"""Export full or segmented mesh-splatting checkpoints as PLY files."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814
DEFAULT_SEED_SAMPLE_SIZE = 4096
DEFAULT_ASSIGN_BATCH_SIZE = 32768
DEFAULT_CLUSTER_REFINE_ITERS = 4


@dataclass(frozen=True)
class MeshCheckpoint:
    active_sh_degree: int
    vertices: torch.Tensor
    triangle_indices: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    vertex_weight: torch.Tensor
    sigma: float
    instance_feature: Optional[torch.Tensor]

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.triangle_indices.shape[0])

    def select_faces(self, mask: Optional[torch.Tensor] = None) -> "MeshCheckpoint":
        if mask is None:
            return self

        face_mask = torch.as_tensor(mask, dtype=torch.bool)
        if face_mask.numel() != self.face_count:
            raise ValueError(
                f"Expected a face mask with {self.face_count} entries, got {face_mask.numel()}."
            )

        selected_face_ids = torch.nonzero(face_mask, as_tuple=False).flatten()
        selected_faces = self.triangle_indices[selected_face_ids].long()
        referenced_vertices, inverse = torch.unique(
            selected_faces.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        remapped_faces = inverse.reshape(selected_faces.shape[0], selected_faces.shape[1]).to(torch.int32)

        selected_instance_feature = None
        if self.instance_feature is not None:
            if self.instance_feature.shape[0] == self.face_count:
                selected_instance_feature = self.instance_feature[selected_face_ids]
            elif self.instance_feature.shape[0] == self.vertex_count:
                selected_instance_feature = self.instance_feature[referenced_vertices]
            else:
                raise ValueError(
                    "instance_feature has an unsupported leading dimension. "
                    f"Expected {self.face_count} faces or {self.vertex_count} vertices, "
                    f"got {self.instance_feature.shape[0]}."
                )

        return MeshCheckpoint(
            active_sh_degree=self.active_sh_degree,
            vertices=self.vertices[referenced_vertices],
            triangle_indices=remapped_faces,
            features_dc=self.features_dc[referenced_vertices],
            features_rest=self.features_rest[referenced_vertices],
            vertex_weight=self.vertex_weight[referenced_vertices],
            sigma=self.sigma,
            instance_feature=selected_instance_feature,
        )


def _str2bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def _normalize_output_name(output_name):
    output_path = Path(output_name)
    if output_path.suffix:
        return output_name, output_name[: -len(output_path.suffix)]
    return f"{output_name}.ply", output_name


def _scene_root_for_export(input_path: Path) -> Path:
    if "point_cloud" not in input_path.parts:
        return input_path.parent
    point_cloud_index = input_path.parts.index("point_cloud")
    return Path(*input_path.parts[:point_cloud_index])


def _build_next_export_dir(input_path) -> Path:
    segment_root = _scene_root_for_export(Path(input_path)) / "segmented_instances"
    segment_root.mkdir(parents=True, exist_ok=True)

    existing_indices = []
    for path in segment_root.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("instance_ply_"):
            continue
        suffix = path.name[len("instance_ply_") :]
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    next_index = max(existing_indices, default=-1) + 1
    export_dir = segment_root / f"instance_ply_{next_index}"
    export_dir.mkdir(exist_ok=True)
    return export_dir


def _load_mesh_checkpoint(path) -> MeshCheckpoint:
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    
    vertex_weight = state_dict["vertex_weight"].detach().cpu()
    if vertex_weight.dim() > 1:
        vertex_weight = vertex_weight.squeeze(-1)

    features_dc = state_dict["features_dc"].detach().cpu()
    features_rest = state_dict["features_rest"].detach().cpu()
    num_sh_coeffs = int(features_dc.shape[1] + features_rest.shape[1])
    max_sh_degree = int(np.sqrt(num_sh_coeffs) - 1)

    sigma = state_dict["sigma"]
    if isinstance(sigma, torch.Tensor):
        sigma = float(sigma.detach().cpu().item())
    else:
        sigma = float(sigma)

    return MeshCheckpoint(
        active_sh_degree=int(state_dict.get("active_sh_degree", max_sh_degree)),
        vertices=state_dict["triangles_points"].detach().cpu(),
        triangle_indices=state_dict["_triangle_indices"].detach().cpu().to(torch.int32),
        features_dc=features_dc,
        features_rest=features_rest,
        vertex_weight=vertex_weight,
        sigma=sigma,
        instance_feature=None
        if state_dict.get("instance_feature") is None
        else state_dict["instance_feature"].detach().cpu(),
    )


def _face_mask_from_indices(face_count, face_indices):
    if face_indices is None:
        return None

    indices = torch.as_tensor(face_indices, dtype=torch.long).flatten()
    valid = indices[(indices >= 0) & (indices < face_count)]
    if valid.numel() == 0:
        raise ValueError("No valid triangle indices were provided.")

    mask = torch.zeros(face_count, dtype=torch.bool)
    mask[valid] = True
    return mask


def _mesh_max_sh_degree(mesh: MeshCheckpoint) -> int:
    return int(np.sqrt(mesh.features_dc.shape[1] + mesh.features_rest.shape[1]) - 1)


def _flatten_feature_tensor(features: torch.Tensor) -> np.ndarray:
    return (
        features.transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _mesh_rgb_colors(mesh: MeshCheckpoint) -> np.ndarray:
    colors = SH_C0 * mesh.features_dc + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    return (colors * 255.0).round().to(torch.uint8).numpy().squeeze(1)


def save_ply_with_sh(
    verts,
    faces,
    features_dc,
    features_rest,
    opacities,
    sigma_value,
    active_sh_degree,
    max_sh_degree,
    path,
):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    num_dc = features_dc.shape[1]
    num_rest = features_rest.shape[1]

    print(f"{num_verts} vertices, {num_faces} faces")
    print(
        f"SH: {num_dc} DC + {num_rest} rest = {num_dc + num_rest} coefficients "
        f"(degree {max_sh_degree}, active {active_sh_degree})"
    )

    vert_props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    for index in range(num_dc):
        vert_props.append((f"f_dc_{index}", "f4"))
    for index in range(num_rest):
        vert_props.append((f"f_rest_{index}", "f4"))
    vert_props.append(("opacity", "f4"))
    vert_props.append(("sigma", "f4"))

    vert_data = np.empty(num_verts, dtype=vert_props)
    vert_data["x"] = verts[:, 0]
    vert_data["y"] = verts[:, 1]
    vert_data["z"] = verts[:, 2]

    for index in range(num_dc):
        vert_data[f"f_dc_{index}"] = features_dc[:, index]
    for index in range(num_rest):
        vert_data[f"f_rest_{index}"] = features_rest[:, index]

    vert_data["opacity"] = opacities
    vert_data["sigma"] = np.full(num_verts, sigma_value, dtype=np.float32)

    face_data = np.empty(num_faces, dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    ply = PlyData(
        [
            PlyElement.describe(vert_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        comments=[
            "triangle_splatting",
            f"active_sh_degree {active_sh_degree}",
            f"max_sh_degree {max_sh_degree}",
            "f_dc: DC SH coefficients (R, G, B)",
            "f_rest: higher-order SH, layout [R0..R14, G0..G14, B0..B14]",
            "opacity: logit space, apply sigmoid for [0,1]",
        ],
        text=False,
    )
    ply.write(path)
    print(f"Saved {path}")


def _write_mesh_sh_ply(mesh: MeshCheckpoint, output_path):
    save_ply_with_sh(
        verts=mesh.vertices.numpy().astype(np.float32, copy=False),
        faces=mesh.triangle_indices.numpy().astype(np.int32, copy=False),
        features_dc=_flatten_feature_tensor(mesh.features_dc),
        features_rest=_flatten_feature_tensor(mesh.features_rest),
        opacities=mesh.vertex_weight.numpy().astype(np.float32, copy=False),
        sigma_value=mesh.sigma,
        active_sh_degree=mesh.active_sh_degree,
        max_sh_degree=_mesh_max_sh_degree(mesh),
        path=str(output_path),
    )


def _write_mesh_rgb_ply(
    mesh: MeshCheckpoint,
    output_path,
    face_property_name: Optional[str] = None,
    face_property_value: Optional[int] = None,
):
    vertices = mesh.vertices.numpy().astype(np.float32, copy=False)
    colors = _mesh_rgb_colors(mesh)
    faces = mesh.triangle_indices.numpy().astype(np.int32, copy=False)

    vert_data = np.empty(
        mesh.vertex_count,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vert_data["x"] = vertices[:, 0]
    vert_data["y"] = vertices[:, 1]
    vert_data["z"] = vertices[:, 2]
    vert_data["red"] = colors[:, 0]
    vert_data["green"] = colors[:, 1]
    vert_data["blue"] = colors[:, 2]

    if face_property_name is None:
        face_data = np.empty(mesh.face_count, dtype=[("vertex_indices", "i4", (3,))])
        face_data["vertex_indices"] = faces
    else:
        face_data = np.empty(
            mesh.face_count,
            dtype=[("vertex_indices", "i4", (3,)), (face_property_name, "i4")],
        )
        face_data["vertex_indices"] = faces
        face_data[face_property_name] = np.full(mesh.face_count, face_property_value, dtype=np.int32)

    PlyData(
        [
            PlyElement.describe(vert_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    ).write(str(output_path))
    print(f"Saved {output_path}")


def _triangle_centroids(mesh: MeshCheckpoint) -> torch.Tensor:
    return mesh.vertices[mesh.triangle_indices.long()].mean(dim=1)


def _face_feature_tensor(mesh: MeshCheckpoint, require_instance_feature=False) -> torch.Tensor:
    if mesh.instance_feature is None:
        if require_instance_feature:
            raise ValueError(
                "The checkpoint does not contain instance_feature, so segmented export is unavailable."
            )
        return _triangle_centroids(mesh).float()

    instance_feature = mesh.instance_feature.detach().cpu().float()
    if instance_feature.shape[0] == mesh.face_count:
        return instance_feature
    if instance_feature.shape[0] == mesh.vertex_count:
        # Vertex-attached features are averaged onto each face so segmentation stays mesh-local.
        return instance_feature[mesh.triangle_indices.long()].mean(dim=1)

    raise ValueError(
        "instance_feature has an unsupported leading dimension. "
        f"Expected {mesh.face_count} faces or {mesh.vertex_count} vertices, "
        f"got {instance_feature.shape[0]}."
    )


def _prepare_feature_tensor(features, normalize_features):
    prepared = features.detach().cpu().float()
    if normalize_features:
        prepared = prepared / prepared.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return prepared


def _sample_rows(values, max_rows):
    if values.shape[0] <= max_rows:
        return values
    sample_idx = torch.linspace(0, values.shape[0] - 1, steps=max_rows).long()
    return values[sample_idx]


def _seed_centroids(sample_features, distance_threshold):
    if sample_features.shape[0] == 0:
        return sample_features.new_empty((0, sample_features.shape[1]))

    centroids = [sample_features[0].clone()]
    counts = [1]
    for feature in sample_features[1:]:
        centroid_tensor = torch.stack(centroids)
        distances = torch.norm(centroid_tensor - feature.unsqueeze(0), dim=1)
        best_index = int(torch.argmin(distances).item())
        if distances[best_index].item() <= distance_threshold:
            counts[best_index] += 1
            centroids[best_index] = centroids[best_index] + (
                feature - centroids[best_index]
            ) / counts[best_index]
        else:
            centroids.append(feature.clone())
            counts.append(1)
    return torch.stack(centroids)


def _batched_nearest_centroid(values, centroids, batch_size=DEFAULT_ASSIGN_BATCH_SIZE):
    assignments = []
    min_distances = []
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = values[start:stop]
        distances = torch.cdist(batch, centroids)
        batch_min_distances, batch_assignments = distances.min(dim=1)
        assignments.append(batch_assignments)
        min_distances.append(batch_min_distances)
    return torch.cat(assignments), torch.cat(min_distances)


def _recompute_centroids(values, assignments):
    num_clusters = int(assignments.max().item()) + 1
    centroids = torch.zeros((num_clusters, values.shape[1]), dtype=values.dtype)
    counts = torch.bincount(assignments, minlength=num_clusters)
    centroids.index_add_(0, assignments, values)

    nonzero = counts > 0
    centroids[nonzero] = centroids[nonzero] / counts[nonzero].unsqueeze(1)
    return centroids, counts


def _compact_and_sort_clusters(values, assignments):
    unique_ids, inverse = torch.unique(assignments, sorted=True, return_inverse=True)
    centroids, counts = _recompute_centroids(values, inverse)
    size_order = torch.argsort(counts, descending=True)

    remapped = torch.empty_like(inverse)
    for new_index, old_index in enumerate(size_order.tolist()):
        remapped[inverse == old_index] = new_index

    return remapped, centroids[size_order], counts[size_order], unique_ids[size_order]


def _merge_small_clusters(values, assignments, centroids, counts, min_cluster_size):
    if min_cluster_size <= 1 or counts.numel() <= 1:
        return assignments, centroids, counts

    large_cluster_mask = counts >= min_cluster_size
    if large_cluster_mask.all() or not torch.any(large_cluster_mask):
        return assignments, centroids, counts

    large_cluster_ids = torch.where(large_cluster_mask)[0]
    small_point_mask = ~large_cluster_mask[assignments]
    if not torch.any(small_point_mask):
        return assignments, centroids, counts

    reassigned = assignments.clone()
    new_targets, _ = _batched_nearest_centroid(values[small_point_mask], centroids[large_cluster_ids])
    reassigned[small_point_mask] = large_cluster_ids[new_targets]

    compacted, compact_centroids, compact_counts, _ = _compact_and_sort_clusters(values, reassigned)
    return compacted, compact_centroids, compact_counts


def _segment_instance_features(
    instance_feature,
    distance_threshold,
    normalize_features=False,
    min_cluster_size=1,
    seed_sample_size=DEFAULT_SEED_SAMPLE_SIZE,
    refine_iters=DEFAULT_CLUSTER_REFINE_ITERS,
):
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive.")
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive.")

    features = _prepare_feature_tensor(instance_feature, normalize_features)
    seed_features = _sample_rows(features, seed_sample_size)
    centroids = _seed_centroids(seed_features, distance_threshold)
    if centroids.shape[0] == 0:
        raise ValueError("Unable to initialize any feature centroids from instance_feature.")

    for _ in range(refine_iters):
        assignments, min_distances = _batched_nearest_centroid(features, centroids)
        far_point_mask = min_distances > distance_threshold
        if torch.any(far_point_mask):
            extra_seed_features = _sample_rows(features[far_point_mask], seed_sample_size)
            extra_centroids = _seed_centroids(extra_seed_features, distance_threshold)
            if extra_centroids.shape[0] > 0:
                _, extra_distances = _batched_nearest_centroid(extra_centroids, centroids)
                keep_extra_mask = extra_distances > distance_threshold
                if torch.any(keep_extra_mask):
                    centroids = torch.cat((centroids, extra_centroids[keep_extra_mask]), dim=0)
                    continue
        centroids, _ = _recompute_centroids(features, assignments)

    assignments, _ = _batched_nearest_centroid(features, centroids)
    assignments, centroids, counts, _ = _compact_and_sort_clusters(features, assignments)
    assignments, centroids, counts = _merge_small_clusters(
        features,
        assignments,
        centroids,
        counts,
        min_cluster_size,
    )
    return assignments, counts


def _run_kmeans(values, n_clusters, max_iters=10):
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    features = values.detach().cpu().float()
    if features.shape[0] <= n_clusters:
        assignments = torch.arange(features.shape[0], dtype=torch.long)
        counts = torch.ones(features.shape[0], dtype=torch.long)
        return assignments, counts

    centroids = _sample_rows(features, n_clusters).clone()
    for _ in range(max_iters):
        assignments, _ = _batched_nearest_centroid(features, centroids)
        centroids, _ = _recompute_centroids(features, assignments)

    assignments, _ = _batched_nearest_centroid(features, centroids)
    assignments, _, counts, _ = _compact_and_sort_clusters(features, assignments)
    return assignments, counts


def _write_segmentation_summary(export_dir, stem, mode_name, counts, **metadata):
    summary_path = Path(export_dir) / f"{stem}_{mode_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"mode={mode_name}\n")
        for key, value in metadata.items():
            handle.write(f"{key}={value}\n")
        handle.write(f"segments={counts.numel()}\n")
        for index, count in enumerate(counts.tolist()):
            handle.write(f"{index}\t{count}\n")


def _export_grouped_sh_plys(mesh: MeshCheckpoint, export_dir, stem, assignments, counts, label):
    for group_index in range(counts.numel()):
        face_mask = assignments == group_index
        if not torch.any(face_mask):
            continue
        grouped_mesh = mesh.select_faces(face_mask)
        output_path = Path(export_dir) / f"{stem}_{label}_{group_index}.ply"
        _write_mesh_sh_ply(grouped_mesh, output_path)
        print(
            f"Saved {label} PLY: {output_path} "
            f"(vertices {grouped_mesh.vertex_count}, faces {grouped_mesh.face_count})"
        )


def create_ply_rgb(path, output_name, instance_trgl=None):
    mesh = _load_mesh_checkpoint(path)
    selected_mesh = mesh.select_faces(_face_mask_from_indices(mesh.face_count, instance_trgl))
    _write_mesh_rgb_ply(selected_mesh, output_name)


def create_ply_sh(path, output_name, instance_trgl=None):
    mesh = _load_mesh_checkpoint(path)
    selected_mesh = mesh.select_faces(_face_mask_from_indices(mesh.face_count, instance_trgl))
    _write_mesh_sh_ply(selected_mesh, output_name)


def export_full_ply(path, output_name):
    normalized_name, _ = _normalize_output_name(output_name)
    create_ply_sh(path, normalized_name)


def create_instance_ply_RGB(path, output_name):
    mesh = _load_mesh_checkpoint(path)
    if mesh.instance_feature is None:
        face_ids = torch.zeros(mesh.face_count, dtype=torch.int64)
    else:
        face_ids = torch.argmax(_face_feature_tensor(mesh, require_instance_feature=True), dim=1).to(torch.int64)

    unique_instance_ids = torch.unique(face_ids).tolist()
    positive_instance_ids = [int(instance_id) for instance_id in unique_instance_ids if int(instance_id) >= 0]
    instance_ids_to_export = positive_instance_ids if positive_instance_ids else [int(i) for i in unique_instance_ids]

    normalized_name, stem = _normalize_output_name(output_name)
    export_dir = _build_next_export_dir(path)
    extension = Path(normalized_name).suffix

    exported = 0
    for instance_id in instance_ids_to_export:
        face_mask = face_ids == instance_id
        if not torch.any(face_mask):
            continue

        instance_mesh = mesh.select_faces(face_mask)
        output_path = Path(export_dir) / f"{stem}_instance_{instance_id}{extension}"
        _write_mesh_rgb_ply(
            instance_mesh,
            output_path,
            face_property_name="instance_id",
            face_property_value=instance_id,
        )
        print(
            f"Saved instance mesh PLY: {output_path.name} "
            f"(instance {instance_id}, vertices {instance_mesh.vertex_count}, faces {instance_mesh.face_count})"
        )
        exported += 1

    if exported == 0:
        print("No instance meshes were exported.")


def export_instance_plys(
    path,
    output_name,
    distance_threshold=0.35,
    min_cluster_size=256,
    normalize_features=False,
):
    mesh = _load_mesh_checkpoint(path)
    face_features = _face_feature_tensor(mesh, require_instance_feature=True)

    _, stem = _normalize_output_name(output_name)
    export_dir = _build_next_export_dir(path)
    assignments, counts = _segment_instance_features(
        face_features,
        distance_threshold=distance_threshold,
        normalize_features=normalize_features,
        min_cluster_size=min_cluster_size,
    )

    _export_grouped_sh_plys(mesh, export_dir, stem, assignments, counts, "instance")
    _write_segmentation_summary(
        export_dir,
        stem,
        "instance",
        counts,
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size,
        normalize_features=normalize_features,
    )
    print(f"Exported {counts.numel()} segmented instance PLY files to {export_dir}")


def export_clustered_plys(path, output_name, n_clusters=10):
    mesh = _load_mesh_checkpoint(path)
    cluster_values = _face_feature_tensor(mesh, require_instance_feature=False)

    _, stem = _normalize_output_name(output_name)
    export_dir = _build_next_export_dir(path)
    assignments, counts = _run_kmeans(cluster_values, n_clusters=n_clusters)

    _export_grouped_sh_plys(mesh, export_dir, stem, assignments, counts, "cluster")
    _write_segmentation_summary(export_dir, stem, "cluster", counts, n_clusters=n_clusters)
    print(f"Exported {counts.numel()} clustered PLY files to {export_dir}")


def cluster_patches(path, output_name="mesh_sh.ply", n_clusters=10):
    return export_clustered_plys(path, output_name, n_clusters=n_clusters)


def create_full_ply_SH(path, output_name):
    return export_full_ply(path, output_name)


def create_instance_ply_SH(
    path,
    output_name,
    distance_threshold=0.35,
    min_cluster_size=256,
    normalize_features=False,
):
    return export_instance_plys(
        path,
        output_name,
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size,
        normalize_features=normalize_features,
    )


def create_clustered_plys(path, output_name, n_clusters=10):
    return export_clustered_plys(path, output_name, n_clusters=n_clusters)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Export PLY files from mesh-splatting checkpoints."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the mesh-splatting checkpoint file (e.g. point_cloud_state_dict.pt).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mesh_sh.ply",
        help="Name stem for the output PLY file(s).",
    )
    parser.add_argument(
        "--instance",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help=(
            "If True, segment by instance_feature distance and export one SH PLY per segment. "
            "Supports `--instance`, `--instance True`, `--instance False`."
        ),
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="If set, cluster mesh faces and export one SH PLY per cluster.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters to produce when using --cluster.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.35,
        help="Euclidean threshold used to group similar instance_feature vectors.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=256,
        help="Segments smaller than this are merged into the nearest larger segment.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="L2-normalize instance_feature vectors before segmentation.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Export a single RGB-colored mesh PLY instead of the SH layout when not segmenting.",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    if args.cluster:
        export_clustered_plys(args.path, args.output_name, n_clusters=args.n_clusters)
    elif args.instance:
        export_instance_plys(
            args.path,
            args.output_name,
            distance_threshold=args.distance_threshold,
            min_cluster_size=args.min_cluster_size,
            normalize_features=args.normalize_features,
        )
    elif args.rgb:
        create_ply_rgb(args.path, args.output_name)
    else:
        export_full_ply(args.path, args.output_name)


if __name__ == "__main__":
    main()
