#!/usr/bin/env python3
"""Cluster checkpoint vertices by instance_feature and export one point PLY per cluster."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MeshCheckpoint:
    """Store only the checkpoint data needed for vertex clustering."""

    vertices: torch.Tensor
    instance_feature: torch.Tensor

    @property
    def vertex_count(self) -> int:
        """Return the number of vertices stored in the checkpoint."""
        return int(self.vertices.shape[0])


def _resolve_checkpoint_path(path_str: str) -> Path:
    """Return a checkpoint file from either a direct path or a directory containing one."""
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected a file or directory, got: {path}")

    checkpoint_path = path / "point_cloud_state_dict.pt"
    if checkpoint_path.is_file():
        return checkpoint_path

    matches = sorted(path.rglob("point_cloud_state_dict*.pt"))
    if not matches:
        raise FileNotFoundError(f"Could not find point_cloud_state_dict*.pt under {path}")
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple checkpoints under {path}. Pass one explicitly.\n"
            + "\n".join(str(match) for match in matches[:10])
        )
    return matches[0]


def _as_vertex_feature_matrix(instance_feature: torch.Tensor | None, vertex_count: int) -> torch.Tensor:
    """Validate instance_feature and return it as a [num_vertices, feature_dim] float tensor."""
    if instance_feature is None:
        raise ValueError("The checkpoint does not contain instance_feature.")

    features = instance_feature.detach().cpu().float()
    if features.dim() == 1:
        features = features.unsqueeze(1)
    elif features.dim() > 2:
        features = features.reshape(features.shape[0], -1)

    if features.shape[0] != vertex_count:
        raise ValueError(
            "instance_feature is not per-vertex. "
            f"Expected {vertex_count} rows, got {features.shape[0]}."
        )
    return features


def _load_mesh_checkpoint(path: str | Path) -> MeshCheckpoint:
    """Load vertex positions and per-vertex instance_feature from a checkpoint."""
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    vertices = state_dict["triangles_points"].detach().cpu().float()
    instance_feature = _as_vertex_feature_matrix(state_dict.get("instance_feature"), vertices.shape[0])
    return MeshCheckpoint(vertices=vertices, instance_feature=instance_feature)


def _prepare_feature_tensor(features: torch.Tensor, normalize_features: bool) -> torch.Tensor:
    """Convert features to float on CPU and optionally L2-normalize each row."""
    prepared = features.detach().cpu().float()
    if normalize_features:
        prepared = prepared / prepared.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return prepared


def _run_kmeans(values: torch.Tensor, n_clusters: int, max_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster feature vectors with a small torch-only k-means implementation."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {tuple(values.shape)}.")
    if values.shape[0] == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    cluster_count = min(n_clusters, values.shape[0])
    seed_ids = torch.linspace(0, values.shape[0] - 1, steps=cluster_count).round().long()
    centroids = values[seed_ids].clone()
    assignments = torch.full((values.shape[0],), -1, dtype=torch.long)

    for _ in range(max_iters):
        distances = torch.cdist(values, centroids)
        new_assignments = distances.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments

        for cluster_id in range(cluster_count):
            mask = assignments == cluster_id
            if torch.any(mask):
                centroids[cluster_id] = values[mask].mean(dim=0)

    counts = torch.bincount(assignments, minlength=cluster_count)
    order = torch.argsort(counts, descending=True)
    remapped = torch.empty_like(assignments)
    sorted_counts = counts[order]
    for new_id, old_id in enumerate(order.tolist()):
        remapped[assignments == old_id] = new_id

    return remapped, sorted_counts


def _make_output_dir(checkpoint_path: Path, output_dir: str | None) -> Path:
    """Create the directory that will receive the exported cluster PLY files."""
    if output_dir:
        export_dir = Path(output_dir).expanduser()
        if not export_dir.is_absolute():
            export_dir = Path.cwd() / export_dir
    else:
        export_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_vertex_clusters"

    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _write_point_ply(path: Path, vertices: torch.Tensor) -> None:
    """Write a position-only ASCII PLY file containing one point per vertex."""
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in vertices.tolist():
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def export_clustered_vertex_plys(
    checkpoint_path: str | Path,
    n_clusters: int,
    output_dir: str | None = None,
    normalize_features: bool = False,
    max_iters: int = 20,
) -> list[Path]:
    """Cluster vertices by instance_feature and export one position-only PLY per cluster."""
    resolved_path = _resolve_checkpoint_path(str(checkpoint_path))
    mesh = _load_mesh_checkpoint(resolved_path)
    features = _prepare_feature_tensor(mesh.instance_feature, normalize_features)
    assignments, counts = _run_kmeans(features, n_clusters=n_clusters, max_iters=max_iters)
    export_dir = _make_output_dir(resolved_path, output_dir)

    exported_paths: list[Path] = []
    for cluster_id in range(counts.shape[0]):
        cluster_vertices = mesh.vertices[assignments == cluster_id]
        if cluster_vertices.shape[0] == 0:
            continue
        output_path = export_dir / f"cluster_{cluster_id:03d}.ply"
        _write_point_ply(output_path, cluster_vertices)
        exported_paths.append(output_path)
        print(f"Saved {output_path} ({cluster_vertices.shape[0]} vertices)")

    print(f"Checkpoint: {resolved_path}")
    print(f"Vertices: {mesh.vertex_count}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Clusters: {counts.shape[0]}")
    return exported_paths


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for checkpoint clustering and PLY export."""
    parser = argparse.ArgumentParser(
        description="Cluster checkpoint vertices by instance_feature and export one point PLY per cluster."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Checkpoint file or directory containing point_cloud_state_dict.pt.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        required=True,
        help="Number of clusters to produce.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the exported PLY files. Defaults to <checkpoint_dir>/<checkpoint>_vertex_clusters.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="L2-normalize each instance_feature vector before clustering.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=20,
        help="Maximum number of k-means refinement iterations.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the vertex clustering export pipeline."""
    args = _build_parser().parse_args()
    export_clustered_vertex_plys(
        checkpoint_path=args.input_path,
        n_clusters=args.clusters,
        output_dir=args.output_dir,
        normalize_features=args.normalize_features,
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
