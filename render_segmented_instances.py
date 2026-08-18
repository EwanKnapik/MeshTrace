#!/usr/bin/env python3
"""Render segmented PLY meshes with SH data from their source checkpoint.

The PLY exporter compacts the vertex table of every segmented instance.  Those
local PLY indices therefore cannot index ``point_cloud_state_dict.pt``
directly.  This script matches each PLY face to the original checkpoint face
by its three vertex positions, then builds a compact render model using the
checkpoint's original vertices, opacity weights, and full SH coefficients.

Typical usage::

    python render_segmented_instances.py \
        -m output/custom/Hotdog \
        --iteration sp_20000 \
        --views test

The source dataset path and other model settings are read from ``cfg_args`` in
the model directory, as they are by ``render.py``.  Use ``-s`` to override the
source path when the saved path is no longer valid.
"""

from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args


CHECKPOINT_FILENAMES = ("point_cloud_state_dict.pt", "point_cloud.pt")


def _iteration_token(value: str) -> str:
    value = str(value)
    prefix = "iteration_"
    return value[len(prefix):] if value.startswith(prefix) else value


def _checkpoint_dir(model_path: Path, iteration: str) -> Path:
    return model_path / "point_cloud" / f"iteration_{_iteration_token(iteration)}"


def _resolve_checkpoint_file(checkpoint_dir: Path) -> Path:
    for filename in CHECKPOINT_FILENAMES:
        candidate = checkpoint_dir / filename
        if candidate.is_file():
            return candidate

    indexed = []
    for candidate in checkpoint_dir.glob("point_cloud_state_dict_*.pt"):
        match = re.fullmatch(r"point_cloud_state_dict_(\d+)\.pt", candidate.name)
        if match:
            indexed.append((int(match.group(1)), candidate))
    if indexed:
        return max(indexed, key=lambda item: item[0])[1]

    expected = ", ".join(CHECKPOINT_FILENAMES)
    raise FileNotFoundError(
        f"No checkpoint found in '{checkpoint_dir}'. Expected {expected}, or "
        "an indexed point_cloud_state_dict_<N>.pt file."
    )


def _find_plys(segment_dir: Path, pattern: str, limit: int) -> list[Path]:
    if not segment_dir.is_dir():
        raise FileNotFoundError(f"Segmented instance directory not found: '{segment_dir}'")

    paths = sorted(path for path in segment_dir.glob(pattern) if path.is_file())
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(
            f"No PLY files matched '{pattern}' under '{segment_dir}'."
        )
    return paths


def _as_numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return np.asarray(tensor.detach().cpu(), dtype=dtype)


def _triangle_geometry_keys(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return winding-independent, exact float32 geometry keys per face."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape [V, 3], got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with shape [F, 3], got {faces.shape}.")
    if faces.size == 0:
        return np.empty((0,), dtype=np.dtype((np.void, 9 * np.dtype(np.float32).itemsize)))
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise ValueError("A face contains a vertex index outside the PLY/checkpoint vertex table.")

    triangles = np.ascontiguousarray(vertices[faces], dtype=np.float32)
    if not np.isfinite(triangles).all():
        raise ValueError("Triangle positions contain NaN or infinity.")

    # Normalize signed zero, because -0.0 and +0.0 are geometrically equal but
    # have different byte representations.
    triangles[triangles == 0.0] = 0.0
    order = np.lexsort(
        (triangles[..., 2], triangles[..., 1], triangles[..., 0]), axis=1
    )
    canonical = np.ascontiguousarray(
        np.take_along_axis(triangles, order[..., None], axis=1)
    )
    key_dtype = np.dtype((np.void, canonical.dtype.itemsize * 9))
    return canonical.reshape(canonical.shape[0], 9).view(key_dtype).reshape(-1)


@dataclass
class TriangleGeometryIndex:
    """Search index from exact triangle geometry to checkpoint face ID."""

    sorted_keys: np.ndarray
    sorted_face_ids: np.ndarray
    face_count: int

    @classmethod
    def build(cls, vertices: np.ndarray, faces: np.ndarray) -> "TriangleGeometryIndex":
        keys = _triangle_geometry_keys(vertices, faces)
        order = np.argsort(keys)
        return cls(keys[order], order.astype(np.int64, copy=False), len(keys))

    def match(self, ply_path: Path) -> np.ndarray:
        mesh = trimesh.load(str(ply_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"'{ply_path}' does not contain one triangular mesh.")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            raise ValueError(f"'{ply_path}' contains no faces.")

        query_keys = _triangle_geometry_keys(vertices, faces)
        left = np.searchsorted(self.sorted_keys, query_keys, side="left")
        right = np.searchsorted(self.sorted_keys, query_keys, side="right")
        match_counts = right - left

        missing = int(np.count_nonzero(match_counts == 0))
        ambiguous = int(np.count_nonzero(match_counts > 1))
        if missing or ambiguous:
            details = []
            if missing:
                details.append(f"{missing} face(s) were not found")
            if ambiguous:
                details.append(
                    f"{ambiguous} face(s) match duplicate checkpoint geometry"
                )
            raise ValueError(
                f"Could not correlate '{ply_path}' with the checkpoint: "
                + "; ".join(details)
                + ". The PLY must preserve the exact float32 positions exported "
                "from this checkpoint."
            )

        checkpoint_face_ids = self.sorted_face_ids[left]
        unique_count = np.unique(checkpoint_face_ids).size
        if unique_count != checkpoint_face_ids.size:
            duplicate_count = checkpoint_face_ids.size - unique_count
            raise ValueError(
                f"'{ply_path}' contains duplicate faces ({duplicate_count} duplicates)."
            )
        return checkpoint_face_ids


def _load_ply_mapping_index_from_checkpoint(
    checkpoint_file: Path,
) -> TriangleGeometryIndex:
    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    try:
        vertices = _as_numpy(state["triangles_points"], np.float32)
        faces = _as_numpy(state["_triangle_indices"], np.int64)
        return TriangleGeometryIndex.build(vertices, faces)
    finally:
        del state
        gc.collect()


def _load_ply_mapping_index_from_model(model) -> TriangleGeometryIndex:
    vertices = _as_numpy(model.get_vertices, np.float32)
    faces = _as_numpy(model.get_triangle_indices, np.int64)
    return TriangleGeometryIndex.build(vertices, faces)


def _subset_render_model(source, checkpoint_face_ids: np.ndarray):
    """Build a compact TriangleModel while retaining the checkpoint SH data."""
    with torch.no_grad():
        face_ids = torch.as_tensor(
            checkpoint_face_ids,
            dtype=torch.long,
            device=source.get_triangle_indices.device,
        )
        source_faces = source.get_triangle_indices[face_ids].long()
        used_vertex_ids, compact_faces = torch.unique(
            source_faces.reshape(-1), sorted=True, return_inverse=True
        )

        instance = type(source)(source.max_sh_degree)
        instance.vertices = source.vertices[used_vertex_ids].contiguous()
        instance._triangle_indices = (
            compact_faces.reshape(-1, 3).to(torch.int32).contiguous()
        )
        instance.vertex_weight = source.vertex_weight[used_vertex_ids].contiguous()
        instance._features_dc = source._features_dc[used_vertex_ids].contiguous()
        instance._features_rest = source._features_rest[used_vertex_ids].contiguous()
    instance._sigma = source._sigma
    instance.active_sh_degree = source.active_sh_degree
    instance.opacity_floor = source.opacity_floor
    instance.scaling = source.scaling
    return instance


def _view_sets(scene, choice: str) -> Sequence[tuple[str, list]]:
    result = []
    if choice in ("train", "both"):
        result.append(("train", scene.getTrainCameras()))
    if choice in ("test", "both"):
        result.append(("test", scene.getTestCameras()))
    return result


def _safe_camera_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).stem)
    return cleaned or "view"


def _render_ply(
    source_model,
    checkpoint_face_ids: np.ndarray,
    ply_path: Path,
    segment_dir: Path,
    output_root: Path,
    view_sets: Sequence[tuple[str, list]],
    pipeline,
    background: torch.Tensor,
    max_views: int,
    skip_existing: bool,
    save_triangle_ids: bool,
) -> int:
    from triangle_renderer import render
    from torchvision.utils import save_image

    relative_ply = ply_path.relative_to(segment_dir)
    instance_output = output_root / relative_ply.parent / relative_ply.stem
    if save_triangle_ids:
        instance_output.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.from_numpy(checkpoint_face_ids.copy()),
            instance_output / "source_triangle_ids.pt",
        )

    instance_model = _subset_render_model(source_model, checkpoint_face_ids)
    rendered = 0
    try:
        with torch.no_grad():
            for split_name, all_views in view_sets:
                views = all_views[:max_views] if max_views > 0 else all_views
                split_output = instance_output / split_name
                split_output.mkdir(parents=True, exist_ok=True)
                for view_index, view in enumerate(
                    tqdm(views, desc=f"{relative_ply} [{split_name}]", leave=False)
                ):
                    camera_name = _safe_camera_name(view.image_name)
                    image_path = split_output / f"{view_index:05d}_{camera_name}.png"
                    if skip_existing and image_path.is_file():
                        continue
                    image = render(view, instance_model, pipeline, background)["render"]
                    save_image(image, image_path)
                    rendered += 1
    finally:
        del instance_model
        gc.collect()
        torch.cuda.empty_cache()
    return rendered


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render segmented PLY instances using SH coefficients from iteration_sp_20000."
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument(
        "--iteration",
        default="sp_20000",
        type=str,
        help="Checkpoint suffix, with or without the 'iteration_' prefix (default: sp_20000).",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="PLY root; defaults to <checkpoint>/segmented_instances.",
    )
    parser.add_argument(
        "--ply-glob",
        default="**/*.ply",
        help="Glob below --segments-dir (default: **/*.ply).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; defaults to <checkpoint>/rendered_segmented_instances.",
    )
    parser.add_argument(
        "--views",
        choices=("train", "test", "both"),
        default="test",
        help="Camera split to render (default: test).",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="Maximum views per PLY and split; 0 renders all views.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Maximum matched PLY files; 0 processes all files.",
    )
    parser.add_argument(
        "--upscaling-factor",
        type=int,
        default=4,
        help="Rasterizer supersampling factor (default: 4).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not replace PNG files that already exist.",
    )
    parser.add_argument(
        "--save-triangle-ids",
        action="store_true",
        help="Save the recovered checkpoint face IDs beside each rendered instance.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate PLY/checkpoint correspondence on CPU without rendering.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, model.extract(args), pipeline.extract(args)


def main() -> None:
    args, dataset, pipeline = _parse_args()
    model_path = Path(dataset.model_path).resolve()
    iteration = _iteration_token(args.iteration)
    checkpoint_dir = _checkpoint_dir(model_path, iteration)
    checkpoint_file = _resolve_checkpoint_file(checkpoint_dir)
    configured_segment_dir = getattr(args, "segments_dir", None)
    configured_output_dir = getattr(args, "output_dir", None)
    segment_dir = (
        configured_segment_dir.resolve()
        if configured_segment_dir is not None
        else checkpoint_dir / "segmented_instances"
    )
    output_root = (
        configured_output_dir.resolve()
        if configured_output_dir is not None
        else checkpoint_dir / "rendered_segmented_instances"
    )
    ply_paths = _find_plys(segment_dir, args.ply_glob, args.max_instances)

    print(f"Checkpoint: {checkpoint_file}")
    print(f"Segmented PLYs: {segment_dir} ({len(ply_paths)} file(s))")

    if args.validate_only:
        geometry_index = _load_ply_mapping_index_from_checkpoint(checkpoint_file)
        total_faces = 0
        for ply_path in tqdm(ply_paths, desc="Validating PLY correspondence"):
            face_ids = geometry_index.match(ply_path)
            total_faces += face_ids.size
            print(f"  {ply_path.relative_to(segment_dir)}: {face_ids.size} faces")
        print(
            f"Validated {len(ply_paths)} PLY file(s), {total_faces} faces total, "
            f"against {geometry_index.face_count} checkpoint faces."
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the triangle rasterizer. Use --validate-only to "
            "check the PLY/checkpoint mapping on a CPU-only machine."
        )

    from scene import Scene
    from triangle_renderer import TriangleModel
    from utils.general_utils import safe_state

    safe_state(args.quiet)
    source_model = TriangleModel(dataset.sh_degree)
    source_model.scaling = args.upscaling_factor
    scene = Scene(
        args=dataset,
        triangles=source_model,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=False,
    )
    geometry_index = _load_ply_mapping_index_from_model(source_model)

    background_rgb = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_rgb, dtype=torch.float32, device="cuda")
    view_sets = _view_sets(scene, args.views)
    if not any(views for _, views in view_sets):
        raise ValueError(f"The requested '{args.views}' camera split contains no views.")

    total_images = 0
    total_faces = 0
    for ply_path in tqdm(ply_paths, desc="Rendering segmented PLYs"):
        face_ids = geometry_index.match(ply_path)
        total_faces += face_ids.size
        total_images += _render_ply(
            source_model=source_model,
            checkpoint_face_ids=face_ids,
            ply_path=ply_path,
            segment_dir=segment_dir,
            output_root=output_root,
            view_sets=view_sets,
            pipeline=pipeline,
            background=background,
            max_views=args.max_views,
            skip_existing=args.skip_existing,
            save_triangle_ids=args.save_triangle_ids,
        )

    print(
        f"Rendered {len(ply_paths)} PLY file(s) ({total_faces} correlated faces) "
        f"to {total_images} image(s) under '{output_root}'."
    )


if __name__ == "__main__":
    main()
