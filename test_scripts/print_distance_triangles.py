#!/usr/bin/env python3
"""Inspect triangle distances to help choose clustering parameters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import cudf
    from cuml.cluster import DBSCAN as CuDBSCAN
except ImportError:
    cudf = None
    CuDBSCAN = None

try:
    from sklearn.cluster import DBSCAN as SkDBSCAN
except ImportError:
    SkDBSCAN = None

DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_PLOT_POINT_LIMIT = 8000
DEFAULT_REFERENCE_K = 50
DEFAULT_NEIGHBOR_K = (1, 4, 8, 16, 32, 50)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CENTROID_CHUNK_SIZE = 32768


@dataclass
class MetricReport:
    name: str
    display_name: str
    unit: str
    distances: np.ndarray
    reference_k: int
    elbow_index: int
    elbow_value: float
    candidate_values: list[float]

    @property
    def reference_distances(self) -> np.ndarray:
        return self.distances[:, self.reference_k - 1]


@dataclass
class SampleMetadata:
    mode: str
    bins_per_axis: int | None
    occupied_voxels: int | None
    sampled_voxels: int | None


@dataclass
class DbscanPrediction:
    metric_name: str
    eps: float
    min_samples: int
    backend: str
    clusters: int
    largest_cluster: int
    median_cluster_size: float
    core_points: int
    border_points: int
    noise_points: int
    core_fraction: float
    border_fraction: float
    noise_fraction: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        rank = self.rank
        parent = self.parent
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _resolve_checkpoint_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected a checkpoint file or directory, got: {path}")

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


def _random_sample_indices(num_rows: int, max_rows: int, seed: int) -> torch.Tensor:
    if max_rows <= 0 or num_rows <= max_rows:
        return torch.arange(num_rows, dtype=torch.long)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sample = torch.randperm(num_rows, generator=generator)[:max_rows]
    return sample.sort().values


def _triangle_centroids(
    state_dict: dict[str, Any],
    triangle_ids: torch.Tensor | None = None,
    chunk_size: int = DEFAULT_CENTROID_CHUNK_SIZE,
) -> torch.Tensor:
    triangle_indices = state_dict["_triangle_indices"]
    if triangle_ids is not None:
        triangle_ids = triangle_ids.to(device=triangle_indices.device)
        triangle_indices = triangle_indices[triangle_ids]

    vertices = state_dict["triangles_points"]
    centroids: list[torch.Tensor] = []
    for start in range(0, int(triangle_indices.shape[0]), chunk_size):
        end = min(start + chunk_size, int(triangle_indices.shape[0]))
        centroids.append(vertices[triangle_indices[start:end]].detach().float().mean(dim=1).cpu())
    return torch.cat(centroids, dim=0)


def _spatial_grid_sample_indices(
    centroids: torch.Tensor,
    sample_size: int,
    seed: int,
    bins_per_axis: int | None,
) -> tuple[torch.Tensor, SampleMetadata]:
    num_rows = int(centroids.shape[0])
    if sample_size <= 0 or num_rows <= sample_size:
        return torch.arange(num_rows, dtype=torch.long), SampleMetadata(
            mode="spatial_grid",
            bins_per_axis=bins_per_axis,
            occupied_voxels=None,
            sampled_voxels=None,
        )

    if bins_per_axis is None:
        bins_per_axis = max(1, int(round(sample_size ** (1.0 / 3.0))))

    coords = centroids.detach().cpu().float()
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    spans = (maxs - mins).clamp_min(1e-12)
    normalized = (coords - mins) / spans
    voxel_coords = torch.floor(normalized * bins_per_axis).to(torch.long)
    voxel_coords = torch.clamp(voxel_coords, min=0, max=max(bins_per_axis - 1, 0))

    voxel_ids = (
        voxel_coords[:, 0]
        + bins_per_axis * (voxel_coords[:, 1] + bins_per_axis * voxel_coords[:, 2])
    ).numpy()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_rows, generator=generator).tolist()

    voxel_buckets: dict[int, list[int]] = {}
    voxel_order: list[int] = []
    for triangle_index in perm:
        voxel_id = int(voxel_ids[triangle_index])
        bucket = voxel_buckets.get(voxel_id)
        if bucket is None:
            bucket = []
            voxel_buckets[voxel_id] = bucket
            voxel_order.append(voxel_id)
        bucket.append(triangle_index)

    selected: list[int] = []
    positions = {voxel_id: 0 for voxel_id in voxel_order}
    active_voxels = voxel_order
    while len(selected) < sample_size and active_voxels:
        next_active: list[int] = []
        for voxel_id in active_voxels:
            position = positions[voxel_id]
            bucket = voxel_buckets[voxel_id]
            if position >= len(bucket):
                continue

            selected.append(bucket[position])
            position += 1
            positions[voxel_id] = position
            if position < len(bucket):
                next_active.append(voxel_id)
            if len(selected) >= sample_size:
                break
        active_voxels = next_active

    selected_tensor = torch.tensor(sorted(selected), dtype=torch.long)
    sampled_voxels = len({int(voxel_ids[index]) for index in selected})
    return selected_tensor, SampleMetadata(
        mode="spatial_grid",
        bins_per_axis=bins_per_axis,
        occupied_voxels=len(voxel_order),
        sampled_voxels=sampled_voxels,
    )


def _sample_triangle_ids(
    state_dict: dict[str, Any],
    sample_size: int,
    sample_mode: str,
    seed: int,
    spatial_bins_per_axis: int | None,
) -> tuple[torch.Tensor, SampleMetadata]:
    total_triangles = int(state_dict["_triangle_indices"].shape[0])
    if sample_size <= 0 or total_triangles <= sample_size:
        return torch.arange(total_triangles, dtype=torch.long), SampleMetadata(
            mode="all",
            bins_per_axis=None,
            occupied_voxels=None,
            sampled_voxels=None,
        )

    if sample_mode == "random":
        return _random_sample_indices(total_triangles, sample_size, seed), SampleMetadata(
            mode="random",
            bins_per_axis=None,
            occupied_voxels=None,
            sampled_voxels=None,
        )

    if sample_mode != "spatial_grid":
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    all_centroids = _triangle_centroids(state_dict, triangle_ids=None, chunk_size=DEFAULT_CENTROID_CHUNK_SIZE)
    return _spatial_grid_sample_indices(
        all_centroids,
        sample_size=sample_size,
        seed=seed,
        bins_per_axis=spatial_bins_per_axis,
    )


def _prepare_feature_tensor(raw_features: torch.Tensor, normalize: bool) -> torch.Tensor:
    features = raw_features.detach().float().contiguous()
    if not normalize:
        return features

    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(1e-12)
    return features / norms


def _triangle_feature_tensor(state_dict: dict[str, Any], triangle_ids: torch.Tensor | None = None) -> torch.Tensor:
    triangle_indices = state_dict["_triangle_indices"]
    if triangle_ids is not None:
        triangle_ids = triangle_ids.to(device=triangle_indices.device)
        triangle_indices = triangle_indices[triangle_ids]

    instance_feature_weighted = state_dict["vertex_weight"] * state_dict["instance_feature"]
    triangle_instance = instance_feature_weighted[triangle_indices].sum(dim=1)
    return triangle_instance


def _compute_triangle_geometry(
    state_dict: dict[str, Any],
    triangle_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triangle_indices = state_dict["_triangle_indices"][triangle_ids.to(state_dict["_triangle_indices"].device)]
    triangle_vertices = state_dict["triangles_points"][triangle_indices].detach().float()

    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]

    centroids = triangle_vertices.mean(dim=1)
    edge_lengths = torch.stack(
        (
            torch.linalg.vector_norm(v1 - v0, dim=1),
            torch.linalg.vector_norm(v2 - v1, dim=1),
            torch.linalg.vector_norm(v0 - v2, dim=1),
        ),
        dim=1,
    )
    mean_edge_length = edge_lengths.mean(dim=1)
    area = 0.5 * torch.linalg.vector_norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    return centroids, mean_edge_length, area


def _effective_k_values(
    requested_k_values: tuple[int, ...],
    reference_k: int,
    sample_count: int,
) -> tuple[list[int], int]:
    if sample_count < 2:
        raise ValueError("Need at least two triangles to compute neighbor distances.")

    unique_k = {int(k) for k in requested_k_values if int(k) >= 1}
    unique_k.add(int(reference_k))
    effective_k = sorted(k for k in unique_k if k < sample_count)

    if not effective_k:
        raise ValueError(
            f"All requested k values are too large for {sample_count} sampled triangles. "
            "Lower the k values or increase the sample size."
        )

    return effective_k, min(reference_k, sample_count - 1)


def _knn_distances(values: torch.Tensor, max_k: int, chunk_size: int) -> np.ndarray:
    values = values.detach().float().contiguous()
    sample_count = int(values.shape[0])
    if max_k >= sample_count:
        raise ValueError(f"max_k must be smaller than the number of samples ({sample_count}).")

    device = values.device
    all_knn: list[torch.Tensor] = []

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        chunk = values[start:end]
        distances = torch.cdist(chunk, values)

        row_ids = torch.arange(end - start, device=device)
        col_ids = torch.arange(start, end, device=device)
        distances[row_ids, col_ids] = torch.inf

        knn = torch.topk(distances, k=max_k, dim=1, largest=False, sorted=True).values
        all_knn.append(knn.cpu())

    return torch.cat(all_knn, dim=0).numpy()


def _radius_neighbor_counts(values: torch.Tensor, eps: float, chunk_size: int) -> np.ndarray:
    values = values.detach().float().contiguous()
    sample_count = int(values.shape[0])
    counts = np.zeros(sample_count, dtype=np.int32)

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        within = torch.cdist(values[start:end], values) <= eps
        counts[start:end] = within.sum(dim=1).cpu().numpy().astype(np.int32, copy=False)

    return counts


def _pca_2d(values: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    centered = values.detach().cpu().float()
    centered = centered - centered.mean(dim=0, keepdim=True)

    if centered.shape[0] == 1:
        return torch.zeros((1, 2), dtype=torch.float32), np.array([0.0, 0.0], dtype=np.float32)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].transpose(0, 1)

    if projection.shape[1] < 2:
        padded = torch.zeros((projection.shape[0], 2), dtype=projection.dtype)
        padded[:, : projection.shape[1]] = projection
        projection = padded

    explained_variance = singular_values.square()
    explained_ratio = (explained_variance / explained_variance.sum().clamp_min(1e-12)).numpy()
    padded_ratio = np.zeros(2, dtype=np.float32)
    padded_ratio[: min(2, explained_ratio.shape[0])] = explained_ratio[:2]
    return projection[:, :2], padded_ratio


def _estimate_elbow(sorted_distances: np.ndarray) -> tuple[int, float]:
    if sorted_distances.shape[0] == 0:
        return 0, 0.0
    if sorted_distances.shape[0] < 3:
        return sorted_distances.shape[0] - 1, float(sorted_distances[-1])

    y = sorted_distances.astype(np.float64, copy=False)
    y_min = float(y[0])
    y_max = float(y[-1])
    if y_max <= y_min:
        midpoint = sorted_distances.shape[0] // 2
        return midpoint, float(sorted_distances[midpoint])

    x = np.linspace(0.0, 1.0, sorted_distances.shape[0])
    y_norm = (y - y_min) / max(y_max - y_min, 1e-12)
    deviations = y_norm - x
    elbow_index = int(np.argmax(deviations))
    return elbow_index, float(sorted_distances[elbow_index])


def _candidate_values(reference_distances: np.ndarray) -> tuple[list[float], int, float]:
    sorted_distances = np.sort(reference_distances)
    elbow_index, elbow_value = _estimate_elbow(sorted_distances)

    candidates = [elbow_value]
    for percentile in (50, 75, 90, 95):
        candidates.append(float(np.percentile(sorted_distances, percentile)))

    deduped: list[float] = []
    for value in sorted(candidates):
        if deduped and np.isclose(value, deduped[-1], rtol=1e-4, atol=1e-8):
            continue
        deduped.append(value)
    return deduped, elbow_index, elbow_value


def _build_metric_report(
    name: str,
    display_name: str,
    unit: str,
    distances: np.ndarray,
    reference_k: int,
) -> MetricReport:
    candidate_values, elbow_index, elbow_value = _candidate_values(distances[:, reference_k - 1])
    return MetricReport(
        name=name,
        display_name=display_name,
        unit=unit,
        distances=distances,
        reference_k=reference_k,
        elbow_index=elbow_index,
        elbow_value=elbow_value,
        candidate_values=candidate_values,
    )


def _torch_dbscan(values: torch.Tensor, eps: float, min_samples: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = values.detach().float().contiguous()
    sample_count = int(values.shape[0])
    neighbor_counts = _radius_neighbor_counts(values, eps=eps, chunk_size=chunk_size)
    core_mask = neighbor_counts >= min_samples
    border_anchor = np.full(sample_count, -1, dtype=np.int32)
    union_find = _UnionFind(sample_count)

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        within = (torch.cdist(values[start:end], values) <= eps).cpu().numpy()
        for local_row, global_row in enumerate(range(start, end)):
            neighbors = np.flatnonzero(within[local_row])
            core_neighbors = neighbors[core_mask[neighbors]]
            if core_mask[global_row]:
                for neighbor in core_neighbors:
                    if int(neighbor) > global_row:
                        union_find.union(global_row, int(neighbor))
            elif core_neighbors.size > 0:
                border_anchor[global_row] = int(core_neighbors[0])

    labels = np.full(sample_count, -1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    core_indices = np.flatnonzero(core_mask)
    for core_index in core_indices:
        root = union_find.find(int(core_index))
        label = root_to_label.get(root)
        if label is None:
            label = len(root_to_label)
            root_to_label[root] = label
        labels[int(core_index)] = label

    border_indices = np.flatnonzero(~core_mask)
    for border_index in border_indices:
        anchor = int(border_anchor[border_index])
        if anchor < 0:
            continue
        labels[int(border_index)] = labels[anchor]

    return labels, core_mask


def _summarize_dbscan_labels(
    metric_name: str,
    labels: np.ndarray,
    core_mask: np.ndarray,
    eps: float,
    min_samples: int,
    backend: str,
) -> DbscanPrediction:
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        cluster_sizes = np.zeros((0,), dtype=np.int64)
    else:
        cluster_sizes = np.bincount(valid_labels.astype(np.int64, copy=False))

    core_points = int(core_mask.sum())
    noise_points = int((labels < 0).sum())
    border_points = int(labels.shape[0] - core_points - noise_points)
    largest_cluster = int(cluster_sizes.max()) if cluster_sizes.size else 0
    median_cluster_size = float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0

    total = max(int(labels.shape[0]), 1)
    return DbscanPrediction(
        metric_name=metric_name,
        eps=float(eps),
        min_samples=int(min_samples),
        backend=backend,
        clusters=int(cluster_sizes.shape[0]),
        largest_cluster=largest_cluster,
        median_cluster_size=median_cluster_size,
        core_points=core_points,
        border_points=border_points,
        noise_points=noise_points,
        core_fraction=core_points / total,
        border_fraction=border_points / total,
        noise_fraction=noise_points / total,
    )


def _predict_dbscan_behavior(
    metric_name: str,
    values: torch.Tensor,
    eps_values: list[float],
    min_samples: int,
    chunk_size: int,
) -> list[DbscanPrediction]:
    predictions: list[DbscanPrediction] = []
    feature_array = values.detach().cpu().numpy().astype(np.float32, copy=False)

    if cudf is not None and CuDBSCAN is not None:
        frame = cudf.DataFrame(feature_array)
        for eps in eps_values:
            labels = CuDBSCAN(eps=float(eps), min_samples=min_samples).fit(frame).labels_.to_numpy()
            core_mask = _radius_neighbor_counts(values, eps=float(eps), chunk_size=chunk_size) >= min_samples
            predictions.append(
                _summarize_dbscan_labels(metric_name, labels.astype(np.int32, copy=False), core_mask, eps, min_samples, "cuml")
            )
        return predictions

    if SkDBSCAN is not None:
        for eps in eps_values:
            labels = SkDBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(feature_array)
            core_mask = _radius_neighbor_counts(values, eps=float(eps), chunk_size=chunk_size) >= min_samples
            predictions.append(
                _summarize_dbscan_labels(metric_name, labels.astype(np.int32, copy=False), core_mask, eps, min_samples, "sklearn")
            )
        return predictions

    for eps in eps_values:
        labels, core_mask = _torch_dbscan(values, eps=float(eps), min_samples=min_samples, chunk_size=chunk_size)
        predictions.append(_summarize_dbscan_labels(metric_name, labels, core_mask, eps, min_samples, "torch"))
    return predictions


def _make_output_prefix(
    checkpoint_path: Path,
    output_dir: str | None,
    output_prefix: str | None,
) -> Path:
    if output_prefix:
        prefix = Path(output_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = Path.cwd() / prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return prefix

    base_dir = Path(output_dir).expanduser() if output_dir else checkpoint_path.parent / "triangle_distance_analysis"
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / checkpoint_path.stem


def _percentile_summary(values: np.ndarray, percentiles: tuple[int, ...] = (5, 25, 50, 75, 90, 95, 99)) -> str:
    return ",".join(f"p{percentile}={np.percentile(values, percentile):.6f}" for percentile in percentiles)


def _write_neighbor_metrics_csv(
    output_path: Path,
    triangle_ids: torch.Tensor,
    centroids: np.ndarray,
    mean_edge_length: np.ndarray,
    triangle_area: np.ndarray,
    feature_norms: np.ndarray,
    reports: list[MetricReport],
    k_values: list[int],
) -> None:
    header = [
        "triangle_index",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "mean_edge_length",
        "triangle_area",
        "feature_norm",
    ]
    for report in reports:
        for k_value in k_values:
            header.append(f"{report.name}_k{k_value}")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row_index in range(triangle_ids.shape[0]):
            row = [
                str(int(triangle_ids[row_index].item())),
                f"{centroids[row_index, 0]:.8f}",
                f"{centroids[row_index, 1]:.8f}",
                f"{centroids[row_index, 2]:.8f}",
                f"{mean_edge_length[row_index]:.8f}",
                f"{triangle_area[row_index]:.8f}",
                f"{feature_norms[row_index]:.8f}",
            ]
            for report in reports:
                for k_value in k_values:
                    row.append(f"{report.distances[row_index, k_value - 1]:.8f}")
            handle.write(",".join(row) + "\n")


def _write_dbscan_prediction_csv(output_path: Path, predictions: list[DbscanPrediction]) -> None:
    header = (
        "metric_name,eps,min_samples,backend,clusters,largest_cluster,median_cluster_size,"
        "core_points,border_points,noise_points,core_fraction,border_fraction,noise_fraction"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for prediction in predictions:
            handle.write(
                f"{prediction.metric_name},{prediction.eps:.6f},{prediction.min_samples},{prediction.backend},"
                f"{prediction.clusters},{prediction.largest_cluster},{prediction.median_cluster_size:.6f},"
                f"{prediction.core_points},{prediction.border_points},{prediction.noise_points},"
                f"{prediction.core_fraction:.6f},{prediction.border_fraction:.6f},{prediction.noise_fraction:.6f}\n"
            )


def _save_distance_overview_plot(output_path: Path, reports: list[MetricReport], k_values: list[int]) -> None:
    fig, axes = plt.subplots(2, len(reports), figsize=(6 * len(reports), 10))

    for column, report in enumerate(reports):
        curve_axis = axes[0, column]
        hist_axis = axes[1, column]

        for k_value in k_values:
            curve_axis.plot(np.sort(report.distances[:, k_value - 1]), linewidth=1.5, label=f"k={k_value}")

        curve_axis.axhline(report.elbow_value, color="tab:red", linestyle="--", linewidth=1.2, label="elbow")
        curve_axis.set_title(f"{report.display_name} k-distance curve")
        curve_axis.set_xlabel("Sorted sampled triangles")
        curve_axis.set_ylabel(report.unit)
        curve_axis.grid(True, alpha=0.2)
        curve_axis.legend(loc="upper left", fontsize=8)

        hist_axis.hist(report.reference_distances, bins=60, color="tab:blue", alpha=0.8)
        hist_axis.axvline(report.elbow_value, color="tab:red", linestyle="--", linewidth=1.5)
        hist_axis.set_title(f"{report.display_name} histogram at k={report.reference_k}")
        hist_axis.set_xlabel(report.unit)
        hist_axis.set_ylabel("Triangles")
        hist_axis.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_feature_pca_plot(
    output_path: Path,
    raw_features: torch.Tensor,
    unit_features: torch.Tensor,
    raw_report: MetricReport,
    unit_report: MetricReport,
    plot_point_limit: int,
    seed: int,
) -> None:
    raw_coords, raw_ratio = _pca_2d(raw_features)
    unit_coords, unit_ratio = _pca_2d(unit_features)
    plot_indices = _random_sample_indices(raw_features.shape[0], plot_point_limit, seed)
    plot_indices_np = plot_indices.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_specs = (
        (
            axes[0],
            raw_coords[plot_indices].numpy(),
            np.log1p(raw_report.reference_distances[plot_indices_np]),
            raw_ratio,
            f"Raw feature PCA colored by log(1 + k={raw_report.reference_k} distance)",
        ),
        (
            axes[1],
            unit_coords[plot_indices].numpy(),
            np.log1p(unit_report.reference_distances[plot_indices_np]),
            unit_ratio,
            "Unit-normalized feature PCA colored by local distance",
        ),
    )

    for axis, coords, colors, explained_ratio, title in plot_specs:
        scatter = axis.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, cmap="viridis", linewidths=0)
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({explained_ratio[0] * 100.0:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({explained_ratio[1] * 100.0:.1f}% variance)")
        axis.grid(True, alpha=0.15)
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(
    output_path: Path,
    checkpoint_path: Path,
    total_triangles: int,
    sample_count: int,
    device: str,
    feature_dim: int,
    chunk_size: int,
    k_values: list[int],
    sample_metadata: SampleMetadata,
    dbscan_min_samples: int,
    feature_norms: np.ndarray,
    mean_edge_length: np.ndarray,
    triangle_area: np.ndarray,
    reports: list[MetricReport],
    dbscan_predictions: list[DbscanPrediction],
) -> None:
    predictions_by_metric: dict[str, list[DbscanPrediction]] = {}
    for prediction in dbscan_predictions:
        predictions_by_metric.setdefault(prediction.metric_name, []).append(prediction)
    for metric_name in predictions_by_metric:
        predictions_by_metric[metric_name].sort(key=lambda prediction: prediction.eps)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"checkpoint={checkpoint_path}\n")
        handle.write(f"triangles_total={total_triangles}\n")
        handle.write(f"triangles_sampled={sample_count}\n")
        handle.write(f"device={device}\n")
        handle.write(f"feature_dim={feature_dim}\n")
        handle.write(f"chunk_size={chunk_size}\n")
        handle.write("k_values=" + ",".join(str(k_value) for k_value in k_values) + "\n")
        handle.write(f"sample_mode={sample_metadata.mode}\n")
        if sample_metadata.bins_per_axis is not None:
            handle.write(f"sample_bins_per_axis={sample_metadata.bins_per_axis}\n")
        if sample_metadata.occupied_voxels is not None:
            handle.write(f"sample_occupied_voxels={sample_metadata.occupied_voxels}\n")
        if sample_metadata.sampled_voxels is not None:
            handle.write(f"sampled_voxels={sample_metadata.sampled_voxels}\n")
        handle.write(f"dbscan_prediction_min_samples={dbscan_min_samples}\n")
        handle.write("notes=\n")
        handle.write("- feature_raw is the exact triangle feature used inside create_full_ply_clustered.py.\n")
        handle.write("- feature_unit is the same feature after L2 normalization; it helps reveal whether vector magnitude dominates.\n")
        handle.write("- centroid_over_edge is a geometry sanity-check: centroid distance divided by median triangle edge length.\n")
        handle.write("- dbscan_prediction tables are computed on the sampled triangles, not on the full checkpoint.\n")
        handle.write(f"feature_norm_percentiles={_percentile_summary(feature_norms)}\n")
        handle.write(f"mean_edge_length_percentiles={_percentile_summary(mean_edge_length)}\n")
        handle.write(f"triangle_area_percentiles={_percentile_summary(triangle_area)}\n")

        for report in reports:
            handle.write(f"\n[{report.name}]\n")
            handle.write(f"display_name={report.display_name}\n")
            handle.write(f"unit={report.unit}\n")
            handle.write(f"reference_k={report.reference_k}\n")
            handle.write(f"elbow_index={report.elbow_index}\n")
            handle.write(f"elbow_value={report.elbow_value:.6f}\n")
            handle.write("candidate_values=" + ",".join(f"{value:.6f}" for value in report.candidate_values) + "\n")
            handle.write(f"reference_distance_percentiles={_percentile_summary(report.reference_distances)}\n")

            metric_predictions = predictions_by_metric.get(report.name)
            if metric_predictions:
                handle.write(f"\n[dbscan_prediction_{report.name}]\n")
                handle.write(
                    "eps,min_samples,backend,clusters,largest_cluster,median_cluster_size,"
                    "core_points,border_points,noise_points,core_fraction,border_fraction,noise_fraction\n"
                )
                for prediction in metric_predictions:
                    handle.write(
                        f"{prediction.eps:.6f},{prediction.min_samples},{prediction.backend},"
                        f"{prediction.clusters},{prediction.largest_cluster},{prediction.median_cluster_size:.6f},"
                        f"{prediction.core_points},{prediction.border_points},{prediction.noise_points},"
                        f"{prediction.core_fraction:.6f},{prediction.border_fraction:.6f},{prediction.noise_fraction:.6f}\n"
                    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot and export triangle-neighbor distances to guide clustering parameter choices."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Checkpoint file or directory containing point_cloud_state_dict.pt",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum number of triangles to analyze. Use 0 or a negative value to analyze all triangles.",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="spatial_grid",
        choices=("random", "spatial_grid"),
        help="How sampled triangles are chosen when --sample-size is smaller than the scene.",
    )
    parser.add_argument(
        "--spatial-bins-per-axis",
        type=int,
        default=None,
        help="Optional voxel count per axis for spatial-grid sampling. Defaults to an auto value derived from sample size.",
    )
    parser.add_argument(
        "--reference-k",
        type=int,
        default=DEFAULT_REFERENCE_K,
        help="Primary k used for the elbow estimate and summary statistics.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_NEIGHBOR_K),
        help="Neighbor counts shown in the k-distance curves.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=None,
        help="min_samples used in the sample-level DBSCAN prediction table. Defaults to --reference-k.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit eps values for the raw-feature DBSCAN prediction table.",
    )
    parser.add_argument(
        "--predict-unit-dbscan",
        action="store_true",
        help="Also generate a sample-level DBSCAN prediction table for unit-normalized features.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of sampled triangles processed at once during pairwise distance computation.",
    )
    parser.add_argument(
        "--plot-point-limit",
        type=int,
        default=DEFAULT_PLOT_POINT_LIMIT,
        help="Maximum number of sampled triangles drawn in the PCA plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for triangle sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Computation device for feature extraction and neighbor distances.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files. Defaults to <checkpoint_dir>/triangle_distance_analysis.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional explicit output prefix. Example: /tmp/scene_triangles",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = _resolve_device(args.device)

    checkpoint_path = _resolve_checkpoint_path(args.input_path)
    state_dict = _torch_load(checkpoint_path, device=device)

    total_triangles = int(state_dict["_triangle_indices"].shape[0])
    sample_size = total_triangles if args.sample_size <= 0 else args.sample_size
    triangle_ids, sample_metadata = _sample_triangle_ids(
        state_dict,
        sample_size=sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
        spatial_bins_per_axis=args.spatial_bins_per_axis,
    )
    sample_count = int(triangle_ids.shape[0])

    k_values, reference_k = _effective_k_values(tuple(args.neighbor_k), args.reference_k, sample_count)
    max_k = max(k_values)
    dbscan_min_samples = args.dbscan_min_samples if args.dbscan_min_samples is not None else reference_k

    raw_features = _triangle_feature_tensor(state_dict, triangle_ids=triangle_ids)
    raw_features = raw_features.detach().to(device=device).float().contiguous()
    unit_features = _prepare_feature_tensor(raw_features, normalize=True)
    feature_norms = torch.linalg.vector_norm(raw_features, dim=1).detach().cpu().numpy()

    centroids, mean_edge_length, triangle_area = _compute_triangle_geometry(state_dict, triangle_ids)
    centroids = centroids.detach().to(device=device).float().contiguous()
    mean_edge_length_np = mean_edge_length.detach().cpu().numpy()
    triangle_area_np = triangle_area.detach().cpu().numpy()
    median_edge_length = float(np.median(mean_edge_length_np))
    edge_scale = max(median_edge_length, 1e-12)
    centroid_over_edge = centroids / edge_scale

    raw_distances = _knn_distances(raw_features, max_k=max_k, chunk_size=args.chunk_size)
    unit_distances = _knn_distances(unit_features, max_k=max_k, chunk_size=args.chunk_size)
    centroid_distances = _knn_distances(centroid_over_edge, max_k=max_k, chunk_size=args.chunk_size)

    raw_report = _build_metric_report(
        name="feature_raw",
        display_name="Raw triangle features",
        unit="feature distance",
        distances=raw_distances,
        reference_k=reference_k,
    )
    unit_report = _build_metric_report(
        name="feature_unit",
        display_name="Unit-normalized triangle features",
        unit="normalized feature distance",
        distances=unit_distances,
        reference_k=reference_k,
    )
    centroid_report = _build_metric_report(
        name="centroid_over_edge",
        display_name="Centroid distance / median edge length",
        unit="scaled centroid distance",
        distances=centroid_distances,
        reference_k=reference_k,
    )
    reports = [raw_report, unit_report, centroid_report]

    raw_prediction_eps = (
        sorted(set(float(eps) for eps in args.dbscan_eps))
        if args.dbscan_eps is not None
        else raw_report.candidate_values
    )
    dbscan_predictions = _predict_dbscan_behavior(
        metric_name=raw_report.name,
        values=raw_features,
        eps_values=raw_prediction_eps,
        min_samples=dbscan_min_samples,
        chunk_size=args.chunk_size,
    )
    if args.predict_unit_dbscan:
        dbscan_predictions.extend(
            _predict_dbscan_behavior(
                metric_name=unit_report.name,
                values=unit_features,
                eps_values=unit_report.candidate_values,
                min_samples=dbscan_min_samples,
                chunk_size=args.chunk_size,
            )
        )

    output_prefix = _make_output_prefix(checkpoint_path, args.output_dir, args.output_prefix)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.txt")
    csv_path = output_prefix.with_name(output_prefix.name + "_neighbor_metrics.csv")
    dbscan_csv_path = output_prefix.with_name(output_prefix.name + "_dbscan_predictions.csv")
    overview_plot_path = output_prefix.with_name(output_prefix.name + "_distance_overview.png")
    pca_plot_path = output_prefix.with_name(output_prefix.name + "_feature_pca.png")

    _write_summary(
        summary_path,
        checkpoint_path=checkpoint_path,
        total_triangles=total_triangles,
        sample_count=sample_count,
        device=device,
        feature_dim=int(raw_features.shape[1]),
        chunk_size=args.chunk_size,
        k_values=k_values,
        sample_metadata=sample_metadata,
        dbscan_min_samples=dbscan_min_samples,
        feature_norms=feature_norms,
        mean_edge_length=mean_edge_length_np,
        triangle_area=triangle_area_np,
        reports=reports,
        dbscan_predictions=dbscan_predictions,
    )
    _write_neighbor_metrics_csv(
        csv_path,
        triangle_ids=triangle_ids,
        centroids=centroids.detach().cpu().numpy(),
        mean_edge_length=mean_edge_length_np,
        triangle_area=triangle_area_np,
        feature_norms=feature_norms,
        reports=reports,
        k_values=k_values,
    )
    _write_dbscan_prediction_csv(dbscan_csv_path, dbscan_predictions)
    _save_distance_overview_plot(overview_plot_path, reports, k_values)
    _save_feature_pca_plot(
        pca_plot_path,
        raw_features=raw_features.detach().cpu(),
        unit_features=unit_features.detach().cpu(),
        raw_report=raw_report,
        unit_report=unit_report,
        plot_point_limit=args.plot_point_limit,
        seed=args.seed,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Triangles analyzed: {sample_count} / {total_triangles}")
    print(f"Sample mode: {sample_metadata.mode}")
    if sample_metadata.bins_per_axis is not None:
        print(f"Spatial bins per axis: {sample_metadata.bins_per_axis}")
    print(f"Reference k: {reference_k}")
    print(f"DBSCAN prediction min_samples: {dbscan_min_samples}")
    print(f"Raw feature elbow value: {raw_report.elbow_value:.6f}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved DBSCAN prediction CSV: {dbscan_csv_path}")
    print(f"Saved distance overview plot: {overview_plot_path}")
    print(f"Saved PCA plot: {pca_plot_path}")


if __name__ == "__main__":
    main()
