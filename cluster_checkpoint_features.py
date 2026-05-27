#!/usr/bin/env python3
"""Project checkpoint features to 2D with PCA and clustering."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from create_full_ply_clustered import MeshCheckpoint, _load_mesh_checkpoint, _prepare_feature_tensor, _run_kmeans

def _vertex_feature_tensor(mesh: MeshCheckpoint) -> torch.Tensor:
    """Return the per-vertex instance_feature matrix from the checkpoint."""
    instance_feature = mesh.instance_feature.detach().cpu().float()
    return instance_feature

DEFAULT_PLOT_POINT_LIMIT = 50000


def _resolve_checkpoint_path(path_str: str) -> Path:
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




def _extract_features(mesh, feature_source: str) -> tuple[torch.Tensor, str]:
    """Select the feature matrix used for plotting and clustering."""
    if feature_source == "instance":
        return _vertex_feature_tensor(mesh), "vertex"
    raise ValueError(f"Unsupported feature source: {feature_source}")


def _sample_indices(num_rows: int, max_rows: int) -> torch.Tensor:
    if max_rows <= 0 or num_rows <= max_rows:
        return torch.arange(num_rows, dtype=torch.long)
    return torch.linspace(0, num_rows - 1, steps=max_rows).round().long()


def _pca_2d(values: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    if values.shape[0] == 0:
        raise ValueError("Cannot compute PCA for an empty feature matrix.")

    centered = values - values.mean(dim=0, keepdim=True)
    if centered.shape[0] == 1:
        return torch.zeros((1, 2), dtype=centered.dtype), np.array([0.0, 0.0], dtype=np.float32)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].transpose(0, 1)

    if projection.shape[1] < 2:
        padded = torch.zeros((projection.shape[0], 2), dtype=projection.dtype)
        padded[:, : projection.shape[1]] = projection
        projection = padded

    explained_variance = singular_values.square()
    explained_ratio = (explained_variance / explained_variance.sum().clamp_min(1e-12)).cpu().numpy()
    padded_ratio = np.zeros(2, dtype=np.float32)
    padded_ratio[: min(2, explained_ratio.shape[0])] = explained_ratio[:2]
    return projection[:, :2], padded_ratio


def _make_output_prefix(
    checkpoint_path: Path, feature_source: str, output_dir: str | None, output_prefix: str | None
) -> Path:
    if output_prefix:
        prefix = Path(output_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = Path.cwd() / prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return prefix

    base_dir = Path(output_dir).expanduser() if output_dir else checkpoint_path.parent / "feature_analysis"
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{checkpoint_path.stem}_{feature_source}"


def _write_projection_csv(
    output_path: Path, coordinates: torch.Tensor, assignments: torch.Tensor
) -> None:
    data = np.column_stack(
        (
            np.arange(coordinates.shape[0], dtype=np.int64),
            assignments.cpu().numpy().astype(np.int64, copy=False),
            coordinates[:, 0].cpu().numpy().astype(np.float32, copy=False),
            coordinates[:, 1].cpu().numpy().astype(np.float32, copy=False),
        )
    )
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="index,cluster_id,pc1,pc2",
        comments="",
        fmt=["%d", "%d", "%.8f", "%.8f"],
    )


def _write_summary(
    output_path: Path,
    checkpoint_path: Path,
    feature_source: str,
    element_kind: str,
    features: torch.Tensor,
    counts: torch.Tensor,
    explained_ratio: np.ndarray,
    normalize_features: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"checkpoint={checkpoint_path}\n")
        handle.write(f"feature_source={feature_source}\n")
        handle.write(f"element_kind={element_kind}\n")
        handle.write(f"num_points={features.shape[0]}\n")
        handle.write(f"feature_dim={features.shape[1]}\n")
        handle.write(f"normalize_features={normalize_features}\n")
        handle.write(f"clusters={counts.shape[0]}\n")
        handle.write(f"pc1_explained_ratio={explained_ratio[0]:.6f}\n")
        handle.write(f"pc2_explained_ratio={explained_ratio[1]:.6f}\n")
        handle.write("cluster_id,count\n")
        for cluster_id, count in enumerate(counts.tolist()):
            handle.write(f"{cluster_id},{count}\n")


def _focus_limits(
    coordinates: torch.Tensor,
    central_fraction: float = 0.9,
    padding_fraction: float = 0.05,
) -> tuple[float, float, float, float]:
    if coordinates.shape[0] == 0:
        raise ValueError("Cannot compute plot limits for empty coordinates.")

    tail_fraction = max(0.0, min((1.0 - central_fraction) / 2.0, 0.5))
    coords_np = coordinates.detach().cpu().numpy()
    lower = np.quantile(coords_np, tail_fraction, axis=0)
    upper = np.quantile(coords_np, 1.0 - tail_fraction, axis=0)

    center = 0.5 * (lower + upper)
    half_extent = 0.5 * (upper - lower)
    min_extent = np.maximum(np.abs(center) * 1e-3, 1e-6)
    half_extent = np.maximum(half_extent, min_extent)
    half_extent *= 1.0 + padding_fraction

    return (
        float(center[0] - half_extent[0]),
        float(center[0] + half_extent[0]),
        float(center[1] - half_extent[1]),
        float(center[1] + half_extent[1]),
    )


def _save_scatter_plot(
    output_path: Path,
    coordinates: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
    explained_ratio: np.ndarray,
    checkpoint_path: Path,
    feature_source: str,
    plot_point_limit: int,
    normalize_features: bool,
) -> int:
    plot_indices = _sample_indices(coordinates.shape[0], plot_point_limit)
    sampled_coords = coordinates[plot_indices].cpu().numpy()
    sampled_assignments = assignments[plot_indices].cpu().numpy()

    cmap = plt.get_cmap("tab20" if counts.shape[0] <= 20 else "gist_ncar", max(int(counts.shape[0]), 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        sampled_coords[:, 0],
        sampled_coords[:, 1],
        c=sampled_assignments,
        cmap=cmap,
        s=6,
        alpha=0.8,
        linewidths=0,
    )

    if counts.shape[0] <= 30:
        centers = []
        for cluster_id in range(counts.shape[0]):
            center = coordinates[assignments == cluster_id].mean(dim=0)
            centers.append(center)
            ax.text(float(center[0]), float(center[1]), str(cluster_id), fontsize=8, ha="center", va="center")
        if centers:
            centers_np = torch.stack(centers).cpu().numpy()
            ax.scatter(
                centers_np[:, 0],
                centers_np[:, 1],
                c=np.arange(counts.shape[0]),
                cmap=cmap,
                s=80,
                marker="x",
                linewidths=1.5,
            )

    if not normalize_features:
        x_min, x_max, y_min, y_max = _focus_limits(coordinates, central_fraction=0.9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.set_title(f"{checkpoint_path.name} | {feature_source} features")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100.0:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100.0:.1f}% variance)")
    ax.grid(True, alpha=0.15)
    ax.text(
        0.01,
        0.99,
        f"points={coordinates.shape[0]:,} plotted={plot_indices.shape[0]:,} clusters={counts.shape[0]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return int(plot_indices.shape[0])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster checkpoint features and save a 2D PCA plot.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Checkpoint path or directory containing point_cloud_state_dict.pt",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default="instance",
        choices=("instance",),
        help="Feature source to analyze.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="L2-normalize each feature vector before PCA and clustering.",
    )
    parser.add_argument(
        "--plot-point-limit",
        type=int,
        default=DEFAULT_PLOT_POINT_LIMIT,
        help="Maximum number of points to draw in the scatter plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files. Defaults to <checkpoint_dir>/feature_analysis.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional explicit output prefix. Example: /tmp/chair_sh_vertex",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    checkpoint_path = _resolve_checkpoint_path(args.input_path)
    mesh = _load_mesh_checkpoint(checkpoint_path)
    raw_features, element_kind = _extract_features(mesh, args.feature_source)
    features = _prepare_feature_tensor(raw_features, args.normalize_features)
    coordinates, explained_ratio = _pca_2d(features)
    assignments, counts = _run_kmeans(features, n_clusters=args.n_clusters)

    output_prefix = _make_output_prefix(
        checkpoint_path,
        args.feature_source,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )
    plot_path = output_prefix.with_name(output_prefix.name + "_pca_clusters.png")
    csv_path = output_prefix.with_name(output_prefix.name + "_projection.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.txt")

    plotted_points = _save_scatter_plot(
        plot_path,
        coordinates,
        assignments,
        counts,
        explained_ratio,
        checkpoint_path,
        args.feature_source,
        args.plot_point_limit,
        args.normalize_features,
    )
    _write_projection_csv(csv_path, coordinates, assignments)
    _write_summary(
        summary_path,
        checkpoint_path,
        args.feature_source,
        element_kind,
        features,
        counts,
        explained_ratio,
        args.normalize_features,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Feature source: {args.feature_source} ({element_kind})")
    print(f"Points: {features.shape[0]}, feature dim: {features.shape[1]}")
    print(f"Clusters: {counts.shape[0]}, largest cluster: {int(counts.max().item())}")
    print(
        f"PCA explained variance: PC1={explained_ratio[0] * 100.0:.2f}% "
        f"PC2={explained_ratio[1] * 100.0:.2f}%"
    )
    print(f"Saved plot: {plot_path}")
    print(f"Saved projection CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if plotted_points < coordinates.shape[0]:
        print(f"Plot sampled {plotted_points} of {coordinates.shape[0]} points for readability.")


if __name__ == "__main__":
    main()
