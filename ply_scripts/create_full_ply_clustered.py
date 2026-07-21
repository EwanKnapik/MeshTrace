from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from create_ply_rgb import _export_ply_from_state
from cuml.cluster import DBSCAN,KMeans, HDBSCAN
from tqdm import tqdm
import cudf
import os

def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def perform_dbscan(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = DBSCAN(eps=15, min_samples=1000, metric="cosine").fit(input_tensor_df)
    return clustering.labels_
    
def perform_hdbscan(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = HDBSCAN(min_cluster_size=1000,min_samples=5,metric="euclidean",alpha=1, cluster_selection_epsilon=1).fit(input_tensor_df)
    return clustering.labels_

def perform_kmeans(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = KMeans(n_clusters=50).fit(input_tensor_df)
    return clustering.labels_


def _sample_feature_rows(input_tensor, sample_size, seed):
    num_rows = input_tensor.shape[0]
    if sample_size <= 0 or sample_size >= num_rows:
        return input_tensor.detach().float().contiguous()

    generator = torch.Generator()
    generator.manual_seed(seed)
    sample_indices = torch.randperm(num_rows, generator=generator)[:sample_size].to(input_tensor.device)
    return input_tensor[sample_indices].detach().float().contiguous()


def _pairwise_distances(left, right, metric):
    if metric == "cosine":
        left = torch.nn.functional.normalize(left, dim=1)
        right = torch.nn.functional.normalize(right, dim=1)
        return (1.0 - left @ right.T).clamp_min_(0.0)
    if metric == "euclidean":
        return torch.cdist(left, right)
    raise ValueError(f"Unsupported k-distance metric: {metric}")


def compute_kdistances(input_tensor, k=1000, metric="cosine", sample_size=10000, chunk_size=512, seed=0):
    if k < 1:
        raise ValueError("k-distance k must be at least 1.")
    if chunk_size < 1:
        raise ValueError("k-distance chunk size must be at least 1.")

    features = _sample_feature_rows(input_tensor, sample_size, seed)
    num_rows = features.shape[0]
    if num_rows < 2:
        raise ValueError("Need at least two rows to compute a k-distance graph.")

    effective_k = min(k, num_rows - 1)
    if effective_k != k:
        print(f"Requested k={k}, but only {num_rows} points are available. Using k={effective_k}.")

    kth_distances = []
    for start in tqdm(range(0, num_rows, chunk_size), desc="k-distance"):
        end = min(start + chunk_size, num_rows)
        distances = _pairwise_distances(features[start:end], features, metric)
        row_indices = torch.arange(end - start, device=features.device)
        distances[row_indices, torch.arange(start, end, device=features.device)] = torch.inf
        kth_distances.append(distances.topk(effective_k, largest=False, dim=1).values[:, -1].detach().cpu())

    return torch.cat(kth_distances).numpy(), effective_k, num_rows


def plot_kdistance_graph(
    input_tensor,
    output_path,
    k=1000,
    metric="cosine",
    sample_size=10000,
    chunk_size=512,
    seed=0,
):
    kth_distances, effective_k, num_rows = compute_kdistances(
        input_tensor,
        k=k,
        metric=metric,
        sample_size=sample_size,
        chunk_size=chunk_size,
        seed=seed,
    )
    sorted_distances = np.sort(kth_distances)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_distances, linewidth=1.5)
    ax.set_title(f"k-distance graph ({metric}, k={effective_k})")
    ax.set_xlabel("Points sorted by k-distance")
    ax.set_ylabel(f"Distance to {effective_k}th nearest neighbor")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved k-distance plot: {output_path}")
    print(
        "k-distance stats: "
        f"points={num_rows}, min={sorted_distances[0]:.6f}, "
        f"median={np.median(sorted_distances):.6f}, max={sorted_distances[-1]:.6f}"
    )
    return output_path


def load_scene(path,device):
    state_dict=torch.load(path,map_location=device)
    return state_dict

def segment_scene(
    path,
    device,
    output_name,
    write=False,
    plot_kdistance=False,
    kdistance_only=False,
    kdistance_k=1000,
    kdistance_metric="cosine",
    kdistance_sample_size=10000,
    kdistance_chunk_size=512,
    kdistance_seed=0,
    kdistance_output=None,
):
    state_dict = load_scene(os.path.join(path,"point_cloud_state_dict.pt"), device=device)
    instance_feature=state_dict["instance_feature"]
    vertex_weight=state_dict["vertex_weight"]
    vertices=state_dict["triangles_points"]
    triangles_indices=state_dict["_triangle_indices"]
    instance_feature_weighted=vertex_weight*instance_feature
    triangle_instance=instance_feature_weighted[triangles_indices].sum(dim=1)
    print(triangle_instance.shape)

    if plot_kdistance or kdistance_only:
        if kdistance_output is None:
            kdistance_output = os.path.join(path, f"kdistance_{kdistance_metric}_k{kdistance_k}.png")
        plot_kdistance_graph(
            triangle_instance,
            kdistance_output,
            k=kdistance_k,
            metric=kdistance_metric,
            sample_size=kdistance_sample_size,
            chunk_size=kdistance_chunk_size,
            seed=kdistance_seed,
        )
    if kdistance_only:
        return



    labels=perform_dbscan(triangle_instance)
    #labels=perform_hdbscan(triangle_instance)
    #labels=perform_kmeans(triangle_instance)
    
    
    
    
    labels=torch.tensor(labels)
    print(labels.unique())

    segment_dir="segmented_instances"
    segment_dir=os.path.join(path,segment_dir)
    os.makedirs(segment_dir, exist_ok=True)

    files_list=os.listdir(segment_dir)
    idxs=[int(file.split("_")[-1]) for file in files_list]
    if len(idxs)==0:
        new_idx=0
    else:
        new_idx=max(idxs)+1

    export_dir =os.path.join(segment_dir, f"instance_ply_{new_idx}")
    if write:
        os.makedirs(export_dir, exist_ok=True)
    print(export_dir)

    for i in tqdm(range(labels.max()+1)):
        print((labels==i).sum(-1))
        if write:
            _export_ply_from_state(state_dict, f"{output_name}_{i}.ply", export_dir, labels==i)
    if write:
        _export_ply_from_state(state_dict, f"{output_name}_other.ply", export_dir, labels==-1)
    print((labels==-1).sum(-1))
    


def main() -> None:
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("--scene_path", type=str, help="path to point_cloud_state_dict.pt")
    p.add_argument("--write", help="pass true to write data", action='store_true', default=False)
    p.add_argument("--plot-kdistance", help="compute and save the sorted k-distance graph", action="store_true", default=False)
    p.add_argument("--kdistance-only", help="only compute the k-distance graph, then skip clustering/export", action="store_true", default=False)
    p.add_argument("--kdistance-k", type=int, default=1000, help="neighbor rank used for the k-distance graph")
    p.add_argument("--kdistance-metric", type=str, choices=("cosine", "euclidean"), default="cosine", help="distance metric for the k-distance graph")
    p.add_argument("--kdistance-sample-size", type=int, default=10000, help="number of triangle features to sample; use 0 for all")
    p.add_argument("--kdistance-chunk-size", type=int, default=512, help="row chunk size used while computing nearest-neighbor distances")
    p.add_argument("--kdistance-seed", type=int, default=0, help="random seed used when sampling triangle features")
    p.add_argument("--kdistance-output", type=str, default=None, help="optional output path for the k-distance PNG")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segment_scene(
        args.scene_path,
        device,
        "test",
        args.write,
        plot_kdistance=args.plot_kdistance,
        kdistance_only=args.kdistance_only,
        kdistance_k=args.kdistance_k,
        kdistance_metric=args.kdistance_metric,
        kdistance_sample_size=args.kdistance_sample_size,
        kdistance_chunk_size=args.kdistance_chunk_size,
        kdistance_seed=args.kdistance_seed,
        kdistance_output=args.kdistance_output,
    )


if __name__ == "__main__":
    main()
