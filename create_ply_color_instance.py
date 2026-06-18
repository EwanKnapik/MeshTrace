#!/usr/bin/env python3
"""Cluster checkpoint vertices by PCA-reduced instance_feature and export one point PLY per cluster."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import trimesh
from create_ply_rgb import _export_ply_from_state
from cuml.cluster import DBSCAN,KMeans
from tqdm import tqdm
import cudf
import os


def perform_dbscan(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = DBSCAN(eps=0.86, min_samples=40).fit(input_tensor_df)
    return clustering.labels_
    

def perform_kmeans(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = KMeans(n_clusters=50).fit(input_tensor_df)
    return clustering.labels_


def load_scene(path,device):
    state_dict=torch.load(path,map_location=device)
    return state_dict

def _resolve_state_dict_path(path):
    if os.path.isdir(path):
        return os.path.join(path, "point_cloud_state_dict.pt")
    return path

def _compute_triangle_instance(state_dict):
    instance_feature=state_dict["instance_feature"]
    vertex_weight=state_dict["vertex_weight"]
    triangles_indices=state_dict["_triangle_indices"]
    instance_feature_weighted=vertex_weight*instance_feature
    return instance_feature_weighted[triangles_indices].sum(dim=1)

def _compute_vertex_instance(state_dict):
    instance_feature=state_dict["instance_feature"]
    vertex_weight=state_dict["vertex_weight"]
    return vertex_weight*instance_feature

def _project_features_to_rgb(features):
    features = features.detach().float()
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(features.shape)}")

    centered = features - features.mean(dim=0, keepdim=True)
    if centered.shape[0] == 0:
        raise ValueError("Cannot project an empty feature tensor.")

    if centered.shape[1] == 1:
        projected = centered.repeat(1, 3)
    else:
        cov = centered.transpose(0, 1) @ centered
        cov = cov / max(centered.shape[0] - 1, 1)
        _, eigenvectors = torch.linalg.eigh(cov)
        components = eigenvectors[:, -min(3, centered.shape[1]):]
        projected = centered @ components
        if projected.shape[1] < 3:
            padded = torch.zeros((projected.shape[0], 3), dtype=projected.dtype, device=projected.device)
            padded[:, :projected.shape[1]] = projected
            projected = padded

    projected = projected[:, :3]
    projected = projected.cpu()
    mins = projected.min(dim=0, keepdim=True).values
    maxs = projected.max(dim=0, keepdim=True).values
    scale = (maxs - mins).clamp_min(1e-12)
    projected = (projected - mins) / scale
    return (projected * 255.0).round().to(torch.uint8).numpy()

def export_feature_ply(path, device, output_name, output_dir=None, feature_level="triangle"):
    state_dict_path = _resolve_state_dict_path(path)
    state_dict = load_scene(state_dict_path, device=device)
    vertices = state_dict["triangles_points"]
    triangles_indices = state_dict["_triangle_indices"]

    if feature_level == "triangle":
        features = _compute_triangle_instance(state_dict)
        colors_u8 = _project_features_to_rgb(features)
        triangle_vertices = vertices[triangles_indices].detach().cpu().numpy().reshape(-1, 3)
        faces_np = np.arange(triangles_indices.shape[0] * 3, dtype=np.int32).reshape(-1, 3)
        vertex_colors = np.repeat(colors_u8, 3, axis=0)
        mesh = trimesh.Trimesh(
            vertices=triangle_vertices.astype(np.float32),
            faces=faces_np,
            vertex_colors=vertex_colors.astype(np.uint8),
            process=False,
        )
    elif feature_level == "vertex":
        features = _compute_vertex_instance(state_dict)
        colors_u8 = _project_features_to_rgb(features)
        mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy().astype(np.float32),
            faces=triangles_indices.detach().cpu().numpy().astype(np.int32),
            vertex_colors=colors_u8.astype(np.uint8),
            process=False,
        )
    else:
        raise ValueError(f"Unsupported feature level: {feature_level}")

    if output_dir is None:
        output_dir = os.path.dirname(state_dict_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    mesh.export(output_path, file_type="ply")
    print(f"Saved PCA feature mesh: {output_path}")

def segment_scene(path,device,output_name):
    state_dict = load_scene(_resolve_state_dict_path(path), device=device)
    instance_feature=state_dict["instance_feature"]
    vertex_weight=state_dict["vertex_weight"]
    vertices=state_dict["triangles_points"]
    triangles_indices=state_dict["_triangle_indices"]
    instance_feature_weighted=vertex_weight*instance_feature
    triangle_instance=instance_feature_weighted[triangles_indices].sum(dim=1)



    labels=perform_dbscan(triangle_instance)
    #labels=perform_kmeans(triangle_instance)
    
    
    
    
    labels=torch.tensor(labels)
    print(labels.unique())

    #segment_dir="segmented_instances"
    #segment_dir=os.path.join(path,segment_dir)
    #os.makedirs(segment_dir, exist_ok=True)

    #files_list=os.listdir(segment_dir)
    #idxs=[int(file.split("_")[-1]) for file in files_list]
    #if len(idxs)==0:
    #    new_idx=0
    #else:
    #    new_idx=max(idxs)+1

    #export_dir =os.path.join(segment_dir, f"instance_ply_{new_idx}")
    #os.makedirs(export_dir, exist_ok=True)

    #for i in tqdm(range(labels.max()+1)):
    #    _export_ply_from_state(state_dict, f"{output_name}_{i}.ply", export_dir, labels==i)
    #_export_ply_from_state(state_dict, f"{output_name}_other.ply", export_dir, labels==-1)
    


def main() -> None:
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("--scene_path", type=str, help="path to point_cloud_state_dict.pt")
    p.add_argument("--mode", type=str, choices=("pca_ply", "cluster"), default="pca_ply")
    p.add_argument("--feature-level", type=str, choices=("triangle", "vertex"), default="triangle")
    p.add_argument("--output-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "pca_ply":
        output_name = args.output_name or "single.ply"
        export_feature_ply(
            args.scene_path,
            device,
            output_name=output_name,
            output_dir=args.output_dir,
            feature_level=args.feature_level,
        )
    else:
        output_name = args.output_name or "test"
        segment_scene(args.scene_path,device,output_name)


if __name__ == "__main__":
    main()
