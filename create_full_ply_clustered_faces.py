from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from create_ply_rgb import _export_ply_from_state
from cuml.cluster import DBSCAN,KMeans
from tqdm import tqdm
import cudf
import os


def perform_dbscan(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = DBSCAN(eps=0.1, min_samples=100).fit(input_tensor_df)
    return clustering.labels_
    

def perform_kmeans(input_tensor):
    input_tensor_df = cudf.DataFrame(input_tensor.detach())
    clustering = KMeans(n_clusters=10).fit(input_tensor_df)
    return clustering.labels_


def load_scene(path,device):
    state_dict=torch.load(path,map_location=device)
    return state_dict

def segment_scene(path,device,output_name):
    state_dict = load_scene(os.path.join(path,"point_cloud_state_dict.pt"), device=device)
    triangle_instance=state_dict["instance_feature"]
    #vertex_weight=state_dict["vertex_weight"]
    #vertices=state_dict["triangles_points"]
    #triangles_indices=state_dict["_triangle_indices"]
    #instance_feature_weighted=vertex_weight*instance_feature
    #triangle_instance=instance_feature_weighted[triangles_indices].sum(dim=1)
    print(triangle_instance.shape)



    labels=perform_dbscan(triangle_instance)
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
    #os.makedirs(export_dir, exist_ok=True)

    for i in tqdm(range(labels.max()+1)):
        print((labels==i).sum(-1))
        #_export_ply_from_state(state_dict, f"{output_name}_{i}.ply", export_dir, labels==i)
    #_export_ply_from_state(state_dict, f"{output_name}_other.ply", export_dir, labels==-1)
    print((labels==-1).sum(-1))
    


def main() -> None:
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("--scene_path", type=str, help="path to point_cloud_state_dict.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segment_scene(args.scene_path,device,"test")


if __name__ == "__main__":
    main()
