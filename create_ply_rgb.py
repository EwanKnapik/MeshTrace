import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import trimesh
import os
from tqdm import tqdm

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _subset_patch_state(sd, instance_trgl=None):
    vertices = sd["triangles_points"]
    triangle_indices = sd["_triangle_indices"]
    f_dc = sd["features_dc"]

    if instance_trgl is None:
        return vertices, triangle_indices, f_dc

    if isinstance(instance_trgl, np.ndarray):
        instance_trgl = torch.from_numpy(instance_trgl)
    instance_trgl = instance_trgl.to(device=triangle_indices.device)
    if instance_trgl.dtype != torch.bool:
        instance_trgl = instance_trgl.long()
    sti = triangle_indices[instance_trgl]
    if sti.numel() == 0:
        return None

    orig_idx, inverse = torch.unique(sti.reshape(-1), sorted=True, return_inverse=True)
    return vertices[orig_idx], inverse.view(-1, 3), f_dc[orig_idx]


def _export_ply_from_state(sd, output_name, output_path="", instance_trgl=None):
    subset = _subset_patch_state(sd, instance_trgl)
    if subset is None:
        return

    vertices, triangle_indices, f_dc = subset
    verts_np = vertices.detach().cpu().numpy()
    faces_np = triangle_indices.detach().cpu().numpy()

    SH_C0 = 0.28209479177387814
    colors = SH_C0 * f_dc + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy().squeeze()

    mesh = trimesh.Trimesh(
        vertices=verts_np.astype(np.float32),
        faces=faces_np.astype(np.int32),
        vertex_colors=colors_u8.astype(np.uint8),
        process=False,
    )
    mesh.export(f"{output_path}/{output_name}", file_type='ply')


def create_ply_rgb(path,output_name,output_path="",instance_trgl=None):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    _export_ply_from_state(sd, output_name, output_path, instance_trgl)


def _patch_has_content(patch) -> bool:
    if isinstance(patch, list):
        return len(patch) > 0

    if isinstance(patch, np.ndarray):
        if patch.dtype == np.bool_:
            return bool(patch.any())
        return patch.size > 0

    if patch.dtype == torch.bool:
        return bool(patch.any())
    return patch.numel() > 0



def create_bulk_ply_rgb(patches:torch.tensor,path:str,input_model:str,output_name:str):
    model_dir = os.path.join(path, "point_cloud", f"iteration_{input_model}")
    segment_dir="segmented_instances"

    segment_dir=os.path.join(model_dir,segment_dir)
    os.makedirs(segment_dir, exist_ok=True)

    files_list=os.listdir(segment_dir)
    idxs=[int(file.split("_")[-1]) for file in files_list]
    if len(idxs)==0:
        new_idx=0
    else:
        new_idx=max(idxs)+1

    export_dir =os.path.join(segment_dir, f"instance_ply_{new_idx}")
    os.makedirs(export_dir, exist_ok=True)

    sd = torch.load(f"{model_dir}/point_cloud_state_dict.pt", map_location="cpu", weights_only=False)
    exported = 0
    for patch in tqdm(patches):
        if _patch_has_content(patch):
            _export_ply_from_state(sd, f"{output_name}_{exported}.ply", export_dir, patch)
            exported+=1
    


def main():
    parser = argparse.ArgumentParser(
        description="Convert a triangle-splatting checkpoint to a PLY file with "
                    "full spherical harmonics data for use in rendering engines."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input checkpoint file (e.g., point_cloud_state_dict.pt)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mesh_sh.ply",
        help="Name of the output PLY file (default: mesh_sh.ply)",
    )
    parser.add_argument(
        "--instance",
        type=bool,
        default=False,
        help="WIP",
    )
    args = parser.parse_args()
    create_ply_rgb(args.path, args.output_name)


if __name__ == "__main__":
    main()
