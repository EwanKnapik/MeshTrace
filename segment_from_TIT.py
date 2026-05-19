from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.con_mask_utils import SegmentationMask
from utils.render_utils import save_img_u8
from triangle_renderer.trace_triangle import trace, trace_masks
from triangle_renderer import TriangleModel
from scene import Scene
from conf.con_masks_conf import *
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

import math
import torch.nn.functional as F
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
import sys
from sixel import converter, sixel
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from create_full_ply import create_ply_rgb
import colorsys
import time
import matplotlib

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def summarize_weight_rows(weights: torch.Tensor, atol: float = 1e-6, near_threshold: float = 0.99):
    row_sums = weights.sum(dim=1)
    row_max = weights.max(dim=1).values
    row_nnz = weights.count_nonzero(dim=1)

    zero_rows = torch.isclose(row_sums, torch.zeros_like(row_sums), atol=atol)
    exact_one_hot_rows = (
        torch.isclose(row_sums, torch.ones_like(row_sums), atol=atol)
        & torch.isclose(row_max, torch.ones_like(row_max), atol=atol)
        & (row_nnz == 1)
    )
    near_one_hot_rows = (
        ~zero_rows
        & torch.isclose(row_sums, torch.ones_like(row_sums), atol=atol)
        & (row_max >= near_threshold)
    )

    print(f"rows: {weights.shape[0]}, cols: {weights.shape[1]}")
    print(f"all-zero rows: {zero_rows.sum().item()}")
    print(f"exact one-hot rows: {exact_one_hot_rows.sum().item()}")
    print(f"exactly valid rows: {(zero_rows | exact_one_hot_rows).sum().item()} / {weights.shape[0]}")
    print(f"near one-hot rows (max >= {near_threshold}): {near_one_hot_rows.sum().item()}")

    nonzero_rows = ~zero_rows
    if nonzero_rows.any():
        nonzero_max = row_max[nonzero_rows]
        print(f"nonzero row max: min={nonzero_max.min().item():.6f}, mean={nonzero_max.mean().item():.6f}")

    suspicious_rows = ~(zero_rows | near_one_hot_rows)
    if suspicious_rows.any():
        sample_idx = torch.nonzero(suspicious_rows, as_tuple=False).squeeze(1)[:5]
        print(f"suspicious row indices: {sample_idx.tolist()}")
        print(weights[sample_idx].detach().cpu())

def compute_similarity(patches1:list,patches2:list,thresh=0.8)->list:
    merge_list=[]

    for i,patch1 in enumerate(patches1):
        for j,patch2 in enumerate(patches2):
            inter=len(intersect(patch1,patch2))
            u=len(union(patch1,patch2))
            IoU=(inter/u if u!=0 else 0)
            if IoU>=thresh:
                merge_list.append((i,j))
    return merge_list







def compute_coherent_W(scene: Scene,triangles,pipe,background):
    camera_stack=scene.getTrainCameras()
    patches=[]

    with tqdm(camera_stack) as pbar:
        for cam in pbar:
            sam_mask = torch.from_numpy(cam.sam_mask.copy()).to(device="cuda", dtype=torch.long)
            weights=trace(cam,triangles,sam_mask,pipe,background)
            weights=weights.argmax(1)
            sorted_vals, sorted_idx = torch.sort(weights)

            # Step 2: find boundaries where value changes
            change = torch.ones_like(sorted_vals, dtype=torch.bool)
            change[1:] = sorted_vals[1:] != sorted_vals[:-1]

            # Step 3: split indices by groups
            group_starts = torch.nonzero(change, as_tuple=True)[0]
            group_ends = torch.cat([group_starts[1:], torch.tensor([len(weights)], device=weights.device)])

            groups = [[] for _ in range(weights.max().item() + 1)]

            # Step 4: fill result
            for start, end in zip(group_starts.tolist(), group_ends.tolist()):
                val = sorted_vals[start].item()
                groups[val] = sorted_idx[start:end].tolist()

            if len(patches)==0:
                patches=groups
            else:
                merge_list = compute_similarity(patches, groups)
                print(merge_list)
                for merge_ids in merge_list:
                    patches[merge_ids[0]]=union(patches[merge_ids[0]],groups[merge_ids[1]])
            pbar.set_postfix({
                "Patches": f"{len(patches)}",
            })
    return patches


def main():
    matplotlib.rcParams["backend"] = "Agg"
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--load_iters", type=str)
    parser.add_argument("--id", default=1, type=int)
    parser.add_argument("--num", default=2, type=int)
    parser.add_argument("--alpha_w", action="store_true")
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=args.load_iters,
                  shuffle=False)
    
    compute_coherent_W(scene,triangles,pipe,background)
    


    





if __name__ == "__main__":
    main()
