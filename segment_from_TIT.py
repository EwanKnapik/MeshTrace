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
from create_ply_rgb import create_bulk_ply_rgb
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


def _labels_to_patch_matrix(labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.zeros((0, 0), device=labels.device, dtype=torch.bool)

    num_groups = int(labels.max().item()) + 1
    group_ids = torch.arange(num_groups, device=labels.device)
    return group_ids.unsqueeze(1).eq(labels.unsqueeze(0))


def _merge_patches_gpu(patches: torch.Tensor, labels: torch.Tensor, thresh: float = 0.8) -> torch.Tensor:
    if patches.numel() == 0 or labels.numel() == 0:
        return patches

    num_groups = int(labels.max().item()) + 1
    patch_membership = patches.to(dtype=torch.float32)
    patch_sizes = patch_membership.sum(dim=1, keepdim=True)
    group_sizes = torch.bincount(labels, minlength=num_groups).to(dtype=torch.float32).unsqueeze(0)

    intersections = torch.zeros(
        (patches.shape[0], num_groups),
        device=labels.device,
        dtype=torch.float32,
    )
    intersections.scatter_add_(
        1,
        labels.unsqueeze(0).expand(patches.shape[0], -1),
        patch_membership,
    )

    unions = patch_sizes + group_sizes - intersections
    merge_mask = (unions > 0) & (intersections >= (thresh * unions))

    if merge_mask.any():
        patches = patches | merge_mask[:, labels]

    return patches

def _merge_patches_optimized_af(
    patches: torch.Tensor,
    labels: torch.Tensor,
    thresh: float = 0.5,
    block_size: int = 8,
) -> torch.Tensor:
    #patches is (240,993329)
    #labels is (217,993329)
    print(patches.shape)
    print(labels.shape)

    if patches.numel() == 0 or labels.numel() == 0:
        return patches

    n = patches.size(0)
    m = labels.size(0)
    device = patches.device
    row_ids = torch.arange(n, device=device)


    labels_large = labels.repeat(block_size,1,1)

    union = torch.empty((n,m), device=device, dtype=torch.int32)
    intersect = torch.empty((n,m), device=device, dtype=torch.int32)

    for start in range(0, m, block_size):
        print(start)
        end = min(start + block_size, m)
        block_rows = torch.arange(start, end, device=device)
        block_indices = (block_rows.unsqueeze(1) - row_ids.unsqueeze(0)) % n
        patches_block = patches[block_indices[:,:m]]
        intersect[start:end,:] = (patches_block & labels_large[:patches_block.shape[0],:]).sum(dim=2, dtype=torch.int32)
        union[start:end,:] = (patches_block | labels_large[:patches_block.shape[0],:]).sum(dim=2, dtype=torch.int32)
    
    print(intersect[:217,:].sum())
    print(intersect[217:,:].sum())


    iou = torch.div(intersect, union.clamp_min(1))
    #want only 1 patch to merge of each patch in labels
    iou_to_merge=iou.argmax(0)

    print(iou_to_merge.shape)
    print(iou_to_merge)
    nb_patches_to_append=len(iou_to_merge[iou_to_merge==0])

    patches_to_add=torch.empty((nb_patches_to_append,patches.shape[1]),device=device, dtype=torch.bool)
    
    for i,j in enumerate(iou_to_merge):
        k=0
        if j==0:
            patches_to_add[k,:]=labels[i,:]
        if j!=0:
            patches[j,:]=(patches[j,:] | labels[i,:])

    patches=torch.cat((patches,patches_to_add),dim=0)
    print(patches.shape)

    

    return patches

def compute_coherent_W(scene: Scene,triangles,pipe,background):
    camera_stack = scene.getTrainCameras()
    device = background.device
    patches = None

    with torch.no_grad():
        with tqdm(camera_stack) as pbar:
            for cam in pbar:
                sam_mask = torch.as_tensor(cam.sam_mask, device=device, dtype=torch.long)
                labels = trace(cam, triangles, sam_mask, pipe, background).argmax(dim=1)
                labels = _labels_to_patch_matrix(labels)[1:]

                if patches is None:
                    patches = labels
                else:
                    patches = _merge_patches_optimized_af(patches, labels)
                    print(patches.shape)

                patch_count = 0 if patches is None else patches.shape[0]
                pbar.set_postfix({"Patches": f"{patch_count}"})

    if patches is None:
        return []
    return patches


def test_fast():
    device = "cuda"
    patches=torch.tensor(np.load("patches.npy"),device=device)
    labels=torch.tensor(np.load("labels.npy"),device=device)
    to_merge = _merge_patches_optimized_af(patches, labels)


def try_load_precomputed_patches(model_path,load_iters):
    patches_dir=os.path.join(model_path,f"point_cloud/iteration_{load_iters}")

    file_name=f'patches_{load_iters}.npy'
    files_list=os.listdir(patches_dir)
    if file_name in files_list:
        print(f"found precomputed patches")
        return torch.as_tensor(np.load(os.path.join(patches_dir,file_name)), device="cuda")
    else:
        print(f"found no precomputed patches")
        return None

def main():
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--load_iters", type=str)
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)


    trngl_patches=try_load_precomputed_patches(dataset.model_path, args.load_iters)
    if trngl_patches!=None:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(args=dataset,
                    triangles=triangles,
                    init_opacity=None,
                    set_sigma=None,
                    load_iteration=args.load_iters,
                    shuffle=False)
        trngl_patches=compute_coherent_W(scene,triangles,pipe,background)
        np.save(os.path.join(dataset.model_path,f"point_cloud/iteration_{args.load_iters}", f'patches_{args.load_iters}.npy'), trngl_patches.cpu())

    create_bulk_ply_rgb(trngl_patches,dataset.model_path,args.load_iters,args.load_iters)


    


    





if __name__ == "__main__":
    main()
