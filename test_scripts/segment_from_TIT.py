from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.con_mask_utils import SegmentationMask
from utils.render_utils import save_img_u8
from triangle_renderer.trace_triangle import trace, trace_dominant_labels, trace_masks
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


def _foreground_label_ids(labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.empty((0,), device=labels.device, dtype=torch.long)

    group_ids = torch.unique(labels)
    return group_ids[group_ids > 0]


def _labels_to_patch_index_list(labels: torch.Tensor) -> List[torch.Tensor]:
    labels = labels.to(device="cpu", dtype=torch.long)
    group_ids = _foreground_label_ids(labels)
    if group_ids.numel() == 0:
        return []

    sorted_labels, sorted_triangle_ids = torch.sort(labels)
    sorted_triangle_ids = sorted_triangle_ids[sorted_labels > 0]
    sorted_labels = sorted_labels[sorted_labels > 0]

    group_change = torch.ones_like(sorted_labels, dtype=torch.bool)
    group_change[1:] = sorted_labels[1:] != sorted_labels[:-1]
    group_starts = torch.nonzero(group_change, as_tuple=False).squeeze(1)
    group_ends = torch.cat(
        (
            group_starts[1:],
            torch.tensor([sorted_labels.numel()], dtype=group_starts.dtype),
        )
    )

    return [
        sorted_triangle_ids[start:end].to(dtype=torch.int32).contiguous()
        for start, end in zip(group_starts.tolist(), group_ends.tolist())
    ]


def _patch_list_from_storage(patches) -> List[torch.Tensor]:
    if patches is None:
        return []
    if isinstance(patches, torch.Tensor):
        if patches.ndim == 2 and patches.dtype == torch.bool:
            return [
                torch.nonzero(patch, as_tuple=False).squeeze(1).to(dtype=torch.int32)
                for patch in patches.cpu()
            ]
        raise TypeError("unsupported tensor patch storage format")
    if isinstance(patches, np.ndarray):
        if patches.dtype == np.bool_ and patches.ndim == 2:
            return [
                torch.from_numpy(np.flatnonzero(patch).astype(np.int32, copy=False))
                for patch in patches
            ]
        if patches.dtype == object:
            return [
                torch.as_tensor(patch, dtype=torch.int32, device="cpu").contiguous()
                for patch in patches.tolist()
            ]
        raise TypeError("unsupported numpy patch storage format")
    if isinstance(patches, list):
        return [
            torch.as_tensor(patch, dtype=torch.int32, device="cpu").contiguous()
            for patch in patches
        ]
    raise TypeError(f"unsupported patch storage type: {type(patches)!r}")


def _merge_patches(
    patches: List[torch.Tensor],
    labels: torch.Tensor,
    iou_thresh: float = 0.3,
    patch_coverage_thresh: float = 0.5,
    group_coverage_thresh: float = 0.5,
) -> List[torch.Tensor]:
    labels = labels.to(device="cpu", dtype=torch.long)
    current_groups = _labels_to_patch_index_list(labels)
    if len(current_groups) == 0:
        return patches
    if len(patches) == 0:
        return current_groups

    # list of labels available in current view, should be of size (nb_labels)
    group_labels = torch.tensor(
        [int(labels[group[0]].item()) for group in current_groups],
        dtype=torch.long,
    )


    # number of Ts in each patch of current view
    group_sizes = torch.tensor([group.numel() for group in current_groups], dtype=torch.long)
    # dict with label id as key from sam mask, and idx as value
    # label id can have gaps between them, but idx are 0,1,2,...
    # useful for idx in list to not have gaps
    group_index_by_label = {int(label.item()): idx for idx, label in enumerate(group_labels)}
    patch_candidate_groups: List[List[int]] = [[] for _ in patches]
    group_candidate_patches: List[List[int]] = [[] for _ in current_groups]

    for patch_idx, patch in enumerate(patches):
        patch_labels = labels[patch.to(dtype=torch.long)]
        visible_patch_labels = patch_labels[patch_labels > 0]
        visible_patch_size = visible_patch_labels.numel()
        if visible_patch_size == 0:
            continue

        patch_group_labels, intersections = torch.unique(visible_patch_labels, return_counts=True)
        for label_value, intersection in zip(patch_group_labels.tolist(), intersections.tolist()):
            group_idx = group_index_by_label.get(label_value)
            if group_idx is None:
                continue

            group_size = int(group_sizes[group_idx].item())
            union = visible_patch_size + group_size - intersection
            if union <= 0:
                continue

            iou = intersection / union
            patch_coverage = intersection / visible_patch_size
            group_coverage = intersection / group_size
            if (
                iou >= iou_thresh
                and patch_coverage >= patch_coverage_thresh
                and group_coverage >= group_coverage_thresh
            ):
                patch_candidate_groups[patch_idx].append(group_idx)
                group_candidate_patches[group_idx].append(patch_idx)

    matched_patch_to_group = {}
    matched_groups = set()
    for patch_idx, candidate_groups in enumerate(patch_candidate_groups):
        if len(candidate_groups) != 1:
            continue

        group_idx = candidate_groups[0]
        if len(group_candidate_patches[group_idx]) != 1:
            continue

        matched_patch_to_group[patch_idx] = group_idx
        matched_groups.add(group_idx)

    merged_patches: List[torch.Tensor] = []
    for patch_idx, patch in enumerate(patches):
        group_idx = matched_patch_to_group.get(patch_idx)
        if group_idx is None:
            merged_patches.append(patch)
            continue

        merged_patch = torch.unique(
            torch.cat([patch.to(dtype=torch.long), current_groups[group_idx].to(dtype=torch.long)]),
            sorted=True,
        )
        merged_patches.append(merged_patch.to(dtype=torch.int32).contiguous())

    merged_patches.extend(
        current_groups[group_idx]
        for group_idx in range(len(current_groups))
        if len(group_candidate_patches[group_idx]) == 0 and group_idx not in matched_groups
    )
    return merged_patches


def compute_coherent_W(
    scene: Scene,
    triangles,
    pipe,
    background,
    merge_iou_thresh: float = 0.3,
    patch_coverage_thresh: float = 0.5,
    group_coverage_thresh: float = 0.5,
):
    camera_stack = scene.getTrainCameras()
    device = background.device
    patches = None

    with torch.no_grad():
        with tqdm(camera_stack) as pbar:
            for cam in pbar:
                sam_mask = torch.as_tensor(cam.sam_mask, device=device, dtype=torch.long)
                labels = trace_dominant_labels(cam, triangles, sam_mask, pipe, background).cpu()

                if patches is None:
                    patches = _labels_to_patch_index_list(labels)
                else:
                    patches = _merge_patches(
                        patches,
                        labels,
                        iou_thresh=merge_iou_thresh,
                        patch_coverage_thresh=patch_coverage_thresh,
                        group_coverage_thresh=group_coverage_thresh,
                    )

                patch_count = 0 if patches is None else len(patches)
                pbar.set_postfix(
                    {
                        "Patches": f"{patch_count}",
                        "IoU": f"{merge_iou_thresh:.2f}",
                        "PatchCov": f"{patch_coverage_thresh:.2f}",
                        "GroupCov": f"{group_coverage_thresh:.2f}",
                    }
                )

    if patches is None:
        return []
    return patches


def try_load_precomputed_patches(model_path,load_iters):
    patches_dir=os.path.join(model_path,f"point_cloud/iteration_{load_iters}")

    file_name_pt=f'patches_{load_iters}.pt'
    file_name=f'patches_{load_iters}.npy'
    files_list=os.listdir(patches_dir)
    if file_name_pt in files_list:
        print(f"found precomputed patches")
        return _patch_list_from_storage(torch.load(os.path.join(patches_dir,file_name_pt), map_location="cpu"))
    if file_name in files_list:
        print(f"found precomputed patches")
        return _patch_list_from_storage(np.load(os.path.join(patches_dir,file_name), allow_pickle=True))
    else:
        print(f"found no precomputed patches")
        return None

def main():
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--load_iters", type=str)
    parser.add_argument("--merge_iou_thresh", default=0.3, type=float)
    parser.add_argument("--patch_coverage_thresh", default=0.5, type=float)
    parser.add_argument("--group_coverage_thresh", default=0.5, type=float)
    parser.add_argument("--force_recompute_patches", action="store_true")
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)


    trngl_patches=try_load_precomputed_patches(dataset.model_path, args.load_iters)
    if args.force_recompute_patches or trngl_patches is None:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(args=dataset,
                    triangles=triangles,
                    init_opacity=None,
                    set_sigma=None,
                    load_iteration=args.load_iters,
                    shuffle=False)
        trngl_patches=compute_coherent_W(
            scene,
            triangles,
            pipe,
            background,
            merge_iou_thresh=args.merge_iou_thresh,
            patch_coverage_thresh=args.patch_coverage_thresh,
            group_coverage_thresh=args.group_coverage_thresh,
        )
        torch.save(
            trngl_patches,
            os.path.join(dataset.model_path, f"point_cloud/iteration_{args.load_iters}", f"patches_{args.load_iters}.pt"),
        )

    create_bulk_ply_rgb(trngl_patches,dataset.model_path,args.load_iters,args.load_iters)


    


    





if __name__ == "__main__":
    main()
