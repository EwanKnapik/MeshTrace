from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.con_mask_utils import SegmentationMask
from utils.render_utils import save_img_u8
from triangle_renderer.trace_triangle import trace
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


# fixed palette: 256 deterministic colors (RGB triplets 0-255) for indexed PNGs
fixed_palette = []
for i in range(256):
    r, g, b = colorsys.hsv_to_rgb(i / 256.0, 0.65, 0.95)
    fixed_palette.extend([int(r * 255), int(g * 255), int(b * 255)])



def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def get_rasterization_results(viewpoint_camera, pc: TriangleModel, bg_color: torch.Tensor, pipe, alpha_w=False, scaling_modifier=1.0, override_color=None):
    """
    Trace triangles to compute per-triangle per-class weights.
    Returns tensor of shape (P, num_class+1) with counts per class.
    """
    # prepare geometry
    triangles_indices = pc.get_triangle_indices
    vertices = pc.get_vertices
    vertex_weights = pc.get_vertex_weight
    sigma = pc.get_sigma

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    # select SHs/colors as in existing triangle renderer
    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pipe, 'convert_SHs_python', False):
            # fallback: let rasterizer handle SH -> RGB
            shs = pc.get_features
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # call rasterizer
    rendered_image, radii, scaling, allmap, max_blending, was_rendered = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs=shs,
        colors_precomp=colors_precomp,
        scaling=torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=False, device="cuda")
    )

    # allmap contains auxiliary maps; index 6 is used in renderer as id map
    if allmap is None:
        P = triangles_indices.shape[0]
        return torch.zeros((P, num_class + 1), device='cuda')

    # ensure id map is on same resolution as id_masks
    id_map = allmap[6].long() if allmap.shape[0] > 6 else torch.full((H, W), -1, dtype=torch.long, device='cuda')

    return id_map, rendered_image


def do_cool_stuff(dataset, pipe, triangles, background, camera_stack, id):
    print("Tracing triangles...")
    trngl_set = set()
    for idx, camera in tqdm(enumerate(camera_stack)):
        print(f"Processing camera {idx}")
        segm_results=np.load(f"{dataset.source_path}/{DEFAULT_SAM_FOLDER}/{SPLIT_FOLDER}/{camera.image_name}.npy")
        id_map, rendered_image = get_rasterization_results(camera, triangles, background, pipe, alpha_w=False, scaling_modifier=1.0)
        id_map = id_map.cpu().numpy()
        # vectorised: select triangle indices where segmentation label == id
        mask = (segm_results == id)
        trngl_set.update(id_map[mask].tolist())
    # remove -1 (background / no triangle rendered)
    trngl_set.discard(-1)
    return list(trngl_set)


def compute_triangle_visibility(pipe, triangles, background, camera_stack, max_triangle_id, num=2):
    # Count capped at 3 (uint8 is enough)
    counts = np.zeros(max_triangle_id + 1, dtype=np.uint8)

    for idx, camera in tqdm(enumerate(camera_stack), total=len(camera_stack)):
        id_map, _ = get_rasterization_results(
            camera, triangles, background, pipe,
            alpha_w=False, scaling_modifier=1.0
        )

        # Move once to CPU + numpy
        id_map = id_map.cpu().numpy()

        # Unique triangle IDs in this image (avoid double counting per image)
        trngl_ids = np.unique(id_map)

        # Increment counts, capped at 3
        for tid in trngl_ids:
            if tid >= 0 and tid <= max_triangle_id:
                if counts[tid] < num+1:
                    counts[tid] += 1

    # Final boolean result
    result = counts > num

    return result



def adjust_id_across_views(dataset, pipe, triangles, background, camera_stack,num):
    
    # Global registry: index k stores triangle-id set for stable instance id (k + 1).
    total_list = []
    # Previous view descriptors: list[(global_instance_id, triangle_id_set)].
    prev_view_segments = []
    sam_path = Path(dataset.source_path) / DEFAULT_SAM_FOLDER
    os.makedirs(sam_path / NORMALIZE_FOLDER, exist_ok=True)

    # Threshold tuned for adjacent views; consecutive frames should overlap more.
    iou_threshold_prev = 0.20
    # Fallback threshold when matching against global registry.
    iou_threshold_global = 0.35

    start = time.perf_counter()

    result = compute_triangle_visibility(pipe, triangles, background, camera_stack, len(triangles.get_triangle_indices),num)

    end = time.perf_counter()

    print(f"Execution time: {end - start:.3f} seconds")

    for idx, camera in tqdm(enumerate(camera_stack)):
        print(f"processing camera {idx}")

        segm_results = np.load(f"{dataset.source_path}/{DEFAULT_SAM_FOLDER}/{SPLIT_FOLDER}/{camera.image_name}.npy")
        id_map, _ = get_rasterization_results(camera, triangles, background, pipe, alpha_w=False, scaling_modifier=1.0)
        id_map = id_map.cpu().numpy()

        # Save original local SAM labels for comparison.
        im = Image.fromarray(segm_results.astype(np.uint8), mode='P')
        im.putpalette(fixed_palette)
        im.save(sam_path / NORMALIZE_FOLDER / f"{idx}_original.png")

        # Build per-segment descriptors: (pixel mask, set of covered triangle ids).
        labels = np.unique(segm_results)
        labels = labels[labels > 0]
        segments = []
        for lbl in labels:
            mask = (segm_results == lbl)
            tri_ids = np.unique(id_map[mask])
            tri_set = set(
                int(x)
                for x in tri_ids
                if int(x) >=0 and int(x) < len(result) and result[int(x)]
            )
            if tri_set:
                segments.append((mask, tri_set))

        # 0 is reserved for background; instance ids start at 1.
        new_mask = np.zeros(segm_results.shape, dtype=np.uint16)

        # Assignment per segment in this view.
        assigned_ids = [None] * len(segments)

        # 1) Primary sequential matching: current view against previous view (one-to-one).
        if prev_view_segments and segments:
            candidate_pairs = []
            for seg_i, (_, tri_set) in enumerate(segments):
                for prev_id, prev_set in prev_view_segments:
                    union_sz = len(tri_set | prev_set)
                    if union_sz == 0:
                        continue
                    iou = len(tri_set & prev_set) / union_sz
                    if iou >= iou_threshold_prev:
                        candidate_pairs.append((iou, seg_i, prev_id))

            # Greedy max-IoU assignment ensures one prev instance maps to at most one segment.
            candidate_pairs.sort(reverse=True, key=lambda x: x[0])
            used_seg = set()
            for _, seg_i, prev_id in candidate_pairs:
                if seg_i in used_seg:
                    continue
                assigned_ids[seg_i] = prev_id
                used_seg.add(seg_i)

        # 2) Fallback: unmatched segments are matched to global registry.
        for seg_i, (_, tri_set) in enumerate(segments):
            if assigned_ids[seg_i] is not None:
                continue

            best_k = -1
            best_iou = 0.0
            for k, existing_set in enumerate(total_list):
                union_sz = len(tri_set | existing_set)
                if union_sz == 0:
                    continue
                iou = len(tri_set & existing_set) / union_sz
                if iou > best_iou:
                    best_iou = iou
                    best_k = k

            if best_k >= 0 and best_iou >= iou_threshold_global:
                assigned_ids[seg_i] = best_k + 1
            else:
                total_list.append(set(tri_set))
                assigned_ids[seg_i] = len(total_list)

        # 3) Commit updates and render normalized instance mask.
        curr_view_segments = []
        for seg_i, (mask, tri_set) in enumerate(segments):
            instance_id = assigned_ids[seg_i]
            total_list[instance_id - 1] |= tri_set
            new_mask[mask] = instance_id
            curr_view_segments.append((instance_id, set(tri_set)))

        prev_view_segments = curr_view_segments

        im = Image.fromarray(new_mask.astype(np.uint8), mode='P')
        im.putpalette(fixed_palette)
        im.save(sam_path / NORMALIZE_FOLDER / f"{idx}.png")

    return total_list






def main():
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
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
                  load_iteration=-1,
                  shuffle=False)
    #trngl_list = do_cool_stuff(dataset, pipe, triangles, background, scene.getTrainCameras(), args.id)
    total_list = adjust_id_across_views(dataset, pipe, triangles, background, scene.getTrainCameras(),args.num)

    checkpoint_path = os.path.join(
        scene.model_path,
        "point_cloud",
        f"iteration_{scene.loaded_iter}",
        "point_cloud_state_dict.pt",
    )
    export_dir = os.path.join(scene.model_path, f"instance_ply_{args.num}",)
    os.makedirs(export_dir, exist_ok=True)

    print(f"Exporting {len(total_list)} instances to {export_dir}")
    for instance_id, tri_ids in enumerate(total_list, start=1):
        if not tri_ids:
            print(f"Skipping instance {instance_id}: empty triangle set")
            continue
        output_name = os.path.join(export_dir, f"instance_{instance_id:04d}.ply")
        create_ply_rgb(checkpoint_path, output_name, list(tri_ids))





if __name__ == "__main__":
    main()
    
