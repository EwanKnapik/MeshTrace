#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import sys
import torch.nn.functional as F

from random import randint
from utils.loss_utils import contrastive_loss
from triangle_renderer.render_feature import render
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

import numpy as np
from sixel import converter, sixel
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from create_full_ply import create_ply_rgb
import colorsys
import time
import matplotlib


def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    scale = values.max().clamp_min(1e-8)
    return values / scale


def _project_triangle_values(rend_ids: torch.Tensor, triangle_values: torch.Tensor) -> torch.Tensor:
    ids = rend_ids.long()
    if ids.dim() == 3 and ids.shape[0] == 1:
        ids = ids.squeeze(0)

    valid = (ids >= 0) & (ids < triangle_values.shape[0])

    if triangle_values.dim() == 1:
        projected = torch.zeros(ids.shape, device=triangle_values.device, dtype=triangle_values.dtype)
        projected[valid] = triangle_values[ids[valid]]
        return projected

    channels = triangle_values.shape[1]
    projected = torch.zeros((*ids.shape, channels), device=triangle_values.device, dtype=triangle_values.dtype)
    projected[valid] = triangle_values[ids[valid]]
    return projected


def _save_heatmap(path: str, values: torch.Tensor, cmap: str, title: str) -> None:
    image = values.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_rgb_image(path: str, values: torch.Tensor) -> None:
    image = values.detach().clamp(0.0, 1.0).cpu().numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def _save_gradient_visualizations(
    iteration: int,
    output_dir: str,
    instance_image: torch.Tensor,
    rend_ids: torch.Tensor,
    feature_before: torch.Tensor,
    feature_after: torch.Tensor,
    feature_grad: torch.Tensor,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if instance_image.grad is None:
        return

    pixel_grad = instance_image.grad.detach().norm(dim=0)
    pixel_grad = _normalize_map(pixel_grad)
    _save_heatmap(
        os.path.join(output_dir, f"iter_{iteration:06d}_pixel_grad.png"),
        pixel_grad,
        "magma",
        "Pixel Gradient Norm",
    )

    grad_norm = feature_grad.detach().norm(dim=-1)
    grad_norm_img = _normalize_map(_project_triangle_values(rend_ids, grad_norm))
    _save_heatmap(
        os.path.join(output_dir, f"iter_{iteration:06d}_triangle_grad_norm.png"),
        grad_norm_img,
        "inferno",
        "Triangle Feature Gradient Norm",
    )

    delta_norm = (feature_after - feature_before).norm(dim=-1)
    delta_norm_img = _normalize_map(_project_triangle_values(rend_ids, delta_norm))
    _save_heatmap(
        os.path.join(output_dir, f"iter_{iteration:06d}_triangle_update_norm.png"),
        delta_norm_img,
        "viridis",
        "Triangle Feature Update Norm",
    )

    grad_scale = feature_grad.detach().abs().amax().clamp_min(1e-8)
    grad_rgb = 0.5 + 0.5 * (feature_grad.detach() / grad_scale)
    grad_rgb_img = _project_triangle_values(rend_ids, grad_rgb)
    _save_rgb_image(
        os.path.join(output_dir, f"iter_{iteration:06d}_triangle_grad_rgb.png"),
        grad_rgb_img,
    )

# fixed palette: 256 deterministic colors (RGB triplets 0-255) for indexed PNGs
fixed_palette = []
for i in range(256):
    r, g, b = colorsys.hsv_to_rgb(i / 256.0, 0.65, 0.95)
    fixed_palette.extend([int(r * 255), int(g * 255), int(b * 255)])


def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    first_iter = 0
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    #triangles.training_setup(opt)

    model_params = torch.load(checkpoint,weights_only=False)
    triangles.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    grad_vis_dir = opt.grad_vis_dir
    if not os.path.isabs(grad_vis_dir):
        grad_vis_dir = os.path.join(dataset.model_path, grad_vis_dir)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    contrastive_func = contrastive_loss

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        triangles.update_learning_rate(iteration)


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, background, include_feature=True)
        instance_image = render_pkg["instance_image"]
        should_visualize = opt.grad_vis_interval > 0 and iteration % opt.grad_vis_interval == 0
        if should_visualize:
            instance_image.retain_grad()
            feature_before = triangles.get_instance_feature.detach().clone()
        instance_features = instance_image.permute(1, 2, 0).reshape(-1, instance_image.shape[0])
        sam_mask = torch.from_numpy(viewpoint_cam.sam_mask.copy()).to(device="cuda", dtype=torch.long).view(-1)
        
        temperature = 100
        main_loss = 0

        #filter out the zero instances
        instance_features = instance_features[sam_mask > 0]
        sam_mask = sam_mask[sam_mask > 0]
        if instance_features.shape[0] == 0:
            continue

        sample_num = opt.sample_num
        n_sample = min(int(len(instance_features) / sample_num) + 1, 5)

        index = torch.randperm(len(instance_features)).cuda()
        for sample_i in range(n_sample):
            sample_idx = index[sample_i*sample_num:(sample_i+1)*sample_num]
            features = instance_features[sample_idx]
            instance_labels = sam_mask[sample_idx]
            con_loss = contrastive_func(features, instance_labels, temperature)
            main_loss += con_loss
        loss = main_loss / n_sample

        total_loss = loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Vertices": f"{triangles.get_vertices.shape[0]}",
                    "Triangles": f"{triangles.get_triangle_indices.shape[0]}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            torch.cuda.empty_cache()
            # Optimizer step
            # Log and save

            if iteration < opt.iterations:
                triangles.optimizer.step()
                if should_visualize and triangles.get_instance_feature.grad is not None:
                    feature_after = triangles.get_instance_feature.detach().clone()
                    _save_gradient_visualizations(
                        iteration,
                        grad_vis_dir,
                        instance_image,
                        render_pkg["rend_ids"],
                        feature_before,
                        feature_after,
                        triangles.get_instance_feature.grad,
                    )
                triangles.optimizer.zero_grad(set_to_none = True)
            elif should_visualize and triangles.get_instance_feature.grad is not None:
                feature_after = triangles.get_instance_feature.detach().clone()
                _save_gradient_visualizations(
                    iteration,
                    grad_vis_dir,
                    instance_image,
                    render_pkg["rend_ids"],
                    feature_before,
                    feature_after,
                    triangles.get_instance_feature.grad,
                )
            
            if (iteration in save_iterations):
                scene.save(f"{save_name}{iteration}")          
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Instance feature training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_name", type=str,help="name of stored checkpoint")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_feature(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations,  args.start_checkpoint,args.save_name)

    # All done
    print("\nTraining complete.")
