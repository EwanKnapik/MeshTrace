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

from random import randint
from utils.loss_utils import contrastive_loss
from triangle_renderer.render_feature import render as render_from_feature_rasterizer
from triangle_renderer import render as render_from_rgb_rasterizer
from scene import Scene, TriangleModel
from scene.triangle_model import resolve_point_cloud_state_path
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def _project_instance_image_for_plot(instance_image):
    with torch.no_grad():
        channels, height, width = instance_image.shape
        flat_pixels = (
            instance_image.detach()
            .permute(1, 2, 0)
            .reshape(-1, channels)
            .float()
            .cpu()
            .numpy()
        )
        projected = PCA(n_components=3).fit_transform(flat_pixels).reshape(height, width, 3)
        projected -= projected.min(axis=(0, 1), keepdims=True)
        scale = projected.max(axis=(0, 1), keepdims=True)
        scale[scale == 0] = 1.0
        projected = projected / scale
        return torch.from_numpy(projected).permute(2, 0, 1).float()



def compute_instance_feature_via_id_map(viewpoint_cam, triangles : TriangleModel, pipe, background):
    render_pkg = render_from_rgb_rasterizer(viewpoint_cam, triangles, pipe, background)
    rend_ids = render_pkg["rend_ids"][0]

    stored_instance_features = triangles.get_instance_feature
    if stored_instance_features is None:
        raise ValueError("Triangle model does not contain instance features.")

    num_triangles = triangles.get_triangle_indices.shape[0]
    num_vertices = triangles.get_vertices.shape[0]
    if stored_instance_features.shape[0] == num_triangles:
        triangle_instances = stored_instance_features
    elif stored_instance_features.shape[0] == num_vertices:
        triangle_indices = triangles.get_triangle_indices.long()
        triangle_instances = (
            triangles.get_vertex_weight * stored_instance_features
        )[triangle_indices].sum(dim=1)
    else:
        raise ValueError(
            f"instance_feature has {stored_instance_features.shape[0]} rows, "
            f"but expected either {num_triangles} triangle features or "
            f"{num_vertices} vertex features."
        )

    num_instances = triangle_instances.shape[0]
    in_bounds = torch.isfinite(rend_ids) & (rend_ids >= 0) & (rend_ids < num_instances)
    rend_ids_inbounds = rend_ids[in_bounds].long()

    height, width = rend_ids.shape
    feature_dim = triangle_instances.shape[1]
    instance_features_for_plot = torch.zeros(
        (height, width, feature_dim),
        dtype=triangle_instances.dtype,
        device=triangle_instances.device,
    )
    if bool(in_bounds.any().item()):
        instance_features_for_plot[in_bounds] = triangle_instances[rend_ids_inbounds]
    instance_features = instance_features_for_plot[in_bounds]

    sam_mask = torch.as_tensor(
        viewpoint_cam.sam_mask,
        dtype=torch.int,
        device=triangle_instances.device,
    )
    if sam_mask.shape != rend_ids.shape:
        if sam_mask.numel() != rend_ids.numel():
            raise ValueError(
                f"sam_mask shape {tuple(sam_mask.shape)} does not match "
                f"rend_ids shape {tuple(rend_ids.shape)}."
            )
        sam_mask = sam_mask.reshape(rend_ids.shape)

    sam_mask = sam_mask[in_bounds]
    return instance_features, sam_mask, instance_features_for_plot


def compute_instance_feature_via_feature_rasterizer(viewpoint_cam, triangles, pipe, background):
    render_pkg = render_from_feature_rasterizer(viewpoint_cam, triangles, pipe, background, include_feature=True)
    instance_image = render_pkg["instance_image"]
    instance_features = instance_image.permute(1, 2, 0).reshape(-1, instance_image.shape[0])

    sam_mask = torch.from_numpy(viewpoint_cam.sam_mask).to(torch.int).view(-1).cuda()
    return instance_features, sam_mask, instance_image.permute(1, 2, 0)


def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    first_iter = 0
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt)

    assert checkpoint is not None, "Checkpoint path must be provided to load a model."

    checkpoint_path = resolve_point_cloud_state_path(checkpoint)
    model_params = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    triangles.restore(model_params, opt)
    triangles.update_min_weight(1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

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

        instance_features, sam_mask, instance_features_for_plot=compute_instance_feature_via_feature_rasterizer(viewpoint_cam,triangles, pipe, background)

        if iteration %1000==0:
            instance_image_rgb = _project_instance_image_for_plot(instance_features_for_plot.permute(2, 0, 1))
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(instance_image_rgb.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(sam_mask.reshape(instance_image_rgb.shape[1:]).cpu().numpy())
            plt.axis("off")
            plt.savefig(f"instance_map/instance_map_new_{iteration}_{viewpoint_cam.image_name}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

        temperature = 100
        main_loss = 0

        #filter out the zero instances
        instance_features = instance_features[sam_mask > 0]
        sam_mask = sam_mask[sam_mask > 0]
        if len(instance_features) == 0:
            triangles.optimizer.zero_grad(set_to_none=True)
            continue

        sample_num = opt.sample_num
        n_sample = min(int(len(sam_mask)/sample_num)+1, 1)
        index = torch.randperm(len(instance_features), device=instance_features.device)
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
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()
                
            torch.cuda.empty_cache()
            # Optimizer step
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)

            if (iteration in save_iterations):
                scene.save(f"{save_name or ''}{iteration}")          
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
