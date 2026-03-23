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
from triangle_renderer import render
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def load_checkpoint_compat(checkpoint_path):
    """Load checkpoint across PyTorch versions (2.6+ defaults to weights_only=True)."""
    try:
        return torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(checkpoint_path, map_location="cuda")

def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    first_iter = 0
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)

    # Optional resume from a previously saved contrastive checkpoint.
    if checkpoint is not None:
        loaded = load_checkpoint_compat(checkpoint)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model_params, first_iter = loaded
            if isinstance(model_params, tuple) and len(model_params) == 4:
                (triangles.active_sh_degree,
                 triangles._features_dc,
                 triangles._features_rest,
                 opt_dict) = model_params
                triangles.optimizer.load_state_dict(opt_dict)
            first_iter = 0

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
        print(f"Iteration: {iteration}, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, background)
        print(f"checkpoint 1, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")

        # Build per-pixel features from rendered triangle ids.
        # Triangle features are the mean SH feature of their 3 vertices.
        rend_ids = render_pkg["rend_ids"][0].long()
        sam_mask = torch.from_numpy(viewpoint_cam.sam_mask.copy()).to(device="cuda", dtype=torch.long)
        if sam_mask.shape != rend_ids.shape:
            sam_mask = F.interpolate(
                sam_mask.float().unsqueeze(0).unsqueeze(0),
                size=rend_ids.shape,
                mode="nearest"
            ).squeeze(0).squeeze(0).long()

        triangle_features = triangles.get_features[triangles.get_triangle_indices.long()].mean(dim=1)
        triangle_features = triangle_features.reshape(triangle_features.shape[0], -1)

        print(f"checkpoint 2, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
        valid = (sam_mask > 0) & (rend_ids >= 0)
        pixel_triangle_ids = rend_ids[valid]
        pixel_labels = sam_mask[valid]

        max_triangle_id = triangle_features.shape[0] - 1
        in_bounds = pixel_triangle_ids <= max_triangle_id
        pixel_triangle_ids = pixel_triangle_ids[in_bounds]
        pixel_labels = pixel_labels[in_bounds]

        if pixel_triangle_ids.numel() > 0:
            instance_features = triangle_features[pixel_triangle_ids]
            sam_mask = pixel_labels
        else:
            instance_features = triangle_features.new_zeros((0, triangle_features.shape[1]))
            sam_mask = pixel_labels
        
        temperature = 100
        main_loss = None

        sample_num = opt.sample_num
        n_features = int(instance_features.shape[0])
        n_sample = max((n_features + sample_num - 1) // sample_num, 1)

        index = torch.randperm(n_features, device="cuda") if n_features > 0 else torch.empty(0, dtype=torch.long, device="cuda")
        n_valid_batches = 0
        print(f"checkpoint 3, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
        for sample_i in range(n_sample):
            sample_idx = index[sample_i*sample_num:(sample_i+1)*sample_num]
            if sample_idx.numel() < 2:
                continue
            features = instance_features[sample_idx]
            print(f"features: {features.shape}")
            instance_labels = sam_mask[sample_idx]
            con_loss = contrastive_func(features, instance_labels, temperature)
            main_loss = con_loss if main_loss is None else (main_loss + con_loss)
            print(f"checkpoint 3.5, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
            n_valid_batches += 1

        print(f"checkpoint 4, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
        if n_valid_batches > 0:
            loss = main_loss / n_valid_batches
        else:
            # Keep graph valid even if no labeled/rendered pixels are available.
            loss = triangles.get_features.sum() * 0.0

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
            print(f"checkpoint 5, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")

            if iteration == opt.iterations:
                progress_bar.close()
                
            torch.cuda.empty_cache()
            # Optimizer step
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
            print(f"checkpoint 6, memory allocated:{torch.cuda.memory_allocated()/(2**30)} GiB")
            
            if (iteration in save_iterations):
                save_path=os.path.join(scene.model_path, dataset.sam_folder, 'chkpnt')
                os.makedirs(save_path, exist_ok=True)
                save_path= os.path.join(save_path, save_name + str(iteration) + ".pth") #TODO distinguish load chkpnt by load_iter
                torch.save((triangles.capture(), iteration),save_path)
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
