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
from triangle_renderer.render_feature_new import render
from scene import Scene, TriangleModel
from scene.triangle_model import resolve_point_cloud_state_path
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    first_iter = 0
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt)

    assert checkpoint is not None, "Checkpoint path must be provided to load a model."

    checkpoint_path = resolve_point_cloud_state_path(checkpoint)
    model_params = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    triangles.restore(model_params, opt)


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

        render_pkg = render(viewpoint_cam, triangles, pipe, background, include_feature=True)
        instance_image = render_pkg["instance_image"]
        instance_features = instance_image.permute(1, 2, 0).reshape(-1, instance_image.shape[0])
        
        temperature = 100
        main_loss = 0

        sam_mask = torch.from_numpy(viewpoint_cam.sam_mask).to(torch.int).view(-1).cuda()
        sample_num = opt.sample_num
        n_sample = min(int(len(sam_mask)/sample_num)+1, 1)

        #filter out the zero instances
        instance_features = instance_features[sam_mask > 0]
        sam_mask = sam_mask[sam_mask > 0]


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
                    "Vertices": f"{triangles.get_vertices.shape[0]}",
                    "Triangles": f"{triangles.get_triangle_indices.shape[0]}"
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
