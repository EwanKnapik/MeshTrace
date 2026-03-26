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


def _optimizer_state_matches_params(optimizer):
    """Return False if Adam state tensors do not match current parameter shapes."""
    for group in optimizer.param_groups:
        if not group.get("params"):
            continue
        param = group["params"][0]
        state = optimizer.state.get(param, None)
        if state is None:
            continue
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)
        if exp_avg is None or exp_avg_sq is None:
            continue
        if exp_avg.shape != param.shape or exp_avg_sq.shape != param.shape:
            return False
    return True


def _maybe_load_checkpoint(triangles, checkpoint, opt):
    """Load compatible checkpoint tensors and optimizer state when shapes match."""
    if checkpoint is None:
        return

    loaded = load_checkpoint_compat(checkpoint)
    if not (isinstance(loaded, tuple) and len(loaded) == 2):
        print(f"[WARN] Unexpected checkpoint format in {checkpoint}. Skipping load.")
        return

    model_params, _ = loaded
    if not (isinstance(model_params, tuple) and len(model_params) == 4):
        print(f"[WARN] Unsupported model tuple in checkpoint {checkpoint}. Skipping load.")
        return

    active_sh_degree, features_dc, features_rest, opt_dict = model_params
    current_vertex_count = triangles.get_vertices.shape[0]

    if features_dc.shape[0] != current_vertex_count or features_rest.shape[0] != current_vertex_count:
        print(
            "[WARN] Checkpoint feature count does not match current geometry "
            f"({features_dc.shape[0]} vs {current_vertex_count}). Skipping checkpoint load."
        )
        return

    triangles.active_sh_degree = active_sh_degree
    triangles._features_dc = torch.nn.Parameter(features_dc.detach().contiguous().requires_grad_(True))
    triangles._features_rest = torch.nn.Parameter(features_rest.detach().contiguous().requires_grad_(True))

    # Rebuild optimizer so it tracks the loaded feature tensors.
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)

    try:
        triangles.optimizer.load_state_dict(opt_dict)
    except Exception as e:
        print(f"[WARN] Could not load optimizer state from {checkpoint}: {e}. Reinitializing optimizer.")
        triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
        return

    if not _optimizer_state_matches_params(triangles.optimizer):
        print(
            f"[WARN] Optimizer state in {checkpoint} is incompatible with current parameter shapes. "
            "Reinitializing optimizer."
        )
        triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)

def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    # def var for optimization
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    
    need_delaunay = False

    run_restricted_delaunay = opt.densify_until_iter + 1000





    first_iter = 0
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)

    # OLD CODE
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

    # GENERATED CODE
    # Optional resume/initialization from a previously saved contrastive checkpoint.
    #_maybe_load_checkpoint(triangles, checkpoint, opt)



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
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False
        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        number_of_training_views = len(viewpoint_stack)

        render_pkg = render(viewpoint_cam, triangles, pipe, background)

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
        sample_num = opt.sample_num
        n_features = int(instance_features.shape[0])

        if n_features > 1:
            # Use a single random contrastive mini-batch per iteration.
            # This avoids retaining many batch graphs before backward.
            if n_features > sample_num:
                sample_idx = torch.randperm(n_features, device="cuda")[:sample_num]
                features = instance_features[sample_idx]
                instance_labels = sam_mask[sample_idx]
            else:
                features = instance_features
                instance_labels = sam_mask

            loss = contrastive_func(features, instance_labels, temperature)
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
            if iteration == opt.iterations:
                progress_bar.close()
                
            torch.cuda.empty_cache()
            # Optimizer step
            # Log and save
            

            # Handle pruning operations
            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                
                # Building masks to delete triangles
                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                ) 
                min_weights = triangle_vertex_weights.min(dim=1).values

                mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                mask_importance  = (triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                mask_size        = (triangles.image_size > prune_size).squeeze()                 # delete if too big

                delete_mask = mask_opacity | mask_size

                if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance

                keep_mask   = ~delete_mask 

                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
             
                # We prune vertices that are no longer used
                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                            dtype=torch.bool, 
                                            device=device)
                if triangles._triangle_indices.numel() > 0:
                    flat_indices = triangles._triangle_indices.flatten()
                    used_vertex_mask[flat_indices] = True
                
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                print(f"shape of mask_out:{triangles.get_vertex_weight.shape}")
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                triangles._prune_vertices(vertex_mask)


                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                )  # [T,3]

                needs_densification = (iteration < opt.densify_until_iter and 
                                     iteration % opt.densification_interval == 0 and 
                                     iteration > opt.densify_from_iter)
                
                if needs_densification:
                    triangles.add_new_gs(iteration, cap_max=opt.max_points, splitt_large_triangles=splitt_large_triangles)
   

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
            
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
