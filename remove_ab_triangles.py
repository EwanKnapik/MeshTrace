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
from random import randint
from utils.loss_utils import l1_loss, ssim
from triangle_renderer import render
from triangle_renderer.trace_triangle import trace
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.render_utils import save_img_f32, save_img_u8
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_obj_by_mask(triangles, mask, inverse=False):
    """Extract a subset of triangles by mask on triangle indices."""
    if inverse:
        mask = ~mask
    obj = TriangleModel(triangles.max_sh_degree)
    with torch.no_grad():
        obj.vertices = triangles.vertices.clone().detach().requires_grad_(False)
        obj._triangle_indices = triangles._triangle_indices[mask].clone()
        obj.vertex_weight = triangles.vertex_weight.clone().detach().requires_grad_(False)
        obj._sigma = triangles._sigma
        obj.active_sh_degree = triangles.active_sh_degree
        obj._features_dc = triangles._features_dc.clone().detach().requires_grad_(False)
        obj._features_rest = triangles._features_rest.clone().detach().requires_grad_(False)
        obj.image_size = torch.zeros((obj._triangle_indices.shape[0],), dtype=torch.float, device="cuda")
        obj.importance_score = torch.zeros((obj._triangle_indices.shape[0],), dtype=torch.float, device="cuda")
        obj.pixel_count = torch.zeros((obj._triangle_indices.shape[0],), dtype=torch.int, device="cuda")
    return obj


def render_by_mask(triangles, mask, path, viewpoints, pipe, background):
    with torch.no_grad():
        if mask is not None:
            obj = get_obj_by_mask(triangles, mask)
        else:
            obj = triangles
        for idx, viewpoint in enumerate(viewpoints):
            render_pkg = render(viewpoint, obj, pipe, background)
            img = render_pkg["render"]
            if img.sum() < 200:
                continue
            os.makedirs(path, exist_ok=True)
            save_img_u8(img.permute(1, 2, 0).cpu().numpy(),
                        os.path.join(path, 'RGB_{0:05d}'.format(idx) + ".png"))


def get_weights(triangles, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    with torch.no_grad():
        num_triangles = triangles._triangle_indices.shape[0]
        weights = torch.zeros((num_triangles, len(viewpoints)), dtype=torch.int).cuda()
        for idx, view in enumerate(viewpoints):
            sam_mask = view.sam_mask.copy()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cpu")
            id_masks = id_masks.cuda()
            id_masks[id_masks > 1] = 0
            w = trace(view, triangles, id_masks, pipe, background, alpha_w=alpha_w)
            unseen_mask = (w.sum(-1) == 0)
            w = torch.argmax(w, dim=-1)
            w[unseen_mask] = -1
            weights[:, idx] = w
    return weights


def split_mask(triangles, viewpoints, pipe, background, threshold=2, sp_th=1, soft_th=0.8, alpha_w=False):
    with torch.no_grad():
        num_triangles = triangles._triangle_indices.shape[0]
        nums = torch.zeros(num_triangles, dtype=torch.int16).cuda()
        ab_nums = torch.zeros(num_triangles, dtype=torch.int16).cuda()
        for idx, view in enumerate(viewpoints):
            sam_mask = view.sam_mask.copy()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cpu")
            id_masks = id_masks.cuda()

            w = trace(view, triangles, id_masks, id_masks.max(), pipe, background, alpha_w)
            seen = w.sum(-1) > threshold
            value, _ = torch.max(w, dim=-1)
            value = value / (w.sum(-1) + 1e-6)
            ab = (value < soft_th) & seen
            nums += seen
            ab_nums += ab

        sp_mask = (ab_nums / (nums + 1e-6)) > sp_th

    return sp_mask


def prune_mask(triangles, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    weights = get_weights(triangles, viewpoints, pipe, background, unseen, alpha_w)
    p_mask = ((weights != unseen).sum(-1) == 0)
    return p_mask.cuda()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, alpha_w):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=-1)
    triangles.training_setup(opt)
    first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ob_ratio = 0

    pcycle = opt.split_cycle_num if hasattr(opt, 'split_cycle_num') else 0
    cycle_from = opt.split_from_iter if hasattr(opt, 'split_from_iter') else 0
    cycle_interval = opt.split_cycle_interval if hasattr(opt, 'split_cycle_interval') else 1000
    scale = 0.8#越大分出来的越小
    threshold = 25       # unseen threshold
    soft_th = 0.8        # lower = less strict
    sp_th = 0.4

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    result_file = os.path.join(dataset.model_path, 'split_result.txt')
    with open(result_file, 'a') as f:
        f.write(f'\nsp_th {sp_th} cycle interval {cycle_interval}, cycle_num {pcycle}\n')
        f.write(f'Prune {getattr(opt, "prune", False)} \n')
    viewpoints = scene.getTrainCameras().copy()
    print(len(viewpoints))

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, background)
        #image, viewspace_point_tensor, visibility_filter, radii, depth= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['surf_depth']
        image = render_pkg["render"]
        depth = render_pkg['surf_depth']

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Normal regularization
        lambda_normal = opt.lambda_normals if hasattr(opt, 'lambda_normals') else 0
        lambda_dist = 0  # triangle renderer does not produce rend_dist by default

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        # loss
        total_loss = loss + normal_loss

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            num_triangles = triangles._triangle_indices.shape[0]
            num_vertices = triangles.vertices.shape[0]

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = ob_ratio

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Triangles": f"{num_triangles}",
                    "Vertices": f"{num_vertices}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if getattr(opt, 'prune', False):
                if iteration == 1 or iteration == opt.iterations:
                    with torch.no_grad():
                        p_mask = prune_mask(triangles, viewpoints, pipe, background, unseen=-1).cuda()
                        # prune_triangles uses a KEEP mask (opposite of GaussianModel)
                        triangles.prune_triangles(~p_mask)
                        print('delete {} triangles, total {} triangles now'.format(
                            p_mask.sum(), triangles._triangle_indices.shape[0]))
                        with open(result_file, 'a') as f:
                            f.write('delete {} triangles, total {} triangles now\n'.format(
                                p_mask.sum(), triangles._triangle_indices.shape[0]))

            if getattr(opt, 'split', False) and iteration >= cycle_from:
                if pcycle > 0 and (iteration - cycle_from) % cycle_interval == 0:
                    sp_mask = split_mask(triangles, viewpoints, pipe, background,
                                        threshold=threshold, sp_th=sp_th, soft_th=soft_th, alpha_w=alpha_w)
                    pre_split_num = sp_mask.sum()
                    tri_areas = triangles.triangle_areas()
                    print(f'sp_mask num:{pre_split_num} total:{num_triangles}, '
                          f'ratio {pre_split_num / num_triangles}, '
                          f'mean area {tri_areas[sp_mask].mean() if sp_mask.any() else 0}')
                    if pcycle == (opt.split_cycle_num if hasattr(opt, 'split_cycle_num') else 0):
                        with open(result_file, 'a') as f:
                            f.write(f'Before: sp_mask num:{pre_split_num} total:{num_triangles}, '
                                    f'ratio {pre_split_num / num_triangles}, '
                                    f'mean area {tri_areas[sp_mask].mean() if sp_mask.any() else 0}\n')

                    # Delete-only behavior: remove ambiguous triangles directly.
                    ob_ratio = sp_mask.sum().item()
                    triangles.prune_triangles(~sp_mask)
                    pcycle -= 1

            #if iteration == opt.iterations:
            #    sp_mask = split_mask(triangles, viewpoints, pipe, background,
            #                        threshold=threshold, sp_th=sp_th, soft_th=soft_th, alpha_w=alpha_w)
            #    print(sp_mask.sum())
            #    tri_areas = triangles.triangle_areas()
            #    with open(result_file, 'a') as f:
            #        f.write(f'After: sp_mask num:{sp_mask.sum()} total:{triangles._triangle_indices.shape[0]}, '
            #                f'ratio {sp_mask.sum() / triangles._triangle_indices.shape[0]}, '
            #                f'mean area {tri_areas[sp_mask].mean() if sp_mask.any() else 0}\n')

            #if iteration % 1000 == 0:
            #    # Current abnormal triangles
            #    sp_mask = split_mask(triangles, viewpoints, pipe, background,
            #                        threshold=threshold, sp_th=sp_th, soft_th=soft_th, alpha_w=alpha_w)
            #    tri_areas = triangles.triangle_areas()
            #    print(sp_mask.sum(), tri_areas[sp_mask].mean() if sp_mask.any() else 0)

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # Optimizer step
            #if iteration < opt.iterations:
            #    triangles.optimizer.step()
            #    triangles.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                print(len(triangles.capture()))
                print(iteration)
                scene.save(f"chkpt_{iteration}")          
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #torch.save((triangles.capture(), iteration),
                #           scene.model_path + "/chkpnt_triangles.pth")

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                           iter_start.elapsed_time(iter_end), testing_iterations,
                           scene, render, (pipe, background), result_file)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, result_file):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_triangles', scene.triangles._triangle_indices.shape[0], iteration)
        tb_writer.add_scalar('total_vertices', scene.triangles.vertices.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                           for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.triangles, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                rend_normal[None], global_step=iteration)
                            tb_writer.add_images(
                                config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                surf_normal[None], global_step=iteration)
                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                rend_alpha[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                with open(result_file, 'a') as f:
                    f.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(
                        iteration, config['name'], l1_test, psnr_test))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--alpha_w", action="store_true", help="True for alpha_w")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.test_iterations, args.checkpoint_iterations, args.alpha_w)

    # All done
    print("\nTraining complete.")
