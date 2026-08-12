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
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# mask of triangles to delete
def split_mask(triangles, viewpoints, pipe, background, sp_th=0.1, soft_th=1.0):
    with torch.no_grad():
        nums = torch.zeros(triangles.get_triangle_indices.shape[0], dtype=torch.int16).cuda()
        ab_nums = torch.zeros(triangles.get_triangle_indices.shape[0], dtype=torch.int16).cuda()
        for idx, view in enumerate(viewpoints):
            sam_mask = view.sam_mask.copy()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cuda")
            
            w = trace(view, triangles, id_masks, pipe, background)
            seen = w.sum(-1) > 0
            value, _ = torch.max(w, dim=-1)
            if (value[seen]).min() !=0:
                print((value[seen]).min())
                print((value[seen]).max())
            ab = (value < soft_th) & seen
            nums += seen
            ab_nums += ab
        # number of time triangle seen with ambiguous value / number of time seen
        # if ratio is too high, means that triangle seen most of the time with ambiguous value
        # if across all views: bad means that generally bad → flag
        print(f"ab nums:{ab_nums.sum()}, nums:{nums.sum()}")
        sp_mask = (ab_nums / (nums + 1e-6)) > sp_th

    return sp_mask

def prune_mask(triangles, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    with torch.no_grad():
        num_triangles = triangles.get_triangle_indices.shape[0]
        mask=torch.zeros(num_triangles).cuda()
        for idx, view in enumerate(viewpoints):
            render_pkg = render(view, triangles, pipe, background)
            was_rendered=render_pkg["triangle_was_rendered"]
            mask+= was_rendered
        return (mask!=0)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, alpha_w):

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    soft_th = opt.soft_th        # lower = less strict
    sp_th = opt.sp_th


    viewpoints = scene.getTrainCameras().copy()


    #p_mask = prune_mask(triangles, viewpoints, pipe, background, unseen=-1).cuda()
    #triangles.prune_triangles(p_mask)
    step=5

    res=torch.zeros(step,step)

    print("")
    for i in range(1,step+1):
        for j in range(1,step+1):
            print(f"\r step {i*j}")
            sp_th=i/step
            soft_th=j/step
            sp_mask = split_mask(triangles, viewpoints, pipe, background,sp_th=i,soft_th=j)

            res[i:j]=sp_mask.sum()

    print(res)





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

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.test_iterations, args.checkpoint_iterations, args.alpha_w)
