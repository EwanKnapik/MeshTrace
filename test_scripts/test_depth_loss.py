from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2

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
import colorsys
import time
import matplotlib
from utils.metric_utils import labels_and_depths, get_ref_view,get_view_ids, get_obj_by_mask, overlay_prediction
from triangle_renderer import render

def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def main():
    matplotlib.rcParams["backend"] = "Agg"
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--id", default=1, type=int)
    parser.add_argument("--num", default=2, type=int)
    parser.add_argument("--alpha_w", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=args.start_checkpoint,
                  shuffle=False)

    idx=100

    viewpoint_cam=scene.getTrainCameras()[idx]

    render_pkg = render(viewpoint_cam, triangles, pipe, background, False)


    path = "/".join((dataset.source_path,f"depth/depth_{idx}.png"))
    depth_image = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    print(depth_image)
    print(f"max :{depth_image.max()} min :{depth_image.min()} mean :{depth_image.mean()}")

    invDepth = 1.0 / (render_pkg["expected_depth"] + 1e-2)
    mono_invdepth = 1/viewpoint_cam.gt_depth.cuda()
    depth_mask = viewpoint_cam.depth_mask.cuda()
    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
    print(f"max :{invDepth.max()} min :{invDepth.min()} mean :{invDepth.mean()}")
    print(f"max :{mono_invdepth.max()} min :{mono_invdepth.min()} mean :{mono_invdepth.mean()}")
    print(f"max :{render_pkg['expected_depth'].max()} min :{render_pkg['expected_depth'].min()} mean :{render_pkg['expected_depth'].mean()}")
    print((render_pkg['expected_depth']==0).sum())
    print(Ll1depth_pure)
    print(invDepth)
    print(mono_invdepth)
    print(depth_mask)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(invDepth.permute(1,2,0).cpu().detach().numpy(),vmin=0,vmax=1)
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(mono_invdepth.permute(1,2,0).cpu().detach().numpy())
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(render_pkg["expected_depth"].permute(1,2,0).cpu().detach().numpy())
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(render_pkg["render"].permute(1,2,0).cpu().detach().numpy())
    plt.axis("off")
    sixel_fig()
    plt.close()
    #Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 




if __name__ == "__main__":
    main()
    
