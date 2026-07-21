from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
from triangle_renderer.render_feature import render


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

    mask=torch.ones((triangles.get_triangle_indices.shape[0]),dtype=bool)
    test=get_obj_by_mask(triangles,mask)
    render_pkg = render(scene.getTrainCameras()[0], test, pipe, background, False)



if __name__ == "__main__":
    main()
    
