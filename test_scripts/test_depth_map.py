import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from argparse import ArgumentParser
from sixel import sixel
from io import BytesIO

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
import tifffile
from glob import glob

from utils.read_write_model import qvec2rotmat, read_model

try:
    from utils.depth_utils import load_depth_image
    from utils.replica_utils import build_replica_pose_map, discover_replica_rgb_frames
    from utils.klevr_utils import build_klevr_pose_map, discover_klevr_train_frames
except ModuleNotFoundError:
    from depth_utils import load_depth_image
    from replica_utils import build_replica_pose_map, discover_replica_rgb_frames
    from klevr_utils import build_klevr_pose_map, discover_klevr_train_frames

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def robust_scale_and_offset(reference_values, predicted_values):
    if reference_values.size <= 10:
        return 0.0, 0.0

    reference_median = np.median(reference_values)
    predicted_median = np.median(predicted_values)

    reference_deviation = np.mean(np.abs(reference_values - reference_median))
    predicted_deviation = np.mean(np.abs(predicted_values - predicted_median))

    if reference_deviation <= 1e-6 or predicted_deviation <= 1e-6:
        return 0.0, 0.0

    scale = reference_deviation / predicted_deviation
    offset = reference_median - predicted_median * scale
    return float(scale), float(offset)

def get_replica_scales(key, depth_gt_path, mono_depth_path):
    image_name = Path(mono_depth_path).stem

    gt_depth_map = 0.01/load_depth_image(Path(depth_gt_path))
    inv_mono_depth = load_depth_image(Path(mono_depth_path))
    print(gt_depth_map)
    print(inv_mono_depth)

    #plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.imshow(gt_depth_map)
    #plt.axis("off")
    #plt.subplot(1, 2, 2)
    #plt.imshow(inv_mono_depth)
    #plt.axis("off")
    #sixel_fig()
    #plt.close()


    if inv_mono_depth is None:
        return None

    if inv_mono_depth.shape[0] != gt_depth_map.shape[0]:
        print("mismatch in depth map size")
        return None

    scale, offset = robust_scale_and_offset(gt_depth_map, inv_mono_depth)
    return {"image_name": image_name, "scale": scale, "offset": offset}

def detect_dataset_type(base_dir, dataset_type):
    if dataset_type != "auto":
        return dataset_type

    if (Path(base_dir) / "traj_w_c.txt").exists():
        return "replica"
    if (Path(base_dir) / "metadata.json").exists():
        return "klevr"
    if (Path(base_dir) / "transforms_train.json").exists():
        return "blender"
    return "colmap"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument("--depths_dir", default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument("--model_type", default="bin")
    parser.add_argument("--dataset_type", choices=("auto", "colmap", "replica", "blender", "klevr"), default="auto")
    args = parser.parse_args()

    dataset_type = detect_dataset_type(args.base_dir, args.dataset_type)

    if dataset_type == "replica":
        gt_depth_path = glob_data(os.path.join(args.depths_dir , f"depth_*.png"))
        mono_depth_path = glob_data(os.path.join(args.depths_dir , f"rgb_*.png"))
        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_replica_scales)(
                key,
                gt_depth_path[key],
                mono_depth_path[key]
            )
            #for key in range(len(gt_depth_path))
            for key in range(1)
        )