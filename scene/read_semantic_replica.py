# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

from utils.graphics_utils import focal2fov
import cv2
import numpy as np
from pathlib import Path

from utils.sh_utils import SH2RGB

from PIL import Image
from scene.gaussian_model import BasicPointCloud
import torch
import torchvision.transforms as transforms
from scene.config import DEFAULT_SAM_FOLDER
from scene.dataset_readers import (
    CameraInfo,
    SceneInfo,
    fetchPly,
    getNerfppNorm,
    load_depth_params,
    read_points3D_binary,
    storePly,
)
from utils.replica_utils import build_replica_pose_map, discover_replica_rgb_frames


def get_replica_semantic_intrisic(img_h:int = 480, img_w:int = 640):
    # replica dataset from semantic nerf used a fixed fov
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = img_w / 2.0 / math.tan(math.radians(hfov / 2.0))
    fy = fx
    cx = (img_w - 1.0) / 2.0
    cy = (img_h - 1.0) / 2.0
    return fx, fy, cx, cy


def find_replica_mono_depth_dir(input_folder: Path):
    for directory_name in ("depth_DA", "depth_any", "depth_anything", "depth_pred", "mono_depth"):
        candidate = input_folder / directory_name
        if candidate.is_dir():
            return candidate
    return None


def read_semantic_ReplicaInfo(input_folder: str, image_stride:int = 1, sam_folder='origin'):
    input_folder = Path(input_folder)
    scene_name = os.path.basename(input_folder)
    traj_path = input_folder / "traj_w_c.txt"
    rgb_frames = discover_replica_rgb_frames(input_folder)

    assert len(rgb_frames) > 0, "No RGB images found at {}".format(str(input_folder / "rgb" / "rgb_*.png"))
    assert os.path.exists(traj_path), "Could not find camera trajectory at {}".format(traj_path)

    frame_ids = [frame_id for frame_id, _ in rgb_frames]
    pose_map = build_replica_pose_map(traj_path, frame_ids)
    depth_params = load_depth_params(input_folder / "sparse/0/depth_params.json")
    mono_depth_dir = find_replica_mono_depth_dir(input_folder)

    first_image = Image.open(rgb_frames[0][1])
    img_w, img_h = first_image.size
    fx, fy, cx, cy = get_replica_semantic_intrisic(img_h, img_w)
    fovx = focal2fov(fx, img_w)
    fovy = focal2fov(fy, img_h)
    transf = transforms.ToTensor()

    train_camera_infos = []
    test_camera_infos = []

    selected_frames = rgb_frames[::image_stride]
    test_positions = set(range(0, len(selected_frames), 4))
    train_frame_ids = {
        frame_id for position, (frame_id, _) in enumerate(selected_frames) if position not in test_positions
    }
    test_frame_ids = {
        frame_id for position, (frame_id, _) in enumerate(selected_frames) if position in test_positions
    }

    reject_view = set()
    if scene_name == 'office_1':
        reject_view = set(range(474, 504))
        print(f"Rejecting views for office_1: {reject_view}")
    elif scene_name == 'office_4':
        reject_view = set(range(618, 734))
        print(f"Rejecting views for office_4: {reject_view}")

    for frame_id, rgb_path in selected_frames:
        if frame_id in reject_view:
            continue

        gt_seg_path = input_folder / "semantic_instance" / f"semantic_instance_{frame_id}.png"
        image = Image.open(rgb_path)
        image_name = rgb_path.stem

        gt_depth_path = input_folder / "depth" / f"depth_{frame_id}.png"
        depth = None
        if gt_depth_path.exists():
            depth = transf(Image.open(gt_depth_path)) / 1000.0

        pose = pose_map[frame_id]
        id_masks = None
        sam_features = None

        sam_path = input_folder / DEFAULT_SAM_FOLDER / sam_folder / f"rgb_{frame_id}.npy"
        if frame_id in train_frame_ids and sam_path.exists():
            sam_mask = np.load(sam_path)
            if len(sam_mask.shape) == 3:
                N, H, W = sam_mask.shape
                sam_mask = torch.from_numpy(sam_mask)
                flat_mask = sam_mask.permute(1, 2, 0).reshape(-1, N)
                _, inverse_indices = torch.unique(flat_mask, return_inverse=True, dim = 0)
                id_masks = inverse_indices.view(H, W).cpu().numpy()
            else:
                id_masks = sam_mask

        try:
            sam_features = torch.load(input_folder / "sam_features" / f"rgb_{frame_id}.pt")
        except:
            pass

        gt_seg = None
        if os.path.exists(gt_seg_path):
            gt_seg=cv2.imread(gt_seg_path,cv2.IMREAD_UNCHANGED)

        pose = np.linalg.inv(pose)
        R = pose[:3,:3]
        R = R.T
        t = pose[:3,3]

        depth_path = ""
        depth_param = None
        if mono_depth_dir is not None:
            mono_depth_path = mono_depth_dir / f"rgb_{frame_id}.png"
            if mono_depth_path.exists():
                depth_path = str(mono_depth_path)
                if depth_params is not None:
                    depth_param = depth_params.get(f"rgb_{frame_id}")

        cam = CameraInfo(
            uid=frame_id,
            R=R,
            T=t,
            FovX=fovx,
            FovY=fovy,
            image=image,
            image_name=image_name,
            width=image.size[0],
            height=image.size[1],
            depth_params=depth_param,
            depth_path=depth_path,
            sam_mask=id_masks,
            instance_image=gt_seg,
            image_path=str(rgb_path),
            depth=depth,
            features=sam_features
        )
        if frame_id in train_frame_ids:
            train_camera_infos.append(cam)
        elif frame_id in test_frame_ids:
            test_camera_infos.append(cam)
        else:
            raise ValueError(f"Replica frame {frame_id} is not assigned to train or test split.")

    nerf_normalization = getNerfppNorm(train_camera_infos)

    ply_path = os.path.join(input_folder, "sparse/0/points3D.ply")
    bin_path = os.path.join(input_folder, "sparse/0/points3D.bin")
       
    if not os.path.exists(ply_path):
        if os.path.exists(bin_path):
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        else:
        # Since this data set has no colmap data, we start with random points
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(ply_path)
    )

    return scene_info
