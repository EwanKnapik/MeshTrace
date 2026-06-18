import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import json

def extract_frame_id_klevr(path_or_name) -> int:
    name = Path(path_or_name).name
    return int(name.split("/")[-1].split("_")[-1].split(".")[0])

def discover_klevr_train_frames(input_folder) -> List[Tuple[int,Path]]:
    train_dir = Path(input_folder) / "train"
    train_paths = sorted(train_dir.glob("r_*.png"), key=extract_frame_id_klevr)
    return [(extract_frame_id_klevr(train_path),train_path) for train_path in train_paths]

def build_klevr_pose_map(traj_path, frame_ids) -> Dict[int, np.ndarray]:
    return

def load_klevr_poses(traj_path) -> np.ndarray:
    with open(Path(traj_path), 'r') as f:
        poses_json = json.load(f)
    poses=[]
    for frame in poses_json["frames"]:
        poses.append(frame["transform_matrix"])
    poses=np.array(poses)
    if poses.size == 0:
        raise ValueError(f"Replica trajectory file '{traj_path}' is empty.")
    if poses.size % 16 != 0:
        raise ValueError(
            f"Replica trajectory file '{traj_path}' has {poses.size} values, expected a multiple of 16."
        )
    return poses.reshape(-1, 4, 4)

def build_klevr_pose_map(traj_path, frame_ids) -> Dict[int, np.ndarray]:
    ordered_frame_ids = sorted(frame_ids)
    poses = load_klevr_poses(traj_path)

    if not ordered_frame_ids:
        return {}

    if max(ordered_frame_ids) < len(poses):
        return {frame_id: poses[frame_id] for frame_id in ordered_frame_ids}

    if len(ordered_frame_ids) == len(poses):
        return {frame_id: poses[idx] for idx, frame_id in enumerate(ordered_frame_ids)}

    raise ValueError(
        "Replica trajectory/image mismatch: "
        f"{len(poses)} poses for frame ids in [{ordered_frame_ids[0]}, {ordered_frame_ids[-1]}]."
    )