import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RGB_FRAME_PATTERN = re.compile(r"^rgb_(\d+)\.png$")


def extract_frame_id(path_or_name, pattern=RGB_FRAME_PATTERN) -> int:
    name = Path(path_or_name).name
    match = pattern.match(name)
    if match is None:
        raise ValueError(f"Could not extract Replica frame id from '{name}'.")
    return int(match.group(1))


def discover_replica_rgb_frames(input_folder) -> List[Tuple[int, Path]]:
    rgb_dir = Path(input_folder) / "rgb"
    rgb_paths = sorted(rgb_dir.glob("rgb_*.png"), key=extract_frame_id)
    return [(extract_frame_id(rgb_path), rgb_path) for rgb_path in rgb_paths]


def load_replica_poses(traj_path) -> np.ndarray:
    poses = np.loadtxt(Path(traj_path))
    if poses.size == 0:
        raise ValueError(f"Replica trajectory file '{traj_path}' is empty.")
    if poses.size % 16 != 0:
        raise ValueError(
            f"Replica trajectory file '{traj_path}' has {poses.size} values, expected a multiple of 16."
        )
    return poses.reshape(-1, 4, 4)


def build_replica_pose_map(traj_path, frame_ids) -> Dict[int, np.ndarray]:
    ordered_frame_ids = sorted(frame_ids)
    poses = load_replica_poses(traj_path)

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
