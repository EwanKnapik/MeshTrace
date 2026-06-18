import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
import tifffile

from read_write_model import qvec2rotmat, read_model

try:
    from utils.depth_utils import load_depth_image
    from utils.replica_utils import build_replica_pose_map, discover_replica_rgb_frames
    from utils.klevr_utils import build_klevr_pose_map, discover_klevr_train_frames
except ModuleNotFoundError:
    from depth_utils import load_depth_image
    from replica_utils import build_replica_pose_map, discover_replica_rgb_frames
    from klevr_utils import build_klevr_pose_map, discover_klevr_train_frames


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


def get_colmap_scales(key, cameras, images, points3d_ordered, depths_dir):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids
    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) == 0:
        return None

    pts = points3d_ordered[pts_idx]
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    valid_depth = pts[..., 2] > 1e-6
    if valid_depth.sum() <= 10:
        return None

    inv_colmap_depth = 1.0 / pts[valid_depth, 2]
    valid_xys = valid_xys[valid_depth]

    image_name = Path(image_meta.name).stem
    inv_mono_depth_map = load_depth_image(Path(depths_dir) / f"{image_name}.png")
    if inv_mono_depth_map is None:
        return None

    scale_factor = inv_mono_depth_map.shape[0] / cam_intrinsic.height
    maps = (valid_xys * scale_factor).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0)
        & (maps[..., 1] >= 0)
        & (maps[..., 0] < cam_intrinsic.width * scale_factor)
        & (maps[..., 1] < cam_intrinsic.height * scale_factor)
    )
    if valid.sum() <= 10:
        return None

    maps = maps[valid]
    inv_colmap_depth = inv_colmap_depth[valid]
    inv_mono_depth = cv2.remap(
        inv_mono_depth_map,
        maps[..., 0],
        maps[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)

    if (inv_colmap_depth.max() - inv_colmap_depth.min()) <= 1e-3:
        return None

    scale, offset = robust_scale_and_offset(inv_colmap_depth, inv_mono_depth)
    return {"image_name": image_name, "scale": scale, "offset": offset}


def get_replica_scales(frame_id, base_dir, depths_dir):
    gt_depth_path = Path(base_dir) / "depth" / f"depth_{frame_id}.png"
    mono_depth_path = Path(depths_dir) / f"rgb_{frame_id}.png"

    gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED)
    inv_mono_depth = load_depth_image(mono_depth_path)
    if gt_depth is None or inv_mono_depth is None:
        return None

    gt_depth = gt_depth.astype(np.float32) / 1000.0
    if inv_mono_depth.shape != gt_depth.shape:
        inv_mono_depth = cv2.resize(
            inv_mono_depth,
            (gt_depth.shape[1], gt_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    valid = (gt_depth > 1e-6) & np.isfinite(inv_mono_depth) & (inv_mono_depth > 0)
    if valid.sum() <= 100:
        return None

    inv_gt_depth = 1.0 / gt_depth[valid]
    inv_mono_depth = inv_mono_depth[valid]
    scale, offset = robust_scale_and_offset(inv_gt_depth, inv_mono_depth)
    return {"image_name": f"rgb_{frame_id}", "scale": scale, "offset": offset}

def get_klevr_scales(frame, base_dir, depths_dir):
    frame_id, image_path = frame
    image_name = image_path.stem          # "r_0"
    split = image_path.parent.name        # "train", "test", "val"

    # Prefer true float depth if available.
    gt_depth_path = Path(base_dir) / "tiff_depth" / f"{split}_r_{frame_id}_depth_0000.tiff"
    if gt_depth_path.exists():
        gt_depth = tifffile.imread(gt_depth_path).astype(np.float32)
    else:
        gt_depth_path = Path(base_dir) / "depth" / f"r_{frame_id}.png"
        gt_depth = load_depth_image(gt_depth_path)

    mono_depth_path = Path(depths_dir) / f"r_{frame_id}.png"
    inv_mono_depth = load_depth_image(mono_depth_path)

    if gt_depth is None or inv_mono_depth is None:
        return None

    if inv_mono_depth.shape != gt_depth.shape:
        inv_mono_depth = cv2.resize(
            inv_mono_depth,
            (gt_depth.shape[1], gt_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    valid = (
        np.isfinite(gt_depth)
        & np.isfinite(inv_mono_depth)
        & (gt_depth > 1e-6)
        & (inv_mono_depth > 0)
        & (gt_depth < 1e8)   # useful for huge TIFF background values
    )
    if valid.sum() <= 100:
        return None

    inv_gt_depth = 1.0 / gt_depth[valid]
    inv_mono_depth = inv_mono_depth[valid]

    scale, offset = robust_scale_and_offset(inv_gt_depth, inv_mono_depth)
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

    if dataset_type == "colmap":
        cam_intrinsics, images_metas, points3d = read_model(
            os.path.join(args.base_dir, "sparse", "0"),
            ext=f".{args.model_type}",
        )

        pts_indices = np.array([points3d[key].id for key in points3d])
        pts_xyzs = np.array([points3d[key].xyz for key in points3d])
        points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
        points3d_ordered[pts_indices] = pts_xyzs

        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_colmap_scales)(
                key,
                cam_intrinsics,
                images_metas,
                points3d_ordered,
                args.depths_dir,
            )
            for key in images_metas
        )
    elif dataset_type == "replica":
        rgb_frames = discover_replica_rgb_frames(args.base_dir)
        frame_ids = [frame_id for frame_id, _ in rgb_frames]
        build_replica_pose_map(Path(args.base_dir) / "traj_w_c.txt", frame_ids)

        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_replica_scales)(frame_id, args.base_dir, args.depths_dir)
            for frame_id in frame_ids
        )
    elif dataset_type == "klevr":
        klevr_frames = discover_klevr_train_frames(args.base_dir)
        frame_ids = [frame_id for frame_id, _ in klevr_frames]
        build_klevr_pose_map(Path(args.base_dir) / "transforms_train.json", frame_ids)

        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_klevr_scales)(frame,args.base_dir, args.depths_dir)
            for frame in klevr_frames
        )

    depth_params = {
        depth_param["image_name"]: {
            "scale": depth_param["scale"],
            "offset": depth_param["offset"],
        }
        for depth_param in depth_param_list
        if depth_param is not None
    }

    output_dir = Path(args.base_dir) / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"Wrote {len(depth_params)} depth parameter entries to {output_dir / 'depth_params.json'}")
