from pathlib import Path
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.con_mask_utils import SegmentationMask, align_instance_ids_across_views
from utils.render_utils import save_img_u8
from triangle_renderer.trace_triangle import trace
from triangle_renderer import TriangleModel
from scene import Scene
from conf.con_masks_conf import *
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


class MaskRepairPipeline:
    """Mask repair pipeline adapted for triangle splatting."""

    def __init__(self, dataset_path: str, scene: Scene, triangles: TriangleModel):
        self.dataset_path = Path(dataset_path)
        self.scene_path = dataset_path
        self.scene = scene
        self.triangles = triangles

        self.camera_stack = scene.getTrainCameras().copy()
        self.setup_directories()
        self.train_idx = None

    def setup_directories(self) -> None:
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        for folder in [SPLIT_FOLDER, COMPARE_FOLDER]:
            (sam_path / folder).mkdir(parents=True, exist_ok=True)

    def get_train_indices(self, sam_paths: List[Path]) -> np.ndarray:
        return np.arange(len(sam_paths))

    def compute_weights(self,
                        masks: List[SegmentationMask],
                        camera_stack: List,
                        pipe,
                        background: torch.Tensor,
                        alpha_w: bool = False,
                        mask_type: str = 'mask') -> torch.Tensor:
        """Compute per-triangle per-view class weights via rasterization tracing."""
        num_triangles = self.triangles._triangle_indices.shape[0]
        weights = torch.zeros((num_triangles, len(camera_stack))).cuda()

        for idx, mask in enumerate(masks):
            if mask_type == 'union':
                p_mask = mask.pre_union_mask.to(torch.int)
            else:
                p_mask = mask.mask.to(torch.int)
            view = camera_stack[mask.view]
            pred, total = trace(
                view,
                self.triangles,
                p_mask,
                p_mask.max(),
                pipe,
                background,
                alpha_w,
                return_assignment=True,
            )
            pred[total == 0] = UNSEEN_VALUE
            weights[:, idx] = pred

        return weights

    def repair_masks(self, pipe, background: torch.Tensor, alpha_w: bool = False) -> None:
        sam_paths = list(self.dataset_path.glob(f"{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/*.npy"))
        print(f"Found {len(sam_paths)} SAM masks")
        self.train_idx = self.get_train_indices(sam_paths)
        masks = []
        for idx, camera in tqdm(enumerate(self.camera_stack), desc="Loading masks"):
            sam_data = np.load(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{camera.image_name}.npy")
            sam_tensor = torch.from_numpy(sam_data).cuda().squeeze()
            sam_tensor = sam_tensor[sam_tensor.sum((-2, -1)) > 96].squeeze()
            mask = SegmentationMask(sam_tensor, view=idx, image_name=camera.image_name)
            masks.append(mask)

        for iteration in range(1):
            self._repair_iteration(masks, iteration, pipe, background)

    def _repair_iteration(self,
                          masks: List[SegmentationMask],
                          iteration: int,
                          pipe,
                          background: torch.Tensor,
                          alpha_w: bool = False) -> None:
        num = 0
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        weights = self.compute_weights(masks, self.camera_stack, pipe, background, alpha_w)
        os.makedirs(sam_path / COMPARE_FOLDER / f'iter={iteration}', exist_ok=True)

        for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Repairing masks"):
            mask.repair(weights, miou_th=0.4)
            num += mask.repaired_num

        aligned_weights = self.compute_weights(masks, self.camera_stack, pipe, background, alpha_w)
        align_instance_ids_across_views(masks, aligned_weights)

        for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):

            if len(mask.rp_iou) > 0:
                miou = torch.stack(mask.rp_iou).mean().item()
            else:
                miou = 0
            plt.imsave(
                sam_path / COMPARE_FOLDER / f'iter={iteration}' / '{}_{}_{:.2f}.png'.format(
                    mask.image_name, mask.repaired_num, miou),
                mask.compare_mask().cpu().numpy()
            )
            np.save(
                sam_path / SPLIT_FOLDER / f'{mask.image_name}.npy',
                mask.mask.cpu().numpy()
            )
        print(f"Repaired {num} mask regions across all views")


def main():
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--alpha_w", action="store_true")
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=-1,
                  shuffle=False)

    repair_pipeline = MaskRepairPipeline(args.source_path, scene, triangles)
    repair_pipeline.repair_masks(pipe, background, args.alpha_w)


if __name__ == "__main__":
    main()
    
