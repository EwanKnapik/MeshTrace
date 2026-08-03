from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.con_mask_utils import SegmentationMask
from utils.render_utils import save_img_u8
from triangle_renderer.trace_triangle import trace, compressed_trace
from scene import Scene
from scene.triangle_model import TriangleModel
from conf.con_masks_conf import *  
from argparse import ArgumentParser
from triangle_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
import torch.nn.functional as F


class MaskRepairPipeline:

    
    def __init__(self, dataset_path: str, scene: Scene, triangles: TriangleModel, precomp_id_map:bool, dataset=ModelParams):
        self.dataset_path = Path(dataset_path)
        self.scene_path = dataset_path
        self.scene = scene
        self.triangles = triangles

        test_cameras = scene.getTestCameras().copy()
        train_cameras = scene.getTrainCameras().copy()
        self.camera_train_stack = train_cameras
        self.camera_test_stack = test_cameras
        self.camera_images_stack = test_cameras + train_cameras

        scene_root = self.dataset_path.parent if self.dataset_path.name in {"train", "test", "images", "rgb"} else self.dataset_path
        self.scene_name = scene_root.name
        self.precomp_id_maps=precomp_id_map

        self.dataset = None
        self.setup_directories()
        self.train_idx = None
        dataset_path_lower = dataset_path.lower()
        if 'llff' in dataset_path_lower:
            self.dataset = 'llff'
            self.sam_name = self.nvos_sam
        elif 'replica' in dataset_path_lower:
            self.sam_name = self.replica_sam
            self.dataset = 'replica'
        elif 'synthetic' in dataset_path_lower:
            self.sam_name = self.replica_sam
            self.dataset = 'blender'
        else:
            self.sam_name = self.replica_sam
            self.dataset = 'colmap'
        
    def setup_directories(self) -> None:
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        for folder in [SPLIT_FOLDER, COMPARE_FOLDER]:
            (sam_path / folder).mkdir(parents=True, exist_ok=True)
            
    def replica_sam(self,image_name):
        # image_id=int(image_name[4:])
        # return f"{image_id:05d}_masks_sam.npy"
        return f"{image_name}.npy"
        #return f"{image_name}_sam.npy"
    
    def nvos_sam(self,image_name):
        return f"{image_name}.npy"
    
    def get_train_indices(self, sam_paths: List[Path]) -> np.ndarray:

        if self.dataset == 'replica':
            all_idx = np.arange(900)
            test_idx = np.arange(0, 900, 4)
            train_idx = np.setdiff1d(all_idx, test_idx)
            if self.scene_name == 'office_1':
                reject_idx = np.arange(474, 504) 
                return np.setdiff1d(train_idx, reject_idx)  
            elif self.scene_name == 'office_4':
                reject_idx = np.arange(618, 734)
                return np.setdiff1d(train_idx, reject_idx)
        elif self.dataset == 'llff':
            test_idx = 1 if 'fern' in self.scene_path  \
            else 8 if 'horns' in self.scene_path \
            else 13 if 'orchids' in self.scene_path \
            else 31 if 'trex' in self.scene_path \
            else 0 
            train_idx = np.arange(len(sam_paths))
            return np.setdiff1d(train_idx, test_idx)
        else:
            train_idx = np.arange(len(sam_paths))
        return train_idx

    def _get_camera_stack(self, eval_mode: bool, directory: str) -> List:
        if not eval_mode:
            return self.camera_images_stack

        normalized_directory = directory.lower()
        if normalized_directory == "train":
            return self.camera_train_stack
        if normalized_directory == "test":
            return self.camera_test_stack
        if normalized_directory in {"images", "rgb"}:
            return self.camera_images_stack

        raise ValueError(f"Unsupported image directory '{directory}' for mask repair")
    

    def trace_from_file(self, viewpoint_camera, pc: TriangleModel, id_masks: torch.Tensor, pipe):
        view_name = getattr(viewpoint_camera, "image_name", "<unknown>")


        source="/".join(self.dataset_path.parts[1:-1])
        rend_ids = torch.from_numpy(np.load(f"/{source}/pre_comp_id_maps/{view_name}.npy")).cuda()


        num_triangles = pc.get_triangle_indices.shape[0]
        max_mask_id = int(id_masks.max().item()) if id_masks.numel() > 0 else 0
        num_mask_ids = max(max_mask_id + 1, 1)

        if id_masks.shape != rend_ids.shape:
            print(f"{'!' * 10} sam_mask not same shape as rend_ids {'!' * 10}")

        num_triangles = pc.get_triangle_indices.shape[0]
        in_bounds = (rend_ids >= 0) & (rend_ids < num_triangles)

        flat_rend_ids = rend_ids[in_bounds].reshape(-1).long()
        flat_masks = id_masks[in_bounds].reshape(-1).long()

        valid_mask_labels = flat_masks >= 0
        flat_rend_ids = flat_rend_ids[valid_mask_labels]
        flat_masks = flat_masks[valid_mask_labels]

        max_mask_id = int(flat_masks.max().item()) if flat_masks.numel() > 0 else 0
        num_mask_ids = max(max_mask_id + 1, 1)

        weights = torch.zeros((num_triangles, num_mask_ids), device=id_masks.device, dtype=torch.float32)
        if flat_rend_ids.numel() > 0:
            linear_idx = flat_rend_ids * num_mask_ids + flat_masks
            counts = torch.bincount(linear_idx, minlength=num_triangles * num_mask_ids).float()
            counts = counts.view(num_triangles, num_mask_ids)
            denom = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
            weights = counts / denom
        return weights


        
    def compute_weights(self, 
                       masks: List[SegmentationMask],
                       camera_stack: List,
                       pipe,
                       background: torch.Tensor,
                       alpha_w:bool=False,
                       mask_type:str='mask') -> torch.Tensor:
        weights = torch.zeros((self.triangles.get_triangle_indices.shape[0], 
                             len(camera_stack))).cuda()#[p,view]
        
        with torch.no_grad():
            for idx, mask in enumerate(masks):
                if mask_type=='union':
                    p_mask = mask.pre_union_mask.to(torch.int)
                else:
                    p_mask = mask.mask.to(torch.int)
                view = camera_stack[mask.view]
                if self.precomp_id_maps:
                    w = self.trace_from_file(
                        view,
                        self.triangles,
                        p_mask,
                        pipe
                    )
                else:
                    #w = trace(
                    #    view,
                    #    self.triangles,
                    #    p_mask,
                    #    pipe,
                    #    background,
                    #    alpha_w,
                    #)
                    w = compressed_trace(
                        view,
                        self.triangles,
                        p_mask,
                        pipe,
                        background,
                        alpha_w,
                    )
                weights[:, idx] = w
            
        return weights
        
    def repair_masks(self, pipe, background: torch.Tensor, dataset, directory, alpha_w:bool=False) -> None:
        sam_paths = sorted(self.dataset_path.glob(f"{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/*.npy"))
        camera_stack = self._get_camera_stack(dataset.eval, directory)
        self.train_idx = self.get_train_indices(sam_paths)
        masks = []
        for idx, camera in tqdm(enumerate(camera_stack)):
            if self.dataset=='replica':
                sam_data = np.load(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{self.sam_name(camera.image_name)}")
            elif self.dataset=='llff':
                sam_data = np.load(sam_paths[self.train_idx[idx]])
            elif self.dataset=='colmap':
                sam_data = np.load(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{self.sam_name(camera.image_name)}")
                print(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{self.sam_name(camera.image_name)}")
            elif self.dataset=='blender':
                sam_data = np.load(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{self.sam_name(camera.image_name)}")
            sam_tensor = torch.from_numpy(sam_data).cuda().squeeze().unsqueeze(0)

            orig_h,orig_w = sam_tensor.shape[2:]

            if camera.resolution in [1, 2, 4, 8]:
                resolution = round(orig_w/(camera.resolution)), round(orig_h/(camera.resolution))
            else:  # should be a type that converts to float
                if camera.resolution == -1:
                    if orig_w > pipe.rescale_res:
                        global_down = orig_w / pipe.rescale_res
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / camera.resolution

                scale = float(global_down)
                resolution = (int(orig_w / scale), int(orig_h / scale))
            
            sam_tensor=F.interpolate(sam_tensor.float(),size=resolution[::-1]).squeeze()>0.5

            mask = SegmentationMask(sam_tensor, view=idx, image_name= camera.image_name)
            # mask.pre_process(sam_tensor, 0.001)
            masks.append(mask)

        for iteration in range(2):
            self._repair_iteration(masks, iteration,pipe,background, camera_stack)

    def _repair_iteration(self, 
                         masks: List[SegmentationMask],
                         iteration,
                         pipe, 
                         background: torch.Tensor, 
                         camera_stack,
                         alpha_w:bool=False ,
                         ) -> None:
        num = 0
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        weights = self.compute_weights(masks, camera_stack, pipe, background, alpha_w)
        os.makedirs(sam_path / COMPARE_FOLDER/f'iter={iteration}',exist_ok=True)
        for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Repairing masks"):
            mask.repair(weights, miou_th=0.4)
            num+=mask.repaired_num

        for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):
            
            if len(mask.rp_iou) > 0:
                miou = torch.stack(mask.rp_iou).mean().item()
            else:
                miou = 0
            plt.imsave(
                sam_path / COMPARE_FOLDER/f'iter={iteration}' / '{}_{}_{:.2f}.png'.format(mask.image_name,mask.repaired_num,miou),
                mask.compare_mask().cpu().numpy()  
            )
            
            np.save(
                sam_path / SPLIT_FOLDER / f'{mask.image_name}.npy',
                mask.mask.cpu().numpy()
            )
    

    def pre_compute_id_maps(self, 
                        dataset,
                       pipe,
                       background: torch.Tensor,
                       alpha_w:bool=False,
                       mask_type:str='mask'):
        camera_stack=self.camera_train_stack
        pc=self.triangles
        source=dataset.source_path
        save_path = os.path.join(source,"pre_comp_id_maps")
        os.makedirs(save_path, exist_ok=True)
        for i,cam in enumerate(camera_stack):
            render_pkg = render(cam, pc, pipe, background)
            view_name = getattr(cam, "image_name", "<unknown>")

            rend_ids = render_pkg["rend_ids"][0].long()

            np.save(os.path.join(save_path, f'{view_name}.npy'), rend_ids.cpu())

            

def main():

    parser = ArgumentParser(description="merge patches")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_pre", action="store_false")
    parser.add_argument("--include_feature", action="store_false")
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--alpha_w", action="store_true")
    parser.add_argument("--large_scene", default=False)
    parser.add_argument("--precomp_id_map", default=False)
    args = get_combined_args(parser)
    print(args._get_args)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    triangles = TriangleModel(dataset.sh_degree)
    dataset.sam_folder = "empty" #prevent sam loading
    if dataset.start_checkpoint:
        scene = Scene(dataset, triangles, init_opacity=None, set_sigma=None, shuffle=False, load_iteration=dataset.start_checkpoint)
    else:
        scene = Scene(dataset, triangles, init_opacity=None, set_sigma=None, shuffle=False, load_iteration=-1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    source_path = Path(args.source_path)
    candidate_directories = [
        directory for directory in ("train", "test", "images", "rgb")
        if (source_path / directory / DEFAULT_SAM_FOLDER / ORIGIN_FOLDER).is_dir()
    ]

    if not candidate_directories and (source_path / DEFAULT_SAM_FOLDER / ORIGIN_FOLDER).is_dir():
        candidate_directories = ["images"]

    if not candidate_directories:
        raise FileNotFoundError(
            f"Could not find any SAM mask directories under '{source_path}'. "
            f"Expected one of train/test/images/rgb/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}."
        )

    for directory in candidate_directories:
        dataset_path = source_path if directory == "images" and (source_path / DEFAULT_SAM_FOLDER / ORIGIN_FOLDER).is_dir() else source_path / directory
        pipeline = MaskRepairPipeline(str(dataset_path), scene, triangles, args.large_scene, dataset)
        if args.large_scene:
            if args.precomp_id_map:
                pipeline.pre_compute_id_maps(dataset, pipe, background)
            else:
                pipeline.repair_masks(pipe, background, dataset, directory, args.alpha_w)
        else:
            pipeline.repair_masks(pipe, background, dataset, directory, args.alpha_w)


if __name__ == "__main__":
    main()
