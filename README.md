<h1 align="center">MeshTrace: 3D segmentation in Mesh Splatting via Triangle Instance Tracing and contrastive lifting.</h1>
<p align="center">
  Ewan Knapik
</p>


## Cloning the Repository + Installation

The code has been used and tested with Python 3.11 and CUDA 12.6.

You should clone the repository with the different submodules by running the following command:

```bash
git clone https://github.com/EwanKnapik/mesh_splatting_with_semantic.git --recursive
cd mesh_splatting_with_semantic
```

Then, we suggest to use a virtual environment to install the dependencies.

```bash
micromamba create -n mesh_splatting python=3.11
micromamba activate mesh_splatting
micromamba install nvidia/label/cuda-12.6.0::cuda

pip install torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
```

Finally, you can compile the custom CUDA kernels by running the following command:

```bash
bash compile.sh
cd submodules/simple-knn
pip install . --no-build-isolation
cd submodules/effrdel
pip install -e .
```

## Training
To train our model, you can use the following command:
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --eval
```

If you want to train the model on indoor scenes, you should add the following command:  
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --indoor --eval
```

### Depth supervision
To have better reconstructed scenes we use depth maps as priors during optimization with each input images.
For real world datasets depth maps should be generated for each input images, to generate them please do the following:

1. Clone [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#usage):
    ```
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    ```
2. Download weights from [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it under `Depth-Anything-V2/checkpoints/`
3. Generate depth maps:
   ```
   python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <path to input images> --outdir <output path>
   ```
   Create a folder named 'depth' to store the depth maps. This folder should be placed alongside the folders containing the RGB images, for example: MipNeRF360/Garden/depth.
5. Generate a `depth_params.json` file using:
    ```
    python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>
    ```

 The depth regularization we integrated is that used in our [Hierarchical 3DGS](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) pape.

## Generating SAM masks
Downloading the Segment Anything Model
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Generating the masks
```
python get_sam_masks.py --sam_checkpoint {SAM_CKPT_PATH} --file_path {IMAGE_FOLDER}
```
Or with a slurm script, after changing necessary fields in the script
```
sbatch slurm_scripts/create_sam_masks.job
```
## Triangle Instance Tracing
```
python merge_patches.py -s ${source}  -m ${output}
```

## Adaptive Density Control
```
python remove_ab_triangles.py -s ${source} -m ${output} --sam_folder split_ms --iterations 9000 --prune --eval
```
To be noted that with the current implementation, the `--iteration` flag doesn't do anything.
Some line of code can be uncommented so that the real Adaptive Density Control is executed, but it is currently not working properly and creates lots of artifacts.

## Contrastive Lifting
```
python train_contrastive.py -s ${source} -m ${output} --iterations 20000 --start_checkpoint ${output}/point_cloud/iteration_chkpt_9000/point_cloud_state_dict.pt --sam_folder split_ms --include_feature --save_name sp_
```

## Whole Pipeline
The whole pipeline, including training, can be run via a slurm script
```
sbatch slurm_scripts/pipeline.job
```

## Clustering
The clustering of the feature vector for a scene can be run with
```
python3 ply_scripts/create_full_ply_clustered.py --scene_path {ply_path}
```

## reconstruction Evaluation
```
sbatch slurm_scripts/metrics.job
```

## Segmentation Evaluation
```
sbatch slurm_scripts/eval_3d.job
```


## Rendering
To render a scene, you can use the following command:
```bash
python render.py -m <path_to_model>
```

To create a video, you can use the following command:
```bash
python create_video.py -m <path_to_model> -s <path_to_scenes>
```

## Create custom PLY files of optimized scenes

To save your optimized scene after training, just run:

```
python create_ply.py <output_model_path>
```

