import torch
import matplotlib.pyplot as plt
import matplotlib

from sklearn.decomposition import PCA
from argparse import ArgumentParser
from sixel import sixel
from io import BytesIO

from triangle_renderer import TriangleModel
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import render
import triangle_renderer.render_feature as render_feature


def _project_instance_image_for_plot(instance_image):
    with torch.no_grad():
        channels, height, width = instance_image.shape
        flat_pixels = (
            instance_image.detach()
            .permute(1, 2, 0)
            .reshape(-1, channels)
            .float()
            .cpu()
            .numpy()
        )
        projected = PCA(n_components=3).fit_transform(flat_pixels).reshape(height, width, 3)
        projected -= projected.min(axis=(0, 1), keepdims=True)
        scale = projected.max(axis=(0, 1), keepdims=True)
        scale[scale == 0] = 1.0
        projected = projected / scale
        return torch.from_numpy(projected).permute(2, 0, 1).float()

def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def print_img_from_id_map(viewpoint_camera, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    vertex_weight=pc.get_vertex_weight
    instance_features=pc.get_instance_feature
    if instance_features is None:
        raise ValueError("The loaded model does not contain instance features.")
    #triangles_indices=pc.get_triangle_indices
    #instance_feature_weighted=vertex_weight*instance_features
    #triangle_instance=instance_feature_weighted[triangles_indices].sum(dim=1)


    render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    rend_ids = render_pkg["rend_ids"][0]
    num_triangles = triangle_instance.shape[0]
    valid_rend_ids = torch.isfinite(rend_ids) & (rend_ids >= 0) & (rend_ids < num_triangles)

    invalid_count = int((~valid_rend_ids).sum().item())

    height, width = rend_ids.shape
    feature_dim = triangle_instance.shape[1]
    instance_image = torch.zeros(
        (height, width, feature_dim),
        dtype=triangle_instance.dtype,
        device=triangle_instance.device,
    )
    if bool(valid_rend_ids.any().item()):
        safe_ids = rend_ids[valid_rend_ids].long()
        #instance_image[valid_rend_ids] = triangle_instance[safe_ids]
        instance_image[valid_rend_ids] = instance_features[safe_ids]

    instance_image = instance_image.permute(2, 0, 1).contiguous()

    instance_image_rgb = _project_instance_image_for_plot(instance_image)
    plt.figure()
    plt.imshow(instance_image_rgb.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    #plt.savefig(f"instance_map/instance_map_{iteration}.png", bbox_inches="tight", pad_inches=0)
    sixel_fig()
    plt.close()

    return

def print_img_from_feature_rasterizer(viewpoint_camera, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    render_pkg = render_feature.render(viewpoint_camera, pc, pipe, bg_color, include_feature=True)
    instance_image = render_pkg["instance_image"]
    instance_image_rgb = _project_instance_image_for_plot(instance_image)
    plt.figure()
    plt.imshow(instance_image_rgb.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    sixel_fig()
    plt.close()

def main():
    matplotlib.rcParams["backend"] = "Agg"
    parser = ArgumentParser(description="Extract objects — triangle splatting mask repair")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default="sp_20000", type=str)
    parser.add_argument("--alpha_w", action="store_true")
    parser.add_argument("--image", default=1, type=int)
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=args.iteration,
                  shuffle=False)
    camera_stack=scene.getTrainCameras().copy()
    viewpoint_cam = camera_stack.pop(10)
    print_img_from_id_map(viewpoint_cam,triangles,pipe,background)
    #print_img_from_feature_rasterizer(viewpoint_cam,triangles,pipe,background)

if __name__ == "__main__":
    main()
