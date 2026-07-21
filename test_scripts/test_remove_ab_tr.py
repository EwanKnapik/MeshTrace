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
from triangle_renderer.trace_triangle import trace

def get_weights_from_trace(triangles, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    with torch.no_grad():
        num_triangles = triangles._triangle_indices.shape[0]
        weights = torch.zeros((num_triangles, len(viewpoints)), dtype=torch.int).cuda()
        for idx, view in enumerate(viewpoints):
            sam_mask = view.sam_mask.copy()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cpu")
            id_masks = id_masks.cuda()
            id_masks[id_masks > 1] = 0
            w = trace(view, triangles, id_masks, pipe, background, alpha_w=alpha_w)
            unseen_mask = (w.sum(-1) == 0)
            w = torch.sum(w, dim=-1)
            w[unseen_mask] = -1
            weights[:, idx] = w
    return weights


def prune_mask_from_was_rendered(triangles, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    with torch.no_grad():
        num_triangles = triangles.get_triangle_indices.shape[0]
        mask=torch.zeros(num_triangles).cuda()
        for idx, view in enumerate(viewpoints):
            render_pkg = render(view, triangles, pipe, background)
            was_rendered=render_pkg["triangle_was_rendered"]
            mask+= was_rendered
        return ~(mask!=0)


def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def print_img_from_id_map(viewpoint_camera, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    image=render_pkg["render"].permute(1,2,0)
    plt.figure()
    plt.imshow(image.detach().cpu().numpy())
    plt.axis("off")
    #plt.savefig(f"instance_map/instance_map_{iteration}.png", bbox_inches="tight", pad_inches=0)
    sixel_fig()
    plt.close()
    return

def do_stuff_trace(viewpoints, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    weights = get_weights_from_trace(pc, viewpoints, pipe, bg_color, -1)
    p_mask = ((weights != -1).sum(-1) == 0)
    print(p_mask.sum())

    p_mask2=prune_mask_from_was_rendered(pc,viewpoints,pipe,bg_color)
    print(p_mask2.sum())
    print((~p_mask2).sum())
    return

def print_render_img_from_feature_rasterizer(viewpoint_camera, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    render_pkg = render_feature.render(viewpoint_camera, pc, pipe, bg_color, include_feature=True)
    image = render_pkg["render"]
    plt.figure()
    plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
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
    #do_stuff_trace(camera_stack,triangles,pipe,background)
    viewpoint_cam = camera_stack.pop(args.image)
    print_img_from_id_map(viewpoint_cam,triangles,pipe,background)
    #print_render_img_from_feature_rasterizer(viewpoint_cam,triangles,pipe,background)

if __name__ == "__main__":
    main()
