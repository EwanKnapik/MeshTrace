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



def sixel_fig():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    writer = sixel.SixelWriter()
    writer.draw(buffer)

def print_img(viewpoint_camera, pc :TriangleModel, pipe, bg_color) -> torch.tensor:
    vertex_weight=pc.get_vertex_weight

    render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    rendered_view = render_pkg["render"]


    plt.figure()
    plt.imshow(rendered_view.permute(1, 2, 0).detach().cpu().numpy())
    plt.axis("off")
    sixel_fig()
    plt.close()


def compute_barycenter(vertices,trngl_idx):
    barycenters = vertices[trngl_idx[:, 0]]
    barycenters = barycenters + vertices[trngl_idx[:, 1]]
    barycenters = barycenters + vertices[trngl_idx[:, 2]]
    barycenters = barycenters * (1.0 / 3.0)
    return barycenters

def distance_cull_scene(distance:int, triangles:TriangleModel, viewpoint_cam,pipe,background):
    points=triangles.get_vertices
    trngl_idx=triangles.get_triangle_indices

    #sq_coords=torch.square(points)
    #distance_from_center=torch.sqrt(sq_coords[:,0]+sq_coords[:,1]+sq_coords[:,2])
    #mask=distance_from_center<distance

    barycenters=compute_barycenter(points,trngl_idx)
    sq_coords=torch.square(barycenters)
    distance_from_center=torch.sqrt(sq_coords[:,0]+sq_coords[:,1]+sq_coords[:,2])
    mask=distance_from_center<distance

    print(mask.shape)
    print(mask.sum())

    triangles.prune_triangles(mask)


    print_img(viewpoint_cam,triangles,pipe,background)







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
    viewpoint_cam = camera_stack.pop(77)
    distance_cull_scene(3,triangles, viewpoint_cam,pipe,background) 
    save_name="culled"
    scene.save(f"{culled}_{args.iteration}")          

if __name__ == "__main__":
    main()
