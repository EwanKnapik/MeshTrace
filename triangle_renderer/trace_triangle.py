import torch
import math
import torch.nn.functional as F
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel

def trace(viewpoint_camera, pc: TriangleModel, id_masks: torch.Tensor, num_class: int, pipe, bg_color: torch.Tensor, alpha_w=False, scaling_modifier=1.0, override_color=None):
    """
    Trace triangles to compute per-triangle per-class weights.
    Returns tensor of shape (P, num_class+1) with counts per class.
    """
    # prepare geometry
    triangles_indices = pc.get_triangle_indices
    vertices = pc.get_vertices
    vertex_weights = pc.get_vertex_weight
    sigma = pc.get_sigma

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    # select SHs/colors as in existing triangle renderer
    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pipe, 'convert_SHs_python', False):
            # fallback: let rasterizer handle SH -> RGB
            shs = pc.get_features
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # call rasterizer
    rendered_image, radii, scaling, allmap, max_blending, was_rendered = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs=shs,
        colors_precomp=colors_precomp,
        scaling=torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=False, device="cuda")
    )

    # allmap contains auxiliary maps; index 6 is used in renderer as id map
    if allmap is None:
        P = triangles_indices.shape[0]
        return torch.zeros((P, num_class + 1), device='cuda')

    # ensure id map is on same resolution as id_masks
    id_map = allmap[6].long() if allmap.shape[0] > 6 else torch.full((H, W), -1, dtype=torch.long, device='cuda')

    # id_masks expected shape [H, W]
    if id_masks.dim() == 3:
        # if channel-first [C,H,W] or [1,H,W]
        if id_masks.shape[0] == 1:
            id_masks = id_masks.squeeze(0)
        else:
            id_masks = id_masks.argmax(0)

    P = triangles_indices.shape[0]
    weights = torch.zeros((P, num_class + 1), dtype=torch.float32, device='cuda')

    # compute counts per triangle per class
    # ignore invalid ids (negative)
    valid_mask = id_map >= 0
    id_map_valid = id_map.clone()
    id_map_valid[~valid_mask] = -1

    for c in range(num_class + 1):
        class_mask = (id_masks == c)
        # pixels belonging to this class and a triangle
        combined = class_mask & valid_mask
        if combined.any():
            ids = id_map_valid[combined]
            ids = ids[ids < P]  # discard out-of-range triangle IDs
            if ids.numel() > 0:
                binc = torch.bincount(ids, minlength=P).float()
                weights[:, c] = binc
    
    return weights
