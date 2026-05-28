#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_triangle_feature_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, include_feature: bool = False, scaling_modifier: float = 1.0, override_color=None, renderid: bool = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_vertices, dtype=pc.get_vertices.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = TriangleRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=include_feature
        # pipe.debug
    )
    
    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    vertices = pc.get_vertices
    triangle_indices=pc.get_triangle_indices
    vertex_weight = pc.get_vertex_weight.reshape(-1).contiguous()
    sigma = pc.get_sigma

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            features = pc.get_features.contiguous()
            shs_view = features.transpose(1, 2).contiguous().view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (vertices - camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features.contiguous()

    else:
        colors_precomp = override_color.contiguous()
        
    if include_feature:
        if not renderid:
            instance_feature_precomp = pc.get_instance_feature

    else:
        instance_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)#lang

    scaling = torch.zeros(
        triangle_indices.shape[0],
        dtype=vertices.dtype,
        device=vertices.device,
    )
    rendered_image, instance_image, radii, scaling, depth, max_blending, was_rendered = rasterizer(
        vertices=vertices,
        triangles_indices=triangle_indices,
        vertex_weights=vertex_weight,
        sigma=sigma,
        scaling = scaling,
        shs = shs,
        colors_precomp = colors_precomp,
        instance_feature_precomp =instance_feature_precomp ,
    )
    

    rets =  {"render": rendered_image,
             "instance_image":instance_image, 
            "radii": radii,
            "was_rendered":was_rendered
    }

    return rets
