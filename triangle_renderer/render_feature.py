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

import math

import torch
import torch.nn.functional as F

from scene.triangle_model import TriangleModel
from triangle_renderer import (
    TriangleRasterizationSettings,
    TriangleRasterizer,
    render as render_mesh,
)


def _build_unshared_triangle_buffers(pc: TriangleModel):
    triangle_indices = pc.get_triangle_indices.long().contiguous()
    flat_vertex_idx = triangle_indices.reshape(-1)

    vertices = pc.get_vertices.detach()[flat_vertex_idx].contiguous()
    vertex_weights = pc.get_vertex_weight.detach().reshape(-1)[flat_vertex_idx].contiguous()
    local_triangle_indices = torch.arange(
        flat_vertex_idx.numel(),
        device=flat_vertex_idx.device,
        dtype=torch.int32,
    ).view(-1, 3)
    return vertices, local_triangle_indices, vertex_weights


def _render_instance_feature_image(
    viewpoint_camera,
    pc: TriangleModel,
    pipe,
    feature_bg: torch.Tensor,
    scaling_modifier: float,
):
    triangle_features = pc.get_instance_feature
    if triangle_features is None:
        raise ValueError("TriangleModel.instance_feature is empty. Enable feature training setup before rendering.")

    num_triangles, feature_dim = triangle_features.shape
    if feature_dim != 3:
        raise ValueError(
            f"Mesh feature rendering is specialized for 3D instance features, got shape {triangle_features.shape}."
        )

    target_h = int(viewpoint_camera.image_height)
    target_w = int(viewpoint_camera.image_width)
    if num_triangles == 0:
        empty = torch.zeros((feature_dim, target_h, target_w), device=feature_bg.device, dtype=feature_bg.dtype)
        return empty, empty

    vertices, triangle_indices, vertex_weights = _build_unshared_triangle_buffers(pc)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    upsample = pc.scaling
    full_h = upsample * target_h
    full_w = upsample * target_w


    raster_settings = TriangleRasterizationSettings(
        image_height=full_h,
        image_width=full_w,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=feature_bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.contiguous(),
        projmatrix=viewpoint_camera.full_proj_transform.contiguous(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.contiguous(),
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    sigma = pc.get_sigma
    scaling = torch.zeros(
        num_triangles,
        dtype=vertices.dtype,
        device=vertices.device,
    )
    colors_precomp = triangle_features.repeat_interleave(3, dim=0).contiguous()

    full_instance_image, _, _, _, _, _ = rasterizer(
        vertices=vertices,
        triangles_indices=triangle_indices,
        vertex_weights=vertex_weights,
        sigma=sigma,
        scaling=scaling,
        colors_precomp=colors_precomp,
    )
    full_instance_image = full_instance_image.contiguous()
    if full_instance_image.shape[-2:] != (target_h, target_w):
        instance_image = F.interpolate(
            full_instance_image.unsqueeze(0),
            size=(target_h, target_w),
            mode="area",
        ).squeeze(0)
    else:
        instance_image = full_instance_image

    return full_instance_image, instance_image


def render(
    viewpoint_camera,
    pc: TriangleModel,
    pipe,
    bg_color: torch.Tensor,
    include_feature: bool = False,
    scaling_modifier: float = 1.0,
    override_color=None,
    renderid: bool = False,
):
    """
    Mesh-splatting equivalent of the Gaussian feature renderer.

    This specialization assumes `instance_feature` has shape [num_triangles, 3]
    and renders it in a single pass through `colors_precomp` on a temporary
    unshared-per-triangle mesh.
    """

    if renderid:
        raise NotImplementedError(
            "Mesh feature rendering does not support renderid=True. Use `rend_ids` from the base triangle renderer instead."
        )

    render_pkg = render_mesh(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
    )

    if not include_feature:
        render_pkg["instance_image"] = torch.zeros(
            (1,),
            dtype=render_pkg["render"].dtype,
            device=render_pkg["render"].device,
        )
        return render_pkg

    feature_bg = torch.zeros(3, dtype=bg_color.dtype, device=bg_color.device)
    full_instance_image, instance_image = _render_instance_feature_image(
        viewpoint_camera,
        pc,
        pipe,
        feature_bg,
        scaling_modifier,
    )
    render_pkg["instance_image_full"] = full_instance_image
    render_pkg["instance_image"] = instance_image
    return render_pkg
