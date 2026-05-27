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
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from scene.triangle_model import TriangleModel
from triangle_renderer import (
    TriangleRasterizationSettings,
    TriangleRasterizer,
    render as render_mesh,
)

_FEATURE_RASTERIZER_SUBMODULE = (
    Path(__file__).resolve().parents[1] / "submodules" / "diff-triangle-feature-rasterization"
)
if str(_FEATURE_RASTERIZER_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_FEATURE_RASTERIZER_SUBMODULE))

try:
    from diff_triangle_feature_rasterization import (
        TriangleRasterizationSettings as FeatureTriangleRasterizationSettings,
    )
    from diff_triangle_feature_rasterization import TriangleRasterizer as FeatureTriangleRasterizer

    _FEATURE_RASTERIZER_AVAILABLE = True
except Exception:
    FeatureTriangleRasterizationSettings = None
    FeatureTriangleRasterizer = None
    _FEATURE_RASTERIZER_AVAILABLE = False

_FEATURE_RASTERIZER_CHANNELS = 16


def _build_unshared_triangle_buffers(pc: TriangleModel):
    triangle_indices = pc.get_triangle_indices.long().contiguous()
    flat_vertex_idx = triangle_indices.reshape(-1)

    vertices = pc.get_vertices[flat_vertex_idx].contiguous()
    vertex_weights = pc.get_vertex_weight.reshape(-1)[flat_vertex_idx].contiguous()
    local_triangle_indices = torch.arange(
        flat_vertex_idx.numel(),
        device=flat_vertex_idx.device,
        dtype=torch.int32,
    ).view(-1, 3)
    return vertices, local_triangle_indices, vertex_weights


def _build_feature_raster_settings(
    viewpoint_camera,
    pipe,
    bg: torch.Tensor,
    scaling_modifier: float,
    include_feature: bool,
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    return dict(
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.contiguous(),
        projmatrix=viewpoint_camera.full_proj_transform.contiguous(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.contiguous(),
        prefiltered=False,
        debug=pipe.debug,
        include_feature=include_feature,
    )


def _render_feature_with_feature_rasterizer(
    viewpoint_camera,
    pc: TriangleModel,
    pipe,
    triangle_features: torch.Tensor,
    scaling_modifier: float,
):
    vertices, triangle_indices, vertex_weights = _build_unshared_triangle_buffers(pc)
    num_triangles = triangle_indices.shape[0]
    upsample = pc.scaling
    target_h = int(viewpoint_camera.image_height)
    target_w = int(viewpoint_camera.image_width)
    full_h = upsample * target_h
    full_w = upsample * target_w

    raster_settings = FeatureTriangleRasterizationSettings(
        image_height=full_h,
        image_width=full_w,
        **_build_feature_raster_settings(
            viewpoint_camera,
            pipe,
            bg=torch.zeros(3, dtype=vertices.dtype, device=vertices.device),
            scaling_modifier=scaling_modifier,
            include_feature=True,
        ),
    )
    rasterizer = FeatureTriangleRasterizer(raster_settings=raster_settings)

    scaling = torch.zeros(num_triangles, dtype=vertices.dtype, device=vertices.device)
    dummy_colors = torch.zeros((vertices.shape[0], 3), dtype=vertices.dtype, device=vertices.device)
    vertex_features = triangle_features.repeat_interleave(3, dim=0).contiguous()

    _, full_instance_image, _, _, _, _, _ = rasterizer(
        vertices=vertices,
        triangles_indices=triangle_indices,
        vertex_weights=vertex_weights,
        sigma=pc.get_sigma,
        scaling=scaling,
        colors_precomp=dummy_colors,
        instance_feature_precomp=vertex_features,
    )
    return full_instance_image.contiguous()


def _render_feature_chunked_with_color_rasterizer(
    viewpoint_camera,
    pc: TriangleModel,
    pipe,
    triangle_features: torch.Tensor,
    scaling_modifier: float,
):
    vertices, triangle_indices, vertex_weights = _build_unshared_triangle_buffers(pc)
    num_triangles, feature_dim = triangle_features.shape
    upsample = pc.scaling
    target_h = int(viewpoint_camera.image_height)
    target_w = int(viewpoint_camera.image_width)
    full_h = upsample * target_h
    full_w = upsample * target_w

    raster_settings = TriangleRasterizationSettings(
        image_height=full_h,
        image_width=full_w,
        **{
            key: value
            for key, value in _build_feature_raster_settings(
                viewpoint_camera,
                pipe,
                bg=torch.zeros(3, dtype=vertices.dtype, device=vertices.device),
                scaling_modifier=scaling_modifier,
                include_feature=False,
            ).items()
            if key != "include_feature"
        },
    )
    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    scaling = torch.zeros(num_triangles, dtype=vertices.dtype, device=vertices.device)
    rendered_chunks = []
    for start in range(0, feature_dim, 3):
        chunk = triangle_features[:, start : start + 3]
        if chunk.shape[1] < 3:
            chunk = F.pad(chunk, (0, 3 - chunk.shape[1]))

        colors_precomp = chunk.repeat_interleave(3, dim=0).contiguous()
        full_chunk, _, _, _, _, _ = rasterizer(
            vertices=vertices,
            triangles_indices=triangle_indices,
            vertex_weights=vertex_weights,
            sigma=pc.get_sigma,
            scaling=scaling,
            colors_precomp=colors_precomp,
        )
        rendered_chunks.append(full_chunk[: min(3, feature_dim - start)])

    return torch.cat(rendered_chunks, dim=0).contiguous()


def _render_instance_feature_image(
    viewpoint_camera,
    pc: TriangleModel,
    pipe,
    triangle_features: torch.Tensor,
    scaling_modifier: float,
):
    feature_dim = triangle_features.shape[1]
    target_h = int(viewpoint_camera.image_height)
    target_w = int(viewpoint_camera.image_width)
    if triangle_features.shape[0] == 0 or feature_dim == 0:
        empty = torch.zeros(
            (feature_dim, target_h, target_w),
            device=triangle_features.device,
            dtype=triangle_features.dtype,
        )
        return empty, empty

    if _FEATURE_RASTERIZER_AVAILABLE and feature_dim == _FEATURE_RASTERIZER_CHANNELS:
        full_instance_image = _render_feature_with_feature_rasterizer(
            viewpoint_camera,
            pc,
            pipe,
            triangle_features,
            scaling_modifier,
        )
    else:
        full_instance_image = _render_feature_chunked_with_color_rasterizer(
            viewpoint_camera,
            pc,
            pipe,
            triangle_features,
            scaling_modifier,
        )

    if full_instance_image.shape[-2:] != (target_h, target_w):
        instance_image = F.interpolate(
            full_instance_image.unsqueeze(0),
            size=(target_h, target_w),
            mode="area",
        ).squeeze(0)
    else:
        instance_image = full_instance_image

    return full_instance_image, instance_image.contiguous()


def _get_triangle_features(pc: TriangleModel, renderid: bool) -> torch.Tensor:
    if not renderid:
        triangle_features = pc.get_instance_feature
        if triangle_features is None:
            raise ValueError(
                "TriangleModel.instance_feature is empty. Enable feature training setup before rendering."
            )
        return triangle_features

    try:
        ids = pc.get_ids
    except AttributeError as exc:
        raise NotImplementedError(
            "renderid=True requires the model to expose `get_ids`, which TriangleModel does not provide."
        ) from exc

    if ids.max() > 15:
        raise ValueError("renderid=True currently supports ids in [0, 15].")

    return torch.nn.functional.one_hot(ids, num_classes=16).to(dtype=torch.float32, device=ids.device)


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

    The base mesh pass is kept identical to `triangle_renderer.render`, then a
    feature-only pass is executed on an unshared-per-triangle mesh so that
    triangle-level instance features render with the same visibility as the RGB
    pass.
    """

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

    triangle_features = _get_triangle_features(pc, renderid=renderid)
    full_instance_image, instance_image = _render_instance_feature_image(
        viewpoint_camera,
        pc,
        pipe,
        triangle_features,
        scaling_modifier,
    )
    render_pkg["instance_image_full"] = full_instance_image
    render_pkg["instance_image"] = instance_image
    return render_pkg
