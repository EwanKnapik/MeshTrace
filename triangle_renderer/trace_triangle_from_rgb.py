import torch
import math
import torch.nn.functional as F
from triangle_renderer import render
from scene.triangle_model import TriangleModel


def trace(viewpoint_camera, pc: TriangleModel, id_masks: torch.Tensor, pipe ,bg_color:torch.Tensor,
          alpha_w=False, scaling_modifier=1.0, override_color=None)->torch.tensor:
    try:
        render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    except RuntimeError as exc:
        if "doesn't have storage" not in str(exc):
            raise
        num_triangles = pc.get_triangle_indices.shape[0]
        max_mask_id = int(id_masks.max().item()) if id_masks.numel() > 0 else 0
        num_mask_ids = max(max_mask_id + 1, 1)
        view_name = getattr(viewpoint_camera, "image_name", "<unknown>")
        print(f"Skipping view '{view_name}' because the triangle rasterizer returned an empty temporary buffer.")
        return torch.zeros((num_triangles, num_mask_ids), device=id_masks.device, dtype=torch.float32)

    rend_ids = render_pkg["rend_ids"][0].long()
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




def trace_masks(viewpoint_camera, pc: TriangleModel, mask1: torch.Tensor, mask2: torch.Tensor, pipe ,bg_color:torch.Tensor,
          alpha_w=False, scaling_modifier=1.0, override_color=None) -> (torch.Tensor, torch.Tensor):
    render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    rend_ids = render_pkg["rend_ids"][0].long()
    if id_masks.shape != rend_ids.shape:
        print(f"{'!' * 10} sam_mask not same shape as rend_ids {'!' * 10}")

    num_triangles = pc.get_triangle_indices.shape[0]
    in_bounds = (rend_ids >= 0) & (rend_ids < num_triangles)

    flat_rend_ids = rend_ids[in_bounds].reshape(-1).long()
    flat_mask1 = mask1[in_bounds].reshape(-1).long()
    flat_mask2 = mask2[in_bounds].reshape(-1).long()

    triangles_id_1=flat_rend_ids[flat_mask1].unique()
    print(f"number of triangles in mask 1 {triangle_id_1.shape}")
    triangles_id_2=flat_rend_ids[flat_mask2].unique()
    print(f"number of triangles in mask 2 {triangle_id_2.shape}")
    return triangles_id_1, triangles_id_2
