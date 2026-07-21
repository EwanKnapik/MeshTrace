#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap, gt_alpha_mask,
                 image_name, uid,instance_image=None,sam_mask=None,depth=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", normal_map=None, resolution=1.0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.normal_map = normal_map

        # to handle image resize in merge patches and render easily
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_instance_image = instance_image.astype(np.int32) if instance_image is not None else None
        self.sam_mask = sam_mask
        self.gt_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            #self.original_image *= gt_alpha_mask.to(self.data_device) # IF YOU WANT A BACKGROUND FOR DTU COMMENT THIS
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None


        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones((self.image_height, self.image_width))
            self.invdepthmap = invdepthmap
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                   # print(depth_params["scale"])
                   # print(depth_params["offset"])
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale),
            dtype=torch.float32,
            device=self.data_device,
        ).transpose(0, 1).contiguous()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(self.data_device).contiguous()
        self.full_proj_transform = torch.matmul(
            self.world_view_transform, self.projection_matrix
        ).contiguous()
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3].contiguous().clone()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform.contiguous()
        self.full_proj_transform = full_proj_transform.contiguous()
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3].contiguous().clone()
