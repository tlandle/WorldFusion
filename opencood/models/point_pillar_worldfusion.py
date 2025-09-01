# -*- coding: utf-8 -*-
#: Tyler Landle <tlandle3@gatech.edu>
"""
This module implements a world‑level fusion network for cooperative perception.

It combines the sensor level encoder from BM2CP (which processes LiDAR and
camera inputs for a single vehicle) with the Where2Comm attention based
communication module to fuse features from multiple collaborating vehicles or
roadside units.  The resulting network is designed to accept a batch of
intermediate fusion data (produced by the dataset class) and output per‑cell
detection logits for classification and regression.  It mirrors the behaviour
of other OpenCOOD models by operating directly on the aggregated batch
dictionary passed in from the dataloader.

Key differences from the original world fusion prototype:

* The ``forward`` method now accepts a single ``data_dict`` argument as
  produced by the intermediate‑fusion dataset.  It internally splits the
  aggregated features into per‑agent chunks, runs the sensor encoder on
  each, fuses them using Where2Comm and then returns the fused logits.
* Post‑processing (e.g. decoding anchor boxes) has been removed from the
  model.  Decoding should be performed externally (in evaluation code) using
  ``VoxelPostprocessor``.
* Single‑agent predictions (``psm_single_v``, ``psm_single_i``, etc.) are
  computed in the same way as the Where2Comm baseline: the first agent in
  each sample is treated as the ego vehicle and the second (if present) as
  an infrastructure unit.

This file should be placed in the ``opencood/models`` package and referenced
from a YAML file via ``model.core_method: point_pillar_worldfusion_updated``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from einops import rearrange

from opencood.models.bm2cp_modules.attentioncomm import ScaledDotProductAttention
from opencood.models.bm2cp_modules.sensor_blocks import ImgCamEncode
from opencood.models.bm2cp_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.where2comm_modules.where2comm_attn import Where2comm
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.utils import transformation_utils as T
import torch.nn.functional as F
import numpy as np

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization


class ImgModalFusion(nn.Module):
    """Voxel‑space fusion of image and LiDAR features (unchanged from BM2CP)."""

    def __init__(self, dim, threshold: float = 0.5) -> None:
        super().__init__()
        self.att = ScaledDotProductAttention(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        self.thres = threshold

    def forward(self, img_voxel: torch.Tensor, pc_voxel: torch.Tensor) -> torch.Tensor:
        # Flatten the 3D voxels along the spatial dimensions
        b, c, z, h, w = pc_voxel.shape
        pc_voxel_flat = pc_voxel.view(b, c, -1)
        img_voxel_flat = img_voxel.view(b, c, -1)
        voxel_mask = self.att(pc_voxel_flat, img_voxel_flat, img_voxel_flat)
        voxel_mask = self.act(self.proj(voxel_mask.permute(0, 2, 1)))
        voxel_mask = voxel_mask.permute(0, 2, 1).view(b, c, z, h, w)
        ones_mask = torch.ones_like(voxel_mask, device=voxel_mask.device)
        zeros_mask = torch.zeros_like(voxel_mask, device=voxel_mask.device)
        mask = torch.where(voxel_mask > self.thres, ones_mask, zeros_mask)
        # Always keep the ego mask as all ones to avoid dropping ego features
        mask[0] = ones_mask[0]
        return mask


class MultiModalFusion(nn.Module):
    """Fuse image and LiDAR voxels into a unified feature volume."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.img_fusion = ImgModalFusion(dim)
        self.multigate = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.multifuse = nn.Conv3d(dim * 2, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, img_voxel: torch.Tensor, pc_dict: Dict[str, torch.Tensor]):
        pc_voxel = pc_dict['spatial_features_3d']
        b, c, z, y, x = pc_voxel.shape
        # Create masks for non‑zero voxels
        ones_mask = torch.ones_like(pc_voxel, device=pc_voxel.device)
        zeros_mask = torch.zeros_like(pc_voxel, device=pc_voxel.device)
        pc_mask = torch.where(pc_voxel != 0, ones_mask, zeros_mask)
        pc_mask, _ = torch.max(pc_mask, dim=1, keepdim=True)
        img_mask = torch.where(img_voxel != 0, ones_mask, zeros_mask)
        img_mask, _ = torch.max(img_mask, dim=1, keepdim=True)
        # Fuse features
        fused_voxel = (
            pc_mask
            * img_mask
            * self.multifuse(torch.cat([
                self.act(self.multigate(pc_voxel)) * img_voxel,
                pc_voxel
            ], dim=1))
        )
        fused_voxel = fused_voxel + pc_voxel * pc_mask * (1 - img_mask) + img_voxel * self.img_fusion(img_voxel, pc_voxel) * (1 - pc_mask) * img_mask
        # Build threshold and mask outputs collapsed along Z
        thres_map = (
            pc_mask * img_mask * 0
            + pc_mask * (1 - img_mask) * 0.5
            + (1 - pc_mask) * img_mask * 0.5
            + (1 - pc_mask) * (1 - img_mask) * 0.5
        )
        thres_map, _ = torch.min(thres_map, dim=2, keepdim=False)
        mask = (
            pc_mask * img_mask
            + pc_mask * (1 - img_mask) * 2
            + (1 - pc_mask) * img_mask * 3
            + (1 - pc_mask) * (1 - img_mask) * 4
        )
        mask, _ = torch.min(mask, dim=2, keepdim=False)
        mask1, _ = torch.max(pc_mask, dim=2, keepdim=False)
        mask2, _ = torch.max(img_mask, dim=2, keepdim=False)
        # Collapse Z dimension for spatial_features
        pc_dict['spatial_features'] = fused_voxel.view(b, c * z, y, x)
        return pc_dict, thres_map, mask, torch.stack([mask1, mask2])


class _SensorEncoder(nn.Module):
    """
    Sensor‑level encoder for a single agent.

    This module implements the BM2CP encoder, consisting of a pillar VFE,
    scatter operation, camera encoding, voxel‑space fusion and a BEV backbone.
    It outputs a dictionary of feature maps suitable for further fusion.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.device = cfg['device'] if 'device' in cfg else 'cpu'
        self.supervise_single = cfg['supervise_single'] if 'supervise_single' in cfg else False
        pc_cfg = cfg['pc_params']
        img_cfg = cfg['img_params']
        fuse_cfg = cfg['modality_fusion']
        # LiDAR branch
        self.pillar_vfe = PillarVFE(
            pc_cfg['pillar_vfe'],
            num_point_features=4,
            voxel_size=pc_cfg['voxel_size'],
            point_cloud_range=pc_cfg['lidar_range'],
        )
        self.scatter = PointPillarScatter(pc_cfg['point_pillar_scatter'])
        # Precompute camera frustum using training resolution
        ogH, ogW = img_cfg['data_aug_conf']['final_dim']
        downsample = img_cfg['img_downsample']
        fH, fW = ogH // downsample, ogW // downsample

        grid_conf = img_cfg['grid_conf']
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        # Register as buffers so they move to the correct device
        self.register_buffer('dx', dx.clone().detach().requires_grad_(False).to(torch.device(self.device)))
        self.register_buffer('bx', bx.clone().detach().requires_grad_(False).to(torch.device(self.device)))
        self.register_buffer('nx', nx.clone().detach().requires_grad_(False).to(torch.device(self.device)))
        # Use efficient cumsum implementation
        self.use_quickcumsum = True
        ds = depth_discretization(*img_cfg['grid_conf']['ddiscr'], img_cfg['grid_conf']['mode'])
        ds = torch.tensor(ds, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        self.register_buffer('frustum', frustum, persistent=False)
        # Camera branch
        self.camenc = ImgCamEncode(
            D,
            img_cfg['bev_dim'],
            img_cfg['img_downsample'],
            img_cfg['grid_conf']['ddiscr'],
            img_cfg['grid_conf']['mode'],
            img_cfg['use_depth_gt'],
            img_cfg['depth_supervision'],
        )
        # Voxel‑space fusion
        self.vox_fuse = MultiModalFusion(img_cfg['bev_dim'])
        self.modal_multi_scale = fuse_cfg['bev_backbone']['multi_scale'] if 'multi_scale' in fuse_cfg['bev_backbone'] else False
        self.num_levels = fuse_cfg['bev_backbone']['layer_num'] if 'layer_num' in fuse_cfg['bev_backbone'] else 1
        assert img_cfg['bev_dim'] == pc_cfg['point_pillar_scatter']['num_features']
        # BEV backbone
        self.backbone = ResNetBEVBackbone(fuse_cfg['bev_backbone'], input_channels=img_cfg['bev_dim'])
        # Optional shrink conv
        self.shrink_flag = 'shrink_header' in fuse_cfg
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(fuse_cfg['shrink_header'])
        # Heads (shared with world fusion)
        C_out = fuse_cfg['shrink_header']['dim'][0] if self.shrink_flag else sum(fuse_cfg['bev_backbone']['num_upsample_filter'])
        self.cls_head = nn.Conv2d(C_out, cfg['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(C_out, 7 * cfg['anchor_number'], kernel_size=1)



    def _get_geometry(self, image_inputs_dict):
        """
        Compute the (x,y,z) location of every pixel in the frustum.
        Returns B × N × D × H/downsample × W/downsample × 3.
        """
        rots    = image_inputs_dict['rots']      # [B,N,3,3]  camera rotation matrices
        trans   = image_inputs_dict['trans']     # [B,N,3]    camera translation vectors
        intrins = image_inputs_dict['intrins']   # [B,N,3,3]  camera intrinsics
        post_rots  = image_inputs_dict['post_rots']  # [B,N,3,3] augmentation rotations
        post_trans = image_inputs_dict['post_trans'] # [B,N,3]    augmentation translations

        B, N, _ = trans.shape

        # 1. Subtract post‑translation and undo post‑rotation.
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if post_rots.device != torch.device('cpu'):
            inv_post_rots = torch.inverse(post_rots.to('cpu')).to(post_rots.device)
        else:
            inv_post_rots = torch.inverse(post_rots)
        # Multiply the 3×3 matrix against the [x,y,z,1] column (unsqueezed)
        points = inv_post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # 2. Project the xy components by depth to get metric image‑plane coords.
        # At this point 'points' has shape [B,N,D,H,W,3,1].
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]), 5)

        # 3. Undo intrinsics and apply extrinsics (camera→ego).
        if intrins.device != torch.device('cpu'):
            inv_intrins = torch.inverse(intrins.to('cpu')).to(intrins.device)
        else:
            inv_intrins = torch.inverse(intrins)
        combine = rots.matmul(inv_intrins)  # [B,N,3,3]
        # Multiply the 3×3 matrix by the 3×1 column (still unsqueezed).
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points  # shape [B,N,D,H,W,3]


    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        # collapsed_final = torch.cat(final.unbind(dim=2), 1)

        # return collapsed_final#, x  # final: 4 x 64 x 240 x 240  # B, C, H, W
        return final

    def forward(self, batch: Dict) -> Dict:
        """
        Encode a single agent's LiDAR and camera data into BEV feature maps.

        Args:
            batch: A dictionary containing:
                processed_lidar: {voxel_features, voxel_coords, voxel_num_points}
                image_inputs: {imgs, rots, trans, intrins, post_rots, post_trans, depth_map}
                record_len: Tensor([1]) (unused here but kept for API consistency)

        Returns:
            A dictionary with keys:
              spatial_features_2d, spatial_features, psm, rm, thres_map, mask, each_mask
        """
        pc = batch['processed_lidar']
        rec_len = batch['record_len']
        bd = {
            'voxel_features': pc['voxel_features'],
            'voxel_coords': pc['voxel_coords'],
            'voxel_num_points': pc['voxel_num_points'],
            'record_len': rec_len,
        }
        bd = self.scatter(self.pillar_vfe(bd))
        # Camera branch

        # image branch
        imgs = batch['image_inputs']['imgs']  # [B,N,3,H,W]
        B,N,C,imH,imW = imgs.shape
        x = imgs.view(B*N, C, imH, imW)
        _, x = self.camenc(x, batch['image_inputs']['depth_map'], batch['record_len'])
        x = rearrange(x, '(b n) c d h w -> b n c d h w', b=B, n=N)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B,N,D,fH,fW,C]
        geom = self._get_geometry(batch['image_inputs'])  # [B,N,D,fH,fW,3]
        # pool into BEV voxels; returns [B,C,Z,Y,X]
        img_voxel = self.voxel_pooling(geom, x)
        # fuse with LiDAR voxels: pc_dict['spatial_features_3d'] also [B,C,Z,Y,X]
        bd, thres_map, mask, each_mask = self.vox_fuse(img_voxel, bd)
        bd = self.backbone(bd)
        bev2d = bd['spatial_features_2d']
        if self.shrink_flag:
            bev2d = self.shrink_conv(bev2d)
        # Classification and regression heads for single agent
        psm = self.cls_head(bev2d)
        rm = self.reg_head(bev2d)
        bd.update({
            'spatial_features_2d': bev2d,
            'psm': psm,
            'rm': rm,
            'thres_map': thres_map,
            'mask': mask,
            'each_mask': each_mask,
        })
        return bd


class PointPillarWorldFusion(nn.Module):
    """
    World‑canvas fusion network for cooperative perception.

    This network accepts a batch of intermediate‑fusion data containing
    concatenated features from multiple collaborative vehicles/RSUs.  It
    processes each agent's LiDAR and camera inputs with a shared sensor
    encoder, fuses the resulting BEV features using the Where2Comm
    communication module and outputs fused classification and regression
    logits.  It also computes single‑agent predictions for ego and
    infrastructure vehicles for use in multitask loss functions.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        # Sensor encoder for individual agents
        self.sensor = _SensorEncoder(cfg)
        # Where2Comm fusion module
        self.fusion = Where2comm(cfg['fusion_args'])
        # Alias shrink, cls_head and reg_head from sensor for convenience
        self.shrink = self.sensor.shrink_conv if self.sensor.shrink_flag else None
        self.cls_head = self.sensor.cls_head
        self.reg_head = self.sensor.reg_head
        self.multi_scale = cfg['fusion_args']['multi_scale']

    @torch.inference_mode()
    def get_feature(self, batch: Dict) -> Dict:
        """Vehicle side encoding wrapper, identical to sensor forward."""
        return self.sensor(batch)

    def regroup(self, x: torch.Tensor, record_len: torch.Tensor) -> List[torch.Tensor]:
        """Regroup a tensor according to record_len across batch dimension."""
        cum_sum = torch.cumsum(record_len, dim=0)
        return list(torch.tensor_split(x, cum_sum[:-1].cpu()))

    def forward(self, data_dict: Dict) -> Dict:
        """
        Forward pass for multi‑agent cooperative detection.

        Args:
            data_dict: A dictionary produced by the intermediate fusion dataset
                with keys:
                    processed_lidar: dict of concatenated voxel features across all agents
                    image_inputs: dict of concatenated camera inputs across all agents
                    record_len: (B,) tensor indicating the number of agents per sample
                    pairwise_t_matrix: (B, max_cav, max_cav, 4, 4) transformation matrices

        Returns:
            output_dict: A dictionary containing fused classification/regression
                logits and auxiliary outputs.  Keys include:
                    'psm': fused classification logits
                    'rm': fused regression logits
                    'psm_single_v', 'psm_single_i', 'rm_single_v', 'rm_single_i'
                    'mask', 'each_mask', 'comm_rate' (if provided by fusion)
        """
        # Unpack data
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_pts = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise = data_dict['pairwise_t_matrix']
        image_inputs = data_dict['image_inputs']

        B = record_len.shape[0]
        device = voxel_features.device
        # Compute cumulative sum to index into concatenated agents
        cum_sum = torch.cumsum(record_len, dim=0).cpu()
        start_indices: List[int] = [0] + cum_sum[:-1].tolist()
        feat_list: List[Dict] = []
        feat_counts: List[int] = []  # keep track of how many features per sample
        # Process each sample in the batch separately
        for b in range(B):
            count = 0
            num_cavs = int(record_len[b].item())
            start = start_indices[b]
            end = start + num_cavs
            # For each agent within this sample
            for cav_offset in range(num_cavs):
                global_idx = start + cav_offset
                # Split LiDAR voxels by batch index
                mask = (voxel_coords[:, 0] == global_idx)
                v_feat = voxel_features[mask]
                v_coord = voxel_coords[mask]
                v_num = voxel_num_pts[mask]
                # Reset batch index to zero for single‑agent processing
                if v_coord.numel() == 0:
                    # Handle empty voxels gracefully
                    continue
                v_coord = torch.cat([
                    torch.zeros((v_coord.shape[0], 1), dtype=v_coord.dtype, device=v_coord.device),
                    v_coord[:, 1:]
                ], dim=1)
                sub_lidar = {
                    'voxel_features': v_feat,
                    'voxel_coords': v_coord,
                    'voxel_num_points': v_num,
                }
                # Split image inputs along first dimension
                sub_imgs: Dict = {}
                for k, v in image_inputs.items():
                    # Each v has shape [sum_cavs, ...]; select the slice for this agent
                    if isinstance(v, torch.Tensor):
                        sub_imgs[k] = v[global_idx:global_idx + 1]
                    else:
                        # For lists, slice directly (unlikely here but kept for completeness)
                        sub_imgs[k] = v[global_idx:global_idx + 1]
                # Build sub‑batch and run sensor encoder
                sub_record_len = torch.tensor([1], dtype=record_len.dtype, device=device)
                sub_batch = {
                    'processed_lidar': sub_lidar,
                    'image_inputs': sub_imgs,
                    'record_len': sub_record_len
                }
                feat = self.sensor(sub_batch)
                feat_list.append(feat)
                count += 1
            feat_counts.append(count)

        if not feat_list:
            raise RuntimeError("No valid agent features were extracted from the input batch.")

        # Concatenate per‑agent spatial features and heads
        spatial_2d = torch.cat([f['spatial_features_2d'] for f in feat_list], dim=0)
        psm_single = torch.cat([f['psm'] for f in feat_list], dim=0)
        rm_single = torch.cat([f['rm'] for f in feat_list], dim=0)
        thres_map = torch.cat([f['thres_map'] for f in feat_list], dim=0)
        spatial_features = torch.cat([f['spatial_features'] for f in feat_list], dim=0)
        psm_single       = torch.cat([f['psm']               for f in feat_list], dim=0)  # logits at first-resnet resolution

        # Now replace record_len with the number of features you actually have
        actual_record_len = torch.tensor(feat_counts, dtype=record_len.dtype, device=record_len.device)


        
        world_anchors = data_dict['world_anchor']  # list of [x,y,z] per batch sample
        pairwise_world = pairwise.clone()

        lidar_poses = data_dict['lidar_pose']

        for b in range(B):
            n = int(record_len[b])
            anchor_pose = [world_anchors[b][0], world_anchors[b][1], world_anchors[b][2], 0, 0, 0]
            
            # Get all transformations to world anchor first
            T_to_anchor = []
            for i in range(n):
                pose_i = lidar_poses[b][i] if isinstance(lidar_poses[b], list) else lidar_poses[i]
                T_i_to_anchor = T.x1_to_x2(pose_i, anchor_pose)
                T_to_anchor.append(T_i_to_anchor)
            
            # Now compute full pairwise matrix
            for i in range(n):
                for j in range(n):
                    if i == j:
                        pairwise_world[b, i, j] = torch.eye(4).to(pairwise.device)
                    else:
                        # Transform from i to j through anchor: j_inv * i
                        T_i_to_j = np.linalg.inv(T_to_anchor[j]) @ T_to_anchor[i]
                        pairwise_world[b, i, j] = torch.from_numpy(T_i_to_j).float().to(pairwise.device)

        # For multi-scale Where2Comm, downsample the 192×704 thres_map to 96×352
        H1 = self.sensor.nx[1] // 2   # = 96 if nx[1]=192
        W1 = self.sensor.nx[0] // 2   # = 352 if nx[0]=704
        if psm_single.shape[-2:] != (H1, W1):
            psm_single = F.interpolate(psm_single, size=(H1, W1), mode='nearest')
        #bd['thres_map'] = thres_map

        # world_pose = [x=0, y=0, z=0, roll=0, yaw=0, pitch=0]
        # Prepare spatial feature for multi‑scale fusion if needed
        if self.multi_scale:
            spatial_features = torch.cat([f['spatial_features'] for f in feat_list], dim=0)
            fused_feature, comm_rate, result_dict = self.fusion(
                spatial_features,
                psm_single,
                actual_record_len,
                pairwise_world,
                backbone=self.sensor.backbone,
                heads=[self.sensor.shrink_conv if self.sensor.shrink_flag else None, self.cls_head, self.reg_head]
            )
            # Downsample fused features if shrink flag
            if self.sensor.shrink_flag:
                fused_feature = self.sensor.shrink_conv(fused_feature)
        else:
            fused_feature, comm_rate, result_dict = self.fusion(
                spatial_features,
                psm_single,
                thres_map,
                actual_record_len,
                pairwise_world
            )

        # Classification and regression on fused feature
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm, 'rm': rm}
        # Merge in Where2Comm auxiliary outputs
        if result_dict:
            output_dict.update(result_dict)
        # Pass along mask and each_mask from the first agent (all agents share the same shape)
        output_dict['mask'] = feat_list[0]['mask']
        output_dict['each_mask'] = feat_list[0]['each_mask']
        output_dict['comm_rate'] = comm_rate
        # Compute single‑agent outputs for ego/infrastructure (following Where2Comm baseline)
        # psm_single and rm_single contain predictions for all agents concatenated.  We
        # regroup them by batch sample and select the first two agents (0:1 and 1:2).
        split_psm_single = self.regroup(psm_single, actual_record_len)
        split_rm_single = self.regroup(rm_single, actual_record_len)
        psm_single_v_list = []
        psm_single_i_list = []
        rm_single_v_list = []
        rm_single_i_list = []
        for b in range(len(split_psm_single)):
            cav_list = split_psm_single[b]
            reg_list = split_rm_single[b]
            # Vehicle (ego) prediction: take the first CAV
            psm_single_v_list.append(cav_list[0:1])
            rm_single_v_list.append(reg_list[0:1])
            # Infrastructure prediction: if a second CAV exists, take it; else repeat the first
            if cav_list.size(0) > 1:
                psm_single_i_list.append(cav_list[1:2])
                rm_single_i_list.append(reg_list[1:2])
            else:
                psm_single_i_list.append(cav_list[0:1])
                rm_single_i_list.append(reg_list[0:1])
        psm_single_v = torch.cat(psm_single_v_list, dim=0)
        psm_single_i = torch.cat(psm_single_i_list, dim=0)
        rm_single_v = torch.cat(rm_single_v_list, dim=0)
        rm_single_i = torch.cat(rm_single_i_list, dim=0)
        output_dict.update({
            'psm_single_v': psm_single_v,
            'psm_single_i': psm_single_i,
            'rm_single_v': rm_single_v,
            'rm_single_i': rm_single_i,
        })
        return output_dict
