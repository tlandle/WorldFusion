# opencood/data_utils/post_processor/world_voxel_postprocessor.py
#: Tyler Landle <tlandle3@gtaech.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import sys
import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils, transformation_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period


class WorldVoxelPostprocessor(VoxelPostprocessor):
    def __init__(self, anchor_params, dataset, train):
        super(WorldVoxelPostprocessor, self).__init__(anchor_params, dataset, train)
        
        # Override with world canvas parameters if present
        if 'canvas_size_m' in anchor_params['anchor_args']:
            self.canvas_size_m = anchor_params['anchor_args']['canvas_size_m']
            self.canvas_res = anchor_params['anchor_args']['canvas_res']
            # Define world range based on canvas
            self.world_range = [-self.canvas_size_m / 2, -self.canvas_size_m / 2, -3,
                                self.canvas_size_m / 2,  self.canvas_size_m / 2, 1]
            print(f"Using world canvas: {self.canvas_size_m}m x {self.canvas_size_m}m @ {self.canvas_res}m/pixel")
    
    def generate_anchor_box(self):
        """Generate anchors in world coordinates using canvas parameters."""
        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w'] 
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']
        
        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]
        
        # Use canvas resolution and world range if available
        if hasattr(self, 'canvas_res'):
            vh = self.canvas_res
            vw = self.canvas_res
            xrange = [self.world_range[0], self.world_range[3]]
            yrange = [self.world_range[1], self.world_range[4]]
            
            # Calculate grid dimensions based on canvas
            W = int((xrange[1] - xrange[0]) / vw)
            H = int((yrange[1] - yrange[0]) / vh)
        else:
            # Fallback to original parameters
            W = self.params['anchor_args']['W']
            H = self.params['anchor_args']['H']
            vh = self.params['anchor_args']['vh']
            vw = self.params['anchor_args']['vw']
            xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                      self.params['anchor_args']['cav_lidar_range'][3]]
            yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                      self.params['anchor_args']['cav_lidar_range'][4]]
        
        feature_stride = self.params['anchor_args'].get('feature_stride', 2)
        
        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)
        
        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0
        
        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h
        
        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]
        
        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')
        
        return anchors
    
    def post_process(self, data_dict, output_dict):
        """
        Process predictions in world coordinates and transform back to ego.
        
        The model outputs predictions in world coordinates (due to world anchor fusion).
        This method transforms them back to ego coordinates for evaluation.
        """
        print(f"[WORLD POST DBG] Entered WorldVoxelPostprocessor.post_process", flush=True)
        
        # For intermediate fusion, we process the 'ego' key
        if 'ego' in data_dict:
            ego_data = data_dict['ego']
            ego_output = output_dict
            
            # Get world anchor and ego pose
            world_anchor = ego_data['world_anchor'][0]  # [x, y, z]
            lidar_poses = ego_data['lidar_pose']
            ego_pose = lidar_poses[0][0] if isinstance(lidar_poses[0], list) else lidar_poses[0]
            
            # Create transformation matrices
            anchor_pose = [world_anchor[0], world_anchor[1], world_anchor[2], 0, 0, 0]
            T_world_to_ego = transformation_utils.x1_to_x2(anchor_pose, ego_pose)
            
            # Use identity matrix for world->world (no transformation needed for processing)
            transformation_matrix = torch.from_numpy(np.eye(4)).float()
            if 'anchor_box' in ego_data:
                anchor_box = ego_data['anchor_box']
            else:
                anchor_box = self.anchor_box
            
            # Process predictions in world coordinates
            prob = ego_output['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)
            
            reg = ego_output['rm']
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            
            mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
            
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])
            
            print(f"[WORLD POST DBG] {boxes3d.shape[0]} boxes after score threshold", flush=True)
            
            if len(boxes3d) == 0:
                return None, None
            
            # Handle direction classification if present
            if 'dm' in ego_output.keys():
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']
                
                dm = ego_output['dm']
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
                dir_cls_preds = dir_cls_preds[mask]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
                
                period = (2 * np.pi / num_bins)
                dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)
            
            # Convert to corners (still in world coordinates)
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
            
            # Transform corners from world to ego coordinates
            boxes3d_corner_np = boxes3d_corner.cpu().numpy()
            transformed_corners = []
            
            for i in range(len(boxes3d_corner_np)):
                corners = boxes3d_corner_np[i]  # (8, 3)
                corners_homo = np.hstack([corners, np.ones((8, 1))])  # (8, 4)
                corners_ego = (T_world_to_ego @ corners_homo.T).T[:, :3]  # Transform to ego
                transformed_corners.append(corners_ego)
            
            if len(transformed_corners) > 0:
                projected_boxes3d = torch.from_numpy(np.array(transformed_corners)).to(boxes3d.device)
                
                # Convert to 2D boxes for NMS
                projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
                
                # Apply filters
                keep_index_1 = box_utils.remove_large_pred_bbx(projected_boxes3d, self.dataset)
                keep_index_2 = box_utils.remove_bbx_abnormal_z(projected_boxes3d)
                keep_index = torch.logical_and(keep_index_1, keep_index_2)
                
                projected_boxes3d = projected_boxes3d[keep_index]
                scores = scores[keep_index]
                
                if len(scores) == 0:
                    return None, None
                
                # NMS
                keep_index = box_utils.nms_rotated(projected_boxes3d, scores, self.params['nms_thresh'])
                projected_boxes3d = projected_boxes3d[keep_index]
                scores = scores[keep_index]
                
                # Range filter (in ego coordinates)
                mask = box_utils.get_mask_for_boxes_within_range_torch(
                    projected_boxes3d, 
                    self.params['anchor_args']['cav_lidar_range']
                )
                projected_boxes3d = projected_boxes3d[mask, :, :]
                scores = scores[mask]
                
                print(f"[WORLD POST DBG] Final: {projected_boxes3d.shape[0]} boxes", flush=True)
                return projected_boxes3d, scores
            
            return None, None
        
        else:
            # Fallback to original behavior for late fusion
            return super().post_process(data_dict, output_dict)

