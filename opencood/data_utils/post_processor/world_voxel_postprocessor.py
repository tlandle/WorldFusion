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
        """Generate anchors matching the model's output dimensions."""
        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w'] 
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']
        
        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]
        
        # Use world coordinates defined by the canvas
        if 'canvas_size_m' in self.params['anchor_args']:
            canvas_size = self.canvas_size_m
            res = self.canvas_res
            
            # CRITICAL FIX: Derive grid size from canvas parameters
            # The pre-processor will create a grid of this size.
            W = H = int(canvas_size / res)
            
            xrange = [-canvas_size / 2, canvas_size / 2]
            yrange = [-canvas_size / 2, canvas_size / 2]
            # Voxel width/height is just the resolution
            vw = vh = res
        else:
            # Fallback to original ego-sized grid logic
            xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                      self.params['anchor_args']['cav_lidar_range'][3]]
            yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                      self.params['anchor_args']['cav_lidar_range'][4]]
            vw = self.params['anchor_args']['vw']
            vh = self.params['anchor_args']['vh']
            W = self.params['anchor_args']['W']
            H = self.params['anchor_args']['H']
        
        feature_stride = self.params['anchor_args'].get('feature_stride', 4)
        
        # Generate anchor centers. We add a half-voxel offset (vw/2) to center the anchors in the grid cells.
        x = np.arange(xrange[0] + vw / 2, xrange[1], vw * feature_stride)
        y = np.arange(yrange[0] + vh / 2, yrange[1], vh * feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0
        
        w_arr = np.ones_like(cx) * w
        l_arr = np.ones_like(cx) * l
        h_arr = np.ones_like(cx) * h
        
        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]
        
        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h_arr, w_arr, l_arr, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l_arr, h_arr, w_arr, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')
        
        return anchors
      
    def generate_label(self, **kwargs):
        """
        Generate training labels in WORLD coordinates to match model predictions.
        """
        world_anchor = kwargs.get('world_anchor', None)
        ego_pose = kwargs.get('ego_pose', None)
        
        #print(f"[LABEL GEN] Called with world_anchor: {world_anchor}, ego_pose: {ego_pose}")
        
        if world_anchor is None or ego_pose is None:
            print("[LABEL GEN] Falling back to parent implementation (ego coords)")
            return super().generate_label(**kwargs)
        
        # Get the required parameters
        gt_box_center = kwargs['gt_box_center']
        anchors = kwargs['anchors']
        mask = kwargs['mask']
        
        #print(f"[LABEL GEN] GT box range BEFORE transform: x=[{gt_box_center[:, 0].min():.1f}, {gt_box_center[:, 0].max():.1f}], y=[{gt_box_center[:, 1].min():.1f}, {gt_box_center[:, 1].max():.1f}]")
        
        # Transform GT boxes from ego to world coordinates
        anchor_pose = [world_anchor[0], world_anchor[1], world_anchor[2], 0, 0, 0]
        T_ego_to_world = transformation_utils.x1_to_x2(ego_pose, anchor_pose)
        
        # Transform GT centers to world frame
        gt_box_center_world = np.copy(gt_box_center)
        for i in range(len(gt_box_center)):
            if mask[i] == 1:
                center_homo = np.append(gt_box_center[i, :3], 1)
                center_world = (T_ego_to_world @ center_homo)[:3]
                gt_box_center_world[i, :3] = center_world
        
        #print(f"[LABEL GEN] GT box range AFTER transform: x=[{gt_box_center_world[:, 0].min():.1f}, {gt_box_center_world[:, 0].max():.1f}], y=[{gt_box_center_world[:, 1].min():.1f}, {gt_box_center_world[:, 1].max():.1f}]")
        #print(f"[LABEL GEN] Anchor range: x=[{anchors[..., 0].min():.1f}, {anchors[..., 0].max():.1f}], y=[{anchors[..., 1].min():.1f}, {anchors[..., 1].max():.1f}]")
        
        # Update kwargs with transformed GT boxes
        kwargs_world = kwargs.copy()
        kwargs_world['gt_box_center'] = gt_box_center_world
        
        # Generate labels and check results
        label_dict = super().generate_label(**kwargs_world)
        
        num_pos = (label_dict['pos_equal_one'] > 0).sum()
        num_neg = (label_dict['neg_equal_one'] > 0).sum()
        #print(f"[LABEL GEN] Generated {num_pos} positive and {num_neg} negative anchors")
        
        return label_dict

    def post_process(self, data_dict, output_dict):
        """
        Process predictions in world coordinates and transform back to ego.
        
        The model outputs predictions in world coordinates (due to world anchor fusion).
        This method transforms them back to ego coordinates for evaluation.
        """
        print(f"[WORLD POST DBG] Entered WorldVoxelPostprocessor.post_process", flush=True)
        print(f"[WORLD POST DBG] output_dict keys: {output_dict.keys()}", flush=True)
        print(f"[WORLD POST DBG] data_dict keys: {data_dict.keys()}", flush=True)

        # Handle the ego wrapper in output_dict
        if 'ego' in output_dict:
            ego_output = output_dict['ego']
        else:
            ego_output = output_dict
        
        # Get data from data_dict
        if 'ego' in data_dict:
            ego_data = data_dict['ego']
            world_anchor = ego_data['world_anchor'][0]
            lidar_poses = ego_data['lidar_pose']
            anchor_box = ego_data.get('anchor_box', self.generate_anchor_box())
        else:
            # Fallback for late fusion
            return super().post_process(data_dict, output_dict)
        
        ego_pose = lidar_poses[0][0] if isinstance(lidar_poses[0], list) else lidar_poses[0]
        
        # Create transformation matrices
        anchor_pose = [world_anchor[0], world_anchor[1], world_anchor[2], 0, 0, 0]
        T_world_to_ego = transformation_utils.x1_to_x2(anchor_pose, ego_pose)
        
        # Process predictions - using ego_output
        prob = ego_output['psm']
        prob = F.sigmoid(prob.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)

     # ============= ADD VERIFICATION TEST HERE =============
        # TEMPORARY DEBUG: Test if spatial misalignment is the issue
        psm_original = ego_output['psm']  # Keep original PSM
        
        # Test 1: Check confidence at different spatial offsets
        print(f"\n[VERIFICATION TEST] Testing spatial alignment hypothesis:")
        print(f"Original max confidence: {prob.max().item():.4f}")
        
        # Try different spatial shifts to see if we can find higher confidences
        shift_amounts = [(-20, -20), (-10, -10), (0, 0), (10, 10), (20, 20)]
        for shift_y, shift_x in shift_amounts:
            shifted_psm = torch.roll(psm_original, shifts=(shift_y, shift_x), dims=(2, 3))
            shifted_prob = F.sigmoid(shifted_psm.permute(0, 2, 3, 1)).reshape(1, -1)
            max_conf = shifted_prob.max().item()
            num_high_conf = (shifted_prob > 0.3).sum().item()
            print(f"  Shift ({shift_y:3d}, {shift_x:3d}): max_conf={max_conf:.4f}, num>0.3={num_high_conf}")
        
        # Test 2: Check the distribution of probabilities
        print(f"\n[VERIFICATION TEST] Probability distribution:")
        print(f"  Num > 0.1: {(prob > 0.1).sum().item()}")
        print(f"  Num > 0.2: {(prob > 0.2).sum().item()}")
        print(f"  Num > 0.3: {(prob > 0.3).sum().item()}")
        print(f"  Num > 0.5: {(prob > 0.5).sum().item()}")
        print(f"  Top 10 confidences: {prob.topk(10)[0].tolist()}")
        # ============= END VERIFICATION TEST =============

        # DEBUG: Check probability statistics
        print(f"[WORLD POST DBG] Prob shape: {prob.shape}")
        print(f"[WORLD POST DBG] Prob max: {prob.max().item():.4f}, min: {prob.min().item():.4f}, mean: {prob.mean().item():.4f}")
        print(f"[WORLD POST DBG] Score threshold: {self.params['target_args']['score_threshold']}")
        print(f"[WORLD POST DBG] Num above 0.1: {(prob > 0.1).sum().item()}")
        print(f"[WORLD POST DBG] Num above 0.3: {(prob > 0.3).sum().item()}")
        print(f"[WORLD POST DBG] Num above threshold: {(prob > self.params['target_args']['score_threshold']).sum().item()}")
    
        
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
    
