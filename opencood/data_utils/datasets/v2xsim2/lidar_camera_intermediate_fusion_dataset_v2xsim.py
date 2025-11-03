# In your new lidar_camera_intermediate_fusion_dataset_v2xsim.py

from collections import defaultdict
import numpy as np
from opencood.data_utils.datasets.v2xsim2.v2xsim2_base_dataset import V2XSIMBaseDataset

# ... other imports ...

class LiDARCameraIntermediateFusionDatasetV2XSim(V2XSIMBaseDataset): # Inherit from the base V2X-Sim dataset
    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        
        # This will store the static anchor for each scenario
        self.world_anchors = {}
        self._compute_static_world_anchors()

    def _compute_static_world_anchors(self):
        """
        Pre-computes a static world anchor for each scenario (log) in the dataset.
        The anchor is the average position of all vehicles at the first timestamp.
        """
        # Group all frames by their scenario ID (log_id)
        frames_by_log = defaultdict(list)
        for i, frame_info in enumerate(self.v2xsim_infos):
            log_id = frame_info['log_id']
            frames_by_log[log_id].append(frame_info)

        print("Computing static world anchors for each scenario...")
        for log_id, frames in frames_by_log.items():
            # The frames are already sorted by timestamp
            first_frame = frames[0]
            
            initial_poses = []
            # Get the pose of every vehicle in the first frame
            for cav_id, cav_content in first_frame['agents'].items():
                initial_poses.append(cav_content['lidar_pose'])
            
            if not initial_poses:
                # Default anchor if a scenario has no agents in the first frame
                world_anchor = [0.0, 0.0, 0.0]
            else:
                # Calculate the mean position (centroid)
                anchor_np = np.mean(np.array(initial_poses)[:, :3], axis=0)
                world_anchor = anchor_np.tolist()
            
            self.world_anchors[log_id] = world_anchor
        print(f"Computed anchors for {len(self.world_anchors)} scenarios.")
