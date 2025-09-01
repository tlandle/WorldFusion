
import torch
from collections import OrderedDict
from opencood.data_utils.datasets.opv2v.lidarcameraintermediatefusionv2 import \
    LiDARCameraIntermediateFusionDataset
from opencood.utils.heter_utils import Adaptor
from opencood.utils.transformation_utils import x1_to_x2, get_pairwise_transformation
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.pre_processor import build_preprocessor

def get_V2XSim2FusionDataset(base_cls):
    class V2XSim2FusionDataset(base_cls):
        """
        Intermediate fusion dataset supporting V2X-Sim 2.0 (heterogeneous modalities).
        Inherits the LiDARCameraIntermediateFusionDataset API but extends it to parse
        V2XSim 2.0 .pkl info files, handle multiple modalities (m1..m4), and
        produce the per-agent batch structure expected by models like f‑Cooper or your
        world‑fusion point pillar.
        """

        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.heterogeneous = True
            # build modality assignment and adaptor (similar to IntermediateheterFusionDataset)
            modality_setting  = params['heter']['modality_setting']
            mapping_dict      = params['heter']['mapping_dict']
            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            cav_preference    = params['heter'].get('cav_preference', None)
            self.modality_name_list = list(modality_setting.keys())
            self.sensor_type_dict = OrderedDict()

            for modality_name, modal_setting in modality_setting.items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}",
                            build_preprocessor(modal_setting['preprocess'], train))
                elif modal_setting['sensor_type'] == 'camera':
                    # if you later add cameras for V2XSim, keep data_aug_conf_*
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])
                else:
                    raise NotImplementedError("Unsupported sensor type")

            self.ego_modality   = params['heter']['ego_modality']
            self.modality_assignment = params['heter'].get('assignment_path', None)
            self.adaptor = Adaptor(self.ego_modality,
                                   self.modality_name_list,
                                   self.modality_assignment,
                                   lidar_channels_dict,
                                   mapping_dict,
                                   cav_preference,
                                   train)

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Override the base class to process V2XSim 2.0 entries.
            Handles multiple sensor modalities via `modality_name`.
            Returns a dict with keys:
                processed_features_<modality>, transformation_matrix, ...
            """
            modality_name = selected_cav_base['modality_name']
            sensor_type   = self.sensor_type_dict[modality_name]

            # Compute transform T_cav→ego in world coordinates
            T_i2ego = x1_to_x2(selected_cav_base['params']['lidar_pose'],
                               ego_cav_base['params']['lidar_pose'])
            selected_cav_processed = {
                'transformation_matrix': T_i2ego,
                'lidar_pose': selected_cav_base['params']['lidar_pose']
            }

            # Lidar processing
            if sensor_type == 'lidar':
                lidar_np = selected_cav_base['lidar_np']
                # shuffle, remove ego hits, etc. (reuse the OPV2V logic)
                # ...
                # Preprocess per modality
                proc = getattr(self, f"pre_processor_{modality_name}")
                processed = proc.preprocess(lidar_np)
                selected_cav_processed[f'processed_features_{modality_name}'] = processed

            # Camera processing (not shown; reuse base if required)
            # If V2XSim2.0 includes images, handle augmentation and depth here.

            # Generate per‑agent labels in ego frame
            # (call the base class or V2XSim utilities for GT)
            obj_center, obj_mask, obj_ids = \
                self.generate_object_center([selected_cav_base],
                                            ego_cav_base['params']['lidar_pose'])
            selected_cav_processed['object_bbx_center'] = obj_center[obj_mask == 1]
            selected_cav_processed['object_bbx_mask']   = obj_mask
            selected_cav_processed['object_ids']        = obj_ids

            return selected_cav_processed

        def __getitem__(self, idx):
            """
            Assemble a sample: read the V2XSim2.0 info at index `idx`, identify
            all agents within communication range, gather their processed features,
            and build the batch dictionary matching LiDARCameraIntermediateFusionV2.
            """
            base_dict = self.retrieve_base_data(idx)
            # Add noise if needed, as in the heter loader
            # base_dict = add_noise_data_dict(base_dict, self.params['noise_setting'])

            ego_id, ego_cav_base = None, None
            for cav_id, cav in base_dict.items():
                if cav['ego']:
                    ego_id = cav_id
                    ego_cav_base = cav
                    break

            # Build per‑agent data lists similar to the heter loader:
            processed_features = []
            cav_id_list       = []
            lidar_pose_list   = []
            object_stack      = []
            object_id_stack   = []
            # For each candidate agent
            for cav_id, cav in base_dict.items():
                # filter by range; adapt modalities if necessary
                # ...
                processed_cav = self.get_item_single_car(cav, ego_cav_base)
                processed_features.append(processed_cav[f'processed_features_{cav["modality_name"]}'])
                cav_id_list.append(cav_id)
                lidar_pose_list.append(processed_cav['lidar_pose'])
                object_stack.append(processed_cav['object_bbx_center'])
                object_id_stack += processed_cav['object_ids']

            # Build pairwise transforms anchored at ego (or world)
            pairwise = get_pairwise_transformation(base_dict,
                                                   self.max_cav,
                                                   proj_first=False)  # or True for proj_first

            # Merge features into the expected dictionary format
            merged_feat_dict = merge_features_to_dict(processed_features)
            label_dict = self.post_processor.generate_label(
                gt_box_center=np.zeros((self.params['postprocess']['max_num'], 7)),
                anchors=self.anchor_box,
                mask=np.zeros(self.params['postprocess']['max_num'])
            )
            data_dict = OrderedDict()
            data_dict['ego'] = {
                'cav_num': len(cav_id_list),
                'processed_lidar': merged_feat_dict,
                'image_inputs': None,  # no cameras if only LiDAR
                'record_len': torch.tensor([len(cav_id_list)]),
                'pairwise_t_matrix': pairwise,
                'label_dict': label_dict,
                'anchor_box': self.anchor_box,
                'lidar_poses': lidar_pose_list,
                'object_ids': object_id_stack
            }
            return data_dict

    return V2XSim2FusionDataset
