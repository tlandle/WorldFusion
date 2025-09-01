# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.world_voxel_postprocessor import WorldVoxelPostprocessor

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'WorldVoxelPostprocessor': WorldVoxelPostprocessor
}


def build_postprocessor(anchor_cfg, dataset, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'WorldVoxelPostprocessor'], \
        f"Not support {process_method_name} postprocessor"
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        dataset=dataset,
        train=train,
    )

    return anchor_generator
