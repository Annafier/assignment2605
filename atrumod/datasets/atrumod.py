"""
ATR-UMOD dataset class for MMRotate.
Handles paired RGB + IR images with DOTA-format rotated bounding box labels.
"""
import os.path as osp
from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset


@DATASETS.register_module()
class ATRUMODDataset(DOTADataset):
    """
    ATR-UMOD paired RGB-IR oriented object detection dataset.
    Expects DOTA-format labels in data/<split>/dota_labels/ and dota_labels_ir/.

    Condition attributes (UAVangle, UAVheight, weather, illumination, time, location)
    are available via the original XML files but not used during standard training.
    """
    METAINFO = {
        'classes':
        ('car', 'suv', 'van', 'bus', 'freight_car', 'truck',
         'motorcycle', 'trailer', 'tank_truck', 'excavator', 'crane'),
        'palette': [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30)
        ]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
