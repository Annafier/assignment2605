"""ATR-UMOD dataset — registered in mmdet scope."""
import os.path as osp
import glob
import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from atrumod.structures.rotated_boxes import RotatedBoxes  # noqa: F401 — register rbox type


@DATASETS.register_module()
class ATRUMODDataset(BaseDetDataset):
    """ATR-UMOD rotated object detection dataset (DOTA format labels).

    11 vehicle classes with rotated bounding boxes.
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

    def load_data_list(self):
        """Load DOTA-format annotations from ann_file directory."""
        data_list = []
        # self.ann_file already has data_root prepended by mmdet
        ann_dir = self.ann_file
        txt_files = glob.glob(osp.join(ann_dir, '*.txt'))

        if not txt_files:
            raise FileNotFoundError(f'No DOTA labels found in {ann_dir}')

        for txt_file in txt_files:
            img_id = osp.splitext(osp.basename(txt_file))[0]
            # self.data_prefix already has data_root prepended by mmdet
            img_path = osp.join(self.data_prefix['img_path'], f'{img_id}.jpg')

            instances = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue

                # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 classname difficult
                try:
                    poly = np.array([float(p) for p in parts[:8]], dtype=np.float32)
                    class_name = parts[8]
                    if class_name not in self.METAINFO['classes']:
                        continue

                    # Convert 4-point polygon to rotated box (cx, cy, w, h, angle)
                    bbox = self._poly2rbox(poly)

                    label = self.METAINFO['classes'].index(class_name)
                    instances.append(dict(bbox=bbox.astype(np.float32),
                                          bbox_label=label,
                                          ignore_flag=0))
                except (ValueError, IndexError):
                    continue

            data_list.append(dict(img_path=img_path, instances=instances))

        return data_list

    @staticmethod
    def _poly2rbox(poly):
        """Convert 4-point polygon to (cx, cy, w, h, angle_deg)."""
        import cv2
        rect = cv2.minAreaRect(poly.reshape(4, 2).astype(np.float32))
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        # Normalize: ensure w >= h, angle in [-90, 90)
        if w < h:
            w, h = h, w
            angle = angle + 90
        angle = (angle + 90) % 180 - 90  # normalize to [-90, 90)
        return np.array([cx, cy, w, h, angle], dtype=np.float32)
