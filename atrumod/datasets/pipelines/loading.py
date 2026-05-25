"""Custom data pipeline transforms for ATR-UMOD paired RGB-IR loading."""
import os.path as osp
from typing import Optional, Tuple
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRGBIRPair(BaseTransform):
    """Load paired RGB and IR images for multimodal detection.

    Required keys: img_path, ir_path (constructed from data_prefix)
    Added keys: img (RGB), img_ir (infrared), img_shape, ori_shape
    """

    def __init__(self, backend_args=None, to_float32=False):
        self.backend_args = backend_args
        self.to_float32 = to_float32

    def transform(self, results):
        from mmcv.transforms import LoadImageFromFile
        rgb_loader = LoadImageFromFile(to_float32=self.to_float32, backend_args=self.backend_args)

        # Load RGB
        rgb_results = rgb_loader.transform(dict(img_path=results['img_path']))
        results['img'] = rgb_results['img']

        # Load IR — derive ir_path from img_path
        ir_path = results['img_path'].replace('/images/', '/images_ir/')
        if not osp.exists(ir_path):
            ir_path = results.get('ir_path', ir_path)
        ir_results = rgb_loader.transform(dict(img_path=ir_path))
        results['img_ir'] = ir_results['img']

        results['img_shape'] = results['img'].shape[:2]
        results['ori_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class PackPairedDetInputs(BaseTransform):
    """Pack paired RGB-IR inputs for dual-stream detection models.

    Stacks img and img_ir as a single tensor with 6 channels (RGB3 + IR3),
    and adds img_ir to the results for models that handle dual streams internally.
    """

    def __init__(self, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')):
        self.meta_keys = meta_keys

    def transform(self, results):
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        import torch

        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if self.to_float32:
                img = img.astype(np.float32)
            packed_results['inputs'] = torch.from_numpy(img.transpose(2, 0, 1))

        # Also include IR as a separate key for dual-stream models
        if 'img_ir' in results:
            img_ir = results['img_ir']
            if len(img_ir.shape) < 3:
                img_ir = np.expand_dims(img_ir, -1)
            if self.to_float32:
                img_ir = img_ir.astype(np.float32)
            packed_results['inputs_ir'] = torch.from_numpy(img_ir.transpose(2, 0, 1))
        elif 'img' in results:
            packed_results['inputs_ir'] = packed_results['inputs']

        data_sample = DetDataSample()
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            gt_labels = results['gt_bboxes_labels']
            gt_instances = InstanceData()
            gt_instances.bboxes = gt_bboxes
            gt_instances.labels = gt_labels
            data_sample.gt_instances = gt_instances

        if 'ignored_labels' in results:
            valid_mask = ~results['ignored_labels']
            data_sample.set_field(valid_mask, 'valid_mask')

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = [data_sample]
        return packed_results
