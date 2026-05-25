"""Rotated boxes (cx, cy, w, h, angle_deg) — mmdet-compatible structure."""
import torch
from mmdet.structures.bbox import BaseBoxes, register_box, register_box_converter
from mmdet.structures.bbox import HorizontalBoxes


@register_box('rbox')
class RotatedBoxes(BaseBoxes):
    """Rotated bounding boxes: (cx, cy, w, h, angle_deg).

    Registered as 'rbox' in mmdet's box registry. Use with:
        dict(type='LoadAnnotations', box_type='rbox')
    """
    box_dim: int = 5

    def __init__(self, data, dtype=None, device=None, clone=True):
        super().__init__(data=data, dtype=dtype, device=device, clone=clone)

    def __repr__(self):
        return f'RotatedBoxes(shape={self.tensor.shape})'

    @property
    def centers(self):
        return self.tensor[..., :2]

    @property
    def widths(self):
        return self.tensor[..., 2]

    @property
    def heights(self):
        return self.tensor[..., 3]

    @property
    def angles(self):
        return self.tensor[..., 4]

    @classmethod
    def cat(cls, box_list):
        return cls(torch.cat([b.tensor for b in box_list], dim=0))

    def flip_(self, img_shape, direction='horizontal'):
        pass  # Simplified — no flip transform needed for now

    def translate_(self, distances):
        self.tensor[..., 0] += distances[..., 0]
        self.tensor[..., 1] += distances[..., 1]

    def clip_(self, img_shape):
        pass

    def rotate_(self, center, angle):
        pass

    def rescale_(self, scale_factor):
        self.tensor[..., :4] *= scale_factor if isinstance(scale_factor, (int, float)) else scale_factor[0]

    def resize_(self, out_shape):
        pass

    @property
    def regular(self):
        return self

    @property
    def areas(self):
        return self.tensor[..., 2] * self.tensor[..., 3]

    def find_inside_points(self, points, is_aligned=False):
        raise NotImplementedError

    @classmethod
    def from_instance_masks(cls, masks):
        raise NotImplementedError

    def is_inside(self, img_shape, homography_matrix=None):
        return torch.ones(self.tensor.shape[0], dtype=torch.bool, device=self.tensor.device)

    def overlaps(self, other, mode='iou', is_aligned=False, eps=1e-6):
        from atrumod.ops.rotated_iou import box_iou_rotated
        return box_iou_rotated(self.tensor, other.tensor, mode=mode, aligned=is_aligned)

    def project_(self, homography_matrix):
        pass


@register_box_converter('rbox', 'rbox')
def rbox2rbox(boxes):
    return boxes
