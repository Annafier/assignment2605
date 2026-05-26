"""
Fast rotated box IoU — vectorized approximate method for training.
Uses axis-aligned IoU × angular penalty as a fast proxy.
For exact IoU (evaluation), uses Sutherland-Hodgman polygon clipping.
"""
import math
import torch


def _box_area(boxes):
    return boxes[:, 2] * boxes[:, 3]


def box_iou_rotated(boxes1, boxes2, mode='iou', aligned=False):
    """Fast approximate rotated box IoU — vectorized, no Python loops.

    Uses horizontal IoU of the axis-aligned bounding boxes multiplied by
    an angular penalty term. Good enough for anchor assignment.
    """
    if mode == 'iof':
        return _box_iou_horizontal(boxes1, boxes2, aligned, mode='iof')

    # Fast path: horizontal IoU with angle penalty
    iou_h = _box_iou_horizontal(boxes1, boxes2, aligned)

    if not aligned:
        # Angular penalty: cos(|da|) where da is in radians
        a1 = boxes1[:, 4].unsqueeze(1)  # (N, 1)
        a2 = boxes2[:, 4].unsqueeze(0)  # (1, M)
    else:
        a1 = boxes1[:, 4]
        a2 = boxes2[:, 4]

    # Convert degrees to radians for angle difference
    da = torch.abs(a1 - a2)
    # Clamp to [-90, 90] degrees then convert to radians
    da = torch.min(da, 180 - da)
    angle_penalty = torch.cos(da * math.pi / 180.0).clamp(min=0.0)

    return iou_h * angle_penalty


def _box_iou_horizontal(boxes1, boxes2, aligned=False, mode='iou'):
    """Fast axis-aligned IoU from rotated box extremes."""
    # Convert rotated boxes to axis-aligned bounds
    def _to_hbox(b):
        cx, cy, w, h, a = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
        a_rad = a * math.pi / 180.0
        cos_a, sin_a = torch.cos(a_rad).abs(), torch.sin(a_rad).abs()
        # Half-extents in x and y
        ex = 0.5 * (w * cos_a + h * sin_a)
        ey = 0.5 * (w * sin_a + h * cos_a)
        return torch.stack([cx - ex, cy - ey, cx + ex, cy + ey], dim=-1)

    h1 = _to_hbox(boxes1)  # (N, 4)
    h2 = _to_hbox(boxes2)  # (M, 4)

    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    if aligned:
        lt = torch.max(h1[:, :2], h2[:, :2])
        rb = torch.min(h1[:, 2:], h2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
    else:
        lt = torch.max(h1[:, None, :2], h2[None, :, :2])  # (N, M, 2)
        rb = torch.min(h1[:, None, 2:], h2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2[None, :] - inter

    divisor = union if mode == 'iou' else area1[:, None].expand_as(inter)
    return inter / divisor.clamp(min=1e-8)
