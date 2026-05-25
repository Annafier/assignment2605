"""
Pure-PyTorch rotated box IoU — replaces mmcv.ops.box_iou_rotated.

Algorithm: Sutherland-Hodgman polygon clipping for intersection area,
then IoU = intersection / (area1 + area2 - intersection).
"""
import math
import torch


def _get_corners(boxes):
    """Convert rotated boxes (cx, cy, w, h, a) to 4 corner points.

    Args:
        boxes: (N, 5) — (cx, cy, w, h, angle_rad)
    Returns:
        corners: (N, 4, 2) — 4 corners in (x, y) order
    """
    cx, cy, w, h, a = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    # Half-dimension offsets from center
    dx = torch.stack([-w/2, w/2, w/2, -w/2], dim=1)  # (N, 4)
    dy = torch.stack([-h/2, -h/2, h/2, h/2], dim=1)  # (N, 4)
    # Rotate
    rx = dx * cos_a.unsqueeze(1) - dy * sin_a.unsqueeze(1) + cx.unsqueeze(1)
    ry = dx * sin_a.unsqueeze(1) + dy * cos_a.unsqueeze(1) + cy.unsqueeze(1)
    return torch.stack([rx, ry], dim=2)  # (N, 4, 2)


def _polygon_area(corners):
    """Compute polygon area via shoelace formula.

    Args:
        corners: (..., N, 2)
    Returns:
        area: (...)
    """
    x = corners[..., 0]
    y = corners[..., 1]
    return 0.5 * torch.abs(
        (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]) +
        (x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1]) +
        (x[..., 2] * y[..., 3] - x[..., 3] * y[..., 2]) +
        (x[..., 3] * y[..., 0] - x[..., 0] * y[..., 3])
    )


def _box_area(boxes):
    """Area of rotated boxes = w * h."""
    return boxes[:, 2] * boxes[:, 3]


def _clip_segment_to_halfplane(p1, p2, edge_p1, edge_p2, inside_fn):
    """Clip line segment (p1, p2) against a half-plane edge.

    Returns list of 0, 1, or 2 points.
    """
    inside1 = inside_fn(p1, edge_p1, edge_p2)
    inside2 = inside_fn(p2, edge_p1, edge_p2)

    if inside1 and inside2:
        return [p1, p2]
    if not inside1 and not inside2:
        return []
    # One inside, one outside — find intersection
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ex, ey = edge_p2[0] - edge_p1[0], edge_p2[1] - edge_p1[1]
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return [p1] if inside1 else [p2]
    t = ((edge_p1[0] - p1[0]) * ey - (edge_p1[1] - p1[1]) * ex) / denom
    inter = (p1[0] + t * dx, p1[1] + t * dy)
    result = [p1] if inside1 else []
    result.append(inter)
    if inside2:
        result.append(p2)
    return result


def _intersection_area(corners1, corners2):
    """Compute intersection area of two convex polygons using Sutherland-Hodgman.

    Args:
        corners1: (4, 2) — first polygon corners
        corners2: (4, 2) — second polygon corners
    Returns:
        area: scalar
    """
    # Start with subject polygon
    output = [(corners1[i, 0].item(), corners1[i, 1].item()) for i in range(4)]

    # Clip against each edge of the clip polygon
    for i in range(4):
        if not output:
            return 0.0
        edge_p1 = (corners2[i, 0].item(), corners2[i, 1].item())
        edge_p2 = (corners2[(i+1) % 4, 0].item(), corners2[(i+1) % 4, 1].item())
        input_list = output
        output = []

        def inside(p, e1, e2):
            cross = (e2[0] - e1[0]) * (p[1] - e1[1]) - (e2[1] - e1[1]) * (p[0] - e1[0])
            return cross >= -1e-12

        for j in range(len(input_list)):
            cur = input_list[j]
            prev = input_list[j - 1]
            clipped = _clip_segment_to_halfplane(prev, cur, edge_p1, edge_p2, inside)
            output.extend(clipped)

    # Shoelace on clipped polygon
    if len(output) < 3:
        return 0.0
    area = 0.0
    for i in range(len(output)):
        j = (i + 1) % len(output)
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return abs(area) * 0.5


def box_iou_rotated(boxes1, boxes2, mode='iou', aligned=False):
    """Pure-PyTorch rotated box IoU — drop-in for mmcv.ops.box_iou_rotated.

    Args:
        boxes1: (N, 5) — (cx, cy, w, h, angle_rad)
        boxes2: (M, 5)
        mode: 'iou' or 'iof'
        aligned: if True, boxes1[i] ↔ boxes2[i] only
    Returns:
        ious: (N, M) or (N,) if aligned
    """
    # Compute box areas
    area1 = _box_area(boxes1)  # (N,)
    area2 = _box_area(boxes2)  # (M,)

    # Get corners
    corners1 = _get_corners(boxes1)  # (N, 4, 2)
    corners2 = _get_corners(boxes2)  # (M, 4, 2)

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if aligned:
        assert N == M
        intersections = torch.zeros(N, device=boxes1.device, dtype=boxes1.dtype)
        for i in range(N):
            intersections[i] = _intersection_area(corners1[i], corners2[i])
        union = area1 + area2 - intersections if mode == 'iou' else area1
        return intersections / union.clamp(min=1e-8)
    else:
        intersections = torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)
        for i in range(N):
            for j in range(M):
                intersections[i, j] = _intersection_area(corners1[i], corners2[j])
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersections
        if mode == 'iof':
            union = area1.unsqueeze(1).expand(N, M)
        return intersections / union.clamp(min=1e-8)
