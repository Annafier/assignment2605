"""
Pure-PyTorch rotated NMS — replaces mmcv.ops.nms_rotated and batched_nms.
"""
import torch


def nms_rotated(dets, scores, iou_threshold):
    """Greedy rotated NMS.

    Args:
        dets: (N, 5) — (cx, cy, w, h, angle_rad) × score-sorted order
        scores: (N,) — detection scores (used to determine sorting, assumed pre-sorted)
        iou_threshold: float — IoU above which boxes are suppressed
    Returns:
        keep_inds: LongTensor of indices to keep
    """
    from atrumod.ops.rotated_iou import box_iou_rotated

    if dets.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=dets.device)

    N = dets.shape[0]
    suppressed = torch.zeros(N, dtype=torch.bool, device=dets.device)
    keep = []

    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(i)
        if i == N - 1:
            break
        ious = box_iou_rotated(dets[i:i+1], dets[i+1:], aligned=False)[0]
        suppressed[i+1:] |= (ious > iou_threshold)

    return torch.tensor(keep, dtype=torch.long, device=dets.device)


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Per-class batched NMS — drop-in for mmcv.ops.batched_nms.

    Args:
        boxes: (N, 4) or (N, 5) — regular or rotated boxes
        scores: (N,)
        idxs: (N,) — class indices
        nms_cfg: dict with 'iou_threshold', 'type' (unused, always greedy)
        class_agnostic: if True, NMS across all classes jointly
    Returns:
        keep: (K,) LongTensor
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    iou_threshold = nms_cfg.get('iou_thr', nms_cfg.get('iou_threshold', 0.5))

    if class_agnostic:
        # Single NMS across all classes
        _, order = scores.sort(descending=True)
        keep = nms_rotated(boxes[order], scores[order], iou_threshold)
        return order[keep]
    else:
        # Per-class NMS
        max_coords = boxes.max()
        offsets = idxs.to(boxes) * (max_coords + 1)
        boxes_offset = boxes + offsets[:, None] if boxes.shape[1] == 4 else boxes.clone()
        if boxes_offset.shape[1] == 5:
            boxes_offset[:, :2] += offsets[:, None]
        _, order = scores.sort(descending=True)
        keep = nms_rotated(boxes_offset[order], scores[order], iou_threshold)
        return order[keep]
