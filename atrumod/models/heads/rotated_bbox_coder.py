"""DeltaXYWHA bbox coder for rotated boxes — self-contained."""
import torch


class DeltaXYWHAOBBoxCoder:
    """Encode/decode rotated box deltas (cx, cy, w, h, angle).

    encoding:  (gt - anchor) → delta
    decoding:  delta → predicted box

    Uses 'le135' angle range ([-135, 135) degrees).
    """

    def __init__(self, angle_range='le135', norm_factor=None,
                 edge_swap=False, proj_xy=True,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        self.angle_range = angle_range
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy
        self.target_means = target_means
        self.target_stds = target_stds

    def encode(self, anchors, gt_bboxes):
        """Encode gt_bboxes relative to anchors.

        Args:
            anchors: (N, 5) — (cx, cy, w, h, angle_deg)
            gt_bboxes: (N, 5) — same format
        Returns:
            deltas: (N, 5)
        """
        means = anchors.new_tensor(self.target_means)
        stds = anchors.new_tensor(self.target_stds)

        da = gt_bboxes[:, 2] / anchors[:, 2]  # dw
        db = gt_bboxes[:, 3] / anchors[:, 3]  # dh
        dw = da.clamp(min=0)  # avoid log of negative
        dh = db.clamp(min=0)

        dx = (gt_bboxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
        dy = (gt_bboxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
        ddw = torch.log(dw)
        ddh = torch.log(dh)
        dangle = gt_bboxes[:, 4] - anchors[:, 4]

        # Normalize angle to [-90, 90)
        dangle = (dangle + 90) % 180 - 90

        deltas = torch.stack([dx, dy, ddw, ddh, dangle], dim=-1)
        deltas = (deltas - means) / stds
        return deltas

    def decode(self, anchors, deltas):
        """Decode deltas to predicted boxes.

        Args:
            anchors: (N, 5)
            deltas: (N, 5)
        Returns:
            bboxes: (N, 5)
        """
        means = anchors.new_tensor(self.target_means)
        stds = anchors.new_tensor(self.target_stds)
        deltas = deltas * stds + means

        dx, dy, ddw, ddh, dangle = deltas.unbind(-1)

        cx = dx * anchors[:, 2] + anchors[:, 0]
        cy = dy * anchors[:, 3] + anchors[:, 1]
        w = anchors[:, 2] * torch.exp(ddw).clamp(min=1e-8)
        h = anchors[:, 3] * torch.exp(ddh).clamp(min=1e-8)
        angle = dangle + anchors[:, 4]

        return torch.stack([cx, cy, w, h, angle], dim=-1)
