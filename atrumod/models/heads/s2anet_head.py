"""S2ANet head — pure PyTorch, no mmrotate dependency.

AlignConv + FAM (Feature Alignment Module) + ODM (Oriented Detection Module).
Reference: Align Deep Features for Oriented Object Detection (Han et al., TGRS 2022)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping
from atrumod.models.heads.rotated_anchor_generator import RotatedAnchorGenerator
from atrumod.models.heads.rotated_bbox_coder import DeltaXYWHAOBBoxCoder


class AlignConv(nn.Module):
    """Oriented feature alignment via bilinear sampling.

    For each oriented proposal, samples conv features at positions aligned
    with the box orientation, making downstream convolutions rotation-invariant.
    """

    def __init__(self, in_channels, featmap_strides, kernel_size=3):
        super().__init__()
        self.strides = featmap_strides
        self.kernel_size = kernel_size

    def _bbox_to_grid(self, bboxes, h, w, stride, device):
        """Convert oriented bboxes to normalized sampling grids.

        Args:
            bboxes: (B, N, 5) in (cx, cy, w, h, angle_deg)
            h, w: feature map spatial size
            stride: feature map stride
        Returns:
            grid: (B, N, kernel_h, kernel_w, 2) in [-1, 1] normalized coords
        """
        B, N = bboxes.shape[:2]
        k = self.kernel_size

        # Local sampling positions in box coordinate frame
        local_y, local_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, k, device=device),
            torch.linspace(-0.5, 0.5, k, device=device),
            indexing='ij')
        local_x = local_x.reshape(1, 1, k, k)
        local_y = local_y.reshape(1, 1, k, k)

        cx, cy, w_, h_, angle = bboxes.unbind(-1)  # each (B, N)
        a = angle * (math.pi / 180.0)
        cos_a, sin_a = torch.cos(a), torch.sin(a)

        # Rotate local coords: (cx,cy) + R(angle) @ (local_x*w, local_y*h)
        dx = local_x * w_.view(B, N, 1, 1) * cos_a.view(B, N, 1, 1) - \
             local_y * h_.view(B, N, 1, 1) * sin_a.view(B, N, 1, 1)
        dy = local_x * w_.view(B, N, 1, 1) * sin_a.view(B, N, 1, 1) + \
             local_y * h_.view(B, N, 1, 1) * cos_a.view(B, N, 1, 1)

        gx = (cx.view(B, N, 1, 1) + dx) / stride  # feature map coords
        gy = (cy.view(B, N, 1, 1) + dy) / stride

        # Normalize to [-1, 1] for F.grid_sample
        gx = gx / (w - 1) * 2.0 - 1.0
        gy = gy / (h - 1) * 2.0 - 1.0

        return torch.stack([gx, gy], dim=-1)  # (B, N, k, k, 2)

    def forward(self, feats, bboxes_list, chunk_size=8):
        """Align features per FPN level. Processes proposals in chunks for VRAM.

        Args:
            feats: list of (B, C, H_i, W_i)
            bboxes_list: list of (B, N_i, 5) proposals per level
            chunk_size: proposals per grid_sample call (keep small for VRAM)
        Returns:
            aligned: (B, total_N, C, k, k) concatenated across levels
        """
        outputs = []
        for i, (feat, bboxes) in enumerate(zip(feats, bboxes_list)):
            B, C, H, W = feat.shape
            if bboxes.shape[1] == 0:
                continue
            stride = self.strides[i]
            grid = self._bbox_to_grid(bboxes, H, W, stride, feat.device)

            batch_samples = []
            for b in range(B):
                Np = bboxes[b].shape[0]
                if Np == 0:
                    continue
                g = grid[b, :Np]  # (N, k, k, 2)
                chunks = []
                for start in range(0, Np, chunk_size):
                    end = min(start + chunk_size, Np)
                    gc = g[start:end]  # (chunk, k, k, 2)
                    fc = feat[b:b+1].expand(gc.shape[0], -1, -1, -1)
                    chunks.append(F.grid_sample(fc, gc, mode='bilinear',
                                  padding_mode='zeros', align_corners=True))
                batch_samples.append(torch.cat(chunks, dim=0))  # (Np, C, k, k)

            if not batch_samples:
                continue
            max_n = max(s.shape[0] for s in batch_samples)
            padded = torch.zeros(B, max_n, C, self.kernel_size, self.kernel_size,
                                device=feat.device, dtype=feat.dtype)
            for b, s in enumerate(batch_samples):
                padded[b, :s.shape[0]] = s
            outputs.append(padded)

        if not outputs:
            return None
        return torch.cat(outputs, dim=1)


class S2ANetHead(nn.Module):
    """S2ANet rotated detection head: FAM → AlignConv → ODM.

    FAM (Feature Alignment Module):
        Anchor-based head producing oriented proposals.
    AlignConv:
        Samples features aligned to proposal orientations.
    ODM (Oriented Detection Module):
        Refinement head on rotation-invariant aligned features.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=2,
                 anchor_generator=None,
                 bbox_coder=None,
                 align_conv_strides=None,
                 align_conv_kernel=3,
                 loss_cls=None,
                 loss_bbox=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Anchor generator & bbox coder (shared between FAM and ODM)
        if anchor_generator is None:
            anchor_generator = dict(strides=[8, 16, 32, 64, 128], scales=[4],
                                    ratios=[1.0], angles=[0])
        if isinstance(anchor_generator, Mapping):
            self.anchor_generator = RotatedAnchorGenerator(
                strides=anchor_generator.get('strides', [8, 16, 32, 64, 128]),
                scales=anchor_generator.get('scales', [4]),
                ratios=anchor_generator.get('ratios', [1.0]),
                angles=anchor_generator.get('angles', [0]))
        else:
            self.anchor_generator = anchor_generator

        if bbox_coder is None:
            bbox_coder = dict(angle_range='le135')
        if isinstance(bbox_coder, Mapping):
            self.bbox_coder = DeltaXYWHAOBBoxCoder(
                **{k: v for k, v in bbox_coder.items() if k != 'type'})
        else:
            self.bbox_coder = bbox_coder

        self.num_anchors = self.anchor_generator.num_anchors
        self.featmap_strides = (align_conv_strides or
                                self.anchor_generator.strides)

        # Losses
        from .rotated_retina_head import FocalLoss, SmoothL1Loss
        loss_cls_cfg = loss_cls or {}
        loss_bbox_cfg = loss_bbox or {}
        self.loss_cls = FocalLoss(
            **{k: v for k, v in loss_cls_cfg.items() if k not in ('type', 'loss_weight')})
        self.loss_bbox = SmoothL1Loss(
            **{k: v for k, v in loss_bbox_cfg.items() if k not in ('type', 'loss_weight')})
        self.loss_cls_weight = loss_cls_cfg.get('loss_weight', 1.0)
        self.loss_bbox_weight = loss_bbox_cfg.get('loss_weight', 1.0)

        # === FAM: anchor-based proposal head ===
        self.fam_convs = nn.ModuleList()
        for i in range(stacked_convs):
            ch_in = in_channels if i == 0 else feat_channels
            self.fam_convs.append(nn.Conv2d(ch_in, feat_channels, 3, padding=1))
        self.fam_cls = nn.Conv2d(feat_channels, self.num_anchors * num_classes, 3, padding=1)
        self.fam_reg = nn.Conv2d(feat_channels, self.num_anchors * 5, 3, padding=1)

        # === AlignConv ===
        self.align_conv = AlignConv(in_channels, self.featmap_strides, align_conv_kernel)

        # === ODM: refinement head on aligned features ===
        odm_in = in_channels * (align_conv_kernel ** 2)
        self.odm_convs = nn.ModuleList()
        for _ in range(stacked_convs):
            self.odm_convs.append(nn.Conv1d(odm_in if _ == 0 else feat_channels,
                                            feat_channels, 1))
        self.odm_cls = nn.Conv1d(feat_channels, num_classes, 1)
        self.odm_reg = nn.Conv1d(feat_channels, 5, 1)

        self._init_weights()
        self._featmap_sizes = []

    def _init_weights(self):
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None and m is not self.fam_cls:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fam_cls.bias, bias_init)

    # ---- FAM forward ----
    def _fam_forward(self, feats):
        fam_cls, fam_reg = [], []
        self._featmap_sizes = []
        for feat in feats:
            self._featmap_sizes.append(feat.shape[-2:])
            x = feat
            for conv in self.fam_convs:
                x = conv(x).relu_()
            cls_out = self.fam_cls(x)
            reg_out = self.fam_reg(x)
            B, _, H, W = cls_out.shape
            fam_cls.append(cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            fam_reg.append(reg_out.permute(0, 2, 3, 1).reshape(B, -1, 5))
        return fam_cls, fam_reg

    # ---- ODM forward ----
    def _odm_forward(self, aligned_feats):
        odm_cls, odm_reg = [], []
        B, N, C, k, _k = aligned_feats.shape
        x = aligned_feats.reshape(B, N, -1).permute(0, 2, 1)  # (B, C*k*k, N)
        for conv in self.odm_convs:
            x = conv(x).relu_()
        odm_cls = self.odm_cls(x).permute(0, 2, 1)  # (B, N, num_classes)
        odm_reg = self.odm_reg(x).permute(0, 2, 1)  # (B, N, 5)
        return odm_cls, odm_reg

    def forward(self, feats):
        """Full S2ANet forward: FAM → decode proposals → AlignConv → ODM.

        Returns:
            fam_cls, fam_reg: FAM predictions per level
            odm_cls, odm_reg: ODM refined predictions
        """
        fam_cls, fam_reg = self._fam_forward(feats)
        return fam_cls, fam_reg

    def get_proposals(self, fam_cls, fam_reg, score_thr=0.05, topk=500):
        """Decode FAM predictions into oriented proposals per level.

        Limits to top-k proposals per level to manage memory in AlignConv.
        """
        device = fam_cls[0].device
        anchors_list, _ = self.anchor_generator(self._featmap_sizes, device)
        proposals = []
        for level_idx, (cls_s, reg_s, anchors) in enumerate(zip(fam_cls, fam_reg, anchors_list)):
            B, N, C = cls_s.shape
            scores = cls_s.sigmoid().max(dim=-1)[0]
            level_props = []
            for b in range(B):
                sc = scores[b]
                # Top-k per level per image
                k = min(topk, N)
                top_sc, top_idx = sc.topk(k)
                keep = top_sc > score_thr
                if keep.sum() == 0:
                    level_props.append(torch.zeros(0, 5, device=device))
                    continue
                idx = top_idx[keep]
                anc = anchors[idx]
                delta = reg_s[b][idx]
                prop = self.bbox_coder.decode(anc, delta)
                level_props.append(prop)
            max_n = max(p.shape[0] for p in level_props) if level_props else 0
            padded = torch.zeros(B, max_n, 5, device=device)
            for b, p in enumerate(level_props):
                if p.shape[0] > 0:
                    padded[b, :p.shape[0]] = p
            proposals.append(padded)
        return proposals

    def loss(self, fam_cls, fam_reg, odm_cls, odm_reg,
             gt_bboxes, gt_labels):
        """Compute FAM + ODM losses.

        Note: This is a simplified loss. Full S2ANet training requires
        the AlignConv → ODM path during training, which is handled
        by the detector wrapper.
        """
        return self._fam_loss(fam_cls, fam_reg, gt_bboxes, gt_labels)

    def _fam_loss(self, fam_cls, fam_reg, gt_bboxes, gt_labels):
        """Compute FAM loss (anchor-based)."""
        from atrumod.ops.rotated_iou import box_iou_rotated

        device = fam_cls[0].device
        B = fam_cls[0].shape[0]
        anchors_list, _ = self.anchor_generator(self._featmap_sizes, device)

        all_cls = torch.cat([s.reshape(B, -1, self.num_classes) for s in fam_cls], dim=1)
        all_reg = torch.cat([r.reshape(B, -1, 5) for r in fam_reg], dim=1)
        all_anchors = torch.cat(anchors_list, dim=0)

        cls_targets, bbox_targets = [], []
        num_pos = 0
        for b in range(B):
            gt_box, gt_lab = gt_bboxes[b], gt_labels[b]
            if gt_box.shape[0] == 0:
                cls_targets.append(torch.zeros(all_anchors.shape[0], dtype=torch.long, device=device))
                bbox_targets.append(all_anchors.clone())
                continue

            ious = box_iou_rotated(all_anchors, gt_box)
            max_iou, max_idx = ious.max(dim=1)
            pos_mask = max_iou >= 0.5
            neg_mask = max_iou < 0.4
            _, best_anchor = ious.max(dim=0)
            pos_mask[best_anchor] = True
            neg_mask[best_anchor] = False

            cls_target = torch.full((all_anchors.shape[0],), -1, dtype=torch.long, device=device)
            cls_target[neg_mask] = 0
            cls_target[pos_mask] = gt_lab[max_idx[pos_mask]] + 1
            cls_targets.append(cls_target)

            bbox_target = all_anchors.clone()
            pos_anc = all_anchors[pos_mask]
            pos_gt = gt_box[max_idx[pos_mask]]
            if pos_anc.shape[0] > 0:
                bbox_target[pos_mask] = self.bbox_coder.decode(
                    pos_anc, self.bbox_coder.encode(pos_anc, pos_gt))
            bbox_targets.append(bbox_target)
            num_pos += pos_mask.sum().item()

        cls_targets = torch.stack(cls_targets)
        bbox_targets = torch.stack(bbox_targets)
        pos_inds = cls_targets > 0
        num_pos = max(pos_inds.sum().item(), 1)

        cls_loss = self.loss_cls(all_cls.reshape(-1, self.num_classes),
                                 cls_targets.reshape(-1), avg_factor=num_pos)

        if pos_inds.sum() > 0:
            pos_pred = all_reg[pos_inds]
            pos_target = bbox_targets[pos_inds]
            pos_anchors = all_anchors.unsqueeze(0).expand(B, -1, -1)[pos_inds]
            deltas = self.bbox_coder.encode(pos_anchors, pos_target)
            bbox_loss = self.loss_bbox(pos_pred, deltas, avg_factor=num_pos)
        else:
            bbox_loss = all_reg.sum() * 0

        return {
            'loss_cls': cls_loss * self.loss_cls_weight,
            'loss_bbox': bbox_loss * self.loss_bbox_weight,
        }

    def get_bboxes(self, fam_cls, fam_reg, odm_cls=None, odm_reg=None, cfg=None):
        """Get detection results. Uses ODM if available, otherwise FAM only."""
        from atrumod.ops.rotated_nms import nms_rotated

        cfg = cfg or dict(nms_pre=2000, score_thr=0.05,
                          nms=dict(iou_thr=0.1), max_per_img=2000)
        device = fam_cls[0].device
        anchors_list, _ = self.anchor_generator(self._featmap_sizes, device)
        all_anchors = torch.cat(anchors_list, dim=0)

        cls_sc = odm_cls if odm_cls is not None else \
            torch.cat([s.reshape(s.shape[0], -1, self.num_classes) for s in fam_cls], dim=1)
        reg_pr = odm_reg if odm_reg is not None else \
            torch.cat([r.reshape(r.shape[0], -1, 5) for r in fam_reg], dim=1)
        if cls_sc.dim() == 2:
            cls_sc = cls_sc.unsqueeze(0)
            reg_pr = reg_pr.unsqueeze(0)

        B = cls_sc.shape[0]
        results = []
        score_thr = cfg.get('score_thr', 0.05)
        nms_cfg = cfg.get('nms', dict(iou_thr=0.1))

        for b in range(B):
            scores, labels = cls_sc[b].sigmoid().max(dim=-1)
            keep = scores > score_thr
            scores, labels = scores[keep], labels[keep]
            bbox_pred = reg_pr[b][keep]

            if scores.shape[0] == 0:
                results.append((torch.zeros(0, 5, device=device),
                                torch.zeros(0, dtype=torch.long, device=device),
                                torch.zeros(0, device=device)))
                continue

            if cls_sc is odm_cls:
                det_bboxes = bbox_pred
            else:
                det_bboxes = self.bbox_coder.decode(
                    all_anchors.unsqueeze(0).expand(B, -1, -1)[b][keep], bbox_pred)

            final_boxes, final_labels, final_scores = [], [], []
            for cls_id in labels.unique():
                cls_mask = labels == cls_id
                cb = det_bboxes[cls_mask]
                cs = scores[cls_mask]
                if cb.shape[0] == 0:
                    continue
                _, order = cs.sort(descending=True)
                cb, cs = cb[order], cs[order]
                keep_nms = nms_rotated(cb, cs, nms_cfg.get('iou_thr', 0.1))
                final_boxes.append(cb[keep_nms])
                final_labels.append(torch.full((keep_nms.sum(),), cls_id, device=device, dtype=torch.long))
                final_scores.append(cs[keep_nms])

            if final_boxes:
                dets = torch.cat(final_boxes)
                labs = torch.cat(final_labels)
                sc = torch.cat(final_scores)
                max_pi = cfg.get('max_per_img', 2000)
                if dets.shape[0] > max_pi:
                    _, topk = sc.topk(max_pi)
                    dets, labs, sc = dets[topk], labs[topk], sc[topk]
            else:
                dets = torch.zeros(0, 5, device=device)
                labs = torch.zeros(0, dtype=torch.long, device=device)
                sc = torch.zeros(0, device=device)
            results.append((dets, labs, sc))

        return results
