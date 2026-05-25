"""Rotated RetinaNet head — self-contained, no mmrotate dependency."""
import torch
import torch.nn as nn
import math
from mmengine.registry import MODELS
from atrumod.models.heads.rotated_anchor_generator import RotatedAnchorGenerator
from atrumod.models.heads.rotated_bbox_coder import DeltaXYWHAOBBoxCoder
from atrumod.ops.rotated_iou import box_iou_rotated


@MODELS.register_module()
class RotatedRetinaHead(nn.Module):
    """Rotated object detection head (RetinaNet-style).

    Produces per-pixel rotated anchor predictions across FPN levels.
    Uses Focal Loss for classification and SmoothL1 for bbox regression.
    """

    def _assert_ok(self, condition, msg):
        if not condition:
            pass  # Silent in production

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stacked_convs=4,
                 feat_channels=256,
                 anchor_generator=None,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels

        # Anchor generator
        if anchor_generator is None:
            anchor_generator = dict(
                type='RotatedAnchorGenerator',
                scales=[4],
                ratios=[1.0],
                angles=[0],
                strides=[8, 16, 32, 64, 128])
        if isinstance(anchor_generator, dict):
            anchor_generator['type'] = 'RotatedAnchorGenerator'  # ensure
            strides = anchor_generator.get('strides', [8, 16, 32, 64, 128])
            scales = anchor_generator.get('scales', [4])
            ratios = anchor_generator.get('ratios', [1.0])
            angles = anchor_generator.get('angles', [0])
            self.anchor_generator = RotatedAnchorGenerator(
                strides=strides, scales=scales, ratios=ratios, angles=angles)
        else:
            self.anchor_generator = anchor_generator

        # Bbox coder
        if bbox_coder is None:
            bbox_coder = dict(type='DeltaXYWHAOBBoxCoder', angle_range='le135')
        if isinstance(bbox_coder, dict):
            self.bbox_coder = DeltaXYWHAOBBoxCoder(**{k: v for k, v in bbox_coder.items() if k != 'type'})
        else:
            self.bbox_coder = bbox_coder

        # Loss configs
        self.loss_cls_cfg = loss_cls or dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        self.loss_bbox_cfg = loss_bbox or dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)

        # Build losses
        from mmdet.models.losses import FocalLoss
        from mmdet.models.losses import SmoothL1Loss
        self.loss_cls = FocalLoss(**{k: v for k, v in self.loss_cls_cfg.items() if k not in ('type', 'loss_weight')})
        self.loss_bbox = SmoothL1Loss(**{k: v for k, v in self.loss_bbox_cfg.items() if k not in ('type', 'loss_weight')})
        self.loss_cls_weight = self.loss_cls_cfg.get('loss_weight', 1.0)
        self.loss_bbox_weight = self.loss_bbox_cfg.get('loss_weight', 1.0)

        # Train/test config
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(
            nms_pre=2000, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000)

        # Build conv layers
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            ch_in = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Conv2d(ch_in, feat_channels, 3, padding=1))
            self.reg_convs.append(
                nn.Conv2d(ch_in, feat_channels, 3, padding=1))

        self.num_anchors = self.anchor_generator.num_anchors
        self.retina_cls = nn.Conv2d(feat_channels, self.num_anchors * num_classes, 3, padding=1)
        self.retina_reg = nn.Conv2d(feat_channels, self.num_anchors * 5, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Classification bias: -log((1-pi)/pi) for focal loss init
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.retina_cls.bias, bias_init)

    def forward(self, feats):
        """Forward pass — produces cls and bbox predictions for each FPN level.

        Args:
            feats: list of (B, C, H_i, W_i)
        Returns:
            cls_scores: list of (B, H_i*W_i*K, num_classes)
            bbox_preds: list of (B, H_i*W_i*K, 5)
        """
        cls_scores = []
        bbox_preds = []
        for feat in feats:
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat).relu_()
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat).relu_()

            cls_out = self.retina_cls(cls_feat)
            reg_out = self.retina_reg(reg_feat)

            B, _, H, W = cls_out.shape
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 5)

            cls_scores.append(cls_out)
            bbox_preds.append(reg_out)

        return cls_scores, bbox_preds

    def get_anchors(self, featmap_sizes, device):
        """Get anchors for all FPN levels.

        Returns:
            anchors: list of (N_i, 5) per level
            valid_flags: list of (N_i,) per level
            num_level_anchors: list of int per level
        """
        return self.anchor_generator(featmap_sizes, device)

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
             img_metas=None, gt_bboxes_ignore=None):
        """Compute detection loss.

        Args:
            cls_scores: list of (B, N_i, num_classes)
            bbox_preds: list of (B, N_i, 5)
            gt_bboxes: list of (G_i, 5) per image
            gt_labels: list of (G_i,) per image
        Returns:
            losses: dict
        """
        featmap_sizes = [(s.shape[1], s.shape[2]) for s in cls_scores]
        device = cls_scores[0].device

        anchors_list, _ = self.get_anchors(featmap_sizes, device)
        B = cls_scores[0].shape[0]

        # Build flat predictions
        all_cls_scores = torch.cat([s.reshape(B, -1, self.num_classes) for s in cls_scores], dim=1)
        all_bbox_preds = torch.cat([p.reshape(B, -1, 5) for p in bbox_preds], dim=1)
        all_anchors = torch.cat(anchors_list, dim=0)  # (total_N, 5)

        # Build targets
        num_pos = 0
        cls_targets = []
        bbox_targets = []
        for b in range(B):
            gt_box = gt_bboxes[b]  # (G, 5)
            gt_lab = gt_labels[b]   # (G,)

            if gt_box.shape[0] == 0:
                cls_targets.append(torch.zeros(all_anchors.shape[0], dtype=torch.long, device=device))
                bbox_targets.append(torch.zeros(all_anchors.shape[0], 5, device=device))
                continue

            # Compute IoU between anchors and GT
            ious = box_iou_rotated(all_anchors, gt_box)  # (N, G)

            # Assign: best GT per anchor, best anchor per GT
            max_iou, max_idx = ious.max(dim=1)  # (N,)

            pos_mask = max_iou >= 0.5
            neg_mask = max_iou < 0.4

            # Ensure each GT has at least one anchor
            _, best_anchor = ious.max(dim=0)  # (G,)
            pos_mask[best_anchor] = True
            neg_mask[best_anchor] = False

            cls_target = torch.full((all_anchors.shape[0],), -1, dtype=torch.long, device=device)
            cls_target[neg_mask] = 0  # background
            cls_target[pos_mask] = gt_lab[max_idx[pos_mask]] + 1  # 1-indexed (0=bg)
            cls_targets.append(cls_target)

            # Bbox target
            bbox_target = all_anchors.clone()
            pos_anchors = all_anchors[pos_mask]
            pos_gts = gt_box[max_idx[pos_mask]]
            if pos_anchors.shape[0] > 0:
                bbox_target[pos_mask] = self.bbox_coder.decode(pos_anchors,
                    self.bbox_coder.encode(pos_anchors, pos_gts))
            bbox_targets.append(bbox_target)
            num_pos += pos_mask.sum().item()

        cls_targets = torch.stack(cls_targets)
        bbox_targets = torch.stack(bbox_targets)

        # Focal loss on classification
        pos_inds = cls_targets > 0
        num_pos = max(pos_inds.sum().item(), 1)

        cls_loss = self.loss_cls(
            all_cls_scores.reshape(-1, self.num_classes),
            cls_targets.reshape(-1),
            avg_factor=num_pos)

        # SmoothL1 on bbox (positives only)
        if pos_inds.sum() > 0:
            pos_bbox_preds = all_bbox_preds[pos_inds]  # (P, 5)
            pos_bbox_targets = bbox_targets[pos_inds]
            # Encode targets back to deltas
            pos_anchors_flat = all_anchors.unsqueeze(0).expand(B, -1, -1)[pos_inds]
            bbox_deltas = self.bbox_coder.encode(pos_anchors_flat, pos_bbox_targets)
            bbox_loss = self.loss_bbox(pos_bbox_preds, bbox_deltas)
        else:
            bbox_loss = all_bbox_preds.sum() * 0

        return {
            'loss_cls': cls_loss * self.loss_cls_weight,
            'loss_bbox': bbox_loss * self.loss_bbox_weight,
        }

    def get_bboxes(self, cls_scores, bbox_preds, img_metas=None,
                   rescale=False, cfg=None):
        """Get detection results with NMS."""
        from atrumod.ops.rotated_nms import nms_rotated

        cfg = cfg or self.test_cfg
        featmap_sizes = [(s.shape[1], s.shape[2]) for s in cls_scores]
        device = cls_scores[0].device

        anchors_list, _ = self.get_anchors(featmap_sizes, device)
        all_anchors = torch.cat(anchors_list, dim=0)

        results = []
        B = cls_scores[0].shape[0]
        for b in range(B):
            cls_score = torch.cat([s[b].reshape(-1, self.num_classes) for s in cls_scores], dim=0)
            bbox_pred = torch.cat([p[b].reshape(-1, 5) for p in bbox_preds], dim=0)

            # Decode
            scores, labels = cls_score.sigmoid().max(dim=-1)

            # Score threshold
            score_thr = cfg.get('score_thr', 0.05)
            keep = scores > score_thr
            scores = scores[keep]
            labels = labels[keep]
            bbox_pred = bbox_pred[keep]
            anchors_keep = all_anchors[keep]

            if scores.shape[0] == 0:
                results.append((torch.zeros(0, 5, device=device),
                                torch.zeros(0, dtype=torch.long, device=device),
                                torch.zeros(0, device=device)))
                continue

            # Decode boxes
            det_bboxes = self.bbox_coder.decode(anchors_keep, bbox_pred)

            # NMS per class
            nms_cfg = cfg.get('nms', dict(iou_thr=0.1))
            final_boxes, final_labels, final_scores = [], [], []
            for cls_id in labels.unique():
                cls_mask = labels == cls_id
                cls_boxes = det_bboxes[cls_mask]
                cls_scores = scores[cls_mask]

                if cls_boxes.shape[0] == 0:
                    continue

                # Sort by score
                _, order = cls_scores.sort(descending=True)
                cls_boxes = cls_boxes[order]
                cls_scores = cls_scores[order]

                keep = nms_rotated(cls_boxes, cls_scores, nms_cfg.get('iou_thr', 0.1))
                final_boxes.append(cls_boxes[keep])
                final_labels.append(torch.full((keep.shape[0],), cls_id, device=device, dtype=torch.long))
                final_scores.append(cls_scores[keep])

            if final_boxes:
                dets = torch.cat(final_boxes)
                labs = torch.cat(final_labels)
                sc = torch.cat(final_scores)

                # Limit max detections
                max_per_img = cfg.get('max_per_img', 2000)
                if dets.shape[0] > max_per_img:
                    _, topk = sc.topk(max_per_img)
                    dets, labs, sc = dets[topk], labs[topk], sc[topk]
            else:
                dets = torch.zeros(0, 5, device=device)
                labs = torch.zeros(0, dtype=torch.long, device=device)
                sc = torch.zeros(0, device=device)

            results.append((dets, labs, sc))

        return results
