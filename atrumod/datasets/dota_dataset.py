"""Pure PyTorch DOTA dataset — no mmdet dependency."""
import glob
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DOTADataset(Dataset):
    """DOTA-format rotated object detection dataset.

    Args:
        data_root: root data directory (e.g. 'data/')
        ann_dir: relative path to DOTA label directory (e.g. 'train/dota_labels/')
        img_dir: relative path to image directory (e.g. 'train/images/')
        classes: tuple of class names
        training: if True, apply augmentations
    """
    CLASSES = ('car', 'suv', 'van', 'bus', 'freight_car', 'truck',
               'motorcycle', 'trailer', 'tank_truck', 'excavator', 'crane')

    def __init__(self, data_root, ann_dir, img_dir, classes=None,
                 training=True):
        self.data_root = Path(data_root)
        self.ann_dir = self.data_root / ann_dir
        self.img_dir = self.data_root / img_dir
        self.classes = classes or self.CLASSES
        self.training = training
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        txt_files = sorted(glob.glob(str(self.ann_dir / '*.txt')))
        self.samples = [osp.splitext(osp.basename(f))[0] for f in txt_files]

    def __len__(self):
        return len(self.samples)

    def _parse_dota(self, txt_path):
        """Parse DOTA label file into list of (cx,cy,w,h,angle_deg, class_idx)."""
        boxes = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    poly = np.array([float(p) for p in parts[:8]], dtype=np.float32)
                    cls_name = parts[8]
                    if cls_name not in self.class_to_idx:
                        continue
                    cx, cy, w, h, angle = DOTADataset._poly2rbox(poly)
                    boxes.append([cx, cy, w, h, angle])
                    labels.append(self.class_to_idx[cls_name])
                except (ValueError, IndexError):
                    continue
        return boxes, labels

    @staticmethod
    def _poly2rbox(poly):
        """4-point polygon → (cx, cy, w, h, angle_deg) via cv2.minAreaRect."""
        rect = cv2.minAreaRect(poly.reshape(4, 2).astype(np.float32))
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        if w < h:
            w, h = h, w
            angle += 90
        angle = (angle + 90) % 180 - 90
        return cx, cy, w, h, angle

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        img_path = self.img_dir / f'{sample_id}.jpg'
        txt_path = self.ann_dir / f'{sample_id}.txt'

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        # Parse labels
        boxes, labels = self._parse_dota(txt_path)

        # Normalize and convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        img = img / 255.0  # to [0, 1]
        img = (img - 0.5) / 0.5  # to [-1, 1]

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        return {
            'img': img,
            'gt_bboxes': boxes_t,
            'gt_labels': labels_t,
            'img_id': sample_id,
        }


class DualInputDataset(Dataset):
    """DOTA dataset that loads paired RGB + IR images for multimodal detection.

    Both modalities share the same label annotations (RGB labels used as ground truth).
    """

    def __init__(self, data_root, ann_dir, img_dir_rgb, img_dir_ir,
                 classes=None, training=True):
        self.data_root = Path(data_root)
        self.ann_dir = self.data_root / ann_dir
        self.img_dir_rgb = self.data_root / img_dir_rgb
        self.img_dir_ir = self.data_root / img_dir_ir
        self.classes = classes or DOTADataset.CLASSES
        self.training = training
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        txt_files = sorted(glob.glob(str(self.ann_dir / '*.txt')))
        self.samples = [osp.splitext(osp.basename(f))[0] for f in txt_files]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_img(path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        return (img - 0.5) / 0.5  # [-1, 1]

    def __getitem__(self, idx):
        sid = self.samples[idx]
        rgb = self._load_img(self.img_dir_rgb / f'{sid}.jpg')
        ir = self._load_img(self.img_dir_ir / f'{sid}.jpg')
        boxes, labels = DOTADataset._parse_dota(self, self.ann_dir / f'{sid}.txt')
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        return {'rgb': rgb, 'ir': ir, 'gt_bboxes': boxes_t, 'gt_labels': labels_t, 'img_id': sid}


def collate_fn(batch):
    """Custom collate — pads images to batch-max size, handles variable shapes."""
    max_h = max(b['img'].shape[1] for b in batch)
    max_w = max(b['img'].shape[2] for b in batch)
    imgs = []
    for b in batch:
        img = b['img']
        h, w = img.shape[1], img.shape[2]
        if h < max_h or w < max_w:
            padded = img.new_zeros(img.shape[0], max_h, max_w)
            padded[:, :h, :w] = img
            img = padded
        imgs.append(img)
    return {
        'img': torch.stack(imgs),
        'gt_bboxes': [b['gt_bboxes'] for b in batch],
        'gt_labels': [b['gt_labels'] for b in batch],
        'img_id': [b['img_id'] for b in batch],
    }


def dual_collate_fn(batch):
    """Collate for DualInputDataset — pads both RGB and IR images."""
    def _pad_images(key):
        max_h = max(b[key].shape[1] for b in batch)
        max_w = max(b[key].shape[2] for b in batch)
        imgs = []
        for b in batch:
            img = b[key]
            h, w = img.shape[1], img.shape[2]
            if h < max_h or w < max_w:
                padded = img.new_zeros(img.shape[0], max_h, max_w)
                padded[:, :h, :w] = img
                img = padded
            imgs.append(img)
        return torch.stack(imgs)

    return {
        'rgb': _pad_images('rgb'),
        'ir': _pad_images('ir'),
        'gt_bboxes': [b['gt_bboxes'] for b in batch],
        'gt_labels': [b['gt_labels'] for b in batch],
        'img_id': [b['img_id'] for b in batch],
    }
