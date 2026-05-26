"""Pure-PyTorch training loop — replaces mmengine.runner.Runner."""
import sys
import time
import torch
import subprocess
import socket
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from .checkpoint import save_checkpoint, load_checkpoint


def _find_free_port(start=6006):
    """Find a free TCP port."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start


class Trainer:
    """Simple training loop with TensorBoard logging and checkpointing."""

    def __init__(self, model, optimizer, train_loader, val_loader=None,
                 scheduler=None, max_epochs=12, work_dir='logs/checkpoints',
                 log_interval=5, device='cuda', use_amp=False, resume=None,
                 grad_clip=35.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.work_dir = Path(work_dir)
        self.log_interval = log_interval
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        self._tb_process = None

        self.work_dir.mkdir(parents=True, exist_ok=True)
        log_dir = str(self.work_dir / 'tensorboard')
        self.writer = SummaryWriter(log_dir)

        # Auto-launch TensorBoard server
        self._tb_port = _find_free_port()
        try:
            self._tb_process = subprocess.Popen(
                [sys.executable, '-m', 'tensorboard.main', '--logdir', log_dir,
                 '--port', str(self._tb_port), '--bind_all',
                 '--reload_multifile', 'true'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self._tb_process = None

        self.start_epoch = 0
        self.best_loss = float('inf')

        if resume is not None:
            self.start_epoch = load_checkpoint(model, resume, optimizer, scheduler, self.scaler) + 1

    def _to_device(self, batch):
        """Move batch tensors to GPU."""
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def _forward_batch(self, batch):
        """Forward pass handling single, dual, and DMM models."""
        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']
        is_dual = 'rgb' in batch

        # DMM detector: has forward_train for full training pipeline
        if hasattr(self.model, 'forward_train') and is_dual:
            if is_dual:
                return self.model.forward_train(batch['rgb'], batch['ir'], gt_bboxes, gt_labels)
            else:
                return self.model.forward_train(batch['img'], gt_bboxes, gt_labels)

        # Standard detector: forward + head.loss
        if is_dual:
            cls_scores, bbox_preds = self.model(batch['rgb'], batch['ir'])
        else:
            cls_scores, bbox_preds = self.model(batch['img'])
        return self.model.bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)

    def _step(self, batch):
        """Single training step."""
        batch = self._to_device(batch)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            loss_dict = self._forward_batch(batch)
            loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        return loss, loss_dict

    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start = time.time()
        data_start = start

        for i, batch in enumerate(self.train_loader):
            data_time = time.time() - data_start

            loss, loss_dict = self._step(batch)

            if not torch.isfinite(loss):
                print(f'[WARN] NaN/inf loss at iter {i} — cls={loss_dict.get("loss_cls", 0):.2f} bbox={loss_dict.get("loss_bbox", 0):.2f}')
                start = time.time()
                data_start = time.time()
                continue

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            step = epoch * len(self.train_loader) + i

            if i % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                batch_time = time.time() - start
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', lr, step)
                print(f'Epoch [{epoch}/{self.max_epochs}] '
                      f'Iter [{i}/{len(self.train_loader)}] '
                      f'loss: {loss.item():.4f} '
                      f'lr: {lr:.6f} '
                      f'data: {data_time:.3f}s '
                      f'batch: {batch_time:.3f}s')

            start = time.time()
            data_start = time.time()

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation epoch."""
        if self.val_loader is None:
            return float('nan')

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss, _ = self._step(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        print(f'Epoch [{epoch}/{self.max_epochs}] val_loss: {avg_loss:.4f}')
        return avg_loss

    def fit(self):
        """Run full training loop."""
        print(f'Training on {self.device} for {self.max_epochs} epochs')
        print(f'Train batches: {len(self.train_loader)}, '
              f'Val batches: {len(self.val_loader) if self.val_loader else 0}')
        print(f'AMP: {self.use_amp}, Grad clip: {self.grad_clip}')
        if self._tb_process:
            print(f'TensorBoard: http://localhost:{self._tb_port}')

        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.train_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            val_loss = self.validate(epoch)

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            save_path = self.work_dir / f'epoch_{epoch}.pth'
            save_checkpoint(self.model, self.optimizer, self.scheduler,
                          self.scaler, epoch, save_path, best=is_best)

        self.writer.close()
        if self._tb_process is not None:
            self._tb_process.terminate()
            self._tb_process.wait(timeout=5)
        print(f'Training complete. Best val loss: {self.best_loss:.4f}')
