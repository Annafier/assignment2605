"""Save/load model checkpoints."""
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, scheduler, epoch, path, best=False):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, path)
    if best:
        best_path = path.parent / 'best.pth'
        torch.save(ckpt, best_path)
        return best_path
    return path


def load_checkpoint(model, path, optimizer=None, scheduler=None):
    """Load training checkpoint. Returns epoch number."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)
