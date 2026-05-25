"""
Pure-PyTorch active rotated filter — replaces mmcv.ops.active_rotated_filter.

Used by ORConv2d (Oriented Response Convolution) in S2ANet's refine head.
Rotates convolution kernels to match the orientation of each anchor box.
"""
import torch
import torch.nn.functional as F


def active_rotated_filter(input, indices):
    """Rotate input filters according to indices.

    In mmcv, this is a CUDA kernel that rotates filter weights for ORConv2d.
    Our pure-PyTorch version uses grid_sample for the rotation.

    Args:
        input: (O*N, C, H, W) — stacked filters
        indices: (O, N) — orientation indices (0 to num_orientations-1)
    Returns:
        output: (O*N, C, H, W) — rotated filters
    """
    if input.numel() == 0:
        return input

    device = input.device
    O, N = indices.shape  # O=num_orientations, N=num_boxes
    C, H, W = input.shape[1], input.shape[2], input.shape[3]

    # indices maps each box to an orientation, with values 0..O-1
    # The rotation angle for orientation k is 2*pi*k / O
    angles = torch.tensor([2 * 3.14159265 * k / O for k in range(O)],
                          device=device, dtype=input.dtype)

    # Reshape input to (O, N, C, H, W)
    input_reshaped = input.view(O, N, C, H, W)

    # For each orientation, rotate the filter by that angle
    outputs = []
    for o in range(O):
        cos_a = torch.cos(angles[o])
        sin_a = torch.sin(angles[o])
        # Rotation matrix for grid_sample
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                             device=device, dtype=input.dtype)
        # Create affine grid: (N, H, W, 2)
        grid = F.affine_grid(
            theta.unsqueeze(0).expand(N, -1, -1),
            (N, C, H, W), align_corners=False)
        # Grid sample each filter
        rotated = F.grid_sample(input_reshaped[o], grid,
                                align_corners=False, mode='bilinear')
        outputs.append(rotated)

    return torch.cat(outputs, dim=0)


# Also provide the forward/backward stubs that mmcv._ext expects
active_rotated_filter_forward = None
active_rotated_filter_backward = None
