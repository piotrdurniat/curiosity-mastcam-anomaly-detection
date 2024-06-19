import torch
import torch.nn as nn

class RealNVP(nn.Module):
    def __init__(self):
        raise NotImplementedError
    

    def _get_mask(
        height: int,
        width: int,
        channels: int,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        device: torch.device = None,
    ):
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

        return mask.unsqueeze(0).repeat(channels, 1, 1)