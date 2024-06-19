import torch

class CheckerboardMask():
    def __init__(
        self,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        device: str = None,
    ):
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        self.mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    def get_mask(self) -> torch.Tensor:
        return self.mask