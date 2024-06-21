import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__(in_dim, out_dim, bias)
        self.mask = None

    def _mask_init(self, mask: Tensor):
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        masked_weights = self.mask * self.weight

        return F.linear(x, masked_weights, self.bias)