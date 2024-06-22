import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

from .coupling_layer import CouplingLayer


class RealNVP(nn.Module):
    def __init__(
    self,
    in_channels: int,
    mid_channels: int,
    n_of_layers: int,
    ):
        super(RealNVP, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.n_of_layers = n_of_layers

        self.couplings = nn.ModuleList(
            [CouplingLayer(in_channels, mid_channels, i % 2 == 1) for i in range(n_of_layers)]
        )


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = x
        log_det_total = 0.0

        for layer in self.couplings:
            out, log_det_jacobian = layer(out)
            log_det_total += log_det_jacobian

        log_det_norm = torch.sum(torch.log(torch.abs(4 * (1 - torch.tanh(out) ** 2))), dim=[1, 2, 3])
        log_det_total += log_det_norm
        y = 4 * torch.tanh(out)
        
        return y, log_det_total
    

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_total = 0.0

        log_det_norm = torch.sum(torch.log(torch.abs(1.0 / 4.0 * 1 / (1 - (y / 4) ** 2))), dim=[1, 2, 3])
        log_det_total += log_det_norm
        x = 0.5 * torch.log((1 + y / 4) / (1 - y / 4))

        for layer in self.couplings:
            x, log_det_jacobian = layer.inverse(x)
            log_det_total += log_det_jacobian

        return x, log_det_total