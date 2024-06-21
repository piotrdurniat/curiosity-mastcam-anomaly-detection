import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple

from .made import Made


class MAFLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int]):
        super(MAFLayer, self).__init__()

        self.in_dim = in_dim
        self.made = Made(in_dim, hidden_dims)


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())

        mu, logp = torch.chunk(out, 2, dim=1)
        z = (x - mu) * torch.exp(0.5 * logp)

        log_det_jacobian = 0.5 * torch.sum(logp, dim=1)
        return z, log_det_jacobian


    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.zeros_like(z)

        for dim in range(self.in_dim):
            out = self.made(x)

            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)

            x[:, dim] = mu[:, dim] + z[:, dim] * torch.exp(mod_logp[:, dim])

        log_det_jacobian = torch.sum(mod_logp, axis=1)
        return x, log_det_jacobian
