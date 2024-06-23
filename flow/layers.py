
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

from typing import Tuple


class MaskedLinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__(in_dim, out_dim, bias)

        self.mask = None

    def _mask_init(self, mask: Tensor):
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        masked_weights = self.mask * self.weight if self.training else self.weight

        return F.linear(x, masked_weights, self.bias)


class BatchNormLayerWithRunning(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.momentum = 0.01

        self.gamma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)

        self.running_mean = torch.zeros(1, dim)
        self.running_var = torch.ones(1, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mu = x.mean(dim=0)
            var = x.var(dim=0) + self.eps

            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * mu

            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * var

        else:
            mu = self.running_mean
            var = self.running_var

        sigma = torch.sqrt(var)

        x_norm = (x - mu) / torch.sqrt(sigma)
        x_norm = x_norm * torch.exp(self.gamma) + self.beta

        log_det_sum = torch.sum(self.gamma) - torch.sum(torch.log(sigma))
        return x_norm, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:

            mu = x.mean(dim=0)
            var = x.var(dim=0) + self.eps

            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * mu

            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * var

        else:
            mu = self.running_mean
            var = self.running_var

        sigma = torch.sqrt(var)

        x_norm = (x - self.beta) * torch.exp(-self.gamma) * sigma + mu
        log_det_sum = torch.sum(-self.gamma + torch.log(sigma))
        return x_norm, log_det_sum
