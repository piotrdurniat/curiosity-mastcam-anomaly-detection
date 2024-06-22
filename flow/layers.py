
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
        masked_weights = self.mask * self.weight

        return F.linear(x, masked_weights, self.bias)


class BatchNormLayer(nn.Module):
    def __init__(self, n_of_features, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, n_of_features))
        self.beta = nn.Parameter(torch.zeros(1, n_of_features))

        self.batch_mean = None
        self.batch_var = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mu = x.mean(dim=0)
            var = x.var(dim=0) + self.eps
            sigma = torch.sqrt(var)

            self.batch_mean = None

        else:
            if self.batch_mean is None:
                self.batch_mean = x.mean(dim=0)
                self.batch_var = x.var(dim=0) + self.eps

            mu = self.batch_mean.clone()
            var = self.batch_var.clone()
            sigma = torch.sqrt(var)

        x_norm = (x - mu) / sigma
        x_norm = x_norm * torch.exp(self.gamma) + self.beta
        log_det_sum = torch.sum(self.gamma - torch.log(sigma))

        return x_norm, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mu = x.mean(dim=0)
            var = x.var(dim=0) + self.eps
            sigma = torch.sqrt(var)
            self.batch_mean = None

        else:
            if self.batch_mean is None:
                self.batch_mean = x.mean(dim=0)
                self.batch_var = x.var(dim=0) + self.eps

            mu = self.batch_mean
            var = self.batch_var
            sigma = torch.sqrt(var)

        x_norm = (x - self.beta) * torch.exp(-self.gamma) * sigma + mu
        log_det = torch.sum(-self.gamma + torch.log(sigma))

        return x_norm, log_det