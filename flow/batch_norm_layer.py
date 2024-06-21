import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

        self.batch_mean = None
        self.batch_var = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0) + self.eps
            self.batch_mean = None

        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)

            mean = self.batch_mean.clone()
            var = self.batch_var.clone()

        std = torch.sqrt(var)
        x_hat = (x - mean) / std
        x_hat = x_hat * torch.exp(self.gamma) + self.beta

        log_det_jacobian = torch.sum(self.gamma - 0.5 * torch.log(var))

        return x_hat, log_det_jacobian


    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0) + self.eps
            self.batch_mean = None

        else:
            if self.batch_mean is None:
                self.batch_mean = x.mean(dim=0)
                self.batch_var = x.var(dim=0) + self.eps

            mean = self.batch_mean
            var = self.batch_var

        std = torch.sqrt(var)
        x_hat = (x - self.beta) * torch.exp(-self.gamma) * std + mean

        log_det_jacobian = torch.sum(-self.gamma + 0.5 * torch.log(var))

        return x_hat, log_det_jacobian