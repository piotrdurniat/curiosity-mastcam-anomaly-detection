from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .maf_layer import MAFLayer
from .batch_norm_layer import BatchNormLayer


class MAF(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int],  n_of_layers: int):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range( n_of_layers):
            self.layers.append(MAFLayer(in_dim, hidden_dims))
            self.layers.append(BatchNormLayer(in_dim))


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])

        for layer in self.layers:
            z, log_det = layer(x)
            log_det_sum += log_det

        return z, log_det_sum
    

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(z.shape[0])

        for layer in reversed(self.layers):
            x, log_det = layer.backward(z)
            log_det_sum += log_det

        return x, log_det_sum
