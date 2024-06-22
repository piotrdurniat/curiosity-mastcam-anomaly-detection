from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .layers import BatchNormLayer
from .maf_layer import MAFLayer


class MAF(nn.Module):
    def __init__(
        self,
        n_of_features: int,
        hidden_dims: List[int], 
        n_layers: int,
        use_reverse: bool = True
    ):
        super().__init__()

        self.n_of_features = n_of_features
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(n_of_features, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(n_of_features))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])

        for layer in self.layers:
            x, log_det_jacobian = layer(x)
            log_det_sum += log_det_jacobian

        return x, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])

        for layer in reversed(self.layers):
            x, log_det_jacobian = layer.backward(x)
            log_det_sum += log_det_jacobian

        return x, log_det_sum