import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

from typing import List, Tuple

from .layers import MaskedLinear


class MAFLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], reverse: bool):
        super(MAFLayer, self).__init__()

        self.in_dim = in_dim
        self.made = MADE(in_dim, hidden_dims, gaussian=True)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())

        mu, logvar = torch.chunk(out, 2, dim=1)
        z = (x - mu) * torch.exp(0.5 * logvar)
        log_det_sum = 0.5 * torch.sum(logvar, dim=1)

        if self.reverse:
            z = z.flip(dims=(1,))

        return z, log_det_sum

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.reverse:
            z = z.flip(dims=(1,))

        x = torch.zeros_like(z)
        for dim in range(self.in_dim):
            out = self.made(x)

            mu, logvar = torch.chunk(out, 2, dim=1)
            mod_logvar = torch.clamp(-0.5 * logvar, max=10)

            x[:, dim] = mu[:, dim] + z[:, dim] * torch.exp(mod_logvar[:, dim])

        log_det_sum = torch.sum(mod_logvar, axis=1)

        return x, log_det_sum
    

class MADE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = 2 * in_dim if gaussian else in_dim

        self.random_order = random_order
        self.gaussian = gaussian

        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        dim_list = [self.in_dim, *hidden_dims, self.out_dim]

        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
            self.layers.append(ReLU())
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))

        self.model = nn.Sequential(*self.layers)
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        if self.gaussian:
            return self.model(x)
        else:
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        hidden_layers = len(self.hidden_dims)

        if self.random_order:
            self.masks[0] = np.random.permutation(self.in_dim)
        else:
            self.masks[0] = np.arange(self.in_dim)

        for layer_index in range(hidden_layers):
            low = self.masks[layer_index].min()
            size = self.hidden_dims[layer_index]
            self.masks[layer_index + 1] = np.random.randint(low=low, high=(self.in_dim - 1), size=size)

        self.masks[hidden_layers + 1] = self.masks[0]

        for i in range(len(self.masks) - 1):
            mask = self.masks[i]
            next_mask = self.masks[i + 1]
            mask_matrix = torch.zeros(len(next_mask), len(mask))

            for j in range(len(next_mask)):
                mask_matrix[j, :] = torch.from_numpy((next_mask[j] >= mask).astype(int))

            self.mask_matrix.append(mask_matrix)

        if self.gaussian:
            mask = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((mask, mask), dim=0))

        mask_iter = iter(self.mask_matrix)
        for mod in self.model.modules():
            if isinstance(mod, MaskedLinear):
                mod._mask_init(next(mask_iter))