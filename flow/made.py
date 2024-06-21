import torch
import torch.nn as nn
import numpy as np

from typing import List
from torch import Tensor
from torch.nn import ReLU

from .masked_linear import MaskedLinear


class Made(nn.Module):
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
        return self.model(x)


    def _create_masks(self):
        L = len(self.hidden_dims)
        D = self.in_dim

        self.masks[0] = np.random.permutation(D) if self.random_order else np.arange(D)

        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]

            self.masks[l + 1] = np.random.randint(low=low, high=D - 1, size=size)
        self.masks[L + 1] = self.masks[0]

        for i in range(len(self.masks) - 1):
            mask = self.masks[i]
            mask_next = self.masks[i + 1]
            mask_matrix = torch.zeros(len(mask_next), len(mask))

            for j in range(len(mask_next)):
                mask_matrix[j, :] = torch.from_numpy((mask_next[j] >= mask).astype(int))

            self.mask_matrix.append(mask_matrix)

        m = self.mask_matrix.pop(-1)
        self.mask_matrix.append(torch.cat((m, m), dim=0))

        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module._mask_init(next(mask_iter))
