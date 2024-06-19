import torch
import torch.nn as nn

from torch import Tensor
from mask import get_mask


class CouplingLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, mask: Tensor):
        super(CouplingLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mask = nn.Parameter(mask, requires_grad=False)

        # Layers for scale computation
        self.scale_conv1 = nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.scale_conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.scale_conv3 = nn.Conv2d(self.hidden_dim, self.in_dim, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.Tensor(self.in_dim, 1, 1))
        nn.init.normal_(self.scale)

        # Layers for translation computation
        self.translation_conv1 = nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.translation_conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.translation_conv3 = nn.Conv2d(self.hidden_dim, self.in_dim, kernel_size=3, padding=1)


    def _compute_scale(self, x: Tensor):
        x_masked = x * self.mask

        s = torch.relu(self.scale_conv1(x_masked))
        s = torch.relu(self.scale_conv2(s))
        s = torch.relu(self.scale_conv3(s)) * self.scale

        return s


    def _compute_translation(self, x: Tensor):
        x_masked = x * self.mask

        t = torch.relu(self.translation_conv1(x_masked))
        t = torch.relu(self.translation_conv2(t))
        t = torch.relu(self.translation_conv3(t))

        return t


    def forward(self, x: Tensor):
        s = self._compute_scale(x)
        t = self._compute_translation(x)

        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask) * s, dim=[1, 2, 3])
    
        return y, logdet
    

    def inverse(self, y: Tensor):
        s = self._compute_scale(y)
        t = self._compute_translation(y)

        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = torch.sum((1 - self.mask) * (-s), dim=[1, 2, 3])

        return x, logdet