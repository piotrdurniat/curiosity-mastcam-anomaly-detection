import torch
import torch.nn as nn

from torch import Tensor


class CouplingLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, reverse_mask: bool = False):
        super(CouplingLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.reverse_mask = reverse_mask
        self.mask = None

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


    def _create_mask(
        self,
        height: int,
        width: int,
        channels: int,
        reverse: bool = False,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        device: torch.device = None,
    ):
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

        if reverse:
            mask = 1 - mask

        mask = mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)

        return mask


    def forward(self, x: Tensor):
        mask = self._create_mask(x.size(2), x.size(3), x.size(1), self.reverse_mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

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