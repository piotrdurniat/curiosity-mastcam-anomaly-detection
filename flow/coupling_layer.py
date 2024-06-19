import torch
import torch.nn as nn
from torch import Tensor


class CouplingLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, mask: Tensor):
        super(CouplingLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mask = nn.Parameter(mask, requires_grad=False)

        # Layers for scale computation
        self.scale_linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.scale_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_linear3 = nn.Linear(self.hidden_dim, self.in_dim)
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        nn.init.normal_(self.scale)

        # Layers for translation computation
        self.translation_linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.translation_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_linear3 = nn.Linear(self.hidden_dim, self.in_dim)


        def _compute_scale(self, x: Tensor):
            x_masked = x * self.mask

            s = torch.relu(self.scale_linear1(x_masked))
            s = torch.relu(self.scale_linear2(s))
            s = torch.relu(self.scale_linear3(s)) * self.scale

            return s


        def _compute_translation(self, x: Tensor):
            x_masked = x * self.mask

            t = torch.relu(self.translation_linear1(x_masked))
            t = torch.relu(self.translation_linear2(t))
            t = torch.relu(self.translation_linear3(t))

            return t
        
        def forward(self, x: Tensor):
            s = self._compute_scale(x)
            t = self._compute_translation(x)

            y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
            logdet = torch.sum((1 - self.mask) * s, -1)

            return y, logdet
    
        def inverse(self, y: Tensor):
            s = self._compute_scale(y)
            t = self._compute_translation(y)

            x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
            logdet = torch.sum((1 - sum.mask) * (-s), -1)

            return x, logdet

