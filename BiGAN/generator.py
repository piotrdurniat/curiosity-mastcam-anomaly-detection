import torch.nn as nn
import torch


class GanGenerator(nn.Module):

    def __init__(self, ):

        super(GanGenerator, self).__init__()

        self.layers_fc = nn.Sequential(
            nn.Linear(800, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU( inplace=False),

            nn.Linear(1024, 16*16*128),
            nn.BatchNorm1d(16*16*128),
            nn.ReLU(inplace=False),)
        
        self.layers_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 6, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z_inp):
        Z_copy = z_inp.clone()
        z =  self.layers_fc(Z_copy)
        z = z.reshape(-1, 128, 16, 16)
        z = self.layers_conv(z)

        return z