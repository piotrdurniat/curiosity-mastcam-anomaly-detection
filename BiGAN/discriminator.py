import torch.nn as nn
import torch

class GanDiscriminator(nn.Module):
    def __init__(self):
        super(GanDiscriminator, self).__init__()

        self.D_X_layers = nn.ModuleList([
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.1),
        ])

        self.D_Z_layers = nn.ModuleList([
            nn.Linear(200, 512),
            nn.LeakyReLU(0.1),
        ])

        self.final_layers = nn.ModuleList([
            nn.Linear(8 * 8 * 256 + 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        ])

        self.prediction = nn.Sigmoid()

    
    def forward(self, x_inp, z_inp):
        for index, layer in enumerate(self.D_X_layers):
            if isinstance(layer, nn.Conv2d):
                if self.training:
                    layer.weight = nn.Parameter(layer.weight.clone())
                    layer.bias = nn.Parameter(layer.bias.clone())
                x_inp = layer(x_inp)
            else:
                x_inp = layer(x_inp)

        for index, layer in enumerate(self.D_Z_layers):
            if isinstance(layer, nn.Linear):
                if self.training:
                    layer.weight = nn.Parameter(layer.weight.clone())
                    layer.bias = nn.Parameter(layer.bias.clone())
                z_inp = layer(z_inp)
            else:
                z_inp = layer(z_inp)

        DX = x_inp.view(-1, 8 * 8 * 256)
        concat = torch.cat((DX, z_inp), dim=1)

        for index, layer in enumerate(self.final_layers):
            if isinstance(layer, nn.Linear):
                if self.training:
                    layer.weight = nn.Parameter(layer.weight.clone())
                    layer.bias = nn.Parameter(layer.bias.clone())
                concat = layer(concat)
            else:
                concat = layer(concat)

        result = self.prediction(concat)
        return result

