import torch.nn as nn
import torch

class GanDiscriminator(nn.Module):

    def __init__(self, ):

        super(GanDiscriminator, self).__init__()

        self.D_X_layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=False),
        )


        self.D_Z_layers = nn.Sequential(
            nn.Linear(200, 512), 
            nn.LeakyReLU(0.1, inplace=False),
        )

        self.final_layers  = nn.Sequential(
            nn.Linear(16896, 1024),
            nn.Linear(1024, 1),
            )
        self.prediction = nn.Sigmoid()

    def forward(self, x_inp, z_inp):
        
        X_copy = x_inp.clone()
        Z_copy = z_inp.clone()

        DX = self.D_X_layers(X_copy)
        DZ = self.D_Z_layers(Z_copy)

        DX = DX.reshape(-1, 8 * 8 * 256)
        concat = torch.concat((DX, DZ), dim=1)
        result = self.final_layers(concat)
        result = self.prediction(result)
        return result