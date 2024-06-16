import torch.nn as nn
import torch

class GanDiscriminator(nn.Module):

    def __init__(self, ):

        super(GanDiscriminator, self).__init__()

        self.D_X_layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Dropout2d(p=0.5),
        )


        self.D_Z_layers = nn.Sequential(
            nn.Linear(200, 512), 
            nn.LeakyReLU(0.1),
        )

        self.final_layers  = nn.Sequential(
            nn.Linear(16896,1024),
            nn.Linear(1024, 1)
        )

        self.prediction = nn.Sigmoid()

    def forward(self, z_inp, x_inp):
        
        DX = self.D_X_layers(x_inp)
        DZ = self.D_Z_layers(z_inp)

        DX = DX.reshape(-1, 8 * 8 * 256)
        concat = torch.cat((DX, DZ), dim=1)

        result = self.final_layers(concat)
        return result