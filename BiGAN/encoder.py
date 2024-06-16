import torch
import torch.nn as nn

class GanEncoder(nn.Module):

    def __init__(self, ):

        super(GanEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 200), 
        )
        
    def forward(self, X):
        return self.layers(X)


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
            nn.Linear(128 * 14 * 14, 512), 
            nn.LeakyReLU(0.1),
        )

        self.final_layers  = nn.Sequential(
            nn.Linear(1024,),
            nn.Linear(1024, 1)
        )

        self.prediction = nn.Sigmoid()

    def forward(self, z_inp, x_inp):
        
        DX = self.D_X_layers(x_inp)
        DZ = self.D_X_layers(z_inp)

        print(DX.shape)
        print(torch.flatten(DX).shape)
        print(DZ.shape)



class GanGenerator():
    
    def __init__(self, ):
        super(GanDiscriminator, self).__init__()



tmp = GanGenerator()
data = torch.rand(64, 200)

