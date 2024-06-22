import torch
import torch.nn as nn

class GanEncoder(nn.Module):

    def __init__(self, ):

        super(GanEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1, inplace=False),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2,  padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 200), 
        )
        
    def forward(self, X):
        X_copy = X.clone()
        return self.layers(X_copy)


# model = GanGenerator()
# z_inp = torch.randn(10, 200)  # Example batch of 10 samples with 200-dimensional noise input
# output = model(z_inp)


# model = GanDiscriminator()
# model(z_inp, output)