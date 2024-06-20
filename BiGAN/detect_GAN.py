import torch 
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class AnomalyScore():
    def __init__(self, generator, encoder, discriminator, loader, device,  alpha=0.5):
        
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator
        self.alpha = alpha
        self.device = device
        self.loader = loader

    def test(self):

        results_list = [] 

        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()


        for x, _ in self.loader: 

            x = x.to(self.device)
            z = self.encoder(x)
            x_hat = self.generator(z)

            L_G = torch.norm(x - x_hat, p=1, dim=1).mean()
            
            dis_output = self.discriminator(x, z)
            
            cross_loss = nn.CrossEntropyLoss()

            y_true = Variable(torch.ones((x.size(0), 1)).to(self.device)) 

            L_D = cross_loss(dis_output, y_true)

            A = self.alpha * L_G + (1 - self.alpha) * L_D
            print(A)
            results_list.append(A)

        return results_list
