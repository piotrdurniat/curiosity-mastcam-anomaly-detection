import torch 
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable


class AnomalyScore():
    def __init__(self, model, loader, device,  alpha=0.5):
        
        self.model = model
        self.alpha = alpha
        self.device = device
        self.loader = loader

    def test(self, type_of_label):

        results_dict = {}

        self.model.eval()


        for index, (x, _) in enumerate(self.loader): 

            x = x.to(self.device)
            z, log_jacobian = self.model.forward(x)

            x_hat, _ = self.model.inverse(z)
            
            log_likelihood = -torch.sum(0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi)))) - log_jacobian
            reconstruction_loss = torch.norm(x - x_hat, p=1, dim=1).mean()

            A = self.alpha * reconstruction_loss + (1 - self.alpha) * log_likelihood.mean()
            results_dict[type_of_label + '_' + str(index)] = A.detach().numpy()
            print(type_of_label + '_' + str(index))

        return results_dict