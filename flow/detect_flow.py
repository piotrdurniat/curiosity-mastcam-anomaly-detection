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
            x = x.view(x.size(0), -1)

            with torch.no_grad():
                nll = self.model.log_prob(x)

            results_dict[type_of_label + '_' + str(index)] = nll.item()
            print(type_of_label + '_' + str(index))

        return results_dict