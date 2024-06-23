import torch 
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np

class AnomalyScore():
    def __init__(self, generator, encoder, discriminator, loader, device,  alpha=0.5):
        
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator
        self.alpha = alpha
        self.device = device
        self.loader = loader

    def test(self, type_of_label):

        results_dict = {}

        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()


        for index, (x, _) in enumerate(self.loader): 

            x = x.to(self.device)
            z = self.encoder(x)
            x_hat = self.generator(z)

            L_G = torch.norm(x - x_hat, p=1, dim=1).mean()
            
            dis_output = self.discriminator(x, z)
            
            cross_loss = nn.CrossEntropyLoss()

            y_true = Variable(torch.ones((x.size(0), 1)).to(self.device)) 

            L_D = cross_loss(dis_output, y_true)
            A = self.alpha * L_G + (1 - self.alpha) * L_D
            results_dict[type_of_label + '_' + str(index)] = A.detach().numpy()
            print(type_of_label + '_' + str(index))

        return results_dict

    def plot_images(self, name, num_images=8):
        self.encoder.eval()
        self.generator.eval()

        images, reconstructions = [], []

        with torch.no_grad():
            for x, _ in self.loader:
                x = x.to(self.device)
                z = self.encoder(x)
                x_hat = self.generator(z)
                images.append(x.cpu())
                reconstructions.append(x_hat.cpu())
                if len(images) * x.size(0) >= num_images:
                    break

        images = torch.cat(images)[:num_images]
        reconstructions = torch.cat(reconstructions)[:num_images]

        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        
        for i in range(num_images):
            print(reconstructions.shape)
            
            # Normalize each image individually
            norm_image = cv2.normalize(images[i].numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image = norm_image.astype(np.uint8)
            norm_rec = cv2.normalize(reconstructions[i].numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_rec = norm_rec.astype(np.uint8)

            axes[0, i].imshow(norm_image[:3].transpose(1, 2, 0).squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(norm_rec[:3].transpose(1, 2, 0).squeeze(), cmap='gray')
            axes[1, i].axis('off')

            
        axes[0, 0].set_ylabel('Original')
        axes[1, 0].set_ylabel('Generated')
        plt.savefig("./results/bigan" + name +".png")
