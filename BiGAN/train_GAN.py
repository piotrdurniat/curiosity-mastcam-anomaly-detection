import torch
import torch.nn as nn
import torch.optim as optim
from . import encoder, discriminator, generator, weights
from tqdm import  tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np

class TrainerBiGAN:

    def __init__(self, epoch_number, lr, train_loader, val_loader, device):

        self.epoch_number = epoch_number
        self.lr = lr
        self.train_loader = train_loader
        self.device = device
        self.val_loader = val_loader

    def plot_loss(self, enc_gen_loss, dis_loss):
        
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epoch_number+1), enc_gen_loss, label='Strata koder-generator')
        plt.xlabel('Liczba epok')
        plt.ylabel('Strata')
        plt.title('Strata koder-generator w zależności od liczby epok')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.epoch_number+1), dis_loss, label='Strata dyskryminator')
        plt.xlabel('Liczba epok')
        plt.ylabel('Strata')
        plt.title('Strata dyskryminator w zależności od liczby epok')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("./results/BiGAN_loss.png")

    
    
    def plot_images(self, epoch, num_images=8):
        self.Encoder.eval()
        self.Generator.eval()

        images, reconstructions = [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                z = self.Encoder(x)
                x_hat = self.Generator(z)
                images.append(x.cpu())
                reconstructions.append(x_hat.cpu())
                if len(images) * x.size(0) >= num_images:
                    break

        images = torch.cat(images)[:num_images]
        reconstructions = torch.cat(reconstructions)[:num_images]

        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        
        for i in range(num_images):

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
        plt.savefig("./results/val_bigan" + str(epoch) + " .png")

        self.Encoder.train()
        self.Generator.train()



    def train(self):

        
        self.Encoder = encoder.GanEncoder().to(self.device)
        self.Discriminator = discriminator.GanDiscriminator().to(self.device)
        self.Generator = generator.GanGenerator().to(self.device)

        self.Generator.apply(weights.init_weights)
        self.Encoder.apply(weights.init_weights)
        self.Discriminator.apply(weights.init_weights)

        optimizer_ge = optim.Adam(list(self.Generator.parameters()) + list(self.Encoder.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=1e-5)
        optimizer_d = optim.Adam(self.Discriminator.parameters(), lr=self.lr,  betas=(0.5, 0.999), weight_decay=1e-5)

        criterion = nn.BCELoss()
        loss_enc_gen = []
        loss_dis = []


        for epoch in tqdm(range(self.epoch_number)):

            print(f"Starting epoch {epoch + 1}")

            ge_losses = 0
            d_losses = 0

            for image, _ in self.train_loader:


                optimizer_d.zero_grad()
                optimizer_ge.zero_grad()
            
                y_true = Variable(torch.ones((image.size(0), 1)).to(self.device)) 
                y_fake = Variable(torch.zeros((image.size(0), 1)).to(self.device))

                z_fake = Variable(torch.randn((image.size(0), 800)).to(self.device), requires_grad=False)
                x_fake = self.Generator(z_fake)

                x_true = image.float().to(self.device)
                z_true = self.Encoder(x_true)

                out_true = self.Discriminator(x_true, z_true)
                out_fake = self.Discriminator(x_fake, z_fake)
            
                loss_d = criterion(out_true, y_true) + criterion(out_fake, y_fake)
                loss_ge = criterion(out_fake, y_true) + criterion(out_true, y_fake)

                loss_d.backward(retain_graph=True)
                optimizer_d.step()

                loss_ge.backward()
                optimizer_ge.step()
                
                ge_losses += loss_ge.item()
                d_losses += loss_d.item()

            print("Training... Epoch: {}, Discrimiantor Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch+1, d_losses/len(self.train_loader), ge_losses/len(self.train_loader)))

            loss_enc_gen.append(ge_losses/len(self.train_loader))
            loss_dis.append(d_losses/len(self.train_loader))

            if epoch%10 ==  0: 
                self.plot_images(epoch)

        print(loss_enc_gen)
        print(loss_dis)

        self.plot_loss(loss_enc_gen, loss_dis)

        return self.Encoder, self.Generator, self.Discriminator