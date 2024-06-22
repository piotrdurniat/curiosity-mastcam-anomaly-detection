import torch
import torch.nn as nn
import torch.optim as optim
from . import encoder, discriminator, generator, weights
from tqdm import  tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt


class TrainerBiGAN:

    def __init__(self, epoch_number, lr, train_loader, device):

        self.epoch_number = epoch_number
        self.lr = lr
        self.train_loader = train_loader
        self.device = device


    def plot_loss(self, enc_gen_loss, dis_loss):
        
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch_number, enc_gen_loss, label='Strata koder-generator')
        plt.xlabel('Liczba epok')
        plt.ylabel('Strata')
        plt.title('Strata koder-generator w zależności od liczby epok')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_number, dis_loss, label='Strata dyskryminator')
        plt.xlabel('Liczba epok')
        plt.ylabel('Strata')
        plt.title('Strata dyskryminator w zależności od liczby epok')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("/results/BiGAN_loss.png")

    def train(self):

        
        self.Encoder = encoder.GanEncoder().to(self.device)
        self.Discriminator = discriminator.GanDiscriminator().to(self.device)
        self.Generator = generator.GanGenerator().to(self.device)

        self.Generator.apply(weights.init_weights)
        self.Encoder.apply(weights.init_weights)
        self.Discriminator.apply(weights.init_weights)

        optimizer_ge = optim.Adam(list(self.Generator.parameters()) + list(self.Encoder.parameters()), lr=self.lr)
        optimizer_d = optim.Adam(self.Discriminator.parameters(), lr=self.lr)

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

                z_fake = Variable(torch.randn((image.size(0), 200)).to(self.device), requires_grad=False)
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

        self.plot_loss(loss_enc_gen, loss_dis)

        return self.Encoder, self.Generator, self.Discriminator