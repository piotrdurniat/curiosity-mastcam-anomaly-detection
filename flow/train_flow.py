import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

from .maf import MAF

from torch import Tensor
from torch.utils.data import DataLoader


class TrainerMAF:
    def __init__(
        self,
        model: MAF,
        epochs: int,
        lr: float,
        train_loader: DataLoader,
        device: torch.device = None
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def plot_loss(self, model_loss):
        epochs = np.arange(1, self.epochs + 1)

        plt.figure(figsize=(12, 6))
    
        plt.plot(epochs, model_loss, label='Strata modelu')
        plt.xlabel('Liczba epok')
        plt.ylabel('Strata')
        plt.title('Strata modelu MAF w zależności od liczby epok')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("/results/Flow_loss.png")


    def train(self):
        self.model.train()

        model_loss = []

        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1}")
            start_time = time.time()

            epoch_loss = 0.0
            for image_batch, _ in self.train_loader:
                x = image_batch.to(self.device)
                x = x.view(x.size(0), -1)

                self.optimizer.zero_grad()

                z, log_det_sum = self.model(x.float())

                loss = self._loss(z, log_det_sum, x.size(1))
                epoch_loss += loss.item()
                loss.backward()

                print(f"Loss: {loss.item()}")

                self.optimizer.step()

            epoch_time = time.time() - start_time
            avg_loss = np.sum(epoch_loss) / len(self.train_loader)

            print(f"Epoch: {epoch + 1} done in {epoch_time:.2f} seconds")
            print(f"Average Loss: {avg_loss:.3f}")

            model_loss.append(avg_loss)

        self.plot_loss(model_loss)

    def _loss(self, z, log_det, n_of_features):
        nll = 0.5 * (z ** 2).sum(dim=1)
        nll += 0.5 * n_of_features * np.log(2 * np.pi)
        nll -= log_det
        nll = torch.mean(nll)

        return nll
