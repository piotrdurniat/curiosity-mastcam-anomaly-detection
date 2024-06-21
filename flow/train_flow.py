import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from .real_nvp import RealNVP

from torch import Tensor
from torch.utils.data import DataLoader



class TrainerRealNVP:
    def __init__(
        self,
        model: RealNVP,
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

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1}")
            start_time = time.time()

            epoch_loss = 0.0
            for image, _ in self.train_loader:
                x = image.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                y, log_det_sum = self.model(x)

                # Loss
                loss = self._loss(y, log_det_sum)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                print(loss.item())

            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch + 1} done in {epoch_time:.2f} seconds")
            print(f"Loss: {epoch_loss / len(self.train_loader):.3f}")


    def _loss(self, y: Tensor, log_det_sum: Tensor):
        prior = torch.distributions.normal.Normal(0.0, 1.0)

        log_pz = prior.log_prob(y).sum(dim=[1, 2, 3])
        log_px = log_det_sum + log_pz
        nll = -log_px.mean()

        return nll