import torch
import torch.nn as nn
import torch.optim as optim
import time

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
                y, log_det_jacobian = self.model(x)

                # Loss
                nll = self._loss(y, log_det_jacobian)

                # Backward pass
                nll.backward()
                self.optimizer.step()

                epoch_loss += nll.item()
                print(nll.item())

            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch + 1} done in {epoch_time:.2f} seconds")
            print(f"Loss: {epoch_loss / len(self.train_loader):.3f}")


    def _loss(self, y: Tensor, log_det: Tensor):
        log_likelihood = torch.sum(-0.5 * (y ** 2 + torch.log(2 * torch.pi * torch.ones_like(y))), dim=[1, 2, 3])
        loss = - (log_likelihood + log_det).mean()

        return loss