
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from .BaseModel import BaseModel
import torch
import torch.nn as nn
from .GaussianAccountant import GaussianAccountant
import numpy as np
class CustomDP_SGD(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        loss_function: str = "RMSE",
        learning_rate: float = 0.001,
        dropout_rate: float = 0.3,
        l2_penalty: float = 1e-4,
        noise_scale: float = 1.0,  # Scale of noise added to gradients
        clip_norm: float = 1.0,  # Max norm for gradient clipping
        delta: float = 1e-5,  # Privacy parameter for DP accounting
        **kwargs
    ):
        super().__init__(
            num_users, num_items, embed_dim, loss_function, learning_rate,
            dropout_rate, l2_penalty, **kwargs
        )
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.delta = delta
        self.privacy_budget = 0  # Tracks accumulated epsilon
        self.automatic_optimization = False  # Required for DP-SGD
    
    def _add_noise(self, grad):
        """Applies Gaussian noise to gradients."""
        noise = torch.randn_like(grad) * self.noise_scale
        return grad + noise

    def _clip_and_noise_gradients(self):
        """Clips gradients and adds noise for differential privacy."""
        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None])
        )
        
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.mul_(clip_coef)  # Clip gradients
        
        # Apply noise after clipping
        for param in self.parameters():
            if param.grad is not None:
                param.grad = self._add_noise(param.grad)

    def _shared_step(self, batch, batch_idx, metrics, prefix):
        """Override shared step to compute loss normally but adjust gradient updates."""
        loss = super()._shared_step(batch, batch_idx, metrics, prefix)
        return loss

    def training_step(self, batch, batch_idx):
        """Perform forward pass, compute loss, and prepare DP-safe gradient update."""
        loss = self._shared_step(batch, batch_idx, self.train_metrics, 'train')

        # Perform gradient clipping and noise addition before optimization step
        self.manual_backward(loss)
        self._clip_and_noise_gradients()
        
        return loss



    def on_train_epoch_end(self):
        """Estimate the privacy budget spent so far using the Gaussian mechanism."""
        sigma = self.noise_scale / self.clip_norm
        epsilon = self._compute_privacy_epsilon(sigma)
        self.privacy_budget += epsilon
        self.log('privacy_epsilon', self.privacy_budget)

    def _compute_privacy_epsilon(self, sigma):
        """Approximates epsilon given sigma and delta using Gaussian DP."""
        return np.sqrt(2 * np.log(1 / self.delta)) / sigma