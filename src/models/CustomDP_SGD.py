
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from .BaseModel import BaseModel
import torch
import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn as nn
import math

class CustomDP_SGD(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        loss_function: str = "RMSE",
        dropout_rate: float = 0.3,
        l2_penalty: float = 1e-4,
        noise_scale: float = 1.0,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
        batch_size: int = 512,
        dataset_size: int = 480189 * 17770,
        **kwargs
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            learning_rate=learning_rate,
            loss_function=loss_function,
            dropout_rate=dropout_rate,
            l2_penalty=l2_penalty,
            **kwargs
        )
        self.noise_multiplier = noise_scale
        self.clip_norm = clip_norm
        self.delta = delta
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sample_rate = batch_size / dataset_size
        self.steps = 0
        self.privacy_budget = 0
        self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        users, items, ratings = batch
        batch_size = users.size(0)

        # Enable individual loss outputs
        predictions = self(users, items)
        criterion = nn.MSELoss(reduction='none')
        losses = criterion(predictions, ratings)

        # Collect per-sample gradients
        per_sample_grads = []
        for i in range(batch_size):
            self.zero_grad()
            losses[i].backward(retain_graph=True)
            grads = [p.grad.detach().clone() if p.grad is not None else None for p in self.parameters()]
            per_sample_grads.append(grads)

        # Clip gradients
        clipped_grads = []
        for grads in per_sample_grads:
            total_norm = torch.sqrt(sum([(g.norm(2) ** 2) for g in grads if g is not None]))
            clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
            clipped = [g * clip_coef if g is not None else None for g in grads]
            clipped_grads.append(clipped)

        # Aggregate and add noise
        final_grads = []
        for param_i in range(len(clipped_grads[0])):
            stacked = torch.stack([
                grads[param_i] for grads in clipped_grads if grads[param_i] is not None
            ])
            mean_grad = stacked.mean(dim=0)
            noise = torch.randn_like(mean_grad) * self.noise_multiplier * self.clip_norm / batch_size
            final_grads.append(mean_grad + noise)

        # Set final noisy gradients
        for p, g in zip(self.parameters(), final_grads):
            if p.requires_grad:
                p.grad = g

        # Step optimizer
        self.optimizers().step()
        self.optimizers().zero_grad()

        # Update privacy accounting
        self.steps += 1
        self._update_privacy_budget()

        return losses.mean()

    def _update_privacy_budget(self):
        if self.noise_multiplier == 0:
            self.privacy_budget = float('inf')
            return

        q = self.sample_rate
        sigma = self.noise_multiplier

        def compute_log_moment(alpha):
            return alpha * (2 * q**2 * sigma**2) / (2 * sigma**2)

        eps_min = float('inf')
        for alpha in self.alphas:
            log_moment = compute_log_moment(alpha)
            eps = (log_moment + math.log(1 / self.delta)) / (alpha - 1)
            if eps < eps_min:
                eps_min = eps

        self.privacy_budget = eps_min
        self.log('privacy_epsilon', self.privacy_budget)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.log('final_privacy_epsilon', self.privacy_budget)
