import torch
import math
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from .BaseModel import BaseModel
class CustomDP_SGD(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        loss_function: str = "RMSE",
        l2_penalty: float = 1e-4,
        dropout_rate: float = 0.2,
        noise_scale: float = 0.0,      # DP parameter
        clip_norm: float = 1.0,        # DP parameter
        delta: float = 1e-5,           # DP parameter
        log_freq: int = 50,
        enable_dp=True
                    # Logging frequency
    ):
        """
        Differentially Private Recommendation Model
        
        Args:
            noise_scale: Noise multiplier for DP (σ)
            clip_norm: Gradient clipping norm (C)
            delta: Privacy parameter (δ)
            log_freq: Frequency of WandB logging (in batches)
        """
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            learning_rate=learning_rate,
            loss_function=loss_function,
            l2_penalty=l2_penalty,
            dropout_rate=dropout_rate,
            enable_dp=False  # Indicate we're using DP
        )
        
        # DP-specific parameters
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.delta = delta
        self.privacy_steps = 0
        self.sample_rate = None
        self.log_freq = log_freq
        self.automatic_optimization = False
        self.epsilon_history = []
        self.metric_history = []  # AUC or RMSE depending on your use

    def on_train_start(self):
        """Initialize sample rate when training begins"""
        if self.trainer.train_dataloader is not None:
            batch_size = self.trainer.train_dataloader.batch_size
            self.sample_rate = batch_size / self.hparams.num_users
            self.log("sample_rate", self.sample_rate)
    
    def configure_dataloader(self, batch_size: int):
        """Set the sample rate for DP accounting"""
        self.sample_rate = batch_size / self.hparams.num_users

    
    def training_step(self, batch, batch_idx):
        # Forward pass and loss calculation
        user_ids, item_ids, targets = batch
        targets_norm = (targets - 1.0) / 4.0
        y_pred = self(user_ids, item_ids)
        loss = self.loss_fn(y_pred, targets_norm).mean()
        
        # Backward pass
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        
        # DP modifications
        if self.noise_scale > 0:
            # 1. Clip gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
            
            # 2. Add noise
            for param in self.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_scale * self.clip_norm
                    param.grad += noise
            
            # 3. Update privacy accounting
            self.privacy_steps += 1
            current_epsilon = self._compute_epsilon()
            
            self.log("train/epsilon", current_epsilon, prog_bar=True)
        
        # Update parameters
        self.optimizers().step()
        
        # Update metrics
        preds_denorm = torch.clamp(y_pred * 4.0 + 1.0, 1.0, 5.0)
        self._update_metrics(preds_denorm, targets, "train")
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def _compute_epsilon(self):
        """Compute epsilon using moments accountant"""
        if self.noise_scale == 0 or self.sample_rate is None:
            return float('inf')
            
        q = self.sample_rate
        sigma = self.noise_scale
        T = self.privacy_steps
        
        # Using analytic moments accountant
        alpha = 1 + 1 / (sigma**2)
        epsilon = (alpha * T * q**2 / (2 * sigma**2)) + (
            math.log(1/self.delta) / (alpha - 1))
        
        return epsilon
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics"""
        super().on_train_epoch_end()  # Call parent's metric logging
        if self.noise_scale > 0:
            epsilon = self._compute_epsilon()
            val_rmse = self.trainer.callback_metrics.get("val_rmse")  # or "val_rmse"
            if val_rmse is not None:
                self.metric_history.append(val_rmse.item())

            self.log("epsilon", epsilon, prog_bar=True)