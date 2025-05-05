
from .BaseModel import BaseModel
import torch
from opacus import PrivacyEngine
from .BaseModel import BaseModel
import torch


import torch
import torch.nn as nn
from torch.optim import Adam
from opacus import PrivacyEngine

class CoreDPModule(nn.Module):
    def __init__(self, user_embedding, item_embedding, dropout, fc):
        super().__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.dropout = nn.Dropout(dropout)
        self.fc = fc

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze()

class DPModel(BaseModel):
    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0, target_delta=1e-5, target_epsilon=1.0, dropout_rate=0.3, enable_dp=True, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.enable_dp = enable_dp
        self.dropout_rate = dropout_rate
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.privacy_engine = PrivacyEngine(accountant='rdp')
        self.dp_model = None  # Will be set in `on_train_start`

    def on_train_start(self):
        if not self.enable_dp:
            return

        train_loader = self.trainer.train_dataloader
        optimizer = self.trainer.optimizers[0]

        # Extract the actual model layers
        self.dp_model = CoreDPModule(self.user_embedding, self.item_embedding, self.dropout_rate, self.fc)

        dp_optimizer = Adam(self.dp_model.parameters(), lr=optimizer.param_groups[0]["lr"])

        # Attach privacy engine
        self.dp_model, dp_optimizer, _ = self.privacy_engine.make_private_with_epsilon(
            module=self.dp_model,
            optimizer=dp_optimizer,
            data_loader=train_loader,
            max_grad_norm=self.max_grad_norm,
            target_delta=self.target_delta,
            target_epsilon=self.target_epsilon,
            epochs=self.trainer.max_epochs
        )

        self.trainer.optimizers = [dp_optimizer]

    def forward(self, user_ids, item_ids):
        if self.enable_dp and self.dp_model is not None:
            return self.dp_model(user_ids, item_ids)
        return super().forward(user_ids, item_ids)

    def on_train_epoch_end(self):
        if self.enable_dp and hasattr(self, 'privacy_engine'):
            epsilon = self.privacy_engine.get_epsilon(self.target_delta)
            self.log("epsilon", epsilon, prog_bar=True)
