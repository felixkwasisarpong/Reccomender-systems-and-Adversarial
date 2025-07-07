# File: models/DPFM_GANTrainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class DPFM_GANTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lambda_adv=0.1, enable_dp=True, noise_multiplier=1.0, max_grad_norm=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Validate and wrap model
        self.generator = ModuleValidator.fix(model)
        self.critic = Critic(input_dim=model.embed_dim * 2)  # Assumes concat of user/item

        self.privacy_engine = None
        self.enable_dp = enable_dp
        self.lambda_adv = lambda_adv

        self.mse = nn.MSELoss()

    def forward(self, batch):
        return self.generator(batch)

    def adversarial_loss(self, user_emb, item_emb):
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        critic_score = self.critic(concat_emb)
        return critic_score.mean(), concat_emb

    def training_step(self, batch, batch_idx, optimizer_idx):
        preds = self.generator(batch)
        target = batch["rating"].float().unsqueeze(1)

        user_emb = self.generator.user_embedding(batch["user_id"])
        item_emb = self.generator.item_embedding(batch["item_id"])
        real_concat = torch.cat([user_emb.detach(), item_emb.detach()], dim=1)

        if optimizer_idx == 0:  # Generator step (DP)
            adv_loss, fake_concat = self.adversarial_loss(user_emb, item_emb)
            prediction_loss = self.mse(preds, target)
            total_loss = prediction_loss + self.lambda_adv * -adv_loss

            self.log_dict({"gen_loss": total_loss, "pred_loss": prediction_loss, "adv_loss": -adv_loss})
            return total_loss

        elif optimizer_idx == 1:  # Critic step (non-DP)
            fake_concat = torch.cat([user_emb.detach(), item_emb.detach()], dim=1)
            real_score = self.critic(real_concat)
            fake_score = self.critic(fake_concat)

            critic_loss = -(real_score.mean() - fake_score.mean())
            self.log("critic_loss", critic_loss)
            return critic_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr * 0.1)

        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()
            self.generator, opt_g, _ = self.privacy_engine.make_private_with_epsilon(
                module=self.generator,
                optimizer=opt_g,
                data_loader=self.trainer.datamodule.train_dataloader(),
                target_epsilon=8.0,
                target_delta=1e-5,
                epochs=self.trainer.max_epochs,
                max_grad_norm=self.hparams.max_grad_norm
            )
        return [opt_g, opt_c], []

    def on_train_epoch_end(self):
        if self.enable_dp and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            self.log("epsilon", epsilon)
