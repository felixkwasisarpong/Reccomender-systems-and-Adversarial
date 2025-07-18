# File: models/DPFM_GANTrainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torchmetrics import MeanSquaredError, MeanAbsoluteError


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
    def __init__(
        self,
        model,
        lr=1e-3,
        lambda_adv=0.1,
        enable_dp=True,
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        loss_type="mse"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.generator = ModuleValidator.fix(model)
        self.critic = Critic(input_dim=model.embed_dim * 2)
        self.privacy_engine = None
        self.enable_dp = enable_dp
        self.lambda_adv = lambda_adv
        self.loss_type = loss_type

        self.loss_fn = nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss()

        for phase in ["train", "val", "test"]:
            setattr(self, f"{phase}_mse", MeanSquaredError())
            setattr(self, f"{phase}_mae", MeanAbsoluteError())
            setattr(self, f"{phase}_rmse", MeanSquaredError(squared=False))

    def forward(self, batch):
        return self.generator(batch)

    def adversarial_loss(self, user_emb, item_emb):
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        critic_score = self.critic(concat_emb)
        return critic_score.mean(), concat_emb

    def _shared_step(self, batch, batch_idx, phase):
        preds = self.generator(batch)
        targets = batch["rating"].float()
        loss = self.loss_fn(preds.squeeze(), targets)
        self.log(f"{phase}_loss", loss, prog_bar=True)
        getattr(self, f"{phase}_mse").update(preds, targets)
        getattr(self, f"{phase}_mae").update(preds, targets)
        getattr(self, f"{phase}_rmse").update(preds, targets)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        user_emb = self.generator.user_embedding(batch["user_id"])
        item_emb = self.generator.item_embedding(batch["item_id"])

        if optimizer_idx == 0:  # Generator step (DP)
            adv_loss, _ = self.adversarial_loss(user_emb, item_emb)
            preds = self.generator(batch)
            target = batch["rating"].float().unsqueeze(1)
            prediction_loss = self.loss_fn(preds, target)
            total_loss = prediction_loss + self.lambda_adv * -adv_loss

            self.log_dict({"gen_loss": total_loss, "pred_loss": prediction_loss, "adv_loss": -adv_loss})
            return total_loss

        elif optimizer_idx == 1:  # Critic step (non-DP)
            with torch.no_grad():
                user_emb = user_emb.detach()
                item_emb = item_emb.detach()
            real_concat = torch.cat([user_emb, item_emb], dim=1)
            fake_concat = torch.cat([user_emb, item_emb], dim=1)
            real_score = self.critic(real_concat)
            fake_score = self.critic(fake_concat)

            critic_loss = -(real_score.mean() - fake_score.mean())
            self.log("critic_loss", critic_loss)
            return critic_loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_mse.compute(), prog_bar=True)
        self.log("train_mae", self.train_mae.compute(), prog_bar=True)
        self.log("train_rmse", self.train_rmse.compute(), prog_bar=True)
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_rmse.reset()

        if self.enable_dp and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=self.hparams.target_delta)
            self.log("epsilon", epsilon)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_rmse", self.val_rmse.compute(), prog_bar=True)
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_rmse.reset()

    def on_test_epoch_end(self):
        self.log("test_mse", self.test_mse.compute(), prog_bar=True)
        self.log("test_mae", self.test_mae.compute(), prog_bar=True)
        self.log("test_rmse", self.test_rmse.compute(), prog_bar=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_rmse.reset()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr * 0.1)

        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()
            self.generator, opt_g, _ = self.privacy_engine.make_private_with_epsilon(
                module=self.generator,
                optimizer=opt_g,
                data_loader=self.trainer.datamodule.train_dataloader(),
                target_epsilon=self.hparams.target_epsilon,
                target_delta=self.hparams.target_delta,
                epochs=self.trainer.max_epochs,
                max_grad_norm=self.hparams.max_grad_norm
            )

        return [opt_g, opt_c], []
