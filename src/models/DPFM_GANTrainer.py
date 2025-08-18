try:
    from opacus.accountants import RDPAccountant
    _HAS_RDP_ACC = True
except Exception:
    RDPAccountant = None
    _HAS_RDP_ACC = False
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
class CustomSGD(torch.optim.Optimizer):
    """Minimal custom SGD optimizer with optional momentum, weight decay, gradient clipping, and DP noise."""
    def __init__(self, params, lr=required, momentum=0, weight_decay=0, noise_multiplier=1.0, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            noise_multiplier = group.get('noise_multiplier', 1.0)
            max_grad_norm = group.get('max_grad_norm', 1.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # Weight decay
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                # Gradient clipping (per-parameter)
                grad_norm = d_p.norm(2)
                if grad_norm > max_grad_norm:
                    d_p = d_p.mul(max_grad_norm / (grad_norm + 1e-6))
                # Add Gaussian noise for DP
                noise_std = noise_multiplier * max_grad_norm
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=d_p.shape,
                    device=d_p.device,
                )
                d_p = d_p + noise
                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(-lr, d_p)
        return loss
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import pytorch_lightning as pl  # BaseModel inherits from this

# --- Import BaseModel (supports package or flat layouts) ---
try:
    from .BaseModel import BaseModel
except Exception:
    try:
        from BaseModel import BaseModel
    except Exception:
        raise ImportError("DPFM_GANTrainer requires BaseModel to be importable (from BaseModel or .BaseModel).")

import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict



# --- Minimal multi-attribute adversary: shared trunk with heads ---

class MultiAttrAdversary(nn.Module):
    def __init__(self, in_dim: int, num_genders: int, num_occupations: int, genre_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        trunk = []
        prev = in_dim
        for h in hidden_dims:
            trunk += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        self.trunk = nn.Sequential(*trunk)
        
        self.gender_head = nn.Linear(prev, num_genders)
        self.occupation_head = nn.Linear(prev, num_occupations)
        self.genre_head = nn.Linear(prev, genre_dim)  # BCEWithLogits for multi-label
        self.age_head = nn.Linear(prev, 1)  # Regression for age

    def forward(self, x):
        h = self.trunk(x)
        return {
            'gender': self.gender_head(h),
            'occupation': self.occupation_head(h),
            'genre': self.genre_head(h),
            'age': self.age_head(h),
        }




# --- DP + Adversarial GAN Trainer ---
class DPFM_GANTrainer(BaseModel):
    """
    GAN-style trainer with Differential Privacy (microbatch clipping, noise addition, accountant) and adversarial privacy pressure.
    Merges CustomDP_SGD training structure with MultiAttrAdversary and adversarial loss scheduling.
    """
    def __init__(
        self,
        *args,
        adv_all_hidden_dims=(128, 64),
        adv_start_epoch: int = 2,
        adv_ramp_epochs: int = 5,
        adv_lambda_start: float = 0.15,
        adv_lambda_end: float = 0.45,
        adv_lambda_cap: float = 0.6,
        repr_dropout: float = 0.05,
        predict_file: str = "dpfm",
        adv_update_freq: int = 1,
        noise_multiplier: float = 1.0,
        dp_max_grad_norm: float = 1.0,
        dp_microbatch_size: int = 1,
        accountant: Any = None,
        delta: float = 1e-5,
        log_freq: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        # --- DP parameters ---
        self.noise_multiplier = float(noise_multiplier)
        self.dp_max_grad_norm = float(dp_max_grad_norm)
        self.dp_microbatch_size = int(dp_microbatch_size)
        self.accountant = accountant
        self.delta = float(delta)
        self.log_freq = int(log_freq)
        self.sample_rate = None
        self.privacy_steps = 0
        self.epsilon_history = []
        self.utility_history = []
        self.accountant = None

        # --- Schedule parameters ---
        self.adv_start_epoch = int(adv_start_epoch)
        self.adv_ramp_epochs = int(adv_ramp_epochs)
        self.adv_lambda_start = float(adv_lambda_start)
        self.adv_lambda_end = float(adv_lambda_end)
        self.adv_lambda_cap = float(adv_lambda_cap)
        self._lambda_adv = 0.0

        # --- Training parameters ---
        self.repr_dropout = nn.Dropout(float(repr_dropout))
        self.adv_update_freq = max(1, int(adv_update_freq))
        self._step_count = 0

        # --- Multi-attribute adversary ---
        num_fields = 2 + (4 if self.use_attrs else 0)  # 2 base (user, item) + 4 attrs (gender, occupation, age, genre) if enabled
        adv_in_dim = self.embed_dim * num_fields
        self.multi_attr_adversary = MultiAttrAdversary(
            in_dim=adv_in_dim,
            num_genders=int(self.hparams.num_genders),
            num_occupations=int(self.hparams.num_occupations),
            genre_dim=int(self.hparams.genre_dim),
            hidden_dims=adv_all_hidden_dims,
        )

    def on_train_start(self):
        try:
            if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "train_dataloader"):
                dl = self.trainer.datamodule.train_dataloader()
            else:
                dl = self.trainer.train_dataloader
            bs = getattr(dl, "batch_size", None)
            ds = getattr(getattr(dl, "dataset", None), "__len__", lambda: None)()
            self.sample_rate = (float(bs) / float(ds)) if (bs and ds) else None
        except Exception:
            self.sample_rate = None

        if self.noise_multiplier > 0 and _HAS_RDP_ACC:
            self.accountant = RDPAccountant()
        else:
            self.accountant = None

    def _build_features_raw(self, batch) -> torch.Tensor:
        user_embed = self.user_embedding(batch['user_id'])
        item_embed = self.item_embedding(batch['item_id'])
        features = [user_embed, item_embed]
        if self.use_attrs:
            gender_embed = self.gender_embedding(batch['gender'])
            occupation_embed = self.occupation_embedding(batch['occupation'])
            age = batch['age'].unsqueeze(-1) if batch['age'].dim() == 1 else batch['age']
            age_embed = F.normalize(self.age_projector(age), dim=1)
            genre_embed = F.normalize(self.genre_projector(batch['genre']), dim=1)
            features.extend([gender_embed, occupation_embed, age_embed, genre_embed])
        return torch.stack(features, dim=1)

    def _build_features(self, batch) -> torch.Tensor:
        user_embed = self.user_embedding(batch['user_id'])
        item_embed = self.item_embedding(batch['item_id'])
        features = [user_embed, item_embed]
        if self.use_attrs:
            gender_embed = self.gender_embedding(batch['gender'])
            occupation_embed = self.occupation_embedding(batch['occupation'])
            age = batch['age'].unsqueeze(-1) if batch['age'].dim() == 1 else batch['age']
            age_embed = F.normalize(self.age_projector(age), dim=1)
            genre_embed = F.normalize(self.genre_projector(batch['genre']), dim=1)
            features.extend([gender_embed, occupation_embed, age_embed, genre_embed])
        return torch.stack(features, dim=1)

    def _predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        B, F, D = features.size()
        linear_term = self.linear(features.view(B, -1))
        summed = features.sum(dim=1)
        summed_squared = summed.pow(2)
        squared_summed = features.pow(2).sum(dim=1)
        bi_interaction = 0.5 * (summed_squared - squared_summed).sum(dim=1, keepdim=True)
        deep_out = self.mlp(features.view(B, -1))
        concat = torch.cat([linear_term, bi_interaction, deep_out], dim=1)
        preds = self.output_layer(concat).squeeze(-1)
        return torch.clamp(preds, self.hparams.target_min, self.hparams.target_max)

    def _compute_entropy_losses(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Gender entropy (categorical)
        p_gender = F.softmax(logits['gender'], dim=-1)
        ent_gender = -(p_gender * torch.log(p_gender.clamp_min(1e-8))).sum(dim=-1).mean()
        # Occupation entropy (categorical)
        p_occupation = F.softmax(logits['occupation'], dim=-1)
        ent_occupation = -(p_occupation * torch.log(p_occupation.clamp_min(1e-8))).sum(dim=-1).mean()
        # Genre entropy (multi-label binary)
        p_genre = torch.sigmoid(logits['genre'])
        ent_genre = -(
            p_genre * torch.log(p_genre.clamp_min(1e-8)) +
            (1 - p_genre) * torch.log((1 - p_genre).clamp_min(1e-8))
        ).mean()
        return ent_gender + ent_occupation + ent_genre

    def _update_lambda_schedule(self):
        e = int(self.current_epoch)
        if e < self.adv_start_epoch:
            lam = 0.0
        elif e < self.adv_start_epoch + self.adv_ramp_epochs:
            progress = (e - self.adv_start_epoch) / max(1, self.adv_ramp_epochs)
            lam = self.adv_lambda_start + progress * (self.adv_lambda_end - self.adv_lambda_start)
        else:
            lam = self.adv_lambda_end
        self._lambda_adv = min(float(lam), self.adv_lambda_cap)
        
    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        """Enable/disable gradients for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad_(flag)
    def on_train_epoch_start(self):
        if hasattr(super(), 'on_train_epoch_start'):
            super().on_train_epoch_start()
        self._update_lambda_schedule()
        self.log("lambda_adv", self._lambda_adv, prog_bar=True)

    def on_fit_start(self):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        """
        Two-pass training per batch (adversary then generator).
        Generator update uses DP-SGD: microbatch clipping, noise addition, and accountant.
        """
        opt_gen, opt_adv = self.optimizers()

        # 1) Adversary pass (no DP noise)
        opt_adv.zero_grad()
        feats = self._build_features_raw(batch)
        with torch.no_grad():
            feat_flat_detached = self.repr_dropout(feats.view(feats.size(0), -1))
        logits_adv = self.multi_attr_adversary(feat_flat_detached)
        y_gender = batch['gender'].view(-1).long()
        y_occupation = batch['occupation'].view(-1).long()
        y_genre = batch['genre'].float()
        y_age = batch['age'].float()
        loss_gender = F.cross_entropy(logits_adv['gender'], y_gender)
        loss_occupation = F.cross_entropy(logits_adv['occupation'], y_occupation)
        loss_genre = F.binary_cross_entropy_with_logits(logits_adv['genre'], y_genre)
        loss_age = F.mse_loss(logits_adv['age'].squeeze(), y_age.squeeze())
        adv_loss = loss_gender + loss_occupation + loss_genre + loss_age
        self.manual_backward(adv_loss)
        opt_adv.step()
        self.log("adv_loss", adv_loss, prog_bar=False)
        self.log("adv_gender_loss", loss_gender, prog_bar=False)
        self.log("adv_occupation_loss", loss_occupation, prog_bar=False)
        self.log("adv_genre_loss", loss_genre, prog_bar=False)
        self.log("adv_age_loss", loss_age, prog_bar=False)

        # 2) Generator pass (DP-SGD)
        opt_gen.zero_grad()
        batch_size = batch['user_id'].shape[0]
        microbatch_size = self.dp_microbatch_size
        n_microbatches = (batch_size + microbatch_size - 1) // microbatch_size
        main_losses = []
        privacy_losses = []
        grads = [torch.zeros_like(p) for p in opt_gen.param_groups[0]['params']]
        for i in range(n_microbatches):
            mb_start = i * microbatch_size
            mb_end = min((i + 1) * microbatch_size, batch_size)
            mb_slice = slice(mb_start, mb_end)
            mb = {k: v[mb_slice] for k, v in batch.items()}
            features = self._build_features(mb)
            preds = self._predict_from_features(features)
            targets = mb['rating'].reshape_as(preds)
            main_loss = self.loss_fn(preds, targets)
            privacy_loss = torch.tensor(0.0, device=self.device)
            if self._lambda_adv > 0:
                self._set_requires_grad(self.multi_attr_adversary, False)
                feat_flat = features.view(features.size(0), -1)
                logits_priv = self.multi_attr_adversary(feat_flat)
                privacy_loss = -self._compute_entropy_losses(logits_priv)
            total_loss = main_loss + self._lambda_adv * privacy_loss
            main_losses.append(main_loss.detach())
            privacy_losses.append(privacy_loss.detach() if self._lambda_adv > 0 else torch.tensor(0.0, device=self.device))
            # Compute gradients for this microbatch
            opt_gen.zero_grad()
            self.manual_backward(total_loss)
            # Clip per-microbatch gradients and accumulate
            for j, p in enumerate(opt_gen.param_groups[0]['params']):
                if p.grad is not None:
                    grad = p.grad.detach()
                    grad_norm = grad.norm(2)
                    if grad_norm > self.dp_max_grad_norm:
                        grad = grad * (self.dp_max_grad_norm / (grad_norm + 1e-6))
                    grads[j] = grads[j] + grad
            if self._lambda_adv > 0:
                self._set_requires_grad(self.multi_attr_adversary, True)
        # Add noise and set gradients
        for j, p in enumerate(opt_gen.param_groups[0]['params']):
            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier * self.dp_max_grad_norm,
                size=p.shape,
                device=p.device,
            )
            p.grad = (grads[j] + noise) / n_microbatches
        opt_gen.step()
        self.privacy_steps += 1
        if self.accountant is not None and self.sample_rate is not None:
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)
        # Logging
        mean_main_loss = torch.stack(main_losses).mean()
        mean_privacy_loss = torch.stack(privacy_losses).mean() if self._lambda_adv > 0 else torch.tensor(0.0, device=self.device)
        total_loss = mean_main_loss + self._lambda_adv * mean_privacy_loss
        self.log("gen_main_loss", mean_main_loss, prog_bar=False)
        self.log("gen_total_loss", total_loss, prog_bar=False)
        if self._lambda_adv > 0:
            self.log("gen_privacy_loss", mean_privacy_loss, prog_bar=False)
        self.log("train_loss", total_loss, prog_bar=True)
        # Only use full batch preds/targets for metrics
        features = self._build_features(batch)
        preds = self._predict_from_features(features)
        targets = batch['rating'].reshape_as(preds)
        self.train_mse.update(preds, targets)
        self.train_mae.update(preds, targets)
        self.train_rmse.update(preds, targets)
        return total_loss

    def on_train_epoch_end(self):
        self.log('train_mse', self.train_mse.compute(), prog_bar=True)
        self.log('train_mae', self.train_mae.compute(), prog_bar=True)
        self.log('train_rmse', self.train_rmse.compute(), prog_bar=True)

        if self.accountant is not None:
            try:
                eps = float(self.accountant.get_epsilon(self.delta))
                self.epsilon_history.append(eps)
                self.log("epsilon", eps, prog_bar=True)
            except Exception:
                pass

        if hasattr(super(), 'on_train_epoch_end'):
            super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        # forward using BaseModel
        preds = super().forward(batch)
        targets = batch['rating']

        # compute the configured training loss (MSE or SmoothL1/Huber)
        main_loss = self.loss_fn(preds, targets)

        # update epoch metrics (handled by torchmetrics)
        self.val_mse.update(preds, targets)
        self.val_mae.update(preds, targets)
        self.val_rmse.update(preds, targets)

        # optional: privacy probe (attribute entropy, higher is better)
        features = self._build_features(batch)
        feat_flat = features.view(features.size(0), -1).detach()
        logits = self.multi_attr_adversary(feat_flat)
        privacy_loss = self._compute_entropy_losses(logits)

        # log only scalar losses here; metrics are logged in on_validation_epoch_end
        self.log("val_loss", main_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_privacy_loss", privacy_loss, prog_bar=False, on_step=False, on_epoch=True)

        return main_loss

    def on_validation_epoch_end(self):
        val_rmse = None
        try:
            val_rmse = float(self.val_rmse.compute().detach().cpu())
        except Exception:
            pass

        if val_rmse is not None:
            self.utility_history.append(val_rmse)

        if hasattr(super(), 'on_validation_epoch_end'):
            super().on_validation_epoch_end()

    def on_train_end(self):
        n = min(len(self.epsilon_history), len(self.utility_history))
        if n <= 0:
            return
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(self.epsilon_history[:n], self.utility_history[:n], marker='o', linestyle='-')
            plt.xlabel('Epsilon (ε)')
            plt.ylabel('Validation RMSE')
            plt.title('Privacy–Utility Trade-off')
            plt.grid(True, linestyle='--', alpha=0.6)
            out_name = f"privacy_utility_tradeoff_{getattr(self.hparams, 'predict_file', 'dpfm')}.png"
            plt.tight_layout()
            plt.savefig(out_name, dpi=300)
            plt.close()
            print(f"[DP] Saved privacy–utility plot to {out_name}")
        except Exception as e:
            print(f"[DP] Failed to save privacy–utility plot: {e}")

    def configure_optimizers(self):
        # Generator parameters (everything except adversary)
        gen_params = [
            p for n, p in self.named_parameters()
            if "multi_attr_adversary" not in n and p.requires_grad
        ]
        adv_params = list(self.multi_attr_adversary.parameters())
        opt_gen = torch.optim.AdamW(
            gen_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_penalty
        )
        opt_adv = torch.optim.AdamW(
            adv_params,
            lr=self.hparams.learning_rate * 0.5,
            weight_decay=1e-4
        )
        return [opt_gen, opt_adv]