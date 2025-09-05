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
        adv_conf_weight_base: float = 1e-3,   # <-- add this
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
        self.adv_conf_weight_base = float(adv_conf_weight_base)
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
        # Match BaseModel's enabled attribute set to avoid shape/attribute mismatches
        attr_count = int(self.use_gender) + int(self.use_occupation) + int(self.use_age) + int(self.use_genre)
        num_fields = 2 + attr_count
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
        # Append only enabled attributes (aligned with BaseModel)
        if self.use_gender:
            gender_embed = self.gender_embedding(batch['gender'])
            features.append(gender_embed)
        if self.use_occupation:
            occupation_embed = self.occupation_embedding(batch['occupation'])
            features.append(occupation_embed)
        if self.use_age:
            age = batch['age']
            if age.dim() == 1:
                age = age.unsqueeze(-1)
            age_embed = F.normalize(self.age_projector(age.float()), dim=1)
            features.append(age_embed)
        if self.use_genre:
            # Prefer projector if defined; otherwise assume genre already in embedding space
            if hasattr(self, 'genre_projector'):
                genre_embed = F.normalize(self.genre_projector(batch['genre'].float()), dim=1)
            else:
                genre_embed = F.normalize(batch['genre'].float(), dim=1)
            features.append(genre_embed)
        return torch.stack(features, dim=1)

    def _build_features(self, batch) -> torch.Tensor:
        user_embed = self.user_embedding(batch['user_id'])
        item_embed = self.item_embedding(batch['item_id'])
        features = [user_embed, item_embed]
        # Append only enabled attributes (aligned with BaseModel)
        if self.use_gender:
            gender_embed = self.gender_embedding(batch['gender'])
            features.append(gender_embed)
        if self.use_occupation:
            occupation_embed = self.occupation_embedding(batch['occupation'])
            features.append(occupation_embed)
        if self.use_age:
            age = batch['age']
            if age.dim() == 1:
                age = age.unsqueeze(-1)
            age_embed = F.normalize(self.age_projector(age.float()), dim=1)
            features.append(age_embed)
        if self.use_genre:
            if hasattr(self, 'genre_projector'):
                genre_embed = F.normalize(self.genre_projector(batch['genre'].float()), dim=1)
            else:
                genre_embed = F.normalize(batch['genre'].float(), dim=1)
            features.append(genre_embed)
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
        total = 0.0
        n = 0
        if self.use_gender and 'gender' in logits:
            p_gender = F.softmax(logits['gender'], dim=-1)
            ent_gender = -(p_gender * torch.log(p_gender.clamp_min(1e-8))).sum(dim=-1).mean()
            total = total + ent_gender
            n += 1
        if self.use_occupation and 'occupation' in logits:
            p_occupation = F.softmax(logits['occupation'], dim=-1)
            ent_occupation = -(p_occupation * torch.log(p_occupation.clamp_min(1e-8))).sum(dim=-1).mean()
            total = total + ent_occupation
            n += 1
        if self.use_genre and 'genre' in logits:
            p_genre = torch.sigmoid(logits['genre'])
            ent_genre = -(
                p_genre * torch.log(p_genre.clamp_min(1e-8)) +
                (1 - p_genre) * torch.log((1 - p_genre).clamp_min(1e-8))
            ).mean()
            total = total + ent_genre
            n += 1
        if self.use_age and 'age' in logits:
            # Encourage wide age predictions by maximizing variance (equiv. to entropy for Gaussian)
            # Here, approximate with negative L2 magnitude to avoid trivial 0.
            ent_age = -torch.mean(logits['age'].squeeze() ** 2)
            total = total + ent_age
            n += 1
        return total if n > 0 else torch.tensor(0.0, device=self.device)

    def _prediction_confidence_penalty(self, preds: torch.Tensor) -> torch.Tensor:
        """Entropy penalty on bounded predictions to discourage overconfident outputs (helps lower MIA).
        Maps prediction range [target_min, target_max] → [0,1] and applies binary entropy.
        """
        p = (preds - self.hparams.target_min) / (self.hparams.target_max - self.hparams.target_min)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return ent.mean()

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

    def _noise_adapt(self, lam: float) -> float:
        """Scale adversarial weight by a tempered inverse noise.
        Instead of lam / noise, use lam / (1 + noise) to avoid over-dampening at large noise.
        Capped by adv_lambda_cap for stability.
        """
        scale = 1.0 / (1.0 + max(float(self.noise_multiplier), 0.0))
        lam_eff = float(lam) * scale
        return float(min(lam_eff, self.adv_lambda_cap))
    
    def _conf_weight(self) -> float:
        """Noise-aware weight for the prediction confidence penalty.
        Higher DP noise -> smaller penalty weight to protect utility.
        """
        return float(self.adv_conf_weight_base) / (1.0 + max(float(self.noise_multiplier), 0.0))
        
    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        """Enable/disable gradients for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad_(flag)
    def on_train_epoch_start(self):
        if hasattr(super(), 'on_train_epoch_start'):
            super().on_train_epoch_start()
        self._update_lambda_schedule()
        self.log("lambda_adv", self._lambda_adv, prog_bar=True)
        try:
            self.log("lambda_adv_eff", self._noise_adapt(self._lambda_adv), prog_bar=True)
        except Exception:
            pass
        try:
            self.log("conf_weight", torch.tensor(self._conf_weight(), dtype=torch.float32, device=self.device), prog_bar=False)
        except Exception:
            pass

    def on_fit_start(self):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                # Enable TF32 where available for speed, keep determinism relaxed for DP noise use
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        """
        Two-pass training per batch (adversary then generator).
        Generator update uses DP-SGD: microbatch clipping, noise addition, and accountant.
        """
        opt_gen, opt_adv = self.optimizers()

        # 1) Adversary pass (no DP noise); allow multiple updates per generator step
        feats = self._build_features_raw(batch)
        with torch.no_grad():
            feat_flat_detached = self.repr_dropout(feats.view(feats.size(0), -1))
        # Under strong privacy (large noise), reduce adversary update frequency
        adv_updates = max(1, int(self.adv_update_freq // 2)) if float(self.noise_multiplier) >= 1.0 else int(self.adv_update_freq)
        # Prepare labels only for enabled attributes
        y_gender = batch['gender'].view(-1).long() if self.use_gender else None
        y_occupation = batch['occupation'].view(-1).long() if self.use_occupation else None
        y_genre = batch['genre'].float() if self.use_genre else None
        y_age = batch['age'].float() if self.use_age else None

        last_adv_loss = None
        for _ in range(adv_updates):
            opt_adv.zero_grad()
            logits_adv = self.multi_attr_adversary(feat_flat_detached)
            adv_loss = torch.tensor(0.0, device=self.device)
            if self.use_gender:
                loss_gender = F.cross_entropy(logits_adv['gender'], y_gender)
                adv_loss = adv_loss + loss_gender
            else:
                loss_gender = None
            if self.use_occupation:
                loss_occupation = F.cross_entropy(logits_adv['occupation'], y_occupation)
                adv_loss = adv_loss + loss_occupation
            else:
                loss_occupation = None
            if self.use_genre:
                loss_genre = F.binary_cross_entropy_with_logits(logits_adv['genre'], y_genre)
                adv_loss = adv_loss + (0.5 * loss_genre if float(self.noise_multiplier) >= 1.0 else loss_genre)
            else:
                loss_genre = None
            if self.use_age:
                loss_age = F.mse_loss(logits_adv['age'].squeeze(), y_age.squeeze())
                adv_loss = adv_loss + (0.5 * loss_age if float(self.noise_multiplier) >= 1.0 else loss_age)
            else:
                loss_age = None
            self.manual_backward(adv_loss)
            opt_adv.step()
            last_adv_loss = adv_loss.detach()

        if last_adv_loss is not None:
            self.log("adv_loss", last_adv_loss, prog_bar=False)
            if self.use_gender and loss_gender is not None:
                self.log("adv_gender_loss", loss_gender.detach(), prog_bar=False)
            if self.use_occupation and loss_occupation is not None:
                self.log("adv_occupation_loss", loss_occupation.detach(), prog_bar=False)
            if self.use_genre and loss_genre is not None:
                self.log("adv_genre_loss", loss_genre.detach(), prog_bar=False)
            if self.use_age and loss_age is not None:
                self.log("adv_age_loss", loss_age.detach(), prog_bar=False)

        # 2) Generator pass (DP-SGD) -- microbatch approximation to per-sample clipping
        # Treat each small microbatch as a unit for clipping; report as an approximation in the paper.
        lambda_eff = self._noise_adapt(self._lambda_adv)
        opt_gen.zero_grad(set_to_none=True)
        batch_size = batch['user_id'].shape[0]
        C = float(self.dp_max_grad_norm)
        sigma = float(self.noise_multiplier)
        m = max(1, int(self.dp_microbatch_size))  # e.g., 4

        main_losses = []
        privacy_losses = []
        conf_penalties = []

        # Generator params only (must match opt_gen param order)
        gen_params = [
            p for n, p in self.named_parameters()
            if "multi_attr_adversary" not in n and p.requires_grad
        ]
        sum_clipped_grads = [torch.zeros_like(p) for p in gen_params]
        had_any_grad = [False for _ in gen_params]

        # Iterate microbatches
        n_mb = (batch_size + m - 1) // m
        for i in range(n_mb):
            mb_start = i * m
            mb_end = min((i + 1) * m, batch_size)
            mb = {k: v[mb_start:mb_end] for k, v in batch.items()}

            # Build features per microbatch to avoid reusing the same graph across backward calls
            features = self._build_features(mb)
            preds = self._predict_from_features(features)
            targets = mb['rating'].reshape_as(preds)

            main_loss = self.loss_fn(preds, targets)
            # Use mean over microbatch for stability
            main_loss_mean = main_loss if main_loss.dim() == 0 else main_loss.mean()

            privacy_loss = torch.tensor(0.0, device=self.device)
            if self._lambda_adv > 0:
                # freeze adversary params while probing privacy
                self._set_requires_grad(self.multi_attr_adversary, False)
                feat_flat = features.view(features.size(0), -1)
                logits_priv = self.multi_attr_adversary(feat_flat)
                privacy_loss = -self._compute_entropy_losses(logits_priv)
                self._set_requires_grad(self.multi_attr_adversary, True)

            conf_pen = self._prediction_confidence_penalty(preds)
            conf_w = self._conf_weight()
            total_loss = main_loss_mean + lambda_eff * privacy_loss + conf_w * conf_pen

            main_losses.append(main_loss_mean.detach())
            privacy_losses.append(privacy_loss.detach() if self._lambda_adv > 0 else torch.tensor(0.0, device=self.device))
            conf_penalties.append(conf_pen.detach())

            # Backward on the microbatch
            self.zero_grad(set_to_none=True)
            self.manual_backward(total_loss)

            # Collect grads, clip global L2 to C, and accumulate (approx. per-sample via microbatches)
            per_param_grads = [p.grad.detach() if p.grad is not None else None for p in gen_params]
            if any(g is not None for g in per_param_grads):
                flat = torch.cat([g.view(-1) for g in per_param_grads if g is not None])
                global_norm = torch.linalg.vector_norm(flat, ord=2)
                scale = min(1.0, C / (global_norm + 1e-6))
                for j, g in enumerate(per_param_grads):
                    if g is not None:
                        sum_clipped_grads[j].add_(g * scale)
                        had_any_grad[j] = True

        # Add Gaussian noise to summed, clipped grads and average by TRUE batch size
        # NOTE: This is an approximation to per-sample DP; we keep noise std = sigma*C per parameter.
        for j, p in enumerate(gen_params):
            if had_any_grad[j]:
                noise = torch.normal(mean=0.0, std=sigma * C, size=p.shape, device=p.device)
                p.grad = (sum_clipped_grads[j] + noise) / float(batch_size)
            else:
                p.grad = None

        opt_gen.step()
        self.privacy_steps += 1
        if self.accountant is not None and self.noise_multiplier > 0:
            if self.sample_rate is None:
                try:
                    dl = self.trainer.datamodule.train_dataloader()
                    bs = getattr(dl, "batch_size", None)
                    ds = getattr(getattr(dl, "dataset", None), "__len__", lambda: None)()
                    self.sample_rate = (float(bs) / float(ds)) if (bs and ds) else None
                except Exception:
                    self.sample_rate = None
            if self.sample_rate is not None:
                self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)
        # Logging
        mean_main_loss = torch.stack(main_losses).mean()
        mean_privacy_loss = torch.stack(privacy_losses).mean() if self._lambda_adv > 0 else torch.tensor(0.0, device=self.device)
        total_loss = mean_main_loss + self._lambda_adv * mean_privacy_loss
        self.log("gen_main_loss", mean_main_loss, prog_bar=False)
        self.log("gen_total_loss", total_loss, prog_bar=False)
        if self._lambda_adv > 0:
            self.log("gen_privacy_loss", mean_privacy_loss, prog_bar=False)
        # confidence penalty mean
        try:
            mean_conf_pen = torch.stack(conf_penalties).mean()
            self.log("gen_conf_pen", mean_conf_pen, prog_bar=False)
        except Exception:
            pass
        # log effective lambda used this step
        try:
            self.log("lambda_adv_eff_step", torch.tensor(lambda_eff, dtype=torch.float32, device=self.device), prog_bar=False)
            self.log("conf_weight", torch.tensor(self._conf_weight(), dtype=torch.float32, device=self.device), prog_bar=False)
            self.log("adv_updates_eff", torch.tensor(float(adv_updates), dtype=torch.float32, device=self.device), prog_bar=False)
        except Exception:
            pass
        
        self.log("train_loss", total_loss, prog_bar=True)
        # Only use full batch preds/targets for metrics
        # Recompute full-batch preds for logging/metrics (no backward here)
        features_fb = self._build_features(batch)
        preds = self._predict_from_features(features_fb)
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
