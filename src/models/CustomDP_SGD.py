import math
import torch
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import os
from .BaseModel import BaseModel
from opacus.accountants import RDPAccountant


class CustomDP_SGD(BaseModel):
    def __init__(
        self,
        # BaseModel-required args
        num_users: int,
        num_items: int,
        num_genders: int,
        num_occupations: int,
        genre_dim: int,
        # DP/optimizer controls
        noise_type: str = 'gaussian',
        noise_scale: float = 0.0,
        enable_dp: bool = False,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
        log_freq: int = 50,
        dp_microbatch_size: int = 32,
        dp_fast_microbatch: bool = True,
        # bookkeeping
        predict_file: str = "predictions",
        # Forward extras to BaseModel (e.g., use_genre, embed_dim, etc.)
        **kwargs
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            num_genders=num_genders,
            num_occupations=num_occupations,
            genre_dim=genre_dim,
            predict_file=predict_file,
            **kwargs
        )
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.delta = delta
        self.log_freq = log_freq
        self.dp_microbatch_size = int(dp_microbatch_size)
        self.dp_fast_microbatch = bool(dp_fast_microbatch)
        self.epsilon_history = []
        self.metric_history = []
        self.privacy_steps = 0
        self.sample_rate = None
        self.automatic_optimization = False
        self.accountant = None

    def on_train_start(self):
        if self.trainer is not None and hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            batch_size = self.trainer.train_dataloader.batch_size
            self.sample_rate = batch_size / len(self.trainer.datamodule.train_dataset)
        else:
            self.sample_rate = None
        if self.noise_scale > 0:
            self.accountant = RDPAccountant()

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        self.zero_grad(set_to_none=True)

        preds_full = self(batch)
        targets_full = batch["rating"].reshape_as(preds_full)
        loss_full = self.loss_fn(preds_full, targets_full)

        # Fast microbatch DP approximation (clip at microbatch level, add noise, average by true batch size)
        if self.dp_fast_microbatch and (self.noise_scale > 0 or self.enable_dp):
            params = [p for p in self.parameters() if p.requires_grad]
            sum_clipped = [torch.zeros_like(p, device=p.device) for p in params]
            had_grad = [False] * len(params)
            B = targets_full.shape[0]
            m = max(1, int(self.dp_microbatch_size))

            n_mb = (B + m - 1) // m
            for i in range(n_mb):
                s = i * m
                e = min((i + 1) * m, B)
                mb = {k: (v[s:e] if torch.is_tensor(v) else v) for k, v in batch.items()}

                self.zero_grad(set_to_none=True)
                out_mb = self(mb)
                loss_mb = self.loss_fn(out_mb, mb["rating"].reshape_as(out_mb))
                loss_mb = loss_mb if loss_mb.dim() == 0 else loss_mb.mean()
                self.manual_backward(loss_mb)

                flat = []
                for p in params:
                    if p.grad is not None:
                        flat.append(p.grad.detach().view(-1))
                if flat:
                    flat = torch.cat(flat)
                    gnorm = torch.linalg.vector_norm(flat, ord=2)
                    scale = min(1.0, float(self.clip_norm) / (float(gnorm) + 1e-6))
                else:
                    scale = 1.0

                for j, p in enumerate(params):
                    if p.grad is not None:
                        sum_clipped[j] = sum_clipped[j] + p.grad.detach() * scale
                        had_grad[j] = True

            # Add noise and average by true batch size
            for j, p in enumerate(params):
                if not had_grad[j]:
                    p.grad = None
                    continue
                g = sum_clipped[j]
                if self.noise_scale > 0:
                    if self.noise_type == 'gaussian':
                        noise = torch.normal(mean=0.0, std=self.noise_scale * self.clip_norm, size=p.shape, device=p.device)
                    elif self.noise_type == 'laplace':
                        s = self.noise_scale * self.clip_norm
                        noise = torch.distributions.Laplace(loc=0.0, scale=s).sample(p.shape).to(p.device)
                    else:
                        raise ValueError(f"Unsupported noise type: {self.noise_type}")
                    g = g + noise
                p.grad = g / float(B)

            if self.noise_scale > 0:
                self.privacy_steps += 1
                if self.accountant is not None and self.sample_rate is not None:
                    self.accountant.step(noise_multiplier=self.noise_scale, sample_rate=self.sample_rate)

            optimizer.step()

        else:
            # Non-DP or slow path: just optimize on full batch
            self.manual_backward(loss_full)
            optimizer.step()

        # Update metrics defined in BaseModel using full-batch preds
        self.train_mse.update(preds_full, targets_full)
        self.train_mae.update(preds_full, targets_full)
        self.train_rmse.update(preds_full, targets_full)
        self.log("train_loss", loss_full, on_step=True, on_epoch=True, prog_bar=True)
        return loss_full

    def on_fit_start(self):
        # minor runtime speed ups (safe defaults)
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()  # Logs + resets metrics
        val_rmse = self.trainer.callback_metrics.get("val_rmse")
        if val_rmse is not None:
            self.metric_history.append(val_rmse.item())

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.noise_scale > 0 and self.accountant is not None:
            eps = self.accountant.get_epsilon(self.delta)
            self.epsilon_history.append(eps)
            self.log("epsilon", eps, on_epoch=True, prog_bar=True)
            print(f"[{self.noise_type.upper()}][Opacus] ε={eps:.4f} at δ={self.delta}")

    def on_train_end(self):
        n = min(len(self.epsilon_history), len(self.metric_history))
        if n > 0:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set(style="whitegrid", palette="muted", font_scale=1.2)
            plt.figure(figsize=(8, 5))
            plt.plot(
                self.epsilon_history[:n],
                self.metric_history[:n],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=7,
                color='#1f77b4',
                label='Validation RMSE'
            )
        
            plt.xlabel('Epsilon (ε)', fontsize=12)
            plt.ylabel('Validation RMSE', fontsize=12)
            plt.title('Privacy–Utility Trade-off Curve', fontsize=14)
            plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.legend()
            plt.tight_layout()

            out_name = f"privacy_utility_tradeoff_{getattr(self.hparams, 'predict_file', 'dpfm')}.png"
          
            plt.savefig(out_name, dpi=300)
            wandb.save(out_name)
            plt.close()

    @staticmethod
    def compute_rdp_gaussian(q, noise_multiplier, steps, orders):
        rdp = []
        for alpha in orders:
            if q == 0 or noise_multiplier == 0:
                rdp.append(0 if q == 0 else float('inf'))
            else:
                rdp_val = (q ** 2) * alpha / (2 * (noise_multiplier ** 2))
                rdp.append(rdp_val)
        return np.array(rdp) * steps

    @staticmethod
    def get_privacy_spent_rdp(orders, rdp, delta):
        eps = rdp - np.log(delta) / (np.array(orders) - 1)
        idx = np.argmin(eps)
        return eps[idx], orders[idx]

    @staticmethod
    def compute_eps_laplace_subsampled(q, noise_scale, steps):
        if noise_scale == 0:
            return float('inf')
        eps_per_step = math.log(1 + q * (math.exp(1 / noise_scale) - 1))
        return eps_per_step * steps
