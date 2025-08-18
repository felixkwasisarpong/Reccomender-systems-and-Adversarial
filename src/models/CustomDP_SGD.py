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
        num_users,
        num_items,
        num_genders,
        num_occupations,
        genre_dim,
        noise_type='gaussian',
        noise_scale=0.0,
        enable_dp=False,
        clip_norm=1.0,
        delta=1e-5,
        log_freq=50,
        dp_microbatch_size: int = 32,
        predict_file="predictions",
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

        preds = self(batch)
        targets = batch["rating"]
        loss = self.loss_fn(preds, targets)

        params = [p for p in self.parameters() if p.requires_grad]
        accum_grads = [torch.zeros_like(p, device=p.device) for p in params]
        B = targets.shape[0]
        eff_count = 0

        for start in range(0, B, self.dp_microbatch_size):
            end = min(start + self.dp_microbatch_size, B)
            for i in range(start, end):
                # zero model grads (faster than optimizer.zero_grad() in a tight loop)
                self.zero_grad(set_to_none=True)
                b_i = {k: (v[i:i+1] if torch.is_tensor(v) else v) for k, v in batch.items()}
                out_i = self(b_i)
                loss_i = self.loss_fn(out_i, b_i["rating"])
                self.manual_backward(loss_i)

                # per-sample grad L2 over all params
                g_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params) + 1e-12)
                c = float(min(1.0, self.clip_norm / (g_norm + 1e-12)))
                for j, p in enumerate(params):
                    accum_grads[j] += p.grad.detach() * c
                eff_count += 1

        # average clipped grads and add noise                                                                                                
        if eff_count == 0:
            eff_count = B  # safety, should not happen
        for p, g in zip(params, accum_grads):
            if self.noise_scale > 0:
                if self.noise_type == 'gaussian':
                    noise = torch.randn_like(g) * (self.noise_scale * self.clip_norm)
                elif self.noise_type == 'laplace':
                    s = self.noise_scale * self.clip_norm
                    noise = torch.distributions.Laplace(loc=0.0, scale=s).sample(g.shape).to(g.device)
                else:
                    raise ValueError(f"Unsupported noise type: {self.noise_type}")
                g = g + noise
            g = g / float(eff_count)
            p.grad = g

        if self.noise_scale > 0:
            self.privacy_steps += 1
            if self.accountant is not None and self.sample_rate is not None:
                self.accountant.step(noise_multiplier=self.noise_scale, sample_rate=self.sample_rate)

        optimizer.step()

        # Update metrics defined in BaseModel
        self.train_mse.update(preds, targets)
        self.train_mae.update(preds, targets)
        self.train_rmse.update(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
            log_dir = "privacy_utility"
            plt.xlabel('Epsilon (ε)', fontsize=12)
            plt.ylabel('Validation RMSE', fontsize=12)
            plt.title('Privacy–Utility Trade-off Curve', fontsize=14)
            plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.legend()
            plt.tight_layout()

            fname = os.path.join(log_dir, f"privacy_utility_tradeoff_custom_mid.png")
          
            plt.savefig(fname, dpi=300)
            wandb.save(fname)
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