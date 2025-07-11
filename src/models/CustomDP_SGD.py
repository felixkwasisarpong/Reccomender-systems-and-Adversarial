import math, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from .BaseModel import BaseModel
import seaborn as sns

class CustomDP_SGD(BaseModel):
    def __init__(
        self,
        num_users,
        num_items,
        noise_type='gaussian',
        noise_scale=0.0,
        enable_dp=False,
        use_attrs=True,
        clip_norm=1.0,
        delta=1e-5,
        log_freq=50,
        **kwargs
    ):
        super().__init__(num_users=num_users, num_items=num_items, use_attrs=use_attrs, **kwargs)
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.delta = delta
        self.log_freq = log_freq
        self.epsilon_history = []
        self.metric_history = []
        self.privacy_steps = 0
        self.sample_rate = None
        self.automatic_optimization = False

    def on_train_start(self):
        batch_size = self.trainer.train_dataloader.batch_size
        self.sample_rate = batch_size / len(self.trainer.datamodule.train_dataset)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        preds = self(batch)  # batch is a dict, as expected by forward()
        targets = batch["rating"]
        loss = self.loss_fn(preds, targets)

        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        if self.noise_scale > 0:
            for p in self.parameters():
                if p.grad is not None:
                    if self.noise_type == 'gaussian':
                        noise = torch.randn_like(p.grad)
                    else:
                        noise = torch.distributions.Laplace(0, 1).sample(p.grad.shape).to(p.grad.device)
                    p.grad += noise * self.noise_scale * self.clip_norm
            self.privacy_steps += 1

        optimizer.step()

        self.train_mse.update(preds, targets)
        self.train_mae.update(preds, targets)
        self.train_rmse.update(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_rmse = self.trainer.callback_metrics.get("val_rmse")
        if val_rmse is not None:
            self.metric_history.append(val_rmse.item())
        super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        if self.noise_scale > 0 and self.sample_rate is not None:
            if self.noise_type == 'gaussian':
                orders = np.arange(2, 64)
                rdp = self.compute_rdp_gaussian(
                    q=self.sample_rate,
                    noise_multiplier=self.noise_scale,
                    steps=self.privacy_steps,
                    orders=orders
                )
                eps, opt_order = self.get_privacy_spent_rdp(orders, rdp, self.delta)
            elif self.noise_type == 'laplace':
                eps = self.compute_eps_laplace_subsampled(
                    q=self.sample_rate,
                    noise_scale=self.noise_scale,
                    steps=self.privacy_steps
                )
                opt_order = None  # Not needed for Laplace
            else:
                raise ValueError(f"Unsupported noise type: {self.noise_type}")

            self.epsilon_history.append(eps)
            self.log("epsilon", eps, prog_bar=True)
            print(f"[{self.noise_type.upper()}] ε={eps:.4f} at δ={self.delta}, α={opt_order if opt_order else 'N/A'}")

        super().on_train_epoch_end()



    def on_train_end(self):
        n = min(len(self.epsilon_history), len(self.metric_history))
        if n > 0:
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

            fname = f"privacy_tradeoff_{int(time.time())}.png"
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
                # Tight RDP bound for subsampled Gaussian (Wang et al. 2019)
                sigma = noise_multiplier
                term1 = alpha / (2 * sigma**2)
                term2 = math.log(1 + q**2 * (alpha - 1) / alpha)
                term3 = math.log(1 + q * (alpha - 1) / (alpha - 1))
                rdp_val = term1 * q**2 + min(term2, term3)
                rdp.append(rdp_val)
        return np.array(rdp) * steps

    @staticmethod
    def get_privacy_spent_rdp(orders, rdp, delta):
        eps = rdp - math.log(delta) / (np.array(orders) - 1)
        idx = np.argmin(eps)
        return eps[idx], orders[idx]

    @staticmethod
    def compute_eps_laplace(noise_scale, steps):
        # Laplace mechanism ε-composition: ε = T / noise_scale
        return steps / noise_scale
    
    @staticmethod
    def compute_eps_laplace_subsampled(q, noise_scale, steps):
        """
        Computes ε for Laplace mechanism with Poisson subsampling.
        Based on: https://arxiv.org/abs/1212.1987 (Theorem 4.1)
        """
        if noise_scale == 0:
            return float('inf')
        # ε = log(1 + q * (exp(1/noise_scale) - 1)) * steps
        eps_per_step = math.log(1 + q * (math.exp(1 / noise_scale) - 1))
        return eps_per_step * steps