import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import matplotlib.pyplot as plt
import os
import numpy as np




class Generator(nn.Module):
    def __init__(self, noise_dim=32, output_dim=25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(critic, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    critic_output = critic(interpolated)
    gradients = grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty, grad_norm.mean()

class WGAN(pl.LightningModule):
    def __init__(self, noise_dim=32, output_dim=24, lr=1e-4, n_critic=5, lambda_gp=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(noise_dim, output_dim)
        self.critic = Critic(output_dim)
        self.automatic_optimization = False
        self.kl_real = []
        self.kl_fake = []
        os.makedirs("wgan_samples_dp_weak", exist_ok=True)

    def forward(self, z):
        return self.generator(z).clamp(0, 1)

    def training_step(self, batch, batch_idx):
        real = batch.float()
        opt_g, opt_c = self.optimizers()
        for _ in range(self.hparams.n_critic):
            z = torch.randn(real.size(0), self.hparams.noise_dim, device=self.device)
            fake = self(z).detach()
            real_score = self.critic(real)
            fake_score = self.critic(fake)
            gp, gn = compute_gradient_penalty(self.critic, real, fake, self.device)
            loss_c = fake_score.mean() - real_score.mean() + self.hparams.lambda_gp * gp
            opt_c.zero_grad(); self.manual_backward(loss_c); opt_c.step()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.log("critic_loss", loss_c, prog_bar=True)
            self.log("grad_norm", gn, prog_bar=True)
        z = torch.randn(real.size(0), self.hparams.noise_dim, device=self.device)
        fake = self(z)
        loss_g = -self.critic(fake).mean()
        opt_g.zero_grad(); self.manual_backward(loss_g); opt_g.step()
        self.log("gen_loss", loss_g, prog_bar=True)
        # collect for KL
        if batch_idx == 0:
            self.kl_real.append(real.detach().cpu().numpy())
            self.kl_fake.append(fake.detach().cpu().numpy())



    def on_fit_start(self):
        if self.logger and self.logger.__class__.__name__.lower().startswith("wandb"):
            import wandb; wandb.define_metric("kl_divergence/*", summary="last")

    def on_train_end(self):
        # make sure we have samples
        if not self.kl_real or not self.kl_fake:
            print("No samples for KL. Skipping."); 
            return

        # stack everything: shape [N, D]
        real_arr = np.concatenate(self.kl_real, axis=0)
        fake_arr = np.concatenate(self.kl_fake, axis=0)

        # pick out the last column (predicted rating)
        # pick out the predictions
        real_r = real_arr[:, -1]
        fake_r = fake_arr[:, -1]


        # Build bins safely
        low, high = float(real_r.min()), float(real_r.max())
        if np.isclose(low, high):
            low -= 1e-3
            high += 1e-3
        bins = np.linspace(low, high, 51)

        # Raw counts
        pr, _ = np.histogram(real_r, bins=bins, density=False)
        pf, _ = np.histogram(fake_r, bins=bins, density=False)

        # Smooth + normalize
        eps = 1e-8
        pr = (pr + eps) / (pr + eps).sum()
        pf = (pf + eps) / (pf + eps).sum()

        # KL
        kl_pred = np.sum(pr * np.log(pr / pf))

        # Plot
        plt.figure(figsize=(4,4))
        centers = 0.5*(bins[:-1] + bins[1:])
        plt.bar(centers, pr, width=bins[1]-bins[0], alpha=0.6, label="Real")
        plt.bar(centers, pf, width=bins[1]-bins[0], alpha=0.6, label="Fake")
        plt.title(f"KL(pred_rating) = {kl_pred:.3f}")
        plt.legend()
        plt.savefig("wgan_samples_dp_weak/kl_pred_rating.png")
        plt.close()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=1e-5, betas=(0.0, 0.9))
        return [opt_g, opt_c], []