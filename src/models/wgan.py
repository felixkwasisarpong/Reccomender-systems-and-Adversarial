import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from torchmetrics import MeanAbsoluteError, MeanSquaredError


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
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


class Generator(nn.Module):
    def __init__(self, noise_dim=32, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # No sigmoid
        )

    def forward(self, x):
        return self.model(x)


class WGAN(pl.LightningModule):
    def __init__(self, noise_dim=32, output_dim=2, lr=1e-4, n_critic=5, lambda_gp=10):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(noise_dim=noise_dim, output_dim=output_dim)
        self.critic = Critic(input_dim=output_dim)

        self.automatic_optimization = False
        self.validation_step_outputs = []

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_data = batch
        opt_g, opt_c = self.optimizers()

        # === Train Critic ===
        for _ in range(self.hparams.n_critic):
            noise = torch.randn(real_data.size(0), self.hparams.noise_dim, device=self.device)
            fake_data = self(z=noise).detach()
            real_scores = self.critic(real_data)
            fake_scores = self.critic(fake_data)
            gp = compute_gradient_penalty(self.critic, real_data, fake_data, self.device)
            critic_loss = fake_scores.mean() - real_scores.mean() + self.hparams.lambda_gp * gp

            opt_c.zero_grad()
            self.manual_backward(critic_loss)
            opt_c.step()

        # === Train Generator ===
        noise = torch.randn(real_data.size(0), self.hparams.noise_dim, device=self.device)
        fake_data = self(z=noise)
        gen_loss = -self.critic(fake_data).mean()

        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()

        self.log_dict({"critic_loss": critic_loss, "gen_loss": gen_loss}, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [opt_g, opt_c], []

    def validation_step(self, batch, batch_idx):
        noise = torch.randn(batch.size(0), self.hparams.noise_dim, device=self.device)
        fake_data = self(z=noise)
        self.validation_step_outputs.append(fake_data)

    def on_validation_epoch_end(self):
        all_generated = torch.cat(self.validation_step_outputs, dim=0)
        self.log("val_mean", all_generated.mean())
        self.validation_step_outputs.clear()
