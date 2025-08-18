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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression



class Generator(nn.Module):
    def __init__(self, noise_dim=32, output_dim=24, cond_dim=0):
        super().__init__()
        input_dim = noise_dim + cond_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z, cond=None):
        if cond is not None:
            z = torch.cat([z, cond], dim=1)
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim=24, cond_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.model(x)


def compute_gradient_penalty(critic, real_data, fake_data, device, cond=None):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    if cond is not None:
        critic_output = critic(interpolated, cond)
    else:
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
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty, grad_norm.mean()

class WGAN(pl.LightningModule):
    def __init__(self, noise_dim=32, output_dim=24, lr=1e-4, n_critic=5, lambda_gp=0.1, pred_file="predictions",num_attack_samples=10000, cond_dim=0, target_model=None, lambda_exploit=0.0, condkl_gender_col: int = 3, condkl_occupation_col: int = 5, condkl_num_occupations: int = 21, condkl_age_col: int = 4, condkl_age_bins: int = 30):
        super().__init__()
        self.save_hyperparameters()
        self.cond_dim = cond_dim
        self.generator = Generator(noise_dim, output_dim, cond_dim)
        self.critic = Critic(output_dim, cond_dim)
        self.automatic_optimization = False
        self.kl_real = []
        self.kl_fake = []
        self.latent_dim = noise_dim
        self.hparams.pred_file = pred_file
        self.num_attack_samples = num_attack_samples
        self.target_model = target_model
        self.lambda_exploit = lambda_exploit
        self.condkl_gender_col = int(condkl_gender_col)
        self.condkl_occupation_col = int(condkl_occupation_col)
        self.condkl_num_occupations = int(condkl_num_occupations)
        self.condkl_age_col = int(condkl_age_col)
        self.condkl_age_bins = int(condkl_age_bins)
        os.makedirs("wgan_samples", exist_ok=True)

        # Freeze target model weights if provided (but allow backprop into fake_g)
        if self.target_model is not None:
            try:
                for p in self.target_model.parameters():
                    p.requires_grad_(False)
                self.target_model.eval()
            except Exception:
                pass

    def forward(self, z, cond=None):
        return self.generator(z, cond)

    def sample(self, z, cond=None):
        """Generate clamped samples in [0,1] for evaluation/exports."""
        return self.generator(z, cond).clamp(0, 1)

    def training_step(self, batch, batch_idx):
        if self.cond_dim > 0:
            real, cond = batch
            cond = cond.float()
        else:
            real = batch.float()
            cond = None
        opt_g, opt_c = self.optimizers()
        # Critic update(s)
        for _ in range(self.hparams.n_critic):
            z = torch.randn(real.size(0), self.hparams.noise_dim, device=self.device)
            fake_c = self(z, cond).detach()  # Detach to avoid gradients to generator
            real_detached = real.detach()
            fake_c = fake_c.detach()
            real_score = self.critic(real_detached, cond)
            fake_score = self.critic(fake_c, cond)
            gp, gn = compute_gradient_penalty(self.critic, real_detached, fake_c, self.device, cond)
            loss_c = fake_score.mean() - real_score.mean() + self.hparams.lambda_gp * gp
            opt_c.zero_grad()
            self.manual_backward(loss_c, retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            opt_c.step()
            self.log("critic_loss", loss_c, prog_bar=True)
            self.log("grad_norm", gn, prog_bar=True)
        # Generator update: use a fresh forward pass for fake with new noise
        z = torch.randn(real.size(0), self.hparams.noise_dim, device=self.device)
        fake_g = self(z, cond)
        loss_g = -self.critic(fake_g, cond).mean()
        if self.target_model is not None and self.lambda_exploit > 0.0:
            with torch.no_grad():
                pred = self.target_model(fake_g)
            if pred.dim() == 2 and pred.size(1) > 1:
                p = torch.softmax(pred, dim=1)
                entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                loss_g = loss_g + self.lambda_exploit * (-entropy)
            else:
                # Single-score target: push away from indecision as a simple proxy
                loss_g = loss_g - self.lambda_exploit * pred.view(-1).abs().mean()
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        opt_g.step()
        self.log("gen_loss", loss_g, prog_bar=True)
        # collect for KL
        if batch_idx == 0:
            self.kl_real.append(real.detach().cpu().numpy())
            self.kl_fake.append(fake_g.detach().cpu().numpy())



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

        # Detect candidate attributes: gender, occupation, and any other columns with integer/categorical values (not the predicted rating column)
        attr_cols = []
        attr_names = []
        # Always include gender and occupation if configured
        if hasattr(self, "condkl_gender_col") and 0 <= self.condkl_gender_col < real_arr.shape[1]:
            attr_cols.append(self.condkl_gender_col)
            attr_names.append("gender")
        if hasattr(self, "condkl_occupation_col") and 0 <= self.condkl_occupation_col < real_arr.shape[1]:
            if self.condkl_occupation_col not in attr_cols:
                attr_cols.append(self.condkl_occupation_col)
                attr_names.append("occupation")
        # Add any other attributes: integer/categorical columns, skip predicted rating
        for col in range(real_arr.shape[1]):
            if col == real_arr.shape[1] - 1:  # skip predicted rating
                continue
            if col in attr_cols:
                continue
            # Heuristic: if all values are close to int, and few unique values, treat as categorical
            vals = np.concatenate([real_arr[:, col], fake_arr[:, col]])
            uniq = np.unique(np.rint(vals))
            if len(uniq) <= 20 and np.allclose(vals, np.rint(vals), atol=1e-2):
                attr_cols.append(col)
                attr_names.append(f"attr{col}")
            # Or, if continuous (float), but not too many unique values, treat as categorical
            elif len(np.unique(vals)) <= 20:
                attr_cols.append(col)
                attr_names.append(f"attr{col}")
            # Otherwise, if not, treat as continuous (will plot as binned)
            elif np.issubdtype(vals.dtype, np.floating):
                attr_cols.append(col)
                attr_names.append(f"attr{col}")

        # Add predicted rating as last plot
        n_attrs = len(attr_cols) + 1
        from math import ceil
        n_cols = min(3, n_attrs)
        n_rows = ceil(n_attrs / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axs = np.array(axs).reshape(-1)  # flatten for easy indexing
        plot_idx = 0

        # Plot predicted rating distribution (last column)
        real_r = real_arr[:, -1]
        fake_r = fake_arr[:, -1]
        low, high = float(real_r.min()), float(real_r.max())
        if np.isclose(low, high):
            low -= 1e-3
            high += 1e-3
        bins = np.linspace(low, high, 51)
        pr, _ = np.histogram(real_r, bins=bins, density=False)
        pf, _ = np.histogram(fake_r, bins=bins, density=False)
        eps = 1e-8
        pr = (pr + eps) / (pr + eps).sum()
        pf = (pf + eps) / (pf + eps).sum()
        kl_pred = np.sum(pr * np.log(pr / pf))
        centers = 0.5*(bins[:-1] + bins[1:])
        ax = axs[plot_idx]
        width = bins[1]-bins[0]
        ax.bar(centers, pr, width=width, alpha=0.6, label="Real")
        ax.bar(centers, pf, width=width, alpha=0.6, label="Fake")
        ax.set_title(f"KL(pred_rating) = {kl_pred:.3f}")
        ax.legend()
        ax.set_xlabel("Predicted Rating")
        ax.set_ylabel("Probability")
        plot_idx += 1

        # For each attribute, plot on a subplot
        for idx, col in enumerate(attr_cols):
            colname = attr_names[idx]
            real_vals = real_arr[:, col]
            fake_vals = fake_arr[:, col]
            ax = axs[plot_idx]
            # Special handling for gender
            if hasattr(self, "condkl_gender_col") and col == self.condkl_gender_col:
                # Binarize: 0/1
                real_v = (real_vals >= 0.5).astype(int)
                fake_v = (fake_vals >= 0.5).astype(int)
                bins = np.arange(-0.5, 2, 1)  # bins for 0 and 1
                pr, _ = np.histogram(real_v, bins=bins, density=False)
                pf, _ = np.histogram(fake_v, bins=bins, density=False)
                pr = (pr + eps) / (pr + eps).sum()
                pf = (pf + eps) / (pf + eps).sum()
                kl = np.sum(pr * np.log(pr / pf))
                x = np.arange(2)
                width = 0.45
                ax.bar(x - width/2, pr, width, alpha=0.6, label="Real")
                ax.bar(x + width/2, pf, width, alpha=0.6, label="Fake")
                ax.set_title(f"KL({colname}) = {kl:.3f}")
                ax.set_xlabel("Gender")
                ax.set_ylabel("Probability")
                ax.set_xticks(x)
                ax.set_xticklabels(["Male", "Female"])
                ax.legend()
                plot_idx += 1
                continue
            # Special handling for occupation
            if hasattr(self, "condkl_occupation_col") and col == self.condkl_occupation_col:
                n_occ = int(getattr(self, "condkl_num_occupations", 0)) or int(np.nanmax(np.concatenate([real_vals, fake_vals])) + 1)
                # Map to int IDs
                def _occ_ids(arr, n_occ):
                    arr = np.asarray(arr)
                    if arr.size == 0:
                        return np.zeros((0,), dtype=int)
                    maxv = float(np.max(arr))
                    if maxv <= 1.5:
                        ids = np.rint(np.clip(arr, 0.0, 1.0) * (n_occ - 1)).astype(int)
                    else:
                        ids = np.rint(np.clip(arr, 0.0, n_occ - 1)).astype(int)
                    return ids
                real_ids = _occ_ids(real_vals, n_occ)
                fake_ids = _occ_ids(fake_vals, n_occ)
                bins = np.arange(-0.5, n_occ + 0.5, 1)
                pr, _ = np.histogram(real_ids, bins=bins, density=False)
                pf, _ = np.histogram(fake_ids, bins=bins, density=False)
                pr = (pr + eps) / (pr + eps).sum()
                pf = (pf + eps) / (pf + eps).sum()
                kl = np.sum(pr * np.log(pr / pf))
                x = np.arange(n_occ)
                width = 0.45
                ax.bar(x - width/2, pr, width, alpha=0.6, label="Real")
                ax.bar(x + width/2, pf, width, alpha=0.6, label="Fake")
                ax.set_title(f"KL({colname}) = {kl:.3f}")
                ax.set_xlabel("Occupation ID")
                ax.set_ylabel("Probability")
                ax.set_xticks(x)
                ax.legend()
                plot_idx += 1
                continue
            # For other attributes, determine if categorical or continuous
            vals = np.concatenate([real_vals, fake_vals])
            uniq = np.unique(np.rint(vals))
            # Categorical/integer: <=20 unique values and all close to int
            if len(uniq) <= 20 and np.allclose(vals, np.rint(vals), atol=1e-2):
                n_cat = int(uniq.max()) + 1
                bins = np.arange(-0.5, n_cat + 0.5, 1)
                pr, _ = np.histogram(np.rint(real_vals), bins=bins, density=False)
                pf, _ = np.histogram(np.rint(fake_vals), bins=bins, density=False)
                pr = (pr + eps) / (pr + eps).sum()
                pf = (pf + eps) / (pf + eps).sum()
                kl = np.sum(pr * np.log(pr / pf))
                x = np.arange(n_cat)
                width = 0.45
                ax.bar(x - width/2, pr, width, alpha=0.6, label="Real")
                ax.bar(x + width/2, pf, width, alpha=0.6, label="Fake")
                ax.set_title(f"KL({colname}) = {kl:.3f}")
                ax.set_xlabel(f"{colname} (categorical)")
                ax.set_ylabel("Probability")
                ax.set_xticks(x)
                ax.legend()
                plot_idx += 1
            else:
                # Treat as continuous: use binning
                low, high = float(min(real_vals.min(), fake_vals.min())), float(max(real_vals.max(), fake_vals.max()))
                if np.isclose(low, high):
                    low -= 1e-3; high += 1e-3
                bins = np.linspace(low, high, 31)
                pr, _ = np.histogram(real_vals, bins=bins, density=False)
                pf, _ = np.histogram(fake_vals, bins=bins, density=False)
                pr = (pr + eps) / (pr + eps).sum()
                pf = (pf + eps) / (pf + eps).sum()
                kl = np.sum(pr * np.log(pr / pf))
                centers = 0.5 * (bins[:-1] + bins[1:])
                width = bins[1]-bins[0]
                ax.bar(centers, pr, width=width, alpha=0.6, label="Real")
                ax.bar(centers, pf, width=width, alpha=0.6, label="Fake")
                ax.set_title(f"KL({colname}) = {kl:.3f}")
                ax.set_xlabel(f"{colname} (binned)")
                ax.set_ylabel("Probability")
                ax.legend()
                plot_idx += 1

        # Hide any unused axes
        for j in range(plot_idx, len(axs)):
            axs[j].axis('off')
        plt.tight_layout()
        plt.savefig(f"wgan_samples/{self.hparams.pred_file}_all_attrs.png")
        plt.close()

        # Evaluate attribute inference attacks
        if self.condkl_gender_col >= 0 and self.condkl_gender_col < real_arr.shape[1]:
            self.evaluate_attribute_attack(real_arr, fake_arr, self.condkl_gender_col, target_names=["Male", "Female"])
        if self.condkl_occupation_col >= 0 and self.condkl_occupation_col < real_arr.shape[1]:
            n_occ = self.condkl_num_occupations if self.condkl_num_occupations > 0 else None
            target_names = [f"Occ_{i}" for i in range(n_occ)] if n_occ else None
            self.evaluate_attribute_attack(real_arr, fake_arr, self.condkl_occupation_col, target_names=target_names)

    def evaluate_attribute_attack(self, real_arr, fake_arr, attr_col, target_names):
        """
        Trains a classifier on fake data to predict sensitive attribute at attr_col,
        then evaluates on real data to simulate black-box attribute inference attack.
        Logs metrics and saves confusion matrix plot.
        """
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        # Prepare data for training classifier
        X_fake = np.delete(fake_arr, attr_col, axis=1)
        y_fake = fake_arr[:, attr_col]

        X_real = np.delete(real_arr, attr_col, axis=1)
        y_real = real_arr[:, attr_col]

        # Discretize/binarize known categorical attributes produced as floats by the WGAN
        # Gender column: threshold at 0.5
        if attr_col == getattr(self, 'condkl_gender_col', -1):
            y_fake_int = (y_fake >= 0.5).astype(int)
            y_real_int = (y_real >= 0.5).astype(int)
        # Occupation column: map from scaled float [0,1] or raw id to integer IDs in [0, n_occ-1]
        elif attr_col == getattr(self, 'condkl_occupation_col', -1):
            n_occ = int(getattr(self, 'condkl_num_occupations', 0)) or int(max(2, np.nanmax(np.concatenate([y_fake, y_real])) + 1))
            def _occ_ids(arr):
                arr = np.asarray(arr)
                if arr.size == 0:
                    return np.zeros((0,), dtype=int)
                maxv = float(np.nanmax(arr))
                if maxv <= 1.5:
                    ids = np.rint(np.clip(arr, 0.0, 1.0) * (n_occ - 1)).astype(int)
                else:
                    ids = np.rint(np.clip(arr, 0.0, n_occ - 1)).astype(int)
                return ids
            y_fake_int = _occ_ids(y_fake)
            y_real_int = _occ_ids(y_real)
        else:
            # Generic heuristic: treat as discrete if few unique integer-like values; otherwise skip (e.g., age)
            y_fake_int = None
            y_real_int = None
            if np.issubdtype(y_fake.dtype, np.floating):
                unique_vals = np.unique(y_fake)
                if len(unique_vals) <= 20 and np.allclose(unique_vals, unique_vals.astype(int)):
                    y_fake_int = y_fake.astype(int)
                    y_real_int = y_real.astype(int)
                else:
                    print(f"Attribute at col {attr_col} appears continuous. Skipping attribute attack.")
                    return
            else:
                y_fake_int = y_fake.astype(int)
                y_real_int = y_real.astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X_fake, y_fake_int, test_size=0.2, random_state=42, stratify=y_fake_int)

        # Choose classifier
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            clf = LogisticRegression(max_iter=200)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)

        # Train classifier on fake data
        clf.fit(X_train, y_train)

        # Evaluate on real data
        y_pred = clf.predict(X_real)

        # Compute metrics
        accuracy = accuracy_score(y_real_int, y_pred)
        precision = precision_score(y_real_int, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_real_int, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_real_int, y_pred, average='weighted', zero_division=0)
        # ROC-AUC only for binary
        roc_auc = None
        if n_classes == 2:
            try:
                y_proba = clf.predict_proba(X_real)[:,1]
                roc_auc = roc_auc_score(y_real_int, y_proba)
            except Exception:
                roc_auc = None

        # Log metrics
        self._log_metric(f"attr_attack_acc_col{attr_col}", accuracy)
        self._log_metric(f"attr_attack_precision_col{attr_col}", precision)
        self._log_metric(f"attr_attack_recall_col{attr_col}", recall)
        self._log_metric(f"attr_attack_f1_col{attr_col}", f1)
        if roc_auc is not None:
            self._log_metric(f"attr_attack_roc_auc_col{attr_col}", roc_auc)

        print(f"Attribute attack on column {attr_col} results:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_real_int, y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        if target_names is not None:
            classes = target_names
        else:
            classes = [str(i) for i in range(n_classes)]
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label',
               title=f'Confusion Matrix for Attr Col {attr_col}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(f"wgan_samples/{self.hparams.pred_file}_attr_attack.png")
        plt.close()

    def _log_metric(self, key, value):
        """
        Helper for logging metrics to the appropriate logger.
        """
        if self.logger is None:
            return
        logger_name = self.logger.__class__.__name__.lower()
        if logger_name.startswith("wandb"):
            import wandb
            wandb.log({key: value})
        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_scalar"):
            self.logger.experiment.add_scalar(key, value)
        else:
            print(f"[metric] {key}: {value}")

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.0, 0.9))
        return [opt_g, opt_c], []