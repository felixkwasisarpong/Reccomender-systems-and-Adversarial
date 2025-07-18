import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any
from .DPModel import DPModel
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os

class DPMembershipInferenceAttack(DPModel):
    def __init__(self, *args, **kwargs):
        kwargs["enable_dp"] = False  # Disable DP during attack evaluation
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        targets = batch["rating"]
        labels = batch["label"]

        # Normalize targets to [0, 1]
        target_min = self.hparams.target_min
        target_max = self.hparams.target_max
        targets_norm = (targets - target_min) / (target_max - target_min)

        # Compute per-sample errors
        errors = torch.abs(preds - targets_norm)
        scores = 1.0 - errors  # Higher score → more likely to be a member

        self.scores.extend(scores.tolist())
        self.labels.extend(labels.tolist())

    def on_test_epoch_end(self):
        if not self.scores:
            return

        auc = roc_auc_score(self.labels, self.scores)
        self.log("mia_auc", auc, prog_bar=True)

        fpr, tpr, _ = roc_curve(self.labels, self.scores)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Membership Inference Attack ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        # Determine output directory
        out_dir = getattr(self, "output_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"mia_roc_curve_opacus_strong.png")
        plt.savefig(fig_path)
        plt.close()

        self.scores.clear()
        self.labels.clear()