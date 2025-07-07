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

class DPMembershipInferenceAttack(DPModel):
    def __init__(self, *args, **kwargs):
        kwargs["enable_dp"] = False  # Disable DP during inference
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits).squeeze()
        
        self.scores.extend(probs.tolist())
        self.labels.extend(batch['label'].tolist())



    def on_test_epoch_end(self):
        if not self.scores:
            return

        auc = roc_auc_score(self.labels, self.scores)
        self.log("mia_auc", auc, prog_bar=True)

        fpr, tpr, _ = roc_curve(self.labels, self.scores)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc:.2f})")  # colored ROC
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")  # baseline
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Membership Inference Attack ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("mia_roc_curve_mid.png")
        plt.close()

        self.scores.clear()
        self.labels.clear()
