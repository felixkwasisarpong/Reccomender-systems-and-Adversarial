import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any
from .DPModel import DPModel
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import matplotlib.pyplot as plt
from models.CustomDP_SGD import CustomDP_SGD


class CustomMembershipInferenceAttack(CustomDP_SGD):
    def __init__(self, *args, **kwargs):
        kwargs["enable_dp"] = False
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        user_ids, item_ids, targets, label = batch  # now label is batch of 0/1
        # normalize target rating to [0,1]
        targets_norm = (targets - 1.0) / 4.0
        
        logits = self(user_ids, item_ids)
        probs = torch.sigmoid(logits).squeeze()  # shape: (batch_size,)
        
        # Use probs directly as “confidence” scores
        # If you want distance from normalized target, you can do:
        # conf = (1.0 - torch.abs(probs - targets_norm)).tolist()
        conf = probs.tolist()  # list of floats length batch_size
        
        # label is a tensor of shape (batch_size,), we want a list of ints
        labels = label.tolist()

        # extend, not append
        self.scores.extend(conf)
        self.labels.extend(labels)

    def on_test_epoch_end(self):
        if not self.scores:
            return

        auc = roc_auc_score(self.labels, self.scores)
        self.log("mia_auc", auc, prog_bar=True)
        fpr, tpr, _ = roc_curve(self.labels, self.scores)
         # Plot ROC Curve
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Membership Inference Attack ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        # Save to file (optional: change path if needed)
        plt.savefig("mia_roc_curve.png")
        plt.close()
        self.scores.clear()
        self.labels.clear()
