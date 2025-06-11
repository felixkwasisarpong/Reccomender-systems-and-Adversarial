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

class CoreDPModule(nn.Module):
    def __init__(self, user_embedding, item_embedding, dropout, fc):
        super().__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.dropout = nn.Dropout(dropout)
        self.fc = fc

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze()


class DPMembershipInferenceAttack(DPModel):
    def __init__(self, *args, **kwargs):
        # Ensure DP is enabled in the underlying model
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


    @classmethod
    def load_dp_checkpoint(cls, checkpoint_path, **kwargs):
        # Load Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = checkpoint["hyper_parameters"]

        # Instantiate model
        model = cls(**hparams, **kwargs)

        # Rebuild the internal DP module
        model.dp_model = CoreDPModule(
            model.user_embedding,
            model.item_embedding,
            model.dropout_rate,
            model.fc
        )

        # Debugging: Print keys in the checkpoint state_dict
        print("Checkpoint keys:", checkpoint["state_dict"].keys())

        # Extract and remap only the dp_model weights
        dp_state = {
            k.replace("dp_model._module.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("dp_model._module.")
        }

        # Debugging: Print remapped keys
        print("Remapped keys:", dp_state.keys())

        # Load state_dict into dp_model
        try:
            model.dp_model.load_state_dict(dp_state, strict=True)
        except RuntimeError as e:
            print("Error loading state_dict:", e)
            raise

        return model
