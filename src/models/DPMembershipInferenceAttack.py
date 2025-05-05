import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any
from .DPModel import DPModel
from sklearn.metrics import roc_auc_score
import torch.nn as nn

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
        kwargs["enable_dp"] = True
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        # Expecting batch = (user_ids, item_ids, targets, label)
        user_ids, item_ids, targets, label = batch

        # Normalize ratings into [0,1]
        targets_norm = (targets - 1.0) / 4.0

        # Forward pass through your DP‐protected model
        logits = self(user_ids, item_ids)
        probs = torch.sigmoid(logits).squeeze()  # shape: (batch_size,)

        # Use model confidence (or any other feature) as attack score
        # Here, distance from true normalized rating
        conf = (1.0 - torch.abs(probs - targets_norm)).tolist()  # list of floats

        # Flatten batch of labels (0 or 1)
        labels = label.tolist()  # list of ints

        # Accumulate
        self.scores.extend(conf)
        self.labels.extend(labels)

    def on_test_epoch_end(self):
        if not self.scores or not self.labels:
            print("⚠️ No scores or labels collected—skipping AUC")
            return

        # Compute single ROC-AUC over all member/nonmember examples
        auc = roc_auc_score(self.labels, self.scores)
        self.log("mia_auc", auc, prog_bar=True)

        # Reset for next run
        self.scores.clear()
        self.labels.clear()

    @classmethod
    def load_dp_checkpoint(cls, checkpoint_path, **kwargs):
        # Load Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams    = checkpoint["hyper_parameters"]

        # Instantiate model
        model = cls(**hparams, **kwargs)

        # Rebuild the internal DP module
        model.dp_model = CoreDPModule(
            model.user_embedding,
            model.item_embedding,
            model.dropout_rate,
            model.fc
        )

        # Extract and remap only the dp_model weights
        dp_state = {
            k.replace("dp_model._module.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("dp_model._module.")
        }
        model.dp_model.load_state_dict(dp_state, strict=True)
        return model
