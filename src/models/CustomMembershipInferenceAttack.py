import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any
from .DPModel import DPModel
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import matplotlib.pyplot as plt
from .CustomDP_SGD import CustomDP_SGD
from .BaseModel import BaseModel
import inspect

class CustomMembershipInferenceAttack(CustomDP_SGD):
    def __init__(self, *args, **kwargs):
        kwargs["enable_dp"] = False
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []

    def forward(self, user_ids, item_ids, attrs=None):
        batch = {
            "user_id": user_ids,
            "item_id": item_ids,
        }

        if self.use_attrs and attrs is not None:
            gender, age, occupation, genre = attrs
            batch.update({
                "gender": gender,
                "age": age,
                "occupation": occupation,
                "genre": genre,
            })

        return super().forward(batch)

    def test_step(self, batch, batch_idx):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        targets = batch["rating"]
        label = batch["label"]

        if self.use_attrs and all(k in batch for k in ["gender", "age", "occupation", "genre"]):
            attrs = (
                batch["gender"],
                batch["age"],
                batch["occupation"],
                batch["genre"],
            )
            preds = self.forward(user_ids, item_ids, attrs)
        else:
            preds = self.forward(user_ids, item_ids)

        targets_norm = (targets - self.hparams.target_min) / (self.hparams.target_max - self.hparams.target_min)
        probs = torch.sigmoid(preds).squeeze()

        self.scores.extend(probs.tolist())
        self.labels.extend(label.tolist())

    def on_test_epoch_end(self):
        if not self.scores:
            return

        auc = roc_auc_score(self.labels, self.scores)
        fpr, tpr, _ = roc_curve(self.labels, self.scores)

        self.log("mia_auc", auc, prog_bar=True)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Membership Inference Attack")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("mia_roc_curve_custom_weak.png")
        plt.close()

        self.scores.clear()
        self.labels.clear()

    @classmethod
    def load_from_dp_checkpoint(cls, checkpoint_path, **override_kwargs):
        """
        Load a checkpoint saved from CustomDP_SGD into this attack class,
        stripping out any CLI/Hydra metadata so we only pass valid init args.
        """
        # 1) Load the raw checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        raw_hparams = ckpt["hyper_parameters"]

        # 2) The checkpoint hparams are flat dict of all model init args.
        #    Merge in any overrides (e.g., enable_dp=False is already forced in __init__).
        raw_hparams = {**raw_hparams, **override_kwargs}

        # 3) Determine which keys CustomDP_SGD.__init__ and BaseModel.__init__ actually accept
        sig_dp = inspect.signature(CustomDP_SGD.__init__)
        sig_base = inspect.signature(BaseModel.__init__)
        allowed = set(sig_dp.parameters) | set(sig_base.parameters)
        allowed.discard("self")

        # 4) Filter out any extraneous keys
        init_kwargs = {k: v for k, v in raw_hparams.items() if k in allowed}

        # 5) Instantiate the attack model and load weights
        model = cls(**init_kwargs)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model