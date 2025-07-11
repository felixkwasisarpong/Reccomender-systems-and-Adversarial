import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError

class AttributeClassifier(pl.LightningModule):
    def __init__(
        self,
        hidden_dims=[128, 64],
        lr=1e-3,
        target_attr="gender",
        pred_file="custom_mid",
        dropout_rate=0.0,
        input_dim=None,  # Dynamically computed if not provided
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_attr = target_attr
        self.pred_file = pred_file

        # Dynamically compute input_dim if not provided
        if input_dim is None:
            input_dim = 1 + 1 + 1 + 19  # user_id, item_id, rating, genre
            if self.target_attr != "occupation":
                input_dim += 1  # occupation
            if self.target_attr != "age":
                input_dim += 1  # age
            if self.target_attr != "gender":
                input_dim += 1  # gender
        self.input_dim = input_dim

        output_dim = 1 if target_attr == "age" else self._get_output_dim()

        # Build MLP with dropout
        layers = [nn.Linear(self.input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(hidden_dims[-1], output_dim)]
        self.model = nn.Sequential(*layers)

        # Loss and metric
        if target_attr == "age":
            self.loss_fn = nn.L1Loss()
            self.metric = MeanAbsoluteError()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric = Accuracy(task="multiclass", num_classes=output_dim)

    def _get_output_dim(self):
        return {"gender": 2, "occupation": 21}[self.target_attr]

    def forward(self, x):
        return self.model(x)

    def _prepare_features(self, batch):
        features = [
            batch['user_id'].float().unsqueeze(1),      # already in [0, 1]
            batch['item_id'].float().unsqueeze(1),      # already in [0, 1]
            batch['rating'].float().unsqueeze(1),       # already in [0, 1]
        ]

        if self.target_attr != "age":
            features.append(batch['age'].unsqueeze(1))          # already in [0, 1]
        if self.target_attr != "occupation":
            features.append(batch['occupation'].float().unsqueeze(1))  # already in [0, 1]
        if self.target_attr != "gender":
            features.append(batch['gender'].float().unsqueeze(1))      # already in {0, 1}

        features.append(batch['genre'])  # 1-hot vector, already in [0, 1]

        return torch.cat(features, dim=1)

    def _get_target(self, batch):
        return batch["age"] if self.target_attr == "age" else batch[self.target_attr]

    def _compute_metric(self, logits, y):
        if self.target_attr == "age":
            return self.metric(logits * 100, y * 100)  # interpret as years
        return self.metric(F.softmax(logits, dim=-1), y)

    def training_step(self, batch, batch_idx):
        x, y = self._prepare_features(batch), self._get_target(batch)
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        metric = self._compute_metric(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log(f"train_{self.target_attr}_metric", metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._prepare_features(batch), self._get_target(batch)
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        metric = self._compute_metric(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_{self.target_attr}_metric", metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = self._prepare_features(batch), self._get_target(batch)
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        metric = self._compute_metric(logits, y)

        self.log("test_loss", loss)
        self.log(f"test_{self.target_attr}_metric", metric, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)