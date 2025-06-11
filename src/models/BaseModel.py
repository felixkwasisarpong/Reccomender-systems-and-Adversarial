import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        loss_function: str = "RMSE",
        l2_penalty: float = 1e-4,
        dropout_rate: float = 0.2,
        enable_dp: bool = False,
    ):
        """
        Base recommendation model with user/item embeddings and MLP.
        
        Args:
            num_users: Number of unique users in the dataset
            num_items: Number of unique items in the dataset
            embed_dim: Dimension of user/item embeddings
            learning_rate: Learning rate for optimizer
            loss_function: One of "RMSE", "MSE", or "Huber"
            l2_penalty: L2 regularization weight
            dropout_rate: Dropout probability
            enable_dp: Flag for differential privacy
        """
        super().__init__()
        self.save_hyperparameters()
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)


        # Network architecture
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        self.embed_dropout = nn.Dropout(dropout_rate/2)

        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Initialize metrics
        self._init_metrics()

        # Loss function
        self.loss_fn = self._get_loss_function(loss_function)

    def _init_metrics(self):
        """Initialize metrics for train/val/test"""



                # MSE
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        
        # MAE
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        
        # RMSE (using MSE with squared=False)
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

    def _get_loss_function(self, loss_function: str):
        """Return the appropriate loss function"""
        if loss_function == "MSE":
            return nn.MSELoss(reduction="none")
        elif loss_function == "Huber":
            return nn.SmoothL1Loss(reduction="none")
        elif loss_function == "RMSE":
            return lambda pred, target: torch.sqrt(nn.MSELoss(reduction="none")(pred, target))
        raise ValueError(f"Unsupported loss function: {loss_function}")
    


    def forward(self, user_ids, item_ids):
        user_emb = self.embed_dropout(self.user_embedding(user_ids))
        item_emb = self.embed_dropout(self.item_embedding(item_ids))

        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.fc(x)  # [batch_size, 1]
        
        user_b = self.user_bias(user_ids)  # [batch_size, 1]
        item_b = self.item_bias(item_ids)  # [batch_size, 1]
        
        return (x + user_b + item_b).squeeze(-1)


    def _shared_step(self, batch, batch_idx, prefix: str):
        user_ids, item_ids, targets = batch
        
        # Normalize targets consistently
        targets_norm = (targets - 1.0) / 4.0
        
        y_pred = self(user_ids, item_ids)
        loss = self.loss_fn(y_pred, targets_norm).mean()
        
        # Denormalize predictions and clamp to valid range
        preds_denorm = torch.clamp(y_pred * 4.0 + 1.0, 1.0, 5.0)
        
        # Use original targets for metrics
        self._update_metrics(preds_denorm, targets, prefix)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        
        return loss

        
    def _update_metrics(self, preds, targets, prefix: str):
        """Update only MSE, MAE, RMSE (no R²)."""
        getattr(self, f"{prefix}_mse").update(preds, targets)
        getattr(self, f"{prefix}_mae").update(preds, targets)
        getattr(self, f"{prefix}_rmse").update(preds, targets)
        
    def _log_epoch_metrics(self, prefix: str):

        self.log(f"{prefix}_mse", getattr(self, f"{prefix}_mse").compute())
        self.log(f"{prefix}_mae", getattr(self, f"{prefix}_mae").compute())
        self.log(f"{prefix}_rmse", getattr(self, f"{prefix}_rmse").compute())

    def _reset_metrics(self, prefix: str):
        """Reset TorchMetrics and prediction storage."""
        getattr(self, f"{prefix}_mse").reset()
        getattr(self, f"{prefix}_mae").reset()
        getattr(self, f"{prefix}_rmse").reset()
 

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")
        self._reset_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")
        self._reset_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
        self._reset_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_penalty
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10  # Number of epochs before restarting the cycle
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # update scheduler every epoch
            }
        }
