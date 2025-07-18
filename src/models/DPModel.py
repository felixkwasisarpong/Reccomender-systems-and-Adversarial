import time
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from opacus import PrivacyEngine
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .BaseModel import BaseModel  # adjust import
from torch.optim import Adam

class DPModel(BaseModel):
    def __init__(
        self,
        # —— all your BaseModel args —— 
        num_users: int,
        num_items: int,
        num_genders: int,
        num_occupations: int,
 
        genre_dim: int,
        embed_dim: int = 16,
        mlp_dims: list = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        l2_penalty: float = 1e-4,
        loss_function: str = "Huber",
        target_min: float = 1.0,
        target_max: float = 5.0,
        use_attrs: bool = True,
        predict_file: str = "predictions",
        # —— DP-specific args ——
        enable_dp: bool = False,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        **kwargs
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            num_genders=num_genders,
            num_occupations=num_occupations,
            genre_dim=genre_dim,
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
            loss_function=loss_function,
            target_min=target_min,
            target_max=target_max,
            predict_file=predict_file,
            use_attrs=use_attrs,
            **kwargs
        )
        self.enable_dp      = enable_dp
        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta
        self.max_grad_norm  = max_grad_norm
        self.privacy_engine = PrivacyEngine()
        self.epsilon_history = []
        self.metric_history  = []
        self.privacy_engine = PrivacyEngine(accountant='rdp')


    def on_train_start(self):
        if not self.enable_dp:
            return

        train_loader = self.trainer.datamodule.train_dataloader()
        optimizer = self.trainer.optimizers[0]

        # Wrap self in-place (do not assign back to self)
        _, dp_optimizer, _ = self.privacy_engine.make_private_with_epsilon(
            module=self,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=self.trainer.max_epochs,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
        )

        self.trainer.optimizers = [dp_optimizer]

    def forward(self, batch):
        return super().forward(batch)
    



    def on_train_epoch_end(self):
        if self.enable_dp:
            try:
                eps = self.privacy_engine.get_epsilon(self.target_delta)
                rmse = self.trainer.callback_metrics.get("val_rmse")

                # Only log when both epsilon and val_rmse are available
                if rmse is not None:
                    self.epsilon_history.append(eps)
                    self.metric_history.append(rmse.cpu().item())
                    self.log("epsilon", eps, prog_bar=True)
            except Exception as e:
                print(f"[Warning] Failed to log ε or RMSE: {e}")
        super().on_train_epoch_end()


    def on_train_end(self):
        # final privacy–utility curve
        if self.enable_dp and self.epsilon_history:
            plt.figure(figsize=(6,4))
            plt.plot(self.epsilon_history, self.metric_history, 'o-', alpha=0.8)
            plt.xlabel("ε"); plt.ylabel("Validation RMSE")
            plt.title("Privacy–Utility Trade-off")
            plt.grid(True)
            fname = f"dp_tradeoff_{self.hparams.predict_file}.png"
            plt.savefig(fname, dpi=300)
            plt.close()