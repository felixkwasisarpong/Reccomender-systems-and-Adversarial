import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        num_users,
        num_items,
        num_genders,
        num_occupations,
        genre_dim,
        embed_dim=16,
        mlp_dims=[128, 64],
        dropout=0.2,
        learning_rate=1e-3,
        l2_penalty=1e-4,
        
        loss_function="MSE",
        target_min=1.0,
        target_max=5.0,
        predict_file="predictions",
        use_attrs=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_attrs = use_attrs
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Always define attr layers
        self.gender_embedding = nn.Embedding(num_genders, embed_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embed_dim)
        self.age_projector = nn.Linear(1, embed_dim)
        self.genre_projector = nn.Linear(genre_dim, embed_dim)

        self.num_fields = 2 + (4 if self.use_attrs else 0)
        input_dim = embed_dim * self.num_fields

        self.linear = nn.Linear(input_dim, 1)

        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.output_layer = nn.Linear(1 + 1 + mlp_dims[-1], 1)

        self.loss_fn = nn.MSELoss() if loss_function == "MSE" else nn.SmoothL1Loss()

        for phase in ["train", "val", "test"]:
            setattr(self, f"{phase}_mse", MeanSquaredError())
            setattr(self, f"{phase}_mae", MeanAbsoluteError())
            setattr(self, f"{phase}_rmse", MeanSquaredError(squared=False))

    def forward(self, batch):
        user_embed = self.user_embedding(batch['user_id'])
        item_embed = self.item_embedding(batch['item_id'])
        features = [user_embed, item_embed]

        if self.use_attrs:
            gender_embed = self.gender_embedding(batch['gender'])
            occupation_embed = self.occupation_embedding(batch['occupation'])
            age = batch['age'].unsqueeze(-1) if batch['age'].dim() == 1 else batch['age']
            age_embed = nn.functional.normalize(self.age_projector(age), dim=1)
            genre_embed = nn.functional.normalize(self.genre_projector(batch['genre']), dim=1)

            features += [gender_embed, occupation_embed, age_embed, genre_embed]

        try:
            features = torch.stack(features, dim=1)
        except RuntimeError as e:
            raise RuntimeError(f"Feature stack failed: {[(f.shape if hasattr(f, 'shape') else 'N/A') for f in features]} | {e}")

        B, F, D = features.size()
        linear_term = self.linear(features.view(B, -1))
        summed = features.sum(dim=1)
        squared = features ** 2
        summed_squared = summed ** 2
        square_summed = squared.sum(dim=1)
        bi_interaction = 0.5 * (summed_squared - square_summed).sum(dim=1, keepdim=True)
        mlp_input = features.view(B, -1)
        deep_out = self.mlp(mlp_input)
        concat = torch.cat([linear_term, bi_interaction, deep_out], dim=1)
        preds = self.output_layer(concat).squeeze(-1)
        return torch.clamp(preds, self.hparams.target_min, self.hparams.target_max)

    def _shared_step(self, batch, batch_idx, phase):
        preds = self(batch)
        targets = batch['rating']
        loss = self.loss_fn(preds, targets)
        self.log(f"{phase}_loss", loss, prog_bar=True)
        getattr(self, f"{phase}_mse").update(preds, targets)
        getattr(self, f"{phase}_mae").update(preds, targets)
        getattr(self, f"{phase}_rmse").update(preds, targets)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_mse.compute(), prog_bar=True)
        self.log("train_mae", self.train_mae.compute(), prog_bar=True)
        self.log("train_rmse", self.train_rmse.compute(), prog_bar=True)
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_rmse.reset()

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_rmse", self.val_rmse.compute(), prog_bar=True)
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_rmse.reset()

    def on_test_epoch_end(self):
        self.log("test_mse", self.test_mse.compute(), prog_bar=True)
        self.log("test_mae", self.test_mae.compute(), prog_bar=True)
        self.log("test_rmse", self.test_rmse.compute(), prog_bar=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_rmse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.l2_penalty)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}