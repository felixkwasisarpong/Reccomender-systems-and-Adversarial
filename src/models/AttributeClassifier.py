import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pytorch_lightning as pl
MAX_AGE_YEARS = 100.0
try:
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
except Exception:
    TensorBoardLogger = None
    try:
        from pytorch_lightning.loggers import CSVLogger
    except Exception:
        CSVLogger = None
from torchmetrics import Accuracy, MeanAbsoluteError

import numpy as np
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        mean_absolute_error,
        r2_score,
        f1_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
except Exception:
    roc_auc_score = average_precision_score = accuracy_score = mean_absolute_error = r2_score = f1_score = balanced_accuracy_score = confusion_matrix = None

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
        use_identifiers=False,      # drop user_id/item_id by default to avoid trivial leakage
        include_rating=True,        # allow excluding rating if desired
        strict_feature_check=True,  # sanity guard against accidental leakage
        exclude_features: list | None = None,
        use_batchnorm: bool = False,  # NEW: optional BatchNorm1d after each linear
        scheduler: str = "none",      # NEW: 'none' or 'cosine'
        use_class_weight: bool = False,
        focal_gamma: float = 0.0,
        input_layernorm: bool = False,
        age_as_buckets: bool = False,
        age_bucket_values: list | None = None,
        genre_dim: int = 19,

        # NEW: accept sweep targets so YAML validation passes
        targets: list | None = None,

    
    ):
        super().__init__()
        self.save_hyperparameters()
        self.genre_dim = int(genre_dim)
        self.input_dim = input_dim
        # Internal: builder so we can rebuild with a new input_dim at runtime
        def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = []
            if self.hparams.input_layernorm:
                layers += [nn.LayerNorm(in_dim)]
            layers += [nn.Linear(in_dim, hidden_dims[0])]
            if self.hparams.use_batchnorm:
                layers += [nn.BatchNorm1d(hidden_dims[0])]
            layers += [nn.ReLU(), nn.Dropout(dropout_rate)]
            for i in range(len(hidden_dims) - 1):
                layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1])]
                if self.hparams.use_batchnorm:
                    layers += [nn.BatchNorm1d(hidden_dims[i + 1])]
                layers += [nn.ReLU(), nn.Dropout(dropout_rate)]
            layers += [nn.Linear(hidden_dims[-1], out_dim)]
            return nn.Sequential(*layers)
        self._make_mlp = _make_mlp
        self.target_attr = target_attr
        self.pred_file = pred_file
        self.use_identifiers = use_identifiers
        self.include_rating = include_rating
        self.strict_feature_check = strict_feature_check
        # Allow users to drop specific features from the auditor (e.g., to bypass bad columns temporarily)
        self.exclude_features = set(exclude_features or [])
        self.targets = targets
        # Age bucketing config
        self._age_as_buckets = bool(age_as_buckets)
        default_buckets = [1., 18., 25., 35., 45., 50., 56.]
        _ages = torch.tensor(age_bucket_values, dtype=torch.float32) if age_bucket_values is not None else torch.tensor(default_buckets, dtype=torch.float32)
        self.register_buffer("age_bucket_values", _ages, persistent=False)
        self._n_age_classes = int(self.age_bucket_values.numel())

        if input_dim is None:
            input_dim = 0
            if self.include_rating and self._use_feature("rating"):
                input_dim += 1
            if self.use_identifiers:
                if self._use_feature("user_id"):
                    input_dim += 1
                if self._use_feature("item_id"):
                    input_dim += 1
            if self._use_feature("age"):
                input_dim += 1
            if self._use_feature("occupation"):
                input_dim += 1
            if self._use_feature("gender"):
                input_dim += 1
            if (self._use_feature("genre") and self.target_attr != "genre"):
                input_dim += int(self.genre_dim)
        self.input_dim = input_dim

        if self.strict_feature_check and self.use_identifiers:
            warnings.warn(
                f"AttributeClassifier: use_identifiers=True; IDs can trivially reveal {self.target_attr}. "
                "Set use_identifiers=False for a fair leakage audit.",
                RuntimeWarning,
            )

        if target_attr == "age":
            output_dim = (self._n_age_classes if self._age_as_buckets else 1)
        else:
            output_dim = self._get_output_dim()

        # Build initial MLP; may be rebuilt in on_fit_start if dataset feature dim differs
        self.model = self._make_mlp(self.input_dim, output_dim)

        # Buffers for richer test-time leakage metrics
        self._test_targets = []
        self._test_preds = []
        self._test_probs = []

        # Buffers for validation-time threshold tuning (binary)
        self._val_targets = []
        self._val_probs = []
        self._best_threshold = 0.5

        # Loss and metric
        if target_attr == "age":
            if self._age_as_buckets:
                # Age as 7-class classification
                self.loss_fn = None  # will use CE/focal in steps
                self.metric = Accuracy(task="multiclass", num_classes=output_dim)
            else:
                # Regression on normalized ages
                self.loss_fn = nn.L1Loss()
                self.metric = MeanAbsoluteError()
        else:
            if target_attr == "genre":
                self.loss_fn = nn.BCEWithLogitsLoss()
                self.metric = None  # macro metrics computed in steps/epoch end
            else:
                self.metric = Accuracy(task="multiclass", num_classes=output_dim)
    def _compute_class_weights(self, y: torch.Tensor, num_classes: int | None = None) -> torch.Tensor | None:
        """Return weights tensor for CrossEntropyLoss given labels y, or None.
        Uses per-batch priors when `use_class_weight=True`.
        """
        if not getattr(self.hparams, "use_class_weight", False):
            return None
        if self.target_attr == "age":
            return None
        # y is [B] long
        with torch.no_grad():
            classes, counts = torch.unique(y, return_counts=True)
            if classes.numel() <= 1:
                return None
            total = counts.sum().float()
            # inverse frequency
            inv_freq = total / (counts.float() + 1e-3)
            # normalize to mean=1 to keep loss scale stable
            inv_freq = inv_freq * (classes.numel() / inv_freq.sum())
            # build weight vector over all classes in order [0..C-1]
            C = int(num_classes) if num_classes is not None else int(y.max().item()) + 1
            w = torch.ones(C, device=y.device, dtype=torch.float)
            for c, val in zip(classes, inv_freq):
                idx = int(c.item())
                if 0 <= idx < C:
                    w[idx] = val
            return w

    def _focal_ce(self, logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
        """Focal loss built on top of CrossEntropy for multiclass (gamma>0 enables)."""
        ce = F.cross_entropy(logits, y, weight=weight, reduction='none')
        if gamma <= 0:
            return ce.mean()
        with torch.no_grad():
            p = torch.softmax(logits, dim=-1).gather(1, y.view(-1,1)).squeeze(1).clamp_min(1e-6)
        loss = ((1 - p) ** gamma) * ce
        return loss.mean()

    def _get_output_dim(self):
        return {"gender": 2, "occupation": 21}.get(
            self.target_attr,
            (self.genre_dim if self.target_attr == "genre" else 0),
        )

    def _age_to_bucket(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map raw age values to bucket indices in [0..C-1].
        If values are already ML-1M codes {1,18,25,35,45,50,56}, map by exact
        lookup; otherwise, map to the nearest bucket center in self.age_bucket_values.
        Returns a long tensor of the same shape as y.
        """
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        y = y.detach().float().view(-1)
        codes = self.age_bucket_values.to(y.device).view(-1)
        # Map every value to nearest bucket center (codes are small: 7)
        diffs = torch.abs(y.unsqueeze(1) - codes.unsqueeze(0))  # [B, C]
        idx = torch.argmin(diffs, dim=1)
        return idx.long().view_as(y)

    def _maybe_to_years(self, t: torch.Tensor) -> torch.Tensor:
        """Heuristic to pretty-print ages in *years* for reporting.
        If values look normalized (<= ~1.5 at 95th percentile), treat them as [0,1] and scale by MAX_AGE_YEARS.
        If values look like ML-1M codes (<= ~60), return as-is (already "years-like").
        Otherwise, return as-is.
        """
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        td = t.detach().float().reshape(-1)
        if td.numel() == 0:
            return t
        try:
            q95 = torch.quantile(td.cpu(), 0.95)
        except Exception:
            q95 = td.cpu().kthvalue(max(1, int(0.95 * td.numel())))[0]
        q95 = float(q95)
        if q95 <= 1.5:
            return t * MAX_AGE_YEARS
        # ML-1M code values are up to 56; treat as already years-like
        if q95 <= 60.0:
            return t
        return t

    def forward(self, x):
        return self.model(x)

    def _use_feature(self, name: str) -> bool:
        # Exclude explicitly requested features
        if name in self.exclude_features:
            return False
        # Never include the target itself
        if name == self.target_attr:
            return False
        return True

    def _feature_names(self):
        names = []
        if self.include_rating and self._use_feature("rating"):
            names.append("rating")
        if self.use_identifiers:
            if self._use_feature("user_id"):
                names.append("user_id")
            if self._use_feature("item_id"):
                names.append("item_id")
        if self._use_feature("age"):
            names.append("age")
        if self._use_feature("occupation"):
            names.append("occupation")
        if self._use_feature("gender"):
            names.append("gender")
        # expand genre into 19 columns (skip only if explicitly excluded)
        if self._use_feature("genre"):
            for i in range(int(self.genre_dim)):
                names.append(f"genre[{i}]")
        return names

    def _prepare_features(self, batch):
        """
        Accepts either:
        - dict-like batch from a standard DataModule (expects keys like 'rating', 'age', 'genre', ...), or
        - (features, label) tuple/list from a Dataset that already concatenates features (e.g., AttributeInferenceDataset).
        Returns a 2D float tensor [B, D].
        """
        # If batch is already a (features, label) pair from a custom Dataset
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x = batch[0]
            # Ensure shape [B, D]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x.float()

        # Otherwise expect a mapping (dict-like) with individual columns
        features = []
        # Optional rating (scale 0..5 -> 0..1)
        if self.include_rating and self._use_feature("rating"):
            r = batch['rating'] if 'rating' in batch else (batch['pred_rating'] if 'pred_rating' in batch else None)
            if r is None:
                # create a zero column if rating missing
                B = (batch['age'].shape[0] if ('age' in batch and torch.is_tensor(batch['age']))
                     else (batch['genre'].shape[0] if ('genre' in batch and torch.is_tensor(batch['genre'])) else 1))
                device = self.model[0].weight.device if isinstance(self.model, nn.Sequential) else None
                r01 = torch.zeros(B, 1, device=device)
            else:
                r = r.float().unsqueeze(1)
                r01 = (r / 5.0) if (r.min() < 0 or r.max() > 1) else r
                r01 = r01.clamp_(0, 1)
            features.append(r01)
        # Optional identifiers (assumed pre-normalized by datamodule; if raw, normalize upstream)
        if self.use_identifiers:
            if self._use_feature("user_id"):
                features.append(batch['user_id'].float().unsqueeze(1))
            if self._use_feature("item_id"):
                features.append(batch['item_id'].float().unsqueeze(1))
        # Age: pipeline uses ML-1M codes {1,18,25,35,45,50,56}; scale by /MAX_AGE_YEARS to match WGAN features
        if self._use_feature("age"):
            features.append((batch['age'].float().unsqueeze(1) / MAX_AGE_YEARS).clamp_(0, 1))
        # Occupation: integer in [0..20]; normalize to [0,1] via /20.0 so 20 -> 1.0, 0 -> 0.0
        if self._use_feature("occupation"):
            features.append((batch['occupation'].float().unsqueeze(1) / 20.0).clamp_(0, 1))
        if self._use_feature("gender"):
            features.append(batch['gender'].float().unsqueeze(1))
        # Genre (multi-hot)
        # Genre (multi-hot)
        if self._use_feature("genre"):
            g = batch['genre'].float()
            if g.dim() == 1:
                g = g.unsqueeze(0)
            G = int(self.genre_dim)
            if g.shape[1] != G:
                if g.shape[1] > G:
                    g = g[:, :G]
                else:
                    pad = torch.zeros(g.shape[0], G - g.shape[1], device=g.device, dtype=g.dtype)
                    g = torch.cat([g, pad], dim=1)
            features.append(g)
        return torch.cat(features, dim=1)

    def _get_target(self, batch):
        # If dataset provided (features, label)
        if self.target_attr == "genre":
            # multilabel (multi-hot)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                y = batch[1]
            else:
                y = batch["genre"]
            y = y.float()
            if y.dim() == 1:
                y = y.unsqueeze(0)
            G = int(self.genre_dim)
            if y.shape[1] != G:
                if y.shape[1] > G:
                    y = y[:, :G]
                else:
                    pad = torch.zeros(y.shape[0], G - y.shape[1], device=y.device, dtype=y.dtype)
                    y = torch.cat([y, pad], dim=1)
            return y
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            y = batch[1]
            if self.target_attr == "age" and self._age_as_buckets:
                return self._age_to_bucket(y)
            return y
        # Else mapping with target column
        if self.target_attr == "age":
            y = batch["age"]
            return self._age_to_bucket(y) if self._age_as_buckets else y
        return batch[self.target_attr]


    def training_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        # Sanity check: input feature dimension must match model input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.input_dim:
            raise RuntimeError(
                f"AttributeClassifier input_dim mismatch: got x.shape[1]={x.size(1)} but model expects {self.input_dim}.\n"
                f"Hint: ensure datamodule.include_rating == model.include_rating and include_identifiers flags match,"
                f" and that the target '{self.target_attr}' is excluded from features."
                " If you see this during training, ensure on_fit_start ran; otherwise set input_dim explicitly."
            )
        y = self._get_target(batch)
        if self.target_attr == "genre":
            y = y.float()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            # macro-F1 @ 0.5
            with torch.no_grad():
                preds = (probs >= 0.5).float()
                tp = (preds * y).sum(dim=0)
                fp = (preds * (1 - y)).sum(dim=0)
                fn = ((1 - preds) * y).sum(dim=0)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                macro_f1 = torch.nanmean(f1).detach()
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_genre_macro_f1", macro_f1, prog_bar=True)
            return loss


        if self.target_attr == "age" and self._age_as_buckets:
            # Age classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)
        elif self.target_attr == "age":
            # Age regression
            y = y.float()
            preds = self(x).squeeze(-1)
            loss = self.loss_fn(preds, y)
            metric = self.metric(self._maybe_to_years(preds), self._maybe_to_years(y))
        else:
            # Gender/Occupation classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log(f"train_{self.target_attr}_metric", metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        # Sanity check: input feature dimension must match model input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.input_dim:
            raise RuntimeError(
                f"AttributeClassifier input_dim mismatch: got x.shape[1]={x.size(1)} but model expects {self.input_dim}.\n"
                f"Hint: ensure datamodule.include_rating == model.include_rating and include_identifiers flags match,"
                f" and that the target '{self.target_attr}' is excluded from features."
                " If you see this during training, ensure on_fit_start ran; otherwise set input_dim explicitly."
            )
        y = self._get_target(batch)

        if self.target_attr == "genre":
            y = y.float()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            with torch.no_grad():
                preds = (probs >= 0.5).float()
                tp = (preds * y).sum(dim=0)
                fp = (preds * (1 - y)).sum(dim=0)
                fn = ((1 - preds) * y).sum(dim=0)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                macro_f1 = torch.nanmean(f1).detach()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_genre_macro_f1", macro_f1, prog_bar=True)
            return

        if self.target_attr == "age" and self._age_as_buckets:
            # Age classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)
            # collect for threshold search if binary (e.g., gender or age as buckets with 2 classes)
            if probs.ndim == 2 and probs.shape[1] == 2:
                self._val_targets.append(y.detach().cpu())
                self._val_probs.append(probs[:, 1].detach().cpu())
        elif self.target_attr == "age":
            # Age regression
            y = y.float()
            preds = self(x).squeeze(-1)
            loss = self.loss_fn(preds, y)
            metric = self.metric(self._maybe_to_years(preds), self._maybe_to_years(y))
        else:
            # Gender/Occupation classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)
            # collect for threshold search if binary (e.g., gender)
            if probs.ndim == 2 and probs.shape[1] == 2:
                self._val_targets.append(y.detach().cpu())
                self._val_probs.append(probs[:, 1].detach().cpu())

        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_{self.target_attr}_metric", metric, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.target_attr == "age":
            return
        if not self._val_targets:
            return
        y = torch.cat(self._val_targets).numpy()
        s = torch.cat(self._val_probs).numpy()
        # Grid search 101 thresholds [0,1]
        best_t = 0.5
        best_ba = -1.0
        try:
            if balanced_accuracy_score is not None:
                for t in np.linspace(0.0, 1.0, 101):
                    pred = (s >= t).astype(int)
                    ba = balanced_accuracy_score(y, pred)
                    if ba > best_ba:
                        best_ba, best_t = ba, float(t)
        except Exception:
            pass
        self._best_threshold = best_t
        # Clear buffers for next epoch
        self._val_targets.clear(); self._val_probs.clear()
        # Log for visibility
        try:
            self.log("val/best_thresh", best_t, prog_bar=False)
            self.log("val/best_bal_acc", best_ba, prog_bar=False)
        except Exception:
            pass

    def test_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        # Sanity check: input feature dimension must match model input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.input_dim:
            raise RuntimeError(
                f"AttributeClassifier input_dim mismatch: got x.shape[1]={x.size(1)} but model expects {self.input_dim}.\n"
                f"Hint: ensure datamodule.include_rating == model.include_rating and include_identifiers flags match,"
                f" and that the target '{self.target_attr}' is excluded from features."
                " If you see this during training, ensure on_fit_start ran; otherwise set input_dim explicitly."
            )
        y = self._get_target(batch)

        if self.target_attr == "genre":
            y = y.float()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            self._test_targets.append(y.detach().cpu())
            self._test_probs.append(probs.detach().cpu())
            preds = (probs >= 0.5).float()
            self._test_preds.append(preds.detach().cpu())
            # quick macro-F1 for progress bar
            with torch.no_grad():
                tp = (preds * y).sum(dim=0)
                fp = (preds * (1 - y)).sum(dim=0)
                fn = ((1 - preds) * y).sum(dim=0)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                macro_f1 = torch.nanmean(f1).detach()
            self.log("test_loss", loss)
            self.log("test_genre_macro_f1", macro_f1, prog_bar=True)
            return
        
        if self.target_attr == "age" and self._age_as_buckets:
            # Age classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)
            # classification: store targets and probabilities for AUC/AP/baselines
            self._test_targets.append(y.detach().cpu())
            # probs: for binary either scalar or 2-col; normalize to 1D positive-class score
            if probs.ndim == 2 and probs.shape[1] > 1:
                pos_scores = probs[:, 1]
            else:
                pos_scores = probs.squeeze(-1)
            self._test_probs.append(pos_scores.detach().cpu())
            # also store hard predictions
            if probs.ndim == 2 and probs.shape[1] > 1:
                # if binary, use tuned threshold; else argmax
                if probs.shape[1] == 2:
                    thr = getattr(self, "_best_threshold", 0.5)
                    hard = (probs[:, 1] >= thr).long()
                else:
                    hard = probs.argmax(dim=-1)
            else:
                thr = getattr(self, "_best_threshold", 0.5)
                hard = (pos_scores >= thr).long()
            self._test_preds.append(hard.detach().cpu())
        elif self.target_attr == "age":
            # Age regression
            y = y.float()
            preds = self(x).squeeze(-1)
            loss = self.loss_fn(preds, y)
            metric = self.metric(self._maybe_to_years(preds), self._maybe_to_years(y))
            # store raw preds and targets for MAE/R2 in on_test_epoch_end
            self._test_preds.append(preds.detach().cpu())
            self._test_targets.append(y.detach().cpu())
        else:
            # Gender/Occupation classification
            y = y.long()
            logits = self(x)
            weight = self._compute_class_weights(y, num_classes=int(logits.shape[1]))
            loss = self._focal_ce(logits, y, weight, float(getattr(self.hparams, 'focal_gamma', 0.0)))
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)
            # classification: store targets and probabilities for AUC/AP/baselines
            self._test_targets.append(y.detach().cpu())
            # probs: for binary either scalar or 2-col; normalize to 1D positive-class score
            if probs.ndim == 2 and probs.shape[1] > 1:
                pos_scores = probs[:, 1]
            else:
                pos_scores = probs.squeeze(-1)
            self._test_probs.append(pos_scores.detach().cpu())
            # also store hard predictions
            if probs.ndim == 2 and probs.shape[1] > 1:
                # if binary, use tuned threshold; else argmax
                if probs.shape[1] == 2:
                    thr = getattr(self, "_best_threshold", 0.5)
                    hard = (probs[:, 1] >= thr).long()
                else:
                    hard = probs.argmax(dim=-1)
            else:
                thr = getattr(self, "_best_threshold", 0.5)
                hard = (pos_scores >= thr).long()
            self._test_preds.append(hard.detach().cpu())

        self.log("test_loss", loss)
        self.log(f"test_{self.target_attr}_metric", metric, prog_bar=True)

    def on_test_epoch_end(self):
        # Aggregate and compute baselines / richer metrics for leakage interpretation
        if not self._test_targets:
            return
        # Safe local import for sklearn metrics to avoid UnboundLocalError from inner imports
        try:
            import sklearn.metrics as _skm
        except Exception:
            _skm = None
        import numpy as np

        # Multilabel GENRE reporting
        if str(getattr(self, "target_attr", "")).lower() == "genre":
            import numpy as _np
            y = torch.cat(self._test_targets).numpy()
            probs = torch.cat(self._test_probs).numpy()
            preds = torch.cat(self._test_preds).numpy()
            G = int(getattr(self, "genre_dim", probs.shape[1]))

            macro_auc = None
            macro_ap = None
            if _skm is not None:
                try:
                    aucs = []
                    for j in range(G):
                        # only compute AUC if both classes present
                        if len(set(y[:, j])) > 1:
                            aucs.append(_skm.roc_auc_score(y[:, j], probs[:, j]))
                    if aucs:
                        macro_auc = float(_np.mean(aucs))
                except Exception:
                    pass
                try:
                    aps = [_skm.average_precision_score(y[:, j], probs[:, j]) for j in range(G)]
                    macro_ap = float(_np.mean(aps))
                except Exception:
                    pass

            # macro-F1 @ 0.5
            try:
                if _skm is not None:
                    macro_f1 = float(_skm.f1_score(y, (probs >= 0.5).astype(int), average="macro", zero_division=0))
                else:
                    macro_f1 = float("nan")
            except Exception:
                macro_f1 = float("nan")

            # --- Consensus-based leakage check (vs chance) ---
            eps_auc = 0.05   # AUC must exceed 0.5 by ≥5pp
            eps_f1  = 0.10   # or macro-F1 reasonably above trivial baselines
            auc_delta = (macro_auc - 0.5) if (macro_auc is not None) else float("nan")
            leak = False
            if (macro_auc is not None) and (macro_auc >= 0.5 + eps_auc):
                leak = True
            if (not leak) and (macro_f1 == macro_f1) and (macro_f1 >= eps_f1):
                leak = True

            out = {
                "test/auc_macro": (macro_auc if macro_auc is not None else float("nan")),
                "test/auc_macro_delta_over_chance": (auc_delta if macro_auc is not None else float("nan")),
                "test/ap_macro":  (macro_ap if macro_ap is not None else float("nan")),
                "test/macro_f1":  macro_f1,
                "test/leakage_detected": float(leak),
            }
            try:
                self.log_dict(out, prog_bar=False)
            except Exception:
                pass

            msg = f"[Leakage/Test] genre: AUC_macro={out['test/auc_macro']:.4f} (Δchance={out['test/auc_macro_delta_over_chance']:.4f}) | AP_macro={out['test/ap_macro']:.4f} | macroF1={macro_f1:.4f}"
            msg += " | LEAKAGE=YES" if leak else " | LEAKAGE=NO"
            self.print(msg)

            self._test_targets.clear(); self._test_preds.clear(); self._test_probs.clear()
            return


        y = torch.cat(self._test_targets).numpy()
        if self.target_attr == "age" and not self._age_as_buckets:
            # regression reporting (scale-aware)
            preds_t = torch.cat(self._test_preds)
            y_t = torch.cat(self._test_targets)

            # MAE in raw units (for debugging)
            mae_norm = float(torch.mean(torch.abs(preds_t - y_t)).item())
            mae_baseline_norm = float(torch.mean(torch.abs(y_t.mean() - y_t)).item())

            # Convert to years only if values look normalized
            preds_y = self._maybe_to_years(preds_t)
            y_y = self._maybe_to_years(y_t)
            mae_years = float(torch.mean(torch.abs(preds_y - y_y)).item())
            mae_baseline_years = float(torch.mean(torch.abs(y_y.mean() - y_y)).item())
            imp = (mae_baseline_years - mae_years) / (mae_baseline_years + 1e-8)
            # Log both for backward-compatibility + human-readable years
            self.log_dict({
                "test/mae_epoch": mae_norm,
                "test/mae_baseline_mean": mae_baseline_norm,
                "test/improvement_over_mean": max(0.0, float(imp)),
                "test/mae_epoch_years": mae_years,
                "test/mae_baseline_mean_years": mae_baseline_years,
            }, prog_bar=False)
            self.print(
                f"[Leakage/Test] age: MAE={mae_years:.2f} yrs | baseline(mean)={mae_baseline_years:.2f} yrs | improvement={imp*100:.2f}%"
            )
        else:
            # classification reporting (covers gender, occupation, and age-as-buckets)
            preds = torch.cat(self._test_preds).numpy()
            probs = torch.cat(self._test_probs).numpy()
            # --- Extra diagnostics for occupation (multi-class) ---
            if str(getattr(self, "target_attr", "")).lower() == "occupation":
                try:
                    import numpy as _np
                    # class prior
                    _vals, _cnts = _np.unique(y, return_counts=True)
                    self.print("[Occ][priors] " + " ".join(f"{int(v)}:{int(c)}" for v, c in zip(_vals, _cnts)))
                    # confusion / per-class accuracy when available
                    if _skm is not None and hasattr(_skm, "confusion_matrix"):
                        # ensure labels cover 0..C-1 even if a class is missing in y
                        C = int(self._get_output_dim())
                        cm = _skm.confusion_matrix(y, preds, labels=list(range(C)))
                        with _np.errstate(divide="ignore", invalid="ignore"):
                            per_class_acc = cm.diagonal() / _np.clip(cm.sum(axis=1), 1, None)
                        # show worst 5 classes
                        order = _np.argsort(per_class_acc)
                        worst = ", ".join(f"{int(i)}={float(per_class_acc[i]):.2f}" for i in order[:5])
                        self.print(f"[Occ][per-class acc] worst5: {worst}")
                except Exception as _e:
                    self.print(f"[Occ][diag] skipped diagnostics: {_e}")
            # Majority baseline accuracy
            (values, counts) = np.unique(y, return_counts=True)
            maj = values[np.argmax(counts)]
            acc_baseline = float((y == maj).mean())
            # Observed accuracy (already logged per-step via torchmetrics; compute again for completeness)
            acc = float((preds == y).mean())
            margin = float(acc - acc_baseline)
            out = {
                "test/acc_epoch": acc,
                "test/acc_baseline_majority": acc_baseline,
                "test/acc_margin_over_majority": max(0.0, margin),
            }
            # Log tuned threshold for test
            out["test/decision_threshold"] = float(getattr(self, "_best_threshold", 0.5))
            # Extra robustness metrics if sklearn is available
            try:
                if _skm is not None:
                    out["test/macro_f1"] = float(_skm.f1_score(y, preds, average="macro", zero_division=0))
                    out["test/balanced_acc"] = float(_skm.balanced_accuracy_score(y, preds))
            except Exception:
                pass
            # AUC/AP if binary and sklearn available
            try:
                if _skm is not None and set(np.unique(y)).issubset({0, 1}):
                    out["test/auc"] = float(_skm.roc_auc_score(y, probs))
                    out["test/ap"] = float(_skm.average_precision_score(y, probs))
            except Exception:
                pass

            # Deltas vs baselines (consensus)
            delta_acc = float(acc - acc_baseline)
            out["test/acc_delta_over_majority"] = max(0.0, delta_acc)

            # number of classes
            C = None
            if self.target_attr == "occupation":
                C = int(self._get_output_dim())
            elif self.target_attr == "gender":
                C = 2
            elif self.target_attr == "age" and self._age_as_buckets:
                C = int(self._n_age_classes)
            chance_bal = (1.0 / C) if (C is not None and C > 0) else None

            # balanced-acc delta over chance
            if "test/balanced_acc" in out and (chance_bal is not None):
                out["test/balanced_acc_delta_over_chance"] = max(0.0, float(out["test/balanced_acc"] - chance_bal))

            # For binary: AUC delta over chance
            if "test/auc" in out:
                out["test/auc_delta_over_chance"] = float(out["test/auc"] - 0.5)

            # --- Consensus-based leakage detection ---
            # Thresholds:
            #  - Binary (gender): AUC ≥ 0.55 OR (Δacc ≥ 0.02 AND Δbal ≥ 0.05)
            #  - Multiclass (occupation/age buckets): Δbal ≥ 0.05 AND Δacc ≥ 0.02
            #  - Supportive: macro-F1 ≥ 0.10
            eps_auc = 0.05
            eps_acc = 0.02
            eps_bal = 0.05

            detected = False
            if self.target_attr == "gender":
                if ("test/auc" in out) and (out["test/auc"] >= 0.5 + eps_auc):
                    detected = True
                elif ("test/balanced_acc_delta_over_chance" in out) and (out["test/balanced_acc_delta_over_chance"] >= eps_bal) and (delta_acc >= eps_acc):
                    detected = True
            else:
                if ("test/balanced_acc_delta_over_chance" in out) and (out["test/balanced_acc_delta_over_chance"] >= eps_bal) and (delta_acc >= eps_acc):
                    detected = True
            if (not detected) and ("test/macro_f1" in out) and (out["test/macro_f1"] >= 0.10):
                detected = True
            out["test/leakage_detected"] = float(detected)
            self.log_dict(out, prog_bar=False)
            msg = (f"[Leakage/Test] {self.target_attr}: ACC={acc:.4f} "
                   f"| baseline(majority)={acc_baseline:.4f} "
                   f"| margin={margin:.4f} "
                   f"| Δacc={out['test/acc_delta_over_majority']:.4f}")
            if "test/macro_f1" in out:
                msg += f" | macroF1={out['test/macro_f1']:.4f}"
            if "test/balanced_acc" in out:
                msg += f" | balACC={out['test/balanced_acc']:.4f}"
            if "test/balanced_acc_delta_over_chance" in out:
                msg += f" | Δbal={out['test/balanced_acc_delta_over_chance']:.4f}"
            if "test/auc" in out:
                msg += f" | AUC={out['test/auc']:.4f}"
            if "test/auc_delta_over_chance" in out:
                msg += f" | ΔAUC={out['test/auc_delta_over_chance']:.4f}"
            if "test/ap" in out:
                msg += f" | AP={out['test/ap']:.4f}"
            msg += f" | thr={out['test/decision_threshold']:.2f}"
            if out.get("test/leakage_detected", 0.0) > 0.0:
                msg += " | LEAKAGE=YES"
            else:
                msg += " | LEAKAGE=NO"
            try:
                if _skm is not None and hasattr(_skm, "confusion_matrix") and set(np.unique(y)).issubset({0, 1}):
                    cm = _skm.confusion_matrix(y, preds)
                    self.print(f"[Leakage/Test] confusion_matrix=\n{cm}")
            except Exception:
                pass
            self.print(msg)
        # clear buffers
        self._test_targets.clear()
        self._test_preds.clear()
        self._test_probs.clear()

    def on_fit_start(self):
        pretty = [f"[{i}] {n}" for i, n in enumerate(self._feature_names())]
        self.print(f"[AttributeClassifier] Using features (target={self.target_attr}): {pretty}")
        self.print(
            f"[AttributeClassifier] opts: input_dim={self.input_dim}, use_class_weight={getattr(self.hparams,'use_class_weight', False)}, "
            f"focal_gamma={getattr(self.hparams,'focal_gamma', 0.0)}, input_layernorm={getattr(self.hparams,'input_layernorm', False)}, "
            f"use_batchnorm={getattr(self.hparams,'use_batchnorm', False)}"
        )

        # --- Probe a real batch to infer feature dim when dataset returns (features, label) ---
        try:
            dl = None
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                if hasattr(dm, "class_attack_train_dataloader"):
                    dl = dm.class_attack_train_dataloader()
                elif hasattr(dm, "train_dataloader"):
                    dl = dm.train_dataloader()
            if dl is not None:
                batch = next(iter(dl))
                x_probe = self._prepare_features(batch)
                if x_probe.dim() == 1:
                    x_probe = x_probe.unsqueeze(0)
                feat_dim = int(x_probe.size(1))
                if (self.input_dim in (None, 0)) or (feat_dim != int(self.input_dim)):
                    # Determine output_dim again (age buckets vs regression vs classification)
                    if self.target_attr == "age":
                        output_dim = (self._n_age_classes if self._age_as_buckets else 1)
                    else:
                        output_dim = self._get_output_dim()
                    self.model = self._make_mlp(feat_dim, output_dim)
                    old_dim = self.input_dim
                    self.input_dim = feat_dim
                    self.print(f"[AttributeClassifier] Rebuilt MLP for input_dim={feat_dim} (was {old_dim})")
        except Exception as e:
            self.print(f"[AttributeClassifier] probe/rebuild skipped ({type(e).__name__}): {e}")

        # ---- One-batch feature & label diagnostics (non-fatal) ----
        try:
            dl = None
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                if hasattr(dm, "class_attack_train_dataloader"):
                    dl = dm.class_attack_train_dataloader()
                elif hasattr(dm, "train_dataloader"):
                    dl = dm.train_dataloader()
            if dl is not None:
                batch = next(iter(dl))
                # Build features the same way training_step does
                x = self._prepare_features(batch)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = x.float()
                try:
                    y = self._get_target(batch)
                except Exception:
                    y = None
                # Basic stats
                x_mean = x.mean(dim=0)
                x_std = x.std(dim=0)
                # Summaries to keep logs short
                self.print(
                    f"[AttributeClassifier][diag] first-batch: X.shape={tuple(x.shape)} | X.mean={float(x.mean()):.4f} | X.std={float(x.std()):.4f}"
                )
                # If labels available, show prior/skew
                if y is not None:
                    if y.dtype.is_floating_point:
                        self.print(f"[AttributeClassifier][diag] first-batch: y(mean)={float(y.mean()):.4f} (regression)")
                    else:
                        y_np = y.detach().cpu().numpy().reshape(-1)
                        # show simple class prior for classification
                        try:
                            import numpy as _np
                            vals, cnts = _np.unique(y_np, return_counts=True)
                            prior = {int(v): int(c) for v, c in zip(vals, cnts)}
                            self.print(f"[AttributeClassifier][diag] first-batch: y prior counts={prior}")
                        except Exception:
                            pass
                # Optional: check input_dim alignment here too
                if x.size(1) != self.input_dim:
                    self.print(
                        f"[WARN] AttributeClassifier: x.shape[1]={x.size(1)} but input_dim={self.input_dim}. "
                        f"Check include_identifiers/include_rating/feature gating."
                    )
        except Exception as e:
            self.print(f"[AttributeClassifier][diag] Skipped batch diagnostics: {e}")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_name = str(getattr(self.hparams, "scheduler", "none")).lower()
        if sched_name == "cosine":
            # T_max falls back to max_epochs when available; else a sane default
            t_max = getattr(getattr(self, "trainer", None), "max_epochs", 50) or 50
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(t_max))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt

# -------------------------------
# XGBoost-based attribute auditor
# -------------------------------

def _build_features_from_batch(batch, target_attr: str, include_rating: bool, use_identifiers: bool):
    # If dataset returns (features, label), just pass them through
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        if x.dim() == 1: x = x.unsqueeze(0)
        X = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        return X, y

    # Otherwise: dict-style batches (old path)
    feats = []
    if include_rating:
        r = batch['rating'] if 'rating' in batch else (batch['pred_rating'] if 'pred_rating' in batch else None)
        if r is None:
            B = batch['age'].shape[0] if ('age' in batch and torch.is_tensor(batch['age'])) else batch['genre'].shape[0]
            feats.append(torch.zeros(B, 1, device=batch['genre'].device))
        else:
            r = r.float().unsqueeze(1)
            feats.append(torch.clamp(r / 5.0 if (r.min() < 0 or r.max() > 1) else r, 0, 1))
    if use_identifiers:
        feats.extend([batch['user_id'].float().unsqueeze(1), batch['item_id'].float().unsqueeze(1)])
    if target_attr != "age":
        feats.append((batch['age'].float().unsqueeze(1) / 100.0).clamp_(0, 1))
    if target_attr != "occupation":
        feats.append((batch['occupation'].float().unsqueeze(1) / 20.0).clamp_(0, 1))
    if target_attr != "gender":
        feats.append(batch['gender'].float().unsqueeze(1))
    feats.append(batch['genre'].float())  # 19-dim multi-hot
    X = torch.cat(feats, dim=1).detach().cpu().numpy()

    if target_attr == 'age':
        y = batch['age'].float().detach().cpu().numpy()
    else:
        y = batch[target_attr].long().detach().cpu().numpy()
    return X, y

def _collect_numpy(loader, target_attr: str, include_rating=True, use_identifiers=False):
    Xs, ys = [], []
    for batch in loader:
        X, y = _build_features_from_batch(batch, target_attr, include_rating, use_identifiers)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def xgb_audit(
    train_loader,
    test_loader,
    target_attr: str = "gender",
    include_rating: bool = True,
    use_identifiers: bool = False,
    params: dict | None = None,
    early_stopping_rounds: int = 50,
    random_state: int = 0,
    n_jobs: int = -1,
    device: str = "auto",
):
    """
    Train an XGBoost auditor on `train_loader` (typically synthetic) and evaluate on `test_loader` (real).
    Returns a metrics dict. Prevents trivial leakage by default (no IDs; target column not in X).
    """
    if xgb is None:
        raise ImportError("xgboost is not installed. pip install xgboost")

    X_train, y_train = _collect_numpy(train_loader, target_attr, include_rating, use_identifiers)
    X_test, y_test = _collect_numpy(test_loader, target_attr, include_rating, use_identifiers)

    # Choose tree method
    if device == "auto":
        tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"
    else:
        tree_method = "gpu_hist" if device in ("cuda", "gpu") else "hist"

    params = dict(params or {})

    metrics = {}
    if target_attr == "age":
        # Regression
        default = dict(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        default.update(params)
        model = xgb.XGBRegressor(**default)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        preds = model.predict(X_test)
        if mean_absolute_error is not None and r2_score is not None:
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["r2"] = float(r2_score(y_test, preds))
        else:
            metrics["mae"] = float(np.mean(np.abs(y_test - preds)))
    else:
        # Classification: gender (binary) or occupation (multi-class)
        is_binary = (target_attr == "gender")
        default = dict(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if is_binary:
            default.update(dict(objective="binary:logistic"))
            eval_metric = "logloss"
            model = xgb.XGBClassifier(**{**default, **params})
        else:
            n_classes = int(np.max(np.concatenate([y_train, y_test])) + 1)
            default.update(dict(objective="multi:softprob", num_class=n_classes))
            eval_metric = "mlogloss"
            model = xgb.XGBClassifier(**{**default, **params})

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=eval_metric,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1) if proba.ndim == 2 and proba.shape[1] > 1 else (proba > 0.5).astype(int)

        if accuracy_score is not None:
            metrics["acc"] = float(accuracy_score(y_test, pred))
        else:
            metrics["acc"] = float((pred == y_test).mean())

        if is_binary and roc_auc_score is not None:
            scores = proba[:, 1] if proba.ndim == 2 else proba
            metrics["auc"] = float(roc_auc_score(y_test, scores))
            if average_precision_score is not None:
                metrics["ap"] = float(average_precision_score(y_test, scores))

    metrics["tree_method"] = tree_method
    metrics["n_train"] = int(X_train.shape[0])
    metrics["n_test"] = int(X_test.shape[0])
    return metrics


# =====================================================================
# YAML-driven classifier attack sweep (lives in the model module)
# Move most orchestration out of Train.py and call this from there:
#   from models.AttributeClassifier import run_yaml_attack_sweep
#   run_yaml_attack_sweep(cli.config, cli.datamodule, cli.trainer)
# =====================================================================

def run_yaml_attack_sweep(cfg, datamodule, trainer, targets=None, age_as_buckets=None):
    """
    Run an attribute-inference attack sweep using YAML (or explicit args), keeping
    Train.py thin. This function reads configuration from, in priority order:
      1) Explicit function arguments: `targets`, `age_as_buckets` (if provided)
      2) Flat YAML keys on cfg: `sweep_targets`, `sweep_age_as_buckets`, `sweep_out_csv`
      3) `model.init_args`: `targets`, `age_as_buckets`

    Required YAML keys (data sources):
      attack_train_source, attack_test_source, synthetic_hdf5_path, blackbox_hdf5_path

    Classifier hyperparams are taken from either:
      - `classifier.init_args` if present, else
      - `model.init_args` (so you can reuse the same block).

    Example YAML minimal:
      do_classifier_attack_sweep: true
      attack_train_source: synthetic_outputs
      attack_test_source:  predicted_data
      synthetic_hdf5_path: synthetic_outputs/base_strong.hdf5
      blackbox_hdf5_path:  predicted_data/base_strong.hdf5
      model:
        class_path: models.AttributeClassifier
        init_args:
          targets: [gender, occupation, age]
          age_as_buckets: true
          ... other classifier args ...
    """
    # Pull datamodule.init_args (so YAML can place keys under datamodule)
    dm_args = getattr(getattr(cfg, "datamodule", None), "init_args", {}) or {}

    def _cfg_get(name, default=None):
        """Lookup order: top-level cfg attr -> datamodule.init_args -> default."""
        if hasattr(cfg, name):
            val = getattr(cfg, name)
            if val is not None:
                return val
        try:
            return dm_args.get(name, default)
        except Exception:
            return default

    # --- Validate top-level flags ---
    if not bool(_cfg_get("do_classifier_attack", getattr(cfg, "do_classifier_attack", False))):
        print("[SWEEP] cfg.do_classifier_attack is False/absent; skipping.")
        return

    # data sources / paths (required)
    attack_train_source = _cfg_get("attack_train_source")
    attack_test_source  = _cfg_get("attack_test_source")
    synthetic_hdf5_path = _cfg_get("synthetic_hdf5_path")
    blackbox_hdf5_path  = _cfg_get("blackbox_hdf5_path")
    blackbox_dir        = _cfg_get("blackbox_dir", None)

    for name, val in (
        ("attack_train_source", attack_train_source),
        ("attack_test_source", attack_test_source),
        ("synthetic_hdf5_path", synthetic_hdf5_path),
        ("blackbox_hdf5_path", blackbox_hdf5_path),
    ):
        if not val:
            raise ValueError(f"[SWEEP] Missing '{name}' in YAML (either top-level or datamodule.init_args).")

    # ---- Gather sweep settings (priority: args > flat cfg > model.init_args) ----
    model_init = getattr(getattr(cfg, "model", None), "init_args", {}) or {}
    if targets is None:
        targets = getattr(cfg, "sweep_targets", None)
        if targets is None:
            targets = model_init.get("targets", None)
    if not targets:
        raise ValueError("[SWEEP] No targets provided. Set sweep_targets at top-level or model.init_args.targets.")

    if age_as_buckets is None:
        age_as_buckets = getattr(cfg, "sweep_age_as_buckets", None)
        if age_as_buckets is None:
            age_as_buckets = bool(model_init.get("age_as_buckets", True))
    out_csv = getattr(cfg, "sweep_out_csv", None)

    # ---- Classifier base args: prefer classifier.init_args, else model.init_args ----
    base_args = {}
    if hasattr(cfg, "classifier") and hasattr(cfg.classifier, "init_args") and cfg.classifier.init_args:
        base_args = dict(cfg.classifier.init_args)
    elif model_init:
        base_args = dict(model_init)
    else:
        raise ValueError("[SWEEP] Missing classifier hyperparams: provide classifier.init_args or model.init_args.")

    print(f"[SWEEP] targets={list(targets)} | train_source={attack_train_source} | test_source={attack_test_source}")
    print(f"[SWEEP] syn={synthetic_hdf5_path} | bb={blackbox_hdf5_path} | bb_dir={blackbox_dir}")

    # convenience accessor
    def _get(m, k, default=None):
        try:
            return m.get(k, default)
        except Exception:
            return default

    results = []

    def _run_one(target_attr: str, age_as_buckets: bool):
        # build args from YAML only; override the target-specific knobs
        args = dict(base_args)
        args["target_attr"] = target_attr
        args["age_as_buckets"] = bool(age_as_buckets if target_attr == "age" else False)
        # Allow the model to infer input_dim unless explicitly fixed in YAML
        if "input_dim" not in args or args["input_dim"] is None:
            args["input_dim"] = None

        clf = AttributeClassifier(**args)

        # Configure datamodule strictly from YAML
        setattr(datamodule, "attack_target", target_attr)
        setattr(datamodule, "attack_train_source", attack_train_source)
        setattr(datamodule, "attack_test_source",  attack_test_source)
        setattr(datamodule, "synthetic_hdf5_path", synthetic_hdf5_path)
        setattr(datamodule, "blackbox_hdf5_path",  blackbox_hdf5_path)
        if blackbox_dir:
            setattr(datamodule, "blackbox_dir", blackbox_dir)

        # --- Optional per-run dataset knobs from YAML (only apply if provided) ---
        for k in (
            "attack_include_identifiers",     # bool
            "attack_include_rating",          # bool
            "attack_identifiers_one_based",   # bool
            "attack_remap_occ_1_based",       # bool
            "attack_age_label_mode",          # "code" | "bucket" | "year"
        ):
            val = _cfg_get(k, None)
            if val is not None:
                setattr(datamodule, k, val)

        # Prepare loaders
        datamodule.setup(stage="classifier_attack")
        train_loader = (
            datamodule.class_attack_train_dataloader()
            if hasattr(datamodule, "class_attack_train_dataloader")
            else datamodule.train_dataloader()
        )
        val_loader = (
            datamodule.class_attack_val_dataloader()
            if hasattr(datamodule, "class_attack_val_dataloader")
            else datamodule.val_dataloader()
        )
        test_loader = (
            datamodule.class_attack_test_dataloader()
            if hasattr(datamodule, "class_attack_test_dataloader")
            else datamodule.test_dataloader()
        )

        print(f"[SWEEP] target={target_attr} | age_as_buckets={args['age_as_buckets']}")

        # Build a fresh Trainer per target to avoid any state/callback leakage
        # Inherit a few common settings from the outer trainer when available
        outer = trainer
        max_epochs  = getattr(outer, "max_epochs", 50)
        accelerator = getattr(outer, "accelerator", None) or "auto"
        devices     = getattr(outer, "devices", None)
        # Lightning expects an int/sequence for devices; avoid passing None
        if devices in (None, [], 0):
            devices = 1
        precision   = getattr(outer, "precision", "32-true")
        enable_pb   = getattr(outer, "enable_progress_bar", True)
        deterministic = getattr(outer, "deterministic", False)
        gradient_clip_val = getattr(outer, "gradient_clip_val", 0.0)
        accumulate_grad_batches = getattr(outer, "accumulate_grad_batches", 1)
        log_every_n_steps = getattr(outer, "log_every_n_steps", 10)
        limit_train_batches = getattr(outer, "limit_train_batches", 1.0)
        limit_val_batches   = getattr(outer, "limit_val_batches", 1.0)
        limit_test_batches  = getattr(outer, "limit_test_batches", 1.0)

        # Per-target logger (separate version so checkpoints & logs don't collide)
        run_name = f"mlp_{args.get('pred_file', 'run')}"
        logger = None
        if TensorBoardLogger is not None:
            try:
                logger = TensorBoardLogger(save_dir="lightning_logs", name=run_name, version=f"{target_attr}")
            except Exception as e:
                print(f"[SWEEP] TensorBoardLogger unavailable ({e}); falling back to CSVLogger.")
        if logger is None and 'CSVLogger' in globals() and CSVLogger is not None:
            try:
                logger = CSVLogger(save_dir="lightning_logs", name=run_name, version=f"{target_attr}")
            except Exception as e:
                print(f"[SWEEP] CSVLogger unavailable ({e}); proceeding without a logger.")

        fresh_trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            enable_progress_bar=enable_pb,
            deterministic=deterministic,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            logger=logger,
        )

        # Force a clean start (no auto-resume) and run
        import time
        t0 = time.perf_counter()
        fresh_trainer.fit(clf, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
        dt_min = (time.perf_counter() - t0) / 60.0
        print(f"[SWEEP] target={target_attr} finished training in {dt_min:.1f} min")

        out = fresh_trainer.test(model=clf, dataloaders=test_loader)
        metrics = out[0] if isinstance(out, list) and out else {}
        results.append({
    "target": target_attr,
    "metrics": metrics,
    "age_as_buckets": bool(args["age_as_buckets"]) if target_attr == "age" else None,
    "is_multilabel": (target_attr == "genre"),
})

    # run all requested targets
    for tgt in targets:
        t = str(tgt).lower()
        _run_one(t, age_as_buckets=(age_as_buckets if t == "age" else False))

    # summary printout
    print("\n[SWEEP] Summary:")
    header = "target | acc | balACC | AUC | AP | macroF1 | MAE | baseline(acc/MAE) | leakage  (AUC/AP are macro for multilabel)"
    print(header)
    print("-" * len(header))

    # optional CSV collector
    rows = []

    for rec in results:
        t = str(rec["target"]).lower()
        m = rec["metrics"]
        is_multilabel = bool(rec.get("is_multilabel", False))
        age_bucketed = bool(rec.get("age_as_buckets", False)) if t == "age" else False

        def _get(d, k, default=float("nan")):
            try:
                return d[k]
            except Exception:
                return default
        def _fmt(x):
            try:
                return "-" if x is None or (isinstance(x, float) and (x != x)) else f"{float(x):.4f}"
            except Exception:
                return "-"

        if is_multilabel:
            aucm = _get(m, "test/auc_macro")
            apm  = _get(m, "test/ap_macro")
            f1m  = _get(m, "test/macro_f1")
            leak = bool(_get(m, "test/leakage_detected", 0.0) > 0.0)
            print(f"{t:9s} |   -  |   -    | {_fmt(aucm)} | {_fmt(apm)} | {_fmt(f1m)} |  -  |   -   | {'YES' if leak else 'NO'}")
            rows.append({"target": t, "auc_macro": aucm, "ap_macro": apm, "macro_f1": f1m, "leakage": leak})
        elif t == "age" and not age_bucketed:
            mae   = _get(m, "test/mae_epoch_years", _get(m, "test/mae_epoch"))
            mae_b = _get(m, "test/mae_baseline_mean_years", _get(m, "test/mae_baseline_mean"))
            print(f"{t:9s} |   -  |   -    |  -  |  -  |   -    | {_fmt(mae)} | {_fmt(mae_b)} | NO")
            rows.append({"target": t, "mae": mae, "mae_baseline": mae_b, "leakage": False})
        else:
            acc   = _get(m, "test/acc_epoch")
            bal   = _get(m, "test/balanced_acc")
            auc   = _get(m, "test/auc")
            ap    = _get(m, "test/ap")
            f1    = _get(m, "test/macro_f1")
            base  = _get(m, "test/acc_baseline_majority")
            leak  = bool(_get(m, "test/leakage_detected", 0.0) > 0.0)
            print(f"{t:9s} | {_fmt(acc)} | {_fmt(bal)} | {_fmt(auc)} | {_fmt(ap)} | {_fmt(f1)} |  -  | {_fmt(base)} | {'YES' if leak else 'NO'}")
            rows.append({
                "target": t, "acc": acc, "balanced_acc": bal, "auc": auc, "ap": ap, "macro_f1": f1,
                "baseline_acc": base, "leakage": leak,
            })
    # Optional CSV output if sweep.out_csv is provided
    if out_csv:
        try:
            import csv, os
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            with open(out_csv, "w", newline="") as fp:
                if rows:
                    keys = sorted({k for r in rows for k in r.keys()})
                    w = csv.DictWriter(fp, fieldnames=keys)
                    w.writeheader(); w.writerows(rows)
            print(f"[SWEEP] wrote CSV -> {out_csv}")
        except Exception as e:
            print(f"[SWEEP] could not write CSV '{out_csv}': {e}")

    print("[SWEEP] done\n")