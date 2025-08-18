import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError

import numpy as np
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error, r2_score
except Exception:
    roc_auc_score = average_precision_score = accuracy_score = mean_absolute_error = r2_score = None

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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_attr = target_attr
        self.pred_file = pred_file
        self.use_identifiers = use_identifiers
        self.include_rating = include_rating
        self.strict_feature_check = strict_feature_check
        # Allow users to drop specific features from the auditor (e.g., to bypass bad columns temporarily)
        self.exclude_features = set(exclude_features or [])

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
            if self._use_feature("genre"):
                input_dim += 19
        self.input_dim = input_dim

        if self.strict_feature_check and self.use_identifiers:
            warnings.warn(
                f"AttributeClassifier: use_identifiers=True; IDs can trivially reveal {self.target_attr}. "
                "Set use_identifiers=False for a fair leakage audit.",
                RuntimeWarning,
            )

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
            for i in range(19):
                names.append(f"genre[{i}]")
        return names

    def _prepare_features(self, batch):
        features = []
        # Optional rating
        if self.include_rating and self._use_feature("rating"):
            features.append(batch['rating'].float().unsqueeze(1))
        # Optional identifiers
        if self.use_identifiers:
            if self._use_feature("user_id"):
                features.append(batch['user_id'].float().unsqueeze(1))
            if self._use_feature("item_id"):
                features.append(batch['item_id'].float().unsqueeze(1))
        # Scalar attributes
        if self._use_feature("age"):
            features.append(batch['age'].unsqueeze(1))
        if self._use_feature("occupation"):
            features.append(batch['occupation'].float().unsqueeze(1))
        if self._use_feature("gender"):
            features.append(batch['gender'].float().unsqueeze(1))
        # Genre (multi-hot)
        if self._use_feature("genre"):
            features.append(batch['genre'].float())
        return torch.cat(features, dim=1)

    def _get_target(self, batch):
        return batch["age"] if self.target_attr == "age" else batch[self.target_attr]


    def training_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        y = self._get_target(batch)

        if self.target_attr == "age":
            # Regression: ensure shapes match [B]
            y = y.float()
            preds = self(x).squeeze(-1)  # [B]
            loss = self.loss_fn(preds, y)
            metric = self.metric(preds * 100, y * 100)
        else:
            # Classification: logits [B, C], targets [B] (long)
            y = y.long()
            logits = self(x)  # [B, C]
            loss = self.loss_fn(logits, y)
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log(f"train_{self.target_attr}_metric", metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        y = self._get_target(batch)

        if self.target_attr == "age":
            y = y.float()
            preds = self(x).squeeze(-1)
            loss = self.loss_fn(preds, y)
            metric = self.metric(preds * 100, y * 100)
        else:
            y = y.long()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_{self.target_attr}_metric", metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = self._prepare_features(batch)
        y = self._get_target(batch)

        if self.target_attr == "age":
            y = y.float()
            preds = self(x).squeeze(-1)
            loss = self.loss_fn(preds, y)
            metric = self.metric(preds * 100, y * 100)
        else:
            y = y.long()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            probs = torch.softmax(logits, dim=-1)
            metric = self.metric(probs, y)

        self.log("test_loss", loss)
        self.log(f"test_{self.target_attr}_metric", metric, prog_bar=True)

    def on_fit_start(self):
        pretty = [f"[{i}] {n}" for i, n in enumerate(self._feature_names())]
        self.print(f"[AttributeClassifier] Using features (target={self.target_attr}): {pretty}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

# -------------------------------
# XGBoost-based attribute auditor
# -------------------------------

def _build_features_from_batch(batch, target_attr: str, include_rating: bool, use_identifiers: bool):
    feats = []
    if include_rating:
        feats.append(batch['rating'].float().unsqueeze(1))
    if use_identifiers:
        feats.extend([
            batch['user_id'].float().unsqueeze(1),
            batch['item_id'].float().unsqueeze(1),
        ])
    if target_attr != "age":
        feats.append(batch['age'].unsqueeze(1))
    if target_attr != "occupation":
        feats.append(batch['occupation'].float().unsqueeze(1))
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