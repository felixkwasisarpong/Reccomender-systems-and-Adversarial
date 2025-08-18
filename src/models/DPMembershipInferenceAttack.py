import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_recall_curve, average_precision_score
)
from scipy.stats import mannwhitneyu
import torch
import inspect
from typing import Optional, Dict, Any
from .DPModel import DPModel
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os

class DPMembershipInferenceAttack(DPModel):
    """
    Membership inference attack head that plugs on top of the DP-trained recommender.
    
    This inherits the full architecture, forward(), and hparams from CustomDP_SGD so you
    can load a privacy-trained checkpoint and immediately evaluate MIA without retraining.
    
    Signals used (per-sample):
      - Confidence score (from normalized absolute error)
      - Loss-based score (MSE or Huber depending on hparams)
      - Entropy-like score (uncertainty proxy on normalized predictions)
    These are ensembled with static weights (0.4/0.4/0.2) into an overall membership score.
    
    The class collects scores during test_step() and aggregates/plots in on_test_epoch_end().
    """

    def __init__(self, *args, **kwargs):
        """
        mia_plot_mode: str, optional
            - 'full' (default): 2x2 grid of plots (ROC, PR, score dist, threshold sweep)
            - 'roc': ROC-only single panel
        """
        super().__init__(*args, **kwargs)
        self.mia_plot_mode = kwargs.get('mia_plot_mode', 'full')
        # Running buffers
        self.scores = []                # ensemble scores
        self.labels = []                # 1=member (positive), 0=non-member
        self.member_scores = []
        self.nonmember_scores = []

        # Individual signals (for diagnostics)
        self.confidence_scores = []
        self.loss_scores = []
        self.entropy_scores = []

        self.pos_label = 1
        self._scores_inverted = False   # whether we flipped scores (non-member mean > member mean)

    # ---------------------------
    # Signal builders
    # ---------------------------
    def compute_confidence_score(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Higher is more likely member.
        Based on normalized absolute error with an exponential transform.
        """
        tmin = float(self.hparams.target_min)
        tmax = float(self.hparams.target_max)
        denom = max(1e-12, (tmax - tmin))
        preds_norm = (preds - tmin) / denom
        targets_norm = (targets - tmin) / denom
        errors = torch.abs(preds_norm - targets_norm)
        # Scale factor (3.0) is a mild temperature; tweakable
        return torch.exp(-errors * 3.0)

    def compute_loss_based_score(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Lower loss ⇒ higher membership probability; map via exp(-loss * 0.5)."""
        if getattr(self.hparams, "loss_function", "MSE").upper() == "MSE":
            losses = (preds - targets) ** 2
        else:  # Huber
            delta = 1.0
            errors = torch.abs(preds - targets)
            losses = torch.where(errors < delta, 0.5 * errors ** 2, delta * (errors - 0.5 * delta))
        return torch.exp(-0.5 * losses)

    def compute_prediction_entropy(self, preds: torch.Tensor) -> torch.Tensor:
        """Entropy proxy from normalized scalar predictions in [0,1]. Lower entropy ⇒ member.
        We map to membership via exp(-entropy).
        """
        tmin = float(self.hparams.target_min)
        tmax = float(self.hparams.target_max)
        denom = max(1e-12, (tmax - tmin))
        p = (preds - tmin) / denom
        p = torch.clamp(p, 1e-2, 1 - 1e-2)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return torch.exp(-entropy)

    # ---------------------------
    # PL hooks
    # ---------------------------
    def on_test_epoch_start(self) -> None:
        # Ensure buffers are empty at the start of evaluation
        self.clear_all_scores()

    def test_step(self, batch, batch_idx):
        # Fetch targets & labels
        targets = batch["rating"].view(-1)
        labels = batch["label"].long().view(-1)  # 1=member, 0=non-member

        # Respect use_attrs, mirroring training inputs
        if getattr(self.hparams, "use_attrs", False) and all(
            k in batch for k in ("gender", "age", "occupation", "genre")
        ):
            model_in = {
                "user_id": batch["user_id"],
                "item_id": batch["item_id"],
                "gender": batch["gender"],
                "age": batch["age"],
                "occupation": batch["occupation"],
                "genre": batch["genre"],
            }
        else:
            model_in = {
                "user_id": batch["user_id"],
                "item_id": batch["item_id"],
            }

        preds = self(model_in)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = preds.view(-1)

        # Individual signals
        conf = self.compute_confidence_score(preds, targets)
        loss_s = self.compute_loss_based_score(preds, targets)
        entr = self.compute_prediction_entropy(preds)

        # Static-weight ensemble
        ensemble = 0.4 * conf + 0.4 * loss_s + 0.2 * entr

        # Persist per-batch scores
        self.confidence_scores.extend(conf.detach().cpu().tolist())
        self.loss_scores.extend(loss_s.detach().cpu().tolist())
        self.entropy_scores.extend(entr.detach().cpu().tolist())
        self.scores.extend(ensemble.detach().cpu().tolist())
        self.labels.extend(labels.detach().cpu().tolist())

        # Split for diagnostics
        member_mask = labels == 1
        nonmember_mask = labels == 0
        if member_mask.any():
            self.member_scores.extend(ensemble[member_mask].detach().cpu().tolist())
        if nonmember_mask.any():
            self.nonmember_scores.extend(ensemble[nonmember_mask].detach().cpu().tolist())

        # Helpful batch-level logs
        self.log("batch_avg_score", ensemble.mean(), on_step=False, on_epoch=True)
        self.log("batch_confidence", conf.mean(), on_step=False, on_epoch=True)
        self.log("batch_loss_score", loss_s.mean(), on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        if not self.scores or len(set(self.labels)) < 2:
            print("Warning: Insufficient data for MIA evaluation")
            return

        y_true = np.asarray(self.labels, dtype=int)
        y_score_raw = np.asarray(self.scores, dtype=float)

        # Orient scores so that higher ⇒ member
        scores_for_metrics = y_score_raw.copy()
        self._scores_inverted = False
        if self.member_scores and self.nonmember_scores:
            m_mean = float(np.mean(self.member_scores))
            nm_mean = float(np.mean(self.nonmember_scores))
            if nm_mean > m_mean:
                scores_for_metrics = 1.0 - scores_for_metrics
                self._scores_inverted = True
        else:
            m_mean = float(np.mean(self.member_scores)) if self.member_scores else float("nan")
            nm_mean = float(np.mean(self.nonmember_scores)) if self.nonmember_scores else float("nan")

        # Primary metrics
        auc = roc_auc_score(y_true, scores_for_metrics)
        fpr, tpr, thr = roc_curve(y_true, scores_for_metrics, pos_label=self.pos_label)
        youden = tpr - fpr
        opt_idx = int(np.argmax(youden))
        opt_thr = float(thr[opt_idx])
        tpr_opt = float(tpr[opt_idx])
        fpr_opt = float(fpr[opt_idx])
        preds_opt = (scores_for_metrics >= opt_thr).astype(int)
        acc = accuracy_score(y_true, preds_opt)
        advantage = tpr_opt - fpr_opt
        prec, rec, _ = precision_recall_curve(y_true, scores_for_metrics, pos_label=self.pos_label)
        ap = average_precision_score(y_true, scores_for_metrics)

        # Log
        self.log("mia_auc", auc, prog_bar=True)
        self.log("mia_accuracy", acc, prog_bar=True)
        self.log("mia_advantage", advantage, prog_bar=True)
        self.log("mia_ap_score", ap, prog_bar=True)

        # Distribution + significance
        if self.member_scores and self.nonmember_scores:
            separation = m_mean - nm_mean
            try:
                m_scores = scores_for_metrics[y_true == 1]
                nm_scores = scores_for_metrics[y_true == 0]
                stat, p_val = mannwhitneyu(m_scores, nm_scores, alternative="greater")
                significant = p_val < 0.05
            except Exception:
                p_val = 1.0
                significant = False

            self.log("member_score_mean", m_mean)
            self.log("nonmember_score_mean", nm_mean)
            self.log("score_separation", separation)
            self.log("mia_p_value", float(p_val))
            self.log("mia_significant", float(significant))
            self.log("mia_scores_inverted", float(self._scores_inverted))
            self.log("mia_n_members", float((y_true == 1).sum()))
            self.log("mia_n_nonmembers", float((y_true == 0).sum()))

            print("\nMembership Inference Attack Results:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Average Precision: {ap:.4f}")
            print(f"  Advantage (TPR-FPR) at J*: {advantage:.4f}")
            print(f"  Optimal threshold (Youden J): {opt_thr:.4f} | TPR: {tpr_opt:.3f} | FPR: {fpr_opt:.3f}")
            print(f"  Member mean: {m_mean:.4f} | Non-member mean: {nm_mean:.4f}")
            if self._scores_inverted:
                print("  Note: Scores were inverted because non-members had higher mean.")

        # Plots + optional CSV dump
        self._create_comprehensive_plots(
            auc, fpr, tpr, prec, rec, ap, opt_thr, preds_opt
        )
        self._dump_scores_csv()

        # clear for next run
        self.clear_all_scores()

    # ---------------------------
    # Visualization & I/O
    # ---------------------------
    def _create_comprehensive_plots(self, auc, fpr, tpr, precision, recall, ap_score, optimal_threshold, predictions):
        y_true = np.asarray(self.labels, dtype=int)
        y_score = np.asarray(self.scores, dtype=float)
        scores_for_metrics = 1.0 - y_score if self._scores_inverted else y_score

        # Always create a single ROC plot with AUC and optimal Youden point
        os.makedirs("mia_results", exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=3, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, alpha=0.7, label="Random (AUC = 0.5)")
        j_idx = int(np.argmax(tpr - fpr))
        plt.scatter(fpr[j_idx], tpr[j_idx], s=80, zorder=5, label="Optimal point")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Membership Inference Attack")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f"mia_results/roc_only_{getattr(self.hparams, 'predict_file', 'predictions')}.png", dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()

        # Also generate signal-wise histograms
        self._create_signal_analysis_plot()

    def _create_signal_analysis_plot(self):
        if not (self.confidence_scores and self.loss_scores and self.entropy_scores):
            return
        os.makedirs("mia_results", exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        signals = [
            (self.confidence_scores, "Confidence Score", axes[0]),
            (self.loss_scores, "Loss-based Score", axes[1]),
            (self.entropy_scores, "Entropy-based Score", axes[2]),
        ]
        for sig, title, ax in signals:
            # Align with labels by index
            mem = [sig[i] for i in range(len(sig)) if self.labels[i] == 1]
            non = [sig[i] for i in range(len(sig)) if self.labels[i] == 0]
            if mem and non:
                bins = np.linspace(0, 1, 30)
                ax.hist(non, bins=bins, alpha=0.7, label='Non-members', density=True)
                ax.hist(mem, bins=bins, alpha=0.7, label='Members', density=True)
                try:
                    auc = roc_auc_score([1 if self.labels[i] == 1 else 0 for i in range(len(sig))], sig)
                    ax.set_title(f"{title}\n(AUC = {auc:.3f})", fontweight='bold')
                except Exception:
                    ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Score')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"mia_results/signal_analysis_{getattr(self.hparams, 'predict_file', 'predictions')}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _dump_scores_csv(self):
        """Optional artifact for downstream analysis."""
        try:
            import csv
            os.makedirs("mia_results", exist_ok=True)
            path = os.path.join("mia_results", f"scores_{getattr(self.hparams, 'predict_file', 'predictions')}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["label", "ensemble", "confidence", "loss_based", "entropy", "inverted"]) 
                inv = int(self._scores_inverted)
                for i in range(len(self.scores)):
                    w.writerow([
                        int(self.labels[i]),
                        float(self.scores[i]),
                        float(self.confidence_scores[i]),
                        float(self.loss_scores[i]),
                        float(self.entropy_scores[i]),
                        inv,
                    ])
        except Exception as e:
            print(f"[MIA] Failed to write CSV: {e}")

    # ---------------------------
    # Utilities
    # ---------------------------
    def clear_all_scores(self):
        self.scores.clear()
        self.labels.clear()
        self.member_scores.clear()
        self.nonmember_scores.clear()
        self.confidence_scores.clear()
        self.loss_scores.clear()
        self.entropy_scores.clear()

    # ---------------------------
    # Loader: build from DP checkpoint while preserving dims
    # ---------------------------
    @classmethod
    def load_from_dp_checkpoint(cls, checkpoint_path, **override_kwargs):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        raw_hparams = ckpt.get("hyper_parameters", {})
        raw_hparams = {**raw_hparams, **override_kwargs}

        # Normalize embedding dimension key from checkpoints
        if "embedding_dim" in raw_hparams and "embed_dim" not in raw_hparams:
            raw_hparams["embed_dim"] = raw_hparams.pop("embedding_dim")

        alias_map = {
            "n_users": "num_users",
            "n_items": "num_items",
            "n_genders": "num_genders",
            "n_occupations": "num_occupations",
            "genre_size": "genre_dim",
        }
        for old, new in alias_map.items():
            if old in raw_hparams and new not in raw_hparams:
                raw_hparams[new] = raw_hparams.pop(old)

        # Fallback: infer embed_dim from state_dict if absent in hyperparameters
        sd = ckpt.get("state_dict", {})
        if "embed_dim" not in raw_hparams and "user_embedding.weight" in sd:
            raw_hparams["embed_dim"] = int(sd["user_embedding.weight"].shape[1])

        required = ["num_users", "num_items", "num_genders", "num_occupations", "genre_dim"]
        missing = [r for r in required if r not in raw_hparams]
        if missing:
            raise ValueError(f"Missing required hyperparameters in checkpoint: {missing}\nAvailable keys: {list(raw_hparams.keys())}")

        allowed_keys = set()
        for base in cls.mro():
            if "__init__" in base.__dict__:
                try:
                    sig = inspect.signature(base.__init__)
                    for name, param in sig.parameters.items():
                        if name not in {"self", "args", "kwargs"}:
                            allowed_keys.add(name)
                except (TypeError, ValueError):
                    pass

        allowed_keys.update({
            "embed_dim", "hidden_dim", "mlp_dims", "dropout", "learning_rate", "l2_penalty",
            "loss_function", "target_min", "target_max", "use_attrs", "predict_file",
            "enable_dp", "noise_type", "noise_scale", "dp_microbatch_size", "clip_norm", "delta",
        })

        def _is_allowed(k: str) -> bool:
            if k.startswith("_"):
                return False
            if k in {"_class_path", "class_path", "_target", "init_args", "_init_args"}:
                return False
            return k in allowed_keys

        init_kwargs = {k: v for k, v in raw_hparams.items() if _is_allowed(k)}

        # Debug print for resolved embed_dim
        print("Resolved embed_dim:", init_kwargs.get("embed_dim"))

        dbg_keys = [k for k in ["embed_dim", "hidden_dim", "num_users", "num_items"] if k in init_kwargs]
        print("Final init_kwargs (subset):", {k: init_kwargs[k] for k in dbg_keys})

        model = cls(**init_kwargs)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model