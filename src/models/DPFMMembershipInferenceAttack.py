import os
import inspect
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from models.DPFM_GANTrainer import DPFM_GANTrainer  # adjust if needed


class DPFMMembershipInferenceAttack(DPFM_GANTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = []
        self.labels = []
        self.member_scores = []
        self.nonmember_scores = []
        
        # Store individual signal scores for analysis
        self.confidence_scores = []
        self.loss_scores = []
        self.entropy_scores = []
        # By convention, label 1 == member (positive class)
        self.pos_label = 1
        self._scores_inverted = False
        # Diagnostics: store abs errors by group
        self._member_abs_err = []
        self._nonmember_abs_err = []

    def forward(self, batch):
        return super().forward(batch)

    def compute_confidence_score(self, preds, targets):
        """
        Compute confidence score based on prediction accuracy.
        Higher confidence suggests membership.
        """
        # Normalize predictions and targets to [0,1]
        preds_norm = (preds - self.hparams.target_min) / (self.hparams.target_max - self.hparams.target_min)
        targets_norm = (targets - self.hparams.target_min) / (self.hparams.target_max - self.hparams.target_min)
        
        # Calculate prediction confidence (lower error = higher confidence)
        errors = torch.abs(preds_norm - targets_norm)
        confidence = torch.exp(-errors * 3.0)  # Exponential confidence function
        
        return confidence

    def compute_loss_based_score(self, preds, targets):
        """
        Compute membership score based on individual sample loss.
        Lower loss suggests membership (overfitting signal).
        """
        # Calculate per-sample loss
        if self.hparams.loss_function == "MSE":
            losses = (preds - targets) ** 2
        else:  # Huber loss
            errors = torch.abs(preds - targets)
            losses = torch.where(errors < 1.0, 0.5 * errors**2, errors - 0.5)
        
        # Convert to membership probability (lower loss = higher membership probability)
        membership_prob = torch.exp(-losses * 0.5)
        return membership_prob

    def compute_prediction_entropy(self, preds):
        """
        Compute prediction entropy as membership signal.
        Lower entropy (more certain predictions) suggests membership.
        """
        # Normalize predictions to [0,1] range
        preds_norm = (preds - self.hparams.target_min) / (self.hparams.target_max - self.hparams.target_min)
        preds_norm = torch.clamp(preds_norm, 0.01, 0.99)  # Avoid log(0)
        
        # Compute entropy-like measure
        entropy = -(preds_norm * torch.log(preds_norm) + (1 - preds_norm) * torch.log(1 - preds_norm))
        membership_score = torch.exp(-entropy)  # Lower entropy = higher membership
        
        return membership_score

    def test_step(self, batch, batch_idx):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        targets = batch["rating"]
        labels = batch["label"].long().view(-1)
        
        # Prepare input batch - Fixed: use self.hparams.use_attrs
        if self.hparams.use_attrs and all(k in batch for k in ["gender", "age", "occupation", "genre"]):
            batch_input = {
                "user_id": user_ids,
                "item_id": item_ids,
                "gender": batch["gender"],
                "age": batch["age"],
                "occupation": batch["occupation"],
                "genre": batch["genre"],
            }
        else:
            batch_input = {
                "user_id": user_ids,
                "item_id": item_ids,
            }

        # Get predictions
        preds = self(batch_input)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Compute multiple membership signals
        confidence_scores = self.compute_confidence_score(preds, targets)
        loss_scores = self.compute_loss_based_score(preds, targets)
        entropy_scores = self.compute_prediction_entropy(preds)
        
        # Ensemble approach: combine multiple signals with weights
        ensemble_scores = (
            0.4 * confidence_scores + 
            0.4 * loss_scores + 
            0.2 * entropy_scores
        )
        
        # Store individual scores for analysis
        self.confidence_scores.extend(confidence_scores.tolist())
        self.loss_scores.extend(loss_scores.tolist())
        self.entropy_scores.extend(entropy_scores.tolist())
        
        # Store scores separately for analysis
        member_mask = labels == 1
        nonmember_mask = labels == 0
        
        if member_mask.any():
            self.member_scores.extend(ensemble_scores[member_mask].tolist())
        if nonmember_mask.any():
            self.nonmember_scores.extend(ensemble_scores[nonmember_mask].tolist())
        
        # Store all scores for ROC calculation
        self.scores.extend(ensemble_scores.tolist())
        self.labels.extend(labels.tolist())
        
        # Log batch-level metrics
        if len(ensemble_scores) > 0:
            self.log("batch_avg_score", ensemble_scores.mean(), on_step=False, on_epoch=True)
            self.log("batch_confidence", confidence_scores.mean(), on_step=False, on_epoch=True)
            self.log("batch_loss_score", loss_scores.mean(), on_step=False, on_epoch=True)

        # Diagnostics: MAE by group
        with torch.no_grad():
            abs_err = (preds - targets).abs()
            if member_mask.any():
                mae_m = abs_err[member_mask].mean()
                self._member_abs_err.append(float(mae_m.detach().cpu()))
                try:
                    self.log("mia/batch_member_mae", mae_m, on_step=False, on_epoch=True)
                except Exception:
                    pass
            if nonmember_mask.any():
                mae_n = abs_err[nonmember_mask].mean()
                self._nonmember_abs_err.append(float(mae_n.detach().cpu()))
                try:
                    self.log("mia/batch_nonmember_mae", mae_n, on_step=False, on_epoch=True)
                except Exception:
                    pass

    def on_test_epoch_end(self):
        if not self.scores or len(set(self.labels)) < 2:
            print("Warning: Insufficient data for MIA evaluation")
            return

        # Ensure numpy arrays
        y_true = np.asarray(self.labels, dtype=int)
        y_score = np.asarray(self.scores, dtype=float)

        # Orient scores so that higher score => member (positive class = 1)
        # If non-members have higher mean, flip the scores for metrics and curves
        if len(self.member_scores) > 0 and len(self.nonmember_scores) > 0:
            member_mean = float(np.mean(self.member_scores))
            nonmember_mean = float(np.mean(self.nonmember_scores))
            scores_for_metrics = y_score.copy()
            self._scores_inverted = False
            if nonmember_mean > member_mean:
                scores_for_metrics = 1.0 - scores_for_metrics
                self._scores_inverted = True
        else:
            scores_for_metrics = y_score
            member_mean = float(np.mean(self.member_scores)) if self.member_scores else float('nan')
            nonmember_mean = float(np.mean(self.nonmember_scores)) if self.nonmember_scores else float('nan')

        # Primary metrics on oriented scores
        auc = roc_auc_score(y_true, scores_for_metrics)
        fpr, tpr, thresholds = roc_curve(y_true, scores_for_metrics, pos_label=self.pos_label)

        # Choose optimal threshold via Youden's J
        youden = tpr - fpr
        optimal_idx = int(np.argmax(youden))
        optimal_threshold = float(thresholds[optimal_idx])
        tpr_opt = float(tpr[optimal_idx])
        fpr_opt = float(fpr[optimal_idx])

        # Predictions and accuracy at optimal threshold
        predictions = (scores_for_metrics >= optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)

        # Advantage (TPR - FPR) at the optimal threshold
        advantage = tpr_opt - fpr_opt

        # Precision-recall metrics
        precision, recall, _ = precision_recall_curve(y_true, scores_for_metrics, pos_label=self.pos_label)
        ap_score = average_precision_score(y_true, scores_for_metrics)

        # Log main metrics
        self.log("mia_auc", auc, prog_bar=True)
        self.log("mia_accuracy", accuracy, prog_bar=True)
        self.log("mia_advantage", advantage, prog_bar=True)
        self.log("mia_ap_score", ap_score, prog_bar=True)

        # Distribution stats
        if self.member_scores and self.nonmember_scores:
            separation = member_mean - nonmember_mean

            # Significance test on **oriented** scores
            try:
                from scipy.stats import mannwhitneyu
                m_scores = scores_for_metrics[y_true == 1]
                nm_scores = scores_for_metrics[y_true == 0]
                statistic, p_value = mannwhitneyu(m_scores, nm_scores, alternative='greater')
                significant = p_value < 0.05
            except Exception:
                p_value = 1.0
                significant = False

            self.log("member_score_mean", member_mean)
            self.log("nonmember_score_mean", nonmember_mean)
            self.log("score_separation", separation)
            self.log("mia_p_value", float(p_value))
            self.log("mia_significant", float(significant))
            self.log("mia_scores_inverted", float(self._scores_inverted))
            self.log("mia_n_members", float((y_true == 1).sum()))
            self.log("mia_n_nonmembers", float((y_true == 0).sum()))

            print(f"\nMembership Inference Attack Results:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Average Precision: {ap_score:.4f}")
            print(f"  Advantage (TPR-FPR) at J*: {advantage:.4f}")
            print(f"  Optimal threshold (Youden J): {optimal_threshold:.4f} | TPR: {tpr_opt:.3f} | FPR: {fpr_opt:.3f}")
            print(f"  Member scores (mean ± std): {member_mean:.4f} ± {np.std(self.member_scores):.4f}")
            print(f"  Non-member scores (mean ± std): {nonmember_mean:.4f} ± {np.std(self.nonmember_scores):.4f}")
            print(f"  Score separation: {separation:.4f}")
            print(f"  Statistical significance (p < 0.05): {significant} (p = {p_value:.4f})")
            if self._scores_inverted:
                print("  Note: Scores were inverted for metrics because non-members had higher mean scores.")

        # Call plot function with predictions
        self._create_comprehensive_plots(
            auc, fpr, tpr, precision, recall,
            ap_score, optimal_threshold, predictions
        )

        self.clear_all_scores()
        # Print final MAE diagnostics
        if self._member_abs_err and self._nonmember_abs_err:
            m_mae = float(np.mean(self._member_abs_err))
            n_mae = float(np.mean(self._nonmember_abs_err))
            print(f"[MIA][diag] mean MAE — members={m_mae:.4f}, nonmembers={n_mae:.4f}, Δ={m_mae - n_mae:.4f}")
        self._member_abs_err.clear(); self._nonmember_abs_err.clear()

    def create_comprehensive_plots(self, auc, fpr, tpr, precision, recall,
                                    ap_score, optimal_threshold, predictions):
        # Use oriented scores for curves and threshold sweeps
        y_true = np.asarray(self.labels, dtype=int)
        y_score = np.asarray(self.scores, dtype=float)
        if self._scores_inverted:
            scores_for_metrics = 1.0 - y_score
        else:
            scores_for_metrics = y_score

        os.makedirs("mia_results", exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. ROC Curve
        ax1.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {auc:.3f})', color='blue')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, alpha=0.7, label='Random (AUC = 0.5)')
        ax1.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)],
                    color='red', s=100, zorder=5, label=f'Optimal point')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve - Membership Inference Attack', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # 2. Score Distribution
        if self.member_scores and self.nonmember_scores:
            bins = np.linspace(0, 1, 40)
            ax2.hist(self.nonmember_scores, bins=bins, alpha=0.7, label='Non-members',
                    color='red', density=True, edgecolor='black', linewidth=0.5)
            ax2.hist(self.member_scores, bins=bins, alpha=0.7, label='Members',
                    color='blue', density=True, edgecolor='black', linewidth=0.5)
            ax2.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
                        label=f'Optimal threshold = {optimal_threshold:.3f}')
            ax2.set_xlabel('Membership Score', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Score Distribution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax3.plot(recall, precision, linewidth=3, label=f'PR (AP = {ap_score:.3f})', color='green')
        ax3.axhline(y=np.mean(y_true), color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label=f'Random (AP = {np.mean(y_true):.3f})')
        ax3.set_xlabel('Recall', fontsize=12)
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])

        # 4. Attack Success Rate vs Threshold
        thresholds_range = np.linspace(0, 1, 100)
        success_rates = []
        for thresh in thresholds_range:
            pred_labels = (scores_for_metrics >= thresh).astype(int)
            success_rate = accuracy_score(y_true, pred_labels)
            success_rates.append(success_rate)

        ax4.plot(thresholds_range, success_rates, linewidth=3, color='purple')
        ax4.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random guessing')
        ax4.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Optimal threshold')
        ax4.scatter(optimal_threshold, accuracy_score(y_true, predictions),
                    color='red', s=100, zorder=5, label=f'Optimal accuracy = {accuracy_score(y_true, predictions):.3f}')
        ax4.set_xlabel('Threshold', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Attack Success Rate vs Threshold', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f"mia_results/comprehensive_mia_{self.hparams.predict_file}.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Generate signal-wise plots
        self.create_signal_analysis_plot()

    def create_signal_analysis_plot(self):
        """Create plot analyzing individual attack signals"""
        if not (self.confidence_scores and self.loss_scores and self.entropy_scores):
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        signals = [
            (self.confidence_scores, "Confidence Score", axes[0]),
            (self.loss_scores, "Loss-based Score", axes[1]),
            (self.entropy_scores, "Entropy-based Score", axes[2])
        ]
        
        for signal_scores, title, ax in signals:
            # Split by membership
            member_signal = [signal_scores[i] for i in range(len(signal_scores)) if self.labels[i] == 1]
            nonmember_signal = [signal_scores[i] for i in range(len(signal_scores)) if self.labels[i] == 0]
            
            if member_signal and nonmember_signal:
                bins = np.linspace(0, 1, 30)
                ax.hist(nonmember_signal, bins=bins, alpha=0.7, label='Non-members', 
                       color='red', density=True)
                ax.hist(member_signal, bins=bins, alpha=0.7, label='Members', 
                       color='blue', density=True)
                
                # Calculate individual AUC
                signal_labels = [1 if self.labels[i] == 1 else 0 for i in range(len(signal_scores))]
                try:
                    signal_auc = roc_auc_score(signal_labels, signal_scores)
                    ax.set_title(f'{title}\n(AUC = {signal_auc:.3f})', fontweight='bold')
                except:
                    ax.set_title(title, fontweight='bold')
                
                ax.set_xlabel('Score')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"mia_results/signal_analysis_{self.hparams.predict_file}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def clear_all_scores(self):
        """Clear all stored scores"""
        self.scores.clear()
        self.labels.clear()
        self.member_scores.clear()
        self.nonmember_scores.clear()
        self.confidence_scores.clear()
        self.loss_scores.clear()
        self.entropy_scores.clear()
        
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
    @classmethod
    def load_from_dp_checkpoint(cls, checkpoint_path, **override_kwargs):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        raw = ckpt.get("hyper_parameters", {}) or {}

        # Some checkpoints store params under 'init_args'
        if isinstance(raw, dict) and "init_args" in raw and isinstance(raw["init_args"], dict):
            raw = raw["init_args"]

        # Strip Lightning/Trainer artifacts and private keys
        forbidden = {
            "_class_path", "class_path", "_target", "_init_args", "init_args",
            "_lightning_module", "_recursive", "_convert_", "_kwargs_",
        }
        base = {k: v for k, v in raw.items() if not k.startswith("_") and k not in forbidden}

        # Provide dataset defaults if missing
        defaults = dict(
            num_users=943,
            num_items=1682,
            num_genders=2,
            num_occupations=21,
            genre_dim=19,
        )

        # Map common aliases from older checkpoints
        alias = {
            "n_users": "num_users",
            "n_items": "num_items",
            "n_genders": "num_genders",
            "n_occupations": "num_occupations",
            "genre_size": "genre_dim",
            "embedding_dim": "embed_dim",
        }
        for old, new in alias.items():
            if old in base and new not in base:
                base[new] = base.pop(old)

        # Allowed constructor args for DPFM_GANTrainer/BaseModel family
        allowed = {
            # dataset/structure
            "num_users", "num_items", "num_genders", "num_occupations", "genre_dim",
            # architecture
            "embed_dim", "mlp_dims", "dropout", "use_attrs",
            # adversarial
            "adv_all_hidden_dims", "adv_start_epoch", "adv_ramp_epochs",
            "adv_lambda_start", "adv_lambda_end", "adv_lambda_cap", "adv_update_freq",
            # regularization/calibration
            "repr_dropout", "l2_penalty", "loss_function",
            # training
            "learning_rate",
            # output
            "target_min", "target_max", "predict_file",
            # dp args
            "enable_dp", "target_epsilon", "target_delta", "max_grad_norm",
            "noise_multiplier",
        }

        # Build init kwargs from defaults + sanitized base + user overrides
        merged = {**defaults, **base, **override_kwargs}
        init_kwargs = {k: merged[k] for k in list(merged.keys()) if k in allowed}

        # Instantiate and load weights
        model = cls(**init_kwargs)
        state_dict = ckpt.get("state_dict", {})
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[MIA] Missing keys:", missing[:10], "...")
        if unexpected:
            print("[MIA] Unexpected keys (ignored):", unexpected[:10], "...")
        return model
