# wgan.py
# WGAN-GP LightningModule + 1×4 plotting callback

from __future__ import annotations
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import os

def make_mlp(in_dim: int, out_dim: int, hidden: List[int], out_act: Optional[nn.Module] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.2, inplace=False)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    if out_act is not None:
        layers += [out_act]
    return nn.Sequential(*layers)

class WGAN(pl.LightningModule):
    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        lr: float = 2e-4,
        n_critic: int = 10,
        lambda_gp: float = 2.5,
        gen_hidden: List[int] = (384, 384, 384, 384),
        disc_hidden: List[int] = (768, 768, 768, 768),
        cond_dim: int = 0,


        # --- new: plot/schema keys coming from YAML ---
        pred_file: str = "base_strong",
        condkl_gender_col: int = 3,
        condkl_occupation_col: int = 4,
        condkl_num_occupations: int = 21,
        condkl_age_col: int = 2,
        condkl_age_bins: int = 7,
        rating_idx: int = 24,
        num_attack_samples: int = 100000,

        # --- auxiliary attribute guidance (disabled by default) ---
        lambda_aux_gender: float = 0.0,
        lambda_aux_occ: float = 0.0,
        lambda_aux_age: float = 0.0,
        attr_hidden: int = 256,
        rebalance_cond_sampling: bool = False,
        cond_include_genre: bool = False,
        cond_genre_proj_dim: int = 0,
        lambda_genre_kl: float = 0.0,
        lambda_aux_genre: float = 0.0,
        occ_class_weights: Optional[List[float]] = None,

        # --- optional: accept these so YAML stays compatible (unused by core WGAN) ---
        use_onehot_occ: bool = False,
        genre_dim: int = 0,

    ):
        super().__init__()
        # Save everything so LightningCLI validation passes
        self.save_hyperparameters()

        self.noise_dim = int(noise_dim)
        self.output_dim = int(output_dim)
        self.lr = float(lr)
        self.n_critic = int(n_critic)
        self.lambda_gp = float(lambda_gp)
        self.cond_dim = int(cond_dim)

        # --- dynamic schema (to interoperate with WGANInputDataset) ---
        self.schema_ready = False
        self._occ_onehot = False
        self._occ_start = 4
        self._occ_width = 1
        self._occ_classes = int(getattr(self.hparams, "condkl_num_occupations", 21))
        self._genre_dim = 0
        self._gender_idx = int(getattr(self.hparams, "condkl_gender_col", 3))
        self._age_idx = int(getattr(self.hparams, "condkl_age_col", 2))
        # Rating defaults to last column when using dynamic layout
        self._rating_idx = int(self.output_dim) - 1
        # --- genre conditioning / aux knobs ---
        self.cond_include_genre = bool(getattr(self.hparams, "cond_include_genre", False))
        self.cond_genre_proj_dim = int(getattr(self.hparams, "cond_genre_proj_dim", 0))
        self.lambda_genre_kl = float(getattr(self.hparams, "lambda_genre_kl", 0.0))
        self.lambda_aux_genre = float(getattr(self.hparams, "lambda_aux_genre", 0.0))
        self._genre_start = None  # set in _maybe_infer_schema
        self.cond_genre_proj = None  # lazy nn.Linear if projecting genre for cond

        lam_g  = float(getattr(self.hparams, "lambda_aux_gender", 0.0))
        lam_o  = float(getattr(self.hparams, "lambda_aux_occ", 0.0))
        lam_a  = float(getattr(self.hparams, "lambda_aux_age", 0.0))
        lam_gn = float(getattr(self.hparams, "lambda_aux_genre", 0.0))
        self.attr_enabled = (lam_g > 0.0) or (lam_o > 0.0) or (lam_a > 0.0) or (lam_gn > 0.0)

        if self.attr_enabled:
            hid = int(getattr(self.hparams, "attr_hidden", 256))
            age_bins = int(getattr(self.hparams, "condkl_age_bins", 7))
            self.C_body   = make_mlp(self.output_dim, hid, [hid, hid])
            self.C_gender = nn.Linear(hid, 1)
            self.C_occ    = nn.Linear(hid, self._occ_classes)
            self.C_age    = nn.Linear(hid, age_bins)
            self.bce      = nn.BCEWithLogitsLoss()
            # optional: prior reweighting for occupation
            occ_w = getattr(self.hparams, "occ_class_weights", None)
            if occ_w is not None:
                self.occ_class_weights_buf = torch.tensor(occ_w, dtype=torch.float32)
            print(f"[WGAN][aux] enabled heads: gender={lam_g>0}, occ={lam_o>0}, age={lam_a>0}, genre={lam_gn>0} | hid={hid}")
        # --- compute effective conditioning dimension (accounts for optional genre) ---
        # We rely on YAML-only schema, so genre_dim is available now.
        self._genre_dim = int(getattr(self.hparams, "genre_dim", 0))  # ensure cached here
        extra_cond = 0
        if self.cond_include_genre and self._genre_dim > 0:
            if self.cond_genre_proj_dim > 0:
                extra_cond = int(self.cond_genre_proj_dim)
                # Build projection layer early so cond length stays consistent
                if self.cond_genre_proj is None:
                    self.cond_genre_proj = nn.Linear(self._genre_dim, self.cond_genre_proj_dim, bias=True)
            else:
                extra_cond = int(self._genre_dim)
        self._cond_dim_eff = int(self.cond_dim + extra_cond)

        # ---- runtime validation (avoids LightningCLI parser checks) ----
        if self.noise_dim <= 0:
            raise ValueError(f"noise_dim must be > 0, got {self.noise_dim}")
        if self.output_dim != 25:
            print(f"[WGAN][warn] output_dim={self.output_dim} (expected 25 for canonical layout)")
        if self.cond_dim not in (0, 1, 3):
            print(f"[WGAN][warn] cond_dim={self.cond_dim} (expected one of 0,1,3); continuing untested setup")

        # used by the plotting callback
        self.pred_file = str(pred_file)

        # ---- always use data-distribution mode ----
        # use effective cond dim (may include projected/raw genre)
        zdim = self.noise_dim + self._cond_dim_eff
        self.G = make_mlp(zdim, self.output_dim, list(gen_hidden), out_act=nn.Tanh())
        self.D = make_mlp(self.output_dim + self._cond_dim_eff, 1, list(disc_hidden))

        self.automatic_optimization = False

        # cache a few real batches for plotting fallback when datamodule is unavailable
        self._real_cache: List[torch.Tensor] = []

        # ---- plotting cache for surrogate mode (used if no DM/loader available) ----
        try:
            cap_default = int(getattr(self.hparams, 'num_attack_samples', 100000))
        except Exception:
            cap_default = 100000
        self._plot_cap = max(1000, min(50000, cap_default))  # keep memory modest
        self._plot_real_buf: List[torch.Tensor] = []  # list of 1-D cpu tensors
        self._plot_fake_buf: List[torch.Tensor] = []
        self._plot_count = 0




    # ----- utils -----
    def _noise(self, n: int, device=None):
        device = device or self.device
        return torch.randn(n, self.noise_dim, device=device)

    def _build_cond_from_real(self, real: torch.Tensor) -> Optional[torch.Tensor]:
        self._maybe_infer_schema()
        cond_dim = int(getattr(self.hparams, "cond_dim", self.cond_dim))
        if cond_dim < 0:
            cond_dim = 0
        if cond_dim == 0 and not self.cond_include_genre:
            return None

        def to01(x):
            return (x + 1.0) * 0.5

        # gender
        g = to01(real[:, self._gender_idx:self._gender_idx+1]).clamp(0, 1)

        # occupation (scalar id in [0,1])
        if self._occ_onehot:
            occ_block = to01(real[:, self._occ_start:self._occ_start + self._occ_width])
            occ_id = torch.argmax(occ_block, dim=1, keepdim=True).float()
            occ_scalar01 = (occ_id / float(max(1, self._occ_classes - 1))).clamp(0, 1)
        else:
            occ_scalar01 = to01(real[:, self._occ_start:self._occ_start+1]).clamp(0, 1)

        # age → nearest bucket 0..6, scaled
        age_bins = int(getattr(self.hparams, "condkl_age_bins", 7))
        age01 = to01(real[:, self._age_idx]).clamp(0, 1)
        codes = real.new_tensor([1., 18., 25., 35., 45., 50., 56.]) / 100.0
        diffs = (age01.unsqueeze(1) - codes.unsqueeze(0)).abs()
        age_bucket = torch.argmin(diffs, dim=1).float().unsqueeze(1)
        age_scalar = (age_bucket / float(max(1, age_bins - 1))).clamp(0, 1)

        parts = []
        if cond_dim >= 1:
            parts.append(g)
        if cond_dim >= 3:
            parts.extend([occ_scalar01, age_scalar])

        # optional genre at end
        if self.cond_include_genre and self._genre_dim and self._genre_dim > 0:
            gstart = int(self._genre_start)
            gend = gstart + int(self._genre_dim)
            genre01 = to01(real[:, gstart:gend]).clamp(0, 1)
            if self.cond_genre_proj is not None:
                genre01 = torch.sigmoid(self.cond_genre_proj(genre01))
            parts.append(genre01)

        if not parts:
            return None
        return torch.cat(parts, dim=1).detach()

    def _cond_with_genre(self, cond_base: Optional[torch.Tensor], real_pm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Append genre conditioning (projected or raw) to cond_base if enabled.
        If cond_base is None and genre conditioning is enabled, create a genre-only cond from real_pm (or zeros if real_pm is None).
        Expects real_pm in [-1,1] space.
        """
        # If genre cond is disabled, return the base as-is
        if not (self.cond_include_genre and self._genre_dim and self._genre_dim > 0):
            return cond_base

        # Build genre part from real batch if available; else zeros
        B = 0
        device = self.device
        genre_part = None
        if real_pm is not None and self._genre_start is not None:
            device = real_pm.device
            s = int(self._genre_start)
            e = s + int(self._genre_dim)
            # real_pm is in [-1,1] during training; convert to [0,1]
            genre01 = ((real_pm[:, s:e] + 1.0) * 0.5).clamp(0, 1)
            if self.cond_genre_proj is not None:
                genre_part = torch.sigmoid(self.cond_genre_proj(genre01))
            else:
                genre_part = genre01
            B = genre_part.size(0)
        else:
            # fall back to zeros if no real batch provided
            if cond_base is not None:
                B = cond_base.size(0)
            genre_part = torch.zeros(B, int(self._genre_dim), device=device)
            if self.cond_genre_proj is not None:
                genre_part = torch.sigmoid(self.cond_genre_proj(genre_part))

        if cond_base is None:
            return genre_part.detach()
        else:
            return torch.cat([cond_base.to(device), genre_part], dim=1).detach()
    def _maybe_infer_schema(self):
        """Infer column indices and shapes directly from YAML hparams. 
        This ignores dataset introspection and only uses provided config."""
        if self.schema_ready:
            return

        # Occupation block setup
        self._occ_onehot = bool(getattr(self.hparams, "use_onehot_occ", False))
        self._occ_classes = int(getattr(self.hparams, "condkl_num_occupations", 21))
        self._occ_start = int(getattr(self.hparams, "condkl_occupation_col", 4))
        self._occ_width = self._occ_classes if self._occ_onehot else 1

        # Genre
        self._genre_dim = int(getattr(self.hparams, "genre_dim", 0))
        self._genre_start = self._occ_start + self._occ_width if self._genre_dim > 0 else None

        # Age & gender
        self._age_idx = int(getattr(self.hparams, "condkl_age_col", 2))
        self._gender_idx = int(getattr(self.hparams, "condkl_gender_col", 3))

        # Rating index
        yaml_rating = getattr(self.hparams, "rating_idx", None)
        if yaml_rating is not None:
            self._rating_idx = int(yaml_rating)
        else:
            self._rating_idx = int(self.output_dim) - 1

        self.schema_ready = True

    def on_fit_start(self):
        try:
            self._maybe_infer_schema()
            print(f"[WGAN][schema] age_idx={self._age_idx} gender_idx={self._gender_idx} "
                  f"occ_onehot={self._occ_onehot} occ_start={self._occ_start} occ_width={self._occ_width} "
                  f"rating_idx={self._rating_idx} occ_classes={self._occ_classes}")
            try:
                yaml_rating = getattr(self.hparams, "rating_idx", None)
                if yaml_rating is not None:
                    print(f"[WGAN][schema] rating_idx overridden by YAML: {int(yaml_rating)}")
            except Exception:
                pass
            if self.cond_include_genre and self._genre_dim > 0 and self.cond_genre_proj is not None:
                print(f"[WGAN][schema] built cond_genre_proj: {self._genre_dim} -> {self.cond_genre_proj_dim}")
            print(f"[WGAN][schema] _cond_dim_eff={self._cond_dim_eff}")
        except Exception as e:
            print(f"[WGAN][schema][WARN] {e}")

            
    def _slice_genre01_from_pm(self, x_pm: torch.Tensor):
        """Given x in [-1,1], return genre slice in [0,1] or None if genre absent."""
        if not (self._genre_dim and self._genre_dim > 0 and self._genre_start is not None):
            return None
        s = int(self._genre_start)
        e = s + int(self._genre_dim)
        return ((x_pm[:, s:e] + 1.0) * 0.5).clamp(0, 1)

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        if getattr(self, "attr_enabled", False):
            params = list(self.C_body.parameters()) + list(self.C_gender.parameters()) + list(self.C_occ.parameters()) + list(self.C_age.parameters())
            opt_c = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.9))
            return [opt_d, opt_g, opt_c]
        return [opt_d, opt_g]

    def _critic(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            cond = cond.to(x.device)
            x = torch.cat([x, cond], dim=1)
        return self.D(x)

    def _grad_penalty(self, real: torch.Tensor, fake: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = real.size(0)
        eps = torch.rand(b, 1, device=real.device).expand_as(real)
        inter = eps * real + (1.0 - eps) * fake
        inter.requires_grad_(True)
        d_inter = self._critic(inter, cond)
        ones = torch.ones_like(d_inter)
        grads = torch.autograd.grad(d_inter, inter, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads = grads.view(b, -1)
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def sample(self, n: int, cond: Optional[torch.Tensor] = None, device=None, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate n samples.
        - Data-distribution mode: returns samples in [-1,1], conditioning on `cond` if provided.
        - Surrogate mode: if `x` is provided, returns \hat{y} ≈ f(x); otherwise uses zeros for x.
        """
        device = device or self.device
        z = self._noise(n, device=device)

        # data-distribution mode (original behavior)
        if cond is not None:
            cond = cond.to(device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(1)
            if cond.size(0) != n:
                cond = cond[:n]
            eff = int(getattr(self, "_cond_dim_eff", self.cond_dim))
            if cond.size(1) != eff:
                # pad with zeros or truncate to expected cond dim
                if cond.size(1) < eff:
                    pad = torch.zeros(n, eff - cond.size(1), device=device, dtype=cond.dtype)
                    cond = torch.cat([cond, pad], dim=1)
                    print(f"[WARN][WGAN] cond dim mismatch: got {cond.size(1) - pad.size(1)} -> padded to {eff}")
                else:
                    cond = cond[:, :eff]
                    print(f"[WARN][WGAN] cond dim mismatch: got more ({cond.size(1)}) -> truncated to {eff}")
            z = torch.cat([z, cond], dim=1)
        x_out = self.G(z)
        return x_out

    def _attr_forward(self, x: torch.Tensor):
        """Forward through auxiliary heads. Returns (gender_logit[B], occ_logits[B,K], age_logits[B,A])."""
        h = self.C_body(x)
        g = self.C_gender(h).squeeze(1)
        o = self.C_occ(h)
        a = self.C_age(h)
        return g, o, a

    def _set_attr_grad(self, flag: bool):
        if not getattr(self, "attr_enabled", False):
            return
        for m in (self.C_body, self.C_gender, self.C_occ, self.C_age):
            for p in m.parameters():
                p.requires_grad = flag

    def _make_balanced_cond(self, b: int, device=None) -> Optional[torch.Tensor]:
        device = device or self.device
        cond_dim = int(getattr(self.hparams, "cond_dim", self.cond_dim))
        if cond_dim <= 0:
            return None
        occ_classes = int(getattr(self.hparams, "condkl_num_occupations", 21))
        age_bins = int(getattr(self.hparams, "condkl_age_bins", 7))
        if cond_dim == 1:
            g = (torch.rand(b, 1, device=device) < 0.5).float()
            return g
        elif cond_dim == 3:
            g = (torch.rand(b, 1, device=device) < 0.5).float()
            occ = torch.randint(0, max(1, occ_classes), (b, 1), device=device).float() / float(max(1, occ_classes - 1))
            age = torch.randint(0, max(1, age_bins), (b, 1), device=device).float() / float(max(1, age_bins - 1))
            return torch.cat([g, occ, age], dim=1)
        else:
            return torch.zeros(b, cond_dim, device=device)


    def training_step(self, batch, batch_idx):
        # -------------------- ORIGINAL DATA-DISTRIBUTION MODE --------------------
        real = batch  # [-1,1]
        try:
            if len(self._real_cache) < 8:
                self._real_cache.append(real.detach().cpu())
        except Exception:
            pass

        cond_real = self._build_cond_from_real(real)
        b = real.size(0)
        opts = self.optimizers()
        opt_d = opt_g = opt_c = None
        if isinstance(opts, (list, tuple)):
            if len(opts) == 3:
                opt_d, opt_g, opt_c = opts
            else:
                opt_d, opt_g = opts
        else:
            opt_d = opts

        # D updates (toggle optimizer to isolate graphs)
        self.toggle_optimizer(opt_d)
        for _ in range(self.n_critic):
            fake_adv = self.sample(b, cond=cond_real).detach()
            d_real = self._critic(real, cond_real)
            d_fake = self._critic(fake_adv, cond_real)
            gp = self._grad_penalty(real, fake_adv, cond=cond_real)
            d_loss = (d_fake.mean() - d_real.mean()) + self.lambda_gp * gp

            opt_d.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Log D statistics
        try:
            self.log("critic_loss", d_loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
            self.log("gp", gp.detach(), prog_bar=False, on_step=True, on_epoch=True)
            self.log("d/real_mean", d_real.mean().detach(), on_step=True, on_epoch=True)
            self.log("d/fake_mean", d_fake.mean().detach(), on_step=True, on_epoch=True)
        except Exception:
            pass

        # G update (adv + aux)
        self.toggle_optimizer(opt_g)
        fake_adv = self.sample(b, cond=cond_real)

        # Diagnostics: saturation of tanh output
        try:
            sat98 = (fake_adv.abs() > 0.98).float().mean()
            self.log("g/tanh_sat98", sat98.detach(), on_step=True, on_epoch=True)
        except Exception:
            pass

        # --- genre KL divergence (real vs fake marginals) ---
        genre_kl = torch.tensor(0.0, device=self.device)
        # component placeholders for logging
        loss_g_comp = torch.tensor(0.0, device=self.device)
        loss_o_comp = torch.tensor(0.0, device=self.device)
        loss_a_comp = torch.tensor(0.0, device=self.device)
        loss_gn_comp = torch.tensor(0.0, device=self.device)
        if self.lambda_genre_kl > 0.0 and self._genre_dim and self._genre_dim > 0:
            real_g01 = self._slice_genre01_from_pm(real)
            fake_g01 = self._slice_genre01_from_pm(fake_adv)
            if real_g01 is not None and fake_g01 is not None:
                p = real_g01.mean(dim=0).clamp(1e-6, 1-1e-6)
                q = fake_g01.mean(dim=0).clamp(1e-6, 1-1e-6)
                kl_pq = (p * (p / q).log()).sum()
                kl_qp = (q * (q / p).log()).sum()
                genre_kl = 0.5 * (kl_pq + kl_qp)
        g_adv = -self._critic(fake_adv, cond_real).mean()

        cond_dim = int(getattr(self.hparams, "cond_dim", self.cond_dim))
        use_balanced = bool(getattr(self.hparams, "rebalance_cond_sampling", False)) and cond_dim > 0 and getattr(self, "attr_enabled", False)
        cond_attr = cond_real
        fake_attr = None
        if use_balanced:
            cond_attr_base = self._make_balanced_cond(b, device=self.device)
            # append genre conditioning to match _cond_dim_eff when enabled (detached inside helper)
            cond_attr = self._cond_with_genre(cond_attr_base, real)
            # safety: ensure no grad flows through cond targets across phases
            cond_attr = cond_attr.detach()
            fake_attr = self.sample(b, cond=cond_attr)

        attr_loss = torch.tensor(0.0, device=self.device)
        if getattr(self, "attr_enabled", False) and cond_attr is not None:
            cond_attr = cond_attr.detach()
            x_for_attr = fake_attr if fake_attr is not None else fake_adv
            self._set_attr_grad(False)
            g_logit, occ_logits, age_logits = self._attr_forward(x_for_attr)

            lam_g = float(getattr(self.hparams, "lambda_aux_gender", 0.0))
            lam_o = float(getattr(self.hparams, "lambda_aux_occ", 0.0))
            lam_a = float(getattr(self.hparams, "lambda_aux_age", 0.0))

            if lam_g > 0.0 and cond_attr.size(1) >= 1:
                tgt_g = cond_attr[:, 0].float()
                loss_g = self.bce(g_logit, tgt_g)
                loss_g_comp = loss_g
                attr_loss = attr_loss + lam_g * loss_g
                self.log("aux/g_ce", loss_g.detach(), on_step=True, on_epoch=True)
            if lam_o > 0.0 and cond_attr.size(1) >= 2:
                occ_classes = int(getattr(self.hparams, "condkl_num_occupations", 21))
                tgt_o = torch.round(cond_attr[:, 1] * float(max(1, occ_classes - 1))).long().clamp_(0, max(0, occ_classes - 1))
                if getattr(self, "occ_class_weights_buf", None) is not None:
                    ce_occ = nn.CrossEntropyLoss(weight=self.occ_class_weights_buf.to(self.device))
                else:
                    ce_occ = nn.CrossEntropyLoss()
                loss_o = ce_occ(occ_logits, tgt_o)
                loss_o_comp = loss_o
                attr_loss = attr_loss + lam_o * loss_o
                self.log("aux/occ_ce", loss_o.detach(), on_step=True, on_epoch=True)
            if lam_a > 0.0 and cond_attr.size(1) >= 3:
                age_bins = int(getattr(self.hparams, "condkl_age_bins", 7))
                tgt_a = torch.round(cond_attr[:, 2] * float(max(1, age_bins - 1))).long().clamp_(0, max(0, age_bins - 1))
                ce_age = nn.CrossEntropyLoss()
                loss_a = ce_age(age_logits, tgt_a)
                loss_a_comp = loss_a
                attr_loss = attr_loss + lam_a * loss_a
                self.log("aux/age_ce", loss_a.detach(), on_step=True, on_epoch=True)
            # Genre (multi-label BCE with logits)
            if float(getattr(self.hparams, "lambda_aux_genre", 0.0)) > 0.0 and self._genre_dim and self._genre_dim > 0:
                s = int(self._genre_start); e = s + int(self._genre_dim)
                tgt_genre = ((real[:, s:e] + 1.0) * 0.5).clamp(0, 1)
                # Lazily create genre head
                if not hasattr(self, "C_genre"):
                    hid = int(getattr(self.hparams, "attr_hidden", 256))
                    self.C_genre = nn.Linear(hid, self._genre_dim).to(self.device)
                h_local = self.C_body(x_for_attr)
                genre_logits = self.C_genre(h_local)
                bce = nn.BCEWithLogitsLoss()
                loss_gn = bce(genre_logits, tgt_genre)
                loss_gn_comp = loss_gn
                attr_loss = attr_loss + float(getattr(self.hparams, "lambda_aux_genre", 0.0)) * loss_gn
                self.log("aux/genre_bce", loss_gn.detach(), on_step=True, on_epoch=True)
            self._set_attr_grad(True)

        g_loss = g_adv + attr_loss + self.lambda_genre_kl * genre_kl
        # Detailed component logs
        try:
            self.log("g/adv", g_adv.detach(), on_step=True, on_epoch=True)
            self.log("g/aux_total", attr_loss.detach(), on_step=True, on_epoch=True)
            self.log("g/aux_gender", loss_g_comp.detach(), on_step=True, on_epoch=True)
            self.log("g/aux_occ", loss_o_comp.detach(), on_step=True, on_epoch=True)
            self.log("g/aux_age", loss_a_comp.detach(), on_step=True, on_epoch=True)
            self.log("g/aux_genre", loss_gn_comp.detach(), on_step=True, on_epoch=True)
            if self.lambda_genre_kl > 0.0:
                self.log("kl/genre_sym", genre_kl.detach(), on_step=True, on_epoch=True)
        except Exception:
            pass

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        self.log("gen_loss", g_loss.detach(), prog_bar=True, on_step=True, on_epoch=True)

        # Aux head update (separate graph on classifier params only)
        if getattr(self, "attr_enabled", False) and (opt_c is not None) and (cond_attr is not None):
            self.toggle_optimizer(opt_c)
            x_for_c = (fake_attr if fake_attr is not None else fake_adv).detach()
            g_logit, occ_logits, age_logits = self._attr_forward(x_for_c)
            loss_c = torch.tensor(0.0, device=self.device)
            cond_tgt = cond_attr.detach()
            if float(getattr(self.hparams, "lambda_aux_gender", 0.0)) > 0.0 and cond_tgt.size(1) >= 1:
                tgt_g = cond_tgt[:, 0].float()
                loss_c = loss_c + self.bce(g_logit, tgt_g)
                try:
                    self.log("c/gender_ce", (self.bce(g_logit, cond_tgt[:,0].float()) if cond_tgt.size(1) >= 1 else torch.tensor(0.0, device=self.device)).detach(), on_step=True, on_epoch=True)
                except Exception:
                    pass
            if float(getattr(self.hparams, "lambda_aux_occ", 0.0)) > 0.0 and cond_tgt.size(1) >= 2:
                occ_classes = int(getattr(self.hparams, "condkl_num_occupations", 21))
                tgt_o = torch.round(cond_tgt[:, 1] * float(max(1, occ_classes - 1))).long().clamp_(0, max(0, occ_classes - 1))
                if getattr(self, "occ_class_weights_buf", None) is not None:
                    ce_occ = nn.CrossEntropyLoss(weight=self.occ_class_weights_buf.to(self.device))
                else:
                    ce_occ = nn.CrossEntropyLoss()
                loss_c = loss_c + ce_occ(occ_logits, tgt_o)
            if float(getattr(self.hparams, "lambda_aux_age", 0.0)) > 0.0 and cond_tgt.size(1) >= 3:
                age_bins = int(getattr(self.hparams, "condkl_age_bins", 7))
                tgt_a = torch.round(cond_tgt[:, 2] * float(max(1, age_bins - 1))).long().clamp_(0, max(0, age_bins - 1))
                ce_age = nn.CrossEntropyLoss()
                loss_c = loss_c + ce_age(age_logits, tgt_a)
            if float(getattr(self.hparams, "lambda_aux_genre", 0.0)) > 0.0 and self._genre_dim and self._genre_dim > 0:
                s = int(self._genre_start); e = s + int(self._genre_dim)
                if not hasattr(self, "C_genre"):
                    hid = int(getattr(self.hparams, "attr_hidden", 256))
                    self.C_genre = nn.Linear(hid, self._genre_dim).to(self.device)
                h_local = self.C_body(x_for_c)
                genre_logits = self.C_genre(h_local)
                tgt_genre = ((real[:, s:e] + 1.0) * 0.5).clamp(0, 1)
                bce = nn.BCEWithLogitsLoss()
                loss_c = loss_c + bce(genre_logits, tgt_genre)

            opt_c.zero_grad(set_to_none=True)
            self.manual_backward(loss_c)
            opt_c.step()
            self.untoggle_optimizer(opt_c)
            self.log("aux/c_loss", loss_c.detach(), on_step=True, on_epoch=True)

    def _convert01_to_original_units(self, x01: torch.Tensor, gender_idx: int, occ_idx, age_idx: int, rating_idx: int, occ_classes: int) -> torch.Tensor:
        """Minimal conversion for columns used in the 1x4 plot when DM helpers are missing.
        Expects x01 in [0,1]. Returns a tensor in 'original' units only for the plotted columns:
          - age: years (using 0..100 scaling)
          - rating: 1..5 scale (MovieLens, not 0..5)
          - gender: {0,1} rounded
          - occupation: integer 0..(occ_classes-1) or one-hot
        Other columns are left in [0,1]; they are not used by the plot.
        """
        x = x01.clone()
        # age years
        x[:, age_idx] = (x[:, age_idx].clamp(0, 1) * 100.0)
        # rating 1..5 (MovieLens scale)
        x[:, rating_idx] = (1.0 + x[:, rating_idx].clamp(0, 1) * 4.0)
        # gender {0,1}
        x[:, gender_idx] = x[:, gender_idx].round().clamp(0, 1)
        # occupation 0..K-1 (supports scalar or one-hot block via occ_idx int or (start,width))
        if occ_classes is None or occ_classes <= 1:
            occ_classes = 21
        if isinstance(occ_idx, tuple):
            # Keep one-hot format in original units. Harden to a 1-of-K vector by argmax.
            s, w = occ_idx
            block = x[:, s:s+w]
            # convert soft [0,1] to hard one-hot
            max_idx = torch.argmax(block, dim=1)
            block.zero_()
            block[torch.arange(x.size(0), device=x.device), max_idx] = 1.0
        else:
            # Scalar occupation id in [0,1] -> integer 0..K-1
            x[:, occ_idx] = (x[:, occ_idx].clamp(0, 1) * float(occ_classes - 1)).round().clamp(0, occ_classes - 1)
        return x

    def _save_plot_safe(self, real_orig: torch.Tensor, fake_orig: torch.Tensor, gender_idx: int, occ_idx: int, age_idx: int, rating_idx: int, occ_classes: int, pred_file: str, genre_start: Optional[int] = None, genre_dim: int = 0) -> None:
        out_dir = "wgan_samples"
        try:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{pred_file}.png")
            print(f"[WGAN][plot] saving 1x4 real-vs-fake to {out_path}")
            save_1x4_real_fake(real_orig, fake_orig, gender_idx, occ_idx, age_idx, rating_idx, occ_classes, out_path, genre_start, genre_dim)
            print(f"[WGAN][plot] saved -> {out_path} (cwd={os.getcwd()})")
        except Exception as e:
            print(f"[WGAN][plot][ERROR] {e}")

    # ---- Helper: build cond from real samples in original units (for plotting/synthesis) ----
    def _cond_from_real_original_units(self, real_orig: torch.Tensor, gender_idx: int, occ_idx, age_idx: int,
                                       occ_classes: int, age_bins: int = 7) -> torch.Tensor:
        """Build conditional vector from *real* samples in original units for plotting/synthesis.
        - gender: keep {0,1}
        - occupation: scale 0..K-1 to [0,1] via /(K-1) or from one-hot
        - age: map years to nearest ML-1M bucket index {0..age_bins-1} then / (age_bins-1)
        Returns a tensor of shape [B, 1] (cond_dim==1) or [B, 3] (cond_dim==3).
        """
        device = real_orig.device
        g = real_orig[:, gender_idx:gender_idx+1].clamp(0, 1).float()
        if isinstance(occ_idx, tuple):
            s, w = occ_idx
            occ_id = torch.argmax(real_orig[:, s:s+w], dim=1, keepdim=True).float()
            o01 = (occ_id / float(max(1, occ_classes - 1))).clamp(0, 1)
        else:
            o01 = (real_orig[:, occ_idx:occ_idx+1].clamp(0, max(0, occ_classes - 1)).float() /
                   float(max(1, occ_classes - 1)))
        years = real_orig[:, age_idx:age_idx+1].clamp(0, 120).float()
        # ML-1M default codes
        codes = years.new_tensor([1., 18., 25., 35., 45., 50., 56.]).view(1, -1)
        idx = torch.argmin((years - codes).abs(), dim=1, keepdim=True).float()
        a01 = (idx / float(max(1, age_bins-1))).clamp(0, 1)
        return torch.cat([g, o01, a01], dim=1)

    # ---- Auto-plot at end of training (no external callback needed) ----
    def on_train_end(self) -> None:
        """Create the 1×4 real-vs-fake figure at the end of **training** (on_train_end).
        Supports conditional sampling:
          - cond_dim == 0: unconditioned
          - cond_dim == 1: gender only  (col=gender_idx)
          - cond_dim == 3: [gender, occ/20, age_bucket/6]
        Enforces canonical columns in normalized [0,1] so downstream uses match the sampled conditions.
        Saves to `wgan_samples/{pred_file}.png` with panels [Rating, Occupation, Gender, Age] and a 5th Genre panel when available.
        """
        dm = getattr(self.trainer, "datamodule", None)
        dm_ok = (dm is not None) and hasattr(dm, "sample_real_original_units") and hasattr(dm, "_to_original_units")

        hp = getattr(self, "hparams", {})
        pred_file   = str(getattr(hp, "pred_file", getattr(self, "pred_file", "preds")))
        self._maybe_infer_schema()
        gender_idx  = int(self._gender_idx)
        age_idx     = int(self._age_idx)
        rating_idx  = int(self._rating_idx)
        occ_classes = int(self._occ_classes)
        # pass occupation as int index or (start, width) tuple for one-hot
        occ_idx = (self._occ_start, self._occ_width) if self._occ_onehot else int(self._occ_start)
        num_samples = int(getattr(hp, "num_attack_samples", 100000))
        cond_dim    = int(getattr(hp, "cond_dim", getattr(self, "cond_dim", 0)))

        print(f"[WGAN][plot] idxs: gender={gender_idx} occ={occ_idx} age={age_idx} rating={rating_idx}")

        print(f"[WGAN][plot] on_train_end: pred_file={pred_file} | cond_dim={cond_dim} | num_samples={num_samples}")

        try:
            if dm_ok:
                real_orig = dm.sample_real_original_units(num_samples).cpu()
                print(f"[WGAN][plot] real_orig shape={tuple(real_orig.shape)} (via DM)")
            else:
                # Fallback: use cached real batches collected during training
                if len(self._real_cache) == 0:
                    print("[WGAN][plot] no datamodule and empty real cache; skipping plot")
                    return
                real_pm = torch.cat(self._real_cache, dim=0)
                if real_pm.size(0) > num_samples:
                    real_pm = real_pm[:num_samples]
                real01 = (real_pm + 1.0) * 0.5
                real_orig = self._convert01_to_original_units(real01, gender_idx, occ_idx, age_idx, rating_idx, occ_classes).cpu()
                print(f"[WGAN][plot] real_orig shape={tuple(real_orig.shape)} (from cache)")
        except Exception as e:
            print(f"[WGAN][plot][ERROR] sampling real: {e}")
            return

        # Ensure we don't request more fakes/conds than real rows available
        n_real = int(real_orig.size(0))
        n = min(int(num_samples), n_real)
        if n < int(num_samples):
            print(f"[WGAN][plot] only {n_real} real rows available; plotting with n={n}")
        # keep real set to the same n
        real_orig = real_orig[:n]
        device = self.device
        with torch.no_grad():
            cond = None
            try:
                # Build base demographic condition (size = cond_dim)
                if cond_dim > 0:
                    if cond_dim == 1:
                        cond = real_orig[:, gender_idx:gender_idx+1].to(device).clamp(0, 1).float()
                    elif cond_dim == 3:
                        age_bins = int(getattr(hp, "condkl_age_bins", 7))
                        cond = self._cond_from_real_original_units(
                            real_orig.to(device), gender_idx, occ_idx, age_idx, occ_classes, age_bins
                        )
                    else:
                        cond = torch.zeros(n, cond_dim, device=device)

                # Optionally append genre conditioning to match _cond_dim_eff used to build G/D
                cond_full = cond
                if self.cond_include_genre and (self._genre_dim and self._genre_dim > 0):
                    s = int(self._genre_start); e = s + int(self._genre_dim)
                    # real_orig is already in original units; genre should be multi-hot in [0,1]
                    genre01 = real_orig[:, s:e].to(device).float().clamp(0, 1)
                    if self.cond_genre_proj is not None:
                        genre_cond = torch.sigmoid(self.cond_genre_proj(genre01))
                    else:
                        genre_cond = genre01
                    cond_full = genre_cond if cond_full is None else torch.cat([cond_full, genre_cond], dim=1)

                fake_pm = self.sample(n, cond=cond_full, device=device)  # [-1,1]
                fake_01 = (fake_pm + 1.0) * 0.5
                if dm_ok:
                    fake_orig = dm._to_original_units(fake_01).cpu()[:n]
                    print(f"[WGAN][plot] fake_orig shape={tuple(fake_orig.shape)} (via DM)")
                else:
                    fake_orig = self._convert01_to_original_units(fake_01, gender_idx, occ_idx, age_idx, rating_idx, occ_classes).cpu()[:n]
                    print(f"[WGAN][plot] fake_orig shape={tuple(fake_orig.shape)} (manual conv)")
            except Exception as e:
                print(f"[WGAN][plot][ERROR] sampling fake: {e}")
                return

        self._save_plot_safe(real_orig, fake_orig, gender_idx, occ_idx, age_idx, rating_idx, occ_classes, pred_file, self._genre_start, int(self._genre_dim))

    # Fallback: call the same plot on fit end in case users run validate/test flows
    def on_fit_end(self):
        try:
            self.on_train_end()
        except Exception as e:
            print(f"[WGAN][plot][on_fit_end][ERROR] {e}")


def save_ratings_hist(y_real: torch.Tensor, y_fake: torch.Tensor, out_path: str):
    """Save a simple histogram overlay of real vs fake ratings (both expected in [0,1])."""
    r = _np(y_real.view(-1))
    f = _np(y_fake.view(-1))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(r, bins=30, alpha=0.6, density=True, label="real")
    ax.hist(f, bins=30, alpha=0.6, density=True, label="fake")
    ax.set_title("Rating (real vs fake)")
    ax.set_xlabel("rating/5 (0..1)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# ----- Plotting (1×4) -----
def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

@torch.no_grad()
def save_1x4_real_fake(real_orig: torch.Tensor, fake_orig: torch.Tensor, gender_idx: int, occ_idx, age_idx: int, rating_idx: int, occ_classes: int, out_path: str, genre_start: Optional[int] = None, genre_dim: int = 0):
    r = _np(real_orig)
    f = _np(fake_orig)

    # Snap ages (years) to canonical ML-1M codes before plotting
    def _snap_years_to_codes(vec: np.ndarray) -> np.ndarray:
        codes = np.array([1., 18., 25., 35., 45., 50., 56.], dtype=np.float64)
        # nearest code per element
        idx = np.abs(vec[:, None] - codes[None, :]).argmin(axis=1)
        return codes[idx]

    # Apply snapping in-place for both real and fake arrays
    try:
        r[:, age_idx] = _snap_years_to_codes(r[:, age_idx])
        f[:, age_idx] = _snap_years_to_codes(f[:, age_idx])
    except Exception:
        pass

    import matplotlib.pyplot as plt
    panels = 5 if (genre_start is not None and int(genre_dim) > 0) else 4
    fig, axes = plt.subplots(1, panels, figsize=(4*panels, 3.5))

    # Helper: symmetric KL divergence on (non-negative) counts/vectors
    def _sym_kl_from_counts(c1: np.ndarray, c2: np.ndarray, eps: float = 1e-8) -> float:
        c1 = np.asarray(c1, dtype=np.float64)
        c2 = np.asarray(c2, dtype=np.float64)
        c1 = np.clip(c1, 0, None) + eps
        c2 = np.clip(c2, 0, None) + eps
        p = c1 / c1.sum()
        q = c2 / c2.sum()
        kl_pq = (p * (np.log(p) - np.log(q))).sum()
        kl_qp = (q * (np.log(q) - np.log(p))).sum()
        return 0.5 * float(kl_pq + kl_qp)

    # (1) Rating (1..5)
    ax = axes[0]
    rating_bins = np.linspace(1.0, 5.0, 21)
    r_hist, _ = np.histogram(r[:, rating_idx], bins=rating_bins)
    f_hist, _ = np.histogram(f[:, rating_idx], bins=rating_bins)
    skl_rating = _sym_kl_from_counts(r_hist, f_hist)
    ax.hist(r[:, rating_idx], bins=rating_bins, alpha=0.6, density=True, label="real")
    ax.hist(f[:, rating_idx], bins=rating_bins, alpha=0.6, density=True, label="fake")
    ax.set_title(f"Rating (1..5) | SKL={skl_rating:.3f}")
    ax.set_xlabel("rating")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # (2) Occupation (0..K-1)
    ax = axes[1]
    if isinstance(occ_idx, tuple):
        s, w = occ_idx
        r_occ = np.argmax(r[:, s:s+w], axis=1)
        f_occ = np.argmax(f[:, s:s+w], axis=1)
    else:
        r_occ = np.clip(np.rint(r[:, occ_idx]).astype(int), 0, occ_classes - 1)
        f_occ = np.clip(np.rint(f[:, occ_idx]).astype(int), 0, occ_classes - 1)
    bins = np.arange(occ_classes + 1) - 0.5
    r_cnt, _ = np.histogram(r_occ, bins=bins)
    f_cnt, _ = np.histogram(f_occ, bins=bins)
    skl_occ = _sym_kl_from_counts(r_cnt, f_cnt)
    ax.hist(r_occ, bins=bins, alpha=0.6, label="real")
    ax.hist(f_occ, bins=bins, alpha=0.6, label="fake")
    ax.set_title(f"Occupation | SKL={skl_occ:.3f}")
    ax.set_xlabel("occ id")
    try:
        ax.set_xticks(range(occ_classes))
    except Exception:
        pass
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # (3) Gender (0/1)
    ax = axes[2]
    r_g = np.clip(np.rint(r[:, gender_idx]).astype(int), 0, 1)
    f_g = np.clip(np.rint(f[:, gender_idx]).astype(int), 0, 1)
    bins_g = np.array([-0.5, 0.5, 1.5])
    r_cnt_g, _ = np.histogram(r_g, bins=bins_g)
    f_cnt_g, _ = np.histogram(f_g, bins=bins_g)
    skl_gender = _sym_kl_from_counts(r_cnt_g, f_cnt_g)
    ax.hist(r_g, bins=bins_g, alpha=0.6, label="real")
    ax.hist(f_g, bins=bins_g, alpha=0.6, label="fake")
    ax.set_xticks([0, 1])
    ax.set_title(f"Gender | SKL={skl_gender:.3f}")
    ax.set_xlabel("gender")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # (4) Age (bucketed years)
    ax = axes[3]
    age_bins = np.linspace(0.0, 100.0, 21)
    r_hist_age, _ = np.histogram(r[:, age_idx], bins=age_bins)
    f_hist_age, _ = np.histogram(f[:, age_idx], bins=age_bins)
    skl_age = _sym_kl_from_counts(r_hist_age, f_hist_age)
    ax.hist(r[:, age_idx], bins=age_bins, alpha=0.6, density=True, label="real")
    ax.hist(f[:, age_idx], bins=age_bins, alpha=0.6, density=True, label="fake")
    ax.set_title(f"Age (bucketed years) | SKL={skl_age:.3f}")
    ax.set_xlabel("age")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    if panels == 5:
        ax = axes[4]
        s = int(genre_start); e = s + int(genre_dim)
        r_g = r[:, s:e].mean(axis=0)
        f_g = f[:, s:e].mean(axis=0)
        # Normalize to distributions for SKL
        skl_genre = _sym_kl_from_counts(r_g, f_g)
        idx = np.arange(int(genre_dim))
        width = 0.45
        ax.bar(idx - width/2, r_g, width, label="real")
        ax.bar(idx + width/2, f_g, width, label="fake")
        ax.set_title(f"Genre marginals | SKL={skl_genre:.3f}")
        ax.set_xlabel("genre idx")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
