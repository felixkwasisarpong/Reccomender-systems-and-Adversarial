# diagnostics.py
from __future__ import annotations
import numpy as np
import torch

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _sym_kl_from_counts(c1: np.ndarray, c2: np.ndarray, eps: float = 1e-8) -> float:
    c1 = np.clip(np.asarray(c1, dtype=np.float64), 0, None) + eps
    c2 = np.clip(np.asarray(c2, dtype=np.float64), 0, None) + eps
    p = c1 / c1.sum()
    q = c2 / c2.sum()
    kl_pq = np.sum(p * (np.log(p) - np.log(q)))
    kl_qp = np.sum(q * (np.log(q) - np.log(p)))
    return 0.5 * float(kl_pq + kl_qp)

def _hist_skl_1d(r, f, bins):
    r_cnt, _ = np.histogram(r, bins=bins)
    f_cnt, _ = np.histogram(f, bins=bins)
    return _sym_kl_from_counts(r_cnt, f_cnt)

def _bin01(x):  # robust rounding for {0,1}
    return np.clip(np.rint(x).astype(int), 0, 1)

def _argmax_block(a, s, w):
    return np.argmax(a[:, s:s+w], axis=1)

def compute_conditional_skl_report(
    real_orig,                 # tensor/ndarray [N, D] in ORIGINAL UNITS
    synth_orig,                # tensor/ndarray [M, D] in ORIGINAL UNITS
    *,
    age_idx: int,
    gender_idx: int,
    rating_idx: int,
    occ_idx,                   # int for scalar or (start,width) for one-hot
    occ_classes: int = 21,
    genre_start: int | None = None,
    genre_dim: int = 0,
    # binning
    rating_bins: int = 20,     # buckets for 1..5
    age_bins: int = 20,        # buckets for 0..100
    # minimum samples per class to evaluate
    min_samples_per_class: int = 50,
    # optional figure path (heatmap)
    save_heatmap_path: str | None = None,
) -> dict:
    """
    Returns a dict with:
      - 'classes': list[int]
      - 'rating': [C] SKL per class
      - 'age':    [C] SKL per class
      - 'gender': [C] SKL per class
      - 'genre':  [C] SKL per class (marginals per genre, KL over the 19-D Bernoulli means)
      - 'counts': {'real': [C], 'synth': [C]}
      - 'overall': {'rating': float, 'age': float, 'gender': float, 'genre': float}  # averaged over evaluated classes
    """
    r = _to_np(real_orig)
    f = _to_np(synth_orig)

    # ---- derive occupation labels ----
    if isinstance(occ_idx, tuple):
        s, w = occ_idx
        r_occ = _argmax_block(r, s, w)
        f_occ = _argmax_block(f, s, w)
    else:
        r_occ = np.clip(np.rint(r[:, occ_idx]).astype(int), 0, occ_classes - 1)
        f_occ = np.clip(np.rint(f[:, occ_idx]).astype(int), 0, occ_classes - 1)

    # ---- bins ----
    rating_edges = np.linspace(1.0, 5.0, rating_bins + 1)
    age_edges    = np.linspace(0.0, 100.0, age_bins + 1)

    # ---- per-class SKLs ----
    cls = list(range(occ_classes))
    skl_rating = np.full(occ_classes, np.nan, dtype=np.float64)
    skl_age    = np.full(occ_classes, np.nan, dtype=np.float64)
    skl_gender = np.full(occ_classes, np.nan, dtype=np.float64)
    skl_genre  = np.full(occ_classes, np.nan, dtype=np.float64)
    cnt_real   = np.zeros(occ_classes, dtype=np.int64)
    cnt_synth  = np.zeros(occ_classes, dtype=np.int64)

    have_genre = (genre_start is not None) and (int(genre_dim) > 0)

    for k in range(occ_classes):
        r_mask = (r_occ == k)
        f_mask = (f_occ == k)
        nr = int(r_mask.sum()); nf = int(f_mask.sum())
        cnt_real[k]  = nr
        cnt_synth[k] = nf
        if nr < min_samples_per_class or nf < min_samples_per_class:
            continue  # skip under-sampled class

        # rating
        skl_rating[k] = _hist_skl_1d(r[r_mask, rating_idx], f[f_mask, rating_idx], rating_edges)
        # age
        skl_age[k]    = _hist_skl_1d(r[r_mask, age_idx],    f[f_mask, age_idx],    age_edges)
        # gender (2 bins)
        r_g = _bin01(r[r_mask, gender_idx]); f_g = _bin01(f[f_mask, gender_idx])
        r_cnt, _ = np.histogram(r_g, bins=np.array([-0.5, 0.5, 1.5]))
        f_cnt, _ = np.histogram(f_g, bins=np.array([-0.5, 0.5, 1.5]))
        skl_gender[k] = _sym_kl_from_counts(r_cnt, f_cnt)

        # genre marginals → KL over mean vectors
        if have_genre:
            s = int(genre_start); e = s + int(genre_dim)
            r_p = np.clip(r[r_mask, s:e].mean(axis=0), 1e-8, 1.0 - 1e-8)
            f_p = np.clip(f[f_mask, s:e].mean(axis=0), 1e-8, 1.0 - 1e-8)
            # treat these as unnormalized “counts” and compute SKL
            skl_genre[k] = _sym_kl_from_counts(r_p, f_p)

    # overall (mean over finite entries)
    def _mean_finite(x):
        x = np.asarray(x, dtype=np.float64)
        m = np.isfinite(x)
        return float(np.nan if not m.any() else np.nanmean(x[m]))

    overall = {
        "rating": _mean_finite(skl_rating),
        "age":    _mean_finite(skl_age),
        "gender": _mean_finite(skl_gender),
        "genre":  _mean_finite(skl_genre) if have_genre else np.nan,
    }

    report = {
        "classes": cls,
        "rating": skl_rating,
        "age": skl_age,
        "gender": skl_gender,
        "genre": skl_genre if have_genre else None,
        "counts": {"real": cnt_real, "synth": cnt_synth},
        "overall": overall,
    }

    # optional heatmap
    if save_heatmap_path is not None:
        try:
            import matplotlib.pyplot as plt
            metrics = ["rating", "age", "gender"] + (["genre"] if have_genre else [])
            data = np.vstack([report[m] for m in metrics])
            fig, ax = plt.subplots(figsize=(max(8, occ_classes * 0.4), 2 + 0.5 * len(metrics)))
            im = ax.imshow(data, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
            ax.set_xticks(range(occ_classes)); ax.set_xticklabels([str(k) for k in range(occ_classes)], rotation=0)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="SKL (real || synth)")
            ax.set_title("Per-occupation SKL (real vs synthetic)")
            fig.tight_layout()
            fig.savefig(save_heatmap_path, dpi=160)
            plt.close(fig)
        except Exception as e:
            print(f"[SKL][warn] heatmap failed: {e}")

    return report