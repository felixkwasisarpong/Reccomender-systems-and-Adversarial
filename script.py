import h5py, torch
from src.Utils.diagnostics import compute_conditional_skl_report
import numpy as np 
# Load real (black-box) and synthetic in ORIGINAL UNITS
def load_h5_rows(path, dataset="predictions"):
    with h5py.File(path, "r") as f:
        if dataset == "predictions":
            d = f["predictions"][:]  # structured array
            # build [N,45] original units: you already write exactly these fields in your generator
            user = d["user_id"]
            item = d["movie_id"]
            age  = d["age"]
            gender = d["gender"]
            occ = d["occupation"]
            genre = d["genre"]  # [N,19]
            rating = d["pred_rating"]
            # pack back to the canonical 45 columns:
            # [uid, iid, age, gender, 21xocc_onehot, 19xgenre, rating]
            import numpy as np
            N = len(user)
            occ_onehot = np.zeros((N,21), dtype=np.float32)
            occ_onehot[np.arange(N), occ.astype(int).clip(0,20)] = 1.0
            X = np.concatenate([
                user.reshape(-1,1), item.reshape(-1,1),
                age.reshape(-1,1), gender.reshape(-1,1),
                occ_onehot.astype(np.float32),
                genre.astype(np.float32),
                rating.reshape(-1,1)
            ], axis=1)
            return torch.from_numpy(X)
        elif dataset == "synthetic":
            X = f["synthetic"][:]  # your normalized 25-D if you saved that; not needed here
            return torch.from_numpy(X)
        else:
            raise ValueError(f"unknown dataset {dataset}")

real_path  = "predicted_data/opacus_strong.hdf5"              # black-box preds
synth_path = "synthetic_outputs/opacus_strong.hdf5"           # WGAN output

real_orig  = load_h5_rows(real_path,  dataset="predictions")
synth_orig = load_h5_rows(synth_path, dataset="predictions")

report = compute_conditional_skl_report(
    real_orig, synth_orig,
    age_idx=2, gender_idx=3, rating_idx=44,
    occ_idx=(4,21), occ_classes=21,
    genre_start=25, genre_dim=19,
    save_heatmap_path="wgan_samples/cond_skl_heatmap.png",
)

print("[SKL] overall:", report["overall"])
bad = []
for k in range(21):
    rr = report["rating"][k]
    aa = report["age"][k]
    gg = report["gender"][k]
    gn = report["genre"][k] if report["genre"] is not None else np.nan
    if np.isfinite(rr) or np.isfinite(aa) or np.isfinite(gg) or np.isfinite(gn):
        bad.append((k, rr, aa, gg, gn))
bad_sorted = sorted(bad, key=lambda t: np.nanmax([t[1], t[2], t[3], t[4]]), reverse=True)
print("Top drifted occupation classes (by max SKL across metrics):")
for k, rr, aa, gg, gn in bad_sorted[:10]:
    print(f"  occ={k:2d} | rating={rr:.3f} age={aa:.3f} gender={gg:.3f} genre={gn:.3f} | "
          f"n_real={report['counts']['real'][k]} n_synth={report['counts']['synth'][k]}")