import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
from models.BaseModel import BaseModel  # noqa
from models.DPModel import DPModel  # noqa
from models.CustomDP_SGD import CustomDP_SGD
from dataloader.NetflixDataModule import NetflixDataModule
from models.MembershipInferenceAttack import MembershipInferenceAttack  # replace with your actual path
from models.DPMembershipInferenceAttack import DPMembershipInferenceAttack  # replace with your actual path
from models.CustomMembershipInferenceAttack import CustomMembershipInferenceAttack  # replace with your actual path
from models.AttributeClassifier import AttributeClassifier,run_yaml_attack_sweep
from models.DPFM_GANTrainer import DPFM_GANTrainer  # replace with your actual path
from models.WGAN import WGAN
from models.DPFMMembershipInferenceAttack import DPFMMembershipInferenceAttack  # noqa
import wandb
import torch
import os
import numpy as np
import torch.nn as nn 


import h5py







_ML1M_AGE_BUCKETS = torch.tensor([1., 18., 25., 35., 45., 50., 56.])

def _snap_age_ml1m(age_years_t: torch.Tensor) -> torch.Tensor:
    """Snap continuous ages (years) to nearest ML-1M age bucket values.
    age_years_t: float tensor (years). Returns float tensor with values in {1,18,25,35,45,50,56}.
    """
    # shape [B,1] vs [7] — broadcast distance then take argmin
    a = age_years_t.view(-1, 1)
    b = _ML1M_AGE_BUCKETS.to(a.device).view(1, -1)
    idx = torch.argmin((a - b).abs(), dim=1)
    return b[0, idx]



torch.set_float32_matmul_precision('high')
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path", "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool, default=True)
        parser.add_argument("--do_validate", type=bool, default=False)
        parser.add_argument("--do_test", type=bool, default=False)
        parser.add_argument("--do_predict", type=bool, default=False)
        parser.add_argument("--do_analyze", default=False, type=bool) 
        parser.add_argument("--do_attack", default=False, type=bool) 
        parser.add_argument("--do_classifier_attack", default=False, type=bool)
        parser.add_argument("--do_create_synthetic_data", default=False, type=bool)
        parser.add_argument("--ckpt_path", type=str, default=None)
        parser.add_argument("--attack_include_identifiers", type=bool, default=False)
        parser.add_argument("--attack_include_rating", type=bool, default=True)
        parser.add_argument("--attack_age_as_buckets", type=bool, default=False)
        # --- attack feature/label formatting knobs (TOP-LEVEL) ---
        parser.add_argument("--attack_identifiers_one_based", type=bool, default=True)
        parser.add_argument("--attack_remap_occ_1_based", type=bool, default=False)
        parser.add_argument("--attack_age_label_mode", type=str,
                            choices=["code", "years", "bucket"], default="code")
        # --- Classifier-attack data source controls ---
        parser.add_argument("--attack_train_source", type=str, choices=["synthetic_outputs", "predicted_data"], default=None)
        parser.add_argument("--attack_test_source", type=str, choices=["synthetic_outputs", "predicted_data"], default=None)
        parser.add_argument("--synthetic_hdf5_path", type=str, default=None)
        parser.add_argument("--blackbox_hdf5_path", type=str, default=None)
        parser.add_argument("--blackbox_dir", type=str, default=None)

        parser.add_argument("--qualifying_file", type=str, default="qualifying.txt")
        parser.add_argument("--output_predictions", type=str, default="submission.txt")

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """Configure WandB with proper metric tracking"""
        if not wandb.run:
            wandb.init(
                project="factorization machine",
                config=self.config.model if hasattr(self.config, 'model') else {}
            )
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
        with open(config_file_name, "w") as f:
            f.write(self.parser.dump(self.config, skip_none=False))
        
        wandb.save(config_file_name, policy="now")
        self._define_wandb_metrics()

    def _define_wandb_metrics(self):
        """Setup WandB metric summaries"""
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("val_rmse", summary="min")
        wandb.define_metric("train_rmse", summary="min")
        # DP-specific metrics
        if hasattr(self.model, "privacy_engine"):
             wandb.define_metric("epsilon", step_metric="epoch", summary="max")


def main():



    cli = MyLightningCLI(None, NetflixDataModule,subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
    #cli.wandb_setup()

    ckpt = "best"
    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule,
                        ckpt_path=cli.config.ckpt_path)



    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        # Run normal test first
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    # If we have trained a model, use the best checkpoint for predicting.
    if cli.config.do_predict:
        # --- 1) Load checkpoint & move to device
        ckpt_path = "/Users/Apple/Documents/assignements/Thesis/lightning_logs/custom_weak/dp_fm/wqa9vz2i/checkpoints/epoch=14-step=37230.ckpt"
        cli.model = CustomDP_SGD.load_from_checkpoint(str(ckpt_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cli.model.to(device)
        cli.model.eval()

        # --- 2) Prepare datamodule & get root dataset (for original ID mapping)
        cli.datamodule.setup('predict')
        pred_loader = cli.datamodule.predict_dataloader()

        root_ds = pred_loader.dataset
        # unwrap Subset/DataLoader chains
        while hasattr(root_ds, "dataset"):
            root_ds = root_ds.dataset

        # Build inverse maps: index -> original ID
        # NOTE: user2idx/movie2idx were built from np.unique(...), which is sorted, so:
        orig_users  = np.array(sorted(np.unique(root_ds.user_ids_all)), dtype=np.int64)
        orig_movies = np.array(sorted(np.unique(root_ds.movie_ids_all)), dtype=np.int32)
        genre_dim   = int(root_ds.genre_all.shape[1])  # typically 19

        # --- 3) Predict loop
        user_idx_buf, movie_idx_buf, pred_buf = [], [], []
        attr_buf = {k: [] for k in ("gender", "age", "occupation", "genre")}

        with torch.no_grad():
            for batch in pred_loader:
                # batch carries *reindexed* ids
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = cli.model(batch)  # shape [B], float

                user_idx_buf.append(batch["user_id"].detach().cpu())  # indices 0..U-1
                movie_idx_buf.append(batch["item_id"].detach().cpu()) # indices 0..I-1
                pred_buf.append(preds.detach().cpu())

                for k in attr_buf:
                    if k in batch:
                        attr_buf[k].append(batch[k].detach().cpu())

        # --- 4) Concatenate and map indices -> original IDs
        user_idx   = torch.cat(user_idx_buf).numpy().astype(np.int64)
        movie_idx  = torch.cat(movie_idx_buf).numpy().astype(np.int64)  # temp int64 for indexing
        preds_np   = torch.cat(pred_buf).numpy().astype(np.float32)

        # map back to original ML-1M IDs
        user_ids_saved  = orig_users[user_idx]                      # int64, original 1..6040
        movie_ids_saved = orig_movies[movie_idx].astype(np.int32)   # int32, original 1..3952

        # attributes
        gender_np = torch.cat(attr_buf["gender"]).numpy().astype(np.int8)
        age_np    = torch.cat(attr_buf["age"]).numpy().astype(np.float32)
        occ_np    = torch.cat(attr_buf["occupation"]).numpy().astype(np.int8)
        genre_np  = torch.cat(attr_buf["genre"]).numpy().astype(np.float32)

        # --- 5) Structured dtype
        dt = np.dtype([
            ("user_id",     np.int64),
            ("movie_id",    np.int32),
            ("pred_rating", np.float32),
            ("gender",      np.int8),
            ("age",         np.float32),
            ("occupation",  np.int8),
            ("genre",       np.float32, (genre_dim,))
        ])

        structured = np.zeros(len(user_ids_saved), dtype=dt)
        structured["user_id"]     = user_ids_saved
        structured["movie_id"]    = movie_ids_saved
        structured["pred_rating"] = preds_np
        structured["gender"]      = gender_np
        structured["age"]         = age_np
        structured["occupation"]  = occ_np
        structured["genre"]       = genre_np

        # ---- 6) Report ranges
        print(f"[Range] user_id:   {user_ids_saved.min()} .. {user_ids_saved.max()} (uniq={np.unique(user_ids_saved).size})")
        print(f"[Range] movie_id:  {movie_ids_saved.min()} .. {movie_ids_saved.max()} (uniq={np.unique(movie_ids_saved).size})")
        print(f"[Range] age:       {float(age_np.min())} .. {float(age_np.max())}")
        print(f"[Range] occupation:{int(occ_np.min())} .. {int(occ_np.max())}")
        print(f"[Range] rating:    {float(preds_np.min()):.3f} .. {float(preds_np.max()):.3f}")

        # --- 7) Save (schema your NetflixDataset expects)
        os.makedirs("predicted_data", exist_ok=True)
        out_name = "predicted_data/custom_weak.hdf5"  # fixed typo
        with h5py.File(out_name, "w") as f:
            f.create_dataset("predictions", data=structured, compression="gzip", compression_opts=3)

            # Flat mirrors (optional but handy)
            f.create_dataset("user_ids",     data=user_ids_saved,  compression="gzip", compression_opts=3)
            f.create_dataset("item_ids",     data=movie_ids_saved, compression="gzip", compression_opts=3)
            f.create_dataset("ratings",      data=preds_np,        compression="gzip", compression_opts=3)
            f.create_dataset("gender",       data=gender_np,       compression="gzip", compression_opts=3)
            f.create_dataset("age",          data=age_np,          compression="gzip", compression_opts=3)
            f.create_dataset("occupation",   data=occ_np,          compression="gzip", compression_opts=3)
            f.create_dataset("genre_onehot", data=genre_np,        compression="gzip", compression_opts=3)

            # attrs (useful for synthetic generation)
            f.attrs["global_mean"]      = float(preds_np.mean())
            f.attrs["num_predictions"]  = int(len(preds_np))
            f.attrs["user_id_min"]      = int(user_ids_saved.min())
            f.attrs["user_id_max"]      = int(user_ids_saved.max())
            f.attrs["movie_id_min"]     = int(movie_ids_saved.min())
            f.attrs["movie_id_max"]     = int(movie_ids_saved.max())
            f.attrs["age_min"]          = float(age_np.min())
            f.attrs["age_max"]          = float(age_np.max())
            f.attrs["occupation_min"]   = int(occ_np.min())
            f.attrs["occupation_max"]   = int(occ_np.max())
            f.attrs["pred_rating_min"]  = float(preds_np.min())
            f.attrs["pred_rating_max"]  = float(preds_np.max())

        print(f"[✓] Saved {len(preds_np)} predictions to {out_name}")

    if cli.config.do_analyze:
        ckpt_path = "/Users/Apple/Documents/assignements/Thesis/lightning_logs/custom_weak/dp_fm/wqa9vz2i/checkpoints/epoch=14-step=37230.ckpt"
        
        # Setup datamodule with improved MIA strategy
        cli.datamodule.setup("fit")
        
        # You can choose different MIA strategies:
        # Option 1: Improved MIA using 1M dataset as true non-members (RECOMMENDED)
        cli.datamodule.mia_strategy = "improved"
        
        # Option 2: User-level MIA approach
        # cli.datamodule.mia_strategy = "user_level"
        
        # Option 3: Original approach (for comparison)
        # cli.datamodule.mia_strategy = "original"
        
        # Load the improved attack model
        attack_model = CustomMembershipInferenceAttack.load_from_dp_checkpoint(ckpt_path)
        
        # Get MIA dataloader with improved strategy
        mia_loader = cli.datamodule.mia_dataloaders()[0]
        
        # Run the attack
        cli.trainer.test(attack_model, dataloaders=[mia_loader])
        
        print("MIA evaluation completed. Check 'mia_results/' directory for detailed plots and analysis.")


    if cli.config.do_attack:
        cli.datamodule.setup(stage="attack")
        attack_loader = cli.datamodule.attack_dataloader()
        cli.trainer.fit(cli.model, train_dataloaders=attack_loader)


    if getattr(cli.config, "do_create_synthetic_data", False):

        # ---- Hard-coded checkpoints (as requested) ----
        engines = {
            "opacus": {
                "strong": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_dp_str/dp_fm/enm7akv6/checkpoints/last.ckpt",
            },
        }

        # Canonical ML-1M original ID ranges (inclusive)
        # Users: 1..6040, Movies: 1..3952
        MAX_USER_ID = 6040
        MAX_ITEM_ID = 3952
        OCC_CLASSES = 21
        AGE_CODES = torch.tensor([1., 18., 25., 35., 45., 50., 56.])  # ML-1M

        def _snap_age_ml1m(age_years_f32: torch.Tensor) -> torch.Tensor:
            """Snap to ML-1M canonical age codes after denorm."""
            codes = AGE_CODES.to(age_years_f32.device)
            idx = torch.argmin(torch.abs(age_years_f32.unsqueeze(1) - codes.view(1, -1)), dim=1)
            return codes[idx]

        def _infer_id_ranges_from_predfile(_pred_file: str | None):
            # Optional: parse pred_file to infer tighter ranges. Stub returns fallbacks.
            return (0, MAX_USER_ID, 0, MAX_ITEM_ID)

        def _estimate_priors_from_real(h5_path, age_codes=(1,18,25,35,45,50,56)):
            import numpy as _np, h5py as _h
            with _h.File(h5_path, "r") as f:
                ds = f["predictions"][()]
            # gender (0/1)
            g = ds["gender"].astype(_np.int64)
            p_gender = _np.bincount(g, minlength=2).astype(_np.float64)
            p_gender = p_gender / max(1, p_gender.sum())
            # occupation 0..20
            o = ds["occupation"].astype(_np.int64)
            p_occ = _np.bincount(o, minlength=21).astype(_np.float64)
            p_occ = p_occ / max(1, p_occ.sum())
            # age codes → bucket idx 0..6
            code_to_idx = {int(c): i for i, c in enumerate(age_codes)}
            a_codes = ds["age"].astype(_np.int64)
            a_idx = _np.array([code_to_idx.get(int(x), 0) for x in a_codes])
            p_age = _np.bincount(a_idx, minlength=len(age_codes)).astype(_np.float64)
            p_age = p_age / max(1, p_age.sum())
            return p_gender, p_occ, p_age

        # ---- Output + generation size ----
        output_dir = "synthetic_outputs"
        os.makedirs(output_dir, exist_ok=True)

        n_samples = 100_000
        batch = 512
        steps = (n_samples + batch - 1) // batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for engine_name, ckpts in engines.items():
            for regime, ckpt_path in ckpts.items():
                if not ckpt_path:
                    continue

                print(f"[{engine_name.upper()} - {regime.upper()}] Generating synthetic samples…")
                print(f"  [load] {ckpt_path}")

                # Load model
                gan: WGAN = WGAN.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
                gan.eval().to(device)

                # Pull dims from hparams
                latent_dim = int(getattr(gan.hparams, "noise_dim", 32))
                cond_dim   = int(getattr(gan.hparams, "cond_dim", 0))       # base cond (1 or 3)
                genre_dim  = int(getattr(gan.hparams, "genre_dim", 19))
                include_genre = bool(getattr(gan.hparams, "cond_include_genre", False))
                proj_dim  = int(getattr(gan.hparams, "cond_genre_proj_dim", 0))

                # Schema indices (from training config)
                gender_idx = int(getattr(gan.hparams, "condkl_gender_col", 3))
                occ_start  = int(getattr(gan.hparams, "condkl_occupation_col", 4))
                occ_width  = int(getattr(gan.hparams, "condkl_num_occupations", 21))
                age_idx    = int(getattr(gan.hparams, "condkl_age_col", 2))
                rating_idx = int(getattr(gan.hparams, "rating_idx", 44))
                age_bins   = int(getattr(gan.hparams, "condkl_age_bins", 7))

                # --- Sanity guardrails (match training) ---
                assert int(getattr(gan.hparams, "output_dim", 45)) == 45, \
                    f"Expected output_dim=45, got {getattr(gan.hparams, 'output_dim', None)}"
                assert bool(getattr(gan.hparams, "use_onehot_occ", True)) is True, \
                    "Model not trained with one-hot occupation (use_onehot_occ=True)."

                print("[GEN] layout:",
                      f"age_idx={age_idx}, gender_idx={gender_idx}, occ_start={occ_start},",
                      f"occ_width={occ_width}, genre_dim={genre_dim}, rating_idx={rating_idx},",
                      f"cond_dim={cond_dim}, include_genre={include_genre}, proj_dim={proj_dim}")

                # Where the genre block lives in 45-D: right after the one-hot occ block
                genre_start = occ_start + occ_width
                genre_end   = genre_start + genre_dim

                # ID ranges
                pred_name = getattr(gan.hparams, "pred_file", None)
                # PURE mode: do not touch any blackbox metadata (no HDF5 reads). Use fixed ML-1M ranges.
                umin, umax, imin, imax = 1, MAX_USER_ID, 1, MAX_ITEM_ID

                # --- Optional: sample conditions using real priors if available ---
                p_gender = p_occ = p_age = None
                real_h5 = os.path.join("predicted_data", f"{pred_name}.hdf5") if pred_name else None
                if real_h5 and os.path.isfile(real_h5):
                    print(f"[GEN] Using real priors from {real_h5}")
                    _pg, _po, _pa = _estimate_priors_from_real(real_h5, tuple(AGE_CODES.tolist()))
                    p_gender = torch.tensor(_pg, device=device, dtype=torch.float32)
                    p_occ    = torch.tensor(_po, device=device, dtype=torch.float32)
                    p_age    = torch.tensor(_pa, device=device, dtype=torch.float32)
                else:
                    print("[GEN] Using uniform priors (no real priors file found).")

                # ------------------------------------------------------------------
                # PURE GENERATION for 45-D layout (one-hot occ, genre present)
                #   - G outputs 45-D in [-1,1] → map to [0,1]
                #   - Condition vector must match training: base cond (cond_dim) + optional genre (proj/raw)
                #   - We repack to a 25-D canonical view for saving convenience
                # ------------------------------------------------------------------
                xs_45 = []
                made = 0
                with torch.no_grad():
                    for _ in range(steps):
                        bsz = min(batch, n_samples - made)
                        if bsz <= 0:
                            break

                        z = torch.randn(bsz, latent_dim, device=device)

                        # ---- Build base cond (size = cond_dim: 0/1/3) ----
                        if cond_dim == 0:
                            cond_base = None
                        elif cond_dim == 1:
                            g = (torch.rand(bsz, 1, device=device) < 0.50).float()  # {0,1}
                            cond_base = g
                        elif cond_dim == 3:
                            # gender
                            if p_gender is not None:
                                g_cls = torch.multinomial(p_gender.expand(bsz, -1), num_samples=1)
                                g = g_cls.float()  # 0/1
                            else:
                                g = (torch.rand(bsz, 1, device=device) < 0.50).float()
                            # occupation (0..20)
                            if p_occ is not None:
                                occ_id = torch.multinomial(p_occ.expand(bsz, -1), num_samples=1)
                            else:
                                occ_id = torch.randint(0, OCC_CLASSES, (bsz, 1), device=device)
                            # age bucket (0..6)
                            if p_age is not None:
                                age_bucket = torch.multinomial(p_age.expand(bsz, -1), num_samples=1)
                            else:
                                age_bucket = torch.randint(0, age_bins, (bsz, 1), device=device)
                            cond_base = torch.cat([
                                g,
                                occ_id.float() / float(OCC_CLASSES - 1),
                                age_bucket.float() / float(max(1, age_bins - 1)),
                            ], dim=1)
                        else:
                            cond_base = torch.zeros(bsz, cond_dim, device=device)

                        # ---- Append genre to cond if enabled (projected or raw) ----
                        cond_full = cond_base
                        if include_genre and genre_dim > 0:
                            # sample a simple multi-label genre prior; tweak p if desired
                            genre01 = (torch.rand(bsz, genre_dim, device=device) < 0.30).float()
                            proj = getattr(gan, "cond_genre_proj", None)
                            if proj is not None and proj_dim > 0:
                                genre_cond = torch.sigmoid(proj(genre01))
                            else:
                                genre_cond = genre01
                            cond_full = genre_cond if cond_full is None else torch.cat([cond_full, genre_cond], dim=1)

                        # ---- Sample and map to [0,1] (shape [bsz,45]) ----
                        fake_pm1 = gan.sample(bsz, cond=cond_full, device=device)  # [-1,1]
                        fake_01  = (fake_pm1 + 1.0) * 0.5

                        # ---- Enforce label columns to equal our sampled *base* condition ----
                        if cond_base is not None:
                            if cond_dim >= 1:
                                # gender
                                fake_01[:, gender_idx] = cond_base[:, 0]
                            if cond_dim == 3:
                                # occupation block (one-hot from occ_id)
                                s_occ, w_occ = occ_start, occ_width
                                fake_01[:, s_occ:s_occ+w_occ] = 0.0
                                fake_01[torch.arange(bsz, device=device), s_occ + occ_id.view(-1)] = 1.0
                                # age years/100 from bucket
                                age_codes = AGE_CODES.to(device)[ (cond_base[:, 2] * float(max(1, age_bins-1))).round().long() ]
                                fake_01[:, age_idx] = (age_codes / 100.0)

                        if made == 0:
                            m, M = float(fake_01.min()), float(fake_01.max())
                            occ_sum = fake_01[:, occ_start:occ_start+occ_width].sum(dim=1).mean().item()
                            print(f"[GEN] fake_01 shape={tuple(fake_01.shape)} range=({m:.3f},{M:.3f}) | mean(sum OCC one-hot)={occ_sum:.3f}")

                        xs_45.append(fake_01.cpu())
                        made += bsz

                data_45 = torch.cat(xs_45, dim=0)[:n_samples]   # [N,45] in [0,1]

                # ---- Build a canonical 25-D view by collapsing the one-hot occ to a scalar ----
                s_occ, w_occ = occ_start, occ_width
                s_genre, e_genre = genre_start, genre_end
                occ_block = data_45[:, s_occ:s_occ+w_occ]
                occ_id_c  = occ_block.argmax(dim=1)
                occ_scalar = occ_id_c.float() / float(max(1, OCC_CLASSES - 1))

                data_25 = torch.cat([
                    data_45[:, 0:2],                                  # uid, iid
                    data_45[:, age_idx:age_idx+1],                    # age (/100)
                    data_45[:, gender_idx:gender_idx+1],              # gender
                    occ_scalar.unsqueeze(1),                          # occ scalar 0..1
                    data_45[:, s_genre:e_genre],                      # 19 genres
                    data_45[:, rating_idx:rating_idx+1],             # rating (/5)
                ], dim=1).clamp(0, 1)                                 # [N,25]

                # ---- Unpack canonical layout ----
                user_sc, item_sc = data_25[:, 0], data_25[:, 1]
                age_sc, gender_sc = data_25[:, age_idx], data_25[:, gender_idx]
                genre = data_25[:, 5:5+genre_dim]
                rating_sc = data_25[:, rating_idx - (w_occ - 1)]  # shift index into 25-D view

                # ---- Denorm (structured view) ----
                user_id   = (user_sc * (umax - umin) + umin).round().clamp_(umin, umax).to(torch.int64)
                item_id   = (item_sc * (imax - imin) + imin).round().clamp_(imin, imax).to(torch.int32)
                gender_i  = (gender_sc >= 0.5).to(torch.int8)
                age_years = _snap_age_ml1m((age_sc * 100.0).clamp_(0, 120)).to(torch.float32)
                occ_id    = occ_id_c.to(torch.int8)
                rating    = (1.0 + rating_sc * 4.0).clamp_(1.0, 5.0).to(torch.float32)

                # Ensure at least one genre active
                genre_bin = (genre >= 0.5).to(torch.float32)
                zero_mask = (genre_bin.sum(dim=1) == 0)
                if zero_mask.any():
                    idx = torch.argmax(genre[zero_mask], dim=1)
                    genre_bin[zero_mask, idx] = 1.0

                # ---- Repack normalized 25-D for downstream ----
                u_den = float(max(1, umax - 1))
                i_den = float(max(1, imax - 1))
                full_vec = torch.cat([
                    ((user_id.float() - 1.0) / u_den).unsqueeze(1),
                    ((item_id.float() - 1.0) / i_den).unsqueeze(1),
                    (age_years.unsqueeze(1) / 100.0),
                    (gender_i.unsqueeze(1).float()),
                    (occ_id.unsqueeze(1).float() / float(OCC_CLASSES - 1)),
                    genre_bin,
                    (rating.unsqueeze(1) / 5.0),
                ], dim=1).clamp(0, 1)  # [N,25]

                # ---- Save HDF5 ----
                tag = f"{engine_name}_{regime}"
                out_path = os.path.join(output_dir, f"{tag}.hdf5")
                with h5py.File(out_path, "w") as f:
                    # normalized 25-D
                    f.create_dataset("synthetic", data=full_vec.numpy())

                    # structured table
                    dt = np.dtype([
                        ("user_id", np.int64),
                        ("movie_id", np.int32),
                        ("pred_rating", np.float32),
                        ("gender", np.int8),
                        ("age", np.float32),
                        ("occupation", np.int8),
                        ("genre", np.float32, (genre.shape[1],)),
                    ])
                    structured = np.zeros(len(user_id), dtype=dt)
                    structured["user_id"] = user_id.numpy()
                    structured["movie_id"] = item_id.numpy()
                    structured["pred_rating"] = rating.numpy()
                    structured["gender"] = gender_i.numpy()
                    structured["age"] = age_years.numpy()
                    structured["occupation"] = occ_id.numpy()
                    structured["genre"] = genre_bin.numpy()
                    f.create_dataset("predictions", data=structured)

                    # metadata
                    f.attrs["source"] = f"wgan:{engine_name}:{regime}"
                    f.attrs["pred_file"] = pred_name if pred_name else ""
                    f.attrs["global_mean"] = float(rating.mean().item())
                    f.attrs["num_predictions"] = int(len(user_id))
                    f.attrs["dataset"] = "ml-1m-compatible"

                print(f"[✓] Saved synthetic data to: {out_path}")



    if getattr(cli.config, "do_classifier_attack", False):
        run_yaml_attack_sweep(cli.config, cli.datamodule, cli.trainer)

if __name__ == "__main__":
    main()
