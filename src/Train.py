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
from models.AttributeClassifier import AttributeClassifier
from models.DPFM_GANTrainer import DPFM_GANTrainer  # replace with your actual path
from models.WGAN import WGAN  # replace with your actual path
from models.DPFMMembershipInferenceAttack import DPFMMembershipInferenceAttack  # noqa
import wandb
import torch
import os
import numpy as np
import h5py



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
        # 1) Load the specified DP-FM checkpoint
        ckpt_path = "/Users/Apple/Documents/assignements/Thesis/lightning_logs/custom_strong/dp_fm/y8fih0tw/checkpoints/epoch=10-step=3432.ckpt"
        cli.model = CustomDP_SGD.load_from_checkpoint(str(ckpt_path))
        cli.model.eval()
        
        # 2) Prepare datamodule & collect predictions
        cli.datamodule.setup('predict')
        user_indices = []
        movie_indices = []
        predicted_ratings = []
        attr_buffers = {k: [] for k in ("gender", "age", "occupation", "genre")}

        with torch.no_grad():
            for batch in cli.datamodule.predict_dataloader():
                batch = {k: v.to(cli.model.device) for k, v in batch.items()}
                preds = cli.model(batch)

                user_indices .append(batch["user_id"].cpu())
                movie_indices.append(batch["item_id"].cpu())
                predicted_ratings.append(preds.cpu())

                
                for k in attr_buffers:
                    if k in batch:
                        attr_buffers[k].append(batch[k].cpu())

        # 3) Concatenate to NumPy
        user_np   = torch.cat(user_indices)   .numpy().astype(np.int64)
        movie_np  = torch.cat(movie_indices)  .numpy().astype(np.int32)
        pred_np   = torch.cat(predicted_ratings).numpy().astype(np.float32)

        # 4) Build structured array dtype
        fields = [
            ("user_id",    np.int64),
            ("movie_id",   np.int32),
            ("pred_rating",np.float32),
        ]

        fields += [
            ("gender",     np.int8),
            ("age",        np.float32),
            ("occupation", np.int8),
            ("genre",      np.float32, (batch["genre"].shape[-1],))
            ]
        dt = np.dtype(fields)

        # 5) Fill it
        structured = np.zeros(len(user_np), dtype=dt)
        structured["user_id"]     = user_np
        structured["movie_id"]    = movie_np
        structured["pred_rating"] = pred_np

        structured["gender"]     = torch.cat(attr_buffers["gender"]).numpy().astype(np.int8)
        structured["age"]        = torch.cat(attr_buffers["age"])   .numpy().astype(np.float32)
        structured["occupation"] = torch.cat(attr_buffers["occupation"]).numpy().astype(np.int8)
        structured["genre"]      = torch.cat(attr_buffers["genre"]) .numpy().astype(np.float32)

        # 6) Save to HDF5, naming by the checkpoint stem
        out_name = "predicted_data/custom_strong.hdf5"
        with h5py.File(out_name, "w") as f:
            f.create_dataset("predictions", data=structured)
            f.attrs["global_mean"]    = float(pred_np.mean())
            f.attrs["num_predictions"]= len(pred_np)

        print(f"[✓] Saved {len(pred_np)} predictions to {out_name}")

    if cli.config.do_analyze:
        ckpt_path = "/Users/Apple/Documents/assignements/Thesis/lightning_logs/opacus_weak/dp_fm/7kyum5gd/checkpoints/epoch=18-step=5928.ckpt"
        
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
        attack_model = DPMembershipInferenceAttack.load_from_dp_checkpoint(ckpt_path)
        
        # Get MIA dataloader with improved strategy
        mia_loader = cli.datamodule.mia_dataloaders()[0]
        
        # Run the attack
        cli.trainer.test(attack_model, dataloaders=[mia_loader])
        
        print("MIA evaluation completed. Check 'mia_results/' directory for detailed plots and analysis.")


    if cli.config.do_attack:
        cli.datamodule.setup(stage="attack")
        attack_loader = cli.datamodule.attack_dataloader()
        cli.trainer.fit(cli.model, train_dataloaders=attack_loader)


    if cli.config.do_create_synthetic_data:
        engines = {
            "opacus": {
                "strong": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_dp_str/dp_fm/j6y7ldj4/checkpoints/epoch=11-step=281376.ckpt",
                "mid": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_dp_mid/dp_fm/ejqhno9h/checkpoints/epoch=17-step=422064.ckpt",
                "weak": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_dp_weak/dp_fm/ikbl6yen/checkpoints/epoch=37-step=891024.ckpt"
            },
            "custom": {
                "strong": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_cus_strong/dp_fm/par67if2/checkpoints/epoch=11-step=281376.ckpt",
                "mid": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_cus_mid/dp_fm/4c9ysf18/checkpoints/epoch=21-step=515856.ckpt",
                "weak": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/wgan_cus_weak/dp_fm/yploz3vr/checkpoints/epoch=31-step=750336.ckpt"
            },"Adv": {
                "strong": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/Adv_strong/dp_fm/8zl6lnxc/checkpoints/epoch=4-step=117240.ckpt",
                "mid": "",
                "weak": "/Users/Apple/Documents/assignements/Thesis/lightning_logs/Adv_weak/dp_fm/edzpd64r/checkpoints/epoch=18-step=445512.ckpt"
            }
        }

        output_dir = "synthetic_outputs"
        os.makedirs(output_dir, exist_ok=True)

        for engine_name, ckpts in engines.items():
            for regime, ckpt_path in ckpts.items():
                print(f"[{engine_name.upper()} - {regime.upper()}] Generating synthetic samples...")

                generator = WGAN.load_from_checkpoint(ckpt_path).to(cli.model.device)
                generator.eval()
                latent_dim = generator.hparams.noise_dim

                samples = []
                with torch.no_grad():
                    for _ in range((100000 + 255) // 256):
                        z = torch.randn(256, latent_dim).to(cli.model.device)
                        batch = generator(z).cpu()

                        samples.append(batch)

                data = torch.cat(samples, dim=0)[:100000]

                # Unpack synthetic features assuming structure:
                # [user_id, item_id, rating, gender, age, occupation, genre (19 dims)]
                user_id    = data[:, 0]
                item_id    = data[:, 1]
                gender     = data[:, 2]
                age        = data[:, 3]
                occupation = data[:, 4]
                genre      = data[:, 5:24]    # 19 dims → columns 5 to 23 inclusive
                rating     = data[:, 24]

                out_path = os.path.join(output_dir, f"{engine_name}_{regime}.hdf5")
                with h5py.File(out_path, "w") as f:

                    f.create_dataset("user_ids",     data=user_id.numpy().astype(np.int32))
                    f.create_dataset("item_ids",     data=item_id.numpy().astype(np.int32))
                    f.create_dataset("ratings",      data=rating.numpy().astype(np.float32))
                    f.create_dataset("gender",       data=gender.numpy().astype(np.int32))
                    f.create_dataset("age",          data=age.numpy().astype(np.float32))
                    f.create_dataset("occupation",   data=occupation.numpy().astype(np.int32))
                    f.create_dataset("genre_onehot", data=genre.numpy().astype(np.float32))

                print(f"[✓] Saved synthetic data to: {out_path}")



    if cli.config.do_classifier_attack:
        cli.datamodule.setup(stage="classifier_attack")

        cli.trainer.fit(
            cli.model,
            train_dataloaders=cli.datamodule.class_attack_train_dataloader(),
            val_dataloaders=cli.datamodule.class_attack_val_dataloader()
        )
        test_results = cli.trainer.test(
            model=cli.model,
            dataloaders=cli.datamodule.class_attack_test_dataloader()
        )


if __name__ == "__main__":
    main()


