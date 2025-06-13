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
import wandb
import torch
import os


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
        # DP-specific metrics
        if hasattr(self.model, "privacy_engine"):
             wandb.define_metric("epsilon", step_metric="epoch", summary="max")


def main():

    import h5py

    def inspect_file(path):
        print(f"\n📂 Inspecting file: {path}")
        with h5py.File(path, 'r') as f:
            def print_structure(name, obj):
                print(f"  {name} -> {type(obj)}")
            f.visititems(print_structure)



    # LightningCLI automatically creates an argparse parser with required arguments and types,
    # and instantiates the model and datamodule. For this, it's important to import the model and datamodule classes above.
    cli = MyLightningCLI(None, NetflixDataModule,subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
    #cli.wandb_setup()

    
    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule,
                        ckpt_path=cli.config.ckpt_path)

    # If we have trained a model, use the best checkpoint for testing and predicting.
    # Without this, the model's state at the end of the training would be used, which is not necessarily the best.
    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        # Run normal test first (with standard model)
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

        # Collect inputs and outputs during testing
        user_indices = []
        movie_indices = []
        ratings = []
        outputs = []

        for batch in cli.datamodule.test_dataloader():
            # Assuming the batch contains user indices and movie indices
            user_idx, movie_idx = batch[:2]  # Adjust based on your dataloader structure
            user_indices.append(user_idx)
            movie_indices.append(movie_idx)

            # Get model outputs
            with torch.no_grad():
                output = cli.model(user_idx, movie_idx)
                outputs.append(output)

            # Assuming ratings are derived from the model's output
            rating = output  # Replace this with the actual logic to compute ratings
            ratings.append(rating)

        # Save the collected data
        torch.save({
            "user_idx": torch.cat(user_indices),
            "movie_idx": torch.cat(movie_indices),
            "ratings": torch.cat(ratings),
            "model_outputs": torch.cat(outputs)
        }, "blackbox_dataset.pt")

    if cli.config.do_predict:
        predictions = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_analyze:
        # Load the MIA model checkpoint
        #mia_model = DPMembershipInferenceAttack.load_dp_checkpoint(
           # "lightning_logs/strong/dp_fm/nnudexlz/checkpoints/epoch=3-step=10000.ckpt"
       # )
        mia_model = CustomMembershipInferenceAttack.load_from_checkpoint("lightning_logs/cus_str/dp_fm/8a7jwgzx/checkpoints/epoch=36-step=92500.ckpt")

        # Define the paths for your member and non-member datasets
        member_data_path = "netflix_data/netflix_data.hdf5"
        nonmember_data_path = "netflix_data/movies.hdf5"

        # Get the DataLoader that includes both member and non-member datasets
        mia_loader = cli.datamodule.mia_dataloaders(member_data_path, nonmember_data_path)[0]

        # Run the test step with the appropriate DataLoader
        cli.trainer.test(mia_model, dataloaders=[mia_loader])



if __name__ == "__main__":
    main()


