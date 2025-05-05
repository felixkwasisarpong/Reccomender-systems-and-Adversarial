import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
from models.BaseModel import BaseModel  # noqa
from models.DPModel import DPModel  # noqa
from models.CustomDP_SGD import CustomDP_SGD
from dataloader.NetflixDataModule import NetflixDataModule
from models.MembershipInferenceAttack import MembershipInferenceAttack  # replace with your actual path
from models.DPMembershipInferenceAttack import DPMembershipInferenceAttack  # replace with your actual path
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



    if cli.config.do_predict:
        predictions = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_analyze:
        #mia_model = MembershipInferenceAttack.load_from_checkpoint("lightning_logs/base/dp_fm/blqsy8mu/checkpoints/epoch=0-step=5000.ckpt")
        mia_model = DPMembershipInferenceAttack.load_dp_checkpoint("lightning_logs/weak/dp_fm/16kn7gdh/checkpoints/epoch=1-step=10000.ckpt")
 
        # Define the paths for your member and non-member datasets
        member_data_path = "netflix_data/member"
        nonmember_data_path = "netflix_data/nonmember"
        

        mia_loaders = cli.datamodule.mia_dataloaders(member_data_path, nonmember_data_path)

        
        # Ensure there are two dataloaders before calling test
        print(f"MIA loader count: {len(mia_loaders)}")
        # Run it for both files:

        # Run the test with the dataloaders
        cli.trainer.test(mia_model, dataloaders=mia_loaders)



if __name__ == "__main__":
    main()


