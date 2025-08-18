#!/bin/bash
#SBATCH --job-name=dp_wgan_strong      # Job name
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=20         # Number of CPUs per node
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=matador          # Partition name
#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=dp_wgan_strong.log          # Output log file
#SBATCH --error=dp_wgan_strong.err      # Separate error log is useful

# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate   wildfire

python src/Train.py --config=cfgs/DPFM_GAN_strong/dpfm_gan.yaml --trainer=cfgs/DPFM_GAN_strong/trainer.yaml  --data=cfgs/DPFM_GAN_strong/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20  --seed_everything=0  --do_test=True --data.data_dir netflix_data
#python src/Train.py --config=cfgs/DPFM_GAN_strong/dpfm_gan.yaml --trainer=cfgs/DPFM_GAN_strong/trainer.yaml  --data=cfgs/DPFM_GAN_strong/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20 --do_analyze=True --do_train=False --do_validate=False --do_test=False --data.data_dir netflix_data
#python src/Train.py --config=cfgs/DPFM_GAN_strong/dpfm_gan.yaml --trainer=cfgs/DPFM_GAN_strong/trainer.yaml  --data=cfgs/DPFM_GAN_strong/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20 --do_predict=True --do_train=False --do_validate=False --do_test=False --data.data_dir netflix_data
#python src/Train.py --config=cfgs/wgan_dp/gan.yaml --trainer=cfgs/wgan_dp/trainer.yaml  --data=cfgs/wgan_dp/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20 --do_attack=True --do_train=False --do_validate=False --do_test=False --data.data_dir predicted_data
