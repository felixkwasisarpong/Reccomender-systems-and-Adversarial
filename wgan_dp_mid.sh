#!/bin/bash
#SBATCH --job-name=wgan_dp_mid        # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=40         # Number of CPUs per node
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=matador          # Partition name
#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=dp_mid.log          # Output log file
#SBATCH --error=dp_mid.err      # Separate error log is useful

# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate wildfire

python src/Train.py --config=cfgs/wgan_dp_mid/gan.yaml --trainer=cfgs/wgan_dp_mid/trainer.yaml  --data=cfgs/wgan_dp_mid/data_loader.yaml --seed_everything=0 --trainer.max_epochs=50 --do_attack=True --do_train=False --do_validate=False --do_test=False --data.data_dir netflix_data