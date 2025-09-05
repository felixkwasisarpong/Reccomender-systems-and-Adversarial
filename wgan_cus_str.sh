#!/bin/bash
#SBATCH --job-name=wgan_cus_str        # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 1 process (don’t oversubscribe GPUs)
#SBATCH --cpus-per-task=2           # all 40 CPUs feed your dataloaders
#SBATCH --gpus-per-node=1            # 2 GPUs on the node
#SBATCH --mem=0                      # full node memory (or set explicitly if needed)
#SBATCH --partition=matador          # Partition name

#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=cus_str.log          # Output log file
#SBATCH --error=cus_str.err      # Separate error log is useful
# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate wildfire

python src/Train.py --config=cfgs/wgan_cus_str/gan.yaml --trainer=cfgs/wgan_cus_str/trainer.yaml --data=cfgs/wgan_cus_str/data_loader.yaml  --seed_everything=0 --trainer.max_epochs=200 --do_attack=True --do_train=False --do_validate=False --do_test=False --data.data_dir predicted_data