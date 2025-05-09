#!/bin/bash
#SBATCH --job-name=Custom_Dp_FM        # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Number of CPUs per node
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=matador          # Partition name
#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=Custom_Dp_fm.log          # Output log file


# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate wildfire

# Run the script
python src/Train.py --config=cfgs/custom/custom_dp.yaml --trainer=cfgs/custom/custom_trainer.yaml --data=cfgs/data_loader.yaml --seed_everything=0 --trainer.max_epochs=5 --do_test=True --data.data_dir netflix_data