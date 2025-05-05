#!/bin/bash
#SBATCH --job-name=Base_FM        # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=8         # Number of CPUs per node
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=matador          # Partition name
#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=base_fm.log
#SBATCH --error=base_fm.err      # Separate error log is useful



# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate wildfire

# Run the script
python src/Train.py --config=cfgs/basemodel/basemodel.yaml --trainer=cfgs/basemodel/basemodel_trainer.yaml --data=cfgs/data_loader.yaml --seed_everything=0 --trainer.max_epochs=120 --do_test=True --data.data_dir netflix_data