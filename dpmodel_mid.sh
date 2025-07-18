#!/bin/bash
#SBATCH --job-name=DP_FM        # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=25         # Number of CPUs per node
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=matador          # Partition name
#SBATCH --time=20:00:00               # Time limit (2 hours)
#SBATCH --output=mid_fm.log          # Output log file
#SBATCH --error=mid_fm-Dp.err      # Separate error log is useful

# Initialize and activate conda environment
source $HOME/conda/etc/profile.d/conda.sh
# Load necessary modules
module load gcc/9.3.0
module load cuda/11.0          # Adjust based on the available CUDA version





conda activate wildfire

# Run the script
python src/Train.py --config=cfgs/dp_model_mid/db_FM.yaml --trainer=cfgs/dp_model_mid/dp_model_trainer.yaml --data=cfgs/data_loader.yaml --seed_everything=0 --trainer.max_epochs=1 --do_test=True --data.data_dir netflix_data

#python src/Train.py --config=cfgs/dp_model_mid/db_FM.yaml --trainer=cfgs/dp_model_mid/dp_model_trainer.yaml --data=cfgs/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20 --do_analyze=True --do_train=False --do_validate=False --do_test=False --data.data_dir netflix_data

#python src/Train.py --config=cfgs/dp_model_mid/db_FM.yaml --trainer=cfgs/dp_model_mid/dp_model_trainer.yaml --data=cfgs/data_loader.yaml --seed_everything=0 --trainer.max_epochs=20 --do_predict=True --do_train=False --do_validate=False --do_test=False --data.data_dir netflix_data
