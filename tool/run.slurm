#!/bin/bash
#SBATCH --job-name=run       # Job name
#SBATCH --output=run_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=run_%j.err         # Error log
#SBATCH --time=00:30:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
module load python/3.12.1

# Go to your project directory
# cd /u/zluo8/qwen
cd /projects/bfdz/zluo8/translate2/my_bfcl
source activate_environment.sh
cd /projects/bfdz/zluo8/translate2/my_bfcl
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

python main.py --config config_slurm.py --num-gpus 1