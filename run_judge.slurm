#!/bin/bash
#SBATCH --job-name=run_judge       # Job name
#SBATCH --output=run_judge_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=run_judge_%j.err         # Error log
#SBATCH --time=00:15:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=16G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
module load python/3.12.1

# Go to your project directory
# cd /u/zluo8/qwen
cd /projects/bfdz/zluo8/tool_and_judge
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

python judge_run.py --config judge_config_slurm.py --num-gpus 1