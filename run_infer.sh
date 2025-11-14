#!/bin/bash

#SBATCH --job-name=[your-job-name]
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:[your-gpu-type]
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=[your-output-dir]/bash_logs/%x-%j.out
#SBATCH --error=[your-output-dir]/bash_logs/%x-%j.err

export PYTHONPATH=$PYTHONPATH:.
#export WANDB_DIR=/scratch/song.liy/wandb
#export WANDB_CACHE_DIR=/scratch/song.liy/.cache/wandb
#export WANDB_DATA_DIR=/scratch/song.liy/.cache/wandb-data
export CUBLAS_WORKSPACE_CONFIG=":4096:8"  # For reproducibility
export PYTHONHASHSEED="42"  # For reproducibility

# Activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate respenv

CONFIG="configs/inference/deepphys_deep_infer.yaml"
CHECKPOINT="checkpoints/deepphys_deep_body.pth"

echo "Running config: $CONFIG with model: $CHECKPOINT"
python infer.py --config "$CONFIG" --checkpoint "$CHECKPOINT"
echo "Finished config: $CONFIG"
