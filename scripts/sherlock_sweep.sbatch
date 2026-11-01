#!/bin/bash
# Run the whole sweep on Sherlock as a Slurm job array: one training per array task.
# Submit from the repo root (the array size is set on the command line):
#   sbatch --array=0-$(( $(python src/sweep.py --list | wc -l) - 1 )) scripts/sherlock_sweep.sbatch
#SBATCH --job-name=hc-sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

source "$HOME/hcenv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
python src/sweep.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
