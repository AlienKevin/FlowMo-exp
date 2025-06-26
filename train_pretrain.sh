#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2880
#SBATCH --cpus-per-task=64
#SBATCH --job-name=rand
#SBATCH --output=%j_output.txt
#SBATCH --error=%j_error.txt

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source .venv/bin/activate

# Generate a random master port to avoid collision
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

torchrun --nproc-per-node=2 --master_port=$MASTER_PORT -m pretrain
