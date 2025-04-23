#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=720
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qwen2.5-coder-0.5b
#SBATCH --output=%j_output.txt
#SBATCH --error=%j_error.txt

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Source conda configuration and activate environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate FlowMo
torchrun -m flowmo.train \
    --experiment-name "flowmo_qwen2.5-coder-0.5b_pretrain" \
    --resume-from-ckpt "results/flowmo_qwen2.5-coder-0.5b_pretrain/checkpoints/00040000.pth" \
    model.context_dim=896 model.quantization_type=qwen2.5-coder-0.5b model.code_length=128 \
    trainer.max_steps=400000

echo "Done"
exit 0
