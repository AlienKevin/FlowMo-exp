#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=360
#SBATCH --cpus-per-task=128
#SBATCH --job-name=eval
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

# Generate a random master port to avoid collision
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

torchrun --nproc-per-node=2 -m flowmo.evaluate \
    --experiment-name "flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain_eval" \
    eval.eval_dir=results/flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain \
    eval.continuous=false \
    eval.subsample_rate=10 \
    eval.force_ckpt_path='results/flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain/checkpoints/00200000.pth' \
    model.context_dim=56 model.codebook_size_for_entropy=14 \
    model.patch_size=8 model.mup_width=4
