#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=2880
#SBATCH --cpus-per-task=32
#SBATCH --job-name=tar50x.006
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

torchrun --nproc-per-node=2 --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_lfq_qwen_hi_targets_sg_50xlr_ce_0.006_pretrain" \
    model.context_dim=56 model.codebook_size_for_entropy=14 model.quantization_type="lfq_qwen" \
    model.patch_size=8 model.mup_width=4 \
    model.qwen_ce_loss_weight=0.006 \
    opt.n_grad_acc=2 \
    opt.lr=0.000025 \
    opt.qwen_lr=0.00125 \
    trainer.max_steps=800000 \
    trainer.checkpoint_every=10000 \
    trainer.keep_every=10000
