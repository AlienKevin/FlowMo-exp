#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:h200:2
#SBATCH --time=1440
#SBATCH --cpus-per-task=64
#SBATCH --job-name=ibq
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

torchrun --nproc-per-node=2 --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_lo_ibq_128x128_pretrain" \
    model.context_dim=18 model.codebook_size_for_entropy=9 model.quantization_type="ibq" \
    model.patch_size=8 model.mup_width=4 model.code_length=128 \
    data.batch_size=128 \
    data.eval_batch_size=40 \
    data.image_size=128 \
    opt.n_grad_acc=1 \
    opt.lr=0.0001 \
    opt.freeze_encoder_after=999999999 \
    trainer.max_steps=200000 \
    trainer.checkpoint_every=50000 \
    trainer.keep_every=50000
