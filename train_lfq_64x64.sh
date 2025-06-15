#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1440
#SBATCH --cpus-per-task=32
#SBATCH --job-name=ibq
#SBATCH --output=%j_output.txt
#SBATCH --error=%j_error.txt

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source fm/bin/activate

# Generate a random master port to avoid collision
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

torchrun --nproc-per-node=1 --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "tiny_flowmo_lo_ibq_64x64_pretrain" \
    data.imagenet_train_index="tiny_imagenet_train_index.json" \
    model.context_dim=18 model.codebook_size_for_entropy=9 model.quantization_type="ibq" \
    model.patch_size=4 model.mup_width=4 model.code_length=64 \
    data.batch_size=256 \
    data.eval_batch_size=40 \
    data.image_size=64 \
    opt.n_grad_acc=1 \
    opt.lr=0.0001 \
    opt.freeze_encoder_after=999999999 \
    trainer.max_steps=20000 \
    trainer.checkpoint_every=1000 \
    trainer.keep_every=1000
