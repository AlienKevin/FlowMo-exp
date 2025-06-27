#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:3090:8
#SBATCH --time=2880
#SBATCH --cpus-per-task=96
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
export MASTER_PORT=$(expr 13370 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

EXPERIMENT_NAME="flowmo_lo_c2i_larp_rand_sg_1.7b_ibq_128x128_pretrain"
# Set cache directories to a non-NFS path to avoid slowdowns
export TRITON_CACHE_DIR="/tmp/kevin02/${EXPERIMENT_NAME}_triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/kevin02/${EXPERIMENT_NAME}_torchinductor_cache"
mkdir -p $TRITON_CACHE_DIR
mkdir -p $TORCHINDUCTOR_CACHE_DIR

# Disable torch.compile to debug potential compilation errors
export TORCH_COMPILE_DISABLE=1

deepspeed --num_gpus=8 --master_port $MASTER_PORT flowmo/train.py \
    --deepspeed_config flowmo/zero3.json \
    --experiment-name "$EXPERIMENT_NAME" \
    model.context_dim=18 model.codebook_size_for_entropy=9 model.quantization_type="larp_ibq" \
    model.patch_size=8 model.mup_width=4 model.code_length=128 \
    prior.model_name="Qwen3-0.6B" \
    prior.random_init=True \
    prior.stop_grad=True \
    prior.loss_weight=0.001 \
    prior.lr_multiplier=1 \
    data.batch_size=16 \
    data.eval_batch_size=40 \
    data.image_size=128 \
    opt.n_grad_acc=2 \
    opt.lr=0.0001 \
    opt.freeze_encoder_after=50000 \
    trainer.max_steps=200000 \
    trainer.checkpoint_every=50000 \
    trainer.keep_every=50000
