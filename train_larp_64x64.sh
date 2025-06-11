#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:h200:4
#SBATCH --time=2880
#SBATCH --cpus-per-task=128
#SBATCH --job-name=64x64
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

torchrun --nproc-per-node=4 --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_hi_c2i_larp_ibq_caption_0.001_prior_0.001_multiplier_10_64x64_pretrain" \
    model.context_dim=56 model.codebook_size_for_entropy=14 model.quantization_type="larp_ibq" \
    model.patch_size=4 model.mup_width=4 model.code_length=64 \
    prior.model_name="Qwen3-0.6B" \
    prior.random_init=False \
    prior.stop_grad=False \
    prior.loss_weight=0.001 \
    prior.caption_loss_weight=0.001 \
    prior.lr_multiplier=10 \
    data.batch_size=32 \
    data.eval_batch_size=40 \
    data.image_size=64 \
    opt.n_grad_acc=1 \
    opt.lr=0.0001 \
    opt.freeze_encoder_after=40000 \
    trainer.max_steps=400000 \
    trainer.checkpoint_every=20000 \
    trainer.keep_every=20000
