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

# Generate a random master port to avoid collision
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

span="0.3"

torchrun --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_qwen2.5-coder-0.5b_span_${span}_posttrain" \
    --resume-from-ckpt "results/flowmo_qwen2.5-coder-0.5b_span_${span}_pretrain/checkpoints/00095000.pth" \
    model.context_dim=896 model.quantization_type="qwen2.5-coder-0.5b_span_${span}" model.code_length=128 \
    trainer.max_steps=100000 \
    opt.lr=0.00005 \
    data.batch_size=8 \
    opt.n_grad_acc=2 \
    model.posttrain_sample=true \
    opt.lpips_mode='resnet' \
    opt.lpips_weight=0.01 \
    trainer.checkpoint_every=5000 \
    trainer.keep_every=5000 \

echo "Done"
exit 0
