#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --time=360
#SBATCH --cpus-per-task=32
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

torchrun --nproc-per-node=1 -m flowmo.evaluate \
    --experiment-name "flowmo_hi_larp_qwen3_0.6b_64x64_pretrain_eval" \
    eval.eval_dir=results/flowmo_hi_larp_qwen3_0.6b_64x64_pretrain \
    eval.continuous=false \
    eval.subsample_rate=10 \
    eval.force_ckpt_path='results/flowmo_hi_larp_qwen3_0.6b_64x64_pretrain/checkpoints/00080000.pth' \
    model.context_dim=56 model.codebook_size_for_entropy=14 model.quantization_type="larp" \
    model.patch_size=4 model.mup_width=4 model.code_length=64 \
    prior.model_name="Qwen3-0.6B" \
    data.eval_batch_size=500 \
    data.image_size=64
