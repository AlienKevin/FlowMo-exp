#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=720
#SBATCH --cpus-per-task=16
#SBATCH --job-name=larp
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

batch_size=8

torchrun --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_lfq_hi_batch_size_${batch_size}_pretrain" \
    model.context_dim=56 model.codebook_size_for_entropy=14 model.quantization_type="lfq" \
    data.batch_size=${batch_size} \
    trainer.max_steps=800000 \
    trainer.checkpoint_every=5000 \
    trainer.keep_every=5000
