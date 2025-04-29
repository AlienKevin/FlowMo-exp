#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2880
#SBATCH --cpus-per-task=16
#SBATCH --job-name=lfq_repa
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

code_length=512
vocab_size=12

torchrun --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_lfq_repa_${code_length}_$(( 2 ** vocab_size ))_pretrain" \
    model.context_dim=${vocab_size} model.codebook_size_for_entropy=$(( vocab_size / 2 )) model.quantization_type=lfq \
    model.code_length=${code_length} \
    model.enable_repa=True \
    model.repa_loss_weight=0.5 \
    model.repa_layer_idx=5 \
    model.enable_cls=True \
    trainer.max_steps=400000 \
    trainer.checkpoint_every=10000 \
    trainer.keep_every=5000

echo "Done"
exit 0
