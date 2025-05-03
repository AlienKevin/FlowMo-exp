#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:8
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

torchrun --nproc-per-node=8 -m flowmo.evaluate \
    --experiment-name "flowmo_lo_posttrain_eval" \
    eval.eval_dir=results/flowmo_lo_posttrain \
    eval.continuous=false \
    eval.force_ckpt_path='flowmo_lo.pth' \
    model.context_dim=18 model.codebook_size_for_entropy=9 \
    model.patch_size=4 model.mup_width=6
