#!/bin/bash
#SBATCH --job-name=pluss_beta_train
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4          # Number of GPUs per node
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --mem=200G                   # Memory per node
#SBATCH --time=72:00:00              # Maximum runtime (3 days)
#SBATCH --partition=gpu              # Partition name
#SBATCH --output=logs/pluss_beta_%j.out
#SBATCH --error=logs/pluss_beta_%j.err

# ============================================================================
# SLURM Multi-Node Multi-GPU Training Script for PLUSS_β
# ============================================================================

# Load modules (adjust according to your cluster)
module purge
module load cuda/11.8
module load python/3.9
module load nccl/2.15

# Activate virtual environment
source /path/to/your/venv/bin/activate

# Configuration
DATA_ROOT="/path/to/imagenet-s"
VARIANT="ImageNetS50"
OUTPUT_DIR="./outputs_slurm_${SLURM_JOB_ID}"
BATCH_SIZE_PER_GPU=4
ACCUMULATION_STEPS=2

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Print job information
echo "================================"
echo "SLURM Job Information"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))"
echo "Master node: $SLURM_NODELIST"
echo "================================"

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Print distributed training info
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))"

# Launch training with srun
srun python pluss_beta/train_multi_gpu.py \
    --data_root $DATA_ROOT \
    --variant $VARIANT \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --dino_config groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --dino_checkpoint groundingdino_swint_ogc.pth \
    --batch_size $BATCH_SIZE_PER_GPU \
    --accumulation_steps $ACCUMULATION_STEPS \
    --num_epochs 1000 \
    --semantic_lr 1e-4 \
    --box_lr 1e-4 \
    --use_amp \
    --memory_capacity 1000 \
    --hard_threshold 0.5 \
    --alpha 0.7 \
    --beta 0.3 \
    --num_prompts 16 \
    --lambda_l1 1.0 \
    --lambda_giou 2.0 \
    --save_freq 100 \
    --eval_freq 50 \
    --num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project pluss-beta-slurm \
    --dist_backend nccl

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
