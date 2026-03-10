#!/bin/bash

# Multi-GPU Training Launch Script for PLUSS_β
# Supports both torch.distributed.launch and torchrun

# ============================================================================
# Configuration - MODIFY THESE VARIABLES
# ============================================================================
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Number of GPUs to use
NUM_GPUS=4  # Change to your available GPU count

# Data path
#DATA_ROOT="/path/to/imagenet-s"
DATA_ROOT="/home/csu/sjj/0-LUSS/ImageNet-S-50"
VARIANT="ImageNetS50"  # ImageNetS50, ImageNetS300, or ImageNetS919

# Output directory
OUTPUT_DIR="./outputs_multi_gpu"

# Model checkpoints
SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
DINO_CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT="groundingdino_swint_ogc.pth"

# Training hyperparameters
BATCH_SIZE_PER_GPU=1 # Batch size per GPU 4
ACCUMULATION_STEPS=2  # Gradient accumulation steps 2
NUM_EPOCHS=500

# Effective batch size = BATCH_SIZE_PER_GPU * NUM_GPUS * ACCUMULATION_STEPS
# Example: 4 * 4 * 2 = 32  8*4*4=128

# Learning rates
SEMANTIC_LR=1e-4
BOX_LR=1e-4

# Mixed precision
USE_AMP="--use_amp"  # Set to "--no_amp" to disable

# Wandb logging
USE_WANDB="--use_wandb"  # Comment out to disable
WANDB_PROJECT="pluss-beta-multi-gpu"

# ============================================================================
# Launch Training
# ============================================================================

echo "================================"
echo "PLUSS_β Multi-GPU Training"
echo "================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Accumulation steps: $ACCUMULATION_STEPS"
echo "Effective batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS * ACCUMULATION_STEPS))"
echo "Dataset: $VARIANT"
echo "Output dir: $OUTPUT_DIR"
echo "================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if data exists
if [ ! -d "$DATA_ROOT/$VARIANT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT/$VARIANT"
    exit 1
fi

# Method 1: Using torch.distributed.launch (PyTorch < 1.10)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     pluss_beta/train_multi_gpu.py \

# Method 2: Using torchrun (PyTorch >= 1.10, Recommended)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    pluss_beta/train_multi_gpu.py \
    --data_root $DATA_ROOT \
    --variant $VARIANT \
    --sam_checkpoint $SAM_CHECKPOINT \
    --dino_config $DINO_CONFIG \
    --dino_checkpoint $DINO_CHECKPOINT \
    --batch_size $BATCH_SIZE_PER_GPU \
    --accumulation_steps $ACCUMULATION_STEPS \
    --num_epochs $NUM_EPOCHS \
    --semantic_lr $SEMANTIC_LR \
    --box_lr $BOX_LR \
    $USE_AMP \
    --memory_capacity 1000 \
    --hard_threshold 0.5 \
    --alpha 0.7 \
    --beta 0.3 \
    --num_prompts 16 \
    --lambda_l1 0.5\
    --lambda_giou 1.0 \
    --save_freq 100 \
    --eval_freq 50 \
    --num_workers 4 \
    --output_dir $OUTPUT_DIR \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT

echo "================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "================================"
