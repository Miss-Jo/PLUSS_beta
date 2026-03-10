#!/bin/bash

# Quick start script for PLUSS_β training on ImageNet-S

# Set your paths here
DATA_ROOT="/path/to/imagenet-s"
OUTPUT_DIR="./outputs"
VARIANT="ImageNetS50"  # or ImageNetS300, ImageNetS919

# Model checkpoints (download these first)
SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
DINO_CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT="groundingdino_swint_ogc.pth"

# Training hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=1000
SEMANTIC_LR=1e-4
BOX_LR=1e-4

# Memory bank settings
MEMORY_CAPACITY=1000
HARD_THRESHOLD=0.5
ALPHA=0.7
BETA=0.3

# Create output directory
mkdir -p $OUTPUT_DIR

echo "================================"
echo "PLUSS_β Training Quick Start"
echo "================================"
echo "Dataset: $VARIANT"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "================================"

# Check if data exists
if [ ! -d "$DATA_ROOT/$VARIANT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT/$VARIANT"
    echo "Please set DATA_ROOT correctly"
    exit 1
fi

# Run training
python pluss_beta/train.py \
    --data_root $DATA_ROOT \
    --variant $VARIANT \
    --sam_checkpoint $SAM_CHECKPOINT \
    --dino_config $DINO_CONFIG \
    --dino_checkpoint $DINO_CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --semantic_lr $SEMANTIC_LR \
    --box_lr $BOX_LR \
    --memory_capacity $MEMORY_CAPACITY \
    --hard_threshold $HARD_THRESHOLD \
    --alpha $ALPHA \
    --beta $BETA \
    --num_prompts 16 \
    --lambda_l1 1.0 \
    --lambda_giou 2.0 \
    --save_freq 100 \
    --eval_freq 50 \
    --num_workers 4 \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project pluss-beta

echo "================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "================================"
