"""
Main training script for PLUSS_β on ImageNet-S
"""

import os
import argparse
import torch
import wandb
from pathlib import Path

# Import models
import clip
from segment_anything import sam_model_registry
from groundingdino.util.inference import load_model as load_grounding_dino

# Import PLUSS_β components
from pluss_beta.trainer import PLUSSBetaTrainer
from pluss_beta.data.imagenet_s import get_imagenet_s_dataloaders
from pluss_beta.utils.evaluation import evaluate_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Train PLUSS_β on ImageNet-S')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of ImageNet-S dataset')
    parser.add_argument('--variant', type=str, default='ImageNetS50',
                       choices=['ImageNetS50', 'ImageNetS300', 'ImageNetS919'],
                       help='ImageNet-S variant')
    
    # Model
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                       help='CLIP model variant')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth',
                       help='SAM checkpoint path')
    parser.add_argument('--dino_config', type=str, 
                       default='groundingdino/config/GroundingDINO_SwinT_OGC.py',
                       help='Grounding DINO config path')
    parser.add_argument('--dino_checkpoint', type=str,
                       default='groundingdino_swint_ogc.pth',
                       help='Grounding DINO checkpoint path')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--semantic_lr', type=float, default=1e-4,
                       help='Learning rate for semantic tuner')
    parser.add_argument('--box_lr', type=float, default=1e-4,
                       help='Learning rate for box tuner')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Memory bank
    parser.add_argument('--memory_capacity', type=int, default=1000,
                       help='Memory bank capacity')
    parser.add_argument('--hard_threshold', type=float, default=0.5,
                       help='Hard example threshold (sigma)')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Alpha for mask loss (IoU weight)')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='Beta for mask loss (Dice weight)')
    
    # Semantic tuner
    parser.add_argument('--num_prompts', type=int, default=16,
                       help='Number of learnable prompts per layer')
    parser.add_argument('--semantic_batch_size', type=int, default=32,
                       help='Batch size for semantic tuner training')
    
    # Box tuner
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                       help='Weight for L1 loss')
    parser.add_argument('--lambda_giou', type=float, default=2.0,
                       help='Weight for GIoU loss')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Checkpoint saving frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=50,
                       help='Evaluation frequency (epochs)')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='pluss-beta',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity name')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    print("=" * 80)
    print("PLUSS_β Training on ImageNet-S")
    print("=" * 80)
    print(f"Dataset: {args.variant}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print("=" * 80)
    
    # Load foundation models
    print("\nLoading foundation models...")
    
    # CLIP
    print(f"Loading CLIP {args.clip_model}...")
    clip_model, _ = clip.load(args.clip_model, device=args.device)
    
    # SAM
    print(f"Loading SAM from {args.sam_checkpoint}...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    
    # Grounding DINO
    print(f"Loading Grounding DINO...")
    grounding_dino = load_grounding_dino(args.dino_config, args.dino_checkpoint)
    
    print("Foundation models loaded successfully!")
    
    # Create dataloaders
    print(f"\nLoading {args.variant} dataset from {args.data_root}...")
    train_loader, val_loader, class_names = get_imagenet_s_dataloaders(
        root_dir=args.data_root,
        variant=args.variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=512
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create config
    config = {
        'num_layers': 12,
        'embed_dim': 512,
        'feature_dim': 512,
        'num_prompts': args.num_prompts,
        'num_heads': 8,
        'hidden_dim': 2048,
        'dropout': 0.1,
        'semantic_lr': args.semantic_lr,
        'box_lr': args.box_lr,
        'weight_decay': args.weight_decay,
        'memory_capacity': args.memory_capacity,
        'hard_threshold': args.hard_threshold,
        'alpha': args.alpha,
        'beta': args.beta,
        'semantic_batch_size': args.semantic_batch_size,
        'lambda_l1': args.lambda_l1,
        'lambda_giou': args.lambda_giou,
        'min_pts': 3,
        'image_size': 512,
        'temperature': 0.07
    }
    
    # Create trainer
    print("\nInitializing PLUSS_β trainer...")
    trainer = PLUSSBetaTrainer(
        clip_model=clip_model,
        sam_model=sam,
        grounding_dino=grounding_dino,
        config=config,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\nTraining configuration:")
    print(f"  Semantic Tuner: {sum(p.numel() for p in trainer.semantic_tuner.parameters())} parameters")
    print(f"  Box Tuner: {sum(p.numel() for p in trainer.box_tuner.parameters())} parameters")
    print(f"  Memory Bank: Capacity {args.memory_capacity}, Threshold {args.hard_threshold}")
    print("=" * 80)
    
    # Training loop
    print("\nStarting training...")
    best_miou = 0.0
    
    for epoch in range(trainer.current_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Log metrics
        print(f"\nTrain metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        if args.use_wandb:
            wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            print(f"\nEvaluating on validation set...")
            val_metrics = evaluate_segmentation(
                trainer, val_loader, args.device
            )
            
            print(f"Validation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            if args.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            
            # Save best model
            if val_metrics.get('mIoU', 0) > best_miou:
                best_miou = val_metrics['mIoU']
                best_path = output_dir / 'best_model.pth'
                trainer.save_checkpoint(str(best_path))
                print(f"New best mIoU: {best_miou:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            trainer.save_checkpoint(str(checkpoint_path))
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    trainer.save_checkpoint(str(final_path))
    
    print("\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Final model saved to: {final_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
