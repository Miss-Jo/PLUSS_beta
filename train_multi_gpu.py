"""
Multi-GPU Training Script for PLUSS_β
Supports distributed training with torch.distributed.launch or torchrun
"""

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import wandb

# Import models
# import clip
import CLIP_Surgery.clip as clip
from groundingdino.util.inference import load_model as load_grounding_dino
# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry

# Import PLUSS_β components
from pluss_beta.trainer_distributed import DistributedPLUSSBetaTrainer
from pluss_beta.data.imagenet_s import ImageNetSDataset, get_imagenet_s_dataloaders
from pluss_beta.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    get_rank, get_world_size, barrier, print_on_main
)
from pluss_beta.utils.evaluation import evaluate_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Training for PLUSS_β')
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=-1,
                       help='World size for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                       help='Distributed backend')
    parser.add_argument('--dist_url', type=str, default='env://',
                       help='URL for distributed training')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of ImageNet-S dataset')
    parser.add_argument('--variant', type=str, default='ImageNetS50',
                       choices=['ImageNetS50', 'ImageNetS300', 'ImageNetS919'],
                       help='ImageNet-S variant')
    
    # Model
    parser.add_argument('--clip_model', type=str, default='CS-ViT-B/16',
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
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--semantic_lr', type=float, default=1e-4,
                       help='Learning rate for semantic tuner')
    parser.add_argument('--box_lr', type=float, default=1e-4,
                       help='Learning rate for box tuner')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Gradient accumulation and mixed precision
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable automatic mixed precision')
    
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
                       help='Number of data loading workers per GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic algorithms')
    
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


def setup_seed(seed, deterministic=False):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def create_dataloaders(args, world_size, rank):
    """
    Create distributed dataloaders
    
    Returns:
        train_loader, val_loader, class_names
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Create datasets
    train_dataset = ImageNetSDataset(
        root_dir=args.data_root,
        split='train',
        variant=args.variant,
        use_semi=False,
        transform=train_transform,
        return_mask=False,
        samples_per_class=10
    )
    
    val_dataset = ImageNetSDataset(
        root_dir=args.data_root,
        split='validation',
        variant=args.variant,
        transform=val_transform,
        return_mask=True
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names


def train_worker(local_rank, args):
    """
    Training worker for each GPU
    
    Args:
        local_rank: Local rank of current process
        args: Training arguments
    """
    # Setup distributed training
    if args.world_size > 1:
        setup_distributed(local_rank, args.world_size, args.dist_backend)
    
    # Set random seed
    setup_seed(args.seed + local_rank, args.deterministic)
    
    rank = get_rank()
    world_size = get_world_size()
    
    print_on_main("=" * 80)
    print_on_main("PLUSS_β Multi-GPU Training")
    print_on_main("=" * 80)
    print_on_main(f"World size: {world_size}")
    print_on_main(f"Dataset: {args.variant}")
    print_on_main(f"Batch size per GPU: {args.batch_size}")
    print_on_main(f"Effective batch size: {args.batch_size * world_size * args.accumulation_steps}")
    print_on_main(f"Gradient accumulation steps: {args.accumulation_steps}")
    print_on_main(f"Mixed precision: {args.use_amp}")
    print_on_main("=" * 80)
    
    # Initialize wandb (only on main process)
    if args.use_wandb and is_main_process():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.variant}_world{world_size}_bs{args.batch_size*world_size}"
        )
    
    # Load foundation models
    print_on_main("\nLoading foundation models...")
    
    # CLIP
    print_on_main(f"Loading CLIP {args.clip_model}...")
    clip_model, _ = clip.load(args.clip_model, device=f'cuda:{local_rank}')
    
    # SAM
    print_on_main(f"Loading SAM from {args.sam_checkpoint}...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)

    
    # Grounding DINO
    print_on_main(f"Loading Grounding DINO...")
    grounding_dino = load_grounding_dino(args.dino_config, args.dino_checkpoint)
    
    print_on_main("Foundation models loaded successfully!")
    
    # Create dataloaders
    print_on_main(f"\nLoading {args.variant} dataset from {args.data_root}...")
    train_loader, val_loader, class_names = create_dataloaders(args, world_size, rank)
    
    print_on_main(f"Train samples: {len(train_loader.dataset)}")
    print_on_main(f"Val samples: {len(val_loader.dataset)}")
    print_on_main(f"Batches per GPU: {len(train_loader)}")
    print_on_main(f"Number of classes: {len(class_names)}")
    
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
        'image_size':512,
        'min_pts': 3,
        'temperature': 0.07,
        'accumulation_steps': args.accumulation_steps,
        'use_amp': args.use_amp,
        'num_epochs': args.num_epochs
    }
    
    # Create distributed trainer
    print_on_main("\nInitializing distributed PLUSS_β trainer...")
    trainer = DistributedPLUSSBetaTrainer(
        clip_model=clip_model,
        sam_model=sam,
        sam_checkpoint=args.sam_checkpoint,
        grounding_dino=grounding_dino,
        config=config,
        local_rank=local_rank,
        world_size=world_size
    )
    trainer.set_variant(args.variant)
    # Resume from checkpoint if specified
    if args.resume:
        print_on_main(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if is_main_process():
        semantic_params = sum(p.numel() for p in trainer.semantic_tuner.parameters())
        box_params = sum(p.numel() for p in trainer.box_tuner.parameters())
        print(f"\nTraining configuration:")
        print(f"  Semantic Tuner: {semantic_params:,} parameters")
        print(f"  Box Tuner: {box_params:,} parameters")
        print(f"  Total trainable: {semantic_params + box_params:,} parameters")
        print(f"  Memory Bank: Capacity {args.memory_capacity}, Threshold {args.hard_threshold}")
    
    print_on_main("=" * 80)
    
    # Training loop
    print_on_main("\nStarting training...")
    best_miou = 0.0
    
    for epoch in range(trainer.current_epoch, args.num_epochs):
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        print_on_main(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Log metrics (only on main process)
        if is_main_process():
            print(f"\nTrain metrics:")
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
            
            if args.use_wandb:
                wandb.log(
                    {f"train/{k}": v for k, v in train_metrics.items()},
                    step=epoch
                )
        
        # Synchronize before evaluation
        barrier()
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            print_on_main(f"\nEvaluating on validation set...")
            val_metrics = evaluate_segmentation(
                trainer, val_loader, f'cuda:{local_rank}'
            )
            
            if is_main_process():
                print(f"Validation metrics:")
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                
                if args.use_wandb:
                    wandb.log(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=epoch
                    )
                
                # Save best model
                if val_metrics.get('mIoU', 0) > best_miou:
                    best_miou = val_metrics['mIoU']
                    best_path = os.path.join(args.output_dir, 'best_model.pth')
                    trainer.save_checkpoint(best_path, is_best=True)
                    print(f"New best mIoU: {best_miou:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)
        
        # Synchronize after each epoch
        barrier()
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    trainer.save_checkpoint(final_path)
    
    print_on_main("\nTraining completed!")
    print_on_main(f"Best mIoU: {best_miou:.4f}")
    print_on_main(f"Final model saved to: {final_path}")
    
    if args.use_wandb and is_main_process():
        wandb.finish()
    
    # Cleanup distributed training
    if world_size > 1:
        cleanup_distributed()


def main():
    args = parse_args()
    
    # Determine world size and local rank
    if args.local_rank == -1:
        # Try to get from environment variables (set by torch.distributed.launch)
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
        else:
            # Single GPU training
            args.local_rank = 0
            args.world_size = 1
    
    # Create output directory
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_worker(args.local_rank, args)


if __name__ == '__main__':
    main()
