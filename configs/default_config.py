"""
Example configuration for PLUSS_β training
"""

# Dataset configuration
DATA_CONFIG = {
    'root_dir': '/path/to/imagenet-s',
    'variant': 'ImageNetS50',  # or 'ImageNetS300', 'ImageNetS919'
    'batch_size': 8,
    'num_workers': 4,
    'image_size': 512
}

# Model configuration
MODEL_CONFIG = {
    # CLIP
    'clip_model': 'ViT-B/16',
    
    # SAM
    'sam_checkpoint': 'sam_vit_h_4b8939.pth',
    'sam_model_type': 'vit_h',
    
    # Grounding DINO
    'dino_config': 'groundingdino/config/GroundingDINO_SwinT_OGC.py',
    'dino_checkpoint': 'groundingdino_swint_ogc.pth',
    
    # Semantic Tuner
    'num_layers': 12,
    'embed_dim': 512,
    'num_prompts': 16,
    'dropout': 0.1,
    
    # Box Tuner
    'feature_dim': 512,
    'num_heads': 8,
    'hidden_dim': 2048,
    
    # Temperature parameter
    'temperature': 0.07
}

# Training configuration
TRAIN_CONFIG = {
    # Optimization
    'num_epochs': 1000,
    'semantic_lr': 1e-4,
    'box_lr': 1e-4,
    'weight_decay': 0.01,
    
    # Memory Bank
    'memory_capacity': 1000,
    'hard_threshold': 0.5,  # sigma in paper
    'alpha': 0.7,  # IoU loss weight
    'beta': 0.3,   # Dice loss weight
    
    # Semantic Tuner Training
    'semantic_batch_size': 32,
    'semantic_train_interval': 100,  # Train every 100 epochs
    
    # Box Tuner Loss
    'lambda_l1': 1.0,
    'lambda_giou': 2.0,
    
    # Point-2-Box
    'min_pts': 3,
    
    # Checkpointing
    'save_freq': 100,
    'eval_freq': 50
}

# System configuration
SYSTEM_CONFIG = {
    'device': 'cuda',
    'seed': 42,
    'output_dir': './outputs',
    'use_wandb': False,
    'wandb_project': 'pluss-beta',
    'wandb_entity': None
}

# Full configuration
CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
    **SYSTEM_CONFIG
}
