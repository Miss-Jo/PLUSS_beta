"""
PLUSS_β Training Pipeline
Implements Algorithm 2 from the paper with separate computational graphs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import wandb
from collections import defaultdict

# Import models
from pluss_beta.models.semantic_tuner import SemanticTuner, SemanticTunerLoss
from pluss_beta.models.box_tuner import BoxTuner, BoxTunerLoss
from pluss_beta.models.memory_bank import MemoryBank
from pluss_beta.utils.point2box import Point2BoxConverter


class PLUSSBetaTrainer:
    """
    PLUSS_β Trainer with architectural principle:
    - Semantic Tuner and Box Tuner have COMPLETELY SEPARATE computational graphs
    - Semantic Tuner: Improves semantic discrimination of visual features
    - Box Tuner: Improves spatial localization accuracy
    - NO gradient flow between them during training
    """
    
    def __init__(self,
                 clip_model,
                 sam_model,
                 grounding_dino,
                 config,
                 device='cuda'):
        """
        Args:
            clip_model: Frozen CLIP model
            sam_model: Frozen SAM model
            grounding_dino: Grounding DINO model
            config: Training configuration
            device: Device to use
        """
        self.device = device
        self.config = config
        
        # Frozen foundation models
        self.clip_model = clip_model.to(device).eval()
        self.sam_model = sam_model.to(device).eval()
        self.grounding_dino = grounding_dino.to(device).eval()
        
        # Freeze foundation models
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.sam_model.parameters():
            param.requires_grad = False
        for param in self.grounding_dino.parameters():
            param.requires_grad = False
        
        # Initialize trainable components
        self.semantic_tuner = SemanticTuner(
            num_layers=config.get('num_layers', 12),
            embed_dim=config.get('embed_dim', 512),
            num_prompts=config.get('num_prompts', 16),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        self.box_tuner = BoxTuner(
            feature_dim=config.get('feature_dim', 512),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 2048),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        # Memory bank
        self.memory_bank = MemoryBank(
            capacity=config.get('memory_capacity', 1000),
            threshold=config.get('hard_threshold', 0.5),
            alpha=config.get('alpha', 0.7),
            beta=config.get('beta', 0.3)
        )
        
        # Point-to-box converter
        self.point2box = Point2BoxConverter(
            min_pts=config.get('min_pts', 3),
            image_size=(config.get('image_size', 512), config.get('image_size', 512))
        )
        
        # Loss functions
        self.semantic_loss_fn = SemanticTunerLoss(
            temperature=config.get('temperature', 0.07)
        ).to(device)
        
        self.box_loss_fn = BoxTunerLoss(
            lambda_l1=config.get('lambda_l1', 1.0),
            lambda_giou=config.get('lambda_giou', 2.0)
        ).to(device)
        
        # CRITICAL: Separate optimizers for separate computational graphs
        self.semantic_optimizer = optim.AdamW(
            self.semantic_tuner.parameters(),
            lr=config.get('semantic_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.box_optimizer = optim.AdamW(
            self.box_tuner.parameters(),
            lr=config.get('box_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def extract_features(self, images, texts):
        """
        Extract features from CLIP (Equations 1, 2)
        
        Args:
            images: Batch of images [B, 3, H, W]
            texts: List of text prompts
            
        Returns:
            f_I: Image features
            f_T: Text features
        """
        with torch.no_grad():
            # Equation 1: f_I = ImageEncoder(I)
            f_I = self.clip_model.encode_image(images)
            
            # Equation 2: f_T = TextEncoder(T)
            # Tokenize texts
            text_tokens = self.clip_model.tokenize(texts).to(self.device)
            f_T = self.clip_model.encode_text(text_tokens)
            
        return f_I, f_T
    
    def forward_point_branch(self, images, texts, f_I):
        """
        Branch 1: Point-Prompt (using CLIP attention maps)
        
        Returns:
            z: Segmentation mask from point-prompt
            points: Generated points
            labels: Point labels
        """
        # Generate points from CLIP attention maps
        # This is a simplified version - actual implementation needs CLIP Surgery
        attention_maps = self.get_attention_maps(images, texts, f_I)
        
        points, labels = self.attention_to_points(attention_maps, images.shape[-2:])
        
        # SAM with point prompts
        with torch.no_grad():
            z = self.sam_predict_with_points(images, points, labels)
        
        return z, points, labels
    
    def forward_box_branch(self, images, texts):
        """
        Branch 2: Box-Prompt (using Grounding DINO)
        
        Returns:
            z_hat: Segmentation mask from box-prompt
            B_init: Initial detection boxes
            sam_tokens: SAM semantic tokens
        """
        # Grounding DINO for box detection
        with torch.no_grad():
            B_init = self.grounding_dino_predict(images, texts)
            
            # SAM with box prompts
            z_hat, sam_tokens = self.sam_predict_with_boxes(images, B_init)
        
        return z_hat, B_init, sam_tokens
    
    def train_box_tuner_step(self, images, texts, B_init, B_pseudo, sam_tokens, f_I):
        """
        Train Box Tuner (separate computational graph from Semantic Tuner)
        
        CRITICAL: This gradient path is INDEPENDENT of semantic tuner
        
        Args:
            images: Input images
            texts: Text prompts
            B_init: Initial boxes from Grounding DINO
            B_pseudo: Pseudo boxes from point-2-box
            sam_tokens: SAM semantic tokens
            f_I: CLIP image features (detached)
            
        Returns:
            loss: Box tuner loss
            metrics: Training metrics
        """
        # Extract CLIP region features using RoIAlign
        clip_feature_map = self.get_clip_feature_map(images, f_I.detach())
        clip_region_features = self.box_tuner.extract_roi_features(
            clip_feature_map, B_init
        )
        
        # Forward through box tuner (Equation 11)
        fused_features = self.box_tuner(sam_tokens.detach(), clip_region_features)
        
        # Predict box adjustments
        delta_boxes = self.box_tuner.predict_box_adjustment(fused_features)
        
        # Refine boxes (Equation 13)
        B_ref = self.box_tuner.refine_boxes(B_init, delta_boxes)
        
        # Compute box loss (Equations 14, 15, 16)
        loss, metrics = self.box_loss_fn(B_ref, B_pseudo)
        
        # Backward and optimize (ONLY box tuner parameters)
        self.box_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.box_tuner.parameters(), max_norm=1.0)
        self.box_optimizer.step()
        
        return loss.item(), metrics
    
    def train_semantic_tuner_step(self, hard_examples_batch):
        """
        Train Semantic Tuner on hard examples (separate computational graph)
        
        CRITICAL: This gradient path is INDEPENDENT of box tuner
        Only trained every 100 epochs on memory bank samples
        
        Args:
            hard_examples_batch: Batch from memory bank
            
        Returns:
            loss: Semantic tuner loss
        """
        if hard_examples_batch is None:
            return 0.0
        
        # Move to device
        f_I = hard_examples_batch['f_I'].to(self.device)
        f_T = hard_examples_batch['f_T'].to(self.device)
        M_pseudo = hard_examples_batch['M_pseudo'].to(self.device)
        
        # Get adapted features through semantic tuner (Equation 9)
        # f_ST = SemanticTuner(f_I)
        # In practice, this requires hooking into CLIP layers
        # Simplified version:
        f_ST = self.semantic_tuner.get_adapted_features(self.clip_model, f_I)
        
        # Compute alignment loss (Equation 10)
        loss = self.semantic_loss_fn(f_ST, f_T, pred_mask=None, pseudo_mask=M_pseudo)
        
        # Backward and optimize (ONLY semantic tuner parameters)
        self.semantic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.semantic_tuner.parameters(), max_norm=1.0)
        self.semantic_optimizer.step()
        
        return loss.item()
    
    def training_step(self, batch):
        """
        Single training step implementing Algorithm 2
        
        Returns:
            metrics: Dictionary of training metrics
        """
        images = batch['image'].to(self.device)
        class_names = batch['class_name']
        
        metrics = defaultdict(float)
        
        # Extract features (Equations 1, 2)
        f_I, f_T = self.extract_features(images, class_names)
        
        # Forward pass through two branches
        # Branch 1: Point-Prompt
        z, points, labels = self.forward_point_branch(images, class_names, f_I)
        
        # Branch 2: Box-Prompt  
        z_hat, B_init, sam_tokens = self.forward_box_branch(images, class_names)
        
        # Hard example mining (Equations 3, 4, 5)
        L_mask = self.memory_bank.compute_mask_loss(z, z_hat)
        
        # Add to memory bank if hard example
        if L_mask.mean().item() >= self.config.get('hard_threshold', 0.5):
            self.memory_bank.add_entry(
                f_I, f_T, z, z_hat,
                iou_point=None, iou_box=None
            )
        
        metrics['mask_loss'] = L_mask.mean().item()
        
        # Point-2-box conversion (Algorithm 1)
        B_pseudo = self.point2box(points, labels)
        
        if len(B_pseudo) > 0 and len(B_init) > 0:
            # Train Box Tuner (SEPARATE computational graph)
            box_loss, box_metrics = self.train_box_tuner_step(
                images, class_names, B_init, 
                torch.tensor(B_pseudo).to(self.device),
                sam_tokens, f_I
            )
            metrics['box_loss'] = box_loss
            metrics.update({f'box_{k}': v for k, v in box_metrics.items()})
        
        # Train Semantic Tuner every 100 epochs (SEPARATE computational graph)
        if self.current_epoch % 100 == 0 and len(self.memory_bank) > 0:
            hard_batch = self.memory_bank.sample_batch(
                batch_size=self.config.get('semantic_batch_size', 32),
                prioritize_high_loss=True
            )
            semantic_loss = self.train_semantic_tuner_step(hard_batch)
            metrics['semantic_loss'] = semantic_loss
        
        self.global_step += 1
        
        return metrics
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            avg_metrics: Average metrics for the epoch
        """
        self.semantic_tuner.train()
        self.box_tuner.train()
        
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch}')
        for batch in pbar:
            metrics = self.training_step(batch)
            
            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                k: f"{v:.4f}" for k, v in metrics.items()
            })
        
        # Compute average metrics
        avg_metrics = {
            k: sum(v) / len(v) for k, v in epoch_metrics.items()
        }
        
        # Add memory bank stats
        mem_stats = self.memory_bank.get_statistics()
        avg_metrics.update({f'memory_{k}': v for k, v in mem_stats.items()})
        
        self.current_epoch += 1
        
        return avg_metrics
    
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'semantic_tuner': self.semantic_tuner.state_dict(),
            'box_tuner': self.box_tuner.state_dict(),
            'semantic_optimizer': self.semantic_optimizer.state_dict(),
            'box_optimizer': self.box_optimizer.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.semantic_tuner.load_state_dict(checkpoint['semantic_tuner'])
        self.box_tuner.load_state_dict(checkpoint['box_tuner'])
        self.semantic_optimizer.load_state_dict(checkpoint['semantic_optimizer'])
        self.box_optimizer.load_state_dict(checkpoint['box_optimizer'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {path}")
    
    # Placeholder methods (to be implemented with actual CLIP Surgery, SAM, DINO integration)
    def get_attention_maps(self, images, texts, f_I):
        """Get attention maps from CLIP - placeholder"""
        # This requires CLIP Surgery implementation
        pass
    
    def attention_to_points(self, attention_maps, image_shape):
        """Convert attention maps to points - placeholder"""
        pass
    
    def sam_predict_with_points(self, images, points, labels):
        """SAM prediction with points - placeholder"""
        pass
    
    def sam_predict_with_boxes(self, images, boxes):
        """SAM prediction with boxes - placeholder"""
        pass
    
    def grounding_dino_predict(self, images, texts):
        """Grounding DINO prediction - placeholder"""
        pass
    
    def get_clip_feature_map(self, images, f_I):
        """Get CLIP feature map for RoIAlign - placeholder"""
        pass
