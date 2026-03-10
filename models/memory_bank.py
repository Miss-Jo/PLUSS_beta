"""
Memory Bank Module for PLUSS_β
Stores "hard examples" where point-prompt and box-prompt branches disagree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


class MemoryBank:
    """
    Dynamic storage structure that maintains hard examples
    Uses mask loss to identify difficult cases where branches disagree
    """
    
    def __init__(self, capacity=1000, threshold=0.5, alpha=0.7, beta=0.3):
        """
        Args:
            capacity: Maximum number of entries (C in paper)
            threshold: Hard example threshold (σ in paper)
            alpha: Weight for IoU loss in mask loss
            beta: Weight for Dice loss in mask loss
        """
        self.capacity = capacity
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        
        # Storage using deque for efficient FIFO
        self.entries = deque(maxlen=capacity)
        
        # Track statistics
        self.total_added = 0
        self.total_rejected = 0
        
    def compute_mask_loss(self, mask1, mask2):
        """
        Compute mask loss between two segmentation masks (Equations 3, 4, 5)
        
        L_mask = α * L_iou + β * L_dice
        
        Args:
            mask1: First mask (z) [H, W] or [batch, H, W]
            mask2: Second mask (ẑ) [H, W] or [batch, H, W]
            
        Returns:
            mask_loss: Inconsistency score
        """
        # Ensure masks are float and same shape
        mask1 = mask1.float()
        mask2 = mask2.float()
        
        # Flatten masks
        if mask1.dim() == 3:
            mask1 = mask1.view(mask1.size(0), -1)
            mask2 = mask2.view(mask2.size(0), -1)
        else:
            mask1 = mask1.view(-1)
            mask2 = mask2.view(-1)
        
        # IoU Loss (Equation 3)
        intersection = (mask1 * mask2).sum(dim=-1)
        union = mask1.sum(dim=-1) + mask2.sum(dim=-1) - intersection
        iou_loss = 1 - intersection / (union + 1e-7)
        
        # Dice Loss (Equation 4)
        dice_numerator = 2 * intersection
        dice_denominator = (mask1 ** 2).sum(dim=-1) + (mask2 ** 2).sum(dim=-1)
        dice_loss = 1 - dice_numerator / (dice_denominator + 1e-7)
        
        # Combined Mask Loss (Equation 5)
        mask_loss = self.alpha * iou_loss + self.beta * dice_loss

        return mask_loss

    def generate_pseudo_label(self, mask_point, mask_box, iou_point, iou_box):
        """
        Generate pseudo-label from the more reliable branch

        Args:
            mask_point: Mask from point-prompt branch
            mask_box: Mask from box-prompt branch
            iou_point: IoU score from point-prompt
            iou_box: IoU score from box-prompt

        Returns:
            pseudo_mask: Pseudo ground truth mask
        """
        # Box prompts are generally more reliable (as stated in paper)
        # But we can also check IoU scores
        if iou_box > iou_point:
            return mask_box
        else:
            return mask_point

    def add_entry(self, image_features, text_features, mask_point, mask_box,
                  iou_point=None, iou_box=None):
        """
        Add new entries to memory bank if they're hard examples

        Handles both single samples and batches

        Entry structure (Equation 6):
        e_i = (f_I, f_T, z, ẑ, L_mask, M_pseudo)

        Args:
            image_features: Image features from CLIP (f_I) [B, ...] or [...]
            text_features: Text features from CLIP (f_T) [B, ...] or [...]
            mask_point: Mask from point-prompt branch (z) [B, H, W] or [H, W]
            mask_box: Mask from box-prompt branch (ẑ) [B, H, W] or [H, W]
            iou_point: Optional IoU score from point branch (single or batch)
            iou_box: Optional IoU score from box branch (single or batch)

        Returns:
            added: Number of entries added
            avg_loss: Average loss value
        """
        # Compute mask loss
        loss = self.compute_mask_loss(mask_point, mask_box)

        # Handle batch vs single sample
        if loss.dim() == 0:
            # Single sample
            loss = loss.unsqueeze(0)
            image_features = image_features.unsqueeze(0)
            text_features = text_features.unsqueeze(0)
            mask_point = mask_point.unsqueeze(0)
            mask_box = mask_box.unsqueeze(0)

        batch_size = loss.shape[0]
        added_count = 0

        # Process each sample in batch
        for i in range(batch_size):
            loss_i = loss[i].item()

            # Check if it's a hard example
            if loss_i >= self.threshold:
                # Get IoU scores for this sample
                iou_p = iou_point[i] if isinstance(iou_point, (list, torch.Tensor)) and len(iou_point) > i else (iou_point if iou_point is not None else 0.5)
                iou_b = iou_box[i] if isinstance(iou_box, (list, torch.Tensor)) and len(iou_box) > i else (iou_box if iou_box is not None else 0.5)

                # Generate pseudo-label for this sample
                pseudo_mask = self.generate_pseudo_label(
                    mask_point[i], mask_box[i],
                    iou_p, iou_b
                )

                # Create entry
                entry = {
                    'f_I': image_features[i].detach().cpu(),
                    'f_T': text_features[i].detach().cpu(),
                    'z': mask_point[i].detach().cpu(),
                    'z_hat': mask_box[i].detach().cpu(),
                    'L_mask': loss_i,
                    'M_pseudo': pseudo_mask.detach().cpu()
                }

                # Add to bank (FIFO if full)
                self.entries.append(entry)
                self.total_added += 1
                added_count += 1
            else:
                self.total_rejected += 1

        return added_count, loss.mean().item()
    
    def sample_batch(self, batch_size, prioritize_high_loss=True):
        """
        Sample a batch of hard examples from memory bank
        
        Args:
            batch_size: Number of samples to draw
            prioritize_high_loss: Whether to prioritize high-loss examples
            
        Returns:
            batch: Dictionary containing batched entries
        """
        if len(self.entries) == 0:
            return None
        
        # Sample size
        sample_size = min(batch_size, len(self.entries))
        
        if prioritize_high_loss:
            # Sort by loss and sample from top entries
            sorted_entries = sorted(self.entries, 
                                   key=lambda x: x['L_mask'], 
                                   reverse=True)
            sampled_entries = sorted_entries[:sample_size]
        else:
            # Random sampling
            indices = np.random.choice(len(self.entries), 
                                      size=sample_size, 
                                      replace=False)
            sampled_entries = [self.entries[i] for i in indices]
        
        # Batch entries
        batch = {
            'f_I': torch.stack([e['f_I'] for e in sampled_entries]),
            'f_T': torch.stack([e['f_T'] for e in sampled_entries]),
            'z': torch.stack([e['z'] for e in sampled_entries]),
            'z_hat': torch.stack([e['z_hat'] for e in sampled_entries]),
            'L_mask': torch.tensor([e['L_mask'] for e in sampled_entries]),
            'M_pseudo': torch.stack([e['M_pseudo'] for e in sampled_entries])
        }
        
        return batch
    
    def get_statistics(self):
        """
        Get memory bank statistics
        
        Returns:
            stats: Dictionary of statistics
        """
        if len(self.entries) == 0:
            return {
                'size': 0,
                'total_added': self.total_added,
                'total_rejected': self.total_rejected,
                'avg_loss': 0.0,
                'max_loss': 0.0,
                'min_loss': 0.0
            }
        
        losses = [e['L_mask'] for e in self.entries]
        
        return {
            'size': len(self.entries),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'total_rejected': self.total_rejected,
            'avg_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'acceptance_rate': self.total_added / (self.total_added + self.total_rejected + 1e-7)
        }
    
    def clear(self):
        """Clear all entries from memory bank"""
        self.entries.clear()
        self.total_added = 0
        self.total_rejected = 0
    
    def __len__(self):
        return len(self.entries)
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"MemoryBank(size={stats['size']}/{stats['capacity']}, " \
               f"avg_loss={stats['avg_loss']:.4f}, " \
               f"acceptance_rate={stats['acceptance_rate']:.2%})"
