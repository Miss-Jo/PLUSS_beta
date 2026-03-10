"""
Evaluation utilities for PLUSS_β
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_iou(pred, target, num_classes, ignore_index=-1):
    """
    Compute IoU for each class
    
    Args:
        pred: Predicted mask [H, W]
        target: Ground truth mask [H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        iou_per_class: IoU for each class
    """
    ious = []
    
    for cls in range(num_classes):
        # Create binary masks for this class
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Skip if class not present in target
        if not target_mask.any():
            continue
        
        # Compute intersection and union
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            iou = 0.0
        else:
            iou = intersection.float() / union.float()
        
        ious.append(iou.item())
    
    return ious


def compute_miou(pred_masks, target_masks, num_classes, ignore_index=-1):
    """
    Compute mean IoU across all samples
    
    Args:
        pred_masks: Predicted masks [B, H, W]
        target_masks: Ground truth masks [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        miou: Mean IoU
        iou_per_class: Average IoU per class
    """
    all_ious = {cls: [] for cls in range(num_classes)}
    
    for pred, target in zip(pred_masks, target_masks):
        ious = compute_iou(pred, target, num_classes, ignore_index)
        
        for cls, iou in enumerate(ious):
            all_ious[cls].append(iou)
    
    # Compute average IoU per class
    iou_per_class = {}
    for cls in range(num_classes):
        if len(all_ious[cls]) > 0:
            iou_per_class[cls] = np.mean(all_ious[cls])
    
    # Compute mean IoU
    if len(iou_per_class) > 0:
        miou = np.mean(list(iou_per_class.values()))
    else:
        miou = 0.0
    
    return miou, iou_per_class


def compute_pixel_accuracy(pred, target, ignore_index=-1):
    """
    Compute pixel accuracy
    
    Args:
        pred: Predicted mask [H, W]
        target: Ground truth mask [H, W]
        ignore_index: Index to ignore
        
    Returns:
        accuracy: Pixel accuracy
    """
    valid_mask = (target != ignore_index)
    
    if valid_mask.sum() == 0:
        return 0.0
    
    correct = ((pred == target) & valid_mask).sum()
    total = valid_mask.sum()
    
    accuracy = correct.float() / total.float()
    
    return accuracy.item()


@torch.no_grad()
def evaluate_segmentation(model, dataloader, device='cuda', num_classes=None):
    """
    Evaluate segmentation performance
    
    Args:
        model: PLUSS_β trainer or inference model
        dataloader: Validation dataloader
        device: Device to use
        num_classes: Number of classes (inferred from data if None)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.semantic_tuner.eval()
    model.box_tuner.eval()
    
    all_pred_masks = []
    all_target_masks = []
    all_pixel_accs = []
    
    # Infer num_classes from first batch if not provided
    if num_classes is None:
        first_batch = next(iter(dataloader))
        if first_batch['has_mask'].any():
            num_classes = first_batch['mask'][0].max().item() + 1
        else:
            num_classes = 50  # Default
    
    print(f"Evaluating with {num_classes} classes...")
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        images = batch['image'].to(device)
        class_names = batch['class_name']
        
        # Skip if no masks available
        if not batch['has_mask'].any():
            continue
        
        target_masks = batch['mask'].to(device)
        
        # Get predictions
        # This is a simplified version - actual implementation depends on model interface
        # For now, we'll use a placeholder
        pred_masks = torch.zeros_like(target_masks)  # Placeholder
        
        # Compute per-sample metrics
        for pred, target in zip(pred_masks, target_masks):
            if (target != -1).any():  # Has valid annotations
                all_pred_masks.append(pred.cpu())
                all_target_masks.append(target.cpu())
                
                # Pixel accuracy
                pixel_acc = compute_pixel_accuracy(pred, target)
                all_pixel_accs.append(pixel_acc)
    
    # Compute overall metrics
    if len(all_pred_masks) == 0:
        return {
            'mIoU': 0.0,
            'pixel_accuracy': 0.0,
            'num_samples': 0
        }
    
    miou, iou_per_class = compute_miou(
        all_pred_masks,
        all_target_masks,
        num_classes
    )
    
    mean_pixel_acc = np.mean(all_pixel_accs) if all_pixel_accs else 0.0
    
    metrics = {
        'mIoU': miou,
        'pixel_accuracy': mean_pixel_acc,
        'num_samples': len(all_pred_masks)
    }
    
    # Add per-class IoU for top/bottom classes
    sorted_classes = sorted(iou_per_class.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_classes) >= 3:
        metrics['top3_classes'] = sorted_classes[:3]
        metrics['bottom3_classes'] = sorted_classes[-3:]
    
    return metrics


def evaluate_on_test_set(model, test_loader, output_dir, device='cuda'):
    """
    Evaluate on test set and save predictions
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        output_dir: Directory to save predictions
        device: Device to use
    """
    import os
    from PIL import Image
    
    model.semantic_tuner.eval()
    model.box_tuner.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    pbar = tqdm(test_loader, desc='Testing')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        image_paths = batch['image_path']
        
        # Get predictions (placeholder)
        pred_masks = torch.zeros(images.shape[0], images.shape[2], images.shape[3])
        
        # Save predictions
        for i, (pred_mask, img_path) in enumerate(zip(pred_masks, image_paths)):
            # Convert mask to RGB format (R + G*256)
            pred_mask_np = pred_mask.cpu().numpy().astype(np.int32)
            
            R = pred_mask_np % 256
            G = pred_mask_np // 256
            B = np.zeros_like(R)
            
            rgb_mask = np.stack([R, G, B], axis=-1).astype(np.uint8)
            
            # Save
            filename = os.path.basename(img_path).replace('.JPEG', '.png')
            save_path = os.path.join(output_dir, filename)
            
            Image.fromarray(rgb_mask).save(save_path)
    
    print(f"Predictions saved to {output_dir}")
