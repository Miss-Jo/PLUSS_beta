"""
Box Tuner Module for PLUSS_β
Refines bounding boxes by fusing CLIP semantic features with SAM spatial features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class BoxTuner(nn.Module):
    """
    Box tuner acts as a decoder module that pools region-level features 
    from the image-level semantic map.
    
    Uses cross-attention where SAM tokens serve as queries to extract 
    relevant features from CLIP region features.
    """
    
    def __init__(self, 
                 feature_dim=512,
                 num_heads=8,
                 hidden_dim=2048,
                 dropout=0.1):
        """
        Args:
            feature_dim: Feature dimension (C in paper)
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Linear projection for SAM tokens to match CLIP feature space
        self.sam_projection = nn.Linear(256, feature_dim)  # SAM uses 256-dim tokens
        
        # Cross-attention: SAM tokens (Q) attend to CLIP features (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # MLP to predict box adjustment offsets
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Output: (Δx, Δy, Δw, Δh)
        )
        
        # Scaling factors for box refinement (learnable)
        self.alpha_x = nn.Parameter(torch.ones(1))
        self.alpha_y = nn.Parameter(torch.ones(1))
        self.alpha_w = nn.Parameter(torch.ones(1))
        self.alpha_h = nn.Parameter(torch.ones(1))
        
    def extract_roi_features(self, clip_feature_map, boxes, output_size=7):
        """
        Extract region-level features using RoIAlign
        
        Args:
            clip_feature_map: CLIP feature map [batch, C, H, W]
            boxes: List of bounding boxes tensors for each image in batch
                   Each element: [num_boxes_i, 4] in (x, y, w, h) or (x1, y1, x2, y2) format
            output_size: RoI output size

        Returns:
            Region features [total_num_boxes, C]
        """
        device = clip_feature_map.device

        # Handle list of boxes (one tensor per image in batch)
        if isinstance(boxes, list):
            all_boxes_xyxy = []
            batch_indices = []

            for batch_idx, box_tensor in enumerate(boxes):
                if len(box_tensor) == 0:
                    continue

                # Ensure boxes are on correct device
                if box_tensor.device != device:
                    box_tensor = box_tensor.to(device)

                # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2) if needed
                # Check if already in xyxy format (x2 > x1 + some threshold)
                if box_tensor.shape[1] == 4:
                    # Assume (x, y, w, h) if w > 0 and h > 0
                    if (box_tensor[:, 2] > 0).all() and (box_tensor[:, 3] > 0).all():
                        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                        boxes_xyxy = torch.zeros_like(box_tensor)
                        boxes_xyxy[:, 0] = box_tensor[:, 0]  # x1 = x
                        boxes_xyxy[:, 1] = box_tensor[:, 1]  # y1 = y
                        boxes_xyxy[:, 2] = box_tensor[:, 0] + box_tensor[:, 2]  # x2 = x + w
                        boxes_xyxy[:, 3] = box_tensor[:, 1] + box_tensor[:, 3]  # y2 = y + h
                    else:
                        # Already in (x1, y1, x2, y2) format
                        boxes_xyxy = box_tensor

                all_boxes_xyxy.append(boxes_xyxy)
                batch_indices.extend([batch_idx] * len(boxes_xyxy))

            if len(all_boxes_xyxy) == 0:
                # No boxes, return empty features
                return torch.zeros(0, clip_feature_map.shape[1], device=device)

            # Concatenate all boxes
            boxes_xyxy = torch.cat(all_boxes_xyxy, dim=0)  # [total_boxes, 4]

            # Create batch indices on correct device
            batch_indices = torch.tensor(
                batch_indices,
                device=device,  # ← Fix: Use same device as feature map
                dtype=torch.float32
            )[:, None]

        else:
            # Single tensor (backward compatibility)
            if boxes.device != device:
                boxes = boxes.to(device)

            # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0]  # x1 = x
            boxes_xyxy[:, 1] = boxes[:, 1]  # y1 = y
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h

            # Assume all boxes from batch index 0
            batch_indices = torch.zeros(
                (boxes.shape[0], 1),
                device=device,  # ← Fix: Use same device
                dtype=torch.float32
            )

        # RoIAlign expects boxes in format [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_indices, boxes_xyxy], dim=1)

        # Apply RoIAlign
        roi_features = roi_align(
            clip_feature_map,
            rois,
            output_size=(output_size, output_size),
            spatial_scale=1.0,  # Assume feature map and boxes are in same scale
            aligned=True
        )
        
        # Global average pooling: [num_boxes, C, H, W] -> [num_boxes, C]
        roi_features = roi_features.mean(dim=[2, 3])
        
        return roi_features
    
    def forward(self, sam_tokens, clip_region_features):
        """
        Fuse SAM spatial tokens with CLIP semantic features
        
        Args:
            sam_tokens: Region-level semantic tokens from SAM [num_boxes, sam_dim]
            clip_region_features: Region features from CLIP [num_boxes, feature_dim]
            
        Returns:
            fused_features: Fused features [num_boxes, feature_dim]
        """
        # Project SAM tokens to same dimension as CLIP features
        sam_features = self.sam_projection(sam_tokens)  # [num_boxes, feature_dim]
        
        # Reshape for attention: [num_boxes, 1, feature_dim]
        # query = sam_features.unsqueeze(1)
        query = sam_features
        key = clip_region_features.unsqueeze(1)
        value = clip_region_features.unsqueeze(1)
        
        # Cross-attention: SAM queries CLIP features (Equation 11)
        # F_fused = softmax((F_S * K_F^T) / sqrt(C)) * V_F
        # print("query:",query.shape)#(4,1,512)
        # print("key:",key.shape)#(4,1,512)
        # print("value:",value.shape)#(4,1,512)
        attn_output, attn_weights = self.cross_attention(
            query, key, value
        )
        
        # Add & Norm
        fused_features = self.norm1(query + attn_output)
        
        # Remove sequence dimension
        fused_features = fused_features.squeeze(1)  # [num_boxes, feature_dim]
        
        return fused_features
    
    def predict_box_adjustment(self, fused_features):
        """
        Predict box adjustment offsets from fused features
        
        Args:
            fused_features: Fused features [num_boxes, feature_dim]
            
        Returns:
            delta_boxes: Box adjustments [num_boxes, 4] (Δx, Δy, Δw, Δh)
        """
        delta_boxes = self.mlp(fused_features)
        return delta_boxes
    
    def refine_boxes(self, initial_boxes, delta_boxes):
        """
        Apply box refinement (Equation 13)
        
        B_ref = (x + α_x*Δx, y + α_y*Δy, w*exp(α_w*Δw), h*exp(α_h*Δh))
        
        Args:
            initial_boxes: Initial boxes [num_boxes, 4] in (x, y, w, h) format
            delta_boxes: Predicted adjustments [num_boxes, 4]
            
        Returns:
            refined_boxes: Refined boxes [num_boxes, 4]
        """
        x, y, w, h = initial_boxes.unbind(dim=-1)
        dx, dy, dw, dh = delta_boxes.unbind(dim=-1)
        
        # Apply refinement with learnable scaling factors
        refined_x = x + self.alpha_x * dx
        refined_y = y + self.alpha_y * dy
        refined_w = w * torch.exp(self.alpha_w * dw)
        refined_h = h * torch.exp(self.alpha_h * dh)
        
        refined_boxes = torch.stack([refined_x, refined_y, refined_w, refined_h], dim=-1)
        
        return refined_boxes


class BoxTunerLoss(nn.Module):
    """
    Loss function for Box Tuner training
    Combines L1 loss for coordinate regression and GIoU loss for shape alignment
    """
    
    def __init__(self, lambda_l1=0.5, lambda_giou=1.0,image_size=512):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.image_size = float(image_size)

    def normalize_boxes(self, boxes):
        """
        核心优化：将boxes从像素坐标归一化到[0,1]

        Args:
            boxes: [N, 4] 像素坐标 (x, y, w, h)
        Returns:
            normalized_boxes: [N, 4] 归一化到[0,1]
        """
        normalized = boxes.clone()
        normalized[:, [0, 2]] = normalized[:, [0, 2]] / self.image_size  # x, w
        normalized[:, [1, 3]] = normalized[:, [1, 3]] / self.image_size  # y, h
        return normalized
        
    def l1_loss(self, pred_boxes, target_boxes):
        """
        L1 loss for coordinate regression (Equation 14)
        
        Args:
            pred_boxes: Predicted boxes [num_boxes, 4]
            target_boxes: Target boxes [num_boxes, 4]
            
        Returns:
            L1 loss
        """
        # print("pred_boxes, target_boxes: ", pred_boxes.shape, target_boxes.shape)
        pred_norm = self.normalize_boxes(pred_boxes)
        target_norm = self.normalize_boxes(target_boxes)
        return F.l1_loss(pred_norm , target_norm, reduction='mean')
    
    def giou_loss(self, pred_boxes, target_boxes):
        """
        Generalized IoU loss (Equation 15)
        
        Args:
            pred_boxes: Predicted boxes [num_boxes, 4] in (x, y, w, h) format
            target_boxes: Target boxes [num_boxes, 4]
            
        Returns:
            GIoU loss
        """
        # Convert to (x1, y1, x2, y2) format
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]
        
        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2]
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3]
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        
        # GIoU loss = 1 - GIoU
        loss = 1 - giou
        
        return loss.mean()
    
    def forward(self, pred_boxes, target_boxes):
        """
        Combined box loss (Equation 16)
        
        L_box = λ_L1 * L_L1 + λ_GIoU * L_GIoU
        
        Args:
            pred_boxes: Predicted refined boxes
            target_boxes: Pseudo-label boxes from point-2-box
            
        Returns:
            Total box loss
        """
        l1 = self.l1_loss(pred_boxes, target_boxes)
        giou = self.giou_loss(pred_boxes, target_boxes)
        
        total_loss = self.lambda_l1 * l1 + self.lambda_giou * giou
        
        return total_loss, {'l1': l1.item(), 'giou': giou.item()}
