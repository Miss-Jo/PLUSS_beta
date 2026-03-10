"""
Semantic Tuner Module for PLUSS_β
Visual prompt tuning of the image encoder while keeping encoders frozen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTuner(nn.Module):
    """
    Semantic tuner inserts learnable prompts into each encoder layer
    while keeping original CLIP parameters frozen.
    
    This module learns to resolve ambiguities causing inconsistencies
    between point-prompt and box-prompt branches.
    """
    
    def __init__(self, 
                 num_layers=12,
                 embed_dim=512,
                 num_prompts=16,
                 dropout=0.1):
        """
        Args:
            num_layers: Number of transformer layers in CLIP image encoder
            embed_dim: Embedding dimension
            num_prompts: Number of learnable prompt tokens per layer
            dropout: Dropout rate for prompts
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        
        # Learnable prompts for each layer P_i = {p_i^k | k=1..m}
        # Shape: [num_layers, num_prompts, embed_dim]
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompts, embed_dim))
            for _ in range(num_layers)
        ])
        
        # Initialize prompts with small values
        for prompt in self.prompts:
            nn.init.normal_(prompt, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, image_features, layer_idx=None):
        """
        Insert learnable prompts into image features
        
        Args:
            image_features: Image features from CLIP [batch_size, num_patches, embed_dim]
            layer_idx: Which layer to get prompts for (if None, return all layers)
            
        Returns:
            Augmented features with prompts inserted
        """
        batch_size = image_features.shape[0]
        
        if layer_idx is not None:
            # Get prompts for specific layer
            prompts = self.prompts[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            prompts = self.dropout(prompts)
            # Concatenate prompts with image features: [CLS, Prompts, Patches]
            augmented_features = torch.cat([
                image_features[:, :1, :],  # CLS token
                prompts,                    # Learnable prompts
                image_features[:, 1:, :]   # Image patches
            ], dim=1)
            return augmented_features
        else:
            # Return all layer prompts for integration
            return [self.prompts[i] for i in range(self.num_layers)]

    def get_adapted_features(self, clip_model, input_data):
        """
        Extract adapted visual features using the semantic tuner

        This method supports two input types:
        1. Raw images [B, 3, H, W] - will encode through CLIP
        2. Pre-computed CLIP features [B, 1025, 512] - will use directly

        The semantic tuner adds learnable prompts to adapt features
        without modifying the frozen CLIP encoder.

        Args:
            clip_model: Frozen CLIP model
            input_data: Either raw images [B, 3, H, W] or pre-computed features [B, 1025, 512]

        Returns:
            f_ST: Adapted semantic features [B, 1025, 512]
        """
        # Determine input type based on tensor dimensions
        if input_data.dim() == 3:
            # Pre-computed CLIP features [B, 1025, 512]
            # This is the case when called from train_semantic_tuner_step
            # with features from memory bank
            image_features = input_data

        elif input_data.dim() == 4 and input_data.shape[1] == 3:
            # Raw images [B, 3, H, W]
            # Encode through frozen CLIP
            with torch.no_grad():
                image_features = clip_model.encode_image(input_data)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Invalid input shape: {input_data.shape}. "
                f"Expected [B, 3, H, W] for images or [B, 1025, 512] for features"
            )

        # Shape: [B, 1025, 512] where 1025 = 1(CLS) + 32*32(patches)
        batch_size = image_features.shape[0]

        # Apply semantic tuning adaptation
        # In full implementation, prompts would be inserted into each CLIP layer
        # For simplified version, we apply prompt-based adaptation to the output features

        # Method 1: Simple weighted combination with learnable prompts
        # Extract CLS token and patch tokens
        cls_token = image_features[:, 0:1, :]  # [B, 1, 512]
        patch_tokens = image_features[:, 1:, :]  # [B, 1024, 512]

        # For simplicity in this implementation, we use the last layer prompts
        # In full implementation, prompts from all layers would be used
        last_layer_prompts = self.prompts[-1]  # [num_prompts, 512]

        # Expand prompts for batch
        prompts = last_layer_prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_prompts, 512]
        prompts = self.dropout(prompts)

        # Compute attention between patch tokens and prompts
        # This allows prompts to modulate the features
        attention_scores = torch.matmul(patch_tokens, prompts.transpose(1, 2))  # [B, 1024, num_prompts]
        attention_scores = F.softmax(attention_scores / (self.embed_dim ** 0.5), dim=-1)

        # Weighted combination of prompts
        prompt_features = torch.matmul(attention_scores, prompts)  # [B, 1024, 512]

        # Residual connection: add prompt features to original patch tokens
        adapted_patch_tokens = patch_tokens + 0.1 * prompt_features  # Small residual weight

        # Recombine CLS and adapted patches
        adapted_features = torch.cat([cls_token, adapted_patch_tokens], dim=1)  # [B, 1025, 512]

        # Normalize
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

        return adapted_features


class SemanticTunerLoss(nn.Module):
    """
    Loss function for training the semantic tuner
    Combines alignment loss with optional segmentation loss
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

    def alignment_loss(self, visual_features, text_features):
        """
        InfoNCE-style alignment loss (Equation 10 in paper)

        L_align = -log(exp(cos(f_ST(i), f_T(i))/τ) / Σ_j exp(cos(f_ST(i), f_T(j))/τ))

        Args:
            visual_features: Adapted visual features [batch_size, embed_dim]
            text_features: Text features from CLIP [batch_size, embed_dim]

        Returns:
            Alignment loss value
        """
        # Normalize features
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute cosine similarity
        logits = torch.matmul(visual_features, text_features.t()) / self.temperature

        # Cross-entropy loss
        batch_size = visual_features.shape[0]
        labels = torch.arange(batch_size, device=visual_features.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def segmentation_loss(self, pred_mask, pseudo_mask):
        """
        Segmentation loss against pseudo-labels from memory bank

        Args:
            pred_mask: Predicted segmentation mask
            pseudo_mask: Pseudo-label mask from memory bank

        Returns:
            Segmentation loss
        """
        # Dice loss
        intersection = (pred_mask * pseudo_mask).sum()
        union = pred_mask.sum() + pseudo_mask.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, pseudo_mask)

        return dice_loss + bce_loss

    def forward(self, visual_features, text_features, pred_mask=None, pseudo_mask=None):
        """
        Total loss for semantic tuner

        Args:
            visual_features: Adapted visual features [B, 1025, 512] or [B, 512]
            text_features: Text features from CLIP [B, 512] or [num_classes, 512]
            pred_mask: Optional predicted mask
            pseudo_mask: Optional pseudo-label mask

        Returns:
            Total loss
        """
        # Extract CLS tokens if input is full feature map
        if visual_features.dim() == 3:
            # visual_features: [B, 1025, 512]
            # Extract CLS token (first token)
            visual_cls = visual_features[:, 0, :]  # [B, 512]
        else:
            # visual_features: [B, 512]
            visual_cls = visual_features

        # Handle text features
        if text_features.dim() == 3:
            # text_features: [B, num_tokens, 512]
            # Extract CLS token or average
            text_cls = text_features[:, 0, :]  # [B, 512]
        elif text_features.dim() == 2:
            # text_features: [B, 512] or [num_classes, 512]
            text_cls = text_features
        else:
            raise ValueError(f"Unexpected text_features shape: {text_features.shape}")

        # Compute alignment loss with CLS tokens
        loss = self.alignment_loss(visual_cls, text_cls)
        
        if pred_mask is not None and pseudo_mask is not None:
            loss = loss + self.segmentation_loss(pred_mask, pseudo_mask)
        
        return loss
