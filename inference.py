"""
PLUSS_β Inference Pipeline
Implements Algorithm 3 from the paper
Uses ONLY the box-prompt branch for efficiency
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class PLUSSBetaInference:
    """
    PLUSS_β Inference using only box-prompt branch
    Maintains computational advantages while using trained tuners
    """
    
    def __init__(self,
                 clip_model,
                 sam_model,
                 grounding_dino,
                 semantic_tuner,
                 box_tuner,
                 device='cuda'):
        """
        Args:
            clip_model: Frozen CLIP model
            sam_model: Frozen SAM model  
            grounding_dino: Grounding DINO model
            semantic_tuner: Trained semantic tuner
            box_tuner: Trained box tuner
            device: Device to use
        """
        self.device = device
        
        # Load models
        self.clip_model = clip_model.to(device).eval()
        self.sam_model = sam_model.to(device).eval()
        self.grounding_dino = grounding_dino.to(device).eval()
        self.semantic_tuner = semantic_tuner.to(device).eval()
        self.box_tuner = box_tuner.to(device).eval()
        
    @torch.no_grad()
    def predict(self, image, text_prompt, return_intermediates=False):
        """
        Inference on a single image (Algorithm 3)
        
        Args:
            image: Input image [3, H, W] tensor or PIL Image
            text_prompt: Text description of target object
            return_intermediates: Return intermediate results
            
        Returns:
            M_final: Final segmentation mask [H, W]
            intermediates: Optional intermediate results
        """
        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            image = self.preprocess_image(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        image = image.to(self.device)
        
        # Step 1: Feature extraction with semantic tuner
        f_I = self.clip_model.encode_image(image)
        
        # Apply semantic tuner for adapted visual features
        f_ST = self.semantic_tuner.get_adapted_features(self.clip_model, image)
        
        # Extract text features
        text_tokens = self.clip_model.tokenize([text_prompt]).to(self.device)
        f_T = self.clip_model.encode_text(text_tokens)
        
        # Step 2: Enhanced detection with box tuner
        # Get initial boxes from Grounding DINO
        B_init = self.grounding_dino_predict(image, text_prompt)
        
        if len(B_init) == 0:
            # No detections
            H, W = image.shape[2], image.shape[3]
            return torch.zeros(H, W, device=self.device), {}
        
        # Refine boxes for each detection
        B_refined_list = []
        
        for b in B_init:
            # Extract SAM semantic tokens for this box
            S_b = self.extract_sam_tokens(image, b)
            
            # Extract CLIP region features using RoIAlign
            clip_feature_map = self.get_clip_feature_map(image, f_ST)
            F_region_b = self.box_tuner.extract_roi_features(
                clip_feature_map, 
                b.unsqueeze(0)
            )
            
            # Fuse features through box tuner
            F_fused_b = self.box_tuner(S_b, F_region_b)
            
            # Predict adjustment
            delta_b = self.box_tuner.predict_box_adjustment(F_fused_b)
            
            # Refine box
            b_refined = self.box_tuner.refine_boxes(
                b.unsqueeze(0), 
                delta_b
            )
            
            B_refined_list.append(b_refined.squeeze(0))
        
        B_refined = torch.stack(B_refined_list)
        
        # Step 3: Final segmentation using refined boxes
        M_final = self.sam_predict_with_boxes(image, B_refined)
        
        if return_intermediates:
            intermediates = {
                'initial_boxes': B_init,
                'refined_boxes': B_refined,
                'image_features': f_I,
                'adapted_features': f_ST,
                'text_features': f_T
            }
            return M_final, intermediates
        
        return M_final
    
    def batch_predict(self, images, text_prompts):
        """
        Batch inference
        
        Args:
            images: Batch of images [B, 3, H, W]
            text_prompts: List of text prompts (length B)
            
        Returns:
            masks: Batch of segmentation masks [B, H, W]
        """
        masks = []
        
        for i, (image, text) in enumerate(zip(images, text_prompts)):
            mask = self.predict(image, text)
            masks.append(mask)
        
        return torch.stack(masks)
    
    def preprocess_image(self, image):
        """
        Preprocess PIL image to tensor
        
        Args:
            image: PIL Image
            
        Returns:
            tensor: Preprocessed tensor [3, H, W]
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        return transform(image)
    
    def visualize_result(self, image, mask, save_path=None):
        """
        Visualize segmentation result
        
        Args:
            image: Original image
            mask: Segmentation mask
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            image = image * std + mean
            image = image.permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1)
        
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.5, cmap='tab20')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    # Placeholder methods (same as trainer)
    def grounding_dino_predict(self, images, text):
        """Grounding DINO prediction - placeholder"""
        # Returns tensor of boxes [num_boxes, 4] in (x, y, w, h) format
        pass
    
    def extract_sam_tokens(self, image, box):
        """Extract SAM semantic tokens for a box - placeholder"""
        # Returns tensor [1, sam_dim]
        pass
    
    def get_clip_feature_map(self, image, features):
        """Get CLIP feature map - placeholder"""
        # Returns tensor [1, C, H, W]
        pass
    
    def sam_predict_with_boxes(self, image, boxes):
        """SAM prediction with boxes - placeholder"""
        # Returns mask [H, W]
        pass


def load_trained_model(checkpoint_path, clip_model, sam_model, grounding_dino, config, device='cuda'):
    """
    Load trained PLUSS_β model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        clip_model: CLIP model
        sam_model: SAM model
        grounding_dino: Grounding DINO model
        config: Model configuration
        device: Device to use
        
    Returns:
        inference_model: Ready-to-use inference model
    """
    from pluss_beta.models.semantic_tuner import SemanticTuner
    from pluss_beta.models.box_tuner import BoxTuner
    
    # Initialize tuners
    semantic_tuner = SemanticTuner(
        num_layers=config.get('num_layers', 12),
        embed_dim=config.get('embed_dim', 512),
        num_prompts=config.get('num_prompts', 16)
    )
    
    box_tuner = BoxTuner(
        feature_dim=config.get('feature_dim', 512),
        num_heads=config.get('num_heads', 8),
        hidden_dim=config.get('hidden_dim', 2048)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    semantic_tuner.load_state_dict(checkpoint['semantic_tuner'])
    box_tuner.load_state_dict(checkpoint['box_tuner'])
    
    # Create inference model
    inference_model = PLUSSBetaInference(
        clip_model=clip_model,
        sam_model=sam_model,
        grounding_dino=grounding_dino,
        semantic_tuner=semantic_tuner,
        box_tuner=box_tuner,
        device=device
    )
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return inference_model
