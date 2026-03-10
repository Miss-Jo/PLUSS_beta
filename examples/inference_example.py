"""
Simple inference example for PLUSS_β
"""

import torch
from PIL import Image
import argparse

from pluss_beta import load_trained_model
import clip
from segment_anything import sam_model_registry
from groundingdino.util.inference import load_model as load_grounding_dino


def main():
    parser = argparse.ArgumentParser(description='PLUSS_β Inference Example')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--text', type=str, required=True,
                       help='Text prompt (e.g., "goldfish")')
    parser.add_argument('--output', type=str, default='result.png',
                       help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    print("Loading models...")
    
    # Load foundation models
    clip_model, _ = clip.load("ViT-B/16", device=args.device)
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    grounding_dino = load_grounding_dino(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "groundingdino_swint_ogc.pth"
    )
    
    # Configuration
    config = {
        'num_layers': 12,
        'embed_dim': 512,
        'feature_dim': 512,
        'num_prompts': 16,
        'num_heads': 8,
        'hidden_dim': 2048
    }
    
    # Load trained model
    model = load_trained_model(
        checkpoint_path=args.checkpoint,
        clip_model=clip_model,
        sam_model=sam,
        grounding_dino=grounding_dino,
        config=config,
        device=args.device
    )
    
    print(f"Loaded model from {args.checkpoint}")
    
    # Load image
    image = Image.open(args.image).convert('RGB')
    print(f"Processing image: {args.image}")
    print(f"Text prompt: {args.text}")
    
    # Predict
    mask, intermediates = model.predict(
        image, 
        args.text,
        return_intermediates=True
    )
    
    print(f"Prediction complete!")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Num detections: {len(intermediates['initial_boxes'])}")
    
    # Visualize
    model.visualize_result(image, mask, save_path=args.output)
    print(f"Result saved to {args.output}")


if __name__ == '__main__':
    main()
