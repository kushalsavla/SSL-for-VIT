#!/usr/bin/env python3
"""
Qualitative Analysis Script for SSL ViT Project
===============================================

This script performs qualitative analysis by classifying individual images
using our trained models (supervised ViT and iBOT fine-tuned model).

Usage:
    python qualitative_analysis.py --image_path path/to/image.jpg --model_type vit
    python qualitative_analysis.py --image_path path/to/image.jpg --model_type ibot
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
# Import vision transformer from external DINO v2 repository
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))
from models.vision_transformer import vit_small

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class FineTunedDinoModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(384, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_vit_model(model_path, device):
    print(f"Loading fine-tuned DINO model from {model_path}")
    backbone = vit_small(img_size=32, patch_size=4)
    model = FineTunedDinoModel(backbone, num_classes=10)
    model.to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded full model (with classifier) from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded raw state_dict into full model.")
    else:
        print(f"⚠️ Model path {model_path} not found, using randomly initialized weights")

    model.eval()
    return model



def load_ibot_model(model_path):
    """Load the iBOT fine-tuned model."""
    import timm
    
    print(f"Loading iBOT model from {model_path}")
    
    try:
        # Load the fine-tuned model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check if this is a FineTunedViT model (has backbone and classifier)
        if any(key.startswith('backbone.') for key in state_dict.keys()):
            print("✅ Detected FineTunedViT model structure")
            
            # Create backbone
            try:
                backbone = timm.create_model('vit_small_patch16_224', 
                                           patch_size=4, 
                                           num_classes=0, 
                                           img_size=32)
                print("✅ Created ViT backbone")
            except Exception as e:
                print(f"❌ Failed to create ViT backbone: {e}")
                return None
            
            
            
            model = FineTunedDinoModel(backbone, num_classes=10)
            
            # Load state dict
            model.load_state_dict(state_dict)
            print("✅ Loaded FineTunedViT model successfully")
            
        else:
            # Fallback to SimpleCNN
            print("⚠️ Could not load iBOT model: Using fallback SimpleCNN model")
            
    except Exception as e:
        print(f"⚠️ Could not load iBOT model: {e}")
        print("Using fallback SimpleCNN model")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("✅ iBOT model loaded successfully")
    return model

def preprocess_image(image_path, target_size=32):
    """Preprocess image for model input."""
    # Load image
    if image_path.endswith('.npy'):
        # Handle numpy array files
        image_array = np.load(image_path)
        image = Image.fromarray(image_array).convert('RGB')
    else:
        # Handle regular image files
        image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array first, then to tensor
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)  # HWC to CHW
    
    # Normalize
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    image_array = (image_array - mean) / std
    
    # Convert to tensor
    tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return tensor, image

def classify_image(model, image_tensor, model_type="vit"):
    """Classify an image using the given model."""
    # Move input tensor to same device as model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get predictions
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5)
        
        # Convert to numpy
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        return top_probs, top_indices

def visualize_results(image, top_probs, top_indices, model_type, image_path):
    """Visualize classification results."""
    try:
        # Create figure with explicit backend
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Display predictions - use simple bar plot to avoid recursion
        y_pos = np.arange(len(top_indices))
        bars = ax2.barh(y_pos, top_probs)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([CIFAR10_CLASSES[i] for i in top_indices])
        ax2.set_xlabel('Probability')
        ax2.set_title(f'Top 5 Predictions ({model_type.upper()} Model)')
        ax2.invert_yaxis()
        
        # Add probability values on bars
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        
        # Save plot
        output_path = f'qualitative_results_{model_type}_{os.path.splitext(os.path.basename(image_path))[0]}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results saved to: {output_path}")
        
        plt.close()  # Close the figure to free memory
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
        print("📊 Classification results still available above")

def print_results(top_probs, top_indices, model_type):
    """Print classification results."""
    print(f"\n🔍 Classification Results ({model_type.upper()} Model)")
    print("=" * 50)
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_name = CIFAR10_CLASSES[idx]
        print(f"{i+1}. {class_name:12} - {prob:.3f} ({prob*100:.1f}%)")
    
    # Highlight top prediction
    top_class = CIFAR10_CLASSES[top_indices[0]]
    top_prob = top_probs[0]
    print(f"\n🎯 Predicted Class: {top_class} (Confidence: {top_prob:.3f})")

def main():
    parser = argparse.ArgumentParser(description='Qualitative Analysis for SSL ViT Models')
    parser.add_argument('--image_path', type=str, default='../final_evaluation/test_airplane.png',
                       help='Path to the image to classify')
    parser.add_argument('--model_type', type=str, choices=['vit', 'ibot', 'dino'], default='dino',
                       help='Type of model to use (vit, ibot, or dino)')
    parser.add_argument('--vit_model_path', type=str, default='models/vit/best_vit_small_model.pth',
                       help='Path to ViT model checkpoint')
    parser.add_argument('--ibot_model_path', type=str, default='models/ibot/best_fine_tuned_model.pth',
                       help='Path to iBOT model checkpoint')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting results')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        return
    
    # Check if model path exists
    if args.model_type == 'dino':
        model_path = args.vit_model_path
    elif args.model_type == 'vit':
        model_path = args.vit_model_path
    else:  # ibot
        model_path = args.ibot_model_path
        
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return
    
    print(f"🚀 Starting Qualitative Analysis")
    print(f"📸 Image: {args.image_path}")
    print(f"🤖 Model: {args.model_type.upper()}")
    print(f"📁 Model Path: {model_path}")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Load model
    if args.model_type == 'dino':
        model = load_vit_model(model_path, device)
    elif args.model_type == 'vit':
        model = load_vit_model(model_path, device)
    else:  # ibot
        model = load_ibot_model(model_path)
    
    # Preprocess image
    print("🔄 Preprocessing image...")
    image_tensor, original_image = preprocess_image(args.image_path)
    
    # Classify image
    print("🔍 Classifying image...")
    top_probs, top_indices = classify_image(model, image_tensor, args.model_type)
    
    # Print results
    print_results(top_probs, top_indices, args.model_type)
    
    # Visualize results
    if not args.no_plot:
        print("📊 Creating visualization...")
        visualize_results(original_image, top_probs, top_indices, args.model_type, args.image_path)
    
    print("\n✅ Qualitative analysis completed!")

if __name__ == '__main__':
    main() 