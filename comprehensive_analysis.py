#!/usr/bin/env python3
"""
Comprehensive Analysis Script for SSL ViT Project
=================================================

This script performs qualitative analysis on all three models (ViT, DINO, iBOT)
and creates a comprehensive comparison visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import sys

# Add DINO v2 path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'dino', 'dinov2'))
from models.vision_transformer import vit_small

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class FineTunedDinoModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(384, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_dino_model(model_path, device):
    """Load DINO model with correct structure."""
    print(f"Loading DINO model from {model_path}")
    backbone = vit_small(img_size=32, patch_size=4)
    model = FineTunedDinoModel(backbone, num_classes=10)
    model.to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded DINO model with classifier from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded DINO model from raw state_dict.")
    else:
        print(f"⚠️ DINO model path {model_path} not found")
        return None
    
    model.eval()
    return model

def load_vit_model(model_path, device):
    """Load ViT model with correct structure."""
    print(f"Loading ViT model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ ViT model path {model_path} not found")
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create a simple ViT model for classification - use timm instead of DINO's vit_small
        import timm
        model = timm.create_model('vit_small_patch16_224', 
                                patch_size=4, 
                                num_classes=10, 
                                img_size=32)
        model.to(device)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        print("✅ Loaded ViT model successfully.")
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading ViT model: {e}")
        return None

def load_ibot_model(model_path, device):
    """Load iBOT model."""
    print(f"Loading iBOT model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ iBOT model path {model_path} not found")
        return None
    
    try:
        import timm
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create backbone
        backbone = timm.create_model('vit_small_patch16_224', 
                                   patch_size=4, 
                                   num_classes=0, 
                                   img_size=32)
        
        # Create full model with classifier
        class FineTunedViT(nn.Module):
            def __init__(self, backbone, num_classes=10):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(384, num_classes)
            
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        model = FineTunedViT(backbone, num_classes=10)
        model.to(device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        print("✅ Loaded iBOT model successfully.")
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading iBOT model: {e}")
        return None

def load_mae_model(model_path, device):
    """Load MAE model."""
    print(f"Loading MAE model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ MAE model path {model_path} not found")
        return None
    
    try:
        # Add MAE scripts to path
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'mae'))
        from models_vit import vit_tiny_patch4_32
        
        # Create model
        model = vit_tiny_patch4_32(num_classes=10, global_pool=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove classification head if shape mismatches
        model_state = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in state_dict and k in model_state and state_dict[k].shape != model_state[k].shape:
                print(f"Removing key {k} from checkpoint due to shape mismatch")
                del state_dict[k]
        
        load_msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {load_msg}")
        print(f"✅ MAE model loaded successfully.")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading MAE model: {e}")
        return None

def preprocess_image(image_path, target_size=32):
    """Preprocess image for model input."""
    if image_path.endswith('.npy'):
        image_array = np.load(image_path)
        image = Image.fromarray(image_array).convert('RGB')
    else:
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

def classify_image(model, image_tensor, device, model_type="unknown"):
    """Classify image and return top 5 predictions."""
    with torch.no_grad():
        model.eval()
        image_tensor = image_tensor.to(device)
        
        # Handle different model types
        if model_type == "mae":
            try:
                outputs = model(image_tensor)
            except TypeError:
                # Try without attn_mask for MAE
                try:
                    outputs = model.forward_features(image_tensor)
                    outputs = model.head(outputs)
                except:
                    print("❌ Failed to run MAE model")
                    return None, None
        else:
            outputs = model(image_tensor)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Handle tuple outputs
        
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5, dim=1)
        
        return top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()

def create_comprehensive_visualization(image, results, image_path):
    """Create comprehensive visualization comparing all models."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Display original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f'Input Image: {os.path.basename(image_path)}')
        axes[0, 0].axis('off')
        
        # Create bar plots for each model
        models = ['ViT', 'DINO', 'iBOT', 'MAE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (model_name, (top_probs, top_indices), color) in enumerate(zip(models, results, colors)):
            if i < 3:
                row = 1
                col = i
            else:
                row = 1
                col = 2
            
            y_pos = np.arange(len(top_indices))
            bars = axes[row, col].barh(y_pos, top_probs, color=color, alpha=0.7)
            axes[row, col].set_yticks(y_pos)
            axes[row, col].set_yticklabels([CIFAR10_CLASSES[idx] for idx in top_indices])
            axes[row, col].set_xlabel('Probability')
            axes[row, col].set_title(f'{model_name} Predictions')
            axes[row, col].invert_yaxis()
            
            # Add probability values on bars
            for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                axes[row, col].text(prob + 0.01, j, f'{prob:.3f}', va='center', fontsize=10)
        
        # Hide the unused subplots
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        output_path = f'comprehensive_model_comparison_{os.path.splitext(os.path.basename(image_path))[0]}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive comparison saved to: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

def print_comprehensive_results(results):
    """Print comprehensive results for all models."""
    models = ['ViT', 'DINO', 'iBOT', 'MAE']
    
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    for i, (model_name, (top_probs, top_indices)) in enumerate(zip(models, results)):
        print(f"\n📊 {model_name} Model Results:")
        print("-" * 40)
        
        for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = CIFAR10_CLASSES[idx]
            print(f"{j+1}. {class_name:12} - {prob:.3f} ({prob*100:.1f}%)")
        
        # Highlight top prediction
        top_class = CIFAR10_CLASSES[top_indices[0]]
        top_prob = top_probs[0]
        print(f"🎯 Top Prediction: {top_class} (Confidence: {top_prob:.3f})")

def main():
    # Configuration
    image_path = "results/final_evaluation/test_airplane.png"
    vit_model_path = "models/vit/best_vit_small_model.pth"
    dino_model_path = "models/dino/best_dino_fine_tuned_model.pth"
    ibot_model_path = "models/ibot/best_fine_tuned_model.pth"
    mae_model_path = "models/mae/best_mae_finetuned_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Comprehensive Analysis on {device}")
    print(f"📸 Image: {image_path}")
    print("="*60)
    
    # Load and preprocess image
    print("🔄 Preprocessing image...")
    image_tensor, original_image = preprocess_image(image_path)
    
    # Load models and get predictions
    results = []
    
    # ViT
    print("\n🤖 Loading ViT model...")
    vit_model = load_vit_model(vit_model_path, device)
    if vit_model is not None:
        vit_probs, vit_indices = classify_image(vit_model, image_tensor, device)
        results.append((vit_probs, vit_indices))
    else:
        print("❌ ViT model failed to load")
        results.append((np.zeros(5), np.zeros(5, dtype=int)))
    
    # DINO
    print("\n🤖 Loading DINO model...")
    dino_model = load_dino_model(dino_model_path, device)
    if dino_model is not None:
        dino_probs, dino_indices = classify_image(dino_model, image_tensor, device)
        results.append((dino_probs, dino_indices))
    else:
        print("❌ DINO model failed to load")
        results.append((np.zeros(5), np.zeros(5, dtype=int)))
    
    # iBOT
    print("\n🤖 Loading iBOT model...")
    ibot_model = load_ibot_model(ibot_model_path, device)
    if ibot_model is not None:
        ibot_probs, ibot_indices = classify_image(ibot_model, image_tensor, device)
        results.append((ibot_probs, ibot_indices))
    else:
        print("❌ iBOT model failed to load")
        results.append((np.zeros(5), np.zeros(5, dtype=int)))
    
    # MAE
    print("\n🤖 Loading MAE model...")
    mae_model = load_mae_model(mae_model_path, device)
    if mae_model is not None:
        mae_probs, mae_indices = classify_image(mae_model, image_tensor, device, "mae")
        results.append((mae_probs, mae_indices))
    else:
        print("❌ MAE model failed to load")
        results.append((np.zeros(5), np.zeros(5, dtype=int)))
    
    # Print results
    print_comprehensive_results(results)
    
    # Create visualization
    print("\n📊 Creating comprehensive visualization...")
    create_comprehensive_visualization(original_image, results, image_path)
    
    print("\n✅ Comprehensive analysis completed!")

if __name__ == '__main__':
    main() 