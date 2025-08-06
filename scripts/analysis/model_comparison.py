#!/usr/bin/env python3
"""
Model Comparison Script for SSL ViT Project
===========================================

This script performs comprehensive comparison of all models (ViT, DINO, iBOT, MAE)
in two modes:
1. Single-image mode: Qualitative analysis on one test image
2. Multi-image mode: Statistical analysis on multiple test images

Usage:
    python model_comparison.py --mode single --image test_airplane.png
    python model_comparison.py --mode multi --num_samples 100
    python model_comparison.py --mode both
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
import argparse
from collections import defaultdict
from tqdm import tqdm

# Add DINO v2 path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'dino', 'dinov2'))
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

def print_single_image_results(results):
    """Print comprehensive results for all models on single image."""
    models = ['ViT', 'DINO', 'iBOT', 'MAE']
    
    print("\n" + "="*80)
    print("🔍 SINGLE IMAGE MODEL COMPARISON RESULTS")
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

def load_test_data(data_dir, num_samples=100):
    """Load test data for multi-image analysis."""
    print(f"📊 Loading {num_samples} test samples from {data_dir}")
    
    try:
        test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        
        # Take a subset if requested
        if num_samples < len(test_images):
            indices = np.random.choice(len(test_images), num_samples, replace=False)
            test_images = test_images[indices]
            test_labels = test_labels[indices]
        
        print(f"✅ Loaded {len(test_images)} test samples")
        return test_images, test_labels
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None

def evaluate_models_on_multiple_images(vit_model, dino_model, ibot_model, mae_model, test_images, test_labels, device):
    """Evaluate all models on multiple test images."""
    print("🔄 Evaluating models on multiple images...")
    
    results = {
        'vit': {'predictions': [], 'confidences': [], 'correct': 0},
        'dino': {'predictions': [], 'confidences': [], 'correct': 0},
        'ibot': {'predictions': [], 'confidences': [], 'correct': 0},
        'mae': {'predictions': [], 'confidences': [], 'correct': 0}
    }
    
    models = [vit_model, dino_model, ibot_model, mae_model]
    model_names = ['vit', 'dino', 'ibot', 'mae']
    
    for i in tqdm(range(len(test_images)), desc="Processing images"):
        image_array = test_images[i]
        true_label = test_labels[i]
        
        # Preprocess image
        image_tensor = preprocess_image_from_array(image_array)
        
        for j, (model, model_name) in enumerate(zip(models, model_names)):
            if model is not None:
                try:
                    if model_name == 'mae':
                        probs, indices = classify_image(model, image_tensor, device, "mae")
                    else:
                        probs, indices = classify_image(model, image_tensor, device)
                    
                    prediction = indices[0]
                    confidence = probs[0]
                    
                    results[model_name]['predictions'].append(prediction)
                    results[model_name]['confidences'].append(confidence)
                    
                    if prediction == true_label:
                        results[model_name]['correct'] += 1
                        
                except Exception as e:
                    print(f"⚠️ Error with {model_name} on image {i}: {e}")
                    results[model_name]['predictions'].append(-1)
                    results[model_name]['confidences'].append(0.0)
            else:
                results[model_name]['predictions'].append(-1)
                results[model_name]['confidences'].append(0.0)
    
    return results

def preprocess_image_from_array(image_array, target_size=32):
    """Preprocess image from numpy array."""
    # Convert to PIL Image
    image = Image.fromarray(image_array.astype(np.uint8))
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def print_multi_image_results(results, num_samples):
    """Print statistical results for multi-image analysis."""
    print("\n" + "="*80)
    print(f"📊 MULTI-IMAGE STATISTICAL ANALYSIS ({num_samples} samples)")
    print("="*80)
    
    models = ['ViT', 'DINO', 'iBOT', 'MAE']
    model_keys = ['vit', 'dino', 'ibot', 'mae']
    
    print(f"\n{'Model':<10} {'Accuracy':<12} {'Avg Confidence':<15} {'Correct/Total':<15}")
    print("-" * 60)
    
    for model_name, model_key in zip(models, model_keys):
        correct = results[model_key]['correct']
        total = len(results[model_key]['predictions'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        avg_confidence = np.mean(results[model_key]['confidences']) * 100
        
        print(f"{model_name:<10} {accuracy:<12.2f}% {avg_confidence:<15.2f}% {correct:<3}/{total:<3}")
    
    print("\n" + "="*60)
    print("🏆 PERFORMANCE RANKINGS")
    print("="*60)
    
    # Sort by accuracy
    accuracies = []
    for model_key in model_keys:
        correct = results[model_key]['correct']
        total = len(results[model_key]['predictions'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracies.append((accuracy, model_key))
    
    accuracies.sort(reverse=True)
    
    for i, (accuracy, model_key) in enumerate(accuracies):
        model_name = models[model_keys.index(model_key)]
        print(f"{i+1}. {model_name}: {accuracy:.2f}%")

def create_multi_image_visualization(results, num_samples):
    """Create comprehensive visualization for multi-image analysis."""
    print("📊 Creating multi-image analysis visualization...")
    
    models = ['ViT', 'DINO', 'iBOT', 'MAE']
    model_keys = ['vit', 'dino', 'ibot', 'mae']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Calculate metrics
    accuracies = []
    avg_confidences = []
    
    for model_key in model_keys:
        correct = results[model_key]['correct']
        total = len(results[model_key]['predictions'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_conf = np.mean(results[model_key]['confidences']) * 100
        
        accuracies.append(accuracy)
        avg_confidences.append(avg_conf)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Multi-Image Model Comparison Analysis ({num_samples} samples)', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    bars1 = axes[0, 0].bar(models, accuracies, color=colors, alpha=0.8)
    axes[0, 0].set_title('Test Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average confidence comparison
    bars2 = axes[0, 1].bar(models, avg_confidences, color=colors, alpha=0.8)
    axes[0, 1].set_title('Average Confidence Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Confidence (%)')
    axes[0, 1].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, conf in zip(bars2, avg_confidences):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Accuracy vs Confidence scatter
    axes[1, 0].scatter(avg_confidences, accuracies, c=colors, s=100, alpha=0.8)
    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (avg_confidences[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
    axes[1, 0].set_xlabel('Average Confidence (%)')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy vs Confidence', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance ranking
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    bars3 = axes[1, 1].bar(range(len(sorted_models)), sorted_accuracies, 
                          color=[colors[i] for i in sorted_indices], alpha=0.8)
    axes[1, 1].set_title('Performance Ranking', fontweight='bold')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_xticks(range(len(sorted_models)))
    axes[1, 1].set_xticklabels([f"{i+1}. {model}" for i, model in enumerate(sorted_models)], rotation=45)
    axes[1, 1].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars3, sorted_accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'results/final_evaluation/multi_image_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Multi-image comparison saved to: {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Script for SSL ViT Project')
    parser.add_argument('--mode', type=str, choices=['single', 'multi', 'both'], default='single',
                       help='Analysis mode: single (one image), multi (multiple images), or both')
    parser.add_argument('--image', type=str, default='results/final_evaluation/test_airplane.png',
                       help='Path to test image for single mode')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test samples for multi mode')
    parser.add_argument('--data_dir', type=str, default='data/cifar10_splits',
                       help='Directory containing test data')
    parser.add_argument('--vit_model_path', type=str, default='models/vit/best_vit_small_model.pth',
                       help='Path to ViT model')
    parser.add_argument('--dino_model_path', type=str, default='models/dino/best_dino_fine_tuned_model.pth',
                       help='Path to DINO model')
    parser.add_argument('--ibot_model_path', type=str, default='models/ibot/best_fine_tuned_model.pth',
                       help='Path to iBOT model')
    parser.add_argument('--mae_model_path', type=str, default='models/mae/best_mae_finetuned_model.pth',
                       help='Path to MAE model')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Model Comparison on {device}")
    print(f"📊 Mode: {args.mode}")
    print("="*60)
    
    # Load all models
    print("🤖 Loading models...")
    vit_model = load_vit_model(args.vit_model_path, device)
    dino_model = load_dino_model(args.dino_model_path, device)
    ibot_model = load_ibot_model(args.ibot_model_path, device)
    mae_model = load_mae_model(args.mae_model_path, device)
    
    # Single image analysis
    if args.mode in ['single', 'both']:
        print("\n" + "="*60)
        print("📸 SINGLE IMAGE ANALYSIS")
        print("="*60)
        
        # Load and preprocess image
        print(f"🔄 Preprocessing image: {args.image}")
        image_tensor, original_image = preprocess_image(args.image)
        
        # Get predictions from all models
        results = []
        
        # ViT
        if vit_model is not None:
            vit_probs, vit_indices = classify_image(vit_model, image_tensor, device)
            results.append((vit_probs, vit_indices))
        else:
            print("❌ ViT model failed to load")
            results.append((np.zeros(5), np.zeros(5, dtype=int)))
        
        # DINO
        if dino_model is not None:
            dino_probs, dino_indices = classify_image(dino_model, image_tensor, device)
            results.append((dino_probs, dino_indices))
        else:
            print("❌ DINO model failed to load")
            results.append((np.zeros(5), np.zeros(5, dtype=int)))
        
        # iBOT
        if ibot_model is not None:
            ibot_probs, ibot_indices = classify_image(ibot_model, image_tensor, device)
            results.append((ibot_probs, ibot_indices))
        else:
            print("❌ iBOT model failed to load")
            results.append((np.zeros(5), np.zeros(5, dtype=int)))
        
        # MAE
        if mae_model is not None:
            mae_probs, mae_indices = classify_image(mae_model, image_tensor, device, "mae")
            results.append((mae_probs, mae_indices))
        else:
            print("❌ MAE model failed to load")
            results.append((np.zeros(5), np.zeros(5, dtype=int)))
        
        # Print results
        print_single_image_results(results)
        
        # Create visualization
        print("\n📊 Creating single image visualization...")
        create_comprehensive_visualization(original_image, results, args.image)
    
    # Multi-image analysis
    if args.mode in ['multi', 'both']:
        print("\n" + "="*60)
        print("📊 MULTI-IMAGE ANALYSIS")
        print("="*60)
        
        # Load test data
        test_images, test_labels = load_test_data(args.data_dir, args.num_samples)
        
        if test_images is not None:
            # Evaluate models on multiple images
            results = evaluate_models_on_multiple_images(
                vit_model, dino_model, ibot_model, mae_model, 
                test_images, test_labels, device
            )
            
            # Print results
            print_multi_image_results(results, args.num_samples)
            
            # Create visualization
            print("\n📊 Creating multi-image visualization...")
            create_multi_image_visualization(results, args.num_samples)
        else:
            print("❌ Failed to load test data")
    
    print("\n✅ Model comparison completed!")

if __name__ == '__main__':
    main() 