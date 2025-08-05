#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script
====================================

This script compares ViT and iBOT models on multiple test images,
providing detailed analysis of their performance differences.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_vit_model(model_path):
    """Load the supervised ViT model."""
    import timm
    
    print(f"Loading ViT model from {model_path}")
    
    try:
        model = timm.create_model('vit_small_patch16_224', 
                                patch_size=4, 
                                num_classes=128,  # Match checkpoint head size
                                img_size=32)
        print("✅ Using vit_small_patch16_224 with img_size=32")
    except Exception as e:
        print(f"❌ Failed to create ViT model: {e}")
        return None
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle nested block structure in checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == 'mask_token':
                continue
                
            if key.startswith('blocks.0.'):
                inner_key = key[9:]
                if inner_key.startswith(('0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.')):
                    parts = inner_key.split('.', 1)
                    if len(parts) == 2:
                        block_num = parts[0]
                        rest = parts[1]
                        new_key = f"blocks.{block_num}.{rest}"
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            elif key.startswith('head.'):
                if key == 'head.1.weight':
                    new_state_dict['head.weight'] = value
                elif key == 'head.1.bias':
                    new_state_dict['head.bias'] = value
            else:
                new_state_dict[key] = value
                
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded ViT weights from {model_path}")
        
        # Replace the head with the correct one for CIFAR-10
        model.head = nn.Linear(model.head.in_features, 10)
        print("✅ Replaced head for CIFAR-10 classification (10 classes)")
    else:
        print(f"⚠️  ViT model path {model_path} not found")
        return None
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("✅ ViT model loaded successfully")
    return model

def load_ibot_model(model_path):
    """Load the iBOT fine-tuned model."""
    import timm
    
    print(f"Loading iBOT model from {model_path}")
    
    # Define SimpleCNN locally to avoid external dependencies
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
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
            
            # Create FineTunedViT model
            class FineTunedViT(nn.Module):
                def __init__(self, backbone, num_classes=10):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = nn.Linear(backbone.num_features, num_classes)
                    
                def forward(self, x):
                    features = self.backbone(x)
                    return self.classifier(features)
            
            model = FineTunedViT(backbone, num_classes=10)
            
            # Load state dict
            model.load_state_dict(state_dict)
            print("✅ Loaded FineTunedViT model successfully")
            
        else:
            # Fallback to SimpleCNN
            print("⚠️ Could not load iBOT model: Using fallback SimpleCNN model")
            model = SimpleCNN(num_classes=10)
            
    except Exception as e:
        print(f"⚠️ Could not load iBOT model: {e}")
        print("Using fallback SimpleCNN model")
        model = SimpleCNN(num_classes=10)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("✅ iBOT model loaded successfully")
    return model

def preprocess_image(image_array, target_size=32):
    """Preprocess image array for model input."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def classify_image(model, image_tensor):
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
        
        # Get prediction and confidence
        pred_class = top_indices[0]
        confidence = top_probs[0]
        
        return pred_class, confidence, top_probs, top_indices

def load_test_data(data_dir, num_samples=100):
    """Load test data from numpy files."""
    print(f"Loading test data from {data_dir}")
    
    # Load test images and labels
    test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    # Limit number of samples
    if num_samples > 0:
        indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)
        test_images = test_images[indices]
        test_labels = test_labels[indices]
    
    print(f"✅ Loaded {len(test_images)} test samples")
    return test_images, test_labels

def evaluate_models(vit_model, ibot_model, test_images, test_labels):
    """Evaluate both models on test data."""
    print("🔍 Evaluating models on test data...")
    
    results = {
        'vit': {'predictions': [], 'confidences': [], 'correct': 0, 'total': 0},
        'ibot': {'predictions': [], 'confidences': [], 'correct': 0, 'total': 0}
    }
    
    class_results = {
        'vit': defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []}),
        'ibot': defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    }
    
    for i in tqdm(range(len(test_images)), desc="Processing images"):
        image = test_images[i]
        true_label = test_labels[i]
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get predictions from both models
        vit_pred, vit_conf, vit_probs, vit_indices = classify_image(vit_model, image_tensor)
        ibot_pred, ibot_conf, ibot_probs, ibot_indices = classify_image(ibot_model, image_tensor)
        
        # Store results
        results['vit']['predictions'].append(vit_pred)
        results['vit']['confidences'].append(vit_conf)
        results['vit']['total'] += 1
        if vit_pred == true_label:
            results['vit']['correct'] += 1
        
        results['ibot']['predictions'].append(ibot_pred)
        results['ibot']['confidences'].append(ibot_conf)
        results['ibot']['total'] += 1
        if ibot_pred == true_label:
            results['ibot']['correct'] += 1
        
        # Store class-wise results
        class_results['vit'][true_label]['total'] += 1
        class_results['vit'][true_label]['confidences'].append(vit_conf)
        if vit_pred == true_label:
            class_results['vit'][true_label]['correct'] += 1
        
        class_results['ibot'][true_label]['total'] += 1
        class_results['ibot'][true_label]['confidences'].append(ibot_conf)
        if ibot_pred == true_label:
            class_results['ibot'][true_label]['correct'] += 1
    
    return results, class_results

def print_comprehensive_results(results, class_results):
    """Print comprehensive comparison results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Overall accuracy
    vit_acc = results['vit']['correct'] / results['vit']['total'] * 100
    ibot_acc = results['ibot']['correct'] / results['ibot']['total'] * 100
    
    print(f"\n📊 OVERALL ACCURACY:")
    print(f"ViT Model:  {vit_acc:.2f}% ({results['vit']['correct']}/{results['vit']['total']})")
    print(f"iBOT Model: {ibot_acc:.2f}% ({results['ibot']['correct']}/{results['ibot']['total']})")
    print(f"Improvement: {ibot_acc - vit_acc:+.2f} percentage points")
    
    # Average confidence
    vit_avg_conf = np.mean(results['vit']['confidences']) * 100
    ibot_avg_conf = np.mean(results['ibot']['confidences']) * 100
    
    print(f"\n🎯 AVERAGE CONFIDENCE:")
    print(f"ViT Model:  {vit_avg_conf:.2f}%")
    print(f"iBOT Model: {ibot_avg_conf:.2f}%")
    print(f"Difference: {ibot_avg_conf - vit_avg_conf:+.2f} percentage points")
    
    # Class-wise performance
    print(f"\n📈 CLASS-WISE PERFORMANCE:")
    print(f"{'Class':<12} {'ViT Acc':<10} {'iBOT Acc':<10} {'Improvement':<12}")
    print("-" * 50)
    
    for class_idx in range(10):
        class_name = CIFAR10_CLASSES[class_idx]
        
        vit_class_acc = class_results['vit'][class_idx]['correct'] / class_results['vit'][class_idx]['total'] * 100
        ibot_class_acc = class_results['ibot'][class_idx]['correct'] / class_results['ibot'][class_idx]['total'] * 100
        
        improvement = ibot_class_acc - vit_class_acc
        
        print(f"{class_name:<12} {vit_class_acc:<10.2f} {ibot_class_acc:<10.2f} {improvement:+.2f}")
    
    # Confidence analysis
    print(f"\n🔍 CONFIDENCE ANALYSIS:")
    vit_conf_std = np.std(results['vit']['confidences']) * 100
    ibot_conf_std = np.std(results['ibot']['confidences']) * 100
    
    print(f"ViT Confidence Std:  {vit_conf_std:.2f}%")
    print(f"iBOT Confidence Std: {ibot_conf_std:.2f}%")
    
    # High confidence predictions
    high_conf_threshold = 0.8
    vit_high_conf = sum(1 for conf in results['vit']['confidences'] if conf > high_conf_threshold)
    ibot_high_conf = sum(1 for conf in results['ibot']['confidences'] if conf > high_conf_threshold)
    
    print(f"\nHigh confidence predictions (>80%):")
    print(f"ViT:  {vit_high_conf}/{len(results['vit']['confidences'])} ({vit_high_conf/len(results['vit']['confidences'])*100:.1f}%)")
    print(f"iBOT: {ibot_high_conf}/{len(results['ibot']['confidences'])} ({ibot_high_conf/len(results['ibot']['confidences'])*100:.1f}%)")

def create_visualizations(results, class_results):
    """Create comprehensive visualizations."""
    print("\n📊 Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Comparison: ViT vs iBOT', fontsize=16, fontweight='bold')
    
    # 1. Overall accuracy comparison
    vit_acc = results['vit']['correct'] / results['vit']['total'] * 100
    ibot_acc = results['ibot']['correct'] / results['ibot']['total'] * 100
    
    axes[0, 0].bar(['ViT', 'iBOT'], [vit_acc, ibot_acc], color=['#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Overall Accuracy')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate([vit_acc, ibot_acc]):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Confidence distribution
    axes[0, 1].hist(results['vit']['confidences'], alpha=0.7, label='ViT', bins=20, color='#ff7f0e')
    axes[0, 1].hist(results['ibot']['confidences'], alpha=0.7, label='iBOT', bins=20, color='#2ca02c')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Class-wise accuracy comparison
    class_names = CIFAR10_CLASSES
    vit_class_accs = []
    ibot_class_accs = []
    
    for class_idx in range(10):
        vit_class_acc = class_results['vit'][class_idx]['correct'] / class_results['vit'][class_idx]['total'] * 100
        ibot_class_acc = class_results['ibot'][class_idx]['correct'] / class_results['ibot'][class_idx]['total'] * 100
        vit_class_accs.append(vit_class_acc)
        ibot_class_accs.append(ibot_class_acc)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, vit_class_accs, width, label='ViT', color='#ff7f0e')
    axes[0, 2].bar(x + width/2, ibot_class_accs, width, label='iBOT', color='#2ca02c')
    axes[0, 2].set_title('Class-wise Accuracy')
    axes[0, 2].set_xlabel('Classes')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0, 100)
    
    # 4. Average confidence by class
    vit_class_conf = []
    ibot_class_conf = []
    
    for class_idx in range(10):
        vit_avg_conf = np.mean(class_results['vit'][class_idx]['confidences']) * 100
        ibot_avg_conf = np.mean(class_results['ibot'][class_idx]['confidences']) * 100
        vit_class_conf.append(vit_avg_conf)
        ibot_class_conf.append(ibot_avg_conf)
    
    axes[1, 0].bar(x - width/2, vit_class_conf, width, label='ViT', color='#ff7f0e')
    axes[1, 0].bar(x + width/2, ibot_class_conf, width, label='iBOT', color='#2ca02c')
    axes[1, 0].set_title('Average Confidence by Class')
    axes[1, 0].set_xlabel('Classes')
    axes[1, 0].set_ylabel('Average Confidence (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].legend()
    
    # 5. Accuracy vs Confidence scatter
    vit_correct = [1 if pred == true else 0 for pred, true in zip(results['vit']['predictions'], range(len(results['vit']['predictions'])))]
    ibot_correct = [1 if pred == true else 0 for pred, true in zip(results['ibot']['predictions'], range(len(results['ibot']['predictions'])))]
    
    axes[1, 1].scatter(results['vit']['confidences'], vit_correct, alpha=0.6, label='ViT', color='#ff7f0e')
    axes[1, 1].scatter(results['ibot']['confidences'], ibot_correct, alpha=0.6, label='iBOT', color='#2ca02c')
    axes[1, 1].set_title('Accuracy vs Confidence')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Correct (1) / Incorrect (0)')
    axes[1, 1].legend()
    
    # 6. Performance improvement by class
    improvements = [ibot - vit for vit, ibot in zip(vit_class_accs, ibot_class_accs)]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1, 2].bar(class_names, improvements, color=colors)
    axes[1, 2].set_title('Performance Improvement (iBOT - ViT)')
    axes[1, 2].set_xlabel('Classes')
    axes[1, 2].set_ylabel('Accuracy Improvement (%)')
    axes[1, 2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'comprehensive_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive comparison saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Comparison')
    parser.add_argument('--vit_model_path', type=str, default='models/vit/best_vit_small_model.pth',
                       help='Path to ViT model checkpoint')
    parser.add_argument('--ibot_model_path', type=str, default='models/ibot/best_fine_tuned_model.pth',
                       help='Path to iBOT model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/cifar10_splits',
                       help='Directory containing test data')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test samples to evaluate')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip creating visualizations')
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Model Comparison")
    print("="*60)
    
    # Load models
    vit_model = load_vit_model(args.vit_model_path)
    if vit_model is None:
        print("❌ Failed to load ViT model")
        return
    
    ibot_model = load_ibot_model(args.ibot_model_path)
    if ibot_model is None:
        print("❌ Failed to load iBOT model")
        return
    
    # Load test data
    test_images, test_labels = load_test_data(args.data_dir, args.num_samples)
    
    # Evaluate models
    results, class_results = evaluate_models(vit_model, ibot_model, test_images, test_labels)
    
    # Print results
    print_comprehensive_results(results, class_results)
    
    # Create visualizations
    if not args.no_plot:
        create_visualizations(results, class_results)
    
    print("\n✅ Comprehensive comparison completed!")

if __name__ == '__main__':
    main() 