#!/usr/bin/env python3
"""
Test script to evaluate ViT and iBOT models on unseen test data.
This script provides final performance metrics for both approaches.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import timm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import time
from tqdm import tqdm

class FineTunedViT(nn.Module):
    """Fine-tuned ViT with classification head."""
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CIFAR10TestDataset(Dataset):
    """Dataset for testing on unseen CIFAR-10 data."""
    
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        print(f"Loaded {len(self.images)} test images")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # CIFAR-10 normalization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img, int(label)

def load_dino_model(model_path, device):
    """Load the DINO fine-tuned model."""
    print(f"Loading DINO model from {model_path}")
    
    try:
        # Import DINO vision transformer
        import sys
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))
        from models.vision_transformer import vit_small
        
        # Create DINO backbone
        backbone = vit_small(img_size=32, patch_size=4)
        
        # Create fine-tuned model with classifier
        class FineTunedDinoModel(nn.Module):
            def __init__(self, backbone, num_classes=10):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(384, num_classes)
            
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        model = FineTunedDinoModel(backbone, num_classes=10)
        print("✅ Created DINO fine-tuned model")
        
    except Exception as e:
        print(f"❌ Failed to create DINO model: {e}")
        return None
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict)
            print("✅ Loaded DINO model successfully")
        except Exception as e:
            print(f"❌ Failed to load DINO model state dict: {e}")
            return None
    else:
        print(f"⚠️ DINO model path {model_path} not found")
        return None
    
    model.to(device)
    model.eval()
    return model

def load_vit_model(model_path, device):
    """Load the supervised ViT model."""
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
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle nested block structure in checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip mask_token as it's not needed for evaluation
            if key == 'mask_token':
                continue
                
            # Convert nested blocks like "blocks.0.1.norm1.weight" to "blocks.1.norm1.weight"
            if key.startswith('blocks.0.'):
                # Extract the inner block number and rest of the key
                inner_key = key[9:]  # Remove "blocks.0."
                if inner_key.startswith(('0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.')):
                    # This is a nested block, extract the block number
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
            # Handle head structure - convert complex head to simple head
            elif key.startswith('head.'):
                if key == 'head.1.weight':
                    new_state_dict['head.weight'] = value
                elif key == 'head.1.bias':
                    new_state_dict['head.bias'] = value
                # Skip other head components (head.3.weight, head.3.bias)
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
    
    model = model.to(device)
    model.eval()
    return model

def load_ibot_model(model_path, device):
    """Load the iBOT SSL model for evaluation."""
    print(f"Loading iBOT model from {model_path}")
    
    try:
        backbone = timm.create_model('vit_small_patch16_224', 
                                   patch_size=4, 
                                   num_classes=0, 
                                   img_size=32)
        print("✅ Using vit_small_patch16_224 backbone")
    except Exception as e:
        print(f"❌ Failed to create ViT backbone: {e}")
        return None
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'student' in checkpoint:
            # SSL pre-trained model
            student_state = checkpoint['student']
            backbone_state = {}
            for key, value in student_state.items():
                if key.startswith('0.'):
                    backbone_state[key[2:]] = value
            backbone.load_state_dict(backbone_state)
            print(f"✅ Loaded SSL student weights from {model_path}")
        elif 'model_state_dict' in checkpoint:
            # Fine-tuned model - load the complete model
            model_state = checkpoint['model_state_dict']
            
            # Create the complete fine-tuned model (backbone + classifier)
            model = FineTunedViT(backbone, num_classes=10)
            model.load_state_dict(model_state)
            print(f"✅ Loaded complete fine-tuned model from {model_path}")
            
            model = model.to(device)
            model.eval()
            return model
        else:
            backbone.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {model_path}")
    else:
        print(f"⚠️  iBOT model path {model_path} not found")
        return None
    
    # Add classifier head for evaluation (only for non-fine-tuned models)
    classifier = nn.Linear(backbone.num_features, 10)
    model = nn.Sequential(backbone, classifier)
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model on test set."""
    print(f"\n🔬 Evaluating {model_name} on test set...")
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_time = time.time() - start_time
    accuracy = 100 * correct / total
    
    # Calculate metrics
    accuracy_sklearn = accuracy_score(all_labels, all_predictions) * 100
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                                             'dog', 'frog', 'horse', 'ship', 'truck'],
                                 digits=3)
    
    print(f"\n📊 {model_name} Test Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    print(f"   Test Time: {test_time:.2f} seconds")
    print(f"   Throughput: {total/test_time:.1f} images/second")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'test_time': test_time,
        'throughput': total/test_time,
        'predictions': all_predictions,
        'labels': all_labels,
        'report': report
    }

def main():
    parser = argparse.ArgumentParser(description='Test ViT, DINO, and iBOT models on unseen data')
    parser.add_argument('--data_path', type=str, default='./data/cifar10_splits/',
                       help='Path to CIFAR-10 test data')
    parser.add_argument('--vit_model_path', type=str, default='./vit/best_vit_small_model.pth',
                       help='Path to ViT model')
    parser.add_argument('--dino_model_path', type=str, default='./dino/best_dino_fine_tuned_model.pth',
                       help='Path to DINO model')
    parser.add_argument('--ibot_model_path', type=str, default='./ibot/fine_tune_results/best_fine_tuned_model.pth',
                       help='Path to iBOT model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CIFAR10TestDataset(
        os.path.join(args.data_path, 'test_images.npy'),
        os.path.join(args.data_path, 'test_labels.npy')
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    results = {}
    
    # Test ViT model
    vit_model = load_vit_model(args.vit_model_path, device)
    if vit_model is not None:
        vit_results = evaluate_model(vit_model, test_loader, device, "Supervised ViT")
        results['vit'] = vit_results
    else:
        print("❌ Could not load ViT model")
    
    # Test DINO model
    dino_model = load_dino_model(args.dino_model_path, device)
    if dino_model is not None:
        dino_results = evaluate_model(dino_model, test_loader, device, "DINO SSL")
        results['dino'] = dino_results
    else:
        print("❌ Could not load DINO model")
    
    # Test iBOT model
    ibot_model = load_ibot_model(args.ibot_model_path, device)
    if ibot_model is not None:
        ibot_results = evaluate_model(ibot_model, test_loader, device, "iBOT SSL")
        results['ibot'] = ibot_results
    else:
        print("❌ Could not load iBOT model")
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL TEST RESULTS COMPARISON")
    print("="*60)
    
    if 'vit' in results and 'dino' in results and 'ibot' in results:
        print(f"{'Model':<20} {'Accuracy':<12} {'Time (s)':<10} {'Throughput':<12}")
        print("-" * 60)
        print(f"{'Supervised ViT':<20} {results['vit']['accuracy']:<12.2f} {results['vit']['test_time']:<10.2f} {results['vit']['throughput']:<12.1f}")
        print(f"{'DINO SSL':<20} {results['dino']['accuracy']:<12.2f} {results['dino']['test_time']:<10.2f} {results['dino']['throughput']:<12.1f}")
        print(f"{'iBOT SSL':<20} {results['ibot']['accuracy']:<12.2f} {results['ibot']['test_time']:<10.2f} {results['ibot']['throughput']:<12.1f}")
        
        # Calculate improvements
        dino_improvement = results['dino']['accuracy'] - results['vit']['accuracy']
        ibot_improvement = results['ibot']['accuracy'] - results['vit']['accuracy']
        
        print(f"\n📈 DINO improvement over ViT: {dino_improvement:+.2f}%")
        print(f"📈 iBOT improvement over ViT: {ibot_improvement:+.2f}%")
        
        if dino_improvement > 0:
            print("✅ DINO SSL outperforms supervised ViT!")
        else:
            print("⚠️  Supervised ViT outperforms DINO SSL")
            
        if ibot_improvement > 0:
            print("✅ iBOT SSL outperforms supervised ViT!")
        else:
            print("⚠️  Supervised ViT outperforms iBOT SSL")
    
    # Save detailed results
    with open('final_test_results.txt', 'w') as f:
        f.write("FINAL TEST RESULTS ON UNSEEN DATA\n")
        f.write("="*50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name.upper()} RESULTS:\n")
            f.write(f"Accuracy: {result['accuracy']:.2f}%\n")
            f.write(f"Correct: {result['correct']}/{result['total']}\n")
            f.write(f"Test Time: {result['test_time']:.2f} seconds\n")
            f.write(f"Throughput: {result['throughput']:.1f} images/second\n\n")
            f.write("Classification Report:\n")
            f.write(result['report'])
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"\n📄 Detailed results saved to: final_test_results.txt")
    print("✅ Testing completed!")

if __name__ == '__main__':
    main() 