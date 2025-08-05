#!/usr/bin/env python3
"""
Linear Probing for Supervised ViT on CIFAR-10
Evaluates the quality of learned features by training a linear classifier on frozen backbone.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import timm
from sklearn.metrics import accuracy_score, classification_report
import argparse
from tqdm import tqdm
import time

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader for probing."""
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img, int(label)

def load_vit_model(model_path, device):
    """Load the supervised ViT model and extract backbone."""
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
        
        # Replace head with identity to get features
        model.head = nn.Identity()
        print("✅ Replaced head with Identity for feature extraction")
    else:
        print(f"⚠️  ViT model path {model_path} not found")
        return None
    
    model = model.to(device)
    model.eval()
    return model

def train_linear_probe(backbone, train_loader, val_loader, device, args):
    """Train a linear classifier on frozen backbone features."""
    print("Training linear classifier on frozen ViT features...")
    
    # Create linear classifier
    classifier = nn.Linear(backbone.num_features, 10).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        # Training
        backbone.eval()
        classifier.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                features = backbone(data)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        backbone.eval()
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                features = backbone(data)
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), os.path.join(args.output_dir, 'best_linear_classifier.pth'))
            print(f"✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    return classifier, best_val_acc, train_losses, val_accuracies

def evaluate_linear_probe(classifier, backbone, test_loader, device):
    """Evaluate the linear classifier on test set."""
    print("Evaluating linear classifier on test set...")
    
    backbone.eval()
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            features = backbone(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                             'dog', 'frog', 'horse', 'ship', 'truck'])
    
    return accuracy, report

def main():
    parser = argparse.ArgumentParser(description='Linear probing for supervised ViT')
    parser.add_argument('--data_path', type=str, default='../data/cifar10_splits/',
                       help='Path to CIFAR-10 data')
    parser.add_argument('--model_path', type=str, default='./best_vit_small_model.pth',
                       help='Path to ViT model')
    parser.add_argument('--output_dir', type=str, default='./linear_probe_results',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CIFAR10Dataset(
        os.path.join(args.data_path, 'probe_images.npy'),
        os.path.join(args.data_path, 'probe_labels.npy')
    )
    val_dataset = CIFAR10Dataset(
        os.path.join(args.data_path, 'val_images.npy'),
        os.path.join(args.data_path, 'val_labels.npy')
    )
    test_dataset = CIFAR10Dataset(
        os.path.join(args.data_path, 'test_images.npy'),
        os.path.join(args.data_path, 'test_labels.npy')
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Load ViT backbone
    backbone = load_vit_model(args.model_path, device)
    if backbone is None:
        print("❌ Could not load ViT model")
        return
    
    # Train linear classifier
    start_time = time.time()
    classifier, best_val_acc, train_losses, val_accuracies = train_linear_probe(
        backbone, train_loader, val_loader, device, args
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_acc, test_report = evaluate_linear_probe(classifier, backbone, test_loader, device)
    
    # Print summary
    print("\n" + "="*50)
    print("LINEAR PROBING RESULTS (Supervised ViT)")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print("\nTest Set Classification Report:")
    print(test_report)
    
    # Save results
    with open(os.path.join(args.output_dir, 'linear_probe_results.txt'), 'w') as f:
        f.write("LINEAR PROBING RESULTS (Supervised ViT)\n")
        f.write("="*50 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Training Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write("\nTest Set Classification Report:\n")
        f.write(test_report)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("Linear probing completed!")

if __name__ == '__main__':
    main() 