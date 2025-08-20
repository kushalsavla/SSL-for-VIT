#!/usr/bin/env python3
"""
Comprehensive iBOT Classification Script
Supports fine-tuning, linear probing, and nonlinear probing on CIFAR-10.
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
import json
from pathlib import Path

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader with configurable augmentation."""
    def __init__(self, images_path, labels_path, is_training=True, augmentation_level='medium'):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.is_training = is_training
        self.augmentation_level = augmentation_level
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]  # Fixed: was self.images[idx]
        
        # Convert to tensor and normalize
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = (img - mean) / std
        
        # Data augmentation for training
        if self.is_training and self.augmentation_level != 'none':
            if self.augmentation_level == 'light':
                # Light augmentation
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, dims=[2])  # Random horizontal flip
            elif self.augmentation_level == 'medium':
                # Medium augmentation
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, dims=[2])
                
                # Random crop with padding
                pad = 4
                img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')
                h, w = img.shape[1], img.shape[2]
                y = torch.randint(0, h - 32 + 1, (1,)).item()
                x = torch.randint(0, w - 32 + 1, (1,)).item()
                img = img[:, y:y+32, x:x+32]
            elif self.augmentation_level == 'heavy':
                # Heavy augmentation
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, dims=[2])
                
                # Random crop with padding
                pad = 4
                img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')
                h, w = img.shape[1], img.shape[2]
                y = torch.randint(0, h - 32 + 1, (1,)).item()
                x = torch.randint(0, w - 32 + 1, (1,)).item()
                img = img[:, y:y+32, x:x+32]
                
                # Color jittering
                if torch.rand(1) > 0.5:
                    brightness = 0.8 + 0.4 * torch.rand(1)
                    img = img * brightness
                    img = torch.clamp(img, 0, 1)
        
        return img, int(label)

class iBOTBackbone(nn.Module):
    """iBOT backbone with configurable output."""
    def __init__(self, model_path, device, freeze_backbone=True):
        super().__init__()
        self.device = device
        self.freeze_backbone = freeze_backbone
        
        # Create the same model architecture as used in iBOT training
        try:
            self.backbone = timm.create_model('vit_small_patch16_224', 
                                           patch_size=4, 
                                           num_classes=0, 
                                           img_size=32)
            print("✅ Using vit_small_patch16_224 with img_size=32")
        except Exception as e:
            print(f"❌ Failed to create vit_small_patch16_224: {e}")
            raise e
        
        # Load pretrained weights
        self._load_weights(model_path)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen")
        else:
            print("🔓 Backbone unfrozen for fine-tuning")
    
    def _load_weights(self, model_path):
        """Load iBOT pretrained weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not found")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'student' in checkpoint:
            # Load student backbone weights from SSL training
            student_state = checkpoint['student']
            backbone_state = {}
            for key, value in student_state.items():
                if key.startswith('0.'):
                    backbone_state[key[2:]] = value  # Remove "0." prefix
                else:
                    continue  # Skip non-backbone keys
            self.backbone.load_state_dict(backbone_state)
            print(f"✅ Loaded iBOT student backbone weights from {model_path}")
        elif 'model_state_dict' in checkpoint:
            # Load fine-tuned model weights
            model_state = checkpoint['model_state_dict']
            backbone_state = {}
            for key, value in model_state.items():
                if key.startswith('backbone.'):
                    backbone_state[key[9:]] = value  # Remove "backbone." prefix
            self.backbone.load_state_dict(backbone_state)
            print(f"✅ Loaded fine-tuned backbone weights from {model_path}")
        else:
            # Direct loading
            self.backbone.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {model_path}")
    
    def forward(self, x):
        """Extract features from backbone."""
        return self.backbone(x)

class FineTunedModel(nn.Module):
    """Fine-tuned iBOT model with trainable backbone."""
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(backbone.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class LinearProbe(nn.Module):
    """Linear classifier on frozen iBOT features."""
    def __init__(self, feature_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

class NonlinearProbe(nn.Module):
    """Nonlinear classifier on frozen iBOT features."""
    def __init__(self, feature_dim, num_classes=10, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

def train_model(model, train_loader, val_loader, device, args):
    """Train the model."""
    print(f"🚀 Starting training with {args.mode} approach...")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if args.mode == 'fine_tune':
        # For fine-tuning, use lower learning rate
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # For probing, use higher learning rate
        optimizer = optim.AdamW(model.parameters(), lr=args.lr * 10, weight_decay=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for fine-tuning
            if args.mode == 'fine_tune':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        print(f"🔧 Train Loss: {train_loss/len(train_loader):.4f} | 🎯 Train Acc: {train_acc:.2f}%")
        print(f"🧪 Val Loss: {val_loss/len(val_loader):.4f} | 🎯 Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"models/ibot/best_{args.mode}_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if args.mode == 'fine_tune':
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'mode': args.mode
                }, save_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'mode': args.mode
                }, save_path)
            
            print(f"✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
            break
    
    return best_val_acc

def evaluate_on_test(model, test_loader, device):
    """Evaluate the model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\n📊 TEST RESULTS:")
    print(f"🎯 Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"📉 Test Loss: {test_loss/len(test_loader):.4f}")
    
    return accuracy, all_predictions, all_labels

def main():
    parser = argparse.ArgumentParser(description='iBOT Classification on CIFAR-10')
    parser.add_argument('--mode', type=str, choices=['fine_tune', 'linear_probe', 'nonlinear_probe'], 
                       default='fine_tune', help='Classification approach')
    parser.add_argument('--model_path', type=str, default='models/ibot/final_model.pth',
                       help='Path to iBOT pretrained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--augmentation', type=str, choices=['none', 'light', 'medium', 'heavy'],
                       default='medium', help='Data augmentation level')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")
    
    # Data loaders
    train_dataset = CIFAR10Dataset('data/cifar10_splits/train_images.npy', 
                                  'data/cifar10_splits/train_labels.npy',
                                  is_training=True, augmentation_level=args.augmentation)
    val_dataset = CIFAR10Dataset('data/cifar10_splits/val_images.npy',
                                'data/cifar10_splits/val_labels.npy',
                                is_training=False, augmentation_level='none')
    test_dataset = CIFAR10Dataset('data/cifar10_splits/test_images.npy',
                                 'data/cifar10_splits/test_labels.npy',
                                 is_training=False, augmentation_level='none')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create model based on mode
    if args.mode == 'fine_tune':
        # Fine-tuning: unfreeze backbone
        backbone = iBOTBackbone(args.model_path, device, freeze_backbone=False)
        model = FineTunedModel(backbone, num_classes=10)
        print("🔓 Created fine-tuning model with unfrozen backbone")
        
    elif args.mode == 'linear_probe':
        # Linear probing: freeze backbone
        backbone = iBOTBackbone(args.model_path, device, freeze_backbone=True)
        model = LinearProbe(backbone.backbone.num_features, num_classes=10)
        print("🔒 Created linear probe with frozen backbone")
        
    elif args.mode == 'nonlinear_probe':
        # Nonlinear probing: freeze backbone
        backbone = iBOTBackbone(args.model_path, device, freeze_backbone=True)
        model = NonlinearProbe(backbone.backbone.num_features, num_classes=10)
        print("🔒 Created nonlinear probe with frozen backbone")
    
    model.to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Train model
    best_val_acc = train_model(model, train_loader, val_loader, device, args)
    
    # Evaluate on test set
    test_acc, predictions, labels = evaluate_on_test(model, test_loader, device)
    
    # Save results
    results = {
        'mode': args.mode,
        'model_path': args.model_path,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'augmentation': args.augmentation,
        'trainable_params': trainable_params,
        'total_params': total_params
    }
    
    results_path = f"results/ibot_experiments/{args.mode}_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"🏁 {args.mode.replace('_', ' ').title()} complete!")

if __name__ == "__main__":
    main()
