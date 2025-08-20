#!/usr/bin/env python3
"""
SSL Classification Testing Script
Tests all SSL methods (MAE, DINO, iBOT) on CIFAR-10 classification tasks.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'external', 'dino', 'dinov2'))

try:
    from models.vision_transformer import vit_small
    print("✅ Successfully imported DINO v2 ViT")
except ImportError:
    print("⚠️ DINO v2 ViT not available, falling back to timm")
    import timm

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader for testing."""
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

def load_ssl_model(method, model_path, device):
    """Load SSL model based on method."""
    print(f"🔍 Loading {method.upper()} model from {model_path}")
    
    if method == 'mae':
        # Load MAE model
        try:
            from working.models_mae import mae_vit_small_patch16
            model = mae_vit_small_patch16(
                img_size=32, patch_size=4, in_chans=3,
                embed_dim=384, depth=8, num_heads=6,
                decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✅ MAE model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading MAE model: {e}")
            return None
            
    elif method == 'dino':
        # Load DINO model
        try:
            model = vit_small(img_size=32, patch_size=4, num_classes=0)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("✅ DINO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading DINO model: {e}")
            return None
            
    elif method == 'ibot':
        # Load iBOT model
        try:
            model = vit_small(img_size=32, patch_size=4, num_classes=0)
            checkpoint = torch.load(model_path, map_location=device)
            if 'student' in checkpoint:
                student_state = checkpoint['student']
                backbone_state = {}
                for key, value in student_state.items():
                    if key.startswith('0.'):
                        backbone_state[key[2:]] = value
                    else:
                        continue
                model.load_state_dict(backbone_state, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("✅ iBOT model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading iBOT model: {e}")
            return None
    
    else:
        print(f"❌ Unknown method: {method}")
        return None
    
    return model

def extract_features(model, dataloader, device):
    """Extract features from the frozen backbone."""
    model.eval()
    features = []
    labels = []
    
    print("🔍 Extracting features...")
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Extracting features"):
            data, target = data.to(device), target.to(device)
            
            # Get features (remove classification head if exists)
            if hasattr(model, 'head'):
                original_head = model.head
                model.head = nn.Identity()
                feat = model(data)
                model.head = original_head
            else:
                feat = model(data)
            
            # Handle different output formats
            if isinstance(feat, tuple):
                feat = feat[0]
            
            features.append(feat.cpu())
            labels.append(target.cpu())
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"✅ Extracted features: {features.shape}")
    print(f"✅ Extracted labels: {labels.shape}")
    
    return features, labels

def create_linear_probe(feature_dim, num_classes=10):
    """Create a linear classifier for probing."""
    return nn.Linear(feature_dim, num_classes)

def create_nonlinear_probe(feature_dim, num_classes=10, hidden_dim=512):
    """Create a nonlinear classifier for probing."""
    return nn.Sequential(
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

def train_probe(classifier, train_features, train_labels, val_features, val_labels, 
                device, epochs=100, lr=0.01, probe_type='linear'):
    """Train a probe classifier on extracted features."""
    print(f"🚀 Training {probe_type} probe on {classifier[0].in_features if probe_type == 'nonlinear' else classifier.in_features}-dimensional features...")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx in range(0, len(train_features), 128):
            end_idx = min(batch_idx + 128, len(train_features))
            batch_features = train_features[batch_idx:end_idx].to(device)
            batch_labels = train_labels[batch_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_features.size(0)
            pred = outputs.argmax(dim=1)
            train_correct += (pred == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(val_features), 128):
                end_idx = min(batch_idx + 128, len(val_features))
                batch_features = val_features[batch_idx:end_idx].to(device)
                batch_labels = val_labels[batch_idx:end_idx].to(device)
                
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        # Calculate accuracies
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss / len(train_features))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_features))
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        if epoch % 10 == 0:
            print(f"📦 Epoch {epoch+1}/{epochs}")
            print(f"🔧 Train Loss: {train_loss/len(train_features):.4f} | 🎯 Train Acc: {train_acc:.4f}")
            print(f"🧪 Val Loss: {val_loss/len(val_features):.4f} | 🎯 Val Acc: {val_acc:.4f}")
            print(f"📚 Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
            break
    
    return best_val_acc, history

def evaluate_on_test(classifier, test_features, test_labels, device):
    """Evaluate the probe on test set."""
    classifier.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx in range(0, len(test_features), 128):
            end_idx = min(batch_idx + 128, len(test_features))
            batch_features = test_features[batch_idx:end_idx].to(device)
            batch_labels = test_labels[batch_idx:end_idx].to(device)
            
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            
            test_loss += loss.item() * batch_features.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    accuracy = correct / total
    print(f"\n📊 TEST RESULTS:")
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"📉 Test Loss: {test_loss/len(test_features):.4f}")
    
    return accuracy

def test_ssl_classification(method, model_path, probe_type='linear', epochs=100, lr=0.01):
    """Test SSL model classification performance."""
    print(f"🚀 Testing {method.upper()} {probe_type} classification")
    print(f"📦 Model path: {model_path}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")
    
    # Load model
    model = load_ssl_model(method, model_path, device)
    if model is None:
        return None
    
    model.to(device)
    
    # Data loaders
    train_dataset = CIFAR10Dataset('data/cifar10_splits/train_images.npy', 
                                  'data/cifar10_splits/train_labels.npy')
    val_dataset = CIFAR10Dataset('data/cifar10_splits/val_images.npy',
                                'data/cifar10_splits/val_labels.npy')
    test_dataset = CIFAR10Dataset('data/cifar10_splits/test_images.npy',
                                 'data/cifar10_splits/test_labels.npy')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Create and train probe
    feature_dim = train_features.shape[1]
    if probe_type == 'linear':
        classifier = create_linear_probe(feature_dim, 10).to(device)
        lr = 0.01  # Higher LR for linear probe
    else:
        classifier = create_nonlinear_probe(feature_dim, 10).to(device)
        lr = 0.001  # Lower LR for nonlinear probe
    
    # Train probe
    best_val_acc, history = train_probe(
        classifier, train_features, train_labels, val_features, val_labels,
        device, epochs=epochs, lr=lr, probe_type=probe_type
    )
    
    # Evaluate on test
    test_acc = evaluate_on_test(classifier, test_features, test_labels, device)
    
    # Save results
    results = {
        'method': method,
        'probe_type': probe_type,
        'model_path': model_path,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'epochs': epochs,
        'learning_rate': lr,
        'feature_dim': feature_dim
    }
    
    results_path = f'results/ssl_experiments/{method}_{probe_type}_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"🏁 {method.upper()} {probe_type} testing complete!")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"🧪 Test Accuracy: {test_acc:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test SSL models on CIFAR-10 classification')
    parser.add_argument('--method', choices=['mae', 'dino', 'ibot'], required=True,
                       help='SSL method to test')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--probe_type', choices=['linear', 'nonlinear'], default='linear',
                       help='Type of probe to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (auto-set if None)')
    
    args = parser.parse_args()
    
    # Auto-set learning rate based on probe type
    if args.lr is None:
        args.lr = 0.01 if args.probe_type == 'linear' else 0.001
    
    print(f"🚀 SSL Classification Testing")
    print(f"📊 Method: {args.method.upper()}")
    print(f"📊 Probe Type: {args.probe_type}")
    print(f"📊 Model: {args.model_path}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📊 Learning Rate: {args.lr}")
    
    # Test the model
    results = test_ssl_classification(
        args.method, args.model_path, args.probe_type, args.epochs, args.lr
    )
    
    if results:
        print(f"\n✅ Testing completed successfully!")
        print(f"🎯 Final Test Accuracy: {results['test_acc']:.4f}")
    else:
        print(f"\n❌ Testing failed!")

if __name__ == "__main__":
    main()
