#!/usr/bin/env python3
"""
Non-Linear Probing for SSL-pretrained ViT-Small on CIFAR-10
Evaluates the quality of learned representations by training an MLP classifier on frozen features.
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
import sys
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import vision transformer from external DINO v2 repository
# The vision_transformer.py is located in external/dino/dinov2/models/vision_transformer.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))
from models.vision_transformer import vit_small

class CIFAR10Dataset(Dataset):
    """Simple CIFAR-10 dataset loader for non-linear probing."""
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

class MLPProbe(nn.Module):
    """MLP classifier on top of frozen SSL features."""
    def __init__(self, feature_dim, num_classes, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = feature_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)

def load_pretrained_model(model_path, device):
    
    """Load the SSL-pretrained model and freeze it."""
    print("Loading pretrained model...")

    backbone = timm.create_model('vit_small_patch16_224', patch_size=4, num_classes=0, img_size=32)
    print(" Created ViT-Small model with patch_size=4 and img_size=32")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        if "backbone.backbone.patch_embed.proj.weight" in checkpoint:
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("backbone."):
                    new_key = k[len("backbone."):].replace("backbone.", "")  
                    new_state_dict[new_key] = v
            backbone.load_state_dict(new_state_dict, strict=False)
            print(f" Loaded backbone weights from DinoModel checkpoint: {model_path}")
        else:
            backbone.load_state_dict(checkpoint, strict=False)
            print(f" Loaded model weights directly from: {model_path}")

    else:
        print(f"  Model path not found: {model_path}, using random weights")

    # Freeze the backbone
    backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


def extract_features(model, dataloader, device):
    """Extract features from the frozen model."""
    model.to(device)
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
            data, target = data.to(device), target.to(device)
            feat = model(data)
            features.append(feat.cpu())
            labels.append(target.cpu())
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels

def train_mlp_probe(train_features, train_labels, val_features, val_labels, 
                   feature_dim, num_classes, device, args):
    """Train MLP classifier on extracted features."""
    print("Training MLP classifier...")
    
    # Create MLP probe
    mlp_probe = MLPProbe(feature_dim, num_classes, hidden_dims=args.hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Convert to tensors
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        # Training
        mlp_probe.train()
        optimizer.zero_grad()
        
        outputs = mlp_probe(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # Validation
        mlp_probe.eval()
        with torch.no_grad():
            val_outputs = mlp_probe(val_features)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            val_accuracies.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(mlp_probe.state_dict(), args.output_dir + '/best_mlp_probe.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
    
    return mlp_probe, best_acc, train_losses, val_accuracies

def evaluate_on_test(mlp_probe, test_features, test_labels, device):
    """Evaluate the trained MLP probe on test set."""
    print("Evaluating on test set...")
    
    mlp_probe.eval()
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        test_outputs = mlp_probe(test_features)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = accuracy_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
        
        # Detailed classification report
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        report = classification_report(test_labels.cpu().numpy(), test_preds.cpu().numpy(), 
                                     target_names=class_names, digits=4)
    
    return test_acc, report

def main():
    parser = argparse.ArgumentParser(description='Non-Linear Probing for SSL-pretrained ViT')
    parser.add_argument('--data_path', type=str, default='../data/cifar10_splits/',
                       help='Path to CIFAR-10 splits')
    parser.add_argument('--model_path', type=str, default='../dino_vit_cifar10-200.pth',
                       help='Path to pretrained model (default: best fine-tuned model)')
    parser.add_argument('--output_dir', type=str, default='./nonlinear_probe_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden dimensions for MLP')
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
        os.path.join(args.data_path, 'train_images.npy'),
        os.path.join(args.data_path, 'train_labels.npy')
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
                            shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Load pretrained model
    backbone = load_pretrained_model(args.model_path, device)
    
    # Extract features
    print("Extracting features from training set...")
    train_features, train_labels = extract_features(backbone, train_loader, device)
    
    print("Extracting features from validation set...")
    val_features, val_labels = extract_features(backbone, val_loader, device)
    
    print("Extracting features from test set...")
    test_features, test_labels = extract_features(backbone, test_loader, device)
    
    print(f"Feature dimensions: {train_features.shape}")
    print(f"Number of classes: {len(torch.unique(train_labels))}")
    print(f"MLP architecture: {args.hidden_dims}")

    train_features = nn.functional.normalize(train_features, dim=1)
    val_features = nn.functional.normalize(val_features, dim=1)
    test_features = nn.functional.normalize(test_features, dim=1)
    
    # Train MLP probe
    feature_dim = train_features.shape[1]
    num_classes = 10
    
    start_time = time.time()
    mlp_probe, best_val_acc, train_losses, val_accuracies = train_mlp_probe(
        train_features, train_labels, val_features, val_labels,
        feature_dim, num_classes, device, args
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_acc, test_report = evaluate_on_test(mlp_probe, test_features, test_labels, device)
    
    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_report': test_report
    }
    
    # Print summary
    print("\n" + "="*50)
    print("NON-LINEAR PROBING RESULTS")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Feature Dimension: {feature_dim}")
    print(f"MLP Architecture: {args.hidden_dims}")
    print("\nTest Set Classification Report:")
    print(test_report)
    
    # Save detailed results
    with open(os.path.join(args.output_dir, 'nonlinear_probe_results.txt'), 'w') as f:
        f.write("NON-LINEAR PROBING RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Feature Dimension: {feature_dim}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Training Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"MLP Architecture: {args.hidden_dims}\n")
        f.write("\nTest Set Classification Report:\n")
        f.write(test_report)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("Non-linear probing completed!")

if __name__ == '__main__':
    main()