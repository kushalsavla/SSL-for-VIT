#!/usr/bin/env python3
"""
Fine-tuning for DINO-pretrained ViT-Small on CIFAR-10
Unfreezes the backbone and trains the entire model end-to-end.
Adapted for friend's DINO pretraining setup.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
import argparse
from tqdm import tqdm
import time

# Import the same modules used in DINO pretraining
# Import vision transformer from external DINO v2 repository
# The vision_transformer.py is located in external/dino/dinov2/models/vision_transformer.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))
from models.vision_transformer import vit_small

# ======== Hyperparameters ========
EPOCHS = 100
BATCH_SIZE = 128
BACKBONE_LR = 5e-5
CLASSIFIER_LR = 5e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT_DIR = 'data/cifar10_splits'
SAVE_PATH = './dino_fine_tuned_model.pth'

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader with data augmentation for fine-tuning."""
    def __init__(self, images_path, labels_path, is_training=True):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.is_training = is_training
        
        # Data augmentation for training
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Apply transforms
        img = self.transform(img)
        
        return img, int(label)

class FineTunedDinoModel(nn.Module):
    """Fine-tuned DINO model with classification head."""
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        # Remove the DINO head and add classification head
        self.classifier = nn.Linear(384, num_classes)  # 384 is the feature dimension from ViT-Small
        
    def forward(self, x):
        # Get features from backbone (CLS token)
        features = self.backbone(x)  # Should return CLS token of shape [batch_size, 384]
        features = self.backbone(x)
        assert features.shape[-1] == 384, f"Unexpected feature dim: {features.shape}"

        return self.classifier(features)

def load_dino_pretrained_model(model_path, device):
    """Load the DINO-pretrained model."""
    print(f"Loading DINO pretrained model from {model_path}")
    
    # Create the same backbone architecture as used in DINO training
    backbone = vit_small(img_size=32, patch_size=4)
    print("✅ Using vit_small with img_size=32, patch_size=4")
    
    # Load pretrained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load the backbone weights (excluding the DINO head)
        backbone_state = {}
        for key, value in checkpoint.items():
            # Skip the DINO head parameters (they start with 'head.')
            if not key.startswith('head.'):
                backbone_state[key] = value

        print(f"Loaded {len(backbone_state)} backbone weights (excluding head)")

        backbone.load_state_dict(backbone_state, strict=False)
        print(f"✅ Loaded DINO backbone weights from {model_path}")
    else:
        print(f"⚠️ Model path {model_path} not found, using random weights")
    
    return backbone

def train_fine_tuned_model(model, train_loader, val_loader, device, args):
    """Train the fine-tuned model."""
    print("Training fine-tuned DINO model...")
    
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for backbone and classifier
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': classifier_params, 'lr': args.classifier_lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_dir + '/best_dino_fine_tuned_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return model, best_acc, train_losses, val_accuracies

def evaluate_on_test(model, test_loader, device):
    """Evaluate the fine-tuned model on test set."""
    print("Evaluating on test set...")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    
    # Detailed classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=CIFAR10_CLASSES, digits=4)
    
    return test_acc, report

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning for DINO-pretrained ViT')
    parser.add_argument('--data_path', type=str, default='data/cifar10_splits/',
                       help='Path to CIFAR-10 splits')
    parser.add_argument('--model_path', type=str, default='./dino_vit_cifar10-200.pth',
                       help='Path to DINO pretrained model')
    parser.add_argument('--output_dir', type=str, default='./dino_fine_tune_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--backbone_lr', type=float, default=5e-5, help='Learning rate for backbone')
    parser.add_argument('--classifier_lr', type=float, default=5e-4, help='Learning rate for classifier')
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
        os.path.join(args.data_path, 'train_images.npy'),
        os.path.join(args.data_path, 'train_labels.npy'),
        is_training=True
    )
    val_dataset = CIFAR10Dataset(
        os.path.join(args.data_path, 'val_images.npy'),
        os.path.join(args.data_path, 'val_labels.npy'),
        is_training=False
    )
    test_dataset = CIFAR10Dataset(
        os.path.join(args.data_path, 'test_images.npy'),
        os.path.join(args.data_path, 'test_labels.npy'),
        is_training=False
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Load DINO pretrained model
    backbone = load_dino_pretrained_model(args.model_path, device)
    
    # Create fine-tuned model
    model = FineTunedDinoModel(backbone, num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    start_time = time.time()
    model, best_val_acc, train_losses, val_accuracies = train_fine_tuned_model(
        model, train_loader, val_loader, device, args
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_acc, test_report = evaluate_on_test(model, test_loader, device)
    
    # Print summary
    print("\n" + "="*50)
    print("DINO FINE-TUNING RESULTS")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Backbone LR: {args.backbone_lr}")
    print(f"Classifier LR: {args.classifier_lr}")
    print("\nTest Set Classification Report:")
    print(test_report)
    
    # Save the best fine-tuned model
    best_model_path = os.path.join(args.output_dir, 'best_dino_fine_tuned_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'epochs': args.epochs,
        'backbone_lr': args.backbone_lr,
        'classifier_lr': args.classifier_lr,
        'training_time': training_time
    }, best_model_path)
    print(f"💾 Best DINO fine-tuned model saved to: {best_model_path}")
    
    # Save detailed results
    with open(os.path.join(args.output_dir, 'dino_fine_tune_results.txt'), 'w') as f:
        f.write("DINO FINE-TUNING RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Training Epochs: {args.epochs}\n")
        f.write(f"Backbone LR: {args.backbone_lr}\n")
        f.write(f"Classifier LR: {args.classifier_lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Best Model Path: {best_model_path}\n")
        f.write("\nTest Set Classification Report:\n")
        f.write(test_report)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("DINO fine-tuning completed!")

if __name__ == '__main__':
    main()
