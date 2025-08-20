#!/usr/bin/env python3
"""
ViT Fine-tuning Script
Simple script to fine-tune ViT models on CIFAR-10.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import math
import copy

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))

try:
    from models.vision_transformer import vit_small
    print("✅ Successfully imported DINO v2 ViT")
except ImportError:
    print("⚠️ DINO v2 ViT not available, falling back to timm")
    import timm

# Configuration
class Config:
    def __init__(self):
        # Training parameters - optimized to prevent memorization
        self.epochs = 100  # Reduced from 200 to prevent overfitting
        self.batch_size = 32  # Smaller batch size for better generalization
        self.learning_rate = 5e-5  # Even lower LR for fine-tuning
        self.weight_decay = 0.1  # Increased weight decay for regularization
        self.grad_clip = 1.0
        self.dropout = 0.3  # Increased dropout for regularization
        self.label_smoothing = 0.1  # Increased label smoothing
        
        # Anti-memorization parameters
        self.warmup_epochs = 5  # Shorter warmup
        self.min_lr = 1e-6  # Minimum learning rate
        self.ema_decay = 0.9995  # Slightly lower EMA for regularization
        self.early_stopping_patience = 15  # Early stopping to prevent overfitting
        
        # Additional regularization
        self.mixup_alpha = 0.1  # Reduced mixup strength
        self.cutmix_prob = 0.5  # Add CutMix augmentation
        self.cutmix_alpha = 1.0  # CutMix alpha parameter
        
        # Paths
        self.split_dir = 'data/cifar10_splits'
        self.models_dir = 'models/vit'
        self.results_dir = 'results/vit_experiments'
        self.save_path = os.path.join(self.models_dir, 'best_vit_finetuned.pth')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

# Data loading functions
def load_tensor_split(split_dir, split_name):
    """Load CIFAR-10 data splits."""
    images = np.load(f"{split_dir}/{split_name}_images.npy")
    labels = np.load(f"{split_dir}/{split_name}_labels.npy")
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def get_dataloaders(config):
    """Get training and validation dataloaders with enhanced augmentation."""
    train_images, train_labels = load_tensor_split(config.split_dir, 'train')
    val_images, val_labels = load_tensor_split(config.split_dir, 'val')

    # Enhanced data augmentation for better generalization
    train_transform = transforms.Compose([
        # Basic augmentations
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color augmentations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        
        # Advanced augmentations
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Normalization
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Validation transforms (no augmentation, just normalization)
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Apply transforms
    train_images = train_transform(train_images)
    val_images = val_transform(val_images)
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# Model creation
def create_vit_model(config, pretrained_path=None):
    """Create ViT model."""
    try:
        model = vit_small(img_size=32, patch_size=4)
        print("✅ Using DINO v2 ViT small")
    except:
        model = timm.create_model('vit_small_patch16_224', 
                                patch_size=4, 
                                num_classes=10,
                                img_size=32)
        print("✅ Using timm ViT small")

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"📦 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove the old head from state_dict since we're replacing it
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        
        # Load the backbone weights (excluding head)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"📦 Loaded pretrained backbone weights. Message: {msg}")
        
        # Now create the new head for CIFAR-10 - improved architecture
        model.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(model.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),  # Less dropout in final layers
            nn.Linear(256, 10)
        )
        print("✅ Created improved head: Dropout -> Linear(384->512) -> LayerNorm -> GELU -> Dropout -> Linear(512->256) -> LayerNorm -> GELU -> Dropout -> Linear(256->10)")
    else:
        # Create improved head if no pretrained weights
        model.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(model.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(256, 10)
        )
    
    return model

# Training functions
def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation for better regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    
    # Generate random box
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=0.1):
    """Mixup augmentation for better regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, optimizer, criterion, scaler, device, config):
    """Train for one epoch with enhanced augmentation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # Randomly choose between Mixup and CutMix for better regularization
        if np.random.random() < config.cutmix_prob:
            # Use CutMix
            x, y_a, y_b, lam = cutmix_data(x, y, alpha=config.cutmix_alpha)
            loss = cutmix_criterion(criterion, model(x), y_a, y_b, lam)
        else:
            # Use Mixup
            x, y_a, y_b, lam = mixup_data(x, y, alpha=config.mixup_alpha)
            loss = mixup_criterion(criterion, model(x), y_a, y_b, lam)
        
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy (approximate for augmentation)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (lam * (pred == y_a).sum().float() + (1 - lam) * (pred == y_b).sum().float()).item()
            total += y.size(0)
            total_loss += loss.item() * x.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader.dataset), correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(val_loader.dataset), correct / total

def train_model(model, train_loader, val_loader, config, device):
    """Main training loop."""
    print(f"🚀 Starting ViT fine-tuning...")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Improved learning rate scheduling with warmup
    def get_lr_scheduler(optimizer):
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                # Warmup phase: linear increase
                return epoch / config.warmup_epochs
            else:
                # Cosine annealing with minimum LR
                progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
                return config.min_lr + (1 - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler()
    
    # Exponential Moving Average for better generalization
    ema_model = None
    if config.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    patience = 20

    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Weight Decay: {config.weight_decay}")
    print(f"   Patience: {patience}")

    # Training loop
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update EMA model if enabled
        if ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(config.ema_decay).add_(param.data, alpha=1 - config.ema_decay)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        elapsed = time.time() - start_time
        
        print(f"\n📦 Epoch {epoch}/{config.epochs}")
        print(f"🔧 Train Loss: {train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f}")
        print(f"🧪 Val Loss: {val_loss:.4f} | 🎯 Val Acc: {val_acc:.4f}")
        print(f"📚 Learning Rate: {current_lr:.2e} | ⏱️ Time: {elapsed:.2f}s")
        
        # Save best model (use EMA model if available)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model = ema_model if ema_model is not None else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': vars(config),
                'history': history,
                'ema_enabled': ema_model is not None
            }, config.save_path)
            print(f"✅ New best model saved! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"🛑 Early stopping at epoch {epoch} (patience: {config.early_stopping_patience})")
            print(f"💡 This prevents overfitting and memorization!")
            break
    
    # Save training history
    history_path = os.path.join(config.results_dir, 'vit_finetune_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved to {history_path}")
    
    return best_val_acc, history

# Main function
def main():
    parser = argparse.ArgumentParser(description='ViT Fine-tuning on CIFAR-10')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides default)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting ViT Fine-tuning")
    
    # Create configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")
    
    # Create model
    model = create_vit_model(config, args.pretrained)
    model.to(device)
    
    # Train model
    best_val_acc, history = train_model(model, train_loader, val_loader, config, device)
    
    print(f"\n🏁 Training finished!")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved to: {config.save_path}")
    print(f"📊 Results saved to: {config.results_dir}")

if __name__ == "__main__":
    main()
