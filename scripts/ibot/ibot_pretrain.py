#!/usr/bin/env python3

import os
import sys
import argparse

# Add iBOT repository to path if it exists
ibot_path = os.path.join(os.path.dirname(__file__), '../../external/ibot')
if os.path.exists(ibot_path):
    sys.path.append(ibot_path)
    print(f"✅ Added iBOT repository to path: {ibot_path}")
else:
    print(f"⚠️ iBOT repository not found at {ibot_path}")
    print("Please clone the official iBOT repository:")
    print("git clone https://github.com/bytedance/ibot.git external/ibot")
import math
import random
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import timm

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 as fallback"""
    def __init__(self):
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
        self.num_features = 256
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def get_args_parser():
    parser = argparse.ArgumentParser('Ultra Simple iBOT for CIFAR-10', add_help=False)
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch size')
    parser.add_argument('--out_dim', default=8192, type=int, help='Dimensionality of the DINO head output')
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help='Base EMA parameter for teacher update')
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.04, type=float, help='Weight decay')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    
    # Misc
    parser.add_argument('--data_path', default='../data/cifar10_splits/', type=str, help='Dataset path')
    parser.add_argument('--output_dir', default="./ultra_simple_output", help='Path to save logs and checkpoints')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    
    return parser

class UltraSimpleDataset:
    """Ultra simple dataset that loads CIFAR-10 .npy files without complex transforms"""
    def __init__(self, images_path, labels_path):
        print(f"Loading images from {images_path}")
        print(f"Loading labels from {labels_path}")
        
        # Load data
        self.images = np.load(images_path)  # (N, 32, 32, 3)
        self.labels = np.load(labels_path)  # (N,)
        
        print(f"Loaded {len(self.images)} images with shape {self.images.shape}")
        print(f"Loaded {len(self.labels)} labels")
        
        # No transform needed - we'll do normalization manually
        self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor directly without numpy issues
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, 32, 32)
        
        # Apply normalization manually
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = (img - mean) / std
        
        # Enhanced SSL augmentation - create two views
        img1 = self.apply_augmentation(img)
        img2 = self.apply_augmentation(img)
        
        return img1, img2, int(label)
    
    def apply_augmentation(self, img):
        """Apply strong augmentation for SSL."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
        
        # Random crop with padding
        pad = 4
        img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - 32)
        x = random.randint(0, w - 32)
        img = img[:, y:y+32, x:x+32]
        
        # Color jittering
        if random.random() > 0.5:
            # Brightness
            brightness_factor = random.uniform(0.8, 1.2)
            img = img * brightness_factor
            img = torch.clamp(img, 0, 1)
            
            # Contrast
            contrast_factor = random.uniform(0.8, 1.2)
            img = (img - img.mean()) * contrast_factor + img.mean()
            img = torch.clamp(img, 0, 1)
        
        # Grayscale (with probability)
        if random.random() > 0.8:
            gray = img.mean(dim=0, keepdim=True)
            img = gray.repeat(3, 1, 1)
        
        # Gaussian blur
        if random.random() > 0.7:
            # Simple blur approximation
            kernel = torch.ones(3, 3) / 9
            kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            img = torch.nn.functional.conv2d(img.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0)
        
        return img

class UltraSimpleDINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def ultra_simple_dino_loss(student_output, teacher_output, student_temp=0.1, center_momentum=0.9):
    """Ultra simple DINO loss"""
    
    # Initialize center if not exists
    if not hasattr(ultra_simple_dino_loss, 'center'):
        ultra_simple_dino_loss.center = torch.zeros(1, student_output.shape[-1]).cuda()
    
    # Compute teacher output
    teacher_cls = F.softmax((teacher_output - ultra_simple_dino_loss.center) / 0.04, dim=-1)
    teacher_cls = teacher_cls.detach()
    
    # Compute student output
    student_cls = F.log_softmax(student_output / student_temp, dim=-1)
    
    # Compute loss
    loss = F.kl_div(student_cls, teacher_cls, reduction='batchmean')
    
    # Update center
    with torch.no_grad():
        center = torch.sum(teacher_output, dim=0, keepdim=True)
        ultra_simple_dino_loss.center = ultra_simple_dino_loss.center * center_momentum + center * (1 - center_momentum)
    
    return loss

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Cosine scheduler for learning rate"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train_ultra_simple_ibot(args):
    """Ultra simple training function"""
    print("Starting Ultra Simple iBOT training for CIFAR-10")
    
    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dataset
    print("Loading dataset...")
    dataset = UltraSimpleDataset(
        os.path.join(args.data_path, 'train_images.npy'),
        os.path.join(args.data_path, 'train_labels.npy')
    )
    
    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Model
    print("Creating model...")
    
    # List available ViT models
    print("Available ViT models in timm:")
    vit_models = [name for name in timm.list_models() if 'vit' in name and 'small' in name]
    print(vit_models[:10])  # Show first 10
    
    try:
        # Try to create ViT-Small model for 32x32 images
        backbone = timm.create_model('vit_small_patch4_32', patch_size=4, num_classes=0, img_size=32)
        print("✅ Using vit_small_patch4_32 for 32x32 images")
    except Exception as e:
        print(f"❌ Failed to create vit_small_patch4_32: {e}")
        try:
            # Try standard ViT-Small
            backbone = timm.create_model('vit_small_patch16_224', patch_size=4, num_classes=0, img_size=32)
            print("✅ Using vit_small_patch16_224 with img_size=32")
        except Exception as e:
            print(f"❌ Failed to create vit_small_patch16_224: {e}")
            try:
                # Try ResNet as fallback
                print("Using ResNet18 as fallback...")
                backbone = timm.create_model('resnet18', num_classes=0)
            except:
                # Last resort: create a simple CNN
                print("Creating simple CNN...")
                backbone = SimpleCNN()
    
    embed_dim = backbone.num_features
    print(f"Model created with embed_dim: {embed_dim}")
    
    # Head
    head = UltraSimpleDINOHead(embed_dim, args.out_dim)
    
    # Student and teacher
    student = nn.Sequential(backbone, head)
    teacher = nn.Sequential(backbone, head)
    
    # Move to GPU
    student = student.cuda()
    teacher = teacher.cuda()
    
    # Teacher doesn't need gradients
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Copy student to teacher
    for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
        param_teacher.data.copy_(param_student.data)
    
    # Ensure student requires gradients
    for param in student.parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        student.train()
        teacher.train()
        
        total_loss = 0
        num_batches = 0
        
        for it, (images, labels) in enumerate(data_loader):
            # Update learning rate
            it = len(data_loader) * epoch + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]

            # Move images to gpu (now we have two views)
            images1, images2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
            
            # Get teacher and student outputs for both views
            with torch.no_grad():
                teacher_output1 = teacher(images1)
                teacher_output2 = teacher(images2)
            
            student_output1 = student(images1)
            student_output2 = student(images2)
            
            # Ensure student_output requires gradients
            if not student_output1.requires_grad:
                print("Warning: student_output1 doesn't require gradients!")
            
            # Compute loss for both views
            loss1 = ultra_simple_dino_loss(student_output1, teacher_output2)
            loss2 = ultra_simple_dino_loss(student_output2, teacher_output1)
            loss = (loss1 + loss2) / 2
            
            # Ensure loss requires gradients
            if not loss.requires_grad:
                print("Warning: loss doesn't require gradients!")
                # Force loss to require gradients by adding a small computation
                loss = loss + 0.0 * student_output1.sum()

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # Student update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update for the teacher
            with torch.no_grad():
                m = args.momentum_teacher
                for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
                    param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

            total_loss += loss.item()
            num_batches += 1
            
            if it % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {it}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
            }
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        
        # Log stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_loss:.4f}")
        print("-" * 50)
        
        # Save to log file
        log_stats = {
            'epoch': epoch,
            'loss': avg_loss,
            'epoch_time': epoch_time,
        }
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    # Save final model
    save_dict = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.epochs,
        'args': args,
    }
    torch.save(save_dict, os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(args.output_dir, 'final_model.pth')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ultra Simple iBOT for CIFAR-10', parents=[get_args_parser()])
    args = parser.parse_args()
    train_ultra_simple_ibot(args) 