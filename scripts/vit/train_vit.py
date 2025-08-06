import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from functools import partial
from torch.cuda.amp import autocast, GradScaler
import argparse

# Path config (adjust to your folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for models
sys.path.append("/work/dlclarge2/savlak-sslViT/dinov2")  # for custom dinov2 code

# Import vision transformer from external DINO v2 repository
# The vision_transformer.py is located in external/dino/dinov2/models/vision_transformer.py
# This avoids duplication and ensures we use the official DINO v2 implementation
# Navigation: external/dino/ -> dinov2/ -> models/ -> vision_transformer.py
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))
from models.vision_transformer import vit_small

# --------- Config ---------
EPOCHS = 200
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
SAVE_PATH = './best_vit_small_model.pth'
SPLIT_DIR = '../data/cifar10_splits'

# --------- Data ---------
class CIFAR10TensorAugment:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

    def __call__(self, x):
        return self.augment(x)

class AugmentedDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = CIFAR10TensorAugment() if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.augment:
            x = self.augment(x)
        return x, y

def load_tensor_split(split_dir, split_name):
    images = np.load(f"{split_dir}/{split_name}_images.npy")
    labels = np.load(f"{split_dir}/{split_name}_labels.npy")
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def get_dataloaders(split_dir, batch_size=128, mode='supervised'):
    if mode in ['linear_probe', 'mlp_probe']:
        train_images, train_labels = load_tensor_split(split_dir, 'probe')
        print('📌 Using probe split for probing mode.')
    else:
        train_images, train_labels = load_tensor_split(split_dir, 'train')
        print('📌 Using train split for supervised mode.')
    val_images, val_labels = load_tensor_split(split_dir, 'val')

    train_dataset = AugmentedDataset(train_images, train_labels, augment=True)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

# --------- Training ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='supervised', choices=['supervised', 'linear_probe', 'mlp_probe'])
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    print(f"🚀 Starting ViT training on CIFAR-10 in {args.mode} mode")

    train_loader, val_loader = get_dataloaders(SPLIT_DIR, batch_size=BATCH_SIZE, mode=args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Model
    model = vit_small(img_size=32, patch_size=4)

    # Custom heads
    if args.mode == 'mlp_probe':
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    else:
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.embed_dim, 10)
        )
    model.to(device)
    print("✅ Model initialized with patch size 4.")

    # Pretrained weights
    if args.pretrained and os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"📦 Loaded pretrained backbone: {args.pretrained}")
        print(f"ℹ️ Load message: {msg}")

    # Freeze backbone for probes
    if args.mode in ['linear_probe', 'mlp_probe']:
        for name, param in model.named_parameters():
            param.requires_grad = 'head' in name
        print("🧊 Backbone frozen. Only head is trainable.")

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_acc = 0.0

    # -------- Training Loop --------
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        elapsed = time.time() - start_time

        print(f"\n📦 Epoch {epoch}/{EPOCHS}")
        print(f"🔧 Train Loss: {avg_loss:.4f} | 🎯 Val Acc: {val_acc:.4f} | ⏱️ Time: {elapsed:.2f}s")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ New best model saved! (Val Acc: {val_acc:.4f})")

    print(f"\n🏁 Training finished. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
