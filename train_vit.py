import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.vision_transformer import vit_base
import sys
import os

# AMP support
from torch.cuda.amp import autocast, GradScaler

# Include DINOv2 layers if used internally
sys.path.append("/work/dlclarge2/savlak-sslViT/dinov2")
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

# --------- Config ---------
EPOCHS = 50
BATCH_SIZE = 64   # Lower to avoid OOM
LR = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
SAVE_PATH = './best_vit_model.pth'

# --------- DataLoader ---------
def get_dataloaders(split_dir='./data/cifar10_splits', batch_size=128, verbose=False):
    train_images = np.load(f"{split_dir}/train_images.npy")
    train_labels = np.load(f"{split_dir}/train_labels.npy")
    val_images = np.load(f"{split_dir}/val_images.npy")
    val_labels = np.load(f"{split_dir}/val_labels.npy")

    train_images = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    val_images = torch.tensor(val_images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    if verbose:
        print(f"Train: {train_images.shape}, Val: {val_images.shape}")

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size)

    return train_loader, val_loader

# --------- Training ---------
def main():
    print("🚀 Starting training script...", flush=True)

    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, verbose=True)
    print("✅ Data loaded.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Using device: {device}")

    model = vit_base(img_size=32, patch_size=4)
    model.head = nn.Linear(model.embed_dim, 10)  # CIFAR-10
    model.to(device)
    print("✅ Model initialized and moved to device.")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📦 Epoch {epoch}/{EPOCHS}")

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
        print(f"🔧 Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"🎯 Val Acc: {val_acc:.4f}")
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Best model saved (Val Acc: {val_acc:.4f})")

    print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
