import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.vision_transformer import vit_base
import sys
sys.path.append("/work/dlclarge2/savlak-sslViT/dinov2")
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


def get_dataloaders(split_dir='./data/cifar10_splits', batch_size=128):
    train_images = np.load(f"{split_dir}/train_images.npy")
    train_labels = np.load(f"{split_dir}/train_labels.npy")
    val_images = np.load(f"{split_dir}/val_images.npy")
    val_labels = np.load(f"{split_dir}/val_labels.npy")
    # Convert to torch tensors and normalize to [0,1]
    train_images = torch.tensor(train_images).permute(0, 3, 1, 2).float() / 255.
    val_images = torch.tensor(val_images).permute(0, 3, 1, 2).float() / 255.
    train_labels = torch.tensor(train_labels).long()
    val_labels = torch.tensor(val_labels).long()
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size)
    return train_loader, val_loader

def main():
    train_loader, val_loader = get_dataloaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For CIFAR-10, use patch_size=4 to get 8x8 patches from 32x32 images
    model = vit_base(img_size=32, patch_size=4)
    model.head = nn.Linear(model.embed_dim, 10)  # Replace identity head with classifier
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

if __name__ == "__main__":
    main()