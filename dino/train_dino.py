import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add DINOv2 to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dinov2.models.vision_transformer import vit_base
from dinov2.train import DINOTrainer  # Assuming DINOTrainer exists in your DINOv2 codebase

# --------- Config ---------
EPOCHS = 50  # For quick test, increase for real run
BATCH_SIZE = 32
LR = 3e-4
SAVE_PATH = './dino_vit_cifar10.pth'
SPLIT_DIR = '../data/cifar10_splits'

# --------- Augmentations ---------
class CIFAR10TensorAugment:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ])

    def __call__(self, x):
        return self.augment(x)

class AugmentedDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.augment = CIFAR10TensorAugment()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = self.augment(x)
        return x

def load_tensor_split(split_dir, split_name):
    images = np.load(f"{split_dir}/{split_name}_images.npy")
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return images

def main():
    print("🚀 Starting DINO pretraining on CIFAR-10 train split...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Using device: {device}")

    # Load train split (no labels)
    train_images = load_tensor_split(SPLIT_DIR, 'train')
    train_dataset = AugmentedDataset(train_images)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize ViT backbone
    model = vit_base(img_size=32, patch_size=4)
    model.to(device)
    print("✅ ViT backbone initialized.")

    # Set up DINO trainer (replace with your DINOv2 training logic)
    trainer = DINOTrainer(model, train_loader, device=device, epochs=EPOCHS, lr=LR)

    # Train
    trainer.train()

    # Save backbone weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ DINO-pretrained backbone saved to {SAVE_PATH}")

if __name__ == "__main__":
    main() 