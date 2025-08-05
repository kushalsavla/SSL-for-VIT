
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from models.vision_transformer import vit_small
from methods.dino import DINOLoss

# ======== Hyperparameters ========
EPOCHS = 200
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.04
OUT_DIM = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT_DIR = 'data/cifar10_splits'
NCROPS = 4
EMA_MOMENTUM = 0.996
SAVE_PATH = './dino_vit_cifar10-200.pth'

# ======== DINO Head ========
class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=384, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        for i in range(nlayers):
            inp_dim = in_dim if i == 0 else hidden_dim
            outp_dim = out_dim if i == nlayers - 1 else hidden_dim
            layers.append(nn.Linear(inp_dim, outp_dim, bias=False))
            if i < nlayers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ======== DINO Wrapper Model ========
class DinoModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(in_dim=384, out_dim=OUT_DIM)

    def forward(self, x):
        x = self.backbone(x)  # should return CLS token
        return self.head(x)

# ======== Dataset Loading ========
def load_tensor_split(split_dir, split_name):
    images = np.load(f"{split_dir}/{split_name}_images.npy")
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return images

class CIFAR10TensorDataset(Dataset):
    def __init__(self, tensor_images):
        self.images = tensor_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# ======== Augmentations for SSL ========
def get_cifar10_ssl_augmentations():
    to_pil = ToPILImage()

    transform_global = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    transform_local = transforms.Compose([
        transforms.RandomResizedCrop(16, scale=(0.05, 0.3)),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    def collate_fn(batch):
        imgs = [to_pil(img) for img in batch]
        batch_views = []
        for img in imgs:
            views = []
            for _ in range(2): views.append(transform_global(img))
            for _ in range(2): views.append(transform_local(img))
            batch_views.append(views)
        crops = list(zip(*batch_views))
        return [torch.stack(v) for v in crops]
    
    return collate_fn

# ======== Main Training Loop ========
def main():
    print("🚀 Starting DINO pretraining on CIFAR-10 train split...")
    train_images = load_tensor_split(SPLIT_DIR, 'train')
    ssl_dataset = CIFAR10TensorDataset(train_images)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=get_cifar10_ssl_augmentations()
    )

    # Build models
    student = DinoModel(vit_small(img_size=32, patch_size=4)).to(DEVICE)
    teacher = DinoModel(vit_small(img_size=32, patch_size=4)).to(DEVICE)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Loss and optimizer
    criterion = DINOLoss(out_dim=OUT_DIM, ncrops=NCROPS).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0

        for views in ssl_loader:
            views = [v.to(DEVICE) for v in views]
            student_outputs = [student(v) for v in views]
            with torch.no_grad():
                teacher_outputs = [teacher(v) for v in views[:2]]

            loss = criterion(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.data = t_param.data * EMA_MOMENTUM + s_param.data * (1 - EMA_MOMENTUM)

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(student.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
