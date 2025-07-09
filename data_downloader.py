import numpy as np
import torchvision
import torchvision.transforms as transforms

def download_cifar10(data_dir='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    print(f"CIFAR-10 downloaded to {data_dir}")

if __name__ == "__main__":
    download_cifar10()