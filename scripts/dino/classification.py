#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
import os
from models.vision_transformer import vit_small
from PIL import Image

# CIFAR-10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class FineTunedDinoModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(384, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_model(model_path, device):
    print(f"Loading fine-tuned DINO model from {model_path}")
    backbone = vit_small(img_size=32, patch_size=4)
    model = FineTunedDinoModel(backbone, num_classes=10)
    model.to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded full model (with classifier) from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded raw state_dict into full model.")
    else:
        print(f"⚠️ Model path {model_path} not found, using randomly initialized weights")

    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0) 

def classify_single_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()

def main():
    parser = argparse.ArgumentParser(description='Classify test_images.npy with fine-tuned DINO model')
    parser.add_argument('--model_path', type=str, default='dino_fine_tune_results_copy/best_dino_fine_tuned_model.pth',
                        help='Path to the fine-tuned DINO model (.pth)')
    parser.add_argument('--image_path', type=str, default='airplane.jpeg',
                        help='Path to the image file (e.g., plane.png)')
    parser.add_argument('--output_file', type=str, default='dino_classification_test_results.txt',
                        help='Output file to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)
    print("Loaded fine-tuned model.")


    image_tensor = preprocess_image(args.image_path)
    pred, conf = classify_single_image(model, image_tensor, device)

    with open(args.output_file, 'w') as f:
        f.write("DINO Test Results\n")
        f.write("="*40 + "\n")
        f.write(f"✅ Prediction: {CIFAR10_CLASSES[pred]}\n")
        f.write(f"🔍 Confidence: {conf*100:.2f}%\n")
        f.write("="*40)


        print("\n" + "="*40)
        print(f"✅ Prediction: {CIFAR10_CLASSES[pred]}\n")
        print(f"🔍 Confidence: {conf*100:.2f}%\n")
        print("="*40)

if __name__ == '__main__':
    main()
