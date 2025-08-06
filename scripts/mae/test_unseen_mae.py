#!/usr/bin/env python3
"""
Evaluate MAE-trained ViT (vit_tiny_patch4_32) on unseen CIFAR-10 test data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Import your MAE ViT model constructor
from models_vit import vit_tiny_patch4_32


class CIFAR10TestDataset(Dataset):
    """Dataset for testing on unseen CIFAR-10 data."""
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        print(f"Loaded {len(self.images)} test images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor and normalize
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = (img - mean) / std

        return img, int(label)


def load_mae_vit_model(model_path, device, num_classes=10, global_pool=False):
    """Load MAE-pretrained ViT classifier for CIFAR-10 evaluation."""
    print(f"Loading MAE ViT model from {model_path}")

    model = vit_tiny_patch4_32(num_classes=num_classes, global_pool=global_pool)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Remove classification head if shape mismatches (likely due to different num_classes)
        model_state = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in state_dict and k in model_state and state_dict[k].shape != model_state[k].shape:
                print(f"Removing key {k} from checkpoint due to shape mismatch (expected classifier)")
                del state_dict[k]

        load_msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {load_msg}")
        print("✅ MAE ViT model loaded successfully.")
    else:
        print(f"⚠️  Model checkpoint not found at: {model_path}")
        return None

    model.to(device)
    model.eval()

    return model


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model performance on CIFAR-10 test set."""
    print(f"\n🔬 Evaluating {model_name} on test set...")

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    start_time = time.time()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Testing {model_name}"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # handle tuple outputs if any
            _, preds = torch.max(outputs, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    elapsed = time.time() - start_time

    accuracy = 100. * correct / total
    report = classification_report(
        all_labels, all_preds,
        target_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck'],
        digits=3
    )

    print(f"\n📊 {model_name} Test Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    print(f"   Test time: {elapsed:.2f} seconds")
    print(f"   Throughput: {total/elapsed:.1f} images/second")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'test_time': elapsed,
        'throughput': total / elapsed,
        'predictions': all_preds,
        'labels': all_labels,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE ViT on unseen CIFAR-10 test data")
    parser.add_argument('--data_path', default='./data/cifar10_splits/',
                        help='Directory containing test_images.npy and test_labels.npy')
    parser.add_argument('--model_path', required=True,
                        help='Path to MAE ViT checkpoint file (e.g. checkpoint-49.pth)')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--global_pool', action='store_true',
                        help='Set this if global pooling was used during training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dataset = CIFAR10TestDataset(
        os.path.join(args.data_path, 'test_images.npy'),
        os.path.join(args.data_path, 'test_labels.npy')
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_mae_vit_model(args.model_path, device, global_pool=args.global_pool)
    if model is None:
        print("❌ Model loading failed.")
        return

    results = evaluate_model(model, test_loader, device, "MAE ViT")

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"MAE ViT        Accuracy: {results['accuracy']:.2f}%  Time: {results['test_time']:.2f}s  Throughput: {results['throughput']:.1f}/s")

    with open('final_test_results.txt', 'w') as f:
        f.write("FINAL TEST RESULTS ON UNSEEN DATA\n")
        f.write("="*50 + "\n\n")
        f.write(f"MAE ViT RESULTS:\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Correct: {results['correct']}/{results['total']}\n")
        f.write(f"Test time: {results['test_time']:.2f} seconds\n")
        f.write(f"Throughput: {results['throughput']:.1f} images/second\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
        f.write("\n" + "-"*50 + "\n\n")

    print(f"\n📄 Detailed results saved to: final_test_results.txt\n✅ Testing completed!")


if __name__ == '__main__':
    main()
