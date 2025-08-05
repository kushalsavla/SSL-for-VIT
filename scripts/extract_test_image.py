#!/usr/bin/env python3
"""
Extract Test Image from CIFAR-10 Dataset
========================================

This script extracts a sample image from our CIFAR-10 dataset for qualitative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def extract_sample_image(data_path, class_idx=None, sample_idx=0, output_path="test_image.png"):
    """Extract a sample image from CIFAR-10 dataset."""
    
    # Load test data
    test_images = np.load(os.path.join(data_path, 'test_images.npy'))
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))
    
    print(f"Loaded {len(test_images)} test images")
    print(f"Image shape: {test_images.shape}")
    print(f"Label shape: {test_labels.shape}")
    
    # Find image
    if class_idx is not None:
        # Find images of specific class
        class_indices = np.where(test_labels == class_idx)[0]
        if len(class_indices) == 0:
            print(f"No images found for class {class_idx} ({CIFAR10_CLASSES[class_idx]})")
            return None
        image_idx = class_indices[sample_idx % len(class_indices)]
    else:
        # Use random image
        image_idx = sample_idx % len(test_images)
    
    # Extract image and label
    image = test_images[image_idx]
    label = test_labels[image_idx]
    
    print(f"Selected image {image_idx}")
    print(f"True class: {label} ({CIFAR10_CLASSES[label]})")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(image)
    pil_image.save(output_path)
    
    print(f"✅ Image saved to: {output_path}")
    
    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Test Image: {CIFAR10_CLASSES[label]} (Class {label})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"test_image_display.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract test image from CIFAR-10')
    parser.add_argument('--data_path', type=str, default='data/cifar10_splits/',
                       help='Path to CIFAR-10 data directory')
    parser.add_argument('--class_idx', type=int, default=None,
                       help='Class index to extract (0-9)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index within the class')
    parser.add_argument('--output_path', type=str, default='test_image.png',
                       help='Output path for the extracted image')
    
    args = parser.parse_args()
    
    # Print available classes
    print("Available CIFAR-10 classes:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        print(f"  {i}: {class_name}")
    print()
    
    # Extract image
    output_path = extract_sample_image(
        args.data_path, 
        args.class_idx, 
        args.sample_idx, 
        args.output_path
    )
    
    if output_path:
        print(f"\n🎯 You can now use this image for qualitative analysis:")
        print(f"   sbatch qualitative_analysis.sh {output_path} vit")
        print(f"   sbatch qualitative_analysis.sh {output_path} ibot")

if __name__ == '__main__':
    main() 