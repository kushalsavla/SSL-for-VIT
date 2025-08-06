#!/usr/bin/env python3
"""
Simple script to create DINO qualitative analysis visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# DINO results from our analysis
model_type = "dino"
image_path = "results/final_evaluation/test_airplane.png"
image_name = "test_airplane"
top_probs = [0.848, 0.131, 0.016, 0.003, 0.003]
top_indices = [0, 2, 9, 3, 8]  # airplane, bird, truck, cat, ship
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the actual image
if os.path.exists(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
else:
    print(f"⚠️ Image not found: {image_path}")
    image_array = None

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Display actual image
if image_array is not None:
    ax1.imshow(image_array)
    ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
else:
    ax1.text(0.5, 0.5, f'Input Image:\n{os.path.basename(image_path)}\n(Not Found)', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=14)
    ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
ax1.axis('off')

# Display predictions
y_pos = np.arange(len(top_indices))
bars = ax2.barh(y_pos, top_probs)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([CIFAR10_CLASSES[i] for i in top_indices])
ax2.set_xlabel('Probability')
ax2.set_title(f'Top 5 Predictions ({model_type.upper()} Model)')
ax2.invert_yaxis()

# Add probability values on bars
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center')

plt.tight_layout()

# Save plot
output_path = f'qualitative_results_{model_type}_{image_name}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Results saved to: {output_path}")

plt.close()
print("✅ DINO visualization created successfully!") 