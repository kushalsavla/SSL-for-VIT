import numpy as np
import pickle
import os

def load_cifar10_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
    labels = np.array(batch[b'labels'])
    return images, labels

def prepare_cifar10_splits(data_dir='./data/cifar-10-batches-py', seed=42):
    # Load all training batches
    images_list, labels_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar10_batch(batch_file)
        images_list.append(images)
        labels_list.append(labels)
    all_train_images = np.concatenate(images_list)
    all_train_labels = np.concatenate(labels_list)
    # Shuffle and split train/val
    np.random.seed(seed)
    indices = np.random.permutation(len(all_train_images))
    split = int(0.9 * len(all_train_images))
    train_idx, val_idx = indices[:split], indices[split:]
    train_images, val_images = all_train_images[train_idx], all_train_images[val_idx]
    train_labels, val_labels = all_train_labels[train_idx], all_train_labels[val_idx]
    # Probe set: 20% of train split
    probe_split = int(0.2 * len(train_images))
    probe_idx = np.random.choice(len(train_images), probe_split, replace=False)
    probe_images, probe_labels = train_images[probe_idx], train_labels[probe_idx]
    # Test set
    test_images, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'probe_images': probe_images,
        'probe_labels': probe_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

def save_splits(splits, save_dir='./data/cifar10_splits'):
    os.makedirs(save_dir, exist_ok=True)
    for k, v in splits.items():
        np.save(os.path.join(save_dir, f"{k}.npy"), v)
    print(f"Splits saved to {save_dir}")

if __name__ == "__main__":
    splits = prepare_cifar10_splits()
    for k, v in splits.items():
        print(f"{k}: {v.shape}")
    save_splits(splits)