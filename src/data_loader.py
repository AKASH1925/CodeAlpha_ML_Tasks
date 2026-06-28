"""
Data Loading and Preprocessing for EMNIST and MNIST Datasets
=============================================================
Handles downloading, loading, and preprocessing of handwritten
character datasets. Primary focus: EMNIST Letters (A-Z, 26 classes).
Also supports:
- MNIST (digits 0-9)
- EMNIST Balanced (47 classes)
- EMNIST ByClass (62 classes)
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# EMNIST ByClass has 62 classes (0-9, A-Z, a-z)
# EMNIST Letters has 26 classes (A-Z)
EMNIST_LETTER_MAP = {i: chr(ord("A") + i) for i in range(26)}
def get_transforms(augment=False):
    """
    Get data transforms for training and testing.
    Args:
        augment (bool): Whether to apply data augmentation
    Returns:
        tuple: (train_transform, test_transform)
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, test_transform
def load_mnist(data_dir="./data", batch_size=64, augment=True):
    """
    Load MNIST dataset (digits 0-9).
    Args:
        data_dir (str): Directory to store downloaded data
        batch_size (int): Batch size for data loaders
        augment (bool): Whether to apply data augmentation
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    print("Loading MNIST dataset...")
    train_transform, test_transform = get_transforms(augment)
    # Download and load training set
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    # Download and load test set
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    # Split training into train + validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    class_names = [str(i) for i in range(10)]
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, class_names
def load_emnist(data_dir="./data", batch_size=64, split="letters", augment=True):
    """
    Load EMNIST dataset (characters).
    Args:
        data_dir (str): Directory to store downloaded data
        batch_size (int): Batch size for data loaders
        split (str): EMNIST split type - 'letters', 'digits', 'byclass', 'balanced'
        augment (bool): Whether to apply data augmentation
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    print(f"Loading EMNIST dataset (split: {split})...")
    train_transform, test_transform = get_transforms(augment)
    # Download and load training set
    train_dataset = datasets.EMNIST(
        root=data_dir,
        split=split,
        train=True,
        download=True,
        transform=train_transform,
    )
    # Download and load test set
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split=split,
        train=False,
        download=True,
        transform=test_transform,
    )
    # Split training into train + validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    # Generate class names based on split
    if split == "letters":
        class_names = [chr(ord("A") + i) for i in range(26)]
        num_classes = 26
    elif split == "digits":
        class_names = [str(i) for i in range(10)]
        num_classes = 10
    elif split == "byclass":
        class_names = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)] + [
            chr(ord("a") + i) for i in range(26)
        ]
        num_classes = 62
    else:  # balanced
        class_names = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)] + [
            chr(ord("a") + i) for i in range(26)
        ]
        num_classes = 47
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, class_names
def get_sample_batch(data_loader, num_samples=16):
    """
    Get a sample batch from the data loader.
    Args:
        data_loader: PyTorch DataLoader
        num_samples (int): Number of samples to retrieve
    Returns:
        tuple: (images, labels)
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    return images[:num_samples], labels[:num_samples]
def preprocess_image(image):
    """
    Preprocess a single image for inference.
    Expects a PIL Image or numpy array.
    Args:
        image: PIL Image or numpy array of shape (H, W)
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, 28, 28)
    """
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor