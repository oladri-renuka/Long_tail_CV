"""
ISIC Skin Lesion Dataset loader with long-tail analysis.
Handles ISIC 2019 Challenge data format.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm


class ISICDataset(Dataset):
    """ISIC 2019 Skin Lesion Dataset."""

    # ISIC 2019 disease types
    DISEASE_TYPES = [
        'MEL',    # Melanoma (0)
        'NV',     # Nevus (1)
        'BCC',    # Basal Cell Carcinoma (2)
        'AK',     # Actinic Keratosis (3)
        'BKL',    # Benign Keratosis-like (4)
        'DF',     # Dermatofibroma (5)
        'VASC',   # Vascular Lesion (6)
        'SCC',    # Squamous Cell Carcinoma (7)
    ]

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        transform=None,
        class_type: str = "all",
        head_threshold: int = 1,  # Top 1 class is head
        tail_threshold: int = 4,  # Bottom 4 classes are tail
    ):
        """
        Args:
            data_dir: Directory containing ISIC_2019_Training_Input/ folder
            csv_path: Path to ISIC_2019_Training_GroundTruth.csv
            transform: Image transformations
            class_type: "all", "head", or "tail"
            head_threshold: Top N classes are head
            tail_threshold: Bottom N classes are tail
        """
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.class_type = class_type
        self.head_threshold = head_threshold
        self.tail_threshold = tail_threshold

        # Load data
        self.load_data()
        self.analyze_distribution()
        self.filter_by_class_type()

    def load_data(self):
        """Load ISIC data from CSV."""
        # Read ground truth CSV
        self.df = pd.read_csv(self.csv_path)

        # Get image directory
        img_dir = self.data_dir / "ISIC_2019_Training_Input"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Map image IDs to file paths
        self.image_paths = {}
        for idx, row in self.df.iterrows():
            img_id = row['image']  # Column is 'image' not 'image_id'
            # Try different file formats
            img_path = img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{img_id}.png"
            if img_path.exists():
                self.image_paths[img_id] = img_path

        print(f"Found {len(self.image_paths)} images in {img_dir}")

        # Filter DF to only include images we found
        self.df = self.df[self.df['image'].isin(self.image_paths.keys())]

    def analyze_distribution(self):
        """Analyze class distribution (long-tail analysis)."""
        # Count images per class
        self.class_counts = {}
        for disease_type in self.DISEASE_TYPES:
            count = (self.df[disease_type] == 1).sum()
            self.class_counts[disease_type] = count

        # Sort by frequency
        sorted_counts = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)

        # Head/tail split
        self.head_classes = set([cls for cls, _ in sorted_counts[:self.head_threshold]])
        self.tail_classes = set([cls for cls, _ in sorted_counts[-self.tail_threshold:]])

        print(f"\n{'='*60}")
        print(f"ISIC Dataset Distribution Analysis")
        print(f"{'='*60}")
        print(f"Total images: {len(self.df)}")
        print(f"Total classes: {len(self.class_counts)}")

        print(f"\nClass Counts:")
        for disease, count in sorted_counts:
            class_type = "HEAD" if disease in self.head_classes else "TAIL" if disease in self.tail_classes else "MID"
            print(f"  {disease:6s}: {count:5d} images  [{class_type}]")

        # Compute Gini coefficient
        counts = np.array(list(self.class_counts.values()))
        sorted_counts_vals = np.sort(counts)
        n = len(sorted_counts_vals)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts_vals)) / (n * np.sum(sorted_counts_vals)) - (n + 1) / n

        print(f"\nGini coefficient (imbalance): {gini:.4f}")
        print(f"Head classes: {self.head_classes}")
        print(f"Tail classes: {self.tail_classes}")
        print(f"{'='*60}\n")

    def filter_by_class_type(self):
        """Filter images based on class type."""
        if self.class_type == "head":
            # Keep only head class images
            self.filtered_df = self.df[self.df[list(self.head_classes)].sum(axis=1) > 0]
        elif self.class_type == "tail":
            # Keep only tail class images
            self.filtered_df = self.df[self.df[list(self.tail_classes)].sum(axis=1) > 0]
        else:  # "all"
            self.filtered_df = self.df

        print(f"Filtered to {len(self.filtered_df)} images ({self.class_type} classes)")

    def __len__(self):
        return len(self.filtered_df)

    def __getitem__(self, idx):
        row = self.filtered_df.iloc[idx]
        img_id = row['image']  # Column is 'image' not 'image_id'

        # Load image
        img_path = self.image_paths[img_id]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        # Get label (one-hot encoded in CSV, convert to class index)
        label = None
        disease = None
        for disease_idx, disease_name in enumerate(self.DISEASE_TYPES):
            if row[disease_name] == 1.0:
                label = disease_idx
                disease = disease_name
                break

        if label is None:
            label = 0  # Default to first class
            disease = self.DISEASE_TYPES[0]

        if self.transform:
            image = self.transform(image)

        return image, label, disease

    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if 0 <= class_id < len(self.DISEASE_TYPES):
            return self.DISEASE_TYPES[class_id]
        return f"Unknown_{class_id}"


def get_isic_loaders(
    data_dir: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, ISICDataset, ISICDataset]:
    """
    Create training and validation loaders for ISIC dataset.

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load full dataset
    full_dataset = ISICDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=None,
        class_type="all"
    )

    # Split into train/val
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = random_split(
        list(range(len(full_dataset))),
        [train_size, val_size]
    )

    # Create subset datasets with transforms
    class TransformedSubset(Subset):
        def __init__(self, dataset, indices, transform=None):
            super().__init__(dataset, indices)
            self.transform = transform

        def __getitem__(self, idx):
            img, label, disease = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label, disease

    train_dataset = TransformedSubset(full_dataset, train_indices.indices, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_indices.indices, val_transform)

    print(f"\n✅ Dataset split:")
    print(f"  - Training: {len(train_dataset)} images (80%)")
    print(f"  - Validation: {len(val_dataset)} images (20%)")

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def get_isic_head_tail_loaders(
    data_dir: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, ISICDataset, ISICDataset]:
    """
    Create separate loaders for head and tail classes.

    Returns:
        (head_loader, tail_loader, head_dataset, tail_dataset)
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Head classes
    head_dataset = ISICDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=val_transform,
        class_type="head"
    )

    # Tail classes
    tail_dataset = ISICDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=val_transform,
        class_type="tail"
    )

    head_loader = DataLoader(head_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tail_loader = DataLoader(tail_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return head_loader, tail_loader, head_dataset, tail_dataset
