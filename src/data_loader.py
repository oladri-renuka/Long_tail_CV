"""
iNaturalist 2021 mini dataset loader with long-tail analysis.
Handles official train/val splits and head/tail class separation.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm
import requests
from zipfile import ZipFile
import io


class iNaturalistDataset(Dataset):
    """iNaturalist 2021 mini dataset with long-tail analysis."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        class_type: str = "all",  # "all", "head", "tail"
        head_threshold: int = 100,
        tail_threshold: int = 1000,
    ):
        """
        Args:
            data_dir: Root directory containing iNaturalist data
            split: "train" or "val"
            transform: Image transformations
            class_type: Filter by class distribution ("all", "head", "tail")
            head_threshold: Top N species are head classes
            tail_threshold: Bottom N species are tail classes
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.class_type = class_type
        self.head_threshold = head_threshold
        self.tail_threshold = tail_threshold

        # Load metadata
        self.load_metadata()
        self.analyze_distribution()
        self.filter_by_class_type()

    def load_metadata(self):
        """Load iNaturalist JSON metadata."""
        # Try multiple naming conventions
        possible_names = [
            f"{self.split}2021_mini.json",      # val2021_mini.json, train2021_mini.json
            f"{self.split}_mini2021.json",      # val_mini2021.json, train_mini2021.json
            f"{self.split}2021.json",           # val2021.json, train2021.json
            f"{self.split}_mini.json",          # val_mini.json, train_mini.json
        ]

        json_path = None
        for name in possible_names:
            candidate = self.data_dir / name
            if candidate.exists():
                json_path = candidate
                print(f"Found metadata: {name}")
                break

        if json_path is None:
            raise FileNotFoundError(
                f"Metadata file not found in {self.data_dir}. "
                f"Tried: {', '.join(possible_names)}"
            )

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}

        # Create mapping: image_id -> annotation
        self.image_to_annotation = {}
        for ann in self.annotations:
            self.image_to_annotation[ann['image_id']] = ann['category_id']

        # Filter images with annotations
        self.valid_images = [
            img for img in self.images
            if img['id'] in self.image_to_annotation
        ]

    def analyze_distribution(self):
        """Analyze species distribution to identify head/tail classes."""
        category_counts = Counter()
        for img in self.valid_images:
            cat_id = self.image_to_annotation[img['id']]
            category_counts[cat_id] += 1

        # Sort by frequency
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Distribution info
        self.category_counts = dict(sorted_categories)
        self.class_distribution = pd.Series(self.category_counts)

        # Head/tail split
        self.head_classes = set(
            cat_id for cat_id, _ in sorted_categories[:self.head_threshold]
        )
        self.tail_classes = set(
            cat_id for cat_id, _ in sorted_categories[-self.tail_threshold:]
        )

        print(f"\n{'='*60}")
        print(f"Distribution Analysis ({self.split})")
        print(f"{'='*60}")
        print(f"Total images: {len(self.valid_images)}")
        print(f"Total classes: {len(self.category_counts)}")
        print(f"Head classes (top {self.head_threshold}): {len(self.head_classes)}")
        print(f"  - Images in head: {sum(self.category_counts[c] for c in self.head_classes)}")
        print(f"  - Min count: {min(self.category_counts[c] for c in self.head_classes)}")
        print(f"  - Max count: {max(self.category_counts[c] for c in self.head_classes)}")
        print(f"Tail classes (bottom {self.tail_threshold}): {len(self.tail_classes)}")
        print(f"  - Images in tail: {sum(self.category_counts[c] for c in self.tail_classes)}")
        print(f"  - Min count: {min(self.category_counts[c] for c in self.tail_classes)}")
        print(f"  - Max count: {max(self.category_counts[c] for c in self.tail_classes)}")

        # Gini coefficient (measure of imbalance)
        counts = np.array(list(self.category_counts.values()))
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        print(f"Gini coefficient: {gini:.4f}")
        print(f"{'='*60}\n")

    def filter_by_class_type(self):
        """Filter images based on class type (head/tail/all)."""
        if self.class_type == "head":
            self.filtered_images = [
                img for img in self.valid_images
                if self.image_to_annotation[img['id']] in self.head_classes
            ]
        elif self.class_type == "tail":
            self.filtered_images = [
                img for img in self.valid_images
                if self.image_to_annotation[img['id']] in self.tail_classes
            ]
        else:  # "all"
            self.filtered_images = self.valid_images

        # Create category to index mapping for filtered classes
        unique_categories = sorted(set(
            self.image_to_annotation[img['id']] for img in self.filtered_images
        ))
        self.category_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        self.idx_to_category = {idx: cat_id for cat_id, idx in self.category_to_idx.items()}
        self.num_classes = len(self.category_to_idx)

    def __len__(self):
        return len(self.filtered_images)

    def __getitem__(self, idx):
        img_info = self.filtered_images[idx]
        cat_id = self.image_to_annotation[img_info['id']]
        label = self.category_to_idx[cat_id]

        # Load image
        img_path = self.data_dir / self.split / "images" / img_info['file_name']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return blank image as fallback
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label, cat_id

    def get_category_name(self, cat_id: int) -> str:
        """Get category name from ID."""
        return self.categories.get(cat_id, f"Unknown_{cat_id}")

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced sampling."""
        counts = np.array([
            self.category_counts[self.idx_to_category[i]]
            for i in range(self.num_classes)
        ])
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, iNaturalistDataset, iNaturalistDataset]:
    """
    Create training and validation data loaders.

    If separate val files exist, uses them. Otherwise uses 90/10 split of train_mini.

    Args:
        data_dir: Dataset directory
        batch_size: Batch size
        num_workers: Data loading workers
        img_size: Image size
        val_split: Validation split (default: 0.1 for 90/10 train/val)

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
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = Path(data_dir)

    # Check if separate val files exist
    has_separate_val = (
        (data_dir / "val2021_mini.json").exists() or
        (data_dir / "val2021.json").exists()
    )

    if has_separate_val:
        print("\n✅ Using separate training and validation sets")
        # Load datasets with separate val
        train_dataset = iNaturalistDataset(
            data_dir,
            split="train",
            transform=train_transform,
            class_type="all"
        )

        val_dataset = iNaturalistDataset(
            data_dir,
            split="val",
            transform=val_transform,
            class_type="all"
        )
    else:
        print(f"\n✅ Using 90/10 split of train_mini (no separate val files found)")
        # Load train_mini and split 90/10
        full_dataset = iNaturalistDataset(
            data_dir,
            split="train",
            transform=None,  # No transform yet
            class_type="all"
        )

        # Split dataset
        train_size = int(0.9 * len(full_dataset))
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
                img, label, cat_id = self.dataset[self.indices[idx]]
                if self.transform:
                    img = self.transform(img)
                return img, label, cat_id

        train_dataset = TransformedSubset(full_dataset, train_indices.indices, train_transform)
        val_dataset = TransformedSubset(full_dataset, val_indices.indices, val_transform)

        print(f"  - Train set: {len(train_dataset)} images (90%)")
        print(f"  - Val set: {len(val_dataset)} images (10%)")

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


def get_head_tail_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create separate loaders for head and tail classes.

    Returns:
        (head_train, head_val, tail_train, tail_val)
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
    head_val = iNaturalistDataset(
        data_dir,
        split="val",
        transform=val_transform,
        class_type="head"
    )

    # Tail classes
    tail_val = iNaturalistDataset(
        data_dir,
        split="val",
        transform=val_transform,
        class_type="tail"
    )

    head_loader = DataLoader(head_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tail_loader = DataLoader(tail_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return head_loader, tail_loader, head_val, tail_val
