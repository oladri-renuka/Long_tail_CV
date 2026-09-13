#!/usr/bin/env python3
"""
Train ViT-B/16 on ISIC Skin Lesion dataset.
Much faster than iNaturalist (25K images vs 500K).
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isic_loader import get_isic_loaders, get_isic_head_tail_loaders
from src.model import ImageClassifier, Trainer


def main(args):
    """Main training pipeline for ISIC."""

    # Create directories
    Path("models").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ========================================================================
    # STAGE 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 1: DATA LOADING & ANALYSIS")
    print("="*60 + "\n")

    try:
        train_loader, val_loader, train_dataset, val_dataset = get_isic_loaders(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        num_classes = 8  # ISIC has 8 disease types
        print(f"\nDataset loaded successfully!")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Train samples: {len(train_dataset)}")
        print(f"  - Val samples: {len(val_dataset)}")
        print(f"  - Batch size: {args.batch_size}")

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print("Ensure ISIC data is at:")
        print(f"  {Path(args.data_dir).absolute()}/ISIC_2019_Training_Input/")
        print(f"  {Path(args.csv_path).absolute()}")
        return

    # ========================================================================
    # STAGE 2: MODEL INITIALIZATION
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 2: MODEL INITIALIZATION")
    print("="*60 + "\n")

    model = ImageClassifier(num_classes=num_classes, pretrained=True)
    print(f"Initialized ViT-B/16 with {num_classes} output classes")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ========================================================================
    # STAGE 3: TRAINING
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 3: TRAINING")
    print("="*60 + "\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir="models",
        patience=args.patience,
    )

    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

    # ========================================================================
    # STAGE 4: HEAD/TAIL EVALUATION
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 4: HEAD vs TAIL EVALUATION")
    print("="*60 + "\n")

    # Load head/tail datasets
    head_loader, tail_loader, head_dataset, tail_dataset = get_isic_head_tail_loaders(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load best model
    best_model_path = Path("models/best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate
    head_tail_metrics = trainer.evaluate_head_tail(head_loader, tail_loader)

    print(f"\n✅ Head/Tail evaluation complete!")
    print(f"Accuracy gap (Head - Tail): {head_tail_metrics['head_accuracy'] - head_tail_metrics['tail_accuracy']:.2f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: ISIC 2019 Skin Lesion")
    print(f"Total images: 25,331")
    print(f"Total classes: 8 disease types")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Best model: models/best_model.pth")
    print(f"\nHead accuracy (most common class): {head_tail_metrics['head_accuracy']:.2f}%")
    print(f"Tail accuracy (rare classes): {head_tail_metrics['tail_accuracy']:.2f}%")
    print(f"Accuracy gap: {head_tail_metrics['head_accuracy'] - head_tail_metrics['tail_accuracy']:.2f}%")
    print("\n✅ Ready for calibration: python scripts/calibrate_isic.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-B/16 on ISIC Skin Lesion dataset")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/isic",
        help="Path to ISIC dataset directory (contains ISIC_2019_Training_Input/)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/isic/ISIC_2019_Training_GroundTruth.csv",
        help="Path to ISIC ground truth CSV"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )

    args = parser.parse_args()
    main(args)
