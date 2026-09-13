#!/usr/bin/env python3
"""
Calibration & Routing for ISIC Skin Lesion Dataset.
Temperature scaling and dynamic threshold optimization.
"""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isic_loader import get_isic_loaders, get_isic_head_tail_loaders
from src.model import ImageClassifier
from src.calibration import CalibrationPipeline, calibrate_predictions
from src.router import DynamicRouter, CostAnalysis
from src.database import ReviewQueueDB


def get_logits_and_labels(model, data_loader, device):
    """Extract logits and labels from data loader."""
    all_logits = []
    all_labels = []
    all_disease_ids = []

    model.eval()
    with torch.no_grad():
        for images, labels, disease_ids in data_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_disease_ids.extend(disease_ids)

    logits = np.vstack(all_logits)
    labels = np.array(all_labels)
    disease_ids = np.array(all_disease_ids)

    return logits, labels, disease_ids


def main(args):
    """Main calibration pipeline for ISIC."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ========================================================================
    # STAGE 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 1: LOAD DATA & MODEL")
    print("="*60 + "\n")

    try:
        train_loader, val_loader, train_dataset, val_dataset = get_isic_loaders(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Load model
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Run: python scripts/train_isic.py")
        return

    model = ImageClassifier(num_classes=8)  # ISIC has 8 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"✅ Loaded model from {model_path}")

    # ========================================================================
    # STAGE 2: SPLIT VALIDATION SET
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 2: SPLIT VALIDATION SET")
    print("="*60 + "\n")

    # Split validation into calibration (80%) and final test (20%)
    val_size = len(val_dataset)
    cal_size = int(0.8 * val_size)
    test_size = val_size - cal_size

    from torch.utils.data import random_split, DataLoader
    cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size])

    cal_loader = DataLoader(
        cal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Validation split:")
    print(f"  - Calibration set: {cal_size} ({cal_size/val_size*100:.1f}%)")
    print(f"  - Test set: {test_size} ({test_size/val_size*100:.1f}%)")

    # ========================================================================
    # STAGE 3: TEMPERATURE SCALING
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 3: TEMPERATURE SCALING CALIBRATION")
    print("="*60 + "\n")

    # Get logits from calibration set
    print("Extracting logits from calibration set...")
    cal_logits, cal_labels, cal_disease_ids = get_logits_and_labels(
        model, cal_loader, device
    )

    # Fit temperature
    calibration = CalibrationPipeline()
    calibration.fit_and_evaluate(
        logits=cal_logits,
        labels=cal_labels,
        num_bins=10,
        device=device
    )

    # ========================================================================
    # STAGE 4: DYNAMIC ROUTING THRESHOLD
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 4: DYNAMIC ROUTING THRESHOLD")
    print("="*60 + "\n")

    # Get logits from test set
    print("Extracting logits from test set...")
    test_logits, test_labels, test_disease_ids = get_logits_and_labels(
        model, test_loader, device
    )

    # Get calibrated predictions
    test_probs = calibrate_predictions(test_logits, calibration.temperature)
    test_preds = np.argmax(test_probs, axis=1)
    test_confs = np.max(test_probs, axis=1)

    # Initialize router
    router = DynamicRouter(target_tail_routing_rate=0.95)

    # Get head/tail info from test dataset
    head_loader, tail_loader, head_dataset, tail_dataset = get_isic_head_tail_loaders(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create sets of head/tail class IDs (0-7 for ISIC)
    head_classes = {0}  # Nevus (most common)
    tail_classes = {4, 5, 7}  # Rare classes

    router.set_class_info(
        head_classes=head_classes,
        tail_classes=tail_classes
    )

    # Optimize threshold
    optimal_threshold = router.optimize_threshold(
        confidences=test_confs,
        labels=test_labels,
        tail_classes=tail_classes,
        head_classes=head_classes
    )

    # ========================================================================
    # STAGE 5: ROUTING EVALUATION
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 5: ROUTING EVALUATION")
    print("="*60 + "\n")

    routing_metrics = router.evaluate_routing(
        confidences=test_confs,
        labels=test_labels,
        tail_classes=tail_classes,
        head_classes=head_classes
    )

    # ========================================================================
    # STAGE 6: COST ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 6: COST-BENEFIT ANALYSIS")
    print("="*60 + "\n")

    cost_analysis = CostAnalysis()
    costs = cost_analysis.compute_costs(
        routing_metrics,
        human_review_cost=0.50,  # Medical review is more expensive
        model_error_cost=1.0
    )

    # ========================================================================
    # STAGE 7: HEAD/TAIL ACCURACY
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 7: HEAD vs TAIL ACCURACY")
    print("="*60 + "\n")

    # Evaluate on test set
    head_mask = np.array([label in head_classes for label in test_labels])
    tail_mask = np.array([label in tail_classes for label in test_labels])

    head_correct = (test_preds[head_mask] == test_labels[head_mask]).sum()
    head_total = head_mask.sum()
    head_acc = 100.0 * head_correct / head_total if head_total > 0 else 0.0

    tail_correct = (test_preds[tail_mask] == test_labels[tail_mask]).sum()
    tail_total = tail_mask.sum()
    tail_acc = 100.0 * tail_correct / tail_total if tail_total > 0 else 0.0

    print(f"Head class accuracy: {head_acc:.2f}% ({head_correct}/{head_total})")
    print(f"Tail class accuracy: {tail_acc:.2f}% ({tail_correct}/{tail_total})")
    print(f"Accuracy gap: {head_acc - tail_acc:.2f}%")

    # ========================================================================
    # STAGE 8: SAVE MODELS & METRICS
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 8: SAVE MODELS & METRICS")
    print("="*60 + "\n")

    # Save calibration
    calibration.save("models/temperature_model.pkl")

    # Save router
    router.save("models/router.pkl")

    # Log metrics to database
    db = ReviewQueueDB()
    db.log_metrics(
        head_accuracy=head_acc,
        tail_accuracy=tail_acc,
        ece_before=calibration.metrics_before['ece'],
        ece_after=calibration.metrics_after['ece'],
        temperature=calibration.temperature,
        routing_threshold=router.threshold,
        tail_routing_rate=routing_metrics['tail_routing_rate'],
        head_false_routing_rate=routing_metrics['head_false_routing_rate'],
        total_routing_rate=routing_metrics['total_routing_rate']
    )
    print("✅ Logged metrics to database")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("CALIBRATION PIPELINE SUMMARY (ISIC)")
    print("="*60)
    print(f"\n📊 CALIBRATION METRICS")
    print(f"  Temperature: {calibration.temperature:.4f}")
    print(f"  ECE Before: {calibration.metrics_before['ece']:.4f}")
    print(f"  ECE After: {calibration.metrics_after['ece']:.4f}")
    ece_improvement = (calibration.metrics_before['ece'] - calibration.metrics_after['ece']) / (calibration.metrics_before['ece'] + 1e-10)
    print(f"  ECE Improvement: {ece_improvement*100:.2f}%")

    print(f"\n📊 ACCURACY METRICS")
    print(f"  Head accuracy: {head_acc:.2f}%")
    print(f"  Tail accuracy: {tail_acc:.2f}%")
    print(f"  Accuracy gap: {head_acc - tail_acc:.2f}%")

    print(f"\n📊 ROUTING METRICS")
    print(f"  Confidence threshold: {router.threshold:.4f}")
    print(f"  Tail routing rate: {routing_metrics['tail_routing_rate']*100:.2f}%")
    print(f"  Head false routing rate: {routing_metrics['head_false_routing_rate']*100:.2f}%")
    print(f"  Total routing rate: {routing_metrics['total_routing_rate']*100:.2f}%")

    print(f"\n💰 COST ANALYSIS (Medical Review: $0.50/image)")
    print(f"  Human review cost: ${costs['human_review_total']:.2f}")
    print(f"  Average cost per prediction: ${costs['average_cost_per_prediction']:.4f}")
    print(f"  Cost reduction: ${costs['cost_reduction']:.4f} per prediction")

    print(f"\n✅ Calibration pipeline complete!")
    print(f"   - Calibration model: models/temperature_model.pkl")
    print(f"   - Router model: models/router.pkl")
    print(f"\n🚀 Ready to deploy: streamlit run app/streamlit_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate ViT-B/16 on ISIC dataset")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/isic",
        help="Path to ISIC dataset directory"
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
        default=64,
        help="Batch size for calibration"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    args = parser.parse_args()
    main(args)
