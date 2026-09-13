"""
Utility functions for long-tail classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def plot_distribution(category_counts: Dict[int, int], save_path: str = None):
    """Plot species distribution (log scale)."""
    counts = np.array(list(category_counts.values()))
    sorted_counts = np.sort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.loglog(range(1, len(sorted_counts) + 1), sorted_counts, linewidth=2)
    ax.set_xlabel("Species Rank", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("iNaturalist Long-Tail Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")

    return fig, ax


def plot_calibration_curve(predictions: np.ndarray, confidences: np.ndarray,
                           labels: np.ndarray, save_path: str = None, num_bins: int = 10):
    """Plot calibration curve (confidence vs accuracy)."""
    # Sort by confidence
    sorted_idx = np.argsort(confidences)
    predictions = predictions[sorted_idx]
    confidences = confidences[sorted_idx]
    labels = labels[sorted_idx]

    # Compute accuracy
    accuracy = (predictions == labels).astype(float)

    # Bin samples
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute bin metrics
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accs.append(accuracy[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Calibration curve
    ax.scatter(bin_confs, bin_accs, s=[s/10 for s in bin_sizes], alpha=0.6, color='blue', label='Calibration')
    ax.plot(bin_confs, bin_accs, 'b-', alpha=0.3)

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot to {save_path}")

    return fig, ax


def plot_accuracy_gap(head_acc: float, tail_acc: float, save_path: str = None):
    """Plot head vs tail accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Head\n(Top 100)', 'Tail\n(Bottom 1000)']
    accuracies = [head_acc, tail_acc]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add gap annotation
    ax.annotate('', xy=(0, head_acc), xytext=(1, tail_acc),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(0.5, (head_acc + tail_acc) / 2, f'Gap: {head_acc - tail_acc:.1f}%',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Gap: Head vs Tail Classes", fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy gap plot to {save_path}")

    return fig, ax


def plot_routing_rates(metrics: Dict, save_path: str = None):
    """Plot routing metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Routing breakdown
    labels = ['Automated\n(Head)', 'Automated\n(Tail)', 'Human\nReview']
    head_automated = metrics['head_samples'] - metrics['head_false_routed']
    tail_automated = metrics['tail_samples'] - metrics['tail_routed']
    sizes = [head_automated, tail_automated, metrics['total_routed']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Routing Decision Breakdown", fontsize=12)

    # Routing rates by class
    class_types = ['Head Classes', 'Tail Classes']
    false_routing = [
        metrics['head_false_routing_rate'] * 100,
        metrics['tail_routing_rate'] * 100
    ]
    colors2 = ['#3498db', '#e74c3c']

    bars = ax2.bar(class_types, false_routing, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')

    for bar, rate in zip(bars, false_routing):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel("Routing Rate (%)", fontsize=12)
    ax2.set_title("Routing Rates by Class Type", fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved routing rates plot to {save_path}")

    return fig, (ax1, ax2)


def plot_confusion_matrix(true_labels: List[int], pred_labels: List[int],
                         label_names: Dict[int, str] = None, save_path: str = None,
                         top_k: int = 20):
    """Plot confusion matrix for top-k species."""
    from sklearn.metrics import confusion_matrix

    # Get top-k classes
    unique_labels = np.unique(true_labels + pred_labels)
    top_k_labels = unique_labels[-top_k:] if len(unique_labels) > top_k else unique_labels

    # Filter to top-k
    mask = np.isin(true_labels, top_k_labels) | np.isin(pred_labels, top_k_labels)
    true_filtered = np.array(true_labels)[mask]
    pred_filtered = np.array(pred_labels)[mask]

    # Compute confusion matrix
    cm = confusion_matrix(true_filtered, pred_filtered)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})

    if label_names:
        labels_list = [label_names.get(i, f"Class {i}") for i in top_k_labels]
    else:
        labels_list = [f"Class {i}" for i in top_k_labels]

    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticklabels(labels_list, rotation=0)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix (Top-{top_k} Species)", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    return fig, ax


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """Compute Gini coefficient (measure of inequality/imbalance)."""
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    return gini


def print_summary_report(metrics: Dict):
    """Print comprehensive summary report."""
    print("\n" + "="*70)
    print("LONG-TAIL CLASSIFICATION SYSTEM - SUMMARY REPORT")
    print("="*70)

    if 'head_accuracy' in metrics:
        print(f"\n📊 ACCURACY METRICS")
        print(f"  Head Class Accuracy: {metrics['head_accuracy']:.2f}%")
        print(f"  Tail Class Accuracy: {metrics['tail_accuracy']:.2f}%")
        print(f"  Accuracy Gap: {metrics['head_accuracy'] - metrics['tail_accuracy']:.2f}%")

    if 'ece_before' in metrics:
        print(f"\n📊 CALIBRATION METRICS")
        print(f"  ECE Before: {metrics['ece_before']:.4f}")
        print(f"  ECE After: {metrics['ece_after']:.4f}")
        print(f"  Temperature: {metrics['temperature']:.4f}")
        ece_improvement = (metrics['ece_before'] - metrics['ece_after']) / (metrics['ece_before'] + 1e-10)
        print(f"  ECE Improvement: {ece_improvement*100:.2f}%")

    if 'routing_threshold' in metrics:
        print(f"\n📊 ROUTING METRICS")
        print(f"  Confidence Threshold: {metrics['routing_threshold']:.4f}")
        print(f"  Tail Routing Rate: {metrics['tail_routing_rate']*100:.2f}%")
        print(f"  Head False Routing Rate: {metrics['head_false_routing_rate']*100:.2f}%")
        print(f"  Total Routing Rate: {metrics['total_routing_rate']*100:.2f}%")

    if 'human_review_total' in metrics:
        print(f"\n💰 COST ANALYSIS")
        print(f"  Human Reviews Cost: ${metrics['human_review_total']:.2f}")
        print(f"  Cost per Prediction: ${metrics['average_cost_per_prediction']:.4f}")
        print(f"  Cost Reduction: ${metrics['cost_reduction']:.4f}/prediction")

    print("\n" + "="*70 + "\n")
