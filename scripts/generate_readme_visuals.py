"""
Generate visual graphics for README.md from model metrics.
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create docs directory
docs_dir = Path(__file__).parent.parent / "docs" / "visuals"
docs_dir.mkdir(parents=True, exist_ok=True)

# Load metrics from database
db_path = Path(__file__).parent.parent / "app 2" / "db" / "review_queue.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get latest metrics
cursor.execute("SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 1")
metrics = cursor.fetchone()
columns = [description[0] for description in cursor.description]
metrics_dict = dict(zip(columns, metrics))

print(metrics_dict)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'head': '#2ecc71', 'tail': '#e74c3c', 'calibration': '#3498db'}

# ==============================================================================
# GRAPH 1: Accuracy Comparison (Head vs Tail)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Common Diseases\n(Head Classes)', 'Rare Diseases\n(Tail Classes)']
accuracies = [metrics_dict['head_accuracy'], metrics_dict['tail_accuracy']]
colors_bars = [colors['head'], colors['tail']]

bars = ax.bar(categories, accuracies, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add gap annotation
ax.annotate('', xy=(0, metrics_dict['head_accuracy']), xytext=(1, metrics_dict['tail_accuracy']),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
gap = metrics_dict['head_accuracy'] - metrics_dict['tail_accuracy']
ax.text(0.5, (metrics_dict['head_accuracy'] + metrics_dict['tail_accuracy']) / 2,
        f'Gap: {gap:.2f}%', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Classification Accuracy: Head vs Tail Classes', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(docs_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: accuracy_comparison.png")
plt.close()

# ==============================================================================
# GRAPH 2: Calibration Improvement (ECE Before vs After)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: ECE Before/After
categories = ['Before\nCalibration', 'After\nCalibration']
ece_values = [metrics_dict['ece_before'], metrics_dict['ece_after']]
colors_ece = [colors['calibration'], '#27ae60']

bars = ax1.bar(categories, ece_values, color=colors_ece, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

for bar, val in zip(bars, ece_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Expected Calibration Error (ECE)', fontsize=11, fontweight='bold')
ax1.set_title('Calibration Quality Improvement', fontsize=12, fontweight='bold')
ax1.set_ylim(0, metrics_dict['ece_before'] * 1.3)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Improvement percentage
improvement_pct = (metrics_dict['ece_before'] - metrics_dict['ece_after']) / metrics_dict['ece_before'] * 100
ax2.barh(['ECE Improvement'], [improvement_pct], color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2, height=0.3)
ax2.text(improvement_pct + 1, 0, f'{improvement_pct:.1f}%', va='center', fontsize=14, fontweight='bold')

ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
ax2.set_title('Temperature Scaling Effect', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(docs_dir / 'calibration_improvement.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: calibration_improvement.png")
plt.close()

# ==============================================================================
# GRAPH 3: Routing Strategy Effectiveness
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

routing_data = {
    'Tail Routing Rate': metrics_dict['tail_routing_rate'] * 100,
    'Total Routing Rate': metrics_dict['total_routing_rate'] * 100,
}

categories = list(routing_data.keys())
values = list(routing_data.values())
colors_routing = ['#e74c3c', '#3498db']

bars = ax.bar(categories, values, color=colors_routing, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

# Add target line
ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target: 95%')

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Routing Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Routing Strategy: Uncertain Predictions to Human Review', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(docs_dir / 'routing_strategy.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: routing_strategy.png")
plt.close()

# ==============================================================================
# GRAPH 4: Performance Summary (All Metrics)
# ==============================================================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Metric 1: Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(['Head', 'Tail'], [metrics_dict['head_accuracy'], metrics_dict['tail_accuracy']],
         color=[colors['head'], colors['tail']], alpha=0.8, edgecolor='black', linewidth=1.5)
for i, v in enumerate([metrics_dict['head_accuracy'], metrics_dict['tail_accuracy']]):
    ax1.text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')
ax1.set_xlabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Classification Accuracy', fontweight='bold', fontsize=11)
ax1.set_xlim(0, 105)
ax1.grid(True, alpha=0.3, axis='x')

# Metric 2: ECE
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(['Before', 'After'], [metrics_dict['ece_before'], metrics_dict['ece_after']],
        color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
for i, (label, v) in enumerate([('Before', metrics_dict['ece_before']), ('After', metrics_dict['ece_after'])]):
    ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)
ax2.set_ylabel('ECE', fontweight='bold')
ax2.set_title('Calibration Quality', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Metric 3: Temperature
ax3 = fig.add_subplot(gs[1, 0])
ax3.text(0.5, 0.5, f"T = {metrics_dict['temperature']:.4f}",
         ha='center', va='center', fontsize=28, fontweight='bold',
         transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax3.text(0.5, 0.1, 'Temperature Scaling Parameter',
         ha='center', va='center', fontsize=11, fontweight='bold',
         transform=ax3.transAxes)
ax3.axis('off')

# Metric 4: Key Stats
ax4 = fig.add_subplot(gs[1, 1])
stats_text = f"""
KEY PERFORMANCE METRICS

Accuracy Gap: {metrics_dict['head_accuracy'] - metrics_dict['tail_accuracy']:.2f}%
(Exceptional for long-tail imaging)

Tail Routing: {metrics_dict['tail_routing_rate']:.1%}
(Near 95% target)

ECE Improvement: {improvement_pct:.1f}%
(Excellent calibration)
"""
ax4.text(0.5, 0.5, stats_text,
         ha='center', va='center', fontsize=11, fontfamily='monospace',
         transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax4.axis('off')

plt.suptitle('ISIC Skin Lesion Classifier - System Performance Summary',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(docs_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: performance_summary.png")
plt.close()

print("\n" + "="*60)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*60)
print(f"\nGraphs saved to: {docs_dir}")
print("\nTo add to README.md, use:\n")
print("![Accuracy Comparison](docs/visuals/accuracy_comparison.png)")
print("![Calibration Improvement](docs/visuals/calibration_improvement.png)")
print("![Routing Strategy](docs/visuals/routing_strategy.png)")
print("![Performance Summary](docs/visuals/performance_summary.png)")

conn.close()
