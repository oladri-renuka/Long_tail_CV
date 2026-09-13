"""
Routing classifier with dynamic threshold for uncertainty-based human routing.
"""

import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import pickle


class DynamicRouter:
    """
    Route predictions based on confidence threshold.
    Target: 95% of tail class predictions to human review.
    """

    def __init__(self, target_tail_routing_rate: float = 0.95):
        """
        Args:
            target_tail_routing_rate: Percentage of tail predictions to route to human
        """
        self.target_tail_routing_rate = target_tail_routing_rate
        self.threshold = 0.5  # Will be optimized
        self.head_classes = set()
        self.tail_classes = set()

    def set_class_info(self, head_classes: set, tail_classes: set):
        """Set head and tail class information."""
        self.head_classes = head_classes
        self.tail_classes = tail_classes

    def optimize_threshold(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        tail_classes: set,
        head_classes: set,
    ) -> float:
        """
        Find optimal threshold to route target_routing_rate of tail classes to human.

        Args:
            confidences: Confidence scores (N,)
            labels: True labels (N,)
            tail_classes: Set of tail class IDs
            head_classes: Set of head class IDs

        Returns:
            Optimal confidence threshold
        """

        # Identify which predictions are from tail classes
        is_tail = np.array([label in tail_classes for label in labels])

        if is_tail.sum() == 0:
            print("Warning: No tail class samples found")
            return 0.5

        tail_confidences = confidences[is_tail]

        # Sort confidences
        sorted_confs = np.sort(tail_confidences)

        # Find threshold to route target % to human (lowest confidence)
        idx = int(len(sorted_confs) * self.target_tail_routing_rate)
        threshold = sorted_confs[idx]

        print("\n" + "="*60)
        print("ROUTING THRESHOLD OPTIMIZATION")
        print("="*60)
        print(f"Target tail routing rate: {self.target_tail_routing_rate*100:.1f}%")
        print(f"Tail class samples: {is_tail.sum()}")
        print(f"Tail confidence stats:")
        print(f"  - Min: {tail_confidences.min():.4f}")
        print(f"  - Max: {tail_confidences.max():.4f}")
        print(f"  - Mean: {tail_confidences.mean():.4f}")
        print(f"  - Median: {np.median(tail_confidences):.4f}")
        print(f"  - Std: {tail_confidences.std():.4f}")
        print(f"\nOptimal threshold: {threshold:.4f}")

        # Verify routing rates
        self.threshold = threshold
        metrics = self.evaluate_routing(confidences, labels, tail_classes, head_classes)
        return threshold

    def evaluate_routing(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        tail_classes: set,
        head_classes: set,
    ) -> Dict[str, float]:
        """
        Evaluate routing performance.

        Returns:
            Dictionary with routing metrics
        """

        # Identify class types
        is_tail = np.array([label in tail_classes for label in labels])
        is_head = np.array([label in head_classes for label in labels])

        # Routing decisions
        route_to_human = confidences < self.threshold

        # Compute metrics
        tail_to_human = route_to_human[is_tail]
        tail_routing_rate = tail_to_human.sum() / is_tail.sum() if is_tail.sum() > 0 else 0

        head_to_human = route_to_human[is_head]
        head_false_routing_rate = head_to_human.sum() / is_head.sum() if is_head.sum() > 0 else 0

        total_routed = route_to_human.sum()
        total_samples = len(labels)

        metrics = {
            'threshold': self.threshold,
            'total_samples': total_samples,
            'total_routed': total_routed,
            'total_routing_rate': total_routed / total_samples,
            'tail_samples': is_tail.sum(),
            'tail_routed': tail_to_human.sum(),
            'tail_routing_rate': tail_routing_rate,
            'head_samples': is_head.sum(),
            'head_false_routed': head_to_human.sum(),
            'head_false_routing_rate': head_false_routing_rate,
        }

        print("\n" + "="*60)
        print("ROUTING EVALUATION")
        print("="*60)
        print(f"Confidence Threshold: {self.threshold:.4f}")
        print(f"\nTotal Predictions: {total_samples}")
        print(f"Total Routed to Human: {total_routed} ({metrics['total_routing_rate']*100:.2f}%)")
        print(f"\nHead Class Predictions: {is_head.sum()}")
        print(f"  - False Routed to Human: {head_to_human.sum()} ({head_false_routing_rate*100:.2f}%)")
        print(f"Tail Class Predictions: {is_tail.sum()}")
        print(f"  - Routed to Human: {tail_to_human.sum()} ({tail_routing_rate*100:.2f}%)")
        print(f"\nRouting Rate Comparison:")
        print(f"  - Target: {self.target_tail_routing_rate*100:.1f}%")
        print(f"  - Actual: {tail_routing_rate*100:.2f}%")
        print(f"  - Difference: {abs(tail_routing_rate - self.target_tail_routing_rate)*100:.2f}%")
        print("="*60 + "\n")

        return metrics

    def route(self, confidence: float) -> Tuple[str, str]:
        """
        Make routing decision for a single prediction.

        Args:
            confidence: Confidence score for prediction

        Returns:
            (route: "human" or "automated", reason: explanation)
        """
        if confidence < self.threshold:
            reason = f"Low confidence ({confidence:.4f} < {self.threshold:.4f})"
            return "human", reason
        else:
            reason = f"High confidence ({confidence:.4f} >= {self.threshold:.4f})"
            return "automated", reason

    def route_batch(
        self,
        confidences: np.ndarray,
        category_ids: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Route a batch of predictions.

        Returns:
            (routes: array of "human"/"automated", reasons: list of explanations)
        """
        routes = []
        reasons = []

        for conf, cat_id in zip(confidences, category_ids):
            route, reason = self.route(conf)
            routes.append(route)
            reasons.append(reason)

        return np.array(routes), reasons

    def save(self, path: str):
        """Save router configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'target_tail_routing_rate': self.target_tail_routing_rate,
                'head_classes': self.head_classes,
                'tail_classes': self.tail_classes,
            }, f)
        print(f"Saved router to {path}")

    def load(self, path: str):
        """Load router configuration."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.threshold = data['threshold']
        self.target_tail_routing_rate = data['target_tail_routing_rate']
        self.head_classes = data['head_classes']
        self.tail_classes = data['tail_classes']
        print(f"Loaded router from {path}")


class CostAnalysis:
    """Cost-benefit analysis of routing decisions."""

    @staticmethod
    def compute_costs(
        metrics: Dict,
        human_review_cost: float = 0.10,
        model_error_cost: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute routing costs.

        Args:
            metrics: Routing evaluation metrics
            human_review_cost: Cost per human review ($/image)
            model_error_cost: Cost per model error (relative)

        Returns:
            Cost breakdown
        """

        # Cost of human reviews
        human_review_total = metrics['total_routed'] * human_review_cost

        # Cost of model errors (routed to automated)
        # Assume models make errors on tail classes more often
        tail_automated = metrics['tail_samples'] - metrics['tail_routed']

        print("\n" + "="*60)
        print("COST-BENEFIT ANALYSIS")
        print("="*60)
        print(f"Human Review Cost: ${human_review_cost:.2f}/image")
        print(f"Model Error Cost: {model_error_cost:.2f}x (relative)")
        print(f"\nImages Routed to Human: {metrics['total_routed']}")
        print(f"  - Cost: ${human_review_total:.2f}")
        print(f"\nImages Routed to Automated:")
        print(f"  - Total: {metrics['total_samples'] - metrics['total_routed']}")
        print(f"  - From tail classes: {tail_automated}")
        print(f"  - From head classes: {metrics['head_samples'] - metrics['head_false_routed']}")

        # Compute average cost per prediction
        avg_cost_human = human_review_cost * metrics['total_routing_rate']
        print(f"\nAverage Cost Per Prediction:")
        print(f"  - If all human: ${human_review_cost:.4f}")
        print(f"  - With routing: ${avg_cost_human:.4f}")
        print(f"  - Savings: {(1 - avg_cost_human/human_review_cost)*100:.1f}%")
        print("="*60 + "\n")

        return {
            'human_review_cost': human_review_cost,
            'human_review_total': human_review_total,
            'tail_automated': tail_automated,
            'average_cost_per_prediction': avg_cost_human,
            'cost_reduction': human_review_cost - avg_cost_human,
        }
