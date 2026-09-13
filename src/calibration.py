"""
Temperature scaling calibration and ECE computation.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.optimize import minimize
from typing import Tuple
import pickle


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def set_temperature(self, temperature: float):
        """Set temperature value."""
        self.temperature.data = torch.tensor([temperature])

    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.temperature.item()


class CalibrationMetrics:
    """Compute calibration metrics: ECE, MCE, etc."""

    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            predictions: Predicted class indices (N,)
            confidences: Confidence scores (N,)
            labels: True labels (N,)
            num_bins: Number of bins for ECE calculation

        Returns:
            ECE score
        """
        # Sort samples by confidence
        sorted_indices = np.argsort(confidences)
        predictions = predictions[sorted_indices]
        confidences = confidences[sorted_indices]
        labels = labels[sorted_indices]

        # Compute accuracy
        accuracy = (predictions == labels).astype(float)

        # Bin samples
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Compute ECE
        ece = 0.0
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = accuracy[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(predictions) * abs(bin_acc - bin_conf)

        return ece

    @staticmethod
    def maximum_calibration_error(
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        accuracy = (predictions == labels).astype(float)
        mce = np.max(np.abs(accuracy - confidences))
        return mce

    @staticmethod
    def brier_score(
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute Brier score."""
        accuracy = (predictions == labels).astype(float)
        brier = np.mean((confidences - accuracy) ** 2)
        return brier

    @staticmethod
    def negative_log_likelihood(
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood."""
        # Convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get probability of true class
        true_probs = probs[np.arange(len(labels)), labels]
        nll = -np.log(true_probs + 1e-10).mean()

        return nll


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    init_temperature: float = 1.0,
) -> float:
    """
    Fit temperature scaling on validation set.

    Args:
        logits: Model outputs (N, K) before softmax
        labels: True labels (N,)
        init_temperature: Initial temperature value

    Returns:
        Optimal temperature
    """
    def nll_loss(temperature):
        """Negative log-likelihood loss."""
        if temperature <= 0:
            return 1e10

        # Apply temperature scaling
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # NLL loss
        true_probs = probs[np.arange(len(labels)), labels]
        nll = -np.log(true_probs + 1e-10).mean()

        return nll

    # Optimize temperature
    result = minimize(
        nll_loss,
        x0=[init_temperature],
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )

    optimal_temperature = result.x[0]
    return optimal_temperature


def calibrate_predictions(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Model outputs (N, K)
        temperature: Temperature value

    Returns:
        Calibrated probabilities (N, K)
    """
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


class CalibrationPipeline:
    """End-to-end calibration pipeline."""

    def __init__(self):
        self.temperature = 1.0
        self.metrics_before = {}
        self.metrics_after = {}

    def fit_and_evaluate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
        device: str = 'cpu',
    ):
        """
        Fit temperature scaling and compute calibration metrics.

        Args:
            logits: Model outputs (N, K)
            labels: True labels (N,)
            num_bins: Number of bins for ECE
            device: Device for computation
        """

        # Get predictions and confidences before calibration
        pred_before = np.argmax(logits, axis=1)
        conf_before = np.max(np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True), axis=1)

        # Metrics before
        self.metrics_before = {
            'ece': CalibrationMetrics.expected_calibration_error(
                pred_before, conf_before, labels, num_bins=num_bins
            ),
            'mce': CalibrationMetrics.maximum_calibration_error(
                pred_before, conf_before, labels
            ),
            'brier': CalibrationMetrics.brier_score(pred_before, conf_before, labels),
            'nll': CalibrationMetrics.negative_log_likelihood(logits, labels),
        }

        print("\n" + "="*60)
        print("CALIBRATION METRICS (Before Temperature Scaling)")
        print("="*60)
        print(f"ECE (Expected Calibration Error): {self.metrics_before['ece']:.4f}")
        print(f"MCE (Max Calibration Error): {self.metrics_before['mce']:.4f}")
        print(f"Brier Score: {self.metrics_before['brier']:.4f}")
        print(f"NLL (Negative Log-Likelihood): {self.metrics_before['nll']:.4f}")

        # Fit temperature
        self.temperature = fit_temperature_scaling(logits, labels)
        print(f"\nOptimal Temperature: {self.temperature:.4f}")

        # Apply temperature and get calibrated predictions
        calibrated_probs = calibrate_predictions(logits, self.temperature)
        pred_after = np.argmax(calibrated_probs, axis=1)
        conf_after = np.max(calibrated_probs, axis=1)

        # Metrics after
        self.metrics_after = {
            'ece': CalibrationMetrics.expected_calibration_error(
                pred_after, conf_after, labels, num_bins=num_bins
            ),
            'mce': CalibrationMetrics.maximum_calibration_error(
                pred_after, conf_after, labels
            ),
            'brier': CalibrationMetrics.brier_score(pred_after, conf_after, labels),
            'nll': CalibrationMetrics.negative_log_likelihood(
                logits / self.temperature, labels
            ),
        }

        print("\n" + "="*60)
        print("CALIBRATION METRICS (After Temperature Scaling)")
        print("="*60)
        print(f"ECE (Expected Calibration Error): {self.metrics_after['ece']:.4f}")
        print(f"MCE (Max Calibration Error): {self.metrics_after['mce']:.4f}")
        print(f"Brier Score: {self.metrics_after['brier']:.4f}")
        print(f"NLL (Negative Log-Likelihood): {self.metrics_after['nll']:.4f}")

        # Improvement
        ece_improvement = (self.metrics_before['ece'] - self.metrics_after['ece']) / (self.metrics_before['ece'] + 1e-10)
        print(f"\nECE Improvement: {ece_improvement*100:.2f}%")
        print("="*60 + "\n")

        return self.metrics_before, self.metrics_after, self.temperature

    def save(self, path: str):
        """Save temperature scaling model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'temperature': self.temperature,
                'metrics_before': self.metrics_before,
                'metrics_after': self.metrics_after
            }, f)
        print(f"Saved calibration model to {path}")

    def load(self, path: str):
        """Load temperature scaling model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.temperature = data['temperature']
        self.metrics_before = data['metrics_before']
        self.metrics_after = data['metrics_after']
        print(f"Loaded calibration model from {path}")
