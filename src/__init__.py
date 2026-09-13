"""
Long-tail image classification system for iNaturalist 2021.
"""

from .data_loader import iNaturalistDataset, get_data_loaders, get_head_tail_loaders
from .model import ImageClassifier, Trainer
from .calibration import CalibrationPipeline, TemperatureScaling, CalibrationMetrics
from .router import DynamicRouter, CostAnalysis
from .database import ReviewQueueDB

__version__ = "1.0.0"
__author__ = "Claude"

__all__ = [
    "iNaturalistDataset",
    "get_data_loaders",
    "get_head_tail_loaders",
    "ImageClassifier",
    "Trainer",
    "CalibrationPipeline",
    "TemperatureScaling",
    "CalibrationMetrics",
    "DynamicRouter",
    "CostAnalysis",
    "ReviewQueueDB",
]
