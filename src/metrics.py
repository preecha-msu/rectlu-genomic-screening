"""
src/metrics.py
Unified Evaluation Metrics, Threshold Calibration, Dead-Unit & Gradient Diagnostics
Authors: Preecha Noiumkar and Thawatchai Chomsiri
Affiliation: Faculty of Informatics, Mahasarakham University
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_true: np.ndarray = None,
    val_prob: np.ndarray = None,
    fixed_threshold: float = 0.50,
) -> dict:
    """
    Computes all metrics required