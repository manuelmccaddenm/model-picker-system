from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error


def primary_metric(task: str, y_true, y_pred, y_prob: Optional[np.ndarray], name: str = "auto") -> float:
    if task == "classification":
        if name in ("roc_auc", "auto") and y_prob is not None:
            # Check if multi-class (more than 2 classes)
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                # Multi-class: use 'ovr' (one-vs-rest) strategy
                return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            else:
                # Binary: standard ROC AUC
                # For binary, y_prob should be 1D (probabilities for positive class)
                if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                return float(roc_auc_score(y_true, y_prob))
        if name == "f1" and y_pred is not None:
            # Use weighted average for multi-class
            return float(f1_score(y_true, y_pred, average='weighted'))
        # fallback
        if y_prob is not None:
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            else:
                if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                return float(roc_auc_score(y_true, y_prob))
        else:
            return float(f1_score(y_true, y_pred, average='weighted'))
    # regression
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name in ("rmse", "auto"):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(mean_absolute_error(y_true, y_pred))


def multi_objective(primary: float, latency_ms: float = 0.0, ece: float = 0.0, weights: Optional[dict] = None) -> float:
    w = {"primary": 0.8, "latency": 0.1, "ece": 0.1}
    if weights:
        w.update(weights)
    # Higher is better; convert latency/ece penalties
    score = w["primary"] * primary - w["latency"] * (latency_ms / 1000.0) - w["ece"] * ece
    return float(score)
