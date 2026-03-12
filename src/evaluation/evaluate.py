# src/evaluation/evaluate.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.evaluation.advanced_metrics import (
    compute_ks,
    recall_at_k
)


def evaluate_binary_model(
    model,
    X_test,
    y_test,
    threshold_strategy="optimal"
):
    """
    Standard binary classification evaluation
    """

    # =========================
    # Predict probabilities
    # =========================
    y_prob = model.predict(X_test).reshape(-1)

    # =========================
    # Threshold
    # =========================
    if threshold_strategy == "optimal":
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, best_thr = 0, 0.5

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1, best_thr = f1, t

        threshold = best_thr
    else:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    # =========================
    # Metrics
    # =========================
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "ks_statistic": compute_ks(y_test, y_prob),
        "recall_at_10pct": recall_at_k(y_test, y_prob, k_ratio=0.1),
        "optimal_threshold": threshold
    }

    return metrics
