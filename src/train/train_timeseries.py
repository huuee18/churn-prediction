
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)
def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=0.5,
    threshold_high=0.7,
    threshold_low=0.3
):
    """
    Đánh giá model deep learning cho churn prediction
    Áp dụng cho TSMixer / N-BEATS / DLinear
    """

    # -------- Predict probability --------
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # -------- Risk segmentation --------
    risk_level = np.where(
        y_prob >= threshold_high, "High churn risk",
        np.where(y_prob <= threshold_low, "Low churn risk", "Medium churn risk")
    )

    # -------- Result DataFrame --------
    results_df = pd.DataFrame({
        "y_true": y_test,
        "churn_prob": y_prob,
        "y_pred": y_pred,
        "risk_level": risk_level
    })

    # -------- Metrics --------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "results_df": results_df
    }

    return metrics
def train_ts_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=20,
    batch_size=64,
    threshold=0.5
):
    """
    Training loop cho time-series deep models
    models: dict[str, keras.Model]
    """

    results = {}

    for name, model in models.items():
        print(f"\n🔵 Training model: {name}")

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            threshold=threshold
        )
        cm_path = save_confusion_matrix(
            metrics["confusion_matrix"],
            name
        )
        roc_pr_path = save_roc_pr_curve(
            y_test,
            metrics["results_df"]["churn_prob"],
            name
        )

        metrics["roc_pr_path"] = roc_pr_path
        metrics["cm_path"] = cm_path
        metrics["model"] = model
        results[name] = metrics

    return results
def save_confusion_matrix(cm, model_name):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    path = f"outputs/plots/cm_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
def save_roc_pr_curve(y_true, y_prob, model_name):
    os.makedirs("outputs/plots", exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot(recall, precision, label="PR")
    plt.xlabel("False Positive Rate / Recall")
    plt.ylabel("True Positive Rate / Precision")
    plt.legend()
    plt.title(f"ROC & PR - {model_name}")

    path = f"outputs/plots/roc_pr_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path
