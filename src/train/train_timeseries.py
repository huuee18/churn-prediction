import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ================================
# EVALUATION
# ================================
# ================================
# EVALUATION
# ================================
def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=0.5,
    threshold_high=0.7,
    threshold_low=0.3
):
    """
    Evaluate time-series churn prediction models
    (TSMixer / LSTM / GRU / CNN / NBEATS / DLinear)
    """

    # ---------- Predict ----------
    y_raw = model.predict(X_test)
    y_prob = np.squeeze(y_raw)

    # Convert logits -> probability if needed
    if y_prob.min() < 0 or y_prob.max() > 1:
        y_prob = 1 / (1 + np.exp(-y_prob))

    y_pred = (y_prob >= threshold).astype(int)

    # ---------- Debug distribution ----------
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("Prediction distribution:", dict(zip(unique_pred, counts_pred)))

    # ---------- Risk segmentation ----------
    risk_level = np.where(
        y_prob >= threshold_high, "High churn risk",
        np.where(y_prob <= threshold_low, "Low churn risk", "Medium churn risk")
    )

    # ---------- Result dataframe ----------
    results_df = pd.DataFrame({
        "y_true": y_test,
        "churn_prob": y_prob,
        "y_pred": y_pred,
        "risk_level": risk_level
    })

    # ---------- Confusion matrix ----------
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (recall_score(y_test, y_pred, zero_division=0) + specificity)

    # ---------- Metrics ----------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "results_df": results_df,
        "threshold": threshold
    }

    # ---------- AUC (safe) ----------
    if len(np.unique(y_test)) == 2:
        metrics["auc_roc"] = roc_auc_score(y_test, y_prob)
        metrics["auc_pr"] = average_precision_score(y_test, y_prob)
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"] = np.nan

    return metrics



# ================================
# TRAIN LOOP
# ================================
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

        metrics["cm_path"] = save_confusion_matrix(
            metrics["confusion_matrix"], name
        )

        metrics["roc_pr_path"] = save_roc_pr_curve(
            y_test,
            metrics["results_df"]["churn_prob"],
            name
        )

        metrics["model"] = model
        results[name] = metrics

    return results


# ================================
# PLOTS
# ================================
def save_confusion_matrix(cm, model_name):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(4, 4))
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

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot(recall, precision, label="PR")
    plt.legend()
    plt.title(f"ROC & PR - {model_name}")

    path = f"outputs/plots/roc_pr_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path
