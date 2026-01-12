# =========================================
# IMPORTS
# =========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
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

from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve


# =========================================
# CLASS WEIGHT (IMBALANCE HANDLING)
# =========================================
def compute_class_weights(y):
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return {0: weights[0], 1: weights[1]}


# =========================================
# THRESHOLD OPTIMIZATION
# =========================================
def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            continue

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


# =========================================
# EVALUATION
# =========================================
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
    """

    # ---------- Predict ----------
    y_prob = model.predict(X_test).ravel()

    # ---------- Prediction ----------
    y_pred = (y_prob >= threshold).astype(int)

    # ---------- Debug ----------
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
    recall = recall_score(y_test, y_pred, zero_division=0)
    balanced_acc = 0.5 * (recall + specificity)

    # ---------- Metrics ----------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "results_df": results_df,
        "threshold": threshold
    }

    # ---------- AUC ----------
    if len(np.unique(y_test)) == 2:
        metrics["auc_roc"] = roc_auc_score(y_test, y_prob)
        metrics["auc_pr"] = average_precision_score(y_test, y_prob)
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"] = np.nan

    return metrics


# =========================================
# TRAIN LOOP
# =========================================


def train_ts_models(
    models,
    X_train, y_train,
    X_test, y_test,
    epochs=10,
    batch_size=32,
    threshold_metric="f1"
):

    results = {}

    # =====================================================
    # 1. Prepare labels (flatten for class_weight)
    # =====================================================
    y_train_flat = y_train.reshape(-1)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_flat),
        y=y_train_flat
    )

    class_weight_dict = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train_flat), class_weights)
    }

    print("⚖️ Class weights:", class_weight_dict)

    # =====================================================
    # 2. Train each model
    # =====================================================
    for name, model in models.items():
        print(f"\n🔵 Training model: {name}")

        # -------------------------------
        # Train
        # -------------------------------
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=1
        )

        # -------------------------------
        # Predict probabilities
        # -------------------------------
        y_prob = model.predict(X_test).ravel()

        # -------------------------------
        # Find optimal threshold
        # -------------------------------
        best_t, best_score = find_best_threshold(
            y_test,
            y_prob,
            metric=threshold_metric
        )

        print(
            f"🎯 Optimal threshold ({threshold_metric}) = "
            f"{best_t:.2f} | score = {best_score:.4f}"
        )

        # -------------------------------
        # Evaluation at optimal threshold
        # -------------------------------
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=best_t
        )

        # =================================================
        # 3. Diagnostics & plots
        # =================================================
        metrics["optimal_threshold"] = best_t
        metrics["threshold_metric"] = threshold_metric

        metrics["prob_dist_path"] = save_prob_distribution(
            y_test,
            metrics["results_df"]["churn_prob"],
            model_name=name
        )

        metrics["ks_statistic"] = compute_ks_statistic(
            y_test,
            metrics["results_df"]["churn_prob"]
        )

        metrics["calibration_path"] = save_calibration_curve(
            y_test,
            metrics["results_df"]["churn_prob"],
            model_name=name
        )

        metrics["cm_path"] = save_confusion_matrix(
            metrics["confusion_matrix"],
            model_name=name
        )

        metrics["roc_pr_path"] = save_roc_pr_curve(
            y_test,
            metrics["results_df"]["churn_prob"],
            model_name=name
        )

        # =================================================
        # 4. Save results
        # =================================================
        results[name] = {
            "model": model,
            "history": history,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,

            # PREDICTIONS (QUAN TRỌNG)
            "y_prob": y_prob,
            "optimal_threshold": best_t,
            "metrics": metrics
        }

    return results




# =========================================
# PLOTS
# =========================================
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


def save_calibration_curve(y_true, y_prob, model_name, n_bins=10):
    os.makedirs("outputs/plots", exist_ok=True)

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed churn rate")
    plt.legend()
    plt.title(f"Calibration Curve - {model_name}")

    path = f"outputs/plots/calibration_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def compute_ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob")

    churn = df[df.y_true == 1]["y_prob"]
    non_churn = df[df.y_true == 0]["y_prob"]

    ks = max(
        abs((churn <= x).mean() - (non_churn <= x).mean())
        for x in df["y_prob"]
    )
    return ks


def save_prob_distribution(y_true, y_prob, model_name):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Non-churn")
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Churn")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Count")
    plt.legend()
    plt.title(f"Churn Probability Distribution - {model_name}")

    path = f"outputs/plots/prob_dist_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
