# =========================================================
# MONITORING PIPELINE
# Prediction Drift + Delayed Label Evaluation
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from scipy.stats import ks_2samp


# =========================================================
# 1. SNAPSHOT DATA AT T-30
# =========================================================
def create_snapshot(df, snapshot_month):
    """
    Create snapshot at time T-30
    """
    snap = df[df["YEAR_MONTH_TRANS"] <= snapshot_month].copy()
    snap = snap.sort_values("YEAR_MONTH_TRANS")
    snap = snap.groupby("POL_NUM").tail(1)
    return snap


# =========================================================
# 2. INFERENCE AT SNAPSHOT
# =========================================================
def run_inference_snapshot(model, X_snapshot, meta_df, threshold=0.5):
    """
    Run inference for snapshot data
    """
    y_prob = np.squeeze(model.predict(X_snapshot))

    snapshot_pred = meta_df.copy()
    snapshot_pred["churn_prob"] = y_prob
    snapshot_pred["predicted_label"] = (y_prob >= threshold).astype(int)

    return snapshot_pred


# =========================================================
# 3. DELAYED LABEL GENERATION
# =========================================================
def generate_actual_labels(snapshot_pred, df_current):
    """
    Generate actual churn labels using current contract status
    """
    active_pols = set(df_current["POL_NUM"].unique())

    snapshot_pred["actual_churn"] = (
        ~snapshot_pred["POL_NUM"].isin(active_pols)
    ).astype(int)

    return snapshot_pred


# =========================================================
# 4. DELAYED LABEL EVALUATION
# =========================================================
def delayed_label_evaluation(df_eval):
    """
    Evaluate model after delayed labels are available
    """
    return {
        "AUC_ROC": roc_auc_score(
            df_eval["actual_churn"],
            df_eval["churn_prob"]
        ),
        "Precision": precision_score(
            df_eval["actual_churn"],
            df_eval["predicted_label"],
            zero_division=0
        ),
        "Recall": recall_score(
            df_eval["actual_churn"],
            df_eval["predicted_label"],
            zero_division=0
        ),
        "F1": f1_score(
            df_eval["actual_churn"],
            df_eval["predicted_label"],
            zero_division=0
        )
    }


# =========================================================
# 5. PREDICTION DRIFT METRICS
# =========================================================
def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI)
    """
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )

    return psi


def compute_ks(expected, actual):
    """
    KS statistic for prediction drift
    """
    ks_stat, p_value = ks_2samp(expected, actual)
    return ks_stat, p_value


# =========================================================
# 6. VISUALIZATION
# =========================================================
def plot_prediction_distribution(train_prob, snapshot_prob, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(train_prob, bins=50, alpha=0.5, label="Train")
    plt.hist(snapshot_prob, bins=50, alpha=0.5, label="Snapshot (T-30)")
    plt.legend()
    plt.title("Prediction Drift - Churn Probability")
    plt.xlabel("Churn Probability")
    plt.ylabel("Count")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =========================================================
# 7. MAIN MONITORING PIPELINE
# =========================================================
def run_monitoring_pipeline(
    model,
    df_history,
    df_current,
    X_snapshot,
    X_train,
    snapshot_month,
    output_dir="outputs/monitoring"
):
    """
    Full monitoring pipeline
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----- Snapshot -----
    snapshot_df = create_snapshot(df_history, snapshot_month)

    # ----- Inference -----
    snapshot_pred = run_inference_snapshot(
        model,
        X_snapshot,
        snapshot_df
    )

    snapshot_pred.to_csv(
        f"{output_dir}/predictions_T_minus_30.csv",
        index=False
    )

    # ----- Delayed label -----
    snapshot_eval = generate_actual_labels(
        snapshot_pred,
        df_current
    )

    delayed_metrics = delayed_label_evaluation(snapshot_eval)

    # ----- Drift metrics -----
    train_prob = np.squeeze(model.predict(X_train))

    psi = compute_psi(train_prob, snapshot_pred["churn_prob"])
    ks_stat, ks_pvalue = compute_ks(
        train_prob,
        snapshot_pred["churn_prob"]
    )

    # ----- Visualization -----
    plot_prediction_distribution(
        train_prob,
        snapshot_pred["churn_prob"],
        f"{output_dir}/prediction_drift.png"
    )

    # ----- Monitoring report -----
    report = {
        "Snapshot_Month": snapshot_month,
        "PSI": psi,
        "KS_Statistic": ks_stat,
        "KS_pvalue": ks_pvalue,
        **delayed_metrics
    }

    report_df = pd.DataFrame([report])
    report_df.to_csv(
        f"{output_dir}/monitoring_report.csv",
        index=False
    )

    return report_df
