# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, average_precision_score


# =====================================================
# 1. Recall@Top-K
# =====================================================
def recall_at_k(y_true, y_prob, k_ratio=0.1):
    """
    Recall@Top-K%
    """
    n_top = int(len(y_prob) * k_ratio)
    idx = np.argsort(y_prob)[::-1][:n_top]

    y_true_top = y_true[idx]
    recall_k = y_true_top.sum() / y_true.sum()

    return recall_k


# =====================================================
# 2. Lift Chart
# =====================================================
def plot_lift_chart(
    y_true,
    y_prob,
    n_bins=10,
    save_path="outputs/figures/lift_chart.png"
):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).sort_values("y_prob", ascending=False)

    df["bin"] = pd.qcut(
        df.index,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    lift_df = df.groupby("bin").agg(
        churn_rate=("y_true", "mean"),
        count=("y_true", "count")
    ).reset_index()

    baseline = y_true.mean()
    lift_df["lift"] = lift_df["churn_rate"] / baseline

    plt.figure(figsize=(7,4))
    plt.plot(lift_df["lift"], marker="o")
    plt.axhline(1, linestyle="--", color="gray")
    plt.title("Lift Chart")
    plt.xlabel("Decile (High → Low risk)")
    plt.ylabel("Lift")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return lift_df, save_path


# =====================================================
# 3. KS Statistic
# =====================================================
def compute_ks(y_true, y_prob):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).sort_values("y_prob")

    df["cum_pos"] = df["y_true"].cumsum() / df["y_true"].sum()
    df["cum_neg"] = (1 - df["y_true"]).cumsum() / (1 - df["y_true"]).sum()

    ks = np.max(np.abs(df["cum_pos"] - df["cum_neg"]))
    return ks


# =====================================================
# 4. SHAP theo timestep (Time-series)
# =====================================================
def shap_timeseries(
    model,
    X,
    feature_names,
    sample_size=200,
    save_path="outputs/figures/shap_timeseries.png"
):
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # =========================
    # Sample
    # =========================
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Background = mean sequence
    background = np.mean(X_sample, axis=0, keepdims=True)

    # =========================
    # SAFE EXPLAINER
    # =========================
    explainer = shap.GradientExplainer(
        model,
        background
    )

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # binary class

    # (N, T, F)
    shap_abs = np.abs(shap_values)
    shap_mean = shap_abs.mean(axis=0)  # (T, F)

    # =========================
    # Feature importance
    # =========================
    shap_feat = shap_mean.mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_feat)
    plt.title("SHAP Feature Importance (Avg over timesteps)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # =========================
    # Time heatmap
    # =========================
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        shap_mean.T,
        cmap="RdBu_r",
        center=0,
        xticklabels=[f"T-{i}" for i in range(shap_mean.shape[0], 0, -1)],
        yticklabels=feature_names
    )
    plt.title("SHAP Importance by Timestep")
    plt.xlabel("Time")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_heatmap.png"))
    plt.close()

    return save_path

