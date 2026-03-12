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

    print("🔍 Running SHAP for time-series model...")

    # =========================
    # Sample data
    # =========================
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # =========================
    # Background as mean sequence
    # =========================
    background = np.mean(X_sample, axis=0, keepdims=True)

    # =========================
    # Create explainer
    # =========================
    explainer = shap.GradientExplainer(model, background)

    # =========================
    # Compute SHAP values
    # =========================
    shap_values = explainer.shap_values(X_sample)

    # Handle list output (binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    print("Original SHAP shape:", shap_values.shape)

    # =========================
    # Normalize dimension safely
    # =========================

    # Case: (N, T, F, 1) -> remove last dim
    if len(shap_values.shape) == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    # Case: (1, N, T, F) -> remove first dim
    elif len(shap_values.shape) == 4 and shap_values.shape[0] == 1:
        shap_values = shap_values.squeeze(0)

    print("After squeeze SHAP shape:", shap_values.shape)

    if len(shap_values.shape) != 3:
        raise ValueError(
            f"Unexpected SHAP shape after processing: {shap_values.shape}. "
            "Expect (N, T, F)"
        )

    if len(shap_values.shape) == 4:
        # Sometimes returns (1, N, T, F)
        shap_values = shap_values.squeeze(0)

    if len(shap_values.shape) != 3:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}. "
            "Expect (N, T, F)"
        )

    N, T, F = shap_values.shape

    print(f"Processed SHAP shape: N={N}, T={T}, F={F}")

    # Validate feature names
    if len(feature_names) != F:
        print(
            f"⚠️ Feature names length mismatch: "
            f"{len(feature_names)} vs SHAP features {F}"
        )
        feature_names = feature_names[:F]

    # =========================
    # Aggregate importance
    # =========================
    shap_abs = np.abs(shap_values)

    # Mean over samples -> (T, F)
    shap_mean = shap_abs.mean(axis=0)

    # Feature importance over all timesteps -> (F,)
    shap_feat = shap_mean.mean(axis=0)

    # Ensure 1D
    shap_feat = shap_feat.flatten()

    # =========================
    # Plot feature importance
    # =========================
    plt.figure(figsize=(10, 6))

    y_pos = np.arange(len(feature_names))

    plt.barh(y_pos, shap_feat)
    plt.yticks(y_pos, feature_names)
    plt.title("SHAP Feature Importance (Avg over timesteps)")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    # =========================
    # Heatmap by timestep
    # =========================
    plt.figure(figsize=(12, 7))

    sns.heatmap(
        shap_mean.T,
        cmap="RdBu_r",
        center=0,
        xticklabels=[f"T-{i}" for i in range(T, 0, -1)],
        yticklabels=feature_names
    )

    plt.title("SHAP Importance by Timestep")
    plt.xlabel("Time")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(save_path.replace(".png", "_heatmap.png"))
    plt.close()

    print(f"✅ SHAP saved to {save_path}")

    return save_path

