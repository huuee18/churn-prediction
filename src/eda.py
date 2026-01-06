# -*- coding: utf-8 -*-
"""
EDA Script for Churn Prediction – Insurance Dataset
Author: Thesis Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# CONFIG
# ===============================
DATA_PATH = "/content/drive/MyDrive/Luận văn/project/data/Sum 1.csv"
TARGET_COL = "CHURN"
TIME_COL = "YEAR_MONTH_TRANS"  # None nếu không có

OUTPUT_DIR = "eda_report"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

# ===============================
# OVERVIEW
# ===============================
overview = {
    "num_rows": df.shape[0],
    "num_columns": df.shape[1],
    "num_numerical": len(df.select_dtypes(include=["int64", "float64"]).columns),
    "num_categorical": len(df.select_dtypes(include=["object"]).columns),
}

pd.DataFrame([overview]).to_csv(
    os.path.join(OUTPUT_DIR, "overview.csv"), index=False
)

# ===============================
# MISSING VALUES
# ===============================
missing_report = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_ratio": df.isna().mean()
}).sort_values("missing_ratio", ascending=False)

missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "missing_report.csv")
)

# ===============================
# FEATURE TYPES
# ===============================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# ===============================
# NUMERICAL SUMMARY
# ===============================
num_summary = df[num_cols].describe().T
num_summary.to_csv(
    os.path.join(OUTPUT_DIR, "numerical_summary.csv")
)

# ===============================
# CATEGORICAL SUMMARY
# ===============================
cat_summary = []

for col in cat_cols:
    cat_summary.append({
        "feature": col,
        "unique_values": df[col].nunique(),
        "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None
    })

pd.DataFrame(cat_summary).to_csv(
    os.path.join(OUTPUT_DIR, "categorical_summary.csv"),
    index=False
)

# ===============================
# TARGET DISTRIBUTION
# ===============================
if TARGET_COL in df.columns:
    churn_dist = df[TARGET_COL].value_counts(normalize=True)
    churn_dist.to_csv(
        os.path.join(OUTPUT_DIR, "churn_distribution.csv")
    )

    plt.figure(figsize=(6,4))
    df[TARGET_COL].value_counts().plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("CHURN")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_distribution.png"))
    plt.close()

# ===============================
# NUMERICAL vs CHURN
# ===============================
if TARGET_COL in df.columns:
    for col in num_cols:
        if col == TARGET_COL:
            continue

        plt.figure(figsize=(6,4))
        sns.boxplot(x=TARGET_COL, y=col, data=df)
        plt.title(f"{col} vs CHURN")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"boxplot_{col}.png"))
        plt.close()

# ===============================
# CORRELATION
# ===============================
corr = df[num_cols].corr()
corr.to_csv(
    os.path.join(OUTPUT_DIR, "correlation.csv")
)

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
plt.close()

# ===============================
# TIME ANALYSIS
# ===============================
if TIME_COL in df.columns and TARGET_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    churn_time = df.groupby(TIME_COL)[TARGET_COL].mean()

    plt.figure(figsize=(10,4))
    churn_time.plot()
    plt.title("Churn Rate Over Time")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_over_time.png"))
    plt.close()

print("✅ EDA completed. Reports saved to:", OUTPUT_DIR)
