# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# ADD PROJECT ROOT TO PATH
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# =========================
# IMPORTS
# =========================
from src.preprocess import preprocess_data
from src.models.models import get_timeseries_models
from src.train.train_timeseries import train_ts_models
from src.sequence_builder import create_fixed_window_sequences

from src.evaluation.benchmark import compare_models
from src.evaluation.advanced_metrics import (
    recall_at_k,
    plot_lift_chart,
    compute_ks,
    shap_timeseries
)

# =========================
# CONFIG
# =========================
DATA_PATH = "/content/drive/MyDrive/Luận văn/project/data/Sum 4.csv"
OUTPUT_DIR = "outputs"

EPOCHS = 15
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
WINDOW_SIZE = 6

TOP_K_LIST = [0.05, 0.1, 0.2]  # Recall@Top-K

# =========================
# PIPELINE
# =========================
def run_pipeline():

    # -------------------------
    # Create output folders
    # -------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/models_ts", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

    # -------------------------
    # 1. Load data
    # -------------------------
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # -------------------------
    # 2. Preprocess
    # -------------------------
    print("🧹 Preprocessing...")
    df, num_cols = preprocess_data(df)

    # -------------------------
    # 3. Build time-series sequences
    # -------------------------
    print("⏱️ Building time-series sequences...")
    X_seq, y_seq = create_fixed_window_sequences(
        df,
        features=num_cols,
        target="CHURN",
        window_size=WINDOW_SIZE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_seq
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -------------------------
    # 4. Load models
    # -------------------------
    ts_models = get_timeseries_models(
        timesteps=X_train.shape[1],
        input_dim=X_train.shape[2]
    )

    # -------------------------
    # 5. Train (ONE TIME)
    # -------------------------
    print("🚀 Training models...")
    results_ts = train_ts_models(
        ts_models,
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # -------------------------
    # 6. Standard benchmark
    # -------------------------
    print("📊 Benchmarking models...")
    df_benchmark = compare_models(results_ts)

    df_benchmark.to_csv(
        f"{OUTPUT_DIR}/results/benchmark_timeseries.csv",
        index=False
    )
    print("✅ Benchmark saved")

    # -------------------------
    # 7. Advanced evaluation
    # -------------------------
    advanced_rows = []

    for name, res in results_ts.items():
        print(f"\n🔎 Advanced evaluation: {name}")

        model = res["model"]
        X_test = res["X_test"]
        y_test = res["y_test"].reshape(-1)
        y_prob = res["y_prob"]
        threshold = res["optimal_threshold"]

        # ---- Recall@Top-K ----
        recall_k_results = {}
        for k in TOP_K_LIST:
            r = recall_at_k(y_test, y_prob, k)
            recall_k_results[f"Recall@{int(k*100)}%"] = r
            print(f"Recall@{int(k*100)}% = {r:.4f}")

        # ---- Lift chart ----
        _, lift_path = plot_lift_chart(
            y_test,
            y_prob,
            save_path=f"{OUTPUT_DIR}/figures/lift_{name}.png"
        )

        # ---- KS statistic ----
        ks = compute_ks(y_test, y_prob)
        print(f"KS = {ks:.4f}")

        # ---- SHAP theo timestep ----
        shap_path = shap_timeseries(
            model,
            X_test,
            feature_names=num_cols,
            save_path=f"{OUTPUT_DIR}/figures/shap_{name}.png"
        )

        advanced_rows.append({
            "Model": name,
            "Optimal_Threshold": threshold,
            "KS": ks,
            **recall_k_results
        })

    # Save advanced metrics
    df_advanced = pd.DataFrame(advanced_rows)
    df_advanced.to_csv(
        f"{OUTPUT_DIR}/results/advanced_metrics_timeseries.csv",
        index=False
    )

    print("✅ Advanced metrics saved")

    # -------------------------
    # 8. Save models
    # -------------------------
    save_models(results_ts, f"{OUTPUT_DIR}/models_ts")

    print("\n🎉 Time-series pipeline FINISHED SUCCESSFULLY!")


# =========================
# SAVE MODELS
# =========================
def save_models(results_ts, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for name, res in results_ts.items():
        model = res["model"]
        model.save(os.path.join(save_dir, f"{name}.keras"))
        print(f"💾 Saved model: {name}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_pipeline()
