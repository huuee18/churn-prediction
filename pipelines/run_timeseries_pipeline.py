# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from src.evaluation.model_efficiency import benchmark_efficiency

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

# ========== CẢI TIẾN 1: Tăng epochs cho training thật ==========
EPOCHS = 20  # Từ 1 lên 50-100 để model học tốt
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
WINDOW_SIZE = 6

# ========== CẢI TIẾN 2: Early stopping config ==========
EARLY_STOPPING_PATIENCE = 5  # Dừng nếu validation không cải thiện sau 5 epochs
USE_EARLY_STOPPING = True

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
    os.makedirs(f"{OUTPUT_DIR}/training_logs", exist_ok=True)

    # -------------------------
    # 1. Load data
    # -------------------------
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

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
    print(f"Churn ratio - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")

    # -------------------------
    # 4. Load models
    # -------------------------
    ts_models = get_timeseries_models(
        timesteps=X_train.shape[1],
        input_dim=X_train.shape[2]
    )

    # -------------------------
    # 5. Train với early stopping
    # -------------------------
    print("🚀 Training models...")
    print(f"⚙️ Config: Epochs={EPOCHS}, Early Stopping Patience={EARLY_STOPPING_PATIENCE if USE_EARLY_STOPPING else 'OFF'}")
    
    results_ts = train_ts_models(
        ts_models,
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        #use_early_stopping=USE_EARLY_STOPPING,
        #early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    # ========== CẢI TIẾN 3: In và lưu thời gian training ==========
    print("\n" + "="*60)
    print("⏱️ TRAINING TIME SUMMARY")
    print("="*60)
    
    training_time_df = []
    for name, res in results_ts.items():
        training_time = res.get("training_time_sec", 0)
        epochs_trained = res.get("epochs_trained", EPOCHS)
        print(f"{name:20} | {training_time:8.2f} seconds | {epochs_trained:3d} epochs")
        training_time_df.append({
            "Model": name,
            "Training_Time_sec": training_time,
            "Training_Time_min": training_time / 60,
            "Epochs_Trained": epochs_trained,
            "Time_per_Epoch_sec": training_time / epochs_trained if epochs_trained > 0 else 0
        })
    
    # Lưu bảng thời gian training
    df_training_time = pd.DataFrame(training_time_df)
    df_training_time.to_csv(f"{OUTPUT_DIR}/results/training_time.csv", index=False)
    print("\n✅ Training time saved to outputs/results/training_time.csv")
    print("="*60)

    # -------------------------
    # 6. Standard benchmark
    # -------------------------
    print("\n📊 Benchmarking models...")
    print("⚙️ Measuring model efficiency...")

    df_efficiency = benchmark_efficiency(
        ts_models,
        X_test
    )

    df_efficiency.to_csv(
        "outputs/results/model_efficiency.csv",
        index=False
    )
    print("✅ Efficiency metrics saved")

    # ========== CẢI TIẾN 4: Lưu benchmark chi tiết hơn ==========
    df_benchmark = compare_models(results_ts)

    if df_benchmark is not None:
        df_benchmark.to_csv("outputs/results/benchmark_ts.csv", index=False)
        print("✅ Benchmark saved to outputs/results/benchmark_ts.csv")
        
        # In bảng benchmark ra console
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        print(df_benchmark[['Model', 'AUC-ROC', 'AUC-PR', 'F1', 'KS', 'Optimal_Threshold']].to_string(index=False))
        print("="*80)
    else:
        print("❌ No benchmark table generated (no model has metrics)")

    # -------------------------
    # 7. Advanced evaluation
    # -------------------------
    print("\n🔎 Running advanced evaluation...")
    advanced_rows = []

    for name, res in results_ts.items():
        print(f"\n{'='*40}")
        print(f"📈 Advanced evaluation: {name}")
        print(f"{'='*40}")

        model = res["model"]
        X_test = res["X_test"]
        y_test = res["y_test"].reshape(-1)
        y_prob = res["y_prob"]
        threshold = res["optimal_threshold"]
        
        # Thêm training time vào advanced metrics
        training_time = res.get("training_time_sec", 0)

        # ---- Recall@Top-K ----
        recall_k_results = {}
        for k in TOP_K_LIST:
            r = recall_at_k(y_test, y_prob, k)
            recall_k_results[f"Recall@{int(k*100)}%"] = r
            print(f"  Recall@{int(k*100)}% = {r:.4f}")

        # ---- Lift chart ----
        _, lift_path = plot_lift_chart(
            y_test,
            y_prob,
            save_path=f"{OUTPUT_DIR}/figures/lift_{name}.png"
        )
        print(f"  📊 Lift chart saved: {lift_path}")

        # ---- KS statistic ----
        ks = compute_ks(y_test, y_prob)
        print(f"  KS = {ks:.4f}")

        # ---- SHAP analysis ----
        try:
            shap_path = shap_timeseries(
                model,
                X_test,
                feature_names=num_cols,
                save_path=f"{OUTPUT_DIR}/figures/shap_{name}.png"
            )
            print(f"  🎯 SHAP analysis saved: {shap_path}")
        except Exception as e:
            print(f"  ⚠️ SHAP failed for {name}: {e}")

        advanced_rows.append({
            "Model": name,
            "Training_Time_sec": training_time,
            "Training_Time_min": training_time / 60,
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
    print("\n✅ Advanced metrics saved to outputs/results/advanced_metrics_timeseries.csv")

    # ========== CẢI TIẾN 5: Lưu summary report ==========
    create_summary_report(results_ts, df_benchmark, df_advanced, df_training_time, OUTPUT_DIR)

    # -------------------------
    # 8. Save models
    # -------------------------
    save_models(results_ts, f"{OUTPUT_DIR}/models_ts")

    print("\n" + "🎉"*30)
    print("TIME-SERIES PIPELINE FINISHED SUCCESSFULLY!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("🎉"*30)


# =========================
# SAVE MODELS
# =========================
def save_models(results_ts, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for name, res in results_ts.items():
        model = res["model"]
        model.save(os.path.join(save_dir, f"{name}.keras"))
        print(f"💾 Saved model: {name}")


# ========== CẢI TIẾN 6: Tạo báo cáo tổng hợp ==========
def create_summary_report(results_ts, df_benchmark, df_advanced, df_training_time, output_dir):
    """Tạo báo cáo tổng hợp Markdown cho luận văn"""
    
    report_path = os.path.join(output_dir, "results", "FINAL_REPORT.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# BÁO CÁO TỔNG HỢP - DỰ ĐOÁN CHURN\n\n")
        f.write(f"*Ngày tạo: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # 1. Model Performance
        f.write("## 1. HIỆU SUẤT CÁC MÔ HÌNH\n\n")
        if df_benchmark is not None:
            f.write("### Bảng 1: So sánh các mô hình\n\n")
            f.write(df_benchmark[['Model', 'AUC-ROC', 'AUC-PR', 'F1', 'KS', 'Optimal_Threshold']].to_markdown())
            f.write("\n\n")
        
        # 2. Training Time
        f.write("## 2. THỜI GIAN HUẤN LUYỆN\n\n")
        f.write("### Bảng 2: Thời gian training\n\n")
        f.write(df_training_time.to_markdown(index=False))
        f.write("\n\n")
        
        # 3. Advanced Metrics
        f.write("## 3. METRICS NÂNG CAO\n\n")
        f.write("### Bảng 3: Recall@Top-K và KS\n\n")
        f.write(df_advanced.to_markdown(index=False))
        f.write("\n\n")
        
        # 4. Best Model Recommendation
        if df_benchmark is not None:
            best_model = df_benchmark.loc[df_benchmark['AUC-ROC'].idxmax(), 'Model']
            best_auc = df_benchmark['AUC-ROC'].max()
            
            f.write("## 4. KHUYẾN NGHỊ\n\n")
            f.write(f"- **Mô hình tốt nhất**: {best_model}\n")
            f.write(f"- **AUC-ROC đạt được**: {best_auc:.4f}\n")
            f.write(f"- **Lý do**: Mô hình có khả năng phân biệt churn/non-churn tốt nhất\n\n")
        
        f.write("---\n")
        f.write("*Báo cáo được tạo tự động từ pipeline*\n")
    
    print(f"📄 Summary report saved: {report_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_pipeline()
