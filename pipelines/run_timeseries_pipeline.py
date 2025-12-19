# pipelines/run_timeseries_pipeline.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_data
from src.sequence_data import create_sequences
from src.models import get_timeseries_models
from src.train.train_timeseries import train_ts_models
from src.evaluation.benchmark import compare_models
from src.evaluation.plots import plot_metrics, plot_roc_pr, plot_confusion_matrices
from src.explain.shap_ts import run_shap_for_models
from src.report.pdf_report import generate_pdf_report


# =========================
# CONFIG
# =========================
DATA_PATH = "data/raw/Sum.csv"
OUTPUT_DIR = "outputs"
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =========================
# PIPELINE
# =========================
def run_pipeline():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Preprocess
    df_tree, df_linear, num_cols = preprocess_data(df)

    # 3. Create sequences
    X_seq, y_seq, timesteps = create_sequences(
        df_tree,
        features=num_cols,
        target="CHURN"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 4. Load models
    ts_models = get_timeseries_models()

    # 5. Train
    results_ts = train_ts_models(
        ts_models,
        X_train, y_train,
        X_test, y_test,
        timesteps,
        input_dim=len(num_cols),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 6. Benchmark
    df_benchmark = compare_models(results_ts)
    df_benchmark.to_csv(f"{OUTPUT_DIR}/results/benchmark_results.csv", index=False)

    # 7. Plots
    metrics_path = plot_metrics(df_benchmark, OUTPUT_DIR)
    roc_pr_path = plot_roc_pr(results_ts, y_test, OUTPUT_DIR)
    cm_paths = plot_confusion_matrices(results_ts, OUTPUT_DIR)

    # 8. SHAP
    shap_results = run_shap_for_models(results_ts, X_test, num_cols, OUTPUT_DIR)

    # 9. Save models
    save_models(results_ts, f"{OUTPUT_DIR}/models_ts")

    # 10. PDF report
    pdf_path = f"{OUTPUT_DIR}/reports/final_model_report.pdf"
    generate_pdf_report(
        roc_pr_path,
        [metrics_path],
        cm_paths,
        pdf_path
    )

    print("✅ Pipeline finished successfully!")


def save_models(results_ts, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for name, res in results_ts.items():
        model = res["model"]
        model.save(os.path.join(save_dir, f"{name}.keras"))
        print(f"✅ Saved model: {name}")


if __name__ == "__main__":
    run_pipeline()
