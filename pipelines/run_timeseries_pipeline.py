# pipelines/run_timeseries_pipeline.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# ADD PROJECT ROOT TO PATH
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.preprocess import preprocess_data
from src.models.models import get_timeseries_models
from src.train.train_timeseries import train_ts_models
from src.evaluation.benchmark import compare_models
from src.sequence_builder import create_fixed_window_sequences

# =========================
# CONFIG
# =========================
DATA_PATH = "/content/drive/MyDrive/Luận văn/project/data/Sum.csv"
OUTPUT_DIR = "outputs"
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
WINDOW_SIZE = 6

# =========================
# PIPELINE
# =========================
def run_pipeline():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/models_ts", exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Preprocess
    df, num_cols = preprocess_data(df)

    # 3. Create time-series sequences
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
        random_state=RANDOM_STATE
    )

    # 4. Load models (KHÔNG TRUYỀN THAM SỐ)
    ts_models = get_timeseries_models(
        timesteps=X_train.shape[1],
        input_dim=X_train.shape[2]
    )

    # 5. Train
    results_ts = train_ts_models(
        ts_models,
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 6. Benchmark
    df_benchmark = compare_models(results_ts)

    if df_benchmark is not None:
        df_benchmark.to_csv(
            f"{OUTPUT_DIR}/results/benchmark_timeseries.csv",
            index=False
        )
        print("✅ Benchmark saved")
    else:
        print("⚠️ Benchmark dataframe is None")

    # 7. Save models
    save_models(results_ts, f"{OUTPUT_DIR}/models_ts")

    print("🎉 Time-series pipeline finished successfully!")


def save_models(results_ts, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for name, res in results_ts.items():
        model = res["model"]
        model.save(os.path.join(save_dir, f"{name}.keras"))
        print(f"✅ Saved model: {name}")


if __name__ == "__main__":
    run_pipeline()
