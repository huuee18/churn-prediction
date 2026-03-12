import numpy as np
import pandas as pd

from src.evaluation.advanced_metrics import recall_at_k
from src.evaluation.advanced_metrics import compute_ks


def rolling_backtest(
    model_builder,
    X_seq,
    y,
    time_index,
    window_train=12,
    window_test=1,
    recall_ks=[0.05, 0.1, 0.2],
    epochs=10,
    batch_size=32
):
    """
    Rolling backtest for time-series churn models

    IMPORTANT:
    - time_index MUST be aligned 1-1 with X_seq and y
    - length(time_index) == len(X_seq)
    """

    # ===== VALIDATION =====
    if len(time_index) != len(X_seq):
        raise ValueError(
            f"[ERROR] Length mismatch:\n"
            f"- X_seq length: {len(X_seq)}\n"
            f"- time_index length: {len(time_index)}\n\n"
            f"👉 time_index must be created AFTER sequence generation!"
        )

    if len(y) != len(X_seq):
        raise ValueError(
            f"[ERROR] y length ({len(y)}) does not match X_seq length ({len(X_seq)})"
        )

    # Ensure numpy arrays
    X_seq = np.array(X_seq)
    y = np.array(y)
    time_index = pd.Series(time_index).reset_index(drop=True)

    unique_times = sorted(time_index.unique())
    results = []

    print(f"\n🔁 Starting rolling backtest over {len(unique_times)} time periods")

    for i in range(len(unique_times) - window_train - window_test + 1):

        train_times = unique_times[i : i + window_train]
        test_times = unique_times[i + window_train : i + window_train + window_test]

        train_mask = time_index.isin(train_times).values
        test_mask = time_index.isin(test_times).values

        X_train, y_train = X_seq[train_mask], y[train_mask]
        X_test, y_test = X_seq[test_mask], y[test_mask]

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        print(
            f"\n🕒 Train {train_times[0]} → {train_times[-1]} "
            f"| Test {test_times}"
        )

        # =========================
        # Train model
        # =========================
        model = model_builder()

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        y_prob = model.predict(X_test).ravel()

        row = {
            "train_start": train_times[0],
            "train_end": train_times[-1],
            "test_time": test_times[0],
            "ks": compute_ks(y_test, y_prob)
        }

        for k in recall_ks:
            row[f"recall@{int(k*100)}"] = recall_at_k(y_test, y_prob, k)

        row["n_train"] = len(X_train)
        row["n_test"] = len(X_test)

        results.append(row)

    df_result = pd.DataFrame(results)

    print("\n✅ Rolling backtest finished")
    print(df_result.head())

    return df_result
