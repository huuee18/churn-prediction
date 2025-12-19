import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)
def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=0.5,
    threshold_high=0.7,
    threshold_low=0.3
):
    """
    Đánh giá model deep learning cho churn prediction
    Áp dụng cho TSMixer / N-BEATS / DLinear
    """

    # -------- Predict probability --------
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # -------- Risk segmentation --------
    risk_level = np.where(
        y_prob >= threshold_high, "High churn risk",
        np.where(y_prob <= threshold_low, "Low churn risk", "Medium churn risk")
    )

    # -------- Result DataFrame --------
    results_df = pd.DataFrame({
        "y_true": y_test,
        "churn_prob": y_prob,
        "y_pred": y_pred,
        "risk_level": risk_level
    })

    # -------- Metrics --------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "results_df": results_df
    }

    return metrics
def train_ts_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=20,
    batch_size=64,
    threshold=0.5
):
    """
    Training loop cho time-series deep models
    """

    results = {}

    for name, build_fn in models.items():
        print(f"\n🔵 Training model: {name}")

        model = build_fn(
            timesteps=X_train.shape[1],
            input_dim=X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            threshold=threshold
        )

        metrics["model"] = model
        results[name] = metrics

    return results
