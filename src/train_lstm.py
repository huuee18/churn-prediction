import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, confusion_matrix
)
from src.models import build_lstm_model


def train_lstm(X_train, y_train, X_test, y_test,
               timesteps, input_dim, epochs=20, batch_size=16,
               threshold_high=0.7, threshold_low=0.3):
    """Huấn luyện LSTM, dự đoán xác suất và phân loại risk level"""
    model = build_lstm_model(timesteps, input_dim)
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=1)

    # Dự đoán
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Risk level
    risk_level = []
    for p in y_prob:
        if p >= threshold_high:
            risk_level.append("High churn risk")
        elif p <= threshold_low:
            risk_level.append("Low churn risk")
        else:
            risk_level.append("Medium churn risk")

    # DataFrame kết quả chi tiết
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "churn_prob": y_prob,
        "risk_level": risk_level
    })

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'results_df': results_df
    }
    return model, metrics, y_pred, y_prob
