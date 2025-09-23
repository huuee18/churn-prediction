import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from models import build_lstm_model

def train_lstm(X_train, y_train, X_test, y_test, timesteps, input_dim, 
               epochs=20, batch_size=16, save_prefix="lstm"):
    # 1. Xây dựng model
    model = build_lstm_model(timesteps, input_dim)

    # 2. Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 3. Dự đoán
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # 4. Tính metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob)
    }

    # 5. Lưu model
    model.save(f"{save_prefix}_model.h5")

    # 6. Lưu metrics ra JSON
    with open(f"{save_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 7. Lưu predictions ra CSV
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }).to_csv(f"{save_prefix}_predictions.csv", index=False)

    return model, results, y_pred, y_prob
