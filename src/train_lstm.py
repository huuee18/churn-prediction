import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from src.models import build_lstm_model

def train_lstm(X_train, y_train, X_test, y_test, timesteps, input_dim, epochs=20, batch_size=16):
    model = build_lstm_model(timesteps, input_dim)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    y_prob = model.predict(X_test).flatten()

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,      # lưu lại để không phải predict lại
        'y_prob': y_prob       # dùng vẽ ROC/PR
    }
    return model, results, y_pred, y_prob
