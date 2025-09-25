import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, confusion_matrix
)


def evaluate_model(model, X_test, y_test,
                   threshold_high=0.7, threshold_low=0.3):
    """Đánh giá model và lưu cả dự đoán với risk level"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Xác định mức độ rủi ro
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
        'results_df': results_df  # lưu DataFrame để xuất/ghi file
    }
    return metrics


def train_ml_models(models_tree, model_linear,
                    X_train_tree, y_train_tree, X_test_tree, y_test_tree,
                    X_train_linear, y_train_linear, X_test_linear, y_test_linear):
    results = {}

    # Train tree-based models
    for name, model in models_tree.items():
        model.fit(X_train_tree, y_train_tree)
        results[name] = evaluate_model(model, X_test_tree, y_test_tree)

    # Logistic Regression
    model_linear.fit(X_train_linear, y_train_linear)
    results['Logistic Regression'] = evaluate_model(model_linear, X_test_linear, y_test_linear)

    return results
