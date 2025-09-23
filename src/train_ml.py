from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, confusion_matrix
)

def evaluate_model(model, X_test, y_test):
    """Đánh giá model bằng các metric chính"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

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
