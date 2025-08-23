from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix

def train_ml_models(models_tree, model_linear, 
                    X_train_tree, y_train_tree, X_test_tree, y_test_tree,
                    X_train_linear, y_train_linear, X_test_linear, y_test_linear):
    results = {}

    # Train tree-based models
    for name, model in models_tree.items():
        model.fit(X_train_tree, y_train_tree)
        y_pred = model.predict(X_test_tree)
        y_prob = model.predict_proba(X_test_tree)[:,1]
        results[name] = {
            'accuracy': accuracy_score(y_test_tree, y_pred),
            'auc_roc': roc_auc_score(y_test_tree, y_prob),
            'auc_pr': average_precision_score(y_test_tree, y_prob),
            'f1': classification_report(y_test_tree, y_pred, output_dict=True)['1']['f1-score'],
            'recall': classification_report(y_test_tree, y_pred, output_dict=True)['1']['recall'],
            'confusion_matrix': confusion_matrix(y_test_tree, y_pred)
        }

    # Logistic Regression
    model_linear.fit(X_train_linear, y_train_linear)
    y_pred = model_linear.predict(X_test_linear)
    y_prob = model_linear.predict_proba(X_test_linear)[:,1]
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test_linear, y_pred),
        'auc_roc': roc_auc_score(y_test_linear, y_prob),
        'auc_pr': average_precision_score(y_test_linear, y_prob),
        'f1': classification_report(y_test_linear, y_pred, output_dict=True)['1']['f1-score'],
        'recall': classification_report(y_test_linear, y_pred, output_dict=True)['1']['recall'],
        'confusion_matrix': confusion_matrix(y_test_linear, y_pred)
    }
    return results
