import pandas as pd

def compare_models(results_ml, results_lstm, lstm_name="LSTM"):
    """So sánh kết quả ML models và LSTM model"""
    rows = []

    # ML models
    for name, res in results_ml.items():
        rows.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'AUC-ROC': res['auc_roc'],
            'AUC-PR': res['auc_pr'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1']
        })

    # LSTM
    rows.append({
        'Model': lstm_name,
        'Accuracy': results_lstm['accuracy'],
        'AUC-ROC': results_lstm['auc_roc'],
        'AUC-PR': results_lstm['auc_pr'],
        'Precision': results_lstm.get('precision', None),  # nếu bạn thêm vào train_lstm
        'Recall': results_lstm.get('recall', None),
        'F1': results_lstm.get('f1', None)
    })

    df_results = pd.DataFrame(rows)
    return df_results
