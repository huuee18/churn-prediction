import pandas as pd

def compare_models(results_ts):
    """
    So sánh tất cả các mô hình time-series.
    
    results_ts: dict chứa kết quả của nhiều mô hình
    {
        "LSTM_Attention": {...},
        "GRU": {...},
        "BiLSTM": {...},
        "Transformer": {...},
        "TCN": {...}
    }
    """
    rows = []

    for name, res in results_ts.items():
        rows.append({
            "Model": name,
            "Accuracy": res.get("accuracy"),
            "AUC-ROC": res.get("roc_auc"),
            "AUC-PR": res.get("auc_pr"),
            "Precision": res.get("precision"),
            "Recall": res.get("recall"),
            "F1": res.get("f1")
        })

    df_results = pd.DataFrame(rows)
    return df_results
