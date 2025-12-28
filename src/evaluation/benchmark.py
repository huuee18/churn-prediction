import pandas as pd

def compare_models(results_ts):
    if not results_ts:
        raise ValueError("results_ts is empty")

    rows = []

    for name, res in results_ts.items():
        rows.append({
            "Model": name,
            "Accuracy": res.get("accuracy"),
            "AUC-ROC": res.get("auc_roc"),   # ✅ sửa key
            "AUC-PR": res.get("auc_pr"),
            "Precision": res.get("precision"),
            "Recall": res.get("recall"),
            "F1": res.get("f1")
        })

    df_results = pd.DataFrame(rows)

    metric_cols = ["Accuracy", "AUC-ROC", "AUC-PR", "Precision", "Recall", "F1"]
    df_results[metric_cols] = df_results[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return df_results   # ✅ DÒNG QUAN TRỌNG NHẤT
