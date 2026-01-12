import pandas as pd


def compare_models(results):
    """
    Compare models using evaluation metrics
    computed at optimal threshold
    """

    rows = []

    for name, res in results.items():
        metrics = res.get("metrics", {})

        if not metrics:
            print(f"⚠️ Skip model {name}: no metrics found")
            continue

        rows.append({
            "Model": name,
            "Accuracy": metrics.get("accuracy"),
            "AUC-ROC": metrics.get("auc_roc"),
            "AUC-PR": metrics.get("auc_pr"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1": metrics.get("f1"),
            "KS": metrics.get("ks_statistic"),
            "Optimal_Threshold": metrics.get("optimal_threshold")
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Sort by business-relevant metric
    df = df.sort_values("AUC-PR", ascending=False)

    return df
