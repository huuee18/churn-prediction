import pandas as pd
from src.evaluation.evaluate import evaluate_binary_model


def compare_models(results):
    rows = []

    for name, res in results.items():
        model = res["model"]
        X_test = res["X_test"]
        y_test = res["y_test"]

        metrics = evaluate_binary_model(
            model,
            X_test,
            y_test,
            threshold_strategy="optimal"
        )

        res["metrics"] = metrics


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
