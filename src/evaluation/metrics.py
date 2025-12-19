import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_pr(models, model_linear, 
                X_test_tree, y_test_tree, 
                X_test_linear, y_test_linear, save_path='outputs/figures/roc_pr.png'):
    plt.figure(figsize=(12,5))

    # ROC
    plt.subplot(1,2,1)
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_tree)[:,1]
        fpr, tpr, _ = roc_curve(y_test_tree, y_prob)
        plt.plot(fpr, tpr, label=name)
    fpr, tpr, _ = roc_curve(y_test_linear, model_linear.predict_proba(X_test_linear)[:,1])
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curves'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()

    # PR
    plt.subplot(1,2,2)
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_tree)[:,1]
        precision, recall, _ = precision_recall_curve(y_test_tree, y_prob)
        plt.plot(recall, precision, label=name)
    precision, recall, _ = precision_recall_curve(y_test_linear, model_linear.predict_proba(X_test_linear)[:,1])
    plt.plot(recall, precision, label='Logistic Regression')
    plt.title('Precision-Recall Curves'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_feature_importance(model, features, top_n=15, save_path='feature_importance.png'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(6,6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Top Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_confusion_matrix(cm, title, save_path='cm.png'):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
