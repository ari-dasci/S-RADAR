import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def measure(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    TP = np.sum((y == 1) & (y_pred == 1))
    FP = np.sum((y == 0) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    FN = np.sum((y == 1) & (y_pred == 0))
    return TP, FP, TN, FN


def metric_precision(y, y_pred):
    TP, FP, _, _ = measure(y, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0


def metric_recall(y, y_pred):
    TP, _, _, FN = measure(y, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0


def metric_accuracy(y, y_pred):
    TP, FP, TN, FN = measure(y, y_pred)
    return (TP + TN) / (TP + TN + FP + FN) * 100


def metric_F1score(y, y_pred):
    precision = metric_precision(y, y_pred)
    recall = metric_recall(y, y_pred)
    return (
        2 * ((precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0
    )


def metric_AUC_ROC(y, y_pred):
    return roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else 0


def print_metrics(metrics_list, y, y_pred):
    metrics = {}
    output = []

    for metric in metrics_list:
        if metric == "Accuracy":
            value = metric_accuracy(y, y_pred)
            metrics["Accuracy"] = f"{value:.3f}%"
            output.append(f"Accuracy: {value:.3f}%")
        elif metric == "Recall":
            value = metric_recall(y, y_pred)
            metrics["Recall"] = f"{value:.3f}"
            output.append(f"Recall: {value:.3f}")
        elif metric == "F1":
            value = metric_F1score(y, y_pred)
            metrics["F1"] = f"{value:.3f}"
            output.append(f"F1 Score: {value:.3f}")
        elif metric == "Precision":
            value = metric_precision(y, y_pred)
            metrics["Precision"] = f"{value:.3f}"
            output.append(f"Precision: {value:.3f}")
        elif metric == "AUC_ROC":
            value = metric_AUC_ROC(y, y_pred)
            metrics["AUC_ROC"] = f"{value:.3f}"
            output.append(f"AUC ROC: {value:.3f}")

    print("\n".join(output))
    return metrics


def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomalía"],
        yticklabels=["Normal", "Anomalía"],
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()
