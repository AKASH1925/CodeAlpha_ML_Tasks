"""
Evaluation — Classification metrics: accuracy, precision, recall, F1.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.
    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix, roc_auc
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results["roc_auc"] = 0.0
    else:
        results["roc_auc"] = 0.0
    return results
def print_report(model_name: str, results: dict):
    """Print a formatted classification report."""
    print(f"--- {model_name} ---")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    if results.get("roc_auc", 0) > 0:
        print(f"  ROC AUC:   {results['roc_auc']:.4f}")
    print()