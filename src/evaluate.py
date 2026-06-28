"""
Evaluation Script for Handwritten Character Recognition
========================================================
Provides comprehensive model evaluation including:
- Accuracy metrics
- Confusion matrix
- Classification report
- Per-class accuracy analysis
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
def evaluate_model(model, test_loader, device="cpu", class_names=None):
    """
    Evaluate the model on the test set.
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device for inference
        class_names: List of class names
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    # Classification report
    if class_names is None:
        class_names = [chr(ord("A") + i) for i in range(len(np.unique(all_labels)))]
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, digits=4
    )
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    # Per-class accuracy
    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_predictions[class_mask] == i).mean()
            per_class_accuracy[name] = float(class_acc)
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities,
    }
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    # Find worst and best performing classes
    if per_class_accuracy:
        worst_class = min(per_class_accuracy, key=per_class_accuracy.get)
        best_class = max(per_class_accuracy, key=per_class_accuracy.get)
        print(f"Best performing class:  {best_class} ({per_class_accuracy[best_class]:.4f})")
        print(f"Worst performing class: {worst_class} ({per_class_accuracy[worst_class]:.4f})")
    print("=" * 60)
    return results