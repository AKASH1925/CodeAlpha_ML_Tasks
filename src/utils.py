"""
Utility Functions for Visualization and Analysis
==================================================
Provides visualization tools for:
- Sample data display
- Training history plots
- Confusion matrix visualization
- Prediction visualizations
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import torch
def show_samples(data_loader, class_names=None, num_samples=16, save_path=None):
    """
    Display sample images from the dataset.
    Args:
        data_loader: Data loader
        class_names: List of class names
        num_samples (int): Number of samples to show
        save_path (str): Path to save the figure
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    # Denormalize for display
    images = images[:num_samples]
    labels = labels[:num_samples]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Sample Training Images", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].squeeze()
            ax.imshow(img, cmap="gray")
            label = labels[i].item()
            if class_names:
                title = f"{class_names[label]}"
            else:
                title = f"Label: {label}"
            ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample images saved to {save_path}")
    plt.close()
def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).
    Args:
        history (dict): Training history from Trainer
        save_path (str): Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to {save_path}")
    plt.close()
def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix.
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path (str): Path to save the figure
        figsize: Figure size
    """
    num_classes = cm.shape[0]
    if num_classes > 30:
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.colorbar(im, ax=ax)
        if class_names and len(class_names) <= 62:
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(class_names, fontsize=7)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(num_classes)
        if class_names:
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(class_names)
        else:
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
        # Add text annotations for smaller matrices
        if num_classes <= 20:
            thresh = cm.max() / 2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()
def plot_per_class_accuracy(per_class_accuracy, save_path=None):
    """
    Plot per-class accuracy bar chart.
    Args:
        per_class_accuracy (dict): Class name -> accuracy
        save_path (str): Path to save the figure
    """
    classes = list(per_class_accuracy.keys())
    accuracies = list(per_class_accuracy.values())
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.5), 6))
    colors = ["green" if acc >= 0.9 else "orange" if acc >= 0.7 else "red" for acc in accuracies]
    bars = ax.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(accuracies), color="blue", linestyle="--", label=f"Mean: {np.mean(accuracies):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45 if len(classes) > 10 else 0, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-class accuracy saved to {save_path}")
    plt.close()
def visualize_predictions(model, data_loader, class_names, device="cpu", num_samples=12, save_path=None):
    """
    Visualize model predictions on sample images.
    Args:
        model: Trained model
        data_loader: Test data loader
        class_names: List of class names
        device: Device for inference
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save the figure
    """
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Model Predictions", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].cpu().squeeze()
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predicted[i].item()]
            confidence = confidences[i].item()
            ax.imshow(img, cmap="gray")
            color = "green" if true_label == pred_label else "red"
            ax.set_title(
                f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2%}",
                fontsize=9,
                color=color,
            )
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Predictions visualization saved to {save_path}")
    plt.close()
def count_parameters(model):
    """
    Count model parameters.
    Args:
        model: PyTorch model
    Returns:
        dict: Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }