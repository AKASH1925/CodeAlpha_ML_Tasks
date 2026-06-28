"""
Visualization — Charts for model comparison and evaluation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
def plot_accuracy_comparison(all_results: dict):
    """
    Bar chart comparing accuracy across all models and datasets.
    all_results: { dataset_name: { model_name: { "accuracy": float, ... } } }
    """
    datasets = list(all_results.keys())
    models = list(next(iter(all_results.values())).keys())
    x = np.arange(len(datasets))
    width = 0.18
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        accuracies = [all_results[d][model]["accuracy"] for d in datasets]
        bars = ax.bar(x + i * width, accuracies, width, label=model, color=colors[i % len(colors)])
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison Across Datasets")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: results/accuracy_comparison.png")
def plot_metric_comparison(all_results: dict, metric: str, filename: str):
    """Generic bar chart for any metric across models and datasets."""
    datasets = list(all_results.keys())
    models = list(next(iter(all_results.values())).keys())
    x = np.arange(len(datasets))
    width = 0.18
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [all_results[d][model].get(metric, 0) for d in datasets]
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Model {metric.capitalize()} Comparison Across Datasets")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: results/{filename}")
def plot_confusion_matrices(all_results: dict):
    """Plot confusion matrices for all models on all datasets."""
    datasets = list(all_results.keys())
    models = list(next(iter(all_results.values())).keys())
    n_datasets = len(datasets)
    n_models = len(models)
    fig, axes = plt.subplots(n_datasets, n_models, figsize=(4 * n_models, 4 * n_datasets))
    if n_datasets == 1:
        axes = [axes]
    if n_models == 1:
        axes = [[ax] for ax in axes]
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            cm = all_results[dataset][model]["confusion_matrix"]
            ax = axes[i][j]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"])
            ax.set_title(f"{dataset}\n{model}", fontsize=10)
            ax.set_ylabel("Actual" if j == 0 else "")
            ax.set_xlabel("Predicted" if i == n_datasets - 1 else "")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("  Saved: results/confusion_matrices.png")
def plot_roc_curves(roc_data: dict):
    """
    Plot ROC curves for all models on all datasets.
    roc_data: { dataset_name: { model_name: { "fpr": array, "tpr": array, "auc": float } } }
    """
    datasets = list(roc_data.keys())
    models = list(next(iter(roc_data.values())).keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        for j, model in enumerate(models):
            data = roc_data[dataset][model]
            ax.plot(data["fpr"], data["tpr"], color=colors[j % len(colors)],
                    label=f'{model} (AUC={data["auc"]:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {dataset}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("  Saved: results/roc_curves.png")
def generate_all_plots(all_results: dict, roc_data: dict):
    """Generate all visualization charts."""
    print("\nGenerating charts...")
    plot_accuracy_comparison(all_results)
    plot_metric_comparison(all_results, "precision", "precision_comparison.png")
    plot_metric_comparison(all_results, "recall", "recall_comparison.png")
    plot_metric_comparison(all_results, "f1", "f1_comparison.png")
    plot_confusion_matrices(all_results)
    plot_roc_curves(roc_data)
    print(f"\nAll charts saved to {RESULTS_DIR}/")