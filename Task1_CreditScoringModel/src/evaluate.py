"""
Evaluation Module
Evaluate model performance with metrics and visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from pathlib import Path
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: True labels
        model_name: Name of the model
        
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics
def print_metrics(metrics: dict):
    """Print formatted metrics."""
    print(f"\n{'='*50}")
    print(f"Model: {metrics['Model']}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"{key:15}: {value:.4f}")
def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Creditworthy', 'Creditworthy'],
                yticklabels=['Not Creditworthy', 'Creditworthy'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: True labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], 
                label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Credit Scoring Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.close()
def plot_metrics_comparison(metrics_list: list, save_path: str = None):
    """
    Plot comparison bar chart of all metrics.
    
    Args:
        metrics_list: List of metric dictionaries
        save_path: Path to save the plot
    """
    df = pd.DataFrame(metrics_list)
    df = df.set_index('Model')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    plt.close()
def plot_feature_importance(model, feature_names: list, model_name: str, save_path: str = None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Feature importance not available for {model_name}")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Feature Importance - {model_name}')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")
    
    plt.close()
def generate_all_plots(models: dict, X_test, y_test, feature_names: list, 
                      save_dir: str = 'reports/figures'):
    """
    Generate all evaluation plots.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: True labels
        feature_names: List of feature names
        save_dir: Directory to save plots
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        clean_name = name.lower().replace(' ', '_')
        plot_confusion_matrix(y_test, y_pred, name, 
                            f'{save_dir}/{clean_name}_confusion.png')
    
    # ROC curves
    plot_roc_curves(models, X_test, y_test, f'{save_dir}/roc_curves.png')
    
    # Feature importance for Random Forest
    if 'Random Forest' in models:
        plot_feature_importance(models['Random Forest'], feature_names, 
                              'Random Forest', f'{save_dir}/feature_importance.png')
    
    print(f"\nAll plots saved to {save_dir}/")
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import prepare_data
    from train import train_models
    
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    models = train_models(X_train, y_train)
    
    # Evaluate
    all_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
    
    # Generate plots
    feature_names = X_train.columns.tolist()
    generate_all_plots(models, X_test, y_test, feature_names)