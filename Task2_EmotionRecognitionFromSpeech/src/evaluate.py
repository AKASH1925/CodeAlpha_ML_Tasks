"""
Evaluation and visualization for speech emotion recognition.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_emotion_labels, get_device
class Evaluator:
    """Evaluation and visualization for emotion recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        results_dir: str = "results"
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device for inference
            results_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.results_dir = results_dir
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.emotion_labels = get_emotion_labels()
        
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=[self.emotion_labels[i] for i in sorted(self.emotion_labels.keys())],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probs
        }
        
        return results
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        save: bool = True
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=list(self.emotion_labels.values()),
            yticklabels=list(self.emotion_labels.values())
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        save: bool = True
    ) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
        axes[0].plot(history['val_loss'], label='Val Loss', marker='o', markersize=3)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
        axes[1].plot(history['val_acc'], label='Val Acc', marker='o', markersize=3)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def plot_per_class_metrics(
        self,
        report: Dict,
        save: bool = True
    ) -> None:
        """Plot per-class precision, recall, and F1-score."""
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='forestgreen')
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'per_class_metrics.png'), dpi=300)
        plt.show()
    
    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"  Precision: {results['precision']*100:.2f}%")
        print(f"  Recall:    {results['recall']*100:.2f}%")
        print(f"  F1-Score:  {results['f1_score']*100:.2f}%")
        
        print(f"\nClassification Report:")
        report = results['classification_report']
        for emotion in list(report.keys())[:-3]:
            metrics = report[emotion]
            print(f"  {emotion:12s}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                  f"Support={int(metrics['support'])}")
        
        print("\n" + "=" * 60)
    
    def run_full_evaluation(
        self,
        history: Optional[Dict] = None
    ) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            history: Training history (optional)
        
        Returns:
            Evaluation results
        """
        print("\nRunning evaluation...")
        
        # Evaluate
        results = self.evaluate()
        
        # Print summary
        self.print_summary(results)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['confusion_matrix'])
        
        # Plot per-class metrics
        self.plot_per_class_metrics(results['classification_report'])
        
        # Plot training history if provided
        if history:
            self.plot_training_history(history)
        
        # Save results to file
        results_file = os.path.join(self.results_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Accuracy:  {results['accuracy']*100:.2f}%\n")
            f.write(f"Precision: {results['precision']*100:.2f}%\n")
            f.write(f"Recall:    {results['recall']*100:.2f}%\n")
            f.write(f"F1-Score:  {results['f1_score']*100:.2f}%\n")
        
        print(f"\nResults saved to {self.results_dir}/")
        
        return results
def predict_emotion(
    model: nn.Module,
    audio: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[str, float]:
    """
    Predict emotion from audio signal.
    
    Args:
        model: Trained model
        audio: Audio signal
        device: Device for inference
    
    Returns:
        Tuple of (predicted emotion, confidence)
    """
    device = device or get_device()
    
    model = model.to(device)
    model.eval()
    
    # Prepare input
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(audio_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    emotion_labels = get_emotion_labels()
    emotion = emotion_labels[predicted.item()]
    
    return emotion, confidence.item()