"""
Training Script for Handwritten Character Recognition
======================================================
Handles the complete training pipeline including:
- Model training with validation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history tracking
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_model_state = None
        self.should_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)
        return self.should_stop
class Trainer:
    """Complete training pipeline for the CNN model."""
    def __init__(self, model, device="cpu", learning_rate=0.001):
        """
        Initialize the trainer.
        Args:
            model: PyTorch model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Initial learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        Args:
            train_loader: Training data loader
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    def validate(self, val_loader):
        """
        Validate the model.
        Args:
            val_loader: Validation data loader
        Returns:
            tuple: (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=50,
        save_dir="./models",
        model_name="best_model",
        patience=10,
    ):
        """
        Complete training loop.
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs (int): Maximum number of epochs
            save_dir (str): Directory to save model checkpoints
            model_name (str): Name for saved model files
            patience (int): Early stopping patience
        Returns:
            dict: Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        early_stopping = EarlyStopping(patience=patience)
        best_val_acc = 0.0
        total_start = time.time()
        print(f"\n{'='*60}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        for epoch in range(num_epochs):
            epoch_start = time.time()
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            epoch_time = time.time() - epoch_start
            # Print progress
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    os.path.join(save_dir, f"{model_name}_best.pth"),
                    epoch,
                    val_acc,
                    val_loss,
                )
                print(f"  -> New best model saved! (Val Acc: {val_acc:.4f})")
            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}\n")
        # Save final model
        self.save_checkpoint(
            os.path.join(save_dir, f"{model_name}_final.pth"),
            epoch,
            val_acc,
            val_loss,
        )
        return self.history
    def save_checkpoint(self, filepath, epoch, accuracy, loss):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": accuracy,
                "loss": loss,
                "num_classes": self.model.num_classes,
            },
            filepath,
        )
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: epoch={checkpoint['epoch']}, acc={checkpoint['accuracy']:.4f}")
        return checkpoint