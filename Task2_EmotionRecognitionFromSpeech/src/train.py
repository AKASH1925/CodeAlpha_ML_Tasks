"""
Training pipeline for speech emotion recognition.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import json
from src.model import EmotionRecognitionModel, SimpleCNNModel, create_model, model_summary
from src.utils import set_seed, get_device, create_directory, format_time, count_parameters
class Trainer:
    """Training pipeline for emotion recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
        save_dir: str = "models"
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
            save_dir: Directory to save models
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.save_dir = save_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        create_directory(save_dir)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int = 50, early_stopping_patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        print(model_summary(self.model, (1, 128, 129)))
        print(f"\nDevice: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("=" * 60)
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
                print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Print statistics
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best model saved to: {self.best_model_path}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training history saved to: {history_path}")
        
        return self.history
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 7,
    model_type: str = "cnn_lstm",
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    save_dir: str = "models"
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with default settings.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of emotion classes
        model_type: Type of model to train
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save models
    
    Returns:
        Tuple of (trained model, training history)
    """
    # Create model
    model = create_model(model_type, num_classes)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    # Train
    history = trainer.train(num_epochs=num_epochs)
    
    return model, history
if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset
    
    # Create dummy data for testing
    batch_size = 32
    num_samples = 100
    audio_length = 22050 * 3  # 3 seconds
    
    X = torch.randn(num_samples, audio_length)
    y = torch.randint(0, 7, (num_samples,))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Train
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=7,
        model_type="simple_cnn",
        num_epochs=5
    )