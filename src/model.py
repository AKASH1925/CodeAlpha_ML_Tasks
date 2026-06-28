"""
CNN Model Architecture for Handwritten Character Recognition
=============================================================
Defines the Convolutional Neural Network architecture used for
classifying handwritten characters (EMNIST Letters A-Z, 26 classes).
Also supports MNIST digits (10 classes) and other EMNIST variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class HandwrittenCNN(nn.Module):
    """
    CNN Architecture for Handwritten Character Recognition.
    Architecture:
        - 3 Convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout)
        - 2 Fully connected layers
        - Output layer with softmax
    Primary: EMNIST Letters (A-Z) - 26 classes
    Also: MNIST Digits (0-9) - 10 classes
    Input: 28x28 grayscale images
    Output: Class probabilities
    """
    def __init__(self, num_classes=26, dropout_rate=0.3):
        """
        Initialize the CNN model.
        Args:
            num_classes (int): Number of output classes
                               26 for EMNIST letters A-Z (default)
                               10 for MNIST digits 0-9
                               36 for EMNIST digits+letters
                               47 for EMNIST balanced
            dropout_rate (float): Dropout rate for regularization
        """
        super(HandwrittenCNN, self).__init__()
        self.num_classes = num_classes
        # Convolutional Block 1: 1 -> 32 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        # Convolutional Block 2: 32 -> 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        # Convolutional Block 3: 64 -> 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        # After 3 MaxPool2d(2,2): 28 -> 14 -> 7 -> 3 (floor division)
        # Feature map size: 128 x 3 x 3 = 1152
        self.feature_size = 128 * 3 * 3
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        # Initialize weights
        self._initialize_weights()
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
    def predict(self, x):
        """
        Predict class probabilities.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
def get_model(dataset="emnist_letters", dropout_rate=0.3):
    """
    Factory function to get the appropriate model.
    Args:
        dataset (str): Dataset type - 'emnist_letters' (default, 26 classes A-Z),
                       'mnist' (10 digits), 'emnist_digits', 'emnist_full',
                       or 'emnist_balanced'
        dropout_rate (float): Dropout rate
    Returns:
        HandwrittenCNN model instance
    """
    num_classes_map = {
        "emnist_letters": 26,
        "mnist": 10,
        "emnist_digits": 10,
        "emnist_full": 36,
        "emnist_balanced": 47,
    }
    num_classes = num_classes_map.get(dataset, 26)
    model = HandwrittenCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    print(f"Model created: {dataset.upper()} ({num_classes} classes)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return model