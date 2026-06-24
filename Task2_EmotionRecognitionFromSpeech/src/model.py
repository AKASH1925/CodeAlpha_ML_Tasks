"""
Deep learning model for speech emotion recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on important features."""
    
    def __init__(self, hidden_size: int):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
        Returns:
            Weighted sum (batch_size, hidden_size)
        """
        attention_weights = self.attention(x)
        weighted = torch.sum(x * attention_weights, dim=1)
        return weighted
class CNNBlock(nn.Module):
    """Convolutional Neural Network block for feature extraction."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super(CNNBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, height, width)
        Returns:
            Extracted features (batch_size, features)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        return x
class LSTMBlock(nn.Module):
    """LSTM block for temporal pattern recognition."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMBlock, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = AttentionBlock(hidden_size * 2 if bidirectional else hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        Returns:
            Output tensor (batch_size, hidden_size * 2 if bidirectional)
        """
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        return attended
class EmotionRecognitionModel(nn.Module):
    """
    Hybrid CNN-LSTM model for speech emotion recognition.
    
    Architecture:
    1. CNN layers extract spatial features from spectrograms
    2. LSTM layers capture temporal patterns
    3. Attention mechanism focuses on important time steps
    4. Fully connected layers classify emotions
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        input_channels: int = 1,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super(EmotionRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN for feature extraction
        self.cnn = CNNBlock(
            in_channels=input_channels,
            out_channels=cnn_channels
        )
        
        # Calculate CNN output size
        # Input: (batch, 1, 128, 129) for mel spectrogram
        # After 3 pooling layers: (batch, cnn_channels*4, 16, 16)
        cnn_output_size = cnn_channels * 4 * 16 * 16
        
        # LSTM for temporal modeling
        self.lstm = LSTMBlock(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 1, height, width) or (batch_size, audio_length)
        Returns:
            Logits (batch_size, num_classes)
        """
        # Reshape if needed (for raw audio input)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        
        # Reshape for LSTM
        batch_size = cnn_features.size(0)
        cnn_features = cnn_features.view(batch_size, -1)
        cnn_features = cnn_features.unsqueeze(1)  # Add sequence dimension
        
        # LSTM temporal modeling
        lstm_features = self.lstm(cnn_features)
        
        # Classification
        logits = self.classifier(lstm_features)
        
        return logits
class SimpleCNNModel(nn.Module):
    """
    Simpler CNN-only model for faster training and comparison.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        input_channels: int = 1
    ):
        super(SimpleCNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.features(x)
        logits = self.classifier(features)
        
        return logits
def create_model(
    model_type: str = "cnn_lstm",
    num_classes: int = 7,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ("cnn_lstm" or "simple_cnn")
        num_classes: Number of emotion classes
        **kwargs: Additional arguments for the model
    
    Returns:
        Model instance
    """
    if model_type == "cnn_lstm":
        return EmotionRecognitionModel(num_classes=num_classes, **kwargs)
    elif model_type == "simple_cnn":
        return SimpleCNNModel(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """Generate a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Architecture Summary:
===========================
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}
Model structure:
{model}
"""
    return summary