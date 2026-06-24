"""
Utility functions for emotion recognition project.
"""
import os
import random
import numpy as np
import torch
from typing import Tuple, List, Dict
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_device() -> torch.device:
    """Get the appropriate device (CPU or GPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
def get_emotion_labels() -> Dict[int, str]:
    """Get emotion label mapping."""
    return {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "angry",
        4: "fear",
        5: "disgust",
        6: "surprise"
    }
def get_emotion_from_filename(filename: str) -> str:
    """
    Extract emotion from filename based on common dataset formats.
    Supports RAVDESS, TESS, and EMO-DB naming conventions.
    """
    filename = filename.lower()
    
    # RAVDESS format: 03-01-01-01-01-01-01.wav (emotion is 3rd part)
    if "03-01" in filename:
        parts = filename.split("-")
        if len(parts) >= 3:
            emotion_map = {
                "01": "neutral",
                "02": "calm",
                "03": "happy",
                "04": "sad",
                "05": "angry",
                "06": "fearful",
                "07": "disgust",
                "08": "surprised"
            }
            return emotion_map.get(parts[2], "unknown")
    
    # TESS format: OAF_happy_angry.wav or YAF_happy_angry.wav
    elif "oaf_" in filename or "yaf_" in filename:
        if "happy" in filename:
            return "happy"
        elif "sad" in filename:
            return "sad"
        elif "angry" in filename:
            return "angry"
        elif "fear" in filename:
            return "fear"
        elif "disgust" in filename:
            return "disgust"
        elif "pleasant_surprise" in filename or "surprise" in filename:
            return "surprise"
        elif "neutral" in filename:
            return "neutral"
    
    # EMO-DB format: 03a01Fa.wav (emotion is 5th character)
    elif len(filename) >= 5:
        emotion_char = filename[4] if len(filename) > 4 else ""
        emotion_map = {
            "W": "angry",
            "L": "boredom",
            "E": "disgust",
            "A": "fear",
            "F": "happy",
            "N": "neutral",
            "T": "sad"
        }
        return emotion_map.get(emotion_char.upper(), "unknown")
    
    return "unknown"
def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    from collections import Counter
    
    counter = Counter(labels)
    total = sum(counter.values())
    weights = []
    
    for i in range(num_classes):
        count = counter.get(i, 0)
        if count > 0:
            weight = total / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.FloatTensor(weights)