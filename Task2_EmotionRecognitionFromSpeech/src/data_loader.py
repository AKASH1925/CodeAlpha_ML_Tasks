"""
Data loader for emotion recognition from speech.
"""
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings("ignore")
class SpeechEmotionDataset(Dataset):
    """Dataset class for speech emotion recognition."""
    
    def __init__(
        self,
        audio_paths: List[str],
        emotions: List[str],
        sr: int = 22050,
        duration: float = 3.0,
        augment: bool = False
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            emotions: List of emotion labels
            sr: Sampling rate
            duration: Duration in seconds to pad/truncate audio
            augment: Whether to apply data augmentation
        """
        self.audio_paths = audio_paths
        self.emotions = emotions
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.max_length = int(sr * duration)
        
        # Emotion to index mapping
        self.emotion_to_idx = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fear": 4,
            "fearful": 4,  # Alias
            "disgust": 5,
            "surprise": 6,
            "calm": 0,  # Map calm to neutral
            "boredom": 0,  # Map boredom to neutral
        }
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        emotion = self.emotions[idx]
        
        # Load audio
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(self.max_length, dtype=np.float32)
        
        # Pad or truncate to fixed length
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")
        else:
            audio = audio[:self.max_length]
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment(audio)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # Get emotion index
        emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)
        
        return audio_tensor, emotion_idx
    
    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques."""
        # Random noise injection
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
        
        # Random time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-2000, 2000)
            audio = np.roll(audio, shift)
        
        # Random pitch shift (subtle)
        if np.random.random() < 0.2:
            n_steps = np.random.uniform(-0.5, 0.5)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        
        return audio
def load_dataset(
    dataset_path: str,
    sr: int = 22050,
    duration: float = 3.0
) -> Tuple[List[str], List[str]]:
    """
    Load dataset from directory structure.
    
    Expected structure:
    dataset_path/
        happy/
            audio1.wav
            audio2.wav
        sad/
            audio1.wav
            ...
    OR flat structure with emotion in filename.
    
    Args:
        dataset_path: Path to dataset directory
        sr: Sampling rate
        duration: Duration in seconds
    
    Returns:
        Tuple of (audio_paths, emotions)
    """
    from src.utils import get_emotion_from_filename
    
    audio_paths = []
    emotions = []
    
    # Check if directory has subdirectories (organized by emotion)
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if subdirs and not any(f.endswith('.wav') for f in os.listdir(dataset_path)):
        # Directory structure: dataset_path/emotion_label/audio.wav
        for subdir in subdirs:
            emotion = subdir.lower()
            subdir_path = os.path.join(dataset_path, subdir)
            
            for filename in os.listdir(subdir_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    audio_paths.append(os.path.join(subdir_path, filename))
                    emotions.append(emotion)
    else:
        # Flat structure: dataset_path/audio.wav
        for filename in os.listdir(dataset_path):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                audio_paths.append(os.path.join(dataset_path, filename))
                emotion = get_emotion_from_filename(filename)
                emotions.append(emotion)
    
    print(f"Loaded {len(audio_paths)} audio files")
    print(f"Emotion distribution: {dict(zip(*np.unique(emotions, return_counts=True)))}")
    
    return audio_paths, emotions
def create_data_loaders(
    audio_paths: List[str],
    emotions: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    sr: int = 22050,
    duration: float = 3.0,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        audio_paths: List of audio file paths
        emotions: List of emotion labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        batch_size: Batch size
        sr: Sampling rate
        duration: Duration in seconds
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        audio_paths, emotions, test_size=1-train_ratio-val_ratio, 
        random_state=42, stratify=emotions
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio/(train_ratio+val_ratio),
        random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = SpeechEmotionDataset(X_train, y_train, sr, duration, augment=True)
    val_dataset = SpeechEmotionDataset(X_val, y_val, sr, duration, augment=False)
    test_dataset = SpeechEmotionDataset(X_test, y_test, sr, duration, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader