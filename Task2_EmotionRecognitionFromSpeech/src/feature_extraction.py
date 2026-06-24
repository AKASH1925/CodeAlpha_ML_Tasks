"""
Feature extraction for speech emotion recognition.
"""
import numpy as np
import librosa
from typing import Dict, Tuple, Optional
import torch
class AudioFeatureExtractor:
    """Extract various audio features for emotion recognition."""
    
    def __init__(
        self,
        sr: int = 22050,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Args:
            sr: Sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs).
        
        MFCCs represent the short-term power spectrum of sound and
        are widely used in speech and audio processing.
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Add delta and delta-delta features
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack all features
        features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel Spectrogram.
        
        Mel spectrograms are visual representations of the spectrum
        of frequencies as they vary with time.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Spectral Contrast features.
        
        Spectral contrast measures the difference in amplitude
        between peaks and valleys in the spectrum.
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Normalize
        contrast = (contrast - np.mean(contrast)) / (np.std(contrast) + 1e-8)
        
        return contrast
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Chroma features.
        
        Chroma features represent the intensity of the 12
        different pitch classes (C, C#, D, etc.).
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return chroma
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Zero Crossing Rate.
        
        ZCR is the rate at which the signal changes sign,
        which is related to the noisiness of the signal.
        """
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return zcr
    
    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Root Mean Square (RMS) energy.
        
        RMS energy represents the loudness of the audio signal.
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return rms
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all audio features.
        
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            "mfcc": self.extract_mfcc(audio),
            "mel_spectrogram": self.extract_mel_spectrogram(audio),
            "spectral_contrast": self.extract_spectral_contrast(audio),
            "chroma": self.extract_chroma(audio),
            "zcr": self.extract_zero_crossing_rate(audio),
            "rms": self.extract_rms_energy(audio)
        }
        
        return features
    
    def extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract and combine MFCC with delta features.
        
        This is the primary feature extraction method used for
        the CNN-LSTM model.
        """
        mfccs = self.extract_mfcc(audio)
        
        # Compute statistics across time axis
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        
        # Concatenate statistics
        combined = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
        
        return combined
def extract_features_for_batch(
    audio_batch: np.ndarray,
    extractor: AudioFeatureExtractor,
    feature_type: str = "mfcc"
) -> np.ndarray:
    """
    Extract features for a batch of audio signals.
    
    Args:
        audio_batch: Batch of audio signals (batch_size, audio_length)
        extractor: Feature extractor instance
        feature_type: Type of features to extract ("mfcc", "mel", "combined")
    
    Returns:
        Array of extracted features
    """
    batch_features = []
    
    for audio in audio_batch:
        if feature_type == "mfcc":
            features = extractor.extract_mfcc(audio)
        elif feature_type == "mel":
            features = extractor.extract_mel_spectrogram(audio)
        elif feature_type == "combined":
            features = extractor.extract_combined_features(audio)
        else:
            features = extractor.extract_mfcc(audio)
        
        batch_features.append(features)
    
    return np.array(batch_features)
def compute_mfcc_statistics(mfccs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation of MFCCs.
    
    Useful for normalizing features across the dataset.
    """
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0) + 1e-8
    
    return mean, std
def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Input features
        mean: Precomputed mean (optional)
        std: Precomputed std (optional)
    
    Returns:
        Normalized features
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0) + 1e-8
    
    return (features - mean) / std