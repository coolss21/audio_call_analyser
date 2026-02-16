"""
Audio processing utilities for feature extraction.
"""

import numpy as np
import librosa
import soundfile as sf
import io
from typing import Tuple, Optional
from .. import config

class AudioProcessor:
    """Helper class for audio loading and feature extraction."""
    
    @staticmethod
    def pad_trim(y: np.ndarray) -> np.ndarray:
        """
        Pad or trim audio to fixed length defined in config.
        
        Args:
            y (np.ndarray): Input audio signal.
            
        Returns:
            np.ndarray: Padded/trimmed audio of length config.Fixed_LENGTH.
        """
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
            
        target_len = config.Fixed_LENGTH
        current_len = len(y)
        
        if current_len > target_len:
            y = y[:target_len]
        else:
            padding = target_len - current_len
            y = np.pad(y, (0, padding), mode='constant')
            
        return y.astype(np.float32)
    
    @staticmethod
    def load_wav(path: str) -> np.ndarray:
        """
        Load audio file from disk.
        
        Args:
            path (str): Path to audio file.
            
        Returns:
            np.ndarray: Processed audio signal.
            
        Raises:
            ValueError: If file cannot be read.
        """
        try:
            y, sr = sf.read(path, always_2d=False)
            return AudioProcessor._process_loaded_audio(y, sr)
        except Exception as e:
            raise ValueError(f"Error loading audio from {path}: {e}")
    
    @staticmethod
    def load_wav_from_bytes(audio_bytes: bytes) -> np.ndarray:
        """
        Load audio from byte content.
        
        Args:
            audio_bytes (bytes): Raw audio file content.
            
        Returns:
            np.ndarray: Processed audio signal.
            
        Raises:
            ValueError: If bytes cannot be decoded.
        """
        try:
            audio_io = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_io)
            return AudioProcessor._process_loaded_audio(y, sr)
        except Exception as e:
            raise ValueError(f"Error loading audio from bytes: {e}")

    @staticmethod
    def _process_loaded_audio(y: np.ndarray, sr: int) -> np.ndarray:
        """Internal method to resample and mixdown audio."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        y = y.astype(np.float32)
        
        if sr != config.SAMPLE_RATE:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            
        return AudioProcessor.pad_trim(y)
    
    @staticmethod
    def mel_spec(y: np.ndarray) -> np.ndarray:
        """
        Extract Mel-spectrogram features.
        
        Args:
            y (np.ndarray): Audio signal.
            
        Returns:
            np.ndarray: Mel-spectrogram in dB scale.
        """
        m = librosa.feature.melspectrogram(
            y=y, 
            sr=config.SAMPLE_RATE, 
            n_mels=config.N_MELS, 
            fmin=20, 
            fmax=config.SAMPLE_RATE//2
        )
        
        max_val = np.max(m)
        if max_val > 1e-10:
            m = librosa.power_to_db(m, ref=max_val)
        else:
            m = librosa.power_to_db(m + 1e-10, ref=1.0)
            
        return m.astype(np.float32)
    
    @staticmethod
    def extra_features(y: np.ndarray) -> np.ndarray:
        """
        Extract additional acoustic features.
        
        Args:
            y (np.ndarray): Audio signal.
            
        Returns:
            np.ndarray: Feature vector of shape (6,).
        """
        # Zero Crossing Rate
        z = librosa.feature.zero_crossing_rate(y)[0]
        z_var = float(np.var(z))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_var = float(np.var(rms)) if len(rms) > 1 else 0.0
        
        # Spectral Flatness
        S = np.abs(librosa.stft(y))
        flat = librosa.feature.spectral_flatness(S=S)[0]
        flat_m = float(np.mean(flat))
        
        # Return fixed-size array (padding with zeros for future features)
        return np.array([z_var, rms_var, flat_m, 0.0, 0.0, 0.0], dtype=np.float32)
