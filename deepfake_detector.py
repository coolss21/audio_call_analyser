"""
Multilingual Deepfake Detection Model Wrapper
==============================================
Handles inference for English, Hindi, Telugu, Malayalam, Tamil
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SR = 16000
DUR = 3.5
LEN = int(SR * DUR)

LANGUAGES = {
    'en': 'english',
    'hi': 'hindi',
    'te': 'telugu',
    'ml': 'malayalam',
    'ta': 'tamil',
    'english': 'english',
    'hindi': 'hindi',
    'telugu': 'telugu',
    'malayalam': 'malayalam',
    'tamil': 'tamil'
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class AudioProcessor:
    """Extract features from audio files"""
    
    @staticmethod
    def pad_trim(y: np.ndarray) -> np.ndarray:
        """Pad or trim audio to fixed length"""
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        if len(y) > LEN:
            y = y[:LEN]
        else:
            y = np.pad(y, (0, LEN - len(y)))
        return y.astype(np.float32)
    
    @staticmethod
    def load_wav(path: str) -> np.ndarray:
        """Load audio file and prepare it"""
        try:
            y, sr = sf.read(path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            if sr != SR:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=SR)
            return AudioProcessor.pad_trim(y)
        except Exception as e:
            raise ValueError(f"Error loading audio from {path}: {e}")
    
    @staticmethod
    def load_wav_from_bytes(audio_bytes: bytes) -> np.ndarray:
        """Load audio from bytes (for streaming)"""
        try:
            import io
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            if sr != SR:
                audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=SR)
            return AudioProcessor.pad_trim(audio_data)
        except Exception as e:
            raise ValueError(f"Error loading audio from bytes: {e}")
    
    @staticmethod
    def mel_spec(y: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Extract mel-spectrogram"""
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=n_mels, fmin=20, fmax=SR//2)
        max_val = np.max(m)
        if max_val > 1e-10:
            m = librosa.power_to_db(m, ref=max_val)
        else:
            m = librosa.power_to_db(m + 1e-10, ref=1.0)
        return m.astype(np.float32)
    
    @staticmethod
    def extra_features(y: np.ndarray) -> np.ndarray:
        """Extract acoustic features"""
        z = librosa.feature.zero_crossing_rate(y)[0]
        z_var = float(np.var(z))
        rms = librosa.feature.rms(y=y)[0]
        rms_var = float(np.var(rms)) if len(rms) > 1 else 0.0
        S = np.abs(librosa.stft(y))
        flat = librosa.feature.spectral_flatness(S=S)[0]
        flat_m = float(np.mean(flat))
        return np.array([z_var, rms_var, flat_m, 0.0, 0.0, 0.0], dtype=np.float32)

# ============================================================================
# FUSION MODEL
# ============================================================================

class FusionModel(nn.Module):
    """Fusion model combining mel-spec, SSL, and extra features"""
    
    def __init__(self):
        super().__init__()
        
        # Mel-spectrogram processor
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(32 + 768 + 6, 256),  # mel + ssl + extra
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # binary: human vs AI
        )
    
    def forward(self, mel: torch.Tensor, ssl: torch.Tensor, ext: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        mel_feat = self.mel_encoder(mel).squeeze(-1)
        fused = torch.cat([mel_feat, ssl, ext], dim=1)
        return self.fusion(fused)

# ============================================================================
# DEEPFAKE DETECTOR
# ============================================================================

class DeepfakeDetector:
    """Main inference class for deepfake detection"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the detector
        
        Args:
            model_path: Path to deepfake_model_multilingual.pt
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load model
        self.model = FusionModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load SSL processor
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
        self.ssl_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Device: {self.device}")
    
    def extract_ssl_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract SSL embedding using wav2vec2"""
        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(self.device)
            out = self.ssl_model(input_values).last_hidden_state
            emb = out.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        return emb[0]
    
    def predict(self, audio_path: str = None, audio_bytes: bytes = None) -> Dict:
        """
        Predict if audio is human or AI
        
        Args:
            audio_path: Path to audio file
            audio_bytes: Audio data as bytes
        
        Returns:
            Dict with prediction results:
                {
                    'label': 'human' or 'ai',
                    'confidence': float (0-1),
                    'probabilities': {'human': float, 'ai': float},
                    'device': 'cpu' or 'cuda'
                }
        """
        try:
            # Load audio
            if audio_path:
                audio = AudioProcessor.load_wav(audio_path)
            elif audio_bytes:
                audio = AudioProcessor.load_wav_from_bytes(audio_bytes)
            else:
                raise ValueError("Either audio_path or audio_bytes must be provided")
            
            # Extract features
            mel = AudioProcessor.mel_spec(audio)  # (128, T)
            ssl = self.extract_ssl_embedding(audio)  # (768,)
            ext = AudioProcessor.extra_features(audio)  # (6,)
            
            # Convert to tensors
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)  # (1, 128, T)
            ssl_tensor = torch.FloatTensor(ssl).unsqueeze(0).to(self.device)  # (1, 768)
            ext_tensor = torch.FloatTensor(ext).unsqueeze(0).to(self.device)  # (1, 6)
            
            # Inference
            with torch.no_grad():
                logits = self.model(mel_tensor, ssl_tensor, ext_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Parse results
            human_prob = float(probs[0])
            ai_prob = float(probs[1])
            
            # Determine label
            label = "ai" if ai_prob > human_prob else "human"
            confidence = max(human_prob, ai_prob)
            
            return {
                'label': label,
                'confidence': confidence,
                'probabilities': {
                    'human': human_prob,
                    'ai': ai_prob
                },
                'device': str(self.device),
                'timestamp': None  # Will be set by API
            }
        
        except Exception as e:
            return {
                'label': 'error',
                'error': str(e),
                'confidence': 0.0,
                'probabilities': {'human': 0.0, 'ai': 0.0},
                'device': str(self.device)
            }
    
    def predict_batch(self, audio_paths: list) -> list:
        """Predict on multiple audio files"""
        results = []
        for path in audio_paths:
            result = self.predict(audio_path=path)
            result['path'] = path
            results.append(result)
        return results

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Deepfake Detection Model Loaded!")
    print("\nUsage:")
    print("  detector = DeepfakeDetector('deepfake_model_multilingual.pt')")
    print("  result = detector.predict(audio_path='path/to/audio.wav')")
    print("\nReturns:")
    print("  {")
    print("    'label': 'human' or 'ai',")
    print("    'confidence': 0.85,")
    print("    'probabilities': {'human': 0.15, 'ai': 0.85}")
    print("  }")
