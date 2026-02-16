"""
Main detector logic for Deepfake classification.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from transformers import AutoProcessor as HFProcessor, AutoModel

from . import config
from .models.fusion_model import FusionModel
from .processors.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """
    Main inference class for deepfake detection.
    
    Handles model loading, feature extraction, and prediction logic.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector.
        
        Args:
            model_path (str, optional): Path to model weights. Defaults to config location.
        """
        self.device = torch.device(config.DEVICE)
        self.model_path = Path(model_path or config.MODEL_PATH)
        
        self.model: Optional[FusionModel] = None
        self.ssl_processor = None
        self.ssl_model = None
        
        self._load_models()
        
    def _load_models(self):
        """Load all necessary models and processors."""
        try:
            # Load Fusion Model
            logger.info(f"Loading fusion model from {self.model_path}")
            self.model = FusionModel().to(self.device)
            
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            else:
                logger.warning(f"Model file not found at {self.model_path}. Inference will fail.")
                
            # Load SSL components (Wav2Vec2)
            # This downloads from HuggingFace Hub if not cached
            logger.info("Loading Wav2Vec2 components...")
            self.ssl_processor = HFProcessor.from_pretrained("facebook/wav2vec2-base")
            self.ssl_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()
            
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def extract_ssl_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract SSL embedding using wav2vec2.
        
        Args:
            audio (np.ndarray): Preprocessed audio.
            
        Returns:
            np.ndarray: Embedding vector (768,)
        """
        with torch.no_grad():
            inputs = self.ssl_processor(
                audio, 
                sampling_rate=config.SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs["input_values"].to(self.device)
            out = self.ssl_model(input_values).last_hidden_state
            # Mean pooling over time dimension
            emb = out.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        return emb[0]

    def predict(self, audio_path: Optional[str] = None, audio_bytes: Optional[bytes] = None) -> Dict:
        """
        Predict if audio is human or AI generated.
        
        Args:
            audio_path (str, optional): Path to audio file.
            audio_bytes (bytes, optional): Audio content as bytes.
            
        Returns:
            Dict: Prediction results conforming to API requirements.
        """
        try:
            # 1. Load Audio
            if audio_path:
                audio = AudioProcessor.load_wav(audio_path)
            elif audio_bytes:
                audio = AudioProcessor.load_wav_from_bytes(audio_bytes)
            else:
                raise ValueError("No audio input provided")
                
            # 2. Extract Features
            mel = AudioProcessor.mel_spec(audio)      # (128, T)
            ssl = self.extract_ssl_embedding(audio)   # (768,)
            ext = AudioProcessor.extra_features(audio) # (6,)
            
            # 3. Prepare Tensors
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            ssl_tensor = torch.FloatTensor(ssl).unsqueeze(0).to(self.device)
            ext_tensor = torch.FloatTensor(ext).unsqueeze(0).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                logits = self.model(mel_tensor, ssl_tensor, ext_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # 5. Parse Results
            human_prob = float(probs[0])
            ai_prob = float(probs[1])
            
            # Decision Logic with Configurable Threshold
            # By default, use config.AI_THRESHOLD (0.4) to be more sensitive to AI
            if ai_prob > config.AI_THRESHOLD:
                label = "ai"
                confidence = ai_prob
            else:
                label = "human"
                confidence = human_prob
                
            return {
                'label': label,
                'confidence': confidence,
                'probabilities': {
                    'human': human_prob,
                    'ai': ai_prob
                },
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'label': 'error',
                'error': str(e),
                'confidence': 0.0
            }
