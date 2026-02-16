"""
Fusion Model Architecture for Deepfake Detection.
"""

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """
    Fusion model combining Mel-spectrogram, SSL embeddings, and acoustic features.
    
    Architecture:
    - Mel-spectrogram encoder (1D-CNN)
    - Concatenation with SSL embeddings and extra features
    - Dense classification head
    """
    
    def __init__(self):
        super().__init__()
        
        # Mel-spectrogram processor
        # Input: (Batch, 128, Time)
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature fusion and classification
        # Mel (32) + SSL (768) + Extra (6) = 806 input features
        self.fusion = nn.Sequential(
            nn.Linear(32 + 768 + 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification: [Human, AI]
        )
    
    def forward(self, mel: torch.Tensor, ssl: torch.Tensor, ext: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion model.
        
        Args:
            mel (torch.Tensor): Mel-spectrogram features (Batch, 128, Time)
            ssl (torch.Tensor): SSL embeddings (Batch, 768)
            ext (torch.Tensor): Extra acoustic features (Batch, 6)
            
        Returns:
            torch.Tensor: Logits (Batch, 2)
        """
        mel_feat = self.mel_encoder(mel).squeeze(-1)
        fused = torch.cat([mel_feat, ssl, ext], dim=1)
        return self.fusion(fused)
