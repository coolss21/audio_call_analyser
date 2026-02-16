"""
Configuration settings for the Voice Detection API.
"""

import os

# Model Configuration
MODEL_FILENAME = "deepfake_model_multilingual.pt"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_FILENAME)
DEVICE = "cpu"  # Force CPU for Hugging Face Spaces

# Audio Processing Configuration
SAMPLE_RATE = 16000
DURATION = 3.5  # Seconds
Fixed_LENGTH = int(SAMPLE_RATE * DURATION)
N_MELS = 128

# Detection Thresholds
# Lower threshold for AI to catch more AI samples (reduce False Negatives)
AI_THRESHOLD = 0.40  # If AI probability > 0.40, classify as AI (Standard was 0.5)

# API Configuration
API_KEY_HEADER = "x-api-key"
DEFAULT_API_KEY = "secret123"  # Should be overridden by env var in production

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
