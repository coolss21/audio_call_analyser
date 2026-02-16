# Audio Call Analyser - Multilingual Voice Deepfake Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)

A high-performance, multilingual voice deepfake detection API built for real-time audio analysis. This system classifies audio as either `HUMAN` or `AI_GENERATED` using a fusion model that combines Mel-spectrogram analysis, SSL embeddings (Wav2Vec 2.0), and acoustic feature extraction.

## 🚀 Features

- **Multilingual Support**: Optimized for English, Hindi, Tamil, Telugu, and Malayalam.
- **Advanced Fusion Model**: Combines multiple feature extraction techniques:
  - **Mel-spectrogram**: Captures frequency-domain patterns.
  - **SSL Embeddings**: Leverages `Wav2Vec2` (facebook/wav2vec2-base) for deep contextual representations.
  - **Acoustic Features**: Monitors Zero Crossing Rate, RMS energy, and Spectral Flatness.
- **Fast Inference**: Optimized for CPU execution, suitable for free-tier deployments like Hugging Face Spaces.
- **Simple API**: Easy-to-integrate REST API with JSON request/response formats.

## 🛠️ Installation & Setup

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coolss21/audio_call_analyser.git
   cd audio_call_analyser
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API server**:
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`.

### Docker Setup

1. **Build the image**:
   ```bash
   docker build -t audio-call-analyser .
   ```

2. **Run the container**:
   ```bash
   docker run -p 5000:5000 audio-call-analyser
   ```

## 🔌 API Usage (Hackathon Evaluation Format)

### Detection Endpoint

`POST /detect` or `POST /api/voice-detection`

**Headers:**
```json
{
  "Content-Type": "application/json",
  "x-api-key": "secret123"
}
```

**Request Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_mp3_data>"
}
```

**Response Format:**
```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.8524
}
```

## 🧠 Model Architecture & Approach

The project uses a **Fusion-based Deep Learning approach**:
1. **Frontend**: Extracts three types of descriptors:
   - **Time-frequency features**: Mel-spectrograms (128 mels).
   - **Self-Supervised Learning (SSL)**: Embeddings from `wav2vec2-base` (768-dim).
   - **Acoustic Features**: Spectral flatness, RMS energy, and zero-crossing rate.
2. **Backbone**: A multi-path encoder that processes spectral data via 1D-CNNs and concatenates them with SSL and acoustic vectors.
3. **Head**: A dense classification layer with Dropout for robust inference.

## 📁 Project Structure

```text
.
├── app.py                   # Flask API implementation
├── deepfake_detector.py      # Core detection logic and Fusion Model
├── deepfake_model_multilingual.pt # Trained model weights
├── requirements.txt         # Python dependencies
├── Dockerfile               # Containerization configuration
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

## 🧪 Testing

You can use the provided `test_api.py` script to evaluate your deployment against the hackathon criteria.

```bash
python test_api.py
```

---

Built for the **AI Voice Detection Hackathon**.

