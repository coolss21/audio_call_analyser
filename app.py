"""
Flask Application for Voice Detection API.
Entry point for Hugging Face Spaces.
"""

import os
import logging
import base64
from flask import Flask, request, jsonify
from src.detector import DeepfakeDetector
from src import config

# Initialize Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB Limit

# Initialize Detector (Global Singleton)
detector: DeepfakeDetector = None

def get_detector():
    """Lazy loader for detector to handle initialization errors gracefully."""
    global detector
    if detector is None:
        try:
            detector = DeepfakeDetector()
            logger.info("Detector initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize detector: {e}")
            detector = None
    return detector

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/detect', methods=['POST'])
@app.route('/api/voice-detection', methods=['POST'])
def detect_voice():
    """
    Main voice detection endpoint.
    
    Expected JSON:
    {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "<base64>"
    }
    """
    model = get_detector()
    if not model:
        return jsonify({
            'status': 'error', 
            'message': 'Service Unavailable: Model not loaded'
        }), 503

    # 1. API Key Validation
    api_key = request.headers.get(config.API_KEY_HEADER)
    expected_key = os.environ.get('API_KEY', config.DEFAULT_API_KEY)
    
    if api_key != expected_key:
        logger.warning(f"Unauthorized access attempt with key: {api_key}")
        return jsonify({
            'status': 'error', 
            'message': 'Invalid API key'
        }), 401

    # 2. Input Validation
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON body'}), 400
            
        required_fields = ['audioBase64']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'status': 'error', 
                'message': f'Missing required fields: {", ".join(missing)}'
            }), 400
            
        # Optional validation for format/language (logged but not strictly enforced to avoid breaking)
        if data.get('audioFormat', '').lower() != 'mp3':
            logger.info(f"Received non-mp3 format: {data.get('audioFormat')}")
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Bad Request: {str(e)}'}), 400

    # 3. Processing
    try:
        audio_b64 = data['audioBase64']
        if ',' in audio_b64:
            audio_b64 = audio_b64.split(',')[1]
            
        audio_bytes = base64.b64decode(audio_b64)
        
        # Inference
        result = model.predict(audio_bytes=audio_bytes)
        
        if result.get('label') == 'error':
            raise RuntimeError(result.get('error'))
            
        # 4. Response Formatting
        classification = "AI_GENERATED" if result['label'] == 'ai' else "HUMAN"
        
        response = {
            "status": "success",
            "classification": classification,
            "confidenceScore": round(float(result['confidence']), 4)
        }
        
        logger.info(f"Processed request: {classification} ({result['confidence']:.4f})")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal Server Error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model = get_detector()
    is_healthy = model is not None
    
    return jsonify({
        'status': 'healthy' if is_healthy else 'unhealthy',
        'model_loaded': is_healthy
    }), 200 if is_healthy else 503

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'status': 'error', 'message': 'Internal request error'}), 500

if __name__ == '__main__':
    # Pre-load model on startup
    get_detector()
    
    print("="*60)
    print(" Voice Detection API - v2.0 (Refactored)")
    print(" Use POST /detect for inference")
    print(f" Threshold: AI > {config.AI_THRESHOLD}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
