"""
Voice Detection API - Hackathon Submission
==========================================
Simplified API matching exact evaluation requirements

Required Response Format:
{
    "status": "success",
    "classification": "HUMAN" or "AI_GENERATED",
    "confidenceScore": 0.85
}
"""

from flask import Flask, request, jsonify
from deepfake_detector import DeepfakeDetector, AudioProcessor
import base64
import os
import logging
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "deepfake_model_multilingual.pt"

# API configuration
API_KEY = "secret123"  # Updated as requested

# Load model
try:
    detector = DeepfakeDetector(MODEL_PATH, device="cpu") # Use CPU for HF Free Tier
    MODEL_LOADED = True
    logger.info("✓ Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    logger.error(f"✗ Failed to load model: {e}")
    detector = None

# ============================================================================
# MAIN DETECTION ENDPOINT (Evaluation uses this)
# ============================================================================

@app.route('/detect', methods=['POST'])
def detect():
    """
    Main voice detection endpoint for hackathon evaluation
    
    Expected Request:
    {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "base64_encoded_audio"
    }
    
    Expected Response:
    {
        "status": "success",
        "classification": "HUMAN" or "AI_GENERATED",
        "confidenceScore": 0.85
    }
    """
    
    if not MODEL_LOADED:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        # Validate API key
        api_key = request.headers.get('x-api-key')
        if api_key != API_KEY:
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key'
            }), 401
        
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body must be JSON'
            }), 400
        
        # Extract fields
        language = data.get('language', '').strip()
        audio_format = data.get('audioFormat', '').strip().lower()
        audio_base64 = data.get('audioBase64', '').strip()
        
        # Validate fields
        if not language:
            return jsonify({
                'status': 'error',
                'message': 'Missing language field'
            }), 400
        
        if not audio_format:
            return jsonify({
                'status': 'error',
                'message': 'Missing audioFormat field'
            }), 400
        
        if audio_format != 'mp3':
            return jsonify({
                'status': 'error',
                'message': 'Only MP3 format is supported'
            }), 400
        
        if not audio_base64:
            return jsonify({
                'status': 'error',
                'message': 'Missing audioBase64 field'
            }), 400
        
        # Decode audio
        try:
            # Handle data URI prefix if present
            if ',' in audio_base64:
                audio_base64 = audio_base64.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to decode audio: {str(e)}'
            }), 400
        
        # Save to temporary file
        temp_path = f"/tmp/audio_{os.getpid()}.mp3"
        try:
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to save audio: {str(e)}'
            }), 500
        
        # Run prediction
        try:
            result = detector.predict(audio_path=temp_path)
            
            # Check for errors
            if result.get('label') == 'error':
                return jsonify({
                    'status': 'error',
                    'message': result.get('error', 'Prediction failed')
                }), 500
            
            # Extract prediction
            label = result['label'].upper()  # 'HUMAN' or 'AI'
            confidence = result['confidence']
            
            # Map to API classification format
            if label == 'AI':
                classification = 'AI_GENERATED'
            else:
                classification = 'HUMAN'
            
            # Return response in exact evaluation format
            return jsonify({
                'status': 'success',
                'classification': classification,
                'confidenceScore': round(confidence, 4)
            }), 200
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

# ============================================================================
# ALTERNATIVE ENDPOINT (for different naming)
# ============================================================================

@app.route('/api/voice-detection', methods=['POST'])
def voice_detection():
    """Alias for /detect endpoint"""
    return detect()

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': MODEL_LOADED
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'status': 'error',
        'message': 'Payload too large'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("VOICE DETECTION API - HACKATHON SUBMISSION")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Status: {'✓ LOADED' if MODEL_LOADED else '✗ NOT LOADED'}")
    print(f"API Key: {API_KEY}")
    print("\nEndpoints:")
    print("  POST /detect              - Main detection endpoint")
    print("  POST /api/voice-detection - Alternative endpoint")
    print("  GET  /health              - Health check")
    print("\nResponse Format:")
    print("  {")
    print("    \"status\": \"success\",")
    print("    \"classification\": \"HUMAN\" or \"AI_GENERATED\",")
    print("    \"confidenceScore\": 0.85")
    print("  }")
    print("="*80)
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
