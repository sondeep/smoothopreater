from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import cv2
import numpy as np
import base64
import os
import requests 

app = Flask(__name__, template_folder='templates')

print("⏳ Initializing Skywatch Backend...")

# ==========================================
# 🔧 ROBOFLOW CONFIGURATION
# ==========================================
ROBOFLOW_API_KEY = "860lkeYzSam08d3D9wl3" 
PROJECT_ID = "damage-assessment" 
VERSION_NUMBER = 1 

# Initialize SDK for REST API (Legacy Frame-by-Frame Fallback)
model = None
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(PROJECT_ID)
    model = project.version(VERSION_NUMBER).model
    print(f"✅ Roboflow REST model loaded!")
except Exception as e:
    print(f"⚠️ REST Model warning (Check API Key): {e}")

@app.route("/")
def index():
    return render_template("index.html")

# ==========================================
# 📡 WEBRTC PROXY ENDPOINT
# Handles the secure handshake for the live stream
# ==========================================
@app.route("/api/init-webrtc", methods=['POST'])
def init_webrtc():
    try:
        # 1. Get the Offer and Params from Frontend
        client_data = request.json
        offer = client_data.get('offer')
        wrtc_params = client_data.get('wrtcParams', {})

        if not offer:
            return jsonify({'error': 'No WebRTC offer provided'}), 400

        # 2. Construct Payload for Roboflow API
        roboflow_payload = {
            "offer": offer,
            "api_key": ROBOFLOW_API_KEY,
            "workspace_name": wrtc_params.get('workspaceName'),
            "workflow_id": wrtc_params.get('workflowId'),
            "stream_output_names": wrtc_params.get('streamOutputNames', ["visualization"]),
            "data_output_names": wrtc_params.get('dataOutputNames', ["predictions"]),
            "processing_timeout": wrtc_params.get('processingTimeout', 600),
            "requested_plan": wrtc_params.get('requestedPlan', "webrtc-gpu-medium"),
            "requested_region": wrtc_params.get('requestedRegion', "us")
        }

        # 3. Forward to Roboflow Inference Server
        roboflow_url = "https://serverless.roboflow.com/webrtc/init"
        
        response = requests.post(roboflow_url, json=roboflow_payload)
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to initialize with Roboflow', 'details': response.json()}), response.status_code

        # 4. Return the Answer to Frontend
        return jsonify(response.json())

    except Exception as e:
        print(f"Server Proxy Error: {e}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# 📸 REST API ENDPOINT (Fallback/Local Cam)
# Frame-by-frame processing
# ==========================================
@app.route("/predict", methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not configured'}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temp file
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, img)

        # Run Inference
        prediction_group = model.predict(temp_filename, confidence=40, overlap=30).json()
        
        detections = []
        if 'predictions' in prediction_group:
            for p in prediction_group['predictions']:
                x1 = p['x'] - (p['width'] / 2)
                y1 = p['y'] - (p['height'] / 2)
                
                conf = p['confidence']
                severity = "critical" if conf > 0.8 else "moderate" if conf > 0.6 else "minor"

                detections.append({
                    "class": p['class'],
                    "type": p['class'], 
                    "confidence": conf,
                    "bbox": [x1, y1, p['width'], p['height']],
                    "severity": severity
                })

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
