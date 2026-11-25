import os
import json
import time
import base64
import requests
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app
from config import APP_ROOT, SCIENTIFIC_NAMES_MAP
from extensions import db
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.image_utils import allowed_file
from services.rag_service import tomato_rag, plant_rag

diagnosis_bp = Blueprint('diagnosis', __name__)

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
N8N_DIAGNOSIS_WEBHOOK = os.getenv("N8N_DIAGNOSIS_WEBHOOK", "http://localhost:5678/webhook/plant-diagnosis")

@diagnosis_bp.route('/api/detect-tomato-disease', methods=['POST'])
def api_detect_tomato_disease():
    """
    Dedicated OpenCV-powered Tomato plant leaf disease detection endpoint.
    Accepts image file under 'image' or 'file' key, or base64 'image_b64' JSON.
    """
    try:
        from plant_disease_detection.tomato_disease_opencv import tomato_detector
        
        img_bytes = None
        if 'image' in request.files:
            img_bytes = request.files['image'].read()
        elif 'file' in request.files:
            img_bytes = request.files['file'].read()
        elif request.is_json:
            data = request.get_json() or {}
            if 'image_b64' in data:
                b64_str = data['image_b64']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)

        if not img_bytes:
            return jsonify({"status": "error", "message": "No tomato leaf image file or base64 provided"}), 400

        result = tomato_detector.predict_tomato_disease(img_bytes)
        return jsonify(result)

    except Exception as e:
        print(f" [Tomato OpenCV API] Error processing request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@diagnosis_bp.route('/api/tools/plant-identify', methods=['POST'])
def n8n_plant_identify():
    """Plant identification tool for n8n."""
    try:
        data = request.get_json(silent=True) or {}
        img_path = data.get('image_path')
        current_app.logger.info("Received image path: %r", img_path)
        current_app.logger.info("Image exists: %s", os.path.isfile(img_path) if img_path else False)

        if not img_path:
            return jsonify({
                "status": "error",
                "message": "image_path is required"
            }), 400

        if not os.path.isfile(img_path):
            return jsonify({
                "status": "error",
                "message": "Image file not found"
            }), 404

        from app.tools.plant_identifier import identify_plant

        plant_info = identify_plant(img_path)

        if not plant_info:
            return jsonify({
                "status": "error",
                "message": "Plant identification returned no result"
            }), 422

        return jsonify({
            "status": "success",
            "tool": "plant_identification",
            "plant": plant_info
        }), 200

    except Exception:
        current_app.logger.exception("Plant identification failed")
        return jsonify({
            "status": "error",
            "message": "Plant identification failed"
        }), 500

@diagnosis_bp.route('/api/tools/disease-identify', methods=['POST'])
def n8n_disease_identify():
    """Disease identification tool for n8n."""
    try:
        data = request.get_json(silent=True) or {}
        plant_name = data.get('plant_name')
        img_path = data.get('image_path')

        if not plant_name or not img_path:
            return jsonify({
                "status": "error",
                "message": "plant_name and image_path are required"
            }), 400

        if not os.path.isfile(img_path):
            return jsonify({
                "status": "error",
                "message": "Image file not found"
            }), 404

        plant_clean = plant_name.strip().title() if plant_name else ""
        if plant_clean not in ["Potato", "Tomato"]:
            return jsonify({
                "status": "success",
                "tool": "disease_identification",
                "disease": {
                    "plant_name": plant_name,
                    "disease_name": "Not Supported",
                    "confidence": 0.0,
                    "status": "unsupported",
                    "image_type": "unknown",
                    "message": f"Disease model is not available for {plant_name}"
                }
            }), 200

        from app.tools.disease_identifier import identify_disease

        disease_info = identify_disease(
            img_path,
            plant_name=plant_name
        )

        if not disease_info:
            return jsonify({
                "status": "error",
                "message": "Disease identification returned no result"
            }), 422

        return jsonify({
            "status": "success",
            "tool": "disease_identification",
            "disease": disease_info
        }), 200

    except Exception:
        current_app.logger.exception("Disease identification failed")
        return jsonify({
            "status": "error",
            "message": "Disease identification failed"
        }), 500

@diagnosis_bp.route('/api/tools/n8n-webhook', methods=['POST'])
def n8n_webhook():
    """n8n Webhook Endpoint for external workflow trigger/logging"""
    try:
        payload = request.json or {}
        return jsonify({
            "status": "success",
            "message": "n8n Webhook received successfully",
            "event": payload.get("event", "n8n_execution"),
            "data": payload.get("data", {})
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@diagnosis_bp.route('/api/plant-diagnosis', methods=['POST'])
@diagnosis_bp.route('/api/identify', methods=['POST'])
def api_plant_diagnosis():
    """
    General plant diagnosis endpoint orchestrating via n8n workflow.
    - Accepts uploaded image files using multipart/form-data ('image' or 'file').
    - Saves image securely in the uploads directory.
    - Generates the correct absolute image path.
    - Sends path to the n8n production webhook (http://localhost:5678/webhook/plant-diagnosis).
    - Uses requests with 30s timeout and graceful connection/timeout handling.
    - Returns diagnosis and remedies to the frontend as unified JSON.
    """
    try:
        # 1. Validate image upload
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided. Please upload an image with key 'image' or 'file'."
            }), 400

        file = request.files.get('image') or request.files.get('file')
        if not file or not file.filename:
            return jsonify({
                "status": "error",
                "message": "Uploaded file is empty or missing a filename."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "message": "Invalid file format. Allowed image formats: JPG, JPEG, PNG, GIF, WEBP."
            }), 400

        # 2. Securely save file with timestamp in uploads directory
        clean_name = secure_filename(file.filename)
        if not clean_name:
            clean_name = "upload.jpg"
        timestamp = int(time.time() * 1000)
        saved_filename = f"{timestamp}_{clean_name}"
        save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(save_path)
        abs_image_path = os.path.abspath(save_path)
        current_app.logger.info("Saved diagnosis image to absolute path: %s", abs_image_path)

        # 3. Call n8n production webhook
        try:
            n8n_res = requests.post(
                N8N_DIAGNOSIS_WEBHOOK,
                json={"image_path": abs_image_path},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
        except requests.exceptions.Timeout:
            current_app.logger.error("n8n plant-diagnosis workflow request timed out")
            return jsonify({
                "status": "error",
                "message": "Plant diagnosis service timed out. Please try again."
            }), 504
        except requests.exceptions.ConnectionError:
            current_app.logger.error("Failed to connect to n8n workflow at %s", N8N_DIAGNOSIS_WEBHOOK)
            return jsonify({
                "status": "error",
                "message": "Could not connect to n8n workflow. Please ensure n8n is running on port 5678."
            }), 503
        except Exception as net_err:
            current_app.logger.error("Error communicating with n8n: %s", net_err)
            return jsonify({
                "status": "error",
                "message": f"n8n communication failure: {str(net_err)}"
            }), 502

        if n8n_res.status_code != 200:
            current_app.logger.error("n8n returned status %s: %s", n8n_res.status_code, n8n_res.text)
            return jsonify({
                "status": "error",
                "message": "Diagnosis workflow execution failed in n8n."
            }), n8n_res.status_code

        # 4. Parse n8n response
        try:
            n8n_data = n8n_res.json()
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Invalid response format received from n8n workflow."
            }), 502

        # 5. Extract and normalize fields
        plant_obj = n8n_data.get("plant") if isinstance(n8n_data.get("plant"), dict) else {}
        plant_name = plant_obj.get("name") or n8n_data.get("plant") or n8n_data.get("plant_name") or "Unknown Plant"
        scientific_name = plant_obj.get("scientific_name") or n8n_data.get("scientific_name") or SCIENTIFIC_NAMES_MAP.get(plant_name, "")
        plant_confidence = plant_obj.get("confidence") or n8n_data.get("plant_confidence")

        disease_obj = n8n_data.get("disease") if isinstance(n8n_data.get("disease"), dict) else {}
        disease_name = disease_obj.get("disease_name") or n8n_data.get("disease") or n8n_data.get("disease_name") or "None Detected"
        disease_confidence = disease_obj.get("confidence") or n8n_data.get("confidence") or n8n_data.get("disease_confidence") or 0.0
        disease_status = (disease_obj.get("status") or n8n_data.get("status") or "detected").lower()
        if disease_status == "success":
            disease_status = "detected"

        treatment = n8n_data.get("treatment") or []
        prevention = n8n_data.get("prevention") or []
        care_tips = n8n_data.get("care_tips") or []

        return jsonify({
            "status": "success",
            "orchestrator": "n8n",
            "plant": {
                "name": plant_name,
                "scientific_name": scientific_name,
                "confidence": plant_confidence
            },
            "disease": {
                "plant_name": plant_name,
                "disease_name": disease_name,
                "confidence": disease_confidence,
                "status": disease_status
            },
            "remedies": {
                "treatment": treatment,
                "prevention": prevention,
                "care_tips": care_tips
            },
            "plant_name": plant_name,
            "scientific_name": scientific_name,
            "disease_name": disease_name,
            "confidence": disease_confidence,
            "treatment": treatment,
            "prevention": prevention,
            "care_tips": care_tips,
            "found": n8n_data.get("found", True),
            "message": n8n_data.get("message", "")
        }), 200

    except Exception as e:
        current_app.logger.exception("General plant diagnosis failed")
        return jsonify({
            "status": "error",
            "message": f"Server error during plant diagnosis: {str(e)}"
        }), 500

