import os
from flask import Blueprint, request, jsonify, current_app
from config import runtime_settings
from extensions import db

config_bp = Blueprint('config', __name__)

@config_bp.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration options"""
    try:
        if request.method == "GET":
            return jsonify({
                "status": "success",
                "config": {
                    "LLM_OPTION": runtime_settings.llm_option,
                    "STT_OPTION": runtime_settings.stt_option,
                    "TTS_OPTION": runtime_settings.tts_option,
                    "LLM_FAILSAFE": runtime_settings.llm_failsafe,
                    "STT_FAILSAFE": runtime_settings.stt_failsafe,
                    "TTS_FAILSAFE": runtime_settings.tts_failsafe
                }
            })

        elif request.method == "POST":
            data = request.json
            
            if "LLM_OPTION" in data:
                runtime_settings.llm_option = data["LLM_OPTION"]
                print(f" LLM changed to: {runtime_settings.llm_option}")
            
            if "STT_OPTION" in data:
                runtime_settings.stt_option = data["STT_OPTION"]
                print(f" STT changed to: {runtime_settings.stt_option}")
            
            if "TTS_OPTION" in data:
                runtime_settings.tts_option = data["TTS_OPTION"]
                print(f" TTS changed to: {runtime_settings.tts_option}")
            
            return jsonify({
                "status": "success",
                "message": "Configuration updated",
                "config": {
                    "LLM_OPTION": runtime_settings.llm_option,
                    "STT_OPTION": runtime_settings.stt_option,
                    "TTS_OPTION": runtime_settings.tts_option
                }
            })

    except Exception as e:
        print(f"Config API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@config_bp.route("/api/stt", methods=["POST", "GET"])
def stt_api():
    """Speech-to-Text (STT) API Endpoint"""
    try:
        if request.method == "GET":
            return jsonify({
                "status": "success",
                "stt_option": runtime_settings.stt_option,
                "stt_failsafe": runtime_settings.stt_failsafe,
                "supported": True
            })

        data = request.get_json() or {}
        text_input = data.get("text", "")
        if text_input:
            return jsonify({
                "status": "success",
                "transcript": text_input,
                "engine": runtime_settings.stt_option
            })

        return jsonify({
            "status": "success",
            "message": "STT active. Use browser Web Speech API or send audio/text transcript.",
            "stt_option": runtime_settings.stt_option
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@config_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
