"""
Speech Service
Text-to-Speech (TTS) and Speech-to-Text (STT) routines with runtime configuration.
"""
import io
import base64
from gtts import gTTS
from config import runtime_settings

VOSK_AVAILABLE = False
vosk_model = None

def tts_failsafe(text):
    """Failsafe TTS using gTTS"""
    print(" Using failsafe TTS (gTTS)...")
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode()
        return {"status": "success", "audio": audio_base64}
    except Exception as e:
        print(f" gTTS failsafe error: {e}")
        return {"status": "error", "message": "Could not generate speech"}

def generate_speech(text):
    """
    Generate speech wrapper for gTTS failsafe
    """
    return tts_failsafe(text)
