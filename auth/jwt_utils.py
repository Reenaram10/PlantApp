import base64
import hashlib
import hmac
import json
import time
from config import JWT_SECRET

def b64_encode(data_bytes: bytes) -> str:
    return base64.urlsafe_b64encode(data_bytes).rstrip(b'=').decode('utf-8')

def b64_decode(data_str: str) -> bytes:
    padding = '=' * (4 - (len(data_str) % 4))
    return base64.urlsafe_b64decode(data_str + padding)

def generate_jwt(payload: dict, secret: str = JWT_SECRET, exp_seconds: int = 86400) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload_copy = dict(payload)
    if "exp" not in payload_copy:
        payload_copy["exp"] = int(time.time()) + exp_seconds
    
    header_b64 = b64_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = b64_encode(json.dumps(payload_copy).encode('utf-8'))
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_b64 = b64_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{sig_b64}"

create_token = generate_jwt

def verify_jwt(token: str, secret: str = JWT_SECRET) -> dict | None:
    try:
        if not token:
            return None
        if token.startswith("Bearer "):
            token = token[7:]
        parts = token.strip().split('.')
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = b64_decode(sig_b64)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
            
        payload = json.loads(b64_decode(payload_b64).decode('utf-8'))
        if payload.get("exp") and time.time() > payload["exp"]:
            return None
            
        return payload
    except Exception:
        return None
