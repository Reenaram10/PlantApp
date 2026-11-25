from functools import wraps
from flask import g, request, jsonify
from extensions import db
from models.tenant import Business
from auth.jwt_utils import verify_jwt

def get_current_tenant() -> str:
    """Helper to extract authenticated user's tenant ID from request context"""
    if hasattr(g, 'current_user') and g.current_user and g.current_user.get('business_id'):
        return g.current_user['business_id']
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        payload = verify_jwt(auth_header)
        if payload and payload.get('business_id'):
            return payload['business_id']
    return 'NURS-0001'

def require_tenant_auth(allowed_roles=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            payload = verify_jwt(auth_header)
            
            if not payload:
                token = request.args.get("token")
                if not token and request.is_json:
                    try:
                        body = request.get_json(silent=True)
                        if isinstance(body, dict):
                            token = body.get("token")
                    except Exception:
                        token = None
                payload = verify_jwt(token)
                
            if not payload:
                # Default guest context for public store operations
                g.current_user = {
                    "user_id": 0,
                    "email": "guest@nursery.com",
                    "role": "STAFF",
                    "business_id": "NURS-0001"
                }
            else:
                g.current_user = payload

            user_biz = g.current_user.get("business_id", "NURS-0001")
            user_role = g.current_user.get("role", "STAFF")

            if user_role != "SUPER_ADMIN" and user_biz != "SYSTEM":
                biz = db.session.get(Business, user_biz)
                if biz and biz.status == "Inactive":
                    return jsonify({"status": "error", "message": "This nursery account has been deactivated."}), 403

            if allowed_roles:
                if user_role not in allowed_roles:
                    return jsonify({"status": "error", "message": "Access denied: Unauthorized role"}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator
