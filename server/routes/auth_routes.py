import os
from flask import Blueprint, request, jsonify, g
from sqlalchemy import text
from server.database import db, utc_now
from server.models import User, Business
from server.auth import generate_jwt, verify_jwt, log_audit_action

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/register", methods=["POST"])
@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    business_id = data.get("business_id", "").strip() or "NURS-0001"
    role = data.get("role", "").strip() or "NURSERY_ADMIN"

    if not username or not email or not password:
        return jsonify({
            "status": "error",
            "message": "Username, email and password are required"
        }), 400

    try:
        existing = db.session.execute(
            text("SELECT id FROM login WHERE LOWER(email) = LOWER(:email)"),
            {"email": email}
        ).fetchone()

        if existing:
            return jsonify({
                "status": "error",
                "message": "Email already registered. Please login."
            }), 409

        # Check nursery validity
        biz = db.session.get(Business, business_id)
        if not biz:
            return jsonify({"status": "error", "message": f"Invalid Business ID: {business_id}"}), 404

        new_user = User(
            username=username,
            email=email,
            password=password,
            role=role,
            business_id=business_id
        )
        db.session.add(new_user)
        db.session.commit()
        user_id = new_user.id

        token_payload = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "business_id": business_id
        }
        token = generate_jwt(token_payload)

        log_audit_action("USER_REGISTER", "AUTH", record_id=user_id, details=f"Registered user {email} under {business_id}", business_id=business_id, user_id=user_id, user_email=email)

        return jsonify({
            "status": "success",
            "message": "Registered successfully",
            "token": token,
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "business_id": business_id,
            "business_name": biz.business_name
        })

    except Exception as e:
        print(f"Register error: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500



# Multi-tenant Login Endpoint
@auth_bp.route("/login", methods=["POST"])
@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    req_business_id = data.get("business_id", "").strip()

    if not email or not password:
        return jsonify({
            "status": "error",
            "message": "Email/Username and password required"
        }), 400

    try:
        user = db.session.execute(text("""
            SELECT id, username, email, password, role, business_id
            FROM login
            WHERE (LOWER(email) = :email OR LOWER(username) = :email) AND password = :password
        """), {
            "email": email,
            "password": password
        }).fetchone()

        if not user:
            return jsonify({
                "status": "error",
                "message": "Invalid credentials"
            }), 401

        user_id, username, user_email, _, role, user_biz_id = user

        # Handle Super Admin login
        if role == 'SUPER_ADMIN' or (user_email and user_email.lower() == 'admin@system.com'):
            token_payload = {
                "user_id": user_id,
                "username": username or "Super Admin",
                "email": user_email,
                "role": "SUPER_ADMIN",
                "business_id": "SYSTEM"
            }
            token = generate_jwt(token_payload)
            log_audit_action("SUPER_ADMIN_LOGIN", "AUTH", record_id=user_id, details="Super Admin logged in", business_id="SYSTEM", user_id=user_id, user_email=user_email)
            return jsonify({
                "status": "success",
                "token": token,
                "user_id": user_id,
                "username": username or "Super Admin",
                "email": user_email,
                "role": "SUPER_ADMIN",
                "business_id": "SYSTEM",
                "business_name": "Platform Administration"
            })

        # For Nursery Admin / Staff, verify Business ID if provided
        target_biz_id = user_biz_id or "NURS-0001"
        if req_business_id and req_business_id.upper() != target_biz_id.upper():
            return jsonify({
                "status": "error",
                "message": f"User does not belong to Business ID: {req_business_id}"
            }), 403

        # Verify Business is Active
        biz = db.session.get(Business, target_biz_id)
        if not biz:
            return jsonify({"status": "error", "message": f"Nursery {target_biz_id} not found."}), 404
        if biz.status == "Inactive":
            return jsonify({"status": "error", "message": "This nursery account has been deactivated. Please contact Super Admin."}), 403

        token_payload = {
            "user_id": user_id,
            "username": username or user_email,
            "email": user_email,
            "role": role or "NURSERY_ADMIN",
            "business_id": target_biz_id
        }
        token = generate_jwt(token_payload)

        # Update login tracking
        try:
            db.session.execute(text("UPDATE login SET last_login = :now, is_online = 1 WHERE id = :id"), {"now": utc_now(), "id": user_id})
            db.session.commit()
        except:
            db.session.rollback()

        log_audit_action("USER_LOGIN", "AUTH", record_id=user_id, details=f"Logged in as {role} for {target_biz_id}", business_id=target_biz_id, user_id=user_id, user_email=user_email)

        return jsonify({
            "status": "success",
            "token": token,
            "user_id": user_id,
            "username": username or user_email,
            "email": user_email,
            "role": role or "NURSERY_ADMIN",
            "business_id": target_biz_id,
            "business_name": biz.business_name
        })

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500




@auth_bp.route('/logout', methods=['POST'])
def logout():
    data = request.json
    user_id = data.get("user_id")

    if user_id:
        db.session.execute(text("""
            UPDATE login
            SET last_logout = CURRENT_TIMESTAMP,
                is_online = 0
            WHERE id = :uid
        """), {"uid": user_id})

        db.session.commit()

    return jsonify({'status': 'success'})




@auth_bp.route('/api/user/phone', methods=['POST'])
def update_user_phone():
    """Update or link phone number for an existing user account."""
    data = request.json or {}
    user_id = str(data.get('user_id', '')).strip()
    phone = str(data.get('phone', '')).strip()

    if not user_id or not phone:
        return jsonify({'status': 'error', 'message': 'user_id and phone are required'}), 400

    user = None
    if user_id.isdigit():
        user = User.query.filter(User.id == int(user_id)).first()
    if not user:
        user = User.query.filter(User.username == user_id).first()

    if not user:
        return jsonify({'status': 'error', 'message': 'User account not found'}), 404

    user.phone = phone
    db.session.commit()
    return jsonify({
        'status': 'success',
        'message': 'Phone number linked successfully',
        'phone': user.phone,
        'username': user.username
    }), 200

