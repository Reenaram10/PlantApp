import os
from flask import Blueprint, request, jsonify, g
from sqlalchemy import text, func
from server.database import db, utc_now
from server.models import Business, User, Plant, Order, AuditLog
from server.auth import generate_jwt, log_audit_action, require_tenant_auth

super_admin_bp = Blueprint('super_admin', __name__)

# ========== SUPER ADMIN PLATFORM ENDPOINTS ==========
@super_admin_bp.route("/api/super-admin/login", methods=["POST"])
@super_admin_bp.route("/api/auth/super-admin/login", methods=["POST"])
def super_admin_login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if email != "admin@system.com" or password != "admin123":
        user = db.session.execute(text("SELECT id, username, email FROM login WHERE LOWER(email) = :email AND password = :pwd AND role = 'SUPER_ADMIN'"), {"email": email, "pwd": password}).fetchone()
        if not user:
            return jsonify({"status": "error", "message": "Invalid Super Admin credentials"}), 401

    token = generate_jwt({
        "user_id": 1,
        "username": "Super Admin",
        "email": email,
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM"
    })
    return jsonify({
        "status": "success",
        "token": token,
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM",
        "username": "Super Admin"
    })

@super_admin_bp.route("/api/super-admin/nurseries", methods=["POST"])
@super_admin_bp.route("/api/super-admin/nursery", methods=["POST"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def register_new_nursery():
    data = request.json or {}
    business_name = (data.get("business_name") or data.get("name") or "").strip()
    owner_name = (data.get("owner_name") or data.get("owner") or "").strip()
    email = data.get("email", "").strip().lower()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    city = data.get("city", "").strip()
    state = data.get("state", "").strip()
    country = data.get("country", "").strip()
    
    admin_name = data.get("admin_name", "").strip() or owner_name or "Nursery Admin"
    admin_email = data.get("admin_email", "").strip().lower() or email
    admin_password = data.get("admin_password", "").strip() or "admin123"

    if not business_name or not admin_email:
        return jsonify({"status": "error", "message": "Nursery Name and Admin Email are required"}), 400

    try:
        # Auto-generate unique Business ID (NURS-XXXX)
        last_biz = db.session.execute(text("SELECT TOP 1 business_id FROM businesses WHERE business_id LIKE 'NURS-%' ORDER BY business_id DESC")).fetchone()
        if last_biz and last_biz[0]:
            try:
                num = int(last_biz[0].replace("NURS-", "")) + 1
            except:
                num = 2
        else:
            num = 1
        
        new_biz_id = f"NURS-{num:04d}"
        
        # Ensure ID is unique
        while db.session.get(Business, new_biz_id) is not None:
            num += 1
            new_biz_id = f"NURS-{num:04d}"

        # Insert new nursery entity
        new_biz = Business(
            business_id=new_biz_id,
            business_name=business_name,
            owner_name=owner_name,
            email=email,
            phone=phone,
            address=address,
            city=city,
            state=state,
            country=country,
            status="Active"
        )
        db.session.add(new_biz)

        # Create Nursery Admin account
        from sqlalchemy import func
        existing_user = User.query.filter(func.lower(User.email) == admin_email.lower()).first()
        if existing_user:
            existing_user.business_id = new_biz_id
            existing_user.role = "NURSERY_ADMIN"
            existing_user.password = admin_password
        else:
            admin_user = User(
                username=admin_name,
                email=admin_email,
                password=admin_password,
                role="NURSERY_ADMIN",
                business_id=new_biz_id
            )
            db.session.add(admin_user)

        db.session.commit()

        log_audit_action("CREATE_NURSERY", "SUPER_ADMIN", record_id=new_biz_id, details=f"Registered nursery {business_name} ({new_biz_id})", business_id="SYSTEM")

        return jsonify({
            "status": "success",
            "message": f"Nursery '{business_name}' registered successfully!",
            "business_id": new_biz_id,
            "business_name": business_name,
            "admin_email": admin_email,
            "business": {
                "business_id": new_biz_id,
                "business_name": business_name,
                "owner_name": owner_name,
                "email": email,
                "phone": phone
            }
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error registering nursery: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@super_admin_bp.route("/api/super-admin/nurseries", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def list_nurseries_super_admin():
    search = request.args.get("search", "").strip().lower()
    query = Business.query
    if search:
        query = query.filter((Business.business_name.ilike(f"%{search}%")) | (Business.business_id.ilike(f"%{search}%")) | (Business.email.ilike(f"%{search}%")))
    
    nurseries = query.order_by(Business.created_at.desc()).all()
    result = []
    for n in nurseries:
        admin = User.query.filter_by(business_id=n.business_id, role="NURSERY_ADMIN").first()
        plant_count = Plant.query.filter_by(business_id=n.business_id).count()
        order_count = Order.query.filter_by(business_id=n.business_id).count()
        result.append({
            "business_id": n.business_id,
            "business_name": n.business_name,
            "owner_name": n.owner_name,
            "email": n.email,
            "phone": n.phone,
            "address": n.address,
            "city": n.city,
            "state": n.state,
            "country": n.country,
            "status": n.status,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "admin_email": admin.email if (admin and admin.email) else (n.email if n.email else None),
            "plant_count": plant_count,
            "order_count": order_count
        })
    return jsonify({"status": "success", "nurseries": result})

@super_admin_bp.route("/api/super-admin/nurseries/<business_id>", methods=["DELETE"])
@super_admin_bp.route("/api/super-admin/nursery/<business_id>", methods=["DELETE"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def delete_nursery_super_admin(business_id):
    if business_id in ["SYSTEM", "NURS-0001"]:
        return jsonify({"status": "error", "message": "Cannot delete primary default nursery"}), 400

    biz = db.session.get(Business, business_id)
    if not biz:
        return jsonify({"status": "error", "message": "Nursery not found"}), 404

    try:
        # Cascade raw SQL cleanup of child tables
        db.session.execute(text("DELETE FROM login WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM plant WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM inventory_transaction WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM user_queries WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM user_interest WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM audit_logs WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM businesses WHERE business_id = :bid"), {"bid": business_id})
        db.session.commit()

        log_audit_action("DELETE_NURSERY", "SUPER_ADMIN", record_id=business_id, details=f"Deleted nursery {business_id}", business_id="SYSTEM")
        return jsonify({"status": "success", "message": f"Nursery {business_id} deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@super_admin_bp.route("/api/super-admin/nurseries/<business_id>/status", methods=["PATCH", "PUT"])
@super_admin_bp.route("/api/super-admin/nursery/<business_id>/status", methods=["PATCH", "PUT"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def toggle_nursery_status(business_id):
    data = request.json or {}
    new_status = data.get("status", "").strip()
    if new_status not in ["Active", "Inactive"]:
        return jsonify({"status": "error", "message": "Status must be 'Active' or 'Inactive'"}), 400

    biz = db.session.get(Business, business_id)
    if not biz:
        return jsonify({"status": "error", "message": "Nursery not found"}), 404

    biz.status = new_status
    db.session.commit()

    log_audit_action("TOGGLE_NURSERY_STATUS", "SUPER_ADMIN", record_id=business_id, details=f"Changed status of {business_id} to {new_status}", business_id="SYSTEM")

    return jsonify({"status": "success", "message": f"Nursery {business_id} status updated to {new_status}", "business_id": business_id, "status": new_status})

@super_admin_bp.route("/api/super-admin/stats", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def super_admin_stats():
    total_nurseries = Business.query.count()
    active_nurseries = Business.query.filter_by(status="Active").count()
    inactive_nurseries = Business.query.filter_by(status="Inactive").count()
    total_users = User.query.count()
    total_plants = Plant.query.count()
    total_orders = Order.query.count()

    return jsonify({
        "status": "success",
        "stats": {
            "total_nurseries": total_nurseries,
            "active_nurseries": active_nurseries,
            "inactive_nurseries": inactive_nurseries,
            "total_users": total_users,
            "total_plants": total_plants,
            "total_orders": total_orders
        }
    })

@super_admin_bp.route("/api/super-admin/audit-logs", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def get_super_admin_audit_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    res = []
    for l in logs:
        res.append({
            "id": l.id,
            "business_id": l.business_id,
            "user_id": l.user_id,
            "user_email": l.user_email,
            "action": l.action,
            "module": l.module,
            "record_id": l.record_id,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None,
            "details": l.details
        })
    return jsonify({"status": "success", "audit_logs": res})

