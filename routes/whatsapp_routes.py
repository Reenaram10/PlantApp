import os
import re
import json
from typing import Any
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.commerce import Cart, Order
from models.catalog import Plant
from models.tenant import User
from auth.decorators import get_current_tenant
from routes.order_routes import checkout, build_order_timeline
from services.image_utils import get_plant_image_url

whatsapp_bp = Blueprint('whatsapp', __name__)

def normalize_phone(phone_str: str) -> str:
    """Normalize phone number to last 10 digits or digits only."""
    if not phone_str:
        return ""
    digits = re.sub(r'\D', '', phone_str)
    if len(digits) > 10:
        return digits[-10:]
    return digits

def find_user_by_phone_or_id(identifier: str):
    """Find registered user by phone or user_id."""
    if not identifier:
        return None
    id_str = identifier.strip()
    norm_phone = normalize_phone(id_str)

    if norm_phone:
        user = db.session.execute(text("""
            SELECT id, username, email, phone, role, business_id
            FROM login
            WHERE phone LIKE :ph OR phone LIKE :fullph
        """), {"ph": f"%{norm_phone}", "fullph": f"%{id_str}%"}).fetchone()
        if user:
            return user

    if id_str.isdigit():
        user = db.session.execute(text("""
            SELECT id, username, email, phone, role, business_id
            FROM login
            WHERE id = :uid
        """), {"uid": int(id_str)}).fetchone()
        if user:
            return user

    return None

@whatsapp_bp.route('/api/whatsapp/cart', methods=['GET'])
def whatsapp_get_cart():
    """
    Retrieve customer cart for WhatsApp n8n workflow.
    Validates customer phone number or user ID.
    Returns structured JSON and WhatsApp message text template.
    """
    phone = request.args.get("phone", "").strip()
    user_id = request.args.get("user_id", "").strip()

    user = find_user_by_phone_or_id(phone or user_id)
    target_uid = str(user[0]) if user else (user_id or phone)

    if not target_uid:
        return jsonify({
            "status": "error",
            "message": "Valid phone number or user_id required"
        }), 400

    rows = db.session.execute(text("""
        SELECT p.plant_id, p.plant_name, p.price, p.description, p.image_path, c.quantity, p.stock
        FROM cart_new c
        JOIN plant p ON p.plant_id = c.plant_id
        WHERE c.user_id = :u
        ORDER BY c.added_at DESC
    """), {"u": target_uid}).fetchall()

    cart_items = []
    subtotal = 0.0
    for r in rows:
        qty = int(r[5] or 1)
        price = float(r[2] or 0.0)
        item_total = round(price * qty, 2)
        subtotal += item_total
        cart_items.append({
            "plant_id": r[0],
            "plant_name": r[1],
            "price": price,
            "quantity": qty,
            "item_total": item_total,
            "image_url": f"/static/{r[4].lstrip('/')}" if r[4] else None
        })

    subtotal = round(subtotal, 2)
    delivery_charge = 0.0 if (subtotal >= 500.0 or subtotal == 0.0) else 50.0
    total_amount = round(subtotal + delivery_charge, 2)

    if not cart_items:
        wa_text = "🛒 Your PlantApp cart is currently empty.\nVisit our catalog to add fresh plants!"
    else:
        item_lines = [f"• *{item['plant_name']}* x{item['quantity']} - ₹{item['item_total']}" for item in cart_items]
        wa_text = (
            f"🌿 *Your PlantApp Cart:*\n\n" +
            "\n".join(item_lines) +
            f"\n\nSubtotal: ₹{subtotal}\n"
            f"Delivery: {'FREE' if delivery_charge == 0 else f'₹{delivery_charge}'}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Total: ₹{total_amount}*\n\n"
            f"Reply *APPROVE* to confirm your order or *CHECKOUT* to proceed."
        )

    return jsonify({
        "status": "success",
        "user_id": target_uid,
        "customer_phone": phone or (user[3] if user else None),
        "cart": cart_items,
        "subtotal": subtotal,
        "delivery_charge": delivery_charge,
        "total_amount": total_amount,
        "whatsapp_text": wa_text,
        "whatsapp_message": wa_text
    }), 200

@whatsapp_bp.route('/api/whatsapp/cart/approve', methods=['POST'])
def whatsapp_cart_approve():
    """
    Approve cart via WhatsApp.
    Validates customer cart and returns approval confirmation with checkout payload.
    """
    data = request.get_json() or {}
    phone = data.get("phone", "").strip()
    user_id = data.get("user_id", "").strip()

    user = find_user_by_phone_or_id(phone or user_id)
    target_uid = str(user[0]) if user else (user_id or phone)

    if not target_uid:
        return jsonify({"status": "error", "message": "Phone or user_id required"}), 400

    rows = db.session.execute(text("SELECT plant_id, quantity FROM cart_new WHERE user_id = :u"), {"u": target_uid}).fetchall()
    if not rows:
        return jsonify({"status": "error", "message": "Cannot approve an empty cart"}), 400

    return jsonify({
        "status": "success",
        "approved": True,
        "is_approved": True,
        "user_id": target_uid,
        "phone": phone or (user[3] if user else None),
        "item_count": len(rows),
        "message": "Cart approved successfully. Proceed to checkout."
    }), 200

@whatsapp_bp.route('/api/whatsapp/cart/update', methods=['POST'])
def whatsapp_cart_update():
    """Update cart item quantity or remove item via WhatsApp."""
    data = request.get_json() or {}
    phone = data.get("phone", "").strip()
    user_id = data.get("user_id", "").strip()
    plant_id = data.get("plant_id")
    plant_name = (data.get("plant_name") or "").strip()
    action = data.get("action", "increase") # 'increase', 'decrease', 'remove'

    user = find_user_by_phone_or_id(phone or user_id)
    target_uid = str(user[0]) if user else (user_id or phone)

    if not plant_id and plant_name:
        plant_row = Plant.query.filter(Plant.plant_name.ilike(f"%{plant_name}%")).first()
        if plant_row:
            plant_id = plant_row.plant_id

    if not target_uid or not plant_id:
        return jsonify({"status": "error", "message": "User identifier and plant_id or plant_name required"}), 400

    item = Cart.query.filter_by(user_id=target_uid, plant_id=plant_id).first()
    if not item:
        if action == "increase":
            item = Cart(user_id=target_uid, plant_id=plant_id, quantity=1, business_id=get_current_tenant() or 'NURS-0001')
            db.session.add(item)
            db.session.commit()
            return jsonify({"status": "success", "message": "Item added to cart via WhatsApp"}), 200
        else:
            return jsonify({"status": "error", "message": "Item not in cart"}), 404

    if action == "increase":
        item.quantity += 1
    elif action == "decrease":
        if item.quantity > 1:
            item.quantity -= 1
        else:
            db.session.delete(item)
    elif action == "remove":
        db.session.delete(item)

    db.session.commit()
    return jsonify({"status": "success", "message": f"Cart updated ({action})"}), 200

@whatsapp_bp.route('/api/whatsapp/checkout', methods=['POST'])
def whatsapp_checkout():
    """Create order directly from WhatsApp n8n webhook payload."""
    data = request.get_json() or {}
    phone = data.get("phone", "").strip()
    user_id = data.get("user_id", "").strip()
    customer_name = data.get("customer_name") or "WhatsApp Customer"
    address = data.get("address", "").strip()
    payment_method = data.get("payment_method", "Cash on Delivery").strip()

    user = find_user_by_phone_or_id(phone or user_id)
    target_uid = str(user[0]) if user else (user_id or phone)

    if not target_uid:
        return jsonify({"status": "error", "message": "Phone number or user_id required"}), 400
    if not address:
        return jsonify({"status": "error", "message": "Delivery address required"}), 400

    data["user_id"] = target_uid
    data["phone"] = phone or (user[3] if user else "")
    data["customer_name"] = customer_name
    data["address"] = address
    data["payment_method"] = payment_method

    # Re-use checkout endpoint logic
    return checkout()

@whatsapp_bp.route('/api/whatsapp/orders', methods=['GET'])
def whatsapp_get_orders():
    """Retrieve orders for customer mapped by WhatsApp phone number."""
    phone = request.args.get("phone", "").strip()
    user_id = request.args.get("user_id", "").strip()

    user = find_user_by_phone_or_id(phone or user_id)
    if not user and not phone:
        return jsonify({"status": "error", "message": "Registered phone number or account required"}), 404

    target_uid = str(user[0]) if user else user_id
    norm_phone = normalize_phone(phone)

    where_clauses = []
    params: dict[str, Any] = {}
    if target_uid and target_uid.isdigit():
        where_clauses.append("o.user_id = :uid")
        params["uid"] = int(target_uid)
    if norm_phone:
        where_clauses.append("(o.shipping_address LIKE :ph OR o.notes LIKE :ph)")
        params["ph"] = f"%{norm_phone}%"

    if not where_clauses:
        return jsonify({"status": "error", "message": "No customer identifier provided"}), 400

    rows = db.session.execute(text(f"""
        SELECT o.order_id, o.order_group_id, o.plant_name, o.price, o.quantity,
               o.total_amount, o.order_status, o.payment_status, o.payment_method,
               o.shipping_address, o.tracking_number, o.delivery_date, o.order_date
        FROM orders o
        WHERE {" OR ".join(where_clauses)}
        ORDER BY o.order_date DESC
    """), params).fetchall()

    grouped = {}
    for r in rows:
        gid = r[1] or f"OLD-{r[0]}"
        if gid not in grouped:
            grouped[gid] = {
                "order_group_id": gid,
                "order_date": r[12].strftime("%Y-%m-%d %H:%M") if (r[12] and hasattr(r[12], 'strftime')) else str(r[12] or "N/A"),
                "order_status": r[6] or "Order Placed",
                "payment_status": r[7] or "Pending",
                "total_amount": 0.0,
                "tracking_number": r[10] or f"GB-TRK-{gid}",
                "items": []
            }
        qty = int(r[4] or 1)
        price = float(r[3] or 0.0)
        item_total = float(r[5]) if r[5] is not None else price * qty
        grouped[gid]["total_amount"] += item_total
        grouped[gid]["items"].append(f"{r[2]} x{qty}")

    orders_list = list(grouped.values())
    for o in orders_list:
        o["total_amount"] = round(o["total_amount"], 2)

    return jsonify({
        "status": "success",
        "phone": phone,
        "orders_count": len(orders_list),
        "orders": orders_list
    }), 200

@whatsapp_bp.route('/api/whatsapp/orders/track', methods=['GET'])
def whatsapp_track_order():
    """Retrieve order tracking status for WhatsApp notifications."""
    order_id = request.args.get("order_id", "").strip()
    phone = request.args.get("phone", "").strip()

    if not order_id:
        return jsonify({"status": "error", "message": "order_id required"}), 400

    row = db.session.execute(text("""
        SELECT TOP 1 order_group_id, order_status, tracking_number, delivery_date, shipping_address
        FROM orders
        WHERE order_group_id = :oid OR CAST(order_id AS VARCHAR(20)) = :oid
    """), {"oid": order_id}).fetchone()

    if not row:
        return jsonify({"status": "error", "message": "Order not found"}), 404

    # Security check: if phone is provided, verify it belongs to this order
    if phone:
        norm_phone = normalize_phone(phone)
        shipping_addr = row[4] or ""
        if norm_phone not in normalize_phone(shipping_addr):
            user = find_user_by_phone_or_id(phone)
            if not user:
                return jsonify({"status": "error", "message": "Unauthorized: Phone number does not match order records"}), 403

    current_status = row[1] or "Order Placed"
    timeline = build_order_timeline(current_status)

    return jsonify({
        "status": "success",
        "order_group_id": row[0] or order_id,
        "current_status": current_status,
        "tracking_number": row[2] or f"GB-TRK-{order_id}",
        "estimated_delivery": row[3] or "3-5 Business Days",
        "timeline": timeline,
        "whatsapp_text": f"📦 *Order Status Update for #{row[0] or order_id}:*\n"
                         f"Status: *{current_status}*\n"
                         f"Tracking No: `{row[2] or 'N/A'}`\n"
                         f"Estimated Delivery: {row[3] or '3-5 Business Days'}"
    }), 200

