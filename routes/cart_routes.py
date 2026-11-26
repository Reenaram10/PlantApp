import os
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.commerce import Cart
from models.catalog import Plant
from auth.decorators import get_current_tenant
from services.image_utils import get_plant_image_url
from services.observability import log_user_interest

cart_bp = Blueprint('cart', __name__)

@cart_bp.route("/api/cart", methods=["POST"])
def add_to_cart():
    data = request.get_json() or {}
    user_id = str(data.get("user_id", "anonymous"))
    plant_id = data.get("plant_id")

    if not plant_id:
        return jsonify({"status": "error", "message": "plant_id required"}), 400

    try:
        existing = Cart.query.filter_by(user_id=user_id, plant_id=plant_id).first()
        if not existing:
            new_item = Cart(user_id=user_id, plant_id=plant_id, business_id=get_current_tenant() or 'NURS-0001')
            db.session.add(new_item)
            db.session.commit()

        #  AUTO LOG PURCHASE
        log_user_interest(user_id, plant_id, "purchase", "Added to cart")

        return jsonify({"status": "success", "message": "Added to cart "})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@cart_bp.route("/api/cart", methods=["GET"])
def get_cart():
    user_id = request.args.get("user_id", "anonymous")

    rows = db.session.execute(text("""
        SELECT p.plant_id, p.plant_name, p.price, p.description, p.image_path, c.quantity, p.stock
        FROM cart_new c
        JOIN plant p ON p.plant_id = c.plant_id
        WHERE c.user_id = :user_id
        ORDER BY c.added_at DESC
    """), {"user_id": user_id}).fetchall()

    cart = []
    subtotal = 0.0
    for r in rows:
        qty = int(r[5] or 1)
        price = float(r[2] or 0.0)
        item_total = round(price * qty, 2)
        subtotal += item_total
        cart.append({
            "plant_id": r[0],
            "plant_name": r[1],
            "price": price,
            "description": r[3],
            "image_url": f"/static/{r[4].lstrip('/')}" if r[4] else None,
            "quantity": qty,
            "stock": r[6] if r[6] is not None else 0,
            "item_total": item_total
        })

    subtotal = round(subtotal, 2)
    delivery_charge = 0.0 if (subtotal >= 500.0 or subtotal == 0.0) else 50.0
    total_amount = round(subtotal + delivery_charge, 2)

    return jsonify({
        "status": "success",
        "cart": cart,
        "subtotal": subtotal,
        "delivery_charge": delivery_charge,
        "total_amount": total_amount,
        "free_delivery_threshold": 500.0,
        "item_count": len(cart)
    })

@cart_bp.route('/api/cart/update', methods=['POST'])
def update_cart():
    try:
        data = request.get_json() or {}
        user_id = str(data.get("user_id", "anonymous"))
        plant_id = data.get("plant_id")
        action = data.get("action") # 'increase' or 'decrease'

        item = Cart.query.filter_by(user_id=user_id, plant_id=plant_id).first()
        if not item:
            return jsonify({"status": "error", "message": "Item not in cart"}), 404

        if action == "increase":
            item.quantity += 1
        elif action == "decrease":
            if item.quantity > 1:
                item.quantity -= 1
            else:
                db.session.delete(item)
        
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@cart_bp.route("/api/cart/remove", methods=["POST"])
def remove_from_cart():
    data = request.get_json() or {}
    user_id = str(data.get("user_id", "anonymous"))
    plant_id = data.get("plant_id")

    if not plant_id:
        return jsonify({"status": "error", "message": "plant_id required"}), 400

    try:
        db.session.execute(text("""
            DELETE FROM cart_new
            WHERE user_id = :user_id AND plant_id = :plant_id
        """), {"user_id": user_id, "plant_id": plant_id})

        db.session.commit()
        return jsonify({"status": "success", "message": "Removed from cart "})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

