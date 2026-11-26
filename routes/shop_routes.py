import os
import json
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.catalog import Plant
from models.chat import UserInterest
from models.tenant import User
from services.image_utils import get_plant_image_url, search_plant_image

shop_bp = Blueprint('shop', __name__)

@shop_bp.route('/plants', methods=['GET'])
def get_plants():
    try:
        sql = text("SELECT plant_name, description, price FROM plant")
        result = db.session.execute(sql).fetchall()
        plants = [{
            "plant_name": row[0], 
            "description": row[1], 
            "price": row[2]
        } for row in result]
        return jsonify({"status": "success", "plants": plants})
    except Exception as e:
        print(f"Error fetching plants: {str(e)}")
        return jsonify({"status": "error", "message": "Unable to fetch plants"}), 500

@shop_bp.route("/api/shop/plants", methods=["GET"])
def get_shop_plants():
    """Get all plants for the dynamically rendered shop"""
    try:
        plants = Plant.query.all()
        return jsonify({
            "status": "success",
            "plants": [{
                "plant_id": p.plant_id,
                "plant_name": p.plant_name,
                "description": p.description,
                "price": p.price,
                "discount_price": 0.0,
                "effective_price": p.price,
                "stock": p.stock if p.stock is not None else 0,
                "availability_status": "Available" if (p.stock is not None and p.stock > 0) else "Out of Stock",
                "image_url": f"/static/{p.image_path.lstrip('/')}" if p.image_path else None
            } for p in plants]
        })
    except Exception as e:
        print(f"Error in get_shop_plants: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@shop_bp.route('/api/suggest_plants/<user_id>', methods=['GET'])
def suggest_plants(user_id):
    try:
        # Get user's last enquiry or purchase
        last_interest = UserInterest.query.filter_by(user_id=user_id if user_id.isdigit() else 0).order_by(UserInterest.created_at.desc()).first()
        
        if last_interest:
            # Suggest based on same plant or similar (simplified for monolith)
            plants = Plant.query.limit(8).all()
        else:
            plants = Plant.query.limit(8).all()
            
        return jsonify({
            "status": "success",
            "plants": [{
                "plant_id": p.plant_id,
                "plant_name": p.plant_name,
                "description": p.description,
                "price": p.price,
                "discount_price": 0.0,
                "effective_price": p.price,
                "stock": p.stock if p.stock is not None else 0,
                "availability_status": "Available" if (p.stock is not None and p.stock > 0) else "Out of Stock",
                "image_url": get_plant_image_url(p.image_path) if p.image_path else None
            } for p in plants]
        })
    except Exception as e:
        print(f"Suggestion error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@shop_bp.route('/api/user_interest', methods=['POST'])
def save_user_interest():
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")
        plant_id = data.get("plant_id")
        interest_type = data.get("interest_type", "enquiry")
        notes = data.get("notes", "")

        if not user_id or not plant_id:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        interest = UserInterest(
            user_id=int(user_id) if str(user_id).isdigit() else 0,
            plant_id=int(plant_id),
            interest_type=interest_type,
            notes=notes
        )
        db.session.add(interest)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

