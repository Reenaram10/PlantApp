import os
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.catalog import Category, PlantCategory, Plant, CategorySynonym
from auth.decorators import require_tenant_auth
from services.image_utils import get_plant_image_url

category_bp = Blueprint('category', __name__)

def add_category_synonym(synonym, category_id):
    """Add a synonym for an existing category"""
    try:
        syn = CategorySynonym(
            synonym=synonym.lower(),
            category_id=category_id
        )
        db.session.add(syn)
        db.session.commit()
        print(f" Synonym '{synonym}' added for category_id {category_id}")
        return syn.synonym_id
    except Exception as e:
        print(f"Error adding synonym: {e}")
        db.session.rollback()
        return None

@category_bp.route('/api/admin/categories', methods=['GET'])
def get_categories_admin():
    """Get all categories for admin"""
    try:
        rows = db.session.execute(text("SELECT category_id, category_name FROM category ORDER BY category_name ASC")).fetchall()
        categories = [{"category_id": r[0], "category_name": r[1]} for r in rows]
        return jsonify({"status": "success", "categories": categories})
    except Exception as e:
        print(f"Error in get_categories_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@category_bp.route('/api/admin/categories', methods=['POST'])
def add_category_admin():
    """Add a new category"""
    try:
        data = request.get_json() or {}
        name = data.get('category_name')
        if not name:
            return jsonify({"status": "error", "message": "Category name required"}), 400
        
        existing_cat = Category.query.filter_by(category_name=name).first()
        if not existing_cat:
            new_cat = Category(category_name=name)
            db.session.add(new_cat)
            db.session.commit()
        return jsonify({"status": "success", "message": "Category saved"})
    except Exception as e:
        print(f"Error in add_category_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@category_bp.route('/api/admin/categories/<int:cat_id>', methods=['DELETE'])
def delete_category_admin(cat_id):
    """Delete a category"""
    try:
        db.session.execute(text("DELETE FROM plant_category WHERE category_id = :id"), {"id": cat_id})
        db.session.execute(text("DELETE FROM category WHERE category_id = :id"), {"id": cat_id})
        db.session.commit()
        return jsonify({"status": "success", "message": "Category deleted"})
    except Exception as e:
        print(f"Error in delete_category_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@category_bp.route('/api/admin/categories/<int:cat_id>/plants', methods=['GET'])
def get_category_plants_admin(cat_id):
    """Get all plants for a category"""
    try:
        rows = db.session.execute(text("""
            SELECT p.plant_id, p.plant_name, p.stock, p.price, p.image_path
            FROM plant p
            JOIN plant_category pc ON pc.plant_id = p.plant_id
            WHERE pc.category_id = :cid
        """), {"cid": cat_id}).fetchall()
        plants = [{
            "plant_id": r[0],
            "plant_name": r[1],
            "stock": r[2],
            "price": float(r[3]) if r[3] else 0,
            "image_url": get_plant_image_url(r[4])
        } for r in rows]
        return jsonify({"status": "success", "plants": plants})
    except Exception as e:
        print(f"Error in get_category_plants_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@category_bp.route('/api/category_synonyms', methods=['GET', 'POST', 'DELETE'])
def manage_category_synonyms():
    """Manage category synonyms"""
    try:
        if request.method == "GET":
            synonyms = CategorySynonym.query.all()
            return jsonify({
                "status": "success",
                "synonyms": [{
                    "synonym_id": s.synonym_id,
                    "synonym": s.synonym,
                    "category_id": s.category_id,
                    "category_name": s.category.category_name,
                    "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for s in synonyms]
            })

        elif request.method == "POST":
            data = request.get_json() or {}
            synonym = data.get("synonym")
            category_id = data.get("category_id")
            
            if not synonym or not category_id:
                return jsonify({"status": "error", "message": "Synonym and category_id required"}), 400
            
            # Verify category exists
            category = db.session.get(Category, category_id)
            if not category:
                return jsonify({"status": "error", "message": "Category not found"}), 404
            
            synonym_id = add_category_synonym(synonym, category_id)
            return jsonify({
                "status": "success",
                "message": f"Synonym '{synonym}' added to category '{category.category_name}'",
                "synonym_id": synonym_id
            })

        elif request.method == "DELETE":
            synonym_id = (request.get_json() or {}).get("synonym_id")
            if not synonym_id:
                return jsonify({"status": "error", "message": "Synonym ID required"}), 400
            
            syn = db.session.get(CategorySynonym, synonym_id)
            if not syn:
                return jsonify({"status": "error", "message": "Synonym not found"}), 404
            
            db.session.delete(syn)
            db.session.commit()
            return jsonify({"status": "success", "message": "Synonym deleted"})

    except Exception as e:
        print(f"Category Synonyms API error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

