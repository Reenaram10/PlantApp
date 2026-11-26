import os
import csv
import json
import io
import traceback
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g, current_app, send_file
from werkzeug.utils import secure_filename
from sqlalchemy import text, func

import server.config as config
from server.config import (
    ALLOWED_EXTENSIONS, PROFILE_DATASETS, APP_ROOT,
    LLM_FAILSAFE, STT_FAILSAFE, TTS_FAILSAFE
)
from server.database import db, utc_now
from server.models import (
    Plant, Variety, Supplier, InventoryTransaction, SupplierPurchase,
    PurchaseItem, Order, Category, PlantCategory, CategorySynonym,
    User, PlantKnowledge, DiseaseKnowledge, Business, AuditLog
)
from server.auth import require_tenant_auth, get_current_tenant, log_audit_action
from server.services.rag_service import faiss_rag
from server.services.llm_service import llm_call
from server.services.chat_service import get_plant_image_url, search_plant_image

admin_bp = Blueprint('admin', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@admin_bp.route('/admin.html')
@admin_bp.route('/admin')
def serve_admin_dashboard():
    return current_app.send_static_file('admin.html')



# ========== ADMIN API ENDPOINTS ==========
@admin_bp.route('/api/admin/suggest-varieties', methods=['POST'])
def suggest_varieties():
    """Suggest plant varieties based on plant name"""
    try:
        data = request.json
        plant_name = data.get('plant_name', '').strip()
        stock_count = data.get('stock_count', 0)
        price = data.get('price', 199.99)
        description = data.get('description', '')

        if not plant_name:
            return jsonify({"status": "error", "message": "Plant name required"}), 400

        #  Check if plant exists FIRST (case-insensitive)
        existing_plant = db.session.execute(
            text("SELECT plant_id, plant_name, stock, price FROM plant WHERE LOWER(plant_name) = LOWER(:name)"),
            {"name": plant_name}
        ).fetchone()

        if existing_plant:
            #  Plant already exists - just return it
            plant_id = existing_plant[0]
            existed = True
            current_stock = existing_plant[2]
            current_price = existing_plant[3]
            
            print(f" Plant '{plant_name}' already exists with ID: {plant_id}")

        else:
            #  Create new plant
            try:
                new_p = Plant(plant_name=plant_name, description=description, price=price, stock=stock_count)
                db.session.add(new_p)
                db.session.commit()
                plant_id = new_p.plant_id
                current_stock = new_p.stock
                current_price = new_p.price

                # Log initial stock if > 0
                if stock_count > 0:
                    db.session.execute(
                        text("""
                            INSERT INTO inventory_transaction (plant_id, type, quantity, notes, created_at)
                            VALUES (:pid, 'ADD', :qty, 'Initial stock from AI suggest', :now)
                        """),
                        {"pid": plant_id, "qty": stock_count, "now": datetime.now(timezone.utc)}
                    )
                    db.session.commit()
                existed = False
                print(f" Created new plant '{plant_name}' with ID: {plant_id}")

            except Exception as insert_error:
                db.session.rollback()
                print(f" Insert error (retrying): {insert_error}")
                
                # Retry: double-check if it exists now
                check_plant = db.session.execute(
                    text("SELECT plant_id, stock, price FROM plant WHERE LOWER(plant_name) = LOWER(:name)"),
                    {"name": plant_name}
                ).fetchone()
                
                if check_plant:
                    plant_id = check_plant[0]
                    existed = True
                    current_stock = check_plant[1]
                    current_price = check_plant[2]
                    print(f" Plant exists (concurrent request): ID {plant_id}")
                else:
                    raise insert_error

        # Suggest varieties using LLM
        prompt = f"""
For the plant "{plant_name}", suggest 5 specific varieties.
Return ONLY the variety names, one per line.
Example: Ruby Red, Golden Tiger, Variegated White, etc.
"""
        varieties_text = llm_call(prompt)
        varieties = [v.strip() for v in varieties_text.split('\n') if v.strip()]

        return jsonify({
            "status": "success",
            "plant_id": plant_id,
            "plant_name": plant_name,
            "existed": existed,
            "current_stock": current_stock,
            "price": current_price,
            "varieties": varieties[:5],
            "message": " Plant already available" if existed else " New plant created"
        })

    except Exception as e:
        print(f" Error in suggest_varieties: {e}")
        print(traceback.format_exc())
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


def reset_plant_sequence():
    """Reset the plant_id sequence to the max ID in the table"""
    try:
        # Get max ID
        max_id_row = db.session.execute(text("SELECT MAX(plant_id) FROM plant")).fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0
        
        if max_id:
            if db.engine.dialect.name == 'mssql':
                db.session.execute(text(f"DBCC CHECKIDENT ('plant', RESEED, {max_id})"))
                db.session.commit()
                print(f" Reseeded plant identity to {max_id}")
            else:
                next_val = max_id + 1
                db.session.execute(text(f"ALTER SEQUENCE plant_plant_id_seq RESTART WITH {next_val}"))
                db.session.commit()
                print(f" Reset plant sequence to start at {next_val} (max was {max_id})")
        else:
            print(" No plants found in database")
        
        return True
    except Exception as e:
        print(f" Sequence reset warning: {e}")
        # Don't fail completely - the retry logic will handle it
        return False


@admin_bp.route('/api/rebuild-index', methods=['POST'])
@admin_bp.route('/api/admin/rag/sync', methods=['POST'])
def sync_rag_index():
    """Force sync/rebuild FAISS RAG index from database and CSV files"""
    try:
        print(" [FAISS] Manual RAG index rebuild requested.")
        faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
        return jsonify({
            "status": "success", 
            "message": "FAISS vector index rebuilt successfully from database and CSV files."
        })
    except Exception as e:
        print(f" [FAISS] RAG rebuild error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



@admin_bp.route('/api/admin/add-varieties', methods=['POST'])
def add_varieties():
    """Add selected varieties to a plant"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        plant_name = data.get('plant_name')
        selected_varieties = data.get('selected_varieties', [])

        if not plant_id or not selected_varieties:
            return jsonify({"status": "error", "message": "Plant ID and varieties required"}), 400

        added = 0
        duplicates = 0

        for variety_name in selected_varieties:
            # Check if variety already exists
            existing = db.session.execute(
                text("""
                    SELECT variety_id FROM variety
                    WHERE plant_id = :pid AND LOWER(variety_name) = :vname
                """),
                {"pid": plant_id, "vname": variety_name.lower()}
            ).fetchone()

            if existing:
                duplicates += 1
                continue

            # Add new variety
            db.session.execute(
                text("""
                    INSERT INTO variety (plant_id, variety_name, variety_price)
                    VALUES (:pid, :vname, :price)
                """),
                {"pid": plant_id, "vname": variety_name, "price": 199.99}
            )
            added += 1

        db.session.commit()

        return jsonify({
            "status": "success",
            "message": f"Added {added} varieties",
            "added": added,
            "duplicates": duplicates
        })

    except Exception as e:
        print(f"Error in add_varieties: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/inventory', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_inventory():
    """Get all plants with inventory info for current nursery"""
    try:
        biz_id = get_current_tenant()
        where_clause = "WHERE p.business_id = :biz_id"
        if biz_id == 'NURS-0001':
            where_clause = "WHERE p.business_id = :biz_id OR p.business_id IS NULL OR p.business_id = ''"

        plants = db.session.execute(text(f"""
            SELECT p.plant_id, p.plant_name, p.description, p.price, p.stock, p.image_path,
                   STRING_AGG(c.category_name, ', ') as categories
            FROM plant p
            LEFT JOIN plant_category pc ON p.plant_id = pc.plant_id
            LEFT JOIN category c ON pc.category_id = c.category_id
            {where_clause}
            GROUP BY p.plant_id, p.plant_name, p.description, p.price, p.stock, p.image_path
            ORDER BY p.plant_name ASC
        """), {"biz_id": biz_id}).fetchall()

        plant_list = []
        total_stock = 0

        for p in plants:
            image_path = p[5]
            image_url = get_plant_image_url(image_path)
            
            plant_list.append({
                "plant_id": p[0],
                "plant_name": p[1],
                "description": p[2],
                "price": float(p[3]) if p[3] else 0,
                "stock": p[4] or 0,
                "image_path": image_path,
                "image_url": image_url,
                "categories": p[6]
            })
            total_stock += (p[4] or 0)

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "total_plants": len(plant_list),
            "total_stock": total_stock,
            "business_id": biz_id
        })

    except Exception as e:
        print(f"Error in get_inventory: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/restock/<int:plant_id>', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def restock_plant(plant_id):
    """Restock a plant by adding quantity"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        quantity = data.get('quantity', 0)

        if quantity <= 0:
            return jsonify({"status": "error", "message": "Quantity must be positive"}), 400

        plant = Plant.query.filter((Plant.plant_id == plant_id) & ((Plant.business_id == biz_id) | (Plant.business_id == 'NURS-0001'))).first()
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        old_stock = plant.stock or 0
        plant.stock = old_stock + quantity

        # Log transaction
        tx = InventoryTransaction(
            business_id=biz_id,
            plant_id=plant_id,
            type='ADD',
            quantity=quantity,
            notes="Manual restock via API"
        )
        db.session.add(tx)
        db.session.commit()

        log_audit_action("RESTOCK_PLANT", "INVENTORY", record_id=plant_id, details=f"Restocked {plant.plant_name} by +{quantity}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Restocked {plant.plant_name}",
            "plant_id": plant_id,
            "plant_name": plant.plant_name,
            "old_stock": old_stock,
            "new_stock": plant.stock,
            "quantity_added": quantity
        })

    except Exception as e:
        print(f"Error in restock_plant: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/get-plant/<int:plant_id>', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_plant_details(plant_id):
    """Get detailed info about a plant"""
    try:
        biz_id = get_current_tenant()
        plant = Plant.query.filter((Plant.plant_id == plant_id) & ((Plant.business_id == biz_id) | (Plant.business_id == 'NURS-0001'))).first()
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        return jsonify({
            "status": "success",
            "plant": {
                "plant_id": plant.plant_id,
                "plant_name": plant.plant_name,
                "description": plant.description,
                "price": float(plant.price) if plant.price else 0,
                "stock": plant.stock or 0,
                "image_path": plant.image_path,
                "business_id": plant.business_id
            }
        })

    except Exception as e:
        print(f"Error in get_plant_details: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/add-plant', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def add_plant():
    """Create a new plant with initial stock and categories for current nursery"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        name = data.get('plant_name')
        price = float(data.get('price', 0))
        stock = int(data.get('stock', 0))
        description = data.get('description', '')
        category_ids = data.get('categories', [])

        if not name:
            return jsonify({"status": "error", "message": "Plant name is required"}), 400

        # Insert plant using SQLAlchemy model (preserves identity handling across all dialects)
        new_plant = Plant(
            business_id=biz_id,
            plant_name=name,
            price=price,
            stock=stock,
            description=description
        )
        db.session.add(new_plant)
        db.session.flush()
        next_pid = new_plant.plant_id

        # Add categories
        for cat_id in category_ids:
            try:
                db.session.execute(
                    text("INSERT INTO plant_category (plant_id, category_id) VALUES (:pid, :cid)"),
                    {"pid": next_pid, "cid": cat_id}
                )
            except Exception as ce:
                print(f"Notice adding category: {ce}")

        # Log initial stock transaction if > 0
        if stock > 0:
            try:
                db.session.execute(
                    text("""
                        INSERT INTO inventory_transaction (business_id, plant_id, type, quantity, notes, created_at)
                        VALUES (:biz, :pid, 'ADD', :qty, 'Initial stock on creation', CURRENT_TIMESTAMP)
                    """),
                    {
                        "biz": biz_id,
                        "pid": next_pid,
                        "qty": stock
                    }
                )
            except Exception as te:
                print(f"Notice adding inventory transaction: {te}")

        db.session.commit()
        log_audit_action("ADD_PLANT", "PLANTS", record_id=next_pid, details=f"Added plant '{name}'", business_id=biz_id, user_id=getattr(g, 'user_id', None), user_email=getattr(g, 'user_email', None))

        return jsonify({
            "status": "success",
            "message": f"Plant '{name}' added successfully",
            "plant_id": next_pid,
            "business_id": biz_id
        })

    except Exception as e:
        print(f"Error in add_plant: {e}")
        traceback.print_exc()
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/update-plant/<int:plant_id>', methods=['PUT'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def update_plant_details(plant_id):
    """Update plant details with strict tenant isolation"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        
        # Enforce tenant data isolation: Local Admin can only modify plants owned by their organization
        plant = Plant.query.filter_by(plant_id=plant_id, business_id=biz_id).first()
        if not plant and biz_id == 'NURS-0001':
            plant = db.session.get(Plant, plant_id)
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found or unauthorized access"}), 404

        if 'plant_name' in data:
            plant.plant_name = data['plant_name']
        if 'description' in data:
            plant.description = data['description']
        if 'price' in data:
            plant.price = float(data['price'])
        if 'stock' in data:
            old_stock = plant.stock or 0
            new_stock = int(data['stock'])
            if old_stock != new_stock:
                diff = new_stock - old_stock
                tx = InventoryTransaction(
                    business_id=biz_id,
                    plant_id=plant_id,
                    type='ADD' if diff > 0 else 'REMOVE',
                    quantity=abs(diff),
                    notes=f"Stock Correction from {old_stock} to {new_stock}"
                )
                db.session.add(tx)
                plant.stock = new_stock

        db.session.commit()
        log_audit_action("UPDATE_PLANT", "PLANTS", record_id=plant_id, details=f"Updated plant {plant.plant_name}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Updated {plant.plant_name}",
            "plant_id": plant_id
        })

    except Exception as e:
        print(f"Error in update_plant_details: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/delete-plant/<int:plant_id>', methods=['DELETE'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def delete_plant_endpoint(plant_id):
    """Delete a plant with strict tenant isolation"""
    try:
        biz_id = get_current_tenant()
        plant = Plant.query.filter_by(plant_id=plant_id, business_id=biz_id).first()
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found or unauthorized access"}), 404

        plant_name = plant.plant_name

        # Delete related child records for this plant
        db.session.execute(text("DELETE FROM plant_category WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM variety WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM cart_new WHERE plant_id = :id"), {"id": plant_id})

        # Delete the plant
        db.session.delete(plant)
        db.session.commit()

        log_audit_action("DELETE_PLANT", "PLANTS", record_id=plant_id, details=f"Deleted plant {plant_name}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Deleted {plant_name}",
            "plant_id": plant_id
        })

    except Exception as e:
        print(f"Error in delete_plant_endpoint: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/search-plants', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def search_plants_admin():
    """Search for plants by name within the current tenant organization"""
    try:
        biz_id = get_current_tenant()
        query = request.args.get('q', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({"status": "error", "message": "Query too short"}), 400

        plants = db.session.execute(text("""
            SELECT TOP 20 plant_id, plant_name, description, price, stock, image_path
            FROM plant
            WHERE business_id = :biz AND LOWER(plant_name) LIKE :q
            ORDER BY plant_name ASC
        """), {"biz": biz_id, "q": f"%{query.lower()}%"}).fetchall()

        plant_list = [{
            "plant_id": p[0],
            "plant_name": p[1],
            "description": p[2],
            "price": float(p[3]) if p[3] else 0,
            "stock": p[4] or 0,
            "image_url": get_plant_image_url(p[5])
        } for p in plants]

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "count": len(plant_list)
        })

    except Exception as e:
        print(f"Error in search_plants_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/suppliers', methods=['GET'])
def get_suppliers():
    """Get all suppliers"""
    try:
        rows = db.session.execute(text("SELECT supplier_id, supplier_name, phone, email, address FROM supplier ORDER BY supplier_name ASC")).fetchall()
        suppliers = [{
            "supplier_id": r[0],
            "name": r[1],
            "contact_number": r[2],
            "email": r[3],
            "address": r[4]
        } for r in rows]
        return jsonify({"status": "success", "suppliers": suppliers})
    except Exception as e:
        print(f"Error in get_suppliers: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/suppliers', methods=['POST'])
def add_supplier():
    """Add a new supplier"""
    try:
        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({"status": "error", "message": "Supplier name required"}), 400
        
        db.session.execute(text("""
            INSERT INTO supplier (supplier_name, phone, email, address)
            VALUES (:name, :phone, :email, :address)
        """), {
            "name": name,
            "phone": data.get('contact_number'),
            "email": data.get('email'),
            "address": data.get('address')
        })
        db.session.commit()
        return jsonify({"status": "success", "message": "Supplier added successfully"})
    except Exception as e:
        print(f"Error in add_supplier: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/categories', methods=['GET'])
def get_categories_admin():
    """Get all categories for admin"""
    try:
        rows = db.session.execute(text("SELECT category_id, category_name FROM category ORDER BY category_name ASC")).fetchall()
        categories = [{"category_id": r[0], "category_name": r[1]} for r in rows]
        return jsonify({"status": "success", "categories": categories})
    except Exception as e:
        print(f"Error in get_categories_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/categories', methods=['POST'])
def add_category_admin():
    """Add a new category"""
    try:
        data = request.json
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


@admin_bp.route('/api/admin/categories/<int:cat_id>', methods=['DELETE'])
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


@admin_bp.route('/api/admin/categories/<int:cat_id>/plants', methods=['GET'])
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


@admin_bp.route('/admin')
@admin_bp.route('/admin.html')
def admin_page():
    from flask import send_from_directory
    return send_from_directory('static', 'admin.html')



# --- INVENTORY MANAGEMENT ROUTES ---
@admin_bp.route('/api/admin/inventory/add', methods=['POST'])
def inventory_add():
    """Add stock via supplier or return"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        quantity = int(data.get('quantity', 0))
        
        if not plant_id or quantity <= 0:
            return jsonify({"status": "error", "message": "Invalid plant or quantity"}), 400
            
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        supplier_id = data.get('supplier_id')
        if supplier_id == "": supplier_id = None
        
        # Create transaction
        tx = InventoryTransaction(
            plant_id=plant_id,
            supplier_id=supplier_id,
            type='ADD',
            quantity=quantity,
            notes=data.get('notes'),
            bill_no=data.get('bill_no'),
            bill_date=data.get('bill_date')
        )
        
        # Update plant stock
        plant.stock = (plant.stock or 0) + quantity
        
        db.session.add(tx)
        db.session.commit()
        
        return jsonify({"status": "success", "message": f"Added {quantity} units to {plant.plant_name}"})
    except Exception as e:
        print(f"Error in inventory_add: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/inventory/remove', methods=['POST'])
def inventory_remove():
    """Remove stock (wastage/return to supplier)"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        quantity = int(data.get('quantity', 0))
        
        if not plant_id or quantity <= 0:
            return jsonify({"status": "error", "message": "Invalid plant or quantity"}), 400
            
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        # Create transaction
        tx = InventoryTransaction(
            plant_id=plant_id,
            type='REMOVE',
            quantity=quantity,
            notes=data.get('notes'),
            bill_no=data.get('bill_no'),
            bill_date=data.get('bill_date')
        )
        
        # Update plant stock
        plant.stock = max(0, (plant.stock or 0) - quantity)
        
        db.session.add(tx)
        db.session.commit()
        
        return jsonify({"status": "success", "message": f"Removed {quantity} units from {plant.plant_name}"})
    except Exception as e:
        print(f"Error in inventory_remove: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/inventory/transactions', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_inventory_transactions():
    """Get recent inventory transactions for current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT TOP 100 it.created_at, p.plant_name, it.type, it.quantity, s.supplier_name as supplier_name, it.bill_no, it.bill_date, it.notes
            FROM inventory_transaction it
            JOIN plant p ON p.plant_id = it.plant_id
            LEFT JOIN supplier s ON s.supplier_id = it.supplier_id
            WHERE it.business_id = :biz_id OR it.business_id IS NULL OR it.business_id = ''
            ORDER BY it.created_at DESC
        """), {"biz_id": biz_id}).fetchall()
        
        transactions = []
        for r in rows:
            transactions.append({
                "date": r[0].strftime("%B %d, %Y %I:%M %p") if r[0] else "N/A",
                "plant_name": r[1],
                "type": r[2],
                "quantity": r[3],
                "supplier_name": r[4] or "N/A",
                "bill_no": r[5],
                "bill_date": r[6],
                "notes": r[7]
            })
        
        return jsonify({"status": "success", "transactions": transactions})
    except Exception as e:
        print(f"Error in get_inventory_transactions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/customers', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_customers_admin():
    """Get all customers for admin within current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT u.id, u.email, u.last_login, u.is_online,
            (SELECT COUNT(*) FROM orders WHERE user_id = u.id AND (business_id = :biz_id OR business_id IS NULL OR business_id = '')) as purchases,
            (SELECT SUM(COALESCE(o.price, p.price, 0)) FROM orders o LEFT JOIN plant p ON o.plant_name = p.plant_name WHERE o.user_id = u.id AND (o.business_id = :biz_id OR o.business_id IS NULL OR o.business_id = '')) as amount
            FROM login u
            WHERE u.business_id = :biz_id OR u.business_id IS NULL OR u.business_id = ''
            ORDER BY u.id DESC
        """), {"biz_id": biz_id}).fetchall()
        
        customers = []
        for r in rows:
            customers.append({
                "id": r[0],
                "email": r[1],
                "last_login": r[2].strftime("%Y-%m-%d %H:%M") if r[2] else None,
                "is_online": bool(r[3]),
                "purchases": r[4],
                "amount_purchased": float(r[5]) if r[5] else 0
            })
        return jsonify({"status": "success", "customers": customers})
    except Exception as e:
        print(f"Error in get_customers_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/orders', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_all_orders_admin():
    """Get all customer orders for admin within current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT 
                COALESCE(o.order_group_id, CONCAT('OLD-', CAST(o.order_id AS VARCHAR(20)))) as gid, 
                u.email, 
                o.order_status, 
                o.order_date, 
                SUM(COALESCE(o.price, p.price, 0)) as total_price, 
                COALESCE(o.payment_method, 'Not Specified') as pm,
                SUM(COALESCE(o.amount_paid, 0)) as amount_paid,
                SUM(COALESCE(o.balance, 0)) as balance
            FROM orders o
            LEFT JOIN login u ON o.user_id = u.id
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE o.business_id = :biz_id OR o.business_id IS NULL OR o.business_id = ''
            GROUP BY COALESCE(o.order_group_id, CONCAT('OLD-', CAST(o.order_id AS VARCHAR(20)))), u.email, o.order_status, o.order_date, COALESCE(o.payment_method, 'Not Specified')
            ORDER BY o.order_date DESC
        """), {"biz_id": biz_id}).fetchall()
        
        orders = []
        for r in rows:
            orders.append({
                "order_group_id": r[0],
                "customer_name": r[1] or "Guest",
                "order_status": r[2],
                "order_date": r[3].strftime("%Y-%m-%d %H:%M") if r[3] else "N/A",
                "total_amount": float(r[4]) if r[4] else 0,
                "payment_method": r[5],
                "amount_paid": float(r[6]) if r[6] else 0,
                "balance": float(r[7]) if r[7] else 0
            })
        return jsonify({"status": "success", "orders": orders})
    except Exception as e:
        print(f"Error in get_all_orders_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/orders/record-payment', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def record_order_payment():
    """Record partial or full payment for an order group within current nursery"""
    biz_id = get_current_tenant()
    data = request.json or {}
    order_group_id = data.get("order_group_id")
    amount = data.get("amount")
    
    if not order_group_id or amount is None:
        return jsonify({"status": "error", "message": "Missing group ID or amount"}), 400
        
    try:
        if order_group_id.startswith('OLD-'):
            actual_id = order_group_id.replace('OLD-', '')
            where_clause = "order_id = :gid"
            gid_val = int(actual_id) if actual_id.isdigit() else 0
        else:
            where_clause = "order_group_id = :gid"
            gid_val = order_group_id
            
        rows = db.session.execute(text(f"SELECT order_id, balance FROM orders WHERE {where_clause} AND (business_id = :biz_id OR business_id IS NULL OR business_id = '') AND balance > 0"), {"gid": gid_val, "biz_id": biz_id}).fetchall()
        
        remaining_amount = float(amount)
        for r in rows:
            if remaining_amount <= 0:
                break
            oid = r[0]
            bal = float(r[1])
            pay_this = min(bal, remaining_amount)
            db.session.execute(text("UPDATE orders SET amount_paid = COALESCE(amount_paid, 0) + :p, balance = balance - :p WHERE order_id = :oid"), {"p": pay_this, "oid": oid})
            remaining_amount -= pay_this
            
        db.session.commit()
        log_audit_action("RECORD_ORDER_PAYMENT", "ORDERS", record_id=order_group_id, details=f"Recorded payment of {amount} for order {order_group_id}", business_id=biz_id)
        return jsonify({"status": "success", "message": "Sales payment recorded successfully"})
    except Exception as e:
        print(f"Error in record_order_payment: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/orders/<order_group_id>', methods=['GET'])
def get_order_details_admin(order_group_id):
    """Get details for a specific order group"""
    try:
        # Handle fallback IDs from get_all_orders_admin
        if order_group_id.startswith('OLD-'):
            actual_id = order_group_id.replace('OLD-', '')
            where_clause = "o.order_id = :gid"
            gid_val = int(actual_id) if actual_id.isdigit() else 0
        else:
            where_clause = "o.order_group_id = :gid"
            gid_val = order_group_id

        # Get orders in this group
        rows = db.session.execute(text(f"""
            SELECT o.order_id, o.plant_name, COALESCE(o.price, p.price, 0), o.quantity, o.order_status
            FROM orders o
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE {where_clause}
        """), {"gid": gid_val}).fetchall()
        
        if not rows:
            return jsonify({"status": "error", "message": "Order not found"}), 404
            
        items = []
        total_amount = 0
        status = rows[0][4]
        
        for r in rows:
            qty = r[3] or 1
            price = float(r[2] or 0)
            items.append({
                "plant_name": r[1],
                "quantity": qty,
                "total_amount": price * qty
            })
            total_amount += price * qty
            
        # Get customer info from first order record
        cust_info = db.session.execute(text(f"""
            SELECT TOP 1 u.email, o.payment_method, o.shipping_address, o.order_date
            FROM orders o
            LEFT JOIN login u ON o.user_id = u.id
            WHERE {where_clause}
        """), {"gid": gid_val}).fetchone()
        
        # Safety check if cust_info is None (though rows was found)
        if not cust_info:
            return jsonify({"status": "error", "message": "Customer info not found"}), 404

        summary = {
            "order_group_id": order_group_id,
            "customer_name": cust_info[0] or "Guest",
            "customer_email": cust_info[0] or "N/A",
            "payment_method": cust_info[1] or "Not Specified",
            "shipping_address": cust_info[2] or "Not Provided",
            "order_date": cust_info[3].strftime("%Y-%m-%d %H:%M") if (cust_info[3] and hasattr(cust_info[3], 'strftime')) else "N/A",
            "order_status": status or "Processing",
            "total_amount": total_amount,
            "tracking_number": "N/A",
            "delivery_date": "N/A",
            "notes": ""
        }
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "items": items
        })
    except Exception as e:
        print(f"Error in get_order_details_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/orders/<order_group_id>', methods=['PUT'])
def update_order_status_admin(order_group_id):
    """Update status of all orders in a group"""
    status = request.json.get("status")
    if not status:
        return jsonify({"status": "error", "message": "Status is required"}), 400
        
    try:
        db.session.execute(text("UPDATE orders SET order_status = :s WHERE order_group_id = :gid"), {"s": status, "gid": order_group_id})
        db.session.commit()
        return jsonify({"status": "success", "message": "Order status updated"})
    except Exception as e:
        print(f"Error in update_order_status_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/purchases', methods=['POST'])
def add_supplier_purchase():
    """Add a new purchase from a supplier"""
    data = request.json
    supplier_id = data.get("supplier_id")
    bill_no = data.get("bill_no")
    bill_date_str = data.get("bill_date")
    items = data.get("items", []) # List of {plant_id, quantity, unit_price}
    
    if not supplier_id or not items:
        return jsonify({"status": "error", "message": "Supplier and items are required"}), 400
        
    try:
        total_amount = sum(item['quantity'] * item['unit_price'] for item in items)
        bill_date = datetime.strptime(bill_date_str, '%Y-%m-%d') if bill_date_str else utc_now()
        
        purchase = SupplierPurchase(
            supplier_id=supplier_id,
            bill_no=bill_no or f"PUR-{random.randint(1000, 9999)}",
            bill_date=bill_date,
            total_amount=total_amount,
            balance=total_amount,
            status='Pending'
        )
        db.session.add(purchase)
        db.session.flush() # Get purchase_id
        
        for itm in items:
            p_id = itm['plant_id']
            qty = itm['quantity']
            price = itm['unit_price']
            
            # 1. Create Purchase Item
            pi = PurchaseItem(
                purchase_id=purchase.purchase_id,
                plant_id=p_id,
                quantity=qty,
                unit_price=price
            )
            db.session.add(pi)
            
            # 2. Update Plant Stock
            plant = db.session.get(Plant, p_id)
            if plant:
                plant.stock += qty
                
            # 3. Log Inventory Transaction
            tx = InventoryTransaction(
                plant_id=p_id,
                supplier_id=supplier_id,
                type='ADD',
                quantity=qty,
                notes=f'Supplier Purchase: {purchase.bill_no}',
                bill_no=purchase.bill_no,
                bill_date=bill_date.strftime('%Y-%m-%d')
            )
            db.session.add(tx)
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Purchase recorded successfully"})
    except Exception as e:
        db.session.rollback()
        print(f"Error in add_supplier_purchase: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/purchases', methods=['GET'])
def get_supplier_purchases():
    """List all supplier purchases"""
    try:
        sql = """
            SELECT p.purchase_id, s.supplier_name, p.bill_no, p.bill_date, 
                   p.total_amount, p.amount_paid, p.balance, p.status
            FROM supplier_purchase p
            JOIN supplier s ON p.supplier_id = s.supplier_id
            ORDER BY p.created_at DESC
        """
        rows = db.session.execute(text(sql)).fetchall()
            
        result = []
        for r in rows:
            result.append({
                "purchase_id": r[0],
                "supplier_name": r[1],
                "bill_no": r[2],
                "bill_date": r[3].strftime('%Y-%m-%d') if hasattr(r[3], 'strftime') else str(r[3]),
                "total_amount": float(r[4]),
                "amount_paid": float(r[5]),
                "balance": float(r[6]),
                "status": r[7]
            })
        return jsonify({"status": "success", "purchases": result})
    except Exception as e:
        print(f"Error in get_supplier_purchases: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/purchases/record-payment', methods=['POST'])
def record_purchase_payment():
    """Record a payment against a purchase balance"""
    data = request.json
    purchase_id = data.get("purchase_id")
    amount = float(data.get("amount", 0))
    
    if not purchase_id or amount <= 0:
        return jsonify({"status": "error", "message": "Valid Purchase ID and amount are required"}), 400
        
    try:
        purchase = db.session.get(SupplierPurchase, purchase_id)
        if not purchase:
            return jsonify({"status": "error", "message": "Purchase not found"}), 404
            
        if amount > purchase.balance:
            return jsonify({"status": "error", "message": f"Payment exceeds balance (₹{purchase.balance})"}), 400
            
        purchase.amount_paid += amount
        purchase.balance = purchase.total_amount - purchase.amount_paid
        
        if purchase.balance <= 0:
            purchase.status = 'Paid'
        elif purchase.amount_paid > 0:
            purchase.status = 'Partial'
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Payment recorded", "new_balance": purchase.balance})
    except Exception as e:
        print(f"Error in record_purchase_payment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/varieties/existing/<int:plant_id>', methods=['GET'])
def get_existing_varieties_admin(plant_id):
    """Get existing varieties for a plant"""
    try:
        rows = db.session.execute(text("SELECT variety_name FROM variety WHERE plant_id = :pid"), {"pid": plant_id}).fetchall()
        varieties = [{"variety_name": r[0]} for r in rows]
        return jsonify({"status": "success", "varieties": varieties})
    except Exception as e:
        print(f"Error in get_existing_varieties_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/update-plant-image/<int:plant_id>', methods=['POST'])
def update_plant_image_admin(plant_id):
    """Update plant image (URL or File)"""
    try:
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        image_url = request.form.get('image_url')
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                # Basic filename cleaning
                filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).strip()
                save_path = os.path.join('static', 'plantpics', filename)
                os.makedirs(os.path.join('static', 'plantpics'), exist_ok=True)
                file.save(save_path)
                plant.image_path = f"plantpics/{filename}"
        elif image_url:
            plant.image_path = image_url # Assuming direct URL or handling path
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Image updated successfully"})
    except Exception as e:
        print(f"Error in update_plant_image_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/category_synonyms', methods=['GET', 'POST', 'DELETE'])
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
            data = request.json
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
            synonym_id = request.json.get("synonym_id")
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


@admin_bp.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration options"""
    # updated via config
    
    try:
        if request.method == "GET":
            return jsonify({
                "status": "success",
                "config": {
                    "LLM_OPTION": config.LLM_OPTION,
                    "STT_OPTION": config.STT_OPTION,
                    "TTS_OPTION": config.TTS_OPTION,
                    "LLM_FAILSAFE": LLM_FAILSAFE,
                    "STT_FAILSAFE": STT_FAILSAFE,
                    "TTS_FAILSAFE": TTS_FAILSAFE
                }
            })

        elif request.method == "POST":
            data = request.json
            
            if "LLM_OPTION" in data:
                config.LLM_OPTION = data["LLM_OPTION"]
                print(f" LLM changed to: {LLM_OPTION}")
            
            if "STT_OPTION" in data:
                config.STT_OPTION = data["STT_OPTION"]
                print(f" STT changed to: {STT_OPTION}")
            
            if "TTS_OPTION" in data:
                config.TTS_OPTION = data["TTS_OPTION"]
                print(f" TTS changed to: {TTS_OPTION}")
            
            return jsonify({
                "status": "success",
                "message": "Configuration updated",
                "config": {
                    "LLM_OPTION": config.LLM_OPTION,
                    "STT_OPTION": config.STT_OPTION,
                    "TTS_OPTION": config.TTS_OPTION
                }
            })

    except Exception as e:
        print(f"Config API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/profile-stats', methods=['GET'])
def get_profile_stats():
    """
    Returns record counts, columns, file details, and preview rows for dataset profiles.
    """
    try:
        profile_type = request.args.get('profile_type', 'all')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results = {}

        targets = [profile_type] if profile_type in PROFILE_DATASETS else list(PROFILE_DATASETS.keys())

        import pandas as pd
        for p_key in targets:
            info = PROFILE_DATASETS[p_key]
            file_path = os.path.join(base_dir, info["file"])
            
            if not os.path.exists(file_path):
                results[p_key] = {
                    "key": p_key,
                    "title": info["title"],
                    "exists": False,
                    "row_count": 0,
                    "columns": info["columns"],
                    "preview": [],
                    "last_modified": "N/A"
                }
                continue

            # Load CSV safely with fallback encodings
            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df is None:
                df = pd.DataFrame()

            # Clean dataframe for JSON response (replace NaN with empty string)
            df_clean = df.fillna("")
            preview_rows = df_clean.head(10).to_dict(orient="records")
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

            results[p_key] = {
                "key": p_key,
                "title": info["title"],
                "exists": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "required_cols": info["required_cols"],
                "preview": preview_rows,
                "last_modified": mod_time,
                "filename": info["file"]
            }

        return jsonify({"status": "success", "datasets": results})

    except Exception as e:
        print(f" [Profile Stats Error]: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@admin_bp.route('/api/admin/upload-profiles', methods=['POST'])
def upload_profile_dataset():
    """
    Accepts Excel (.xlsx / .xls) or CSV file uploads to update Geography, Plant, or Disease dataset profiles.
    Supports 'replace' or 'append' mode. Rebuilds FAISS RAG index automatically.
    """
    import io
    import pandas as pd
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded in form payload ('file' key expected)."}), 400

        file_obj = request.files['file']
        profile_type = request.form.get('profile_type', 'plant_profile')
        mode = request.form.get('mode', 'replace').lower() # 'replace' or 'append'

        if profile_type not in PROFILE_DATASETS:
            return jsonify({
                "status": "error",
                "message": f"Invalid profile_type. Must be one of: {list(PROFILE_DATASETS.keys())}"
            }), 400

        meta = PROFILE_DATASETS[profile_type]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_file_path = os.path.join(base_dir, str(meta["file"]))

        filename = file_obj.filename.lower()
        content = file_obj.read()

        if not content:
            return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

        # Parse file based on extension
        new_df = None
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            new_df = pd.read_excel(io.BytesIO(content))
        else:
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    new_df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    break
                except Exception:
                    continue

        if new_df is None or new_df.empty:
            return jsonify({"status": "error", "message": "Could not parse spreadsheet data or sheet is empty."}), 400

        # Clean column names (strip whitespace)
        new_df.columns = [str(c).strip() for c in new_df.columns]

        # Check required columns
        missing_cols = [col for col in meta["required_cols"] if col not in new_df.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Uploaded file is missing required columns for {meta['title']}: {missing_cols}. Found columns: {list(new_df.columns)}"
            }), 400

        # Process dataset depending on mode
        final_df = None
        if mode == 'append' and os.path.exists(target_file_path):
            existing_df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    existing_df = pd.read_csv(target_file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if existing_df is not None and not existing_df.empty:
                existing_df.columns = [str(c).strip() for c in existing_df.columns]
                # Combine dataframes
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop exact duplicate rows
                key_col = str(meta.get("key_col") or "")
                if key_col and key_col in combined.columns:
                    final_df = combined.drop_duplicates(subset=[key_col], keep="last")
                else:
                    final_df = combined.drop_duplicates(keep="last")
            else:
                final_df = new_df
        else:
            final_df = new_df

        # Fill NaNs with empty string
        final_df = final_df.fillna("")

        # Save back to target CSV file using latin1 or utf8
        try:
            final_df.to_csv(target_file_path, index=False, encoding='utf-8')
        except Exception:
            final_df.to_csv(target_file_path, index=False, encoding='latin1')

        # Trigger RAG FAISS re-indexing
        print(f" [Admin Upload] Syncing RAG vector database after {meta['title']} update...")
        rag_reindexed = False
        try:
            faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
            rag_reindexed = True
        except Exception as rag_err:
            print(f" [Admin Upload] FAISS RAG sync warning: {rag_err}")

        return jsonify({
            "status": "success",
            "message": f"Successfully updated {meta['title']} ({mode} mode). Total rows: {len(final_df)}.",
            "profile_type": profile_type,
            "mode": mode,
            "new_rows_uploaded": len(new_df),
            "total_rows": len(final_df),
            "rag_reindexed": rag_reindexed,
            "filename": meta["file"]
        })

    except Exception as e:
        print(f" [Upload Profile Error]: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error processing upload: {str(e)}"}), 500


@admin_bp.route('/api/admin/download-template/<profile_type>', methods=['GET'])
def download_profile_template(profile_type):
    """
    Generates and returns a downloadable CSV template with standard headers and sample data for admins.
    """
    try:
        if profile_type not in PROFILE_DATASETS:
            return jsonify({"status": "error", "message": "Invalid profile type."}), 400

        meta = PROFILE_DATASETS[profile_type]
        cols = list(meta["columns"]) if isinstance(meta["columns"], (list, tuple)) else []
        example = dict(meta["example"]) if isinstance(meta["example"], dict) else {}

        import pandas as pd
        template_df = pd.DataFrame([example], columns=cols)
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={profile_type}_template.csv"
        response.headers["Content-type"] = "text/csv; charset=utf-8"
        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
