import os
import json
import re
import traceback
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from sqlalchemy import text
from extensions import db
from config import APP_ROOT
from models.catalog import Plant, Variety
from models.inventory import InventoryTransaction, Supplier
from auth.decorators import require_tenant_auth, get_current_tenant
from auth.audit import log_audit_action
from services.image_utils import allowed_file, get_plant_image_url, search_plant_image
from services.llm_service import llm_call

admin_inventory_bp = Blueprint('admin_inventory', __name__)

@admin_inventory_bp.route('/api/admin/suggest-varieties', methods=['POST'])
def suggest_varieties():
    """Suggest plant varieties based on plant name"""
    try:
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/add-varieties', methods=['POST'])
def add_varieties():
    """Add selected varieties to a plant"""
    try:
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/varieties/existing/<int:plant_id>', methods=['GET'])
def get_existing_varieties_admin(plant_id):
    """Get existing varieties for a plant"""
    try:
        rows = db.session.execute(text("SELECT variety_name FROM variety WHERE plant_id = :pid"), {"pid": plant_id}).fetchall()
        varieties = [{"variety_name": r[0]} for r in rows]
        return jsonify({"status": "success", "varieties": varieties})
    except Exception as e:
        print(f"Error in get_existing_varieties_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@admin_inventory_bp.route('/api/admin/inventory', methods=['GET'])
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

@admin_inventory_bp.route('/api/admin/restock/<int:plant_id>', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def restock_plant(plant_id):
    """Restock a plant by adding quantity"""
    try:
        biz_id = get_current_tenant()
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/get-plant/<int:plant_id>', methods=['GET'])
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

@admin_inventory_bp.route('/api/admin/add-plant', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def add_plant():
    """Create a new plant with initial stock and categories for current nursery"""
    try:
        biz_id = get_current_tenant()
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/update-plant/<int:plant_id>', methods=['PUT'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def update_plant_details(plant_id):
    """Update plant details with strict tenant isolation"""
    try:
        biz_id = get_current_tenant()
        data = request.get_json() or {}
        
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

@admin_inventory_bp.route('/api/admin/delete-plant/<int:plant_id>', methods=['DELETE'])
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

@admin_inventory_bp.route('/api/admin/search-plants', methods=['GET'])
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

@admin_inventory_bp.route('/api/admin/update-plant-image/<int:plant_id>', methods=['POST'])
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

@admin_inventory_bp.route('/api/admin/inventory/add', methods=['POST'])
def inventory_add():
    """Add stock via supplier or return"""
    try:
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/inventory/remove', methods=['POST'])
def inventory_remove():
    """Remove stock (wastage/return to supplier)"""
    try:
        data = request.get_json() or {}
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

@admin_inventory_bp.route('/api/admin/inventory/transactions', methods=['GET'])
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

@admin_inventory_bp.route('/api/plant/<int:plant_id>/image', methods=['GET', 'POST', 'DELETE'])
def manage_plant_image(plant_id):
    """Upload, retrieve, or delete plant image"""
    try:
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        if request.method == "GET":
            # Return image path if present
            if plant.image_path:
                return jsonify({
                    "status": "success",
                    "image_url": f"/static/{plant.image_path.lstrip('/')}",
                    "source": "filesystem"
                })
            else:
                return jsonify({"status": "error", "message": "No image found"}), 404

        elif request.method == "POST":
            # Upload image
            if 'image' not in request.files:
                return jsonify({"status": "error", "message": "No image file provided"}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No selected file"}), 400

            if not allowed_file(file.filename):
                return jsonify({"status": "error", "message": "Only JPG, PNG, GIF allowed"}), 400

            # Read image data
            image_data = file.read()
            
            # Validate image size (max 5MB)
            if len(image_data) > 5 * 1024 * 1024:
                return jsonify({"status": "error", "message": "Image too large (max 5MB)"}), 400

            # Store in static/plantpics
            file_name = file.filename or ""
            filename = "".join(c for c in file_name if c.isalnum() or c in ('.', '_', '-')).strip()
            save_path = os.path.join('static', 'plantpics', filename)
            os.makedirs(os.path.join('static', 'plantpics'), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(image_data)
            plant.image_path = f"plantpics/{filename}"
            db.session.commit()

            return jsonify({
                "status": "success",
                "message": f"Image uploaded for {plant.plant_name}",
                "plant_id": plant_id,
                "size": len(image_data),
                "image_url": f"/static/{plant.image_path}"
            })

        elif request.method == "DELETE":
            # Delete image reference
            plant.image_path = None
            db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"})

    except Exception as e:
        print(f"Plant image API error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@admin_inventory_bp.route('/api/debug/list-plant-images', methods=['GET'])
def list_plant_images():
    import os
    static_dir = os.path.join(APP_ROOT, 'static')
    plants_dir = os.path.join(static_dir, 'images', 'plants')
    
    files = []
    if os.path.exists(plants_dir):
        files = os.listdir(plants_dir)
    
    return jsonify({
        "static_dir": static_dir,
        "plants_dir": plants_dir,
        "exists": os.path.exists(plants_dir),
        "file_count": len(files),
        "files_sample": files[:20],
        "root_path": current_app.root_path
    })

