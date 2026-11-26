import os
import time
import random
import re
from datetime import datetime, timezone
from contextlib import nullcontext
from flask import Blueprint, request, jsonify, g, current_app, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import text

from server.config import APP_ROOT, TRACKING_STAGES, ALLOWED_EXTENSIONS
from server.database import db, utc_now
from server.models import Plant, Cart, Order, User
from server.auth import get_current_tenant, verify_jwt
from server.services.payment_service import encrypt, decrypt
from server.services.chat_service import get_plant_image_url

store_bp = Blueprint('store', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def app_context_guard():
    from flask import has_app_context
    if has_app_context():
        return nullcontext()
    try:
        import app as _root_app
        return _root_app.app.app_context()
    except Exception:
        return nullcontext()

@store_bp.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('.', 'auth.html')


@store_bp.route('/<path:filename>')
def serve_root_files(filename):
    from flask import send_from_directory
    # If the file exists in root, serve it
    if os.path.exists(os.path.join(current_app.root_path, filename)):
        return send_from_directory('.', filename)
    # fall back to static
    return send_from_directory('static', filename)


@store_bp.route('/api/cart/update', methods=['POST'])
def update_cart():
    try:
        data = request.json
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



@store_bp.route('/api/plant/<int:plant_id>/image', methods=['GET', 'POST', 'DELETE'])
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
            filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).strip()
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



@store_bp.route('/plants', methods=['GET'])
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


@store_bp.route("/api/cart", methods=["POST"])
def add_to_cart():
    data = request.json
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
    


@store_bp.route("/api/cart", methods=["GET"])
def get_cart():
    user_id = str(request.args.get("user_id", "anonymous"))

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



@store_bp.route("/api/checkout", methods=["POST"])
def checkout():
    data = request.json or {}
    user_id = str(data.get("user_id", "anonymous"))

    # Customer Details
    customer_name = (data.get("customer_name") or data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    address = (data.get("shipping_address") or data.get("address") or "").strip()
    city = (data.get("city") or "").strip()
    state = (data.get("state") or "").strip()
    pincode = (data.get("pincode") or "").strip()
    payment_method = (data.get("payment_method") or "Cash on Delivery").strip()
    notes = (data.get("notes") or "").strip()

    # Legacy payment_info support
    payment_info = data.get("payment_info", {})
    if not address and payment_info.get("address"):
        address = payment_info.get("address")
    if not phone and payment_info.get("phone"):
        phone = payment_info.get("phone")

    if not customer_name:
        customer_name = "Valued Customer"
    if not address:
        return jsonify({"status": "error", "message": "Delivery address is required"}), 400

    address_parts = [address]
    if city: address_parts.append(city)
    if state: address_parts.append(state)
    if pincode: address_parts.append(f"PIN: {pincode}")
    full_address = ", ".join(address_parts)
    if phone:
        full_address += f" | Phone: {phone}"

    # Verify user cart
    rows = db.session.execute(text("""
        SELECT c.plant_id, c.quantity, p.plant_name, p.price, p.stock
        FROM cart_new c
        JOIN plant p ON p.plant_id = c.plant_id
        WHERE c.user_id = :u
    """), {"u": user_id}).fetchall()

    if not rows:
        return jsonify({"status": "error", "message": "Your cart is empty"}), 400

    # Stock validation
    for r in rows:
        stock = r[4] if r[4] is not None else 0
        qty = int(r[1] or 1)
        if stock < qty:
            return jsonify({
                "status": "error",
                "message": f"'{r[2]}' is out of stock or requested quantity ({qty}) exceeds available stock ({stock})."
            }), 400

    # Calculate totals
    subtotal = sum(float(r[3] or 0.0) * int(r[1] or 1) for r in rows)
    delivery_charge = 0.0 if subtotal >= 500.0 else 50.0
    total_amount = round(subtotal + delivery_charge, 2)

    # Link phone to user account if logged in
    if phone and user_id.isdigit():
        try:
            db.session.execute(text("UPDATE login SET phone = :ph WHERE id = :uid AND (phone IS NULL OR phone = '')"), {"ph": phone, "uid": int(user_id)})
            db.session.commit()
        except Exception:
            db.session.rollback()

    # Generate unique IDs
    order_timestamp = int(time.time())
    rand_code = random.randint(1000, 9999)
    order_group_id = f"ORD-{order_timestamp}-{rand_code}"
    tracking_number = f"GB-TRK-{order_timestamp}-{rand_code}"
    biz_id = get_current_tenant() or 'NURS-0001'

    order_items_summary = []
    is_cod = "cod" in payment_method.lower() or "cash" in payment_method.lower()
    pay_status = "Pending (COD)" if is_cod else "Pending (Online)"

    target_uid = None
    if user_id.isdigit():
        target_uid = int(user_id)
    else:
        user_rec = User.query.filter(User.username == user_id).first()
        if user_rec:
            target_uid = user_rec.id

    notes_parts = []
    if not user_id.isdigit():
        notes_parts.append(f"User: {user_id}")
    if customer_name:
        notes_parts.append(f"Customer: {customer_name}")
    if notes:
        notes_parts.append(notes)
    order_notes = " | ".join(notes_parts)

    for r in rows:
        plant_id = r[0]
        qty = int(r[1] or 1)
        p_name = r[2]
        p_price = float(r[3] or 0.0)
        item_total = round(p_price * qty, 2)

        # Deduct stock
        plant = db.session.get(Plant, plant_id)
        if plant and plant.stock is not None:
            plant.stock = max(0, plant.stock - qty)

        new_order = Order(
            business_id=biz_id,
            plant_name=p_name,
            user_id=target_uid,
            order_date=utc_now(),
            total_amount=item_total,
            payment_status=pay_status,
            order_status='Order Placed',
            order_group_id=order_group_id,
            price=p_price,
            quantity=qty,
            payment_method=payment_method,
            shipping_address=full_address,
            delivery_date="3-5 Business Days",
            tracking_number=tracking_number,
            notes=order_notes,
            amount_paid=0.0,
            balance=item_total
        )
        db.session.add(new_order)
        order_items_summary.append({
            "plant_name": p_name,
            "quantity": qty,
            "price": p_price,
            "item_total": item_total
        })

    # Clear user cart
    db.session.execute(text("DELETE FROM cart_new WHERE user_id = :u"), {"u": user_id})
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Order placed successfully!",
        "order_group_id": order_group_id,
        "tracking_number": tracking_number,
        "customer_name": customer_name,
        "phone": phone,
        "shipping_address": full_address,
        "payment_method": payment_method,
        "payment_status": pay_status,
        "order_status": "Order Placed",
        "items": order_items_summary,
        "subtotal": round(subtotal, 2),
        "delivery_charge": round(delivery_charge, 2),
        "total_amount": total_amount,
        "estimated_delivery": "3-5 Business Days"
    }), 201


@store_bp.route("/api/payment/response", methods=["POST"])
def payment_response():
    """Handle CCAvenue callback"""
    working_key = os.getenv("CCAVENUE_WORKING_KEY", "7D0D9D9D9D9D9D9D9D9D9D9D9D9D9D9D")
    enc_resp = request.form.get("encResp")
    
    if not enc_resp:
        return "Invalid response", 400
        
    try:
        dec_resp = decrypt(enc_resp, working_key)
        # Parse response string (it's in param1=val1&param2=val2 format)
        resp_dict = {p.split('=')[0]: p.split('=')[1] for p in dec_resp.split('&') if '=' in p}
        
        order_status = resp_dict.get("order_status")
        order_group_id = resp_dict.get("order_id")
        user_id = resp_dict.get("merchant_param1")
        total_order_amount = float(resp_dict.get("merchant_param2", 0))
        amount_paid = float(resp_dict.get("amount", 0))

        if order_status == "Success":
            # Finalize the order in DB
            with app_context_guard():
                # Retrieve cart again to reduce stock accurately
                rows = db.session.execute(text("SELECT plant_id, quantity FROM cart_new WHERE user_id = :u"), {"u": user_id}).fetchall()
                
                remaining_paid = amount_paid
                remaining_total = total_order_amount
                
                for i, r in enumerate(rows):
                    plant = db.session.get(Plant, r[0])
                    if plant:
                        plant.stock -= r[1]
                        item_total = plant.price * r[1]
                        
                        # Allocate paid amount proportionally or just dump on first items
                        # Here we use a safe approach: items get their share
                        if i == len(rows) - 1:
                            # Last item gets the rest to avoid rounding issues
                            item_paid = remaining_paid
                            item_balance = remaining_total - remaining_paid
                        else:
                            ratio = item_total / total_order_amount if total_order_amount > 0 else 0
                            item_paid = round(amount_paid * ratio, 2)
                            item_balance = item_total - item_paid
                            remaining_paid -= item_paid
                            remaining_total -= item_total

                        new_order = Order(
                            plant_name=plant.plant_name,
                            user_id=int(user_id) if str(user_id).isdigit() else None,
                            order_status='Processing',
                            order_group_id=order_group_id,
                            price=plant.price,
                            quantity=r[1],
                            total_amount=item_total,
                            payment_status='Paid' if item_balance <= 0 else 'Partial',
                            payment_method='CCAvenue',
                            shipping_address=resp_dict.get("billing_address", "Provided via PG"),
                            amount_paid=item_paid,
                            balance=item_balance
                        )
                        db.session.add(new_order)
                
                db.session.execute(text("DELETE FROM cart_new WHERE user_id=:u"), {"u": user_id})
                db.session.commit()
                
            return f"""
                <html>
                    <body onload="window.parent.postMessage('payment_success', '*')">
                        <h2>Payment Successful! Order ID: {order_group_id}</h2>
                        <p>You can close this tab and return to the shop.</p>
                        <script>setTimeout(() => window.close(), 3000);</script>
                    </body>
                </html>
            """
        else:
            return f"Payment status: {order_status}. Please try again."
            
    except Exception as e:
        print("Payment Error:", e)
        return f"Payment Verification Failed: {str(e)}", 500



@store_bp.route("/api/shop/plants", methods=["GET"])
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



@store_bp.route("/api/cart/remove", methods=["POST"])
def remove_from_cart():
    data = request.json
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



def build_order_timeline(current_status: str):
    """Generate the 8-milestone visual tracking timeline."""
    is_cancelled = (current_status or "").lower() == "cancelled"
    timeline = []

    current_idx = -1
    for i, stage in enumerate(TRACKING_STAGES):
        if stage.lower() == (current_status or "").lower():
            current_idx = i
            break

    if current_idx == -1 and not is_cancelled:
        current_idx = 0

    for i, stage in enumerate(TRACKING_STAGES):
        if is_cancelled:
            timeline.append({
                "stage": stage,
                "completed": (i == 0),
                "current": False,
                "cancelled": True
            })
        else:
            timeline.append({
                "stage": stage,
                "completed": (i <= current_idx),
                "current": (i == current_idx),
                "cancelled": False
            })

    if is_cancelled:
        timeline.append({
            "stage": "Cancelled",
            "completed": True,
            "current": True,
            "cancelled": True
        })

    return timeline


@store_bp.route('/api/orders/<user_id>', methods=['GET'])
@store_bp.route('/api/orders/user/<user_id>', methods=['GET'])
def get_orders(user_id):
    """
    Retrieve all previous orders for a customer (grouped by order_group_id).
    Supports numeric user_id or guest session identifier with complete data isolation.
    """
    try:
        user_str = str(user_id).strip()
        params: dict[str, Any] = {}
        target_uid = None
        if user_str.isdigit():
            target_uid = int(user_str)
        else:
            user_rec = User.query.filter(User.username == user_str).first()
            if user_rec:
                target_uid = user_rec.id

        if target_uid is not None:
            where_clause = "o.user_id = :uid"
            params["uid"] = target_uid
        else:
            where_clause = "(o.notes LIKE :ug OR o.shipping_address LIKE :ug)"
            params["ug"] = f"%{user_str}%"

        rows = db.session.execute(text(f"""
            SELECT o.order_id, o.order_group_id, o.plant_name, o.price, o.quantity,
                   o.total_amount, o.order_status, o.payment_status, o.payment_method,
                   o.shipping_address, o.tracking_number, o.delivery_date, o.order_date,
                   p.image_path
            FROM orders o
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE {where_clause}
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
                    "payment_method": r[8] or "Cash on Delivery",
                    "shipping_address": r[9] or "Not Provided",
                    "tracking_number": r[10] or f"GB-TRK-{gid}",
                    "delivery_date": r[11] or "3-5 Business Days",
                    "total_amount": 0.0,
                    "items": []
                }
            qty = int(r[4] or 1)
            price = float(r[3] or 0.0)
            item_total = float(r[5]) if r[5] is not None else round(price * qty, 2)
            grouped[gid]["total_amount"] += item_total
            grouped[gid]["items"].append({
                "order_id": r[0],
                "plant_name": r[2],
                "price": price,
                "quantity": qty,
                "total_amount": round(item_total, 2),
                "image_url": f"/static/{r[13].lstrip('/')}" if r[13] else None
            })

        orders_list = list(grouped.values())
        for o in orders_list:
            o["total_amount"] = round(o["total_amount"], 2)

        return jsonify({"status": "success", "orders": orders_list}), 200

    except Exception as e:
        print("Orders API error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500



@store_bp.route('/api/orders/details/<order_group_id>', methods=['GET'])
def get_order_details(order_group_id):
    """Public customer endpoint for retrieving specific order details."""
    try:
        rows = db.session.execute(text("""
            SELECT o.order_id, o.plant_name, COALESCE(o.price, p.price, 0), o.quantity,
                   o.total_amount, o.order_status, o.payment_status, o.payment_method,
                   o.shipping_address, o.tracking_number, o.delivery_date, o.order_date,
                   p.image_path
            FROM orders o
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE o.order_group_id = :gid OR CAST(o.order_id AS VARCHAR(20)) = :gid
        """), {"gid": order_group_id}).fetchall()

        if not rows:
            return jsonify({"status": "error", "message": "Order not found"}), 404

        first_row = rows[0]
        items = []
        total_amount = 0.0
        for r in rows:
            qty = int(r[3] or 1)
            price = float(r[2] or 0.0)
            item_total = float(r[4]) if r[4] is not None else round(price * qty, 2)
            total_amount += item_total
            items.append({
                "order_id": r[0],
                "plant_name": r[1],
                "price": price,
                "quantity": qty,
                "total_amount": round(item_total, 2),
                "image_url": f"/static/{r[12].lstrip('/')}" if r[12] else None
            })

        summary = {
            "order_group_id": order_group_id,
            "order_date": first_row[11].strftime("%Y-%m-%d %H:%M") if (first_row[11] and hasattr(first_row[11], 'strftime')) else str(first_row[11] or "N/A"),
            "order_status": first_row[5] or "Order Placed",
            "payment_status": first_row[6] or "Pending",
            "payment_method": first_row[7] or "Cash on Delivery",
            "shipping_address": first_row[8] or "Not Provided",
            "tracking_number": first_row[9] or f"GB-TRK-{order_group_id}",
            "delivery_date": first_row[10] or "3-5 Business Days",
            "total_amount": round(total_amount, 2),
            "items": items
        }

        return jsonify({"status": "success", "order": summary}), 200

    except Exception as e:
        print("Order details error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500



@store_bp.route('/api/orders/track/<order_group_id>', methods=['GET'])
def track_order(order_group_id):
    """
    Returns order tracking milestone timeline across the 8 required statuses:
    Order Placed, Order Confirmed, Processing, Packed, Shipped, Out for Delivery, Delivered, Cancelled.
    """
    try:
        row = db.session.execute(text("""
            SELECT TOP 1 order_group_id, order_status, tracking_number, delivery_date, order_date, shipping_address
            FROM orders
            WHERE order_group_id = :gid OR CAST(order_id AS VARCHAR(20)) = :gid
        """), {"gid": order_group_id}).fetchone()

        if not row:
            return jsonify({"status": "error", "message": "Order not found"}), 404

        current_status = row[1] or "Order Placed"
        timeline = build_order_timeline(current_status)

        return jsonify({
            "status": "success",
            "order_group_id": row[0] or order_group_id,
            "current_status": current_status,
            "tracking_number": row[2] or f"GB-TRK-{order_group_id}",
            "estimated_delivery": row[3] or "3-5 Business Days",
            "order_date": row[4].strftime("%Y-%m-%d %H:%M") if (row[4] and hasattr(row[4], 'strftime')) else str(row[4] or "N/A"),
            "shipping_address": row[5] or "Not Provided",
            "timeline": timeline
        }), 200

    except Exception as e:
        print("Track order error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

