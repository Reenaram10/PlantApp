import os
import random
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.commerce import Order
from models.tenant import User
from models.inventory import Supplier, SupplierPurchase, PurchaseItem, InventoryTransaction, utc_now
from models.catalog import Plant
from auth.decorators import require_tenant_auth, get_current_tenant
from auth.audit import log_audit_action

admin_orders_bp = Blueprint('admin_orders', __name__)

@admin_orders_bp.route('/api/admin/customers', methods=['GET'])
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

@admin_orders_bp.route('/api/admin/orders', methods=['GET'])
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

@admin_orders_bp.route('/api/admin/orders/record-payment', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def record_order_payment():
    """Record partial or full payment for an order group within current nursery"""
    biz_id = get_current_tenant()
    data = request.get_json() or {}
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

@admin_orders_bp.route('/api/admin/orders/<order_group_id>', methods=['GET'])
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

@admin_orders_bp.route('/api/admin/orders/<order_group_id>', methods=['PUT'])
def update_order_status_admin(order_group_id):
    """Update status of all orders in a group"""
    status = (request.get_json() or {}).get("status")
    if not status:
        return jsonify({"status": "error", "message": "Status is required"}), 400
        
    try:
        db.session.execute(text("UPDATE orders SET order_status = :s WHERE order_group_id = :gid"), {"s": status, "gid": order_group_id})
        db.session.commit()
        return jsonify({"status": "success", "message": "Order status updated"})
    except Exception as e:
        print(f"Error in update_order_status_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@admin_orders_bp.route('/api/admin/purchases', methods=['POST'])
def add_supplier_purchase():
    """Add a new purchase from a supplier"""
    data = request.get_json() or {}
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

@admin_orders_bp.route('/api/admin/purchases', methods=['GET'])
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

@admin_orders_bp.route('/api/admin/purchases/record-payment', methods=['POST'])
def record_purchase_payment():
    """Record a payment against a purchase balance"""
    data = request.get_json() or {}
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

@admin_orders_bp.route('/api/admin/suppliers', methods=['GET'])
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

@admin_orders_bp.route('/api/admin/suppliers', methods=['POST'])
def add_supplier():
    """Add a new supplier"""
    try:
        data = request.get_json() or {}
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

