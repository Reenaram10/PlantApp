from datetime import datetime, timezone
from extensions import db, BaseModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Supplier(BaseModel):
    __tablename__ = "supplier"
    supplier_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    supplier_name = db.Column(db.String(100), nullable=False)
    contact_person = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    address = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utc_now)

class InventoryTransaction(BaseModel):
    __tablename__ = "inventory_transaction"
    transaction_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    supplier_id = db.Column(db.Integer, db.ForeignKey("supplier.supplier_id", ondelete="SET NULL"), nullable=True)
    type = db.Column(db.String(10), nullable=False) # 'ADD' or 'REMOVE'
    quantity = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    bill_no = db.Column(db.String(50))
    bill_date = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=utc_now)

    plant = db.relationship('Plant', backref='transactions')
    supplier = db.relationship('Supplier', backref='transactions')

class SupplierPurchase(BaseModel):
    __tablename__ = "supplier_purchase"
    purchase_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    supplier_id = db.Column(db.Integer, db.ForeignKey("supplier.supplier_id", ondelete="CASCADE"), nullable=False)
    bill_no = db.Column(db.String(50), nullable=False)
    bill_date = db.Column(db.DateTime, default=utc_now)
    total_amount = db.Column(db.Float, nullable=False)
    amount_paid = db.Column(db.Float, default=0.0)
    balance = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='Pending') # Pending, Partial, Paid
    created_at = db.Column(db.DateTime, default=utc_now)

    supplier = db.relationship('Supplier', backref='purchases')

class PurchaseItem(BaseModel):
    __tablename__ = "purchase_item"
    item_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    purchase_id = db.Column(db.Integer, db.ForeignKey("supplier_purchase.purchase_id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)

    purchase = db.relationship('SupplierPurchase', backref='items')
    plant = db.relationship('Plant', backref='purchases')
