from datetime import datetime, timezone
from extensions import db, BaseModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Cart(BaseModel):
    __tablename__ = "cart_new"
    cart_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.String(100), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id"), nullable=False)
    added_at = db.Column(db.DateTime, default=utc_now)
    quantity = db.Column(db.Integer, default=1)

    plant = db.relationship('Plant', backref='cart_items')

class Order(BaseModel):
    __tablename__ = 'orders'
    order_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_name = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('login.id'))
    order_date = db.Column(db.DateTime, default=utc_now)
    total_amount = db.Column(db.Float)
    payment_status = db.Column(db.String(20), default='Pending')
    order_status = db.Column(db.String(20), default='Processing')
    order_group_id = db.Column(db.String(50))
    price = db.Column(db.Float)
    quantity = db.Column(db.Integer)
    payment_method = db.Column(db.String(50))
    shipping_address = db.Column(db.Text)
    notes = db.Column(db.Text)
    delivery_date = db.Column(db.String(50))
    tracking_number = db.Column(db.String(100))
    amount_paid = db.Column(db.Float, default=0.0)
    balance = db.Column(db.Float, default=0.0)
