from datetime import datetime, timezone
from extensions import db, BaseModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Business(BaseModel):
    __tablename__ = "businesses"

    business_id = db.Column(db.String(50), primary_key=True)
    business_name = db.Column(db.String(100), nullable=False)
    owner_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    country = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Active') # 'Active' / 'Inactive'
    created_at = db.Column(db.DateTime, default=utc_now)
    updated_at = db.Column(db.DateTime, default=utc_now)

class User(BaseModel):
    __tablename__ = 'login'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='NURSERY_ADMIN') # 'SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'
    business_id = db.Column(db.String(50), default='NURS-0001')
    phone = db.Column(db.String(20), nullable=True)

    last_login = db.Column(db.DateTime)
    last_logout = db.Column(db.DateTime)
    is_online = db.Column(db.Boolean, default=False)

class AuditLog(BaseModel):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50))
    user_id = db.Column(db.Integer)
    user_email = db.Column(db.String(100))
    action = db.Column(db.String(100), nullable=False)
    module = db.Column(db.String(50), nullable=False)
    record_id = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=utc_now)
    details = db.Column(db.Text)
