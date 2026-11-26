import inspect
from datetime import datetime, timezone
from extensions import db, BaseModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class UserQuery(BaseModel):
    __tablename__ = "user_queries"

    uq_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="SET NULL"), nullable=True)
    query_text = db.Column('query', db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now)

class _UserQueryDescriptor:
    """Descriptor that preserves UserQuery.query for Flask-SQLAlchemy model queries,
    while allowing instance-level uq.query to access or mutate query_text seamlessly."""
    def __init__(self, original_desc):
        self.original_desc = original_desc

    def __get__(self, instance, owner):
        if instance is None:
            return self.original_desc.__get__(None, owner)
        return getattr(instance, 'query_text', None)

    def __set__(self, instance, value):
        instance.query_text = value

UserQuery.query = _UserQueryDescriptor(inspect.getattr_static(db.Model, 'query'))

class UserInterest(BaseModel):
    __tablename__ = "user_interest"

    interest_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)

    interest_type = db.Column(db.String(20), nullable=False)  # enquiry / purchase
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utc_now)

class Conversation(BaseModel):
    __tablename__ = 'conversation'
    conversation_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'))
    conversation = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now)

class LLMResult(BaseModel):
    __tablename__ = 'llm_results'
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_query = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text, nullable=False)
    category_suggested = db.Column(db.String(100))
    plants_found = db.Column(db.Integer, default=0)
    llm_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=utc_now)
    expires_at = db.Column(db.DateTime)

class Query(BaseModel):
    __tablename__ = 'query'
    query_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    description = db.Column(db.Text, nullable=False)
    canonical_key = db.Column(db.String(100))   
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'), nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now)
    user = db.relationship('User', backref='queries')

class QueryResult(BaseModel):
    __tablename__ = 'query_result'
    result_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    query_id = db.Column(db.Integer, db.ForeignKey('query.query_id', ondelete='CASCADE'), nullable=False)
    result_description = db.Column(db.Text, nullable=False)
    result_source = db.Column(db.String(50), nullable=False)
    
    response_type = db.Column(db.String(50), default='text')
    response_data = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=utc_now)
    query = db.relationship('Query', backref='results')
