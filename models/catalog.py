from datetime import datetime, timezone
from extensions import db, BaseModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Plant(BaseModel):
    __tablename__ = "plant"

    plant_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    price = db.Column(db.Float)
    image_path = db.Column(db.String(255))
    stock = db.Column(db.Integer, default=0)
    discount_price = db.Column(db.Float, default=0.0)

class Variety(BaseModel):
    __tablename__ = "variety"

    variety_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    variety_name = db.Column(db.String(100), nullable=False)
    variety_description = db.Column(db.Text)
    variety_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=utc_now)

    plant = db.relationship('Plant', backref='varieties')

class Category(BaseModel):
    __tablename__ = 'category'
    category_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    category_name = db.Column(db.String(100), nullable=False)

class PlantCategory(BaseModel):
    __tablename__ = 'plant_category'
    plant_category_id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('plant.plant_id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id'), nullable=False)

class CategorySynonym(BaseModel):
    __tablename__ = 'category_synonym'
    synonym_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    synonym = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id', ondelete='CASCADE'), nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now)

    category = db.relationship('Category', backref='synonyms')

class PlantKnowledge(BaseModel):
    __tablename__ = "plant_knowledge"

    id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(255))
    scientific_name = db.Column(db.String(255))
    description = db.Column(db.Text)
    sunlight = db.Column(db.Text)
    water_requirement = db.Column(db.Text)
    soil_type = db.Column(db.Text)
    fertilizer = db.Column(db.Text)
    medicinal_uses = db.Column(db.Text)
    common_diseases = db.Column(db.Text)
    care_tips = db.Column(db.Text)

class DiseaseKnowledge(BaseModel):
    __tablename__ = "disease_knowledge"

    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(255))
    symptoms = db.Column(db.Text)
    causes = db.Column(db.Text)
    organic_treatment = db.Column(db.Text)
    chemical_treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)
