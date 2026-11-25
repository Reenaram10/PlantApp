"""
Image Utilities Service
File validation, image path resolution, and plant image lookups.
"""
import os
from sqlalchemy import text
from config import APP_ROOT, ALLOWED_EXTENSIONS
from extensions import db

def allowed_file(filename):
    """Validate uploaded file has an allowed image extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_plant_image_url(image_path):
    """
    Resolves local plant image path or returns default/remote image URL.
    Anchored to APP_ROOT/static/images/plants.
    """
    if not image_path:
        return "/static/images/default_plant.png"
    if image_path.startswith(('http://', 'https://')):
        return image_path
    
    clean_name = image_path.lstrip('/').split('/')[-1]  # Just the filename
    
    # 1. Try direct path first
    static_dir = os.path.join(APP_ROOT, 'static')
    plants_dir = os.path.join(static_dir, 'images', 'plants')
    
    direct_path = os.path.join(plants_dir, clean_name)
    if os.path.exists(direct_path):
        return f"/static/images/plants/{clean_name}"
        
    # 2. Case-insensitive and prefix-insensitive search in static/images/plants
    if os.path.exists(plants_dir):
        files = os.listdir(plants_dir)
        name_base = os.path.splitext(clean_name.lower())[0]
        
        # Try exact match first (case-insensitive)
        for f in files:
            if f.lower() == clean_name.lower():
                return f"/static/images/plants/{f}"
        
        # Try finding the name as part of the filename (handles prefixes like plant_76_)
        for f in files:
            if name_base in f.lower():
                return f"/static/images/plants/{f}"
                
    # 3. Fallback to default
    return "/static/images/default_plant.png"

def search_plant_image(query):
    """
    Search for a plant image in the local database
    """
    try:
        # Search in plant table
        plant = db.session.execute(text("""
            SELECT TOP 1 image_path, plant_name 
            FROM plant 
            WHERE LOWER(plant_name) LIKE :q
        """), {"q": f"%{query.lower()}%"}).fetchone()
        
        if plant and plant[0]:
            return {
                "status": "success",
                "image_url": get_plant_image_url(plant[0])
            }
            
        # Search in variety table as fallback
        variety = db.session.execute(text("""
            SELECT TOP 1 p.image_path, v.variety_name
            FROM variety v
            JOIN plant p ON p.plant_id = v.plant_id
            WHERE LOWER(v.variety_name) LIKE :q
        """), {"q": f"%{query.lower()}%"}).fetchone()
        
        if variety and variety[0]:
            return {
                "status": "success",
                "image_url": get_plant_image_url(variety[0])
            }
            
        return {"status": "error", "message": "No local image found for this plant."}
    except Exception as e:
        print(f"Local image search error: {e}")
        return {"status": "error", "message": str(e)}
