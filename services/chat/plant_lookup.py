"""
Plant Lookup & Botanical Catalog Service
Handles fuzzy plant matching, phonetic resolution, synonym lookups, DB profile retrieval,
and catalog interest logging.
"""
import re
import difflib
import time
from sqlalchemy import text
from flask import has_app_context, current_app

from extensions import db
from models.catalog import Plant, PlantKnowledge, Category, CategorySynonym
from models.chat import Query
from services.image_utils import get_plant_image_url

def resolve_plant_synonym(user_message):
    """Resolve plant from plant_synonym table matching substring."""
    row = db.session.execute(text("""
        SELECT TOP 1 p.plant_id, p.plant_name, p.description, p.price, p.image_path
        FROM plant_synonym ps
        JOIN plant p ON p.plant_id = ps.plant_id
        WHERE LOWER(:msg) LIKE CONCAT('%', LOWER(ps.synonym), '%')
        ORDER BY LEN(ps.synonym) DESC
    """), {"msg": user_message.lower()}).fetchone()

    if row:
        return {
            "plant_id": row[0],
            "plant_name": row[1],
            "description": row[2],
            "price": row[3],
            "image_url": get_plant_image_url(row[4])
        }
    return None

def fuzzy_resolve_plant(user_message):
    """
    NLP Fuzzy plant matching mapping user query (including misspellings/phonetic inputs like 'rase nuh', 'roz', 'hibiscs')
    to a canonical plant in the database.
    """
    msg_clean = re.sub(r'[^\w\s]', ' ', user_message.lower()).strip()
    if not msg_clean:
        return None

    phonetic_fixes = {
        "rase nuh": "rose",
        "rase": "rose",
        "roz": "rose",
        "ros": "rose",
        "hibiscs": "hibiscus",
        "snak": "snake plant",
        "alo": "aloe vera",
        "money": "money plant",
        "jasmin": "jasmine",
        "tulsi": "holy basil",
        "tulas": "tulsi"
    }

    def _fetch_rows():
        try:
            return db.session.execute(text("""
                SELECT p.plant_id, p.plant_name, p.price, p.image_path, ps.synonym
                FROM plant p
                LEFT JOIN plant_synonym ps ON p.plant_id = ps.plant_id
            """)).fetchall()
        except Exception as e:
            print(f"Error fetching plants for fuzzy resolution: {e}")
            return []

    rows = _fetch_rows()
    direct = resolve_plant_synonym(user_message)

    if direct:
        direct["score"] = 1.0
        direct["is_exact"] = True

    plant_map = {}
    for row in rows:
        pid, name, price, img, syn = row[0], row[1], row[2], row[3], row[4]
        if pid not in plant_map:
            plant_map[pid] = {
                "plant_id": pid,
                "plant_name": name,
                "price": price,
                "image_url": get_plant_image_url(img),
                "synonyms": set()
            }
        if name:
            plant_map[pid]["synonyms"].add(name.lower())
        if syn:
            plant_map[pid]["synonyms"].add(syn.lower())

    # Check phonetic fix dictionary
    for trigger, target in phonetic_fixes.items():
        if trigger in msg_clean:
            for pid, info in plant_map.items():
                for s in info["synonyms"]:
                    if target in s:
                        return {
                            "plant_id": info["plant_id"],
                            "plant_name": info["plant_name"],
                            "price": info["price"],
                            "image_url": info["image_url"],
                            "score": 0.95,
                            "is_exact": False,
                            "raw_query": trigger
                        }

    # Direct match check
    if direct:
        return direct

    # Difflib SequenceMatcher fuzzy matching
    best_match = None
    best_score = 0.0

    words = msg_clean.split()
    for pid, info in plant_map.items():
        for syn in info["synonyms"]:
            ratio = difflib.SequenceMatcher(None, msg_clean, syn).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = info

            for w in words:
                w_ratio = difflib.SequenceMatcher(None, w, syn).ratio()
                if w_ratio > best_score and w_ratio > 0.6:
                    best_score = w_ratio
                    best_match = info

    if best_match and best_score >= 0.55:
        return {
            "plant_id": best_match["plant_id"],
            "plant_name": best_match["plant_name"],
            "price": best_match["price"],
            "image_url": best_match["image_url"],
            "score": best_score,
            "is_exact": (best_score >= 0.98)
        }

    return None

def fuzzy_find_plant(name):
    """Find plant by name pattern."""
    return db.session.execute(text("""
        SELECT TOP 1 plant_id, plant_name, description, price
        FROM plant
        WHERE LOWER(plant_name) LIKE LOWER(:x)
    """), {"x": f"%{name}%"}).fetchone()

def get_category_from_synonym(user_input):
    """Check if user input matches any category synonym."""
    synonym_row = db.session.execute(text("""
        SELECT c.category_id, c.category_name
        FROM category_synonym cs
        JOIN category c ON cs.category_id = c.category_id
        WHERE LOWER(cs.synonym) = :syn
    """), {"syn": user_input.lower()}).fetchone()
    
    if synonym_row:
        return {
            "category_id": synonym_row[0],
            "category_name": synonym_row[1]
        }
    return None

def add_category_synonym(synonym, category_id):
    """Add a synonym for an existing category"""
    try:
        syn = CategorySynonym(
            synonym=synonym.lower(),
            category_id=category_id
        )
        db.session.add(syn)
        db.session.commit()
        print(f" Synonym '{synonym}' added for category_id {category_id}")
        return syn.synonym_id
    except Exception as e:
        print(f"Error adding synonym: {e}")
        db.session.rollback()
        return None

def handle_count_plants(canonical_key, query_id, user_message, user_id=None):
    """Handles plant count queries safely."""
    if not canonical_key.startswith("COUNT_PLANTS"):
        return None

    row = db.session.execute(
        text("SELECT COUNT(*) FROM plant")
    ).fetchone()

    count = row[0] if row else 0
    reply = f" There are {count} plants available in our nursery."

    return {
        "status": "success",
        "type": "text",
        "reply": reply,
        "canonical_key": canonical_key
    }

def log_category_interest(user_id, category_id, interest_type="search"):
    """Log user interest in a plant category."""
    if not user_id or not str(user_id).isdigit():
        return

    try:
        if db.engine.dialect.name == 'mssql':
            db.session.execute(text("""
                IF EXISTS (SELECT 1 FROM user_category_interest WHERE user_id = :u AND category_id = :c AND interest_type = :t)
                    UPDATE user_category_interest 
                    SET count = count + 1, last_interaction = CURRENT_TIMESTAMP 
                    WHERE user_id = :u AND category_id = :c AND interest_type = :t
                ELSE
                    INSERT INTO user_category_interest (user_id, category_id, interest_type, count, last_interaction)
                    VALUES (:u, :c, :t, 1, CURRENT_TIMESTAMP)
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })
        else:
            db.session.execute(text("""
                INSERT INTO user_category_interest
                (user_id, category_id, interest_type, count)
                VALUES (:u, :c, :t, 1)
                ON CONFLICT (user_id, category_id, interest_type)
                DO UPDATE SET
                    count = user_category_interest.count + 1,
                    last_interaction = CURRENT_TIMESTAMP
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(" Category interest failed:", e)

def log_user_interest(user_id, plant_id, interest_type, notes=""):
    """Log user interest in a specific plant."""
    try:
        if not user_id or str(user_id) in ["anonymous", "guest_user"]:
            return

        db.session.execute(text("""
            INSERT INTO user_interest (user_id, plant_id, interest_type, notes)
            VALUES (:user_id, :plant_id, :interest_type, :notes)
        """), {
            "user_id": int(user_id),
            "plant_id": int(plant_id),
            "interest_type": interest_type,
            "notes": notes
        })
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(" user_interest insert failed:", e)

def store_query(description, user_id, canonical_key=None):
    """Store user query in database."""
    try:
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            print(f" Invalid user_id: {user_id}, using 1 as default")
            user_id_int = 1

        query = Query(
            description=description,
            canonical_key=canonical_key,
            user_id=user_id_int
        )
        db.session.add(query)
        db.session.commit()
        print(f" Stored query: {description}")
        return query.query_id
    except Exception as e:
        print(f" Error storing query: {e}")
        db.session.rollback()
        return None

def fetch_db_plant_details(plant_name_or_query):
    """Retrieve plant details, stock, price, status, and care info directly from database tables (Plant & PlantKnowledge)."""
    try:
        clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy|price of|cost of|stock of|how many)\s+', '', plant_name_or_query, flags=re.IGNORECASE).strip().lower()
        if not clean_target:
            clean_target = plant_name_or_query.strip().lower()
            
        p_db = Plant.query.filter(Plant.plant_name.ilike(f"%{clean_target}%")).first()
        pk_db = PlantKnowledge.query.filter(PlantKnowledge.plant_name.ilike(f"%{clean_target}%")).first()

        if not p_db and not pk_db:
            return None

        plant_name = p_db.plant_name if p_db else pk_db.plant_name
        price = p_db.price if p_db else 0.0
        disc_price = 0.0
        stock_cnt = p_db.stock if p_db and p_db.stock is not None else 0
        status_str = "Available" if stock_cnt > 0 else "Out of Stock"
        desc = p_db.description if p_db and p_db.description else (pk_db.description if pk_db else "")

        reply_parts = [f"🌱 **Plant Profile: {plant_name}**"]

        price_str = f"₹{price:.2f}"
        if disc_price and disc_price > 0 and disc_price < price:
            price_str = f"~~₹{price:.2f}~~ **₹{disc_price:.2f}** (Special Price!)"
            
        reply_parts.append(f"💵 **Price**: {price_str}\n📦 **Stock**: **{stock_cnt}** in stock (**Status**: {status_str})")

        if pk_db:
            if pk_db.scientific_name:
                reply_parts.append(f"🔬 **Scientific Name**\n*{pk_db.scientific_name.strip()}*")
            
            conds = []
            if pk_db.sunlight:
                conds.append(f"• Sunlight: {pk_db.sunlight.strip()}")
            if pk_db.water_requirement:
                conds.append(f"• Water: {pk_db.water_requirement.strip()}")
            if pk_db.soil_type:
                conds.append(f"• Soil: {pk_db.soil_type.strip()}")
            if conds:
                reply_parts.append("🌡️ **Growing Conditions**\n" + "\n".join(conds))

            if pk_db.fertilizer:
                reply_parts.append(f"🌾 **Fertilizer & Nutrients**\n{pk_db.fertilizer.strip()}")
            if pk_db.care_tips:
                reply_parts.append(f"🌿 **Care Tips**\n{pk_db.care_tips.strip()}")
            if pk_db.medicinal_uses:
                reply_parts.append(f"💊 **Medicinal / Extra Uses**\n{pk_db.medicinal_uses.strip()}")
        elif desc:
            reply_parts.append(f"📝 **Description**\n{desc.strip()}")

        return "\n\n".join(reply_parts)
    except Exception as err:
        print(f"Error in fetch_db_plant_details: {err}")
        return None

def check_db_plant_availability(query_or_name, failures=None, timings=None):
    """Check if plant exists in Plant or PlantKnowledge database tables."""
    clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy)\s+', '', query_or_name, flags=re.IGNORECASE).strip().lower()
    if not clean_target:
        clean_target = query_or_name.strip().lower()
    
    t_db_start = time.perf_counter()
    matched_name = None
    try:
        plant_db = Plant.query.filter(Plant.plant_name.ilike(f"%{clean_target}%")).first()
        if plant_db:
            matched_name = plant_db.plant_name
    except Exception as e:
        if failures is not None:
            failures["db_plant_query_error"] = str(e)

    if not matched_name:
        try:
            pk_db = PlantKnowledge.query.filter(PlantKnowledge.plant_name.ilike(f"%{clean_target}%")).first()
            if pk_db:
                matched_name = pk_db.plant_name
        except Exception as e:
            if failures is not None:
                failures["db_pk_query_error"] = str(e)

    if not matched_name:
        try:
            match = db.session.execute(text("SELECT plant_name FROM plant WHERE LOWER(plant_name) LIKE LOWER(:x)"), {"x": f"%{clean_target}%"}).fetchone()
            if match:
                matched_name = match[0]
        except Exception as e:
            if failures is not None:
                failures["db_raw_sql_error"] = str(e)

    if timings is not None:
        timings["db_search_ms"] = round((time.perf_counter() - t_db_start) * 1000, 2)
    return matched_name

def format_plant_profile(bp):
    """Format dictionary plant profile into markdown reply."""
    def is_avail(val):
        if val is None:
            return False
        val_str = str(val).strip().lower()
        return val_str not in ("", "n/a", "none", "null", "undefined", "unspecified", "-")

    reply_parts = [f"🌱 Plant Profile: {bp['plant_name'].strip()}"]
    
    if is_avail(bp.get('scientific_name')):
        reply_parts.append(f"🔬 Scientific Name\n{bp['scientific_name'].strip()}")
        
    cond_parts = []
    if is_avail(bp.get('temperature')):
        cond_parts.append(f"• Temperature: {bp['temperature'].strip()}")
    if is_avail(bp.get('humidity')):
        cond_parts.append(f"• Humidity: {bp['humidity'].strip()}")
    if is_avail(bp.get('moisture')):
        cond_parts.append(f"• Soil Moisture: {bp['moisture'].strip()}")
    if is_avail(bp.get('sunlight')):
        cond_parts.append(f"• Sunlight: {bp['sunlight'].strip()}")
        
    if cond_parts:
        reply_parts.append("🌡️ Growing Conditions\n" + "\n".join(cond_parts))
        
    if is_avail(bp.get('soil_type')):
        reply_parts.append(f"🌾 Soil\n{bp['soil_type'].strip()}")
        
    if is_avail(bp.get('watering_frequency')):
        reply_parts.append(f"💧 Watering\n{bp['watering_frequency'].strip()}")
        
    nutr_parts = []
    if is_avail(bp.get('nitrogen')):
        nutr_parts.append(f"🟢 Nitrogen: {bp['nitrogen'].strip()}")
    if is_avail(bp.get('phosphorus')):
        nutr_parts.append(f"🟡 Phosphorus: {bp['phosphorus'].strip()}")
    if is_avail(bp.get('potassium')):
        nutr_parts.append(f"🟣 Potassium: {bp['potassium'].strip()}")
        
    if nutr_parts:
        reply_parts.append("🌿 Nutrient Requirements\n" + "\n".join(nutr_parts))
        
    if is_avail(bp.get('growth_duration')):
        reply_parts.append(f"⏳ Growth Duration\n{bp['growth_duration'].strip()}")
        
    return "\n\n".join(reply_parts)

def make_plant_unavailable_guardrail(query_or_name):
    """Generate availability guardrail notice for unavailable plant queries."""
    clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy)\s+', '', query_or_name, flags=re.IGNORECASE).strip()
    if not clean_target:
        clean_target = query_or_name.strip()
    clean_target = clean_target.title()
    
    return (
        f"🌿 **Nursery Availability Notice**\n\n"
        f"Sorry, **{clean_target}** is currently not available in our nursery catalog or database. "
        f"We are actively expanding our plant collection and hope to bring it to our nursery soon! 🪴\n\n"
        f"In the meantime, feel free to explore our available plants in the shop or ask about our indoor plants, "
        f"flowering saplings, and fruit trees!"
    )
