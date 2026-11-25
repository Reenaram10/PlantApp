import os
import time
from sqlalchemy import text
from extensions import db
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.rag_service import faiss_rag, ask_rebuild_index
from create_app import create_app

app = create_app()

def reset_plant_sequence():
    """Reset the plant_id sequence to the max ID in the table"""
    try:
        max_id_row = db.session.execute(text("SELECT MAX(plant_id) FROM plant")).fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0
        
        if max_id:
            if db.engine.dialect.name == 'mssql':
                db.session.execute(text(f"DBCC CHECKIDENT ('plant', RESEED, {max_id})"))
                db.session.commit()
                print(f" Reseeded plant identity to {max_id}")
            else:
                next_val = max_id + 1
                db.session.execute(text(f"ALTER SEQUENCE plant_plant_id_seq RESTART WITH {next_val}"))
                db.session.commit()
                print(f" Reset plant sequence to start at {next_val} (max was {max_id})")
        else:
            print(" No plants found in database")
        
        return True
    except Exception as e:
        print(f" Sequence reset warning: {e}")
        return False

def ensure_admin_user():
    """Ensure the default super admin user exists with valid credentials."""
    try:
        user = db.session.execute(text("""
            SELECT id FROM login WHERE LOWER(username) = 'admin' OR LOWER(email) = 'admin' OR LOWER(email) = 'admin@gmail.com'
        """)).fetchone()
        if not user:
            db.session.execute(text("""
                INSERT INTO login (username, email, password, role, business_id, is_online)
                VALUES ('admin', 'admin@gmail.com', 'admin@123', 'SUPER_ADMIN', 'SYSTEM', 0)
            """))
            db.session.commit()
            print(" [DB] Admin user created: admin / admin@123")
        else:
            db.session.execute(text("""
                UPDATE login SET username = 'admin', password = 'admin@123' WHERE id = :id
            """), {"id": user[0]})
            db.session.commit()
            print(" [DB] Admin user credentials verified: admin / admin@123")
    except Exception as e:
        db.session.rollback()
        print(f" [DB] Admin user setup notice: {e}")

if __name__ == "__main__":
    for attempt in range(1, 4):
        try:
            with app.app_context():
                db.session.remove()
                db.create_all()
                print(" DB schemas verified.")
                ensure_admin_user()
                
                # (1) DB Connectivity Check
                print("(1) Result:", db.session.execute(text("SELECT 1")).fetchone())
                print(" DB CONNECTED SUCCESSFULLY")
                    
                # (2) Reset Plant Sequence
                reset_plant_sequence()
                
                # (3) Sync FAISS RAG from database / CSV files
                if ask_rebuild_index():
                    faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
                else:
                    faiss_rag.load_index()
                break
                
        except Exception as e:
            print(f" DB initialization attempt {attempt}/3 error: {e}")
            if attempt < 3:
                print(" Retrying database connection in 2 seconds...")
                time.sleep(2)
            else:
                print(" Could not establish database connection after 3 attempts.")

    port = int(os.getenv("PORT", 8082))
    print(f" Starting on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
