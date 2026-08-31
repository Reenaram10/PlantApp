from app import app, db

def ensure_tables():
    with app.app_context():
        print(" [DB] Verifying and creating tables...")
        db.create_all()
        print(" [DB] All tables created successfully.")

if __name__ == "__main__":
    ensure_tables()
