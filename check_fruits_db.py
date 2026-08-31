import os
from sqlalchemy import text
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

DATABASE_URL = "postgresql://neondb_owner:npg_KBStXxq52HPZ@ep-gentle-grass-adnpzd0p-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
db = SQLAlchemy(app)

def check_db():
    with app.app_context():

        print("--- category table columns ---")
        try:
            cols = db.session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_name='category'
            """)).fetchall()
            for c in cols:
                print(f"Column: {c[0]}, Type: {c[1]}")
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

        print("\n--- All categories ---")
        try:
            rows = db.session.execute(text("SELECT * FROM category")).fetchall()
            for r in rows:
                print(r)
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

        print("\n--- plant_category columns ---")
        try:
            cols = db.session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_name='plant_category'
            """)).fetchall()
            for c in cols:
                print(f"Column: {c[0]}, Type: {c[1]}")
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

        print("\n--- category_synonym columns ---")
        try:
            cols = db.session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_name='category_synonym'
            """)).fetchall()
            for c in cols:
                print(f"Column: {c[0]}, Type: {c[1]}")
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

        print("\n--- Sample category_synonym rows ---")
        try:
            rows = db.session.execute(text("SELECT * FROM category_synonym LIMIT 10")).fetchall()
            for r in rows:
                print(r)
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

        print("\n--- Sample plants ---")
        try:
            rows = db.session.execute(text("SELECT plant_id, plant_name FROM plant LIMIT 20")).fetchall()
            for r in rows:
                print(r)
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

if __name__ == "__main__":
    check_db()
