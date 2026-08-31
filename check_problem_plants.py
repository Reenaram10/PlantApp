import sqlite3
import os

db_path = 'plantapp.db'
# Note: I'll try to find where the actual DB is. app.py says default_db_uri is postgres, 
# but maybe there's a local sqlite one that is being used?
# I'll check app.py again to see if it fallbacks to sqlite.

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT plant_id, plant_name, image_path FROM plant WHERE plant_name IN ('Mango', 'Lichi', 'strawberry', 'Acanthocereus', 'orange');")
    rows = cursor.fetchall()
    print("--- Specific Plants ---")
    for r in rows:
        print(f"ID: {r[0]}, Name: {r[1]}, Image: {r[2]}")
except Exception as e:
    print(f"Error: {e}")

conn.close()
