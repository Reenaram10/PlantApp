import sqlite3
import os

db_path = 'plantapp.db'
if not os.path.exists(db_path):
    print(f"Database {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("--- Plants ---")
cursor.execute("SELECT plant_id, plant_name, image_path FROM plant LIMIT 10;")
rows = cursor.fetchall()
for r in rows:
    print(f"ID: {r[0]}, Name: {r[1]}, Image: {r[2]}")

conn.close()
