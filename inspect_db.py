import sqlite3
import os

db_path = 'plantapp.db'
if not os.path.exists(db_path):
    print(f"Database {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("--- Tables ---")
for t in tables:
    print(t[0])

conn.close()
