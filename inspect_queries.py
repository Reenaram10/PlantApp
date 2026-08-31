from app import app, db, Query
import sqlite3
import os

def inspect_queries():
    with app.app_context():
        try:
            queries = Query.query.limit(20).all()
            print(f"Found {len(queries)} queries in database.")
            for q in queries:
                print(f"[{q.query_id}] {q.description[:100]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    inspect_queries()
