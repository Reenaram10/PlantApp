import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from create_app import create_app
from extensions import db
from sqlalchemy import text

app = create_app()
with app.app_context():
    s = db.session.execute(text("SELECT * FROM category_synonym WHERE synonym LIKE '%test%'")).fetchall()
    c = db.session.execute(text("SELECT * FROM category WHERE category_name LIKE '%TEST%'")).fetchall()
    ident = db.session.execute(text(
        "SELECT IDENT_SEED('category_synonym') AS seed_val, "
        "IDENT_INCR('category_synonym') AS incr_val, "
        "IDENT_CURRENT('category_synonym') AS curr_val"
    )).fetchone()
    print("category_synonym test rows:", len(s), [dict(r._mapping) for r in s])
    print("category TEST rows:", len(c), [dict(r._mapping) for r in c])
    print("category_synonym identity:", dict(ident._mapping))
