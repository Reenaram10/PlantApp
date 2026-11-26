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
    c_old = db.session.execute(text("SELECT COUNT(*) FROM audit_logs WHERE id BETWEEN 255 AND 270")).scalar()
    print("IDs 255-270 remaining:", c_old)
    rows = db.session.execute(text(
        "SELECT id, business_id, user_id, user_email, action, module, record_id, details, timestamp "
        "FROM audit_logs WHERE (user_email IN ('test_smoke_leg@example.com', 'test_smoke_new@example.com') "
        "OR details IN ('Added plant ''TEST_Fern_Leg''', 'Added plant ''TEST_Fern_New''', 'Restocked TEST_Fern_Leg by +10', 'Restocked TEST_Fern_New by +10')) "
        "AND id >= 279 ORDER BY id"
    )).fetchall()
    print(f"New test rows found: {len(rows)}")
    for r in rows:
        print(dict(r._mapping))
    if rows:
        res = db.session.execute(text(
            "DELETE FROM audit_logs WHERE (user_email IN ('test_smoke_leg@example.com', 'test_smoke_new@example.com') "
            "OR details IN ('Added plant ''TEST_Fern_Leg''', 'Added plant ''TEST_Fern_New''', 'Restocked TEST_Fern_Leg by +10', 'Restocked TEST_Fern_New by +10')) "
            "AND id >= 279"
        ))
        db.session.commit()
        print("Deleted new test rows:", res.rowcount)
    rem = db.session.execute(text("SELECT COUNT(*) FROM audit_logs WHERE id >= 279")).scalar()
    print("Remaining test audit_logs:", rem)
