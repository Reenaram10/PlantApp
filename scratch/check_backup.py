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
    rows = db.session.execute(text(
        "SELECT bs.database_name, bs.backup_start_date, bs.backup_finish_date, bs.type, bmf.physical_device_name "
        "FROM msdb.dbo.backupset bs "
        "JOIN msdb.dbo.backupmediafamily bmf ON bs.media_set_id = bmf.media_set_id "
        "WHERE bs.database_name = 'PlantAppDB' "
        "ORDER BY bs.backup_start_date DESC"
    )).fetchall()
    print("Backupset rows for PlantAppDB:", len(rows))
    for r in rows:
        print(dict(r._mapping))
