import sys
import os
import importlib.util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app_py_path = os.path.join(ROOT_DIR, "app.py")

spec = importlib.util.spec_from_file_location("root_flask_app", app_py_path)
assert spec is not None and spec.loader is not None, "Failed to load app.py spec"
root_flask_app = importlib.util.module_from_spec(spec)
sys.modules["root_flask_app"] = root_flask_app
spec.loader.exec_module(root_flask_app)

app = root_flask_app.app
db = root_flask_app.db

def ensure_tables():
    with app.app_context():
        print(" [DB] Verifying and creating tables...")
        db.create_all()
        print(" [DB] All tables verified successfully.")

if __name__ == "__main__":
    ensure_tables()
