import os
from flask import Blueprint, send_from_directory, current_app, render_template, request
from config import APP_ROOT

static_bp = Blueprint('static_routes', __name__)

@static_bp.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('.', 'auth.html')

@static_bp.route('/admin')
@static_bp.route('/admin.html')
def admin_page():
    from flask import send_from_directory
    return send_from_directory('static', 'admin.html')

@static_bp.route('/<path:filename>')
def serve_root_files(filename):
    from flask import send_from_directory
    # If the file exists in root, serve it
    if os.path.exists(os.path.join(current_app.root_path, filename)):
        return send_from_directory('.', filename)
    # fall back to static
    return send_from_directory('static', filename)

