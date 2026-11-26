import ast
import re

with open('app.py', 'r', encoding='utf-8') as f:
    app_text = f.read()
    app_lines = app_text.splitlines(keepends=True)

tree = ast.parse(app_text)
funcs_by_name = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

blueprint_defs = [
    {
        'file': 'auth_routes.py',
        'bp_var': 'auth_bp',
        'bp_name': 'auth',
        'helpers': [],
        'funcs': ['register', 'login', 'logout', 'update_user_phone'],
        'header': '''import os
from flask import Blueprint, request, jsonify, current_app, g
from extensions import db
from models.tenant import User, Business
from auth.jwt_utils import create_token
from auth.decorators import require_tenant_auth
from auth.audit import log_audit_action

auth_bp = Blueprint('auth', __name__)
'''
    },
    {
        'file': 'super_admin_routes.py',
        'bp_var': 'super_admin_bp',
        'bp_name': 'super_admin',
        'helpers': [],
        'funcs': [
            'super_admin_login', 'register_new_nursery', 'list_nurseries_super_admin',
            'delete_nursery_super_admin', 'toggle_nursery_status', 'super_admin_stats',
            'get_super_admin_audit_logs'
        ],
        'header': '''import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from extensions import db
from models.tenant import User, Business, AuditLog
from models.catalog import Plant
from models.commerce import Order
from auth.jwt_utils import create_token
from auth.decorators import require_tenant_auth
from auth.audit import log_audit_action

super_admin_bp = Blueprint('super_admin', __name__)
'''
    },
    {
        'file': 'admin_inventory_routes.py',
        'bp_var': 'admin_inventory_bp',
        'bp_name': 'admin_inventory',
        'helpers': [],
        'funcs': [
            'suggest_varieties', 'add_varieties', 'get_existing_varieties_admin',
            'get_inventory', 'restock_plant', 'get_plant_details', 'add_plant',
            'update_plant_details', 'delete_plant_endpoint', 'search_plants_admin',
            'update_plant_image_admin', 'inventory_add', 'inventory_remove',
            'get_inventory_transactions', 'manage_plant_image', 'list_plant_images'
        ],
        'header': '''import os
import json
import re
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from sqlalchemy import text
from extensions import db
from config import APP_ROOT
from models.catalog import Plant, Variety
from models.inventory import InventoryTransaction, Supplier
from auth.decorators import require_tenant_auth
from auth.audit import log_audit_action
from services.image_utils import allowed_file, get_plant_image_url, search_plant_image
from services.llm_service import llm_call

admin_inventory_bp = Blueprint('admin_inventory', __name__)
'''
    },
    {
        'file': 'admin_orders_routes.py',
        'bp_var': 'admin_orders_bp',
        'bp_name': 'admin_orders',
        'helpers': [],
        'funcs': [
            'get_customers_admin', 'get_all_orders_admin', 'record_order_payment',
            'get_order_details_admin', 'update_order_status_admin', 'add_supplier_purchase',
            'get_supplier_purchases', 'record_purchase_payment', 'get_suppliers', 'add_supplier'
        ],
        'header': '''import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.commerce import Order
from models.tenant import User
from models.inventory import Supplier, SupplierPurchase, PurchaseItem, InventoryTransaction
from models.catalog import Plant
from auth.decorators import require_tenant_auth
from auth.audit import log_audit_action

admin_orders_bp = Blueprint('admin_orders', __name__)
'''
    },
    {
        'file': 'admin_profiles_routes.py',
        'bp_var': 'admin_profiles_bp',
        'bp_name': 'admin_profiles',
        'helpers': [],
        'funcs': [
            'sync_rag_index', 'get_profile_stats', 'upload_profile_dataset',
            'download_profile_template'
        ],
        'header': '''import os
import json
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, send_file, Response
from werkzeug.utils import secure_filename
from config import APP_ROOT, PROFILE_DATASETS
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.rag_service import faiss_rag

admin_profiles_bp = Blueprint('admin_profiles', __name__)
'''
    },
    {
        'file': 'category_routes.py',
        'bp_var': 'category_bp',
        'bp_name': 'category',
        'helpers': ['add_category_synonym'],
        'funcs': [
            'get_categories_admin', 'add_category_admin', 'delete_category_admin',
            'get_category_plants_admin', 'manage_category_synonyms'
        ],
        'header': '''import os
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.catalog import Category, PlantCategory, Plant, CategorySynonym
from auth.decorators import require_tenant_auth

category_bp = Blueprint('category', __name__)
'''
    },
    {
        'file': 'shop_routes.py',
        'bp_var': 'shop_bp',
        'bp_name': 'shop',
        'helpers': [],
        'funcs': ['get_plants', 'get_shop_plants', 'suggest_plants', 'save_user_interest'],
        'header': '''import os
import json
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.catalog import Plant
from models.chat import UserInterest
from models.tenant import User
from services.image_utils import get_plant_image_url, search_plant_image

shop_bp = Blueprint('shop', __name__)
'''
    },
    {
        'file': 'cart_routes.py',
        'bp_var': 'cart_bp',
        'bp_name': 'cart',
        'helpers': [],
        'funcs': ['add_to_cart', 'get_cart', 'update_cart', 'remove_from_cart'],
        'header': '''import os
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.commerce import Cart
from models.catalog import Plant
from services.image_utils import get_plant_image_url

cart_bp = Blueprint('cart', __name__)
'''
    },
    {
        'file': 'order_routes.py',
        'bp_var': 'order_bp',
        'bp_name': 'order',
        'helpers': ['build_order_timeline'],
        'funcs': ['checkout', 'payment_response', 'get_orders', 'get_order_details', 'track_order'],
        'extra_top': '''
TRACKING_STAGES = [
    "Order Placed",
    "Order Confirmed",
    "Processing",
    "Packed",
    "Shipped",
    "Out for Delivery",
    "Delivered"
]
''',
        'header': '''import os
import json
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, render_template
from sqlalchemy import text
from extensions import db
from models.commerce import Order, Cart
from models.catalog import Plant
from models.tenant import User
from services.payment import verify_checksum, get_payment_status
from services.image_utils import get_plant_image_url

order_bp = Blueprint('order', __name__)
'''
    },
    {
        'file': 'chat_routes.py',
        'bp_var': 'chat_bp',
        'bp_name': 'chat',
        'helpers': ['extract_json'],
        'funcs': [
            'chat', 'save_user_query', 'get_user_history', 'get_history',
            'manage_queries', 'flask_history_query', 'flask_history_profile',
            'flask_history_variable', 'flask_history_analysis', 'manage_query_results'
        ],
        'header': '''import os
import re
import json
import time
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, Response
from extensions import db, langfuse_client
from models.chat import UserQuery, Query, QueryResult
from models.tenant import User
from services.chat.chat_service import process_message
from services.chat.dialog_state import USER_DIALOG_STATE
from services.llm_service import normalize_query

chat_bp = Blueprint('chat', __name__)
'''
    },
    {
        'file': 'whatsapp_routes.py',
        'bp_var': 'whatsapp_bp',
        'bp_name': 'whatsapp',
        'helpers': ['normalize_phone', 'find_user_by_phone_or_id'],
        'funcs': [
            'whatsapp_get_cart', 'whatsapp_cart_approve', 'whatsapp_cart_update',
            'whatsapp_checkout', 'whatsapp_get_orders', 'whatsapp_track_order'
        ],
        'header': '''import os
import re
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.commerce import Cart, Order
from models.catalog import Plant
from models.tenant import User
from routes.order_routes import checkout
from services.image_utils import get_plant_image_url

whatsapp_bp = Blueprint('whatsapp', __name__)
'''
    },
    {
        'file': 'diagnosis_routes.py',
        'bp_var': 'diagnosis_bp',
        'bp_name': 'diagnosis',
        'helpers': [],
        'funcs': [
            'api_detect_tomato_disease', 'n8n_plant_identify', 'n8n_disease_identify',
            'n8n_webhook', 'api_plant_diagnosis'
        ],
        'extra_top': '''
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
N8N_DIAGNOSIS_WEBHOOK = os.getenv("N8N_DIAGNOSIS_WEBHOOK", "http://localhost:5678/webhook/plant-diagnosis")
''',
        'header': '''import os
import json
import requests
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app
from config import APP_ROOT
from extensions import db
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.image_utils import allowed_file
from services.rag_service import tomato_rag, plant_rag
from services.disease_models import (
    predict_tomato_disease, predict_crop_type, predict_potato_disease, predict_flower_crop
)

diagnosis_bp = Blueprint('diagnosis', __name__)
'''
    },
    {
        'file': 'config_routes.py',
        'bp_var': 'config_bp',
        'bp_name': 'config',
        'helpers': [],
        'funcs': ['manage_config', 'stt_api', 'health_check'],
        'header': '''import os
from flask import Blueprint, request, jsonify, current_app
from config import runtime_settings
from services.speech_service import process_audio
from extensions import db

config_bp = Blueprint('config', __name__)
'''
    },
    {
        'file': 'static_routes.py',
        'bp_var': 'static_bp',
        'bp_name': 'static_routes',
        'helpers': [],
        'funcs': ['index', 'admin_page', 'serve_root_files'],
        'header': '''import os
from flask import Blueprint, send_from_directory, current_app, render_template, request
from config import APP_ROOT

static_bp = Blueprint('static_routes', __name__)
'''
    }
]

print("Blueprint definitions configured successfully!")
print(f"Total blueprints: {len(blueprint_defs)}")
