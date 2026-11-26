import os
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
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.tenant import User, Business, utc_now
from auth.jwt_utils import create_token, generate_jwt
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
from sqlalchemy import text
from extensions import db
from models.tenant import User, Business, AuditLog, utc_now
from models.catalog import Plant
from models.commerce import Order
from auth.jwt_utils import create_token, generate_jwt
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
import traceback
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from sqlalchemy import text
from extensions import db
from config import APP_ROOT
from models.catalog import Plant, Variety
from models.inventory import InventoryTransaction, Supplier
from auth.decorators import require_tenant_auth, get_current_tenant
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
import random
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from extensions import db
from models.commerce import Order
from models.tenant import User
from models.inventory import Supplier, SupplierPurchase, PurchaseItem, InventoryTransaction, utc_now
from models.catalog import Plant
from auth.decorators import require_tenant_auth, get_current_tenant
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
import io
import json
import traceback
from datetime import datetime
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, send_file, Response, make_response
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
from services.image_utils import get_plant_image_url

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
from auth.decorators import get_current_tenant
from services.image_utils import get_plant_image_url
from services.observability import log_user_interest

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
import time
import random
import uuid
from typing import Any
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, render_template
from sqlalchemy import text
from extensions import db
from models.commerce import Order, Cart, utc_now
from models.catalog import Plant
from models.tenant import User
from auth.decorators import get_current_tenant
from services.payment import decrypt
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
from sqlalchemy import text
from extensions import db, langfuse_client
from models.chat import UserQuery, Query, QueryResult
from models.tenant import User
from auth.decorators import get_current_tenant
from services.chat.chat_service import process_message
from services.chat.dialog_state import USER_DIALOG_STATE
from services.llm_service import normalize_query
from services.speech_service import generate_speech
from services.rag_service import history_faiss
from services.observability import store_query

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
from typing import Any
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from extensions import db
from models.commerce import Cart, Order
from models.catalog import Plant
from models.tenant import User
from auth.decorators import get_current_tenant
from routes.order_routes import checkout, build_order_timeline
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
import time
import base64
import requests
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app
from config import APP_ROOT, SCIENTIFIC_NAMES_MAP
from extensions import db
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.image_utils import allowed_file
from services.rag_service import tomato_rag, plant_rag

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

os.makedirs('routes', exist_ok=True)

def extract_function_code(func_name, bp_var):
    node = funcs_by_name[func_name]
    start_line = min([d.lineno for d in node.decorator_list]) if node.decorator_list else node.lineno
    end_line = node.end_lineno
    
    # Slice the lines (0-indexed)
    fn_lines = app_lines[start_line - 1 : end_line]
    code = ''.join(fn_lines)
    
    # 1. Transform decorators: @app.route -> @{bp_var}.route
    code = re.sub(r'@app\.route\(', f'@{bp_var}.route(', code)
    
    # 2. Transform app.<property/method> -> current_app.<property/method>
    code = re.sub(r'\bapp\.logger\b', 'current_app.logger', code)
    code = re.sub(r'\bapp\.config\b', 'current_app.config', code)
    code = re.sub(r'\bapp\.root_path\b', 'current_app.root_path', code)
    code = re.sub(r'\bapp\.app_context\(\)', 'current_app.app_context()', code)
    
    # 3. Path rule: replace moved module __file__ references if any
    code = code.replace("os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')", "os.path.join(APP_ROOT, 'static')")
    code = code.replace('os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")', "os.path.join(APP_ROOT, 'static')")
    
    # In admin_profiles_routes.py:
    # replace base_dir = os.path.dirname(os.path.abspath(__file__)) with base_dir = APP_ROOT
    code = code.replace("base_dir = os.path.dirname(os.path.abspath(__file__))", "base_dir = APP_ROOT")
    
    # In config_routes.py:
    # replace STT_OPTION and STT_FAILSAFE with runtime_settings equivalents
    if bp_var == 'config_bp':
        code = re.sub(r'\bSTT_OPTION\b', 'runtime_settings.stt_option', code)
        code = re.sub(r'\bSTT_FAILSAFE\b', 'runtime_settings.stt_failsafe', code)
        
    return code

for bp in blueprint_defs:
    file_name = str(bp['file'])
    bp_var = str(bp['bp_var'])
    file_path = os.path.join('routes', file_name)
    content_parts = [str(bp['header']).strip(), '']
    
    extra_top = str(bp.get('extra_top', '')).strip()
    if extra_top:
        content_parts.append(extra_top)
        content_parts.append('')
        
    helpers = bp.get('helpers', [])
    if isinstance(helpers, list):
        for helper_name in helpers:
            h_code = extract_function_code(str(helper_name), bp_var)
            content_parts.append(h_code.strip())
            content_parts.append('')
        
    funcs = bp.get('funcs', [])
    if isinstance(funcs, list):
        for func_name in funcs:
            f_code = extract_function_code(str(func_name), bp_var)
            content_parts.append(f_code.strip())
            content_parts.append('')
        
    final_content = '\n'.join(content_parts) + '\n'
    
    with open(file_path, 'w', encoding='utf-8') as out_f:
        out_f.write(final_content)
        
    print(f"Generated {file_path} with {len(bp['funcs'])} routes and {len(bp['helpers'])} helpers.")

print("All 14 blueprints successfully generated!")
