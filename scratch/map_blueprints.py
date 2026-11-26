import ast

with open('app.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

funcs = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        route_decs = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'route':
                route_decs.append(dec)
        if route_decs:
            funcs.append((node.name, node.lineno, node.end_lineno, [ast.unparse(d) for d in route_decs]))

print(f"Total route functions: {len(funcs)}")

blueprint_mapping = {
    'auth_bp': ['register', 'login', 'logout', 'update_user_phone'],
    'super_admin_bp': ['super_admin_login', 'register_new_nursery', 'list_nurseries_super_admin', 'delete_nursery_super_admin', 'toggle_nursery_status', 'super_admin_stats', 'get_super_admin_audit_logs'],
    'admin_inventory_bp': ['suggest_varieties', 'add_varieties', 'get_existing_varieties_admin', 'get_inventory', 'restock_plant', 'get_plant_details', 'add_plant', 'update_plant_details', 'delete_plant_endpoint', 'search_plants_admin', 'update_plant_image_admin', 'inventory_add', 'inventory_remove', 'get_inventory_transactions', 'manage_plant_image', 'list_plant_images'],
    'admin_orders_bp': ['get_customers_admin', 'get_all_orders_admin', 'record_order_payment', 'get_order_details_admin', 'update_order_status_admin', 'add_supplier_purchase', 'get_supplier_purchases', 'record_purchase_payment', 'get_suppliers', 'add_supplier'],
    'admin_profiles_bp': ['sync_rag_index', 'get_profile_stats', 'upload_profile_dataset', 'download_profile_template'],
    'category_bp': ['get_categories_admin', 'add_category_admin', 'delete_category_admin', 'get_category_plants_admin', 'manage_category_synonyms'],
    'shop_bp': ['get_plants', 'get_shop_plants', 'suggest_plants', 'save_user_interest'],
    'cart_bp': ['add_to_cart', 'get_cart', 'update_cart', 'remove_from_cart'],
    'order_bp': ['checkout', 'payment_response', 'get_orders', 'get_order_details', 'track_order'],
    'chat_bp': ['chat', 'save_user_query', 'get_user_history', 'get_history', 'manage_queries', 'flask_history_query', 'flask_history_profile', 'flask_history_variable', 'flask_history_analysis', 'manage_query_results'],
    'whatsapp_bp': ['whatsapp_get_cart', 'whatsapp_cart_approve', 'whatsapp_cart_update', 'whatsapp_checkout', 'whatsapp_get_orders', 'whatsapp_track_order'],
    'diagnosis_bp': ['api_detect_tomato_disease', 'n8n_plant_identify', 'n8n_disease_identify', 'n8n_webhook', 'api_plant_diagnosis'],
    'config_bp': ['manage_config', 'stt_api', 'health_check'],
    'static_bp': ['index', 'admin_page', 'serve_root_files'],
}

all_mapped = []
for bp, fns in blueprint_mapping.items():
    all_mapped.extend(fns)

found_names = [f[0] for f in funcs]

print(f"Total mapped functions in 14 blueprints: {len(all_mapped)}")
print(f"Total found functions in app.py: {len(found_names)}")

unmapped = set(found_names) - set(all_mapped)
print(f"Unmapped functions in app.py: {unmapped}")
missing_from_app = set(all_mapped) - set(found_names)
print(f"Mapped but missing in app.py: {missing_from_app}")
