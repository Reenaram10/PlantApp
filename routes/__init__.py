from flask import Flask

from routes.auth_routes import auth_bp
from routes.super_admin_routes import super_admin_bp
from routes.admin_inventory_routes import admin_inventory_bp
from routes.admin_orders_routes import admin_orders_bp
from routes.admin_profiles_routes import admin_profiles_bp
from routes.category_routes import category_bp
from routes.shop_routes import shop_bp
from routes.cart_routes import cart_bp
from routes.order_routes import order_bp
from routes.chat_routes import chat_bp
from routes.whatsapp_routes import whatsapp_bp
from routes.diagnosis_routes import diagnosis_bp
from routes.config_routes import config_bp
from routes.static_routes import static_bp

def register_routes(app: Flask) -> None:
    """Register all 14 application blueprints.
    Note: Blueprints have NO url_prefix.
    static_routes (with the /<path:filename> catch-all) is registered LAST.
    """
    app.register_blueprint(auth_bp)
    app.register_blueprint(super_admin_bp)
    app.register_blueprint(admin_inventory_bp)
    app.register_blueprint(admin_orders_bp)
    app.register_blueprint(admin_profiles_bp)
    app.register_blueprint(category_bp)
    app.register_blueprint(shop_bp)
    app.register_blueprint(cart_bp)
    app.register_blueprint(order_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(whatsapp_bp)
    app.register_blueprint(diagnosis_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(static_bp)  # MUST BE REGISTERED LAST
