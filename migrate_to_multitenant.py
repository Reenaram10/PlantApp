import os
import sys
import warnings
# pyrefly: ignore[missing-import]
import pyodbc
# pyrefly: ignore[missing-import]
from sqlalchemy import create_engine, text
# pyrefly: ignore[missing-import]
from sqlalchemy.exc import SAWarning

# Force UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore', category=SAWarning, message=r".*Unrecognized server version info.*")

def get_db_uri():
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]

    _db_server = os.getenv("DB_SERVER", "localhost")
    _db_name = os.getenv("DB_NAME", "PlantAppDB")
    _db_trusted = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() == "yes"
    _db_user = os.getenv("DB_USER", "")
    _db_password = os.getenv("DB_PASSWORD", "")

    drivers = pyodbc.drivers()
    if "ODBC Driver 18 for SQL Server" in drivers:
        _driver = "ODBC+Driver+18+for+SQL+Server"
    elif "ODBC Driver 17 for SQL Server" in drivers:
        _driver = "ODBC+Driver+17+for+SQL+Server"
    else:
        _driver = "SQL+Server"

    if _db_trusted:
        return f"mssql+pyodbc://@{_db_server}/{_db_name}?driver={_driver}&TrustServerCertificate=yes&MARS_Connection=yes"
    else:
        from urllib.parse import quote_plus
        return f"mssql+pyodbc://{quote_plus(_db_user)}:{quote_plus(_db_password)}@{_db_server}/{_db_name}?driver={_driver}&TrustServerCertificate=yes&MARS_Connection=yes"

def run_migration():
    db_uri = get_db_uri()
    print("Connecting to database...")
    engine = create_engine(db_uri, pool_pre_ping=True)
    dialect = engine.dialect.name
    print(f"Database engine dialect: {dialect}")

    def exec_sql(sql_str, params=None):
        with engine.begin() as conn:
            result = conn.execute(text(sql_str), params or {})
            if result.returns_rows:
                return result.fetchall()
            return None

    # 1. Create businesses table
    print("Creating 'businesses' table if not exists...")
    if dialect == 'mssql':
        exec_sql("""
            IF OBJECT_ID('dbo.businesses', 'U') IS NULL
            BEGIN
                CREATE TABLE dbo.businesses (
                    business_id VARCHAR(50) PRIMARY KEY,
                    business_name VARCHAR(100) NOT NULL,
                    owner_name VARCHAR(100),
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    address NVARCHAR(MAX),
                    city VARCHAR(50),
                    state VARCHAR(50),
                    country VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'Active',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            END
        """)
    else:
        exec_sql("""
            CREATE TABLE IF NOT EXISTS businesses (
                business_id VARCHAR(50) PRIMARY KEY,
                business_name VARCHAR(100) NOT NULL,
                owner_name VARCHAR(100),
                email VARCHAR(100),
                phone VARCHAR(20),
                address TEXT,
                city VARCHAR(50),
                state VARCHAR(50),
                country VARCHAR(50),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # 2. Create audit_logs table
    print("Creating 'audit_logs' table if not exists...")
    if dialect == 'sqlite':
        exec_sql("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id VARCHAR(50),
                user_id INTEGER,
                user_email VARCHAR(100),
                action VARCHAR(100) NOT NULL,
                module VARCHAR(50) NOT NULL,
                record_id VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            );
        """)
    elif dialect == 'mssql':
        exec_sql("""
            IF OBJECT_ID('dbo.audit_logs', 'U') IS NULL
            BEGIN
                CREATE TABLE dbo.audit_logs (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    business_id VARCHAR(50),
                    user_id INTEGER,
                    user_email VARCHAR(100),
                    action VARCHAR(100) NOT NULL,
                    module VARCHAR(50) NOT NULL,
                    record_id VARCHAR(50),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    details NVARCHAR(MAX)
                );
            END
        """)
    else:
        exec_sql("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                business_id VARCHAR(50),
                user_id INTEGER,
                user_email VARCHAR(100),
                action VARCHAR(100) NOT NULL,
                module VARCHAR(50) NOT NULL,
                record_id VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            );
        """)

    # 3. Seed default nursery NURS-0001
    print("Seeding default nursery 'NURS-0001'...")
    try:
        rows = exec_sql("SELECT business_id FROM businesses WHERE business_id = 'NURS-0001'")
        if not rows:
            exec_sql("""
                INSERT INTO businesses (business_id, business_name, owner_name, email, phone, status)
                VALUES ('NURS-0001', 'Green Valley Nursery', 'Green Valley Admin', 'admin@greenvalley.com', '1234567890', 'Active');
            """)
    except Exception as e:
        print(f"  Nursery seed check notice: {e}")

    # 4. Add business_id, role, username to login table
    print("Updating 'login' table schema...")
    for col_def in [
        ("business_id", "VARCHAR(50) DEFAULT 'NURS-0001'"),
        ("role", "VARCHAR(20) DEFAULT 'NURSERY_ADMIN'"),
        ("username", "VARCHAR(100)")
    ]:
        col_name, col_type = col_def
        try:
            if dialect == 'mssql':
                exec_sql(f"""
                    IF NOT EXISTS (
                        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = 'login' AND COLUMN_NAME = '{col_name}'
                    )
                    BEGIN
                        ALTER TABLE dbo.login ADD {col_name} {col_type};
                    END
                """)
            else:
                exec_sql(f"ALTER TABLE login ADD COLUMN {col_name} {col_type};")
            print(f"  Added/verified column {col_name} in login")
        except Exception as e:
            print(f"  Column {col_name} in login notice: {e}")

    try:
        exec_sql("UPDATE login SET business_id = 'NURS-0001' WHERE business_id IS NULL OR business_id = '';")
        exec_sql("UPDATE login SET role = 'NURSERY_ADMIN' WHERE role IS NULL OR role = '';")
    except Exception as e:
        print(f"  Error updating login defaults: {e}")

    # Seed Super Admin in login table if not present
    try:
        super_admin_rows = exec_sql("SELECT id FROM login WHERE email = 'admin@system.com' OR role = 'SUPER_ADMIN'")
        if not super_admin_rows:
            print("Seeding Super Admin user...")
            exec_sql("""
                INSERT INTO login (email, password, username, role, business_id)
                VALUES ('admin@system.com', 'admin123', 'Super Admin', 'SUPER_ADMIN', 'SYSTEM');
            """)
        else:
            exec_sql("UPDATE login SET role = 'SUPER_ADMIN' WHERE email = 'admin@system.com';")
    except Exception as e:
        print(f"  Super Admin seed notice: {e}")

    # 5. List of tenant tables
    tenant_tables = [
        'plant',
        'category',
        'variety',
        'supplier',
        'inventory_transaction',
        'supplier_purchase',
        'purchase_item',
        'orders',
        'cart_new',
        'user_queries',
        'user_interest',
        'conversation',
        'query',
        'query_result',
        'llm_results',
        'category_synonym'
    ]

    for table in tenant_tables:
        print(f"Migrating table '{table}'...")
        # 5a. Add column if missing
        try:
            if dialect == 'mssql':
                exec_sql(f"""
                    IF NOT EXISTS (
                        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = 'business_id'
                    )
                    BEGIN
                        ALTER TABLE dbo.[{table}] ADD business_id VARCHAR(50) DEFAULT 'NURS-0001';
                    END
                """)
            else:
                exec_sql(f"ALTER TABLE {table} ADD COLUMN business_id VARCHAR(50) DEFAULT 'NURS-0001';")
            print(f"  business_id column verified on {table}")
        except Exception as e:
            print(f"  Notice column business_id on {table}: {e}")

        # 5b. Backfill nulls
        try:
            exec_sql(f"UPDATE dbo.[{table}] SET business_id = 'NURS-0001' WHERE business_id IS NULL OR business_id = '';" if dialect == 'mssql' else f"UPDATE {table} SET business_id = 'NURS-0001' WHERE business_id IS NULL OR business_id = '';")
        except Exception as e:
            print(f"  Notice updating {table}: {e}")

        # 5c. Single-column index for business_id
        idx_name = f"idx_{table}_business_id"
        try:
            if dialect == 'mssql':
                exec_sql(f"""
                    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx_name}' AND object_id = OBJECT_ID('dbo.[{table}]'))
                    BEGIN
                        CREATE INDEX {idx_name} ON dbo.[{table}] (business_id);
                    END
                """)
            else:
                exec_sql(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} (business_id);")
            print(f"  Created index {idx_name}")
        except Exception:
            pass

    # 6. Add discount_price to plant if missing
    try:
        if dialect == 'mssql':
            exec_sql("""
                IF NOT EXISTS (
                    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = 'plant' AND COLUMN_NAME = 'discount_price'
                )
                BEGIN
                    ALTER TABLE dbo.plant ADD discount_price FLOAT DEFAULT 0.0;
                END
            """)
        else:
            exec_sql("ALTER TABLE plant ADD COLUMN IF NOT EXISTS discount_price FLOAT DEFAULT 0.0;")
        print("  discount_price column verified on plant")
    except Exception as e:
        print(f"  Notice discount_price column: {e}")

    # 7. Composite indexes for high-performance per-business queries
    composite_indexes = [
        ("idx_plant_biz_plant",          "plant",                  "business_id, plant_id"),
        ("idx_plant_biz_name",           "plant",                  "business_id, plant_name"),
        ("idx_orders_biz_order",         "orders",                 "business_id, order_id"),
        ("idx_orders_biz_date",          "orders",                 "business_id, order_date"),
        ("idx_orders_biz_user",          "orders",                 "business_id, user_id"),
        ("idx_orders_biz_status",        "orders",                 "business_id, order_status"),
        ("idx_inventory_biz_plant",      "inventory_transaction",  "business_id, plant_id"),
        ("idx_inventory_biz_date",       "inventory_transaction",  "business_id, created_at"),
        ("idx_supplier_biz",             "supplier",               "business_id, supplier_id"),
        ("idx_supplier_purchase_biz",    "supplier_purchase",      "business_id, supplier_id"),
        ("idx_purchase_item_biz",        "purchase_item",          "business_id, purchase_id"),
        ("idx_cart_biz_user",            "cart_new",               "business_id, user_id"),
        ("idx_user_queries_biz_user",    "user_queries",           "business_id, user_id"),
        ("idx_user_interest_biz_user",   "user_interest",          "business_id, user_id"),
        ("idx_conversation_biz_user",    "conversation",           "business_id, user_id"),
        ("idx_category_synonym_biz",     "category_synonym",       "business_id, category_id"),
        ("idx_llm_results_biz",          "llm_results",            "business_id, created_at"),
        ("idx_audit_logs_biz",           "audit_logs",             "business_id, timestamp"),
    ]
    print("\nCreating composite indexes for per-business query performance...")
    for idx_name, table_name, cols in composite_indexes:
        try:
            if dialect == 'mssql':
                exec_sql(f"""
                    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx_name}' AND object_id = OBJECT_ID('dbo.[{table_name}]'))
                    BEGIN
                        CREATE INDEX {idx_name} ON dbo.[{table_name}] ({cols});
                    END
                """)
            else:
                exec_sql(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({cols});")
            print(f"  [OK] {idx_name} ON {table_name}({cols})")
        except Exception as e:
            print(f"  Notice {idx_name}: {e}")

    print("\n[OK] Multi-tenant schema migration & data seeding completed successfully!")

if __name__ == '__main__':
    run_migration()
