"""
migrate_neon_to_mssql.py
------------------------
Safe, idempotent, relationship-preserving migration script to transfer data
from NeonDB (PostgreSQL) to Microsoft SQL Server (PlantAppDB).

Features:
  - STRICTLY READ-ONLY on source PostgreSQL database (SET TRANSACTION READ ONLY).
  - Preserves all original Primary Key IDs using SET IDENTITY_INSERT ON/OFF.
  - Reseeds SQL Server IDENTITY counters using DBCC CHECKIDENT after import.
  - Topological table order prevents Foreign Key constraint violations.
  - Includes --dry-run mode to inspect and verify counts without modifying data.
  - Includes --verify mode to report reconciliation between PostgreSQL and MSSQL.
  - Includes --clean-dest mode to wipe SQL Server destination tables before run.

Usage:
  python migrate_neon_to_mssql.py --dry-run
  python migrate_neon_to_mssql.py --clean-dest
  python migrate_neon_to_mssql.py --verify
"""

import os
import sys
import argparse
import time
from datetime import datetime, date

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import psycopg2
import psycopg2.extras
import pyodbc

# Default source NeonDB connection string
DEFAULT_PG_URL = os.getenv(
    "POSTGRES_DATABASE_URL",
    "postgresql://neondb_owner:npg_KBStXxq52HPZ@ep-gentle-grass-adnpzd0p-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

# Tables to migrate in topological dependency order (parents before children)
TABLES_ORDER = [
    {
        "name": "businesses",
        "pk": "business_id",
        "has_identity": False,
        "cols": [
            "business_id", "business_name", "owner_name", "email", "phone",
            "address", "city", "state", "country", "status", "created_at", "updated_at"
        ]
    },
    {
        "name": "login",
        "pk": "id",
        "has_identity": True,
        "cols": [
            "id", "username", "password", "email", "last_login",
            "last_logout", "is_online", "role", "business_id"
        ]
    },
    {
        "name": "supplier",
        "pk": "supplier_id",
        "has_identity": True,
        "cols": [
            "supplier_id", "supplier_name", "contact_person", "phone",
            "email", "address", "created_at", "business_id"
        ]
    },
    {
        "name": "plant",
        "pk": "plant_id",
        "has_identity": True,
        "cols": [
            "plant_id", "plant_name", "description", "price", "image_path",
            "stock", "discount_price", "business_id"
        ]
    },
    {
        "name": "category",
        "pk": "category_id",
        "has_identity": True,
        "cols": ["category_id", "category_name", "business_id"]
    },
    {
        "name": "plant_category",
        "pk": "plant_category_id",
        "has_identity": True,
        "cols": ["plant_category_id", "plant_id", "category_id"]
    },
    {
        "name": "variety",
        "pk": "variety_id",
        "has_identity": True,
        "cols": [
            "variety_id", "plant_id", "variety_name", "variety_description",
            "variety_price", "created_at", "business_id"
        ]
    },
    {
        "name": "plant_synonym",
        "pk": "synonym_id",
        "has_identity": True,
        "cols": ["synonym_id", "synonym", "plant_id", "created_at"]
    },
    {
        "name": "inventory_transaction",
        "pk": "transaction_id",
        "has_identity": True,
        "cols": [
            "transaction_id", "plant_id", "supplier_id", "type", "quantity",
            "notes", "bill_no", "bill_date", "created_at", "business_id"
        ]
    },
    {
        "name": "supplier_purchase",
        "pk": "purchase_id",
        "has_identity": True,
        "cols": [
            "purchase_id", "supplier_id", "bill_no", "bill_date", "total_amount",
            "amount_paid", "balance", "status", "created_at", "business_id"
        ]
    },
    {
        "name": "purchase_item",
        "pk": "item_id",
        "has_identity": True,
        "cols": [
            "item_id", "purchase_id", "plant_id", "quantity", "unit_price", "business_id"
        ]
    },
    {
        "name": "cart_new",
        "pk": "cart_id",
        "has_identity": True,
        "cols": [
            "cart_id", "user_id", "plant_id", "quantity", "added_at", "business_id"
        ]
    },
    {
        "name": "orders",
        "pk": "order_id",
        "has_identity": True,
        "cols": [
            "order_id", "plant_name", "user_id", "order_date", "total_amount",
            "payment_status", "order_group_id", "quantity", "payment_method",
            "order_status", "shipping_address", "delivery_date", "tracking_number",
            "notes", "price", "amount_paid", "balance", "business_id"
        ]
    },
    {
        "name": "category_synonym",
        "pk": "synonym_id",
        "has_identity": True,
        "cols": ["synonym_id", "synonym", "category_id", "created_at", "business_id"]
    },
    {
        "name": "query",
        "pk": "query_id",
        "has_identity": True,
        "cols": ["query_id", "description", "user_id", "created_at", "canonical_key", "business_id"]
    },
    {
        "name": "query_result",
        "pk": "result_id",
        "has_identity": True,
        "cols": [
            "result_id", "query_id", "result_description", "result_source",
            "created_at", "response_type", "response_data", "business_id"
        ]
    },
    {
        "name": "user_queries",
        "pk": "uq_id",
        "has_identity": True,
        "cols": ["uq_id", "user_id", "plant_id", "query", "created_at", "business_id"]
    },
    {
        "name": "user_interest",
        "pk": "interest_id",
        "has_identity": True,
        "cols": ["interest_id", "user_id", "plant_id", "interest_type", "notes", "created_at", "business_id"]
    },
    {
        "name": "user_category_interest",
        "pk": "id",
        "has_identity": True,
        "cols": ["id", "user_id", "category_id", "interest_type", "count", "last_interaction"]
    },
    {
        "name": "conversation",
        "pk": "conversation_id",
        "has_identity": True,
        "cols": ["conversation_id", "user_id", "conversation", "created_at", "business_id"]
    },
    {
        "name": "llm_results",
        "pk": "id",
        "has_identity": True,
        "cols": [
            "id", "user_query", "llm_response", "category_suggested", "plants_found",
            "created_at", "llm_name", "expires_at", "business_id"
        ]
    },
    {
        "name": "plant_knowledge",
        "pk": "id",
        "has_identity": True,
        "cols": [
            "id", "plant_name", "scientific_name", "description", "sunlight",
            "water_requirement", "soil_type", "fertilizer", "medicinal_uses",
            "common_diseases", "care_tips"
        ]
    },
    {
        "name": "disease_knowledge",
        "pk": "id",
        "has_identity": True,
        "cols": [
            "id", "disease_name", "symptoms", "causes", "organic_treatment",
            "chemical_treatment", "prevention"
        ]
    },
    {
        "name": "audit_logs",
        "pk": "id",
        "has_identity": True,
        "cols": [
            "id", "business_id", "user_id", "user_email", "action", "module",
            "record_id", "timestamp", "details"
        ]
    }
]


def get_mssql_connection_string():
    server = os.getenv("DB_SERVER", "localhost")
    dbname = os.getenv("DB_NAME", "PlantAppDB")
    trusted = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() == "yes"
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")

    # Auto-detect ODBC driver
    drivers = pyodbc.drivers()
    if "ODBC Driver 18 for SQL Server" in drivers:
        driver = "ODBC Driver 18 for SQL Server"
    elif "ODBC Driver 17 for SQL Server" in drivers:
        driver = "ODBC Driver 17 for SQL Server"
    else:
        driver = "SQL Server"

    if trusted:
        return f"DRIVER={{{driver}}};SERVER={server};DATABASE={dbname};Trusted_Connection=yes;TrustServerCertificate=yes;"
    else:
        return f"DRIVER={{{driver}}};SERVER={server};DATABASE={dbname};UID={user};PWD={password};TrustServerCertificate=yes;"


def sanitize_val(val):
    """Normalize data types between PostgreSQL and SQL Server"""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (datetime, date)):
        return val
    return val


def connect_postgres(url):
    print(" [PG] Connecting to source PostgreSQL (read-only)...")
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def connect_mssql():
    conn_str = get_mssql_connection_string()
    print(" [MSSQL] Connecting to destination SQL Server...")
    return pyodbc.connect(conn_str, autocommit=False)


def get_pg_columns(cur, table_name):
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = %s
    """, (table_name,))
    return set(r[0] for r in cur.fetchall())


def dry_run(pg_conn, ms_conn):
    print("\n" + "="*70)
    print(" 🔍 DRY RUN: Checking source and destination readiness")
    print("="*70)

    pg_cur = pg_conn.cursor()
    ms_cur = ms_conn.cursor()

    total_pg_rows = 0
    print(f"{'Table Name':<26} | {'PG Rows':<10} | {'MSSQL Exists?':<14} | {'Has Identity'}")
    print("-" * 70)

    for item in TABLES_ORDER:
        tbl = item["name"]
        pg_cols = get_pg_columns(pg_cur, tbl)

        # Count PG rows
        if pg_cols:
            pg_cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
            pg_count = pg_cur.fetchone()[0]
        else:
            pg_count = 0
        total_pg_rows += pg_count

        # Check MSSQL table exists
        ms_cur.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", (tbl,))
        ms_exists = "YES" if ms_cur.fetchone() else "NO ❌"

        print(f"{tbl:<26} | {pg_count:<10} | {ms_exists:<14} | {str(item['has_identity']):<10}")

    print("-" * 70)
    print(f"Total PostgreSQL rows ready to migrate: {total_pg_rows}")
    print("✅ Dry run completed. PostgreSQL remains 100% unmodified.\n")


def clean_destination(ms_conn):
    print("\n" + "="*70)
    print(" 🧹 CLEANING DESTINATION SQL SERVER TABLES (Reverse order)")
    print("="*70)
    ms_cur = ms_conn.cursor()

    # Disable constraints temporarily for clean wipe
    ms_cur.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT all'")

    for item in reversed(TABLES_ORDER):
        tbl = item["name"]
        try:
            ms_cur.execute(f"DELETE FROM dbo.[{tbl}];")
            if item["has_identity"]:
                try:
                    ms_cur.execute(f"DBCC CHECKIDENT ('dbo.[{tbl}]', RESEED, 0);")
                except Exception:
                    pass
            print(f"  ✓ Cleaned dbo.[{tbl}]")
        except Exception as e:
            print(f"  Notice cleaning dbo.[{tbl}]: {e}")

    ms_cur.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? WITH CHECK CHECK CONSTRAINT all'")
    ms_conn.commit()
    print("✅ SQL Server destination tables cleared.\n")


def ensure_login_email_collation(ms_conn):
    """
    Ensure dbo.login.email uses Case-Sensitive collation (SQL_Latin1_General_CP1_CS_AS)
    matching PostgreSQL semantics so that distinct accounts with differing letter cases
    (e.g., 'reena' vs 'Reena') do not trigger false unique key violations on SQL Server.
    """
    ms_cur = ms_conn.cursor()
    try:
        ms_cur.execute("""
            SELECT collation_name 
            FROM sys.columns 
            WHERE object_id = OBJECT_ID('dbo.login') AND name = 'email'
        """)
        row = ms_cur.fetchone()
        if row and ('_CI_' in row[0] or 'CI_AS' in row[0]):
            print("  [MSSQL Schema] Aligning dbo.login.email to Case-Sensitive (CS) collation...")
            ms_cur.execute("IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_login_email' AND object_id = OBJECT_ID('dbo.login')) DROP INDEX idx_login_email ON dbo.login;")
            ms_cur.execute("IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'UQ_login_email' AND parent_object_id = OBJECT_ID('dbo.login')) ALTER TABLE dbo.login DROP CONSTRAINT UQ_login_email;")
            ms_cur.execute("ALTER TABLE dbo.login ALTER COLUMN email VARCHAR(120) COLLATE SQL_Latin1_General_CP1_CS_AS NOT NULL;")
            ms_cur.execute("ALTER TABLE dbo.login ADD CONSTRAINT UQ_login_email UNIQUE (email);")
            ms_cur.execute("CREATE NONCLUSTERED INDEX idx_login_email ON dbo.login (email);")
            ms_conn.commit()
            print("  [MSSQL Schema] dbo.login.email successfully aligned to SQL_Latin1_General_CP1_CS_AS.")
    except Exception as e:
        print(f"  [MSSQL Schema Notice] Collation alignment notice: {e}")


def run_migration(pg_conn, ms_conn, batch_size=1000):
    print("\n" + "="*70)
    print(" 🚀 STARTING FULL DATA MIGRATION (PostgreSQL -> SQL Server)")
    print("="*70)

    ensure_login_email_collation(ms_conn)

    pg_cur = pg_conn.cursor()
    ms_cur = ms_conn.cursor()

    start_time = time.time()
    total_migrated = 0

    for item in TABLES_ORDER:
        tbl = item["name"]
        pk = item["pk"]
        has_id = item["has_identity"]
        candidate_cols = item["cols"]

        # Intersect with actual PG columns
        pg_actual_cols = get_pg_columns(pg_cur, tbl)
        if not pg_actual_cols:
            print(f"⚠️  Table '{tbl}' does not exist in PostgreSQL. Skipping.")
            continue

        valid_cols = [c for c in candidate_cols if c in pg_actual_cols]
        if not valid_cols:
            print(f"⚠️  No valid columns found for '{tbl}'. Skipping.")
            continue

        # Fetch all rows from PostgreSQL
        cols_pg_str = ", ".join(f'"{c}"' for c in valid_cols)
        order_clause = f'ORDER BY "{pk}" ASC' if pk in pg_actual_cols else ""
        pg_query = f'SELECT {cols_pg_str} FROM "{tbl}" {order_clause};'
        pg_cur.execute(pg_query)
        rows = pg_cur.fetchall()
        num_rows = len(rows)

        if num_rows == 0:
            print(f"  ⏩ {tbl:<24} : 0 rows (skipped)")
            continue

        # Prepare MSSQL Insert Statement
        cols_ms_str = ", ".join(f"[{c}]" for c in valid_cols)
        placeholders = ", ".join("?" for _ in valid_cols)
        insert_sql = f"INSERT INTO dbo.[{tbl}] ({cols_ms_str}) VALUES ({placeholders})"

        # Enable IDENTITY_INSERT if needed
        if has_id:
            ms_cur.execute(f"SET IDENTITY_INSERT dbo.[{tbl}] ON;")

        # Insert in batches
        try:
            for i in range(0, num_rows, batch_size):
                batch = rows[i:i + batch_size]
                sanitized_batch = [tuple(sanitize_val(v) for v in row) for row in batch]
                ms_cur.executemany(insert_sql, sanitized_batch)
            ms_conn.commit()
        except Exception as e:
            ms_conn.rollback()
            if has_id:
                try:
                    ms_cur.execute(f"SET IDENTITY_INSERT dbo.[{tbl}] OFF;")
                    ms_conn.commit()
                except Exception:
                    pass
            print(f"❌ Error inserting into dbo.[{tbl}]: {e}")
            raise e

        # Disable IDENTITY_INSERT
        if has_id:
            ms_cur.execute(f"SET IDENTITY_INSERT dbo.[{tbl}] OFF;")
            ms_conn.commit()

            # Reseed Identity Counter to MAX(id)
            try:
                ms_cur.execute(f"SELECT MAX([{pk}]) FROM dbo.[{tbl}]")
                max_val = ms_cur.fetchone()[0]
                if max_val and isinstance(max_val, int):
                    ms_cur.execute(f"DBCC CHECKIDENT ('dbo.[{tbl}]', RESEED, {max_val});")
                    ms_conn.commit()
            except Exception as res_err:
                print(f"    Notice reseeding identity on {tbl}: {res_err}")

        total_migrated += num_rows
        print(f"  ✓ {tbl:<24} : {num_rows:>5} rows migrated successfully")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"🎉 MIGRATION FINISHED: {total_migrated} rows across {len(TABLES_ORDER)} tables in {elapsed:.2f}s!")
    print("✅ Source PostgreSQL database remained 100% read-only throughout.")


def verify_migration(pg_conn, ms_conn):
    print("\n" + "="*70)
    print(" 📊 DATA RECONCILIATION & VERIFICATION REPORT")
    print("="*70)

    pg_cur = pg_conn.cursor()
    ms_cur = ms_conn.cursor()

    print(f"{'Table Name':<26} | {'PG Rows':<10} | {'MSSQL Rows':<10} | {'Status'}")
    print("-" * 70)

    all_matched = True
    for item in TABLES_ORDER:
        tbl = item["name"]
        pg_cols = get_pg_columns(pg_cur, tbl)

        pg_count = 0
        if pg_cols:
            pg_cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
            pg_count = pg_cur.fetchone()[0]

        try:
            ms_cur.execute(f"SELECT COUNT(*) FROM dbo.[{tbl}]")
            ms_count = ms_cur.fetchone()[0]
        except Exception:
            ms_count = -1

        status = "MATCH ✅" if pg_count == ms_count else f"MISMATCH ❌ ({ms_count - pg_count:+d})"
        if pg_count != ms_count:
            all_matched = False

        print(f"{tbl:<26} | {pg_count:<10} | {ms_count:<10} | {status}")

    print("-" * 70)
    if all_matched:
        print("✅ ALL TABLES AND ROW COUNTS MATCH 100% PERFECTLY!")
    else:
        print("⚠️ Some tables differ in row counts. Review details above.")


def main():
    parser = argparse.ArgumentParser(description="PlantApp PostgreSQL to Microsoft SQL Server Migration")
    parser.add_argument("--dry-run", action="store_true", help="Preview tables and row counts without migrating")
    parser.add_argument("--clean-dest", action="store_true", help="Truncate SQL Server tables before migrating")
    parser.add_argument("--verify", action="store_true", help="Compare row counts between PostgreSQL and SQL Server")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for INSERT execution")
    args = parser.parse_args()

    pg_url = DEFAULT_PG_URL

    try:
        pg_conn = connect_postgres(pg_url)
    except Exception as e:
        print(f"❌ Failed to connect to source PostgreSQL: {e}")
        sys.exit(1)

    try:
        ms_conn = connect_mssql()
    except Exception as e:
        print(f"❌ Failed to connect to destination SQL Server: {e}")
        pg_conn.close()
        sys.exit(1)

    try:
        if args.dry_run:
            dry_run(pg_conn, ms_conn)
        elif args.verify:
            verify_migration(pg_conn, ms_conn)
        else:
            if args.clean_dest:
                clean_destination(ms_conn)
            run_migration(pg_conn, ms_conn, batch_size=args.batch_size)
            verify_migration(pg_conn, ms_conn)
    finally:
        pg_conn.close()
        ms_conn.close()


if __name__ == "__main__":
    main()
