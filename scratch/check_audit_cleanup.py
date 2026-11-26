import pyodbc

conn = pyodbc.connect('Driver={ODBC Driver 18 for SQL Server};Server=localhost;Database=PlantAppDB;Trusted_Connection=yes;TrustServerCertificate=yes')
cur = conn.cursor()

tables = ['inventory_transaction', 'audit_logs', 'user_interest', 'user_queries', 'query', 'conversation', 'llm_results']

print('=== AUDIT OF TABLES FOR ROWS FROM PART C RUN (2026-11-24) ===')
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{t}'")
    if cur.fetchone()[0] == 0:
        print(f"Table [{t}]: DOES NOT EXIST")
        continue
    
    cur.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{t}'")
    col_info = cur.fetchall()
    cols = [c[0] for c in col_info]
    
    time_col = None
    for c in cols:
        if 'date' in c.lower() or 'time' in c.lower() or 'created' in c.lower():
            time_col = c
            break
            
    where_clauses = []
    if time_col:
        where_clauses.append(f"{time_col} >= '2026-11-24'")
        
    for ci in col_info:
        c, dtype = ci[0], ci[1]
        if dtype in ['varchar', 'nvarchar', 'text', 'ntext', 'char']:
            where_clauses.append(f"{c} LIKE '%TEST_%' OR {c} LIKE '%test_%'")
            
    sql = f"SELECT * FROM {t}"
    if where_clauses:
        sql += " WHERE " + " OR ".join(where_clauses)
    sql += " ORDER BY 1 DESC"
    
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        print(f"\nTable [{t}]: {len(rows)} matching rows found.")
        for r in rows:
            print("  ->", r)
    except Exception as e:
        print(f"Error querying [{t}]: {e}")
