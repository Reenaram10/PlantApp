import pyodbc

bak = r"D:\Microsoft\MSSQL17.MSSQLSERVER\MSSQL\Backup\PlantAppDB_before_security_20261125.bak"
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=master;Trusted_Connection=yes;TrustServerCertificate=yes",
    autocommit=True,
)
cur = conn.cursor()
files = cur.execute(f"RESTORE FILELISTONLY FROM DISK = N'{bak}'").fetchall()
data_name = files[0][0]
log_name = files[1][0]
print("Logical names:", data_name, log_name)

cur.execute(
    f"RESTORE DATABASE [PlantAppDB_test] FROM DISK = N'{bak}' "
    f"WITH MOVE N'{data_name}' TO N'D:\\Microsoft\\MSSQL17.MSSQLSERVER\\MSSQL\\DATA\\PlantAppDB_test.mdf', "
    f"MOVE N'{log_name}' TO N'D:\\Microsoft\\MSSQL17.MSSQLSERVER\\MSSQL\\DATA\\PlantAppDB_test_log.ldf', "
    f"REPLACE"
)
while cur.nextset():
    pass

dbs = [r[0] for r in cur.execute("SELECT name FROM sys.databases WHERE name LIKE 'PlantApp%'").fetchall()]
print("Databases now:", dbs)
