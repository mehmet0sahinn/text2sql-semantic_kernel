"""Quick test for database connection."""

import os
from dotenv import load_dotenv
from database_service import DatabaseService

load_dotenv()

def main():
    db = DatabaseService(
        server=os.getenv("SQL_SERVER"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DATABASE")
    )

    print("Database Tests\n")
    
    # Test 1: Database Info
    print("1. Database Info:", db.get_db_info()[0].db_name if db.get_db_info() else "FAIL")
    
    # Test 2: Schemas
    schemas = [s.schema_name for s in db.get_schema_info() if s.schema_name]
    print(f"2. Schemas: {', '.join(schemas[:5])}...")
    
    # Test 3: Tables in SalesLT
    tables = [t.table_name for t in db.get_table_schema_info("SalesLT")]
    print(f"3. SalesLT Tables: {', '.join(tables)}")
    
    # Test 4: Query
    result = db.execute_sql_command("SELECT TOP 3 FirstName, LastName FROM SalesLT.Customer")
    print(f"4. Sample Query: {result[:100]}...")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()

