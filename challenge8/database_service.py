"""Database Service for SQL Server operations."""

import json
import pyodbc
from typing import List, Optional
from models import DbInfo, SchemaInfo, TableInfo, ColumnsInfo


class DatabaseService:
    """Service class for SQL Server database operations."""

    def __init__(self, server: str, user: str, password: str, database: str):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={server};DATABASE={database};"
            f"UID={user};PWD={password};TrustServerCertificate=yes;"
        )

    def _get_connection(self) -> pyodbc.Connection:
        return pyodbc.connect(self.connection_string)

    def get_db_info(self) -> List[DbInfo]:
        """Get database name and description."""
        sql = """
            SELECT DISTINCT CATALOG_NAME as Db, s.value as Description
            FROM INFORMATION_SCHEMA.SCHEMATA
            LEFT JOIN sys.extended_properties s ON s.major_id = 0 AND s.name = 'MS_Description'
        """
        with self._get_connection() as conn:
            rows = conn.cursor().execute(sql).fetchall()
            return [DbInfo(db_name=r.Db, db_description=r.Description) for r in rows]

    def get_schema_info(self) -> List[SchemaInfo]:
        """Get all schemas with descriptions."""
        sql = """
            SELECT CATALOG_NAME as Db, SCHEMA_NAME as [Schema], s.value as Description
            FROM INFORMATION_SCHEMA.SCHEMATA
            LEFT JOIN sys.extended_properties s ON s.major_id = SCHEMA_ID(SCHEMA_NAME) AND s.name = 'MS_Description'
            ORDER BY SCHEMA_NAME
        """
        with self._get_connection() as conn:
            rows = conn.cursor().execute(sql).fetchall()
            return [SchemaInfo(db_name=r.Db, schema_name=r.Schema, schema_description=r.Description) for r in rows]

    def get_table_schema_info(self, schema_name: Optional[str] = None) -> List[TableInfo]:
        """Get tables with descriptions, optionally filtered by schema."""
        sql = """
            SELECT TABLE_CATALOG as Db, TABLE_SCHEMA as [Schema], TABLE_NAME as [Table], s.value as Description
            FROM INFORMATION_SCHEMA.TABLES
            LEFT JOIN sys.extended_properties s 
                ON s.major_id = OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME) AND s.name = 'MS_Description' AND s.minor_id = 0
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        with self._get_connection() as conn:
            rows = conn.cursor().execute(sql).fetchall()
            tables = [TableInfo(db_name=r.Db, schema_name=r.Schema, table_name=r.Table, table_description=r.Description) for r in rows]
            return [t for t in tables if t.schema_name == schema_name] if schema_name else tables

    def get_column_schema_info(self, schema_name: Optional[str] = None, table_name: Optional[str] = None) -> List[ColumnsInfo]:
        """Get columns with data types and descriptions."""
        sql = """
            SELECT TABLE_CATALOG as Db, TABLE_SCHEMA as [Schema], TABLE_NAME as [Table], 
                   COLUMN_NAME as [Column], DATA_TYPE as DataType, s.value as Description
            FROM INFORMATION_SCHEMA.COLUMNS
            LEFT JOIN sys.extended_properties s 
                ON s.major_id = OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME) AND s.minor_id = ORDINAL_POSITION AND s.name = 'MS_Description'
            WHERE OBJECTPROPERTY(OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME), 'IsMsShipped') = 0
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        with self._get_connection() as conn:
            rows = conn.cursor().execute(sql).fetchall()
            columns = [ColumnsInfo(db_name=r.Db, schema_name=r.Schema, table_name=r.Table, 
                                   column_name=r.Column, column_data_type=r.DataType, column_description=r.Description) for r in rows]
            if schema_name:
                columns = [c for c in columns if c.schema_name == schema_name]
            if table_name:
                columns = [c for c in columns if c.table_name == table_name]
            return columns

    def execute_sql_command(self, sql_command: str) -> str:
        """Execute SQL query and return results as JSON."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_command)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            result = []
            for row in rows:
                row_dict = {columns[i]: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) 
                           for i, v in enumerate(row)}
                result.append(row_dict)
            return json.dumps(result, ensure_ascii=False, indent=2)

