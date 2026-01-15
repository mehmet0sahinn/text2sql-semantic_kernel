"""Semantic Kernel Plugin for Database Operations."""

from typing import Annotated
from semantic_kernel.functions import kernel_function
from database_service import DatabaseService


class DatabasePlugin:
    """Plugin providing database operations for AI agents."""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    @kernel_function(name="get_database_info", description="Get database name and description.")
    def get_database_info(self) -> str:
        info = self.db_service.get_db_info()
        if not info:
            return "No database information available."
        return "\n".join([f"Database: {db.db_name} - {db.db_description or 'No description'}" for db in info])

    @kernel_function(name="get_database_schema_info", description="Get all database schemas with descriptions.")
    def get_database_schema_info(self) -> str:
        info = self.db_service.get_schema_info()
        if not info:
            return "No schema information available."
        return "Schemas:\n" + "\n".join([f"- {s.schema_name}" for s in info if s.schema_name])

    @kernel_function(name="get_database_schema_table_info", description="Get tables in a specific schema.")
    def get_database_schema_table_info(
        self, schema_name: Annotated[str, "Schema name (e.g., 'SalesLT')"]
    ) -> str:
        info = self.db_service.get_table_schema_info(schema_name)
        if not info:
            return f"No tables found in schema '{schema_name}'."
        return f"Tables in '{schema_name}':\n" + "\n".join([f"- {t.table_name}" for t in info])

    @kernel_function(name="get_database_schema_table_columns_info", description="Get columns for a specific table.")
    def get_database_schema_table_columns_info(
        self,
        schema_name: Annotated[str, "Schema name (e.g., 'SalesLT')"],
        table_name: Annotated[str, "Table name"]
    ) -> str:
        info = self.db_service.get_column_schema_info(schema_name, table_name)
        if not info:
            return f"No columns found in '{schema_name}.{table_name}'."
        return f"Columns in '{schema_name}.{table_name}':\n" + "\n".join(
            [f"- {c.column_name} ({c.column_data_type})" for c in info]
        )

    @kernel_function(name="execute_sql_command", description="Execute a SELECT query. Use TOP 10 to limit results. Never use UPDATE/DELETE/INSERT.")
    def execute_sql_command(
        self, sql_command: Annotated[str, "SQL SELECT query to execute"]
    ) -> str:
        sql_upper = sql_command.strip().upper()
        
        # Security checks
        if not sql_upper.startswith("SELECT"):
            return "Error: Only SELECT statements allowed."
        
        forbidden = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE", "EXEC"]
        for kw in forbidden:
            if kw in sql_upper:
                return f"Error: '{kw}' not allowed."
        
        try:
            return self.db_service.execute_sql_command(sql_command)
        except Exception as e:
            return f"Error: {str(e)}"

