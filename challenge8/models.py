"""Data models for database schema information."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DbInfo:
    db_name: Optional[str] = None
    db_description: Optional[str] = None


@dataclass
class SchemaInfo:
    db_name: Optional[str] = None
    schema_name: Optional[str] = None
    schema_description: Optional[str] = None


@dataclass
class TableInfo:
    db_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    table_description: Optional[str] = None


@dataclass
class ColumnsInfo:
    db_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    column_data_type: Optional[str] = None
    column_description: Optional[str] = None

