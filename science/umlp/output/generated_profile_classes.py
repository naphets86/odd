"""Auto-generated classes from UML Profile"""
from dataclasses import dataclass, field
from typing import Optional, List, Any
from abc import ABC

@dataclass
class EntityStereotype:
    """Stereotype: Entity (extends Class)"""
    name: str
    tableName: str = None
    schema: str = "public"

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"class {self.name}:")
        code.append("    pass")
        return "\n".join(code)


@dataclass
class ColumnStereotype:
    """Stereotype: Column (extends Property)"""
    name: str
    columnName: str = None
    nullable: bool = True
    primaryKey: bool = False

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"    {self.name}: Any")
        return "\n".join(code)
