"""Auto-generated classes from UML Profile"""
from dataclasses import dataclass, field
from typing import Optional, List, Any
from abc import ABC
import time
import warnings

@dataclass
class TimedActionStereotype:
    """Stereotype: TimedAction (extends Class)"""
    name: str
    description: str = ""
    min_duration: float = 0.0
    max_duration: float = 10000.0
    typical_duration: float = 3000.0
    timeout: float = 15000.0

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"class {self.name}:")
        code.append("    pass")
        return "\n".join(code)


@dataclass
class TimedOperationStereotype:
    """Stereotype: TimedOperation (extends Operation)"""
    name: str
    is_blocking: bool = True
    min_duration: float = 0.0
    max_duration: float = 5000.0
    typical_duration: float = 1000.0
    timeout: float = 10000.0

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"    def {self.name}(self):")
        code.append("        start_time = time.time()")
        code.append("        # TODO: Implement operation")
        code.append("        pass")
        code.append("        duration = time.time() - start_time")
        code.append("        if duration < 0.0:")
        code.append("            warnings.warn(f\"Operation {{self.name}} completed too fast: {{duration}}s\")")
        code.append("        if duration > 5000.0:")
        code.append("            warnings.warn(f\"Operation {{self.name}} exceeded max duration: {{duration}}s\")")
        return "\n".join(code)


@dataclass
class TimedStateStereotype:
    """Stereotype: TimedState (extends Class)"""
    name: str
    is_stable: bool = True
    min_duration: float = 0.0

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"class {self.name}:")
        code.append("    pass")
        return "\n".join(code)


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


