import pytest
import sys
import os
from pathlib import Path

from src.code_generator import CodeGenerator
from src.model_parser import ModelInstance

class TestCodeGenerator:
    """Tests für CodeGenerator Klasse"""
    
    @pytest.fixture
    def setup_generated_module(self, tmp_path):
        """Erstelle ein Test-Modul mit generierten Klassen"""
        module_content = '''"""Test generated classes"""
from dataclasses import dataclass
from typing import Any

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
        return "\\n".join(code)


@dataclass
class ColumnStereotype:
    """Stereotype: Column (extends Property)"""
    name: str
    columnName: str = None
    nullable: bool = True

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"    {self.name}: Any")
        return "\\n".join(code)


@dataclass
class TimedActionStereotype:
    """Stereotype: TimedAction (extends Class)"""
    name: str
    min_duration: float = 0.0
    max_duration: float = 10.0
    time_unit: str = "s"

    def generate_code(self) -> str:
        """Generiert Python-Code aus diesem Stereotype"""
        code = []
        code.append(f"class {self.name}:")
        doc_parts = []
        doc_parts.append("    \\"\\"\\\"")
        if hasattr(self, "min_duration") and self.min_duration:
            doc_parts.append(f"    Min Duration: {self.min_duration}{self.time_unit}")
        if hasattr(self, "max_duration") and self.max_duration:
            doc_parts.append(f"    Max Duration: {self.max_duration}{self.time_unit}")
        doc_parts.append("    \\"\\"\\\"")
        code.extend(doc_parts)
        code.append("    pass")
        return "\\n".join(code)
'''
        module_file = tmp_path / "test_generated.py"
        module_file.write_text(module_content)
        
        # Füge tmp_path zum Python-Pfad hinzu
        sys.path.insert(0, str(tmp_path))
        
        yield "test_generated"
        
        # Cleanup
        sys.path.remove(str(tmp_path))
        if "test_generated" in sys.modules:
            del sys.modules["test_generated"]
    
    def test_generator_initialization(self, setup_generated_module):
        """Test Generator-Initialisierung"""
        generator = CodeGenerator(setup_generated_module)
        assert generator is not None
        assert generator.module is not None
    
    def test_generate_single_instance(self, setup_generated_module):
        """Test Generierung eines einzelnen Elements"""
        generator = CodeGenerator(setup_generated_module)
        
        instance = ModelInstance(
            stereotype_name="Entity",
            element_name="Customer",
            tagged_values={"tableName": "customers", "schema": "public"}
        )
        
        code = generator.generate_code([instance])
        
        assert "class Customer:" in code
        assert "pass" in code
        assert '"""Auto-generated code from UML model"""' in code
    
    def test_generate_empty_list(self, setup_generated_module):
        """Test Generierung mit leerer Liste"""
        generator = CodeGenerator(setup_generated_module)
        
        code = generator.generate_code([])
        
        assert '"""Auto-generated code from UML model"""' in code
