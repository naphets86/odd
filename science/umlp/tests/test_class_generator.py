import pytest
import tempfile
import os
from src.class_generator import ProfileClassGenerator
from src.uml_profile_base import (
    UMLProfile, 
    Stereotype, 
    TaggedValue,
    TimingConstraint,
    TimeUnit
)

class TestProfileClassGenerator:
    """Tests für ProfileClassGenerator"""
    
    @pytest.fixture
    def simple_profile(self):
        """Erstellt ein einfaches UML-Profil"""
        tv1 = TaggedValue(name="tableName", type="String")
        tv2 = TaggedValue(name="schema", type="String", default_value="public")
        
        st = Stereotype(
            name="Entity",
            base_class="Class",
            tagged_values=[tv1, tv2]
        )
        
        return UMLProfile(name="SimpleProfile", stereotypes=[st])
    
    @pytest.fixture
    def timing_profile(self):
        """Erstellt ein Profil mit Timing Constraints"""
        tv = TaggedValue(name="description", type="String")
        tc = TimingConstraint(
            min_duration=100.0,
            max_duration=3000.0,
            typical_duration=1500.0,
            timeout=5000.0,
            unit=TimeUnit.MILLISECONDS
        )
        
        st = Stereotype(
            name="DoorOperation",
            base_class="Operation",
            tagged_values=[tv],
            timing_constraint=tc
        )
        
        return UMLProfile(name="TimingProfile", stereotypes=[st])
    
    @pytest.fixture
    def complex_profile(self):
        """Erstellt ein komplexes Profil mit mehreren Stereotypes"""
        st1 = Stereotype(
            name="Entity",
            base_class="Class",
            tagged_values=[
                TaggedValue(name="tableName", type="String"),
                TaggedValue(name="schema", type="String", default_value="public")
            ]
        )
        
        st2 = Stereotype(
            name="Column",
            base_class="Property",
            tagged_values=[
                TaggedValue(name="columnName", type="String"),
                TaggedValue(name="nullable", type="Boolean", default_value=True),
                TaggedValue(name="primaryKey", type="Boolean", default_value=False)
            ]
        )
        
        tc = TimingConstraint(
            min_duration=50.0,
            max_duration=2000.0,
            unit=TimeUnit.MILLISECONDS
        )
        
        st3 = Stereotype(
            name="TimedMethod",
            base_class="Operation",
            tagged_values=[TaggedValue(name="critical", type="Boolean", default_value=False)],
            timing_constraint=tc
        )
        
        return UMLProfile(
            name="ComplexProfile",
            stereotypes=[st1, st2, st3]
        )
    
    def test_generator_initialization(self, simple_profile):
        """Test dass Generator korrekt initialisiert wird"""
        gen = ProfileClassGenerator(simple_profile)
        assert gen.profile == simple_profile
    
    def test_generate_simple_class(self, simple_profile):
        """Test Generierung einer einfachen Klasse"""
        gen = ProfileClassGenerator(simple_profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            # Lese generierte Datei
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Prüfe dass Imports vorhanden sind
            assert 'from dataclasses import dataclass' in content
            assert 'from typing import Optional, List, Any' in content
            
            # Prüfe dass Klasse generiert wurde
            assert 'class EntityStereotype:' in content
            assert 'tableName: str' in content
            assert 'schema: str = "public"' in content
            
        finally:
            os.unlink(temp_path)
    
    def test_generate_class_with_timing(self, timing_profile):
        """Test Generierung einer Klasse mit Timing Constraints"""
        gen = ProfileClassGenerator(timing_profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Prüfe Timing-Felder
            assert 'min_duration: float = 100.0' in content
            assert 'max_duration: float = 3000.0' in content
            assert 'typical_duration: float = 1500.0' in content
            assert 'timeout: float = 5000.0' in content
                 
        finally:
            os.unlink(temp_path)
    
    def test_generate_multiple_stereotypes(self, complex_profile):
        """Test Generierung mehrerer Stereotypes"""
        gen = ProfileClassGenerator(complex_profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Prüfe dass alle Klassen generiert wurden
            assert 'class EntityStereotype:' in content
            assert 'class ColumnStereotype:' in content
            assert 'class TimedMethodStereotype:' in content
            
        finally:
            os.unlink(temp_path)
    
    def test_map_uml_type_to_python(self, simple_profile):
        """Test Type-Mapping"""
        gen = ProfileClassGenerator(simple_profile)
        
        assert gen._map_uml_type_to_python('String') == 'str'
        assert gen._map_uml_type_to_python('Integer') == 'int'
        assert gen._map_uml_type_to_python('Boolean') == 'bool'
        assert gen._map_uml_type_to_python('Real') == 'float'
        assert gen._map_uml_type_to_python('UnlimitedNatural') == 'int'
        assert gen._map_uml_type_to_python('Duration') == 'float'
        assert gen._map_uml_type_to_python('Time') == 'float'
        assert gen._map_uml_type_to_python('Unknown') == 'Any'
    
    def test_format_default_value_string(self, simple_profile):
        """Test Formatierung von String-Defaults"""
        gen = ProfileClassGenerator(simple_profile)
        
        assert gen._format_default_value("test", "str") == '"test"'
        assert gen._format_default_value("public", "str") == '"public"'
    
    def test_format_default_value_boolean(self, simple_profile):
        """Test Formatierung von Boolean-Defaults"""
        gen = ProfileClassGenerator(simple_profile)
        
        assert gen._format_default_value(True, "bool") == 'True'
        assert gen._format_default_value(False, "bool") == 'False'
        assert gen._format_default_value("true", "bool") == 'True'
        assert gen._format_default_value("false", "bool") == 'False'
        assert gen._format_default_value("1", "bool") == 'True'
        assert gen._format_default_value("yes", "bool") == 'True'
    
    def test_format_default_value_number(self, simple_profile):
        """Test Formatierung von Zahlen-Defaults"""
        gen = ProfileClassGenerator(simple_profile)
        
        assert gen._format_default_value(42, "int") == '42'
        assert gen._format_default_value(3.14, "float") == '3.14'
    
    def test_format_default_value_none(self, simple_profile):
        """Test Formatierung von None-Defaults"""
        gen = ProfileClassGenerator(simple_profile)
        
        assert gen._format_default_value(None, "str") == 'None'
        assert gen._format_default_value(None, "int") == 'None'
    
    def test_generate_code_method_for_class(self):
        """Test dass generate_code Methode für Class korrekt ist"""
        st = Stereotype(name="TestClass", base_class="Class")
        profile = UMLProfile(name="Test", stereotypes=[st])
        gen = ProfileClassGenerator(profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert 'def generate_code(self) -> str:' in content
            assert 'code.append(f"class {self.name}:")' in content
            
        finally:
            os.unlink(temp_path)
    
    def test_generate_code_method_for_property(self):
        """Test dass generate_code Methode für Property korrekt ist"""
        st = Stereotype(name="TestProp", base_class="Property")
        profile = UMLProfile(name="Test", stereotypes=[st])
        gen = ProfileClassGenerator(profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert 'code.append(f"    {self.name}: Any")' in content
            
        finally:
            os.unlink(temp_path)
    
    def test_generate_code_method_for_operation(self):
        """Test dass generate_code Methode für Operation korrekt ist"""
        st = Stereotype(name="TestOp", base_class="Operation")
        profile = UMLProfile(name="Test", stereotypes=[st])
        gen = ProfileClassGenerator(profile)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
        
        try:
            gen.generate_classes(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert 'code.append(f"    def {self.name}(self):")' in content
            
        finally:
            os.unlink(temp_path)
