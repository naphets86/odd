import pytest
import os
import tempfile
from src.model_parser import ModelParser, ModelInstance

class TestModelInstance:
    """Tests für ModelInstance"""
    
    def test_model_instance_creation(self):
        """Test Erstellung einer ModelInstance"""
        instance = ModelInstance(
            stereotype_name="Entity",
            element_name="Customer",
            tagged_values={"tableName": "customers", "schema": "public"}
        )
        
        assert instance.stereotype_name == "Entity"
        assert instance.element_name == "Customer"
        assert instance.tagged_values["tableName"] == "customers"
        assert instance.tagged_values["schema"] == "public"
    
    def test_model_instance_empty_tagged_values(self):
        """Test ModelInstance mit leeren Tagged Values"""
        instance = ModelInstance(
            stereotype_name="SimpleClass",
            element_name="Test",
            tagged_values={}
        )
        
        assert len(instance.tagged_values) == 0


class TestModelParser:
    """Tests für ModelParser"""
    
    @pytest.fixture
    def parser(self):
        """Erstellt einen ModelParser"""
        return ModelParser()
    
    @pytest.fixture
    def simple_model_xmi(self):
        """Erstellt eine einfache Modell-XMI"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="SimpleModel">
  <element stereotype="Entity" name="Customer">
    <taggedValue tag="tableName" value="customers"/>
    <taggedValue tag="schema" value="public"/>
  </element>
</uml:Model>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def complex_model_xmi(self):
        """Erstellt eine komplexe Modell-XMI mit mehreren Elementen"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="ComplexModel">
  <element stereotype="Entity" name="Customer">
    <taggedValue tag="tableName" value="customers"/>
    <taggedValue tag="schema" value="public"/>
  </element>
  <element stereotype="Column" name="customerId">
    <taggedValue tag="columnName" value="customer_id"/>
    <taggedValue tag="primaryKey" value="true"/>
    <taggedValue tag="nullable" value="false"/>
  </element>
  <element stereotype="DoorOperation" name="openDoor">
    <taggedValue tag="doorType" value="automatic"/>
    <taggedValue tag="safetyLevel" value="5"/>
  </element>
</uml:Model>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def empty_model_xmi(self):
        """Erstellt eine leere Modell-XMI"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="EmptyModel">
</uml:Model>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_parser_initialization(self, parser):
        """Test dass Parser korrekt initialisiert wird"""
        assert parser is not None
    
    def test_parse_simple_model(self, parser, simple_model_xmi):
        """Test Parsen eines einfachen Modells"""
        instances = parser.parse_model(simple_model_xmi)
        
        assert len(instances) == 1
        assert instances[0].stereotype_name == "Entity"
        assert instances[0].element_name == "Customer"
    
    def test_parse_tagged_values(self, parser, simple_model_xmi):
        """Test Parsen von Tagged Values"""
        instances = parser.parse_model(simple_model_xmi)
        
        instance = instances[0]
        assert "tableName" in instance.tagged_values
        assert "schema" in instance.tagged_values
        assert instance.tagged_values["tableName"] == "customers"
        assert instance.tagged_values["schema"] == "public"
    
    def test_parse_complex_model(self, parser, complex_model_xmi):
        """Test Parsen eines komplexen Modells"""
        instances = parser.parse_model(complex_model_xmi)
        
        assert len(instances) == 3
        
        # Prüfe Stereotype-Namen
        stereotype_names = [inst.stereotype_name for inst in instances]
        assert "Entity" in stereotype_names
        assert "Column" in stereotype_names
        assert "DoorOperation" in stereotype_names
    
    def test_parse_element_names(self, parser, complex_model_xmi):
        """Test dass Element-Namen korrekt geparst werden"""
        instances = parser.parse_model(complex_model_xmi)
        
        element_names = [inst.element_name for inst in instances]
        assert "Customer" in element_names
        assert "customerId" in element_names
        assert "openDoor" in element_names
    
    def test_parse_empty_model(self, parser, empty_model_xmi):
        """Test Parsen eines leeren Modells"""
        instances = parser.parse_model(empty_model_xmi)
        
        assert len(instances) == 0
    
    def test_parse_element_without_name(self, parser):
        """Test Parsen eines Elements ohne Namen"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="TestModel">
  <element stereotype="Entity">
    <taggedValue tag="tableName" value="test"/>
  </element>
</uml:Model>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            instances = parser.parse_model(temp_path)
            assert len(instances) == 1
            assert instances[0].element_name == "unnamed"
        finally:
            os.unlink(temp_path)
    
    def test_parse_element_without_tagged_values(self, parser):
        """Test Parsen eines Elements ohne Tagged Values"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="TestModel">
  <element stereotype="SimpleClass" name="Test">
  </element>
</uml:Model>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            instances = parser.parse_model(temp_path)
            assert len(instances) == 1
            assert len(instances[0].tagged_values) == 0
        finally:
            os.unlink(temp_path)
    
    def test_parse_boolean_tagged_values(self, parser, complex_model_xmi):
        """Test Parsen von Boolean Tagged Values"""
        instances = parser.parse_model(complex_model_xmi)
        
        column_instance = next(i for i in instances if i.stereotype_name == "Column")
        assert column_instance.tagged_values["primaryKey"] == "true"
        assert column_instance.tagged_values["nullable"] == "false"
    
    def test_parse_numeric_tagged_values(self, parser, complex_model_xmi):
        """Test Parsen von numerischen Tagged Values"""
        instances = parser.parse_model(complex_model_xmi)
        
        door_instance = next(i for i in instances if i.stereotype_name == "DoorOperation")
        assert door_instance.tagged_values["safetyLevel"] == "5"
