import pytest
import tempfile
import os
from src.profile_parser import ProfileParser
from src.uml_profile_base import (
    UMLProfile, 
    Stereotype, 
    TaggedValue,
    TimingConstraint,
    TimeUnit
)

class TestProfileParser:
    """Tests für ProfileParser"""
    
    @pytest.fixture
    def parser(self):
        """Erstellt einen ProfileParser"""
        return ProfileParser()
    
    @pytest.fixture
    def simple_profile_xmi(self):
        """Erstellt eine einfache Profil-XMI-Datei"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="SimpleProfile">
  <stereotype name="Entity" baseClass="Class">
    <ownedAttribute name="tableName" type="String"/>
    <ownedAttribute name="schema" type="String" defaultValue="public"/>
  </stereotype>
</uml:Profile>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def timing_profile_xmi(self):
        """Erstellt eine Profil-XMI mit Timing Constraints"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="TimingProfile">
  <stereotype name="TimedOperation" baseClass="Operation">
    <ownedAttribute name="description" type="String"/>
    <timingConstraint minDuration="100" maxDuration="5000" typicalDuration="2000" 
                      timeout="10000" unit="ms"/>
  </stereotype>
</uml:Profile>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def complex_profile_xmi(self):
        """Erstellt eine komplexe Profil-XMI"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="ComplexProfile">
  <stereotype name="Entity" baseClass="Class">
    <ownedAttribute name="tableName" type="String"/>
    <ownedAttribute name="schema" type="String" defaultValue="public"/>
  </stereotype>
  <stereotype name="DoorOperation" baseClass="Operation">
    <ownedAttribute name="doorType" type="String" defaultValue="automatic"/>
    <ownedAttribute name="safetyLevel" type="Integer" defaultValue="3"/>
    <timingConstraint minDuration="100" maxDuration="3000" typicalDuration="1500" 
                      timeout="5000" unit="ms"/>
  </stereotype>
  <stereotype name="Sensor" baseClass="Property">
    <ownedAttribute name="sensorType" type="String" defaultValue="optical"/>
    <ownedAttribute name="critical" type="Boolean" defaultValue="false"/>
  </stereotype>
</uml:Profile>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_parser_initialization(self, parser):
        """Test dass Parser korrekt initialisiert wird"""
        assert parser is not None
        assert 'uml' in parser.namespaces
        assert 'xmi' in parser.namespaces
    
    def test_parse_simple_profile(self, parser, simple_profile_xmi):
        """Test Parsen eines einfachen Profils"""
        profile = parser.parse_profile(simple_profile_xmi)
        
        assert profile.name == "SimpleProfile"
        assert len(profile.stereotypes) == 1
        assert profile.stereotypes[0].name == "Entity"
        assert profile.stereotypes[0].base_class == "Class"
    
    def test_parse_tagged_values(self, parser, simple_profile_xmi):
        """Test Parsen von Tagged Values"""
        profile = parser.parse_profile(simple_profile_xmi)
        
        stereotype = profile.stereotypes[0]
        assert len(stereotype.tagged_values) == 2
        
        # Prüfe Tagged Values
        tv_names = [tv.name for tv in stereotype.tagged_values]
        assert "tableName" in tv_names
        assert "schema" in tv_names
        
        # Prüfe Default-Werte
        schema_tv = next(tv for tv in stereotype.tagged_values if tv.name == "schema")
        assert schema_tv.default_value == "public"
    
    def test_parse_timing_constraint(self, parser, timing_profile_xmi):
        """Test Parsen von Timing Constraints"""
        profile = parser.parse_profile(timing_profile_xmi)
        
        stereotype = profile.stereotypes[0]
        assert stereotype.timing_constraint is not None
        
        tc = stereotype.timing_constraint
        assert tc.min_duration == 100.0
        assert tc.max_duration == 5000.0
        assert tc.typical_duration == 2000.0
        assert tc.timeout == 10000.0
        assert tc.unit == TimeUnit.SECONDS
    
    def test_parse_complex_profile(self, parser, complex_profile_xmi):
        """Test Parsen eines komplexen Profils mit mehreren Stereotypes"""
        profile = parser.parse_profile(complex_profile_xmi)
        
        assert profile.name == "ComplexProfile"
        assert len(profile.stereotypes) == 3
        
        # Prüfe Stereotype-Namen
        stereotype_names = [st.name for st in profile.stereotypes]
        assert "Entity" in stereotype_names
        assert "DoorOperation" in stereotype_names
        assert "Sensor" in stereotype_names
    
    def test_parse_different_base_classes(self, parser, complex_profile_xmi):
        """Test dass verschiedene Base Classes korrekt geparst werden"""
        profile = parser.parse_profile(complex_profile_xmi)
        
        base_classes = {st.name: st.base_class for st in profile.stereotypes}
        assert base_classes["Entity"] == "Class"
        assert base_classes["DoorOperation"] == "Operation"
        assert base_classes["Sensor"] == "Property"
    
    def test_parse_timing_constraint_with_different_units(self, parser):
        """Test Parsen von Timing Constraints mit verschiedenen Einheiten"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="TestProfile">
  <stereotype name="FastOp" baseClass="Operation">
    <timingConstraint minDuration="1" maxDuration="10" unit="us"/>
  </stereotype>
</uml:Profile>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            profile = parser.parse_profile(temp_path)
            tc = profile.stereotypes[0].timing_constraint
            assert tc.unit == TimeUnit.SECONDS
        finally:
            os.unlink(temp_path)
    
    def test_parse_profile_without_timing(self, parser, simple_profile_xmi):
        """Test dass Profile ohne Timing Constraints funktionieren"""
        profile = parser.parse_profile(simple_profile_xmi)
        
        stereotype = profile.stereotypes[0]
        assert stereotype.timing_constraint is None
    
    def test_parse_empty_profile(self, parser):
        """Test Parsen eines leeren Profils"""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="EmptyProfile">
</uml:Profile>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmi', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            profile = parser.parse_profile(temp_path)
            assert profile.name == "EmptyProfile"
            assert len(profile.stereotypes) == 0
        finally:
            os.unlink(temp_path)
