import pytest
import sys
import os
from pathlib import Path
import tempfile

from src.profile_parser import ProfileParser
from src.model_parser import ModelParser
from src.class_generator import ProfileClassGenerator
from src.code_generator import CodeGenerator

class TestEndToEndIntegration:
    """End-to-End Integrationstests"""
    
    @pytest.fixture
    def test_profile_xml(self):
        """Test-Profil XML"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Profile xmi:version="2.1" 
             xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
             xmlns:uml="http://www.omg.org/spec/UML/20131001" 
             name="IntegrationTestProfile">
    <stereotype name="Entity" baseClass="Class">
        <ownedAttribute name="tableName" type="String"/>
        <ownedAttribute name="schema" type="String" defaultValue="public"/>
    </stereotype>
    <stereotype name="TimedAction" baseClass="Class">
        <ownedAttribute name="min_duration" type="Float" defaultValue="0.0"/>
        <ownedAttribute name="max_duration" type="Float" defaultValue="10.0"/>
        <ownedAttribute name="time_unit" type="String" defaultValue="s"/>
    </stereotype>
</uml:Profile>'''
    
    @pytest.fixture
    def test_model_xml(self):
        """Test-Modell XML"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<uml:Model xmi:version="2.1" 
           xmlns:xmi="http://www.omg.org/spec/XMI/20131001" 
           xmlns:uml="http://www.omg.org/spec/UML/20131001" 
           name="IntegrationTestModel">
    <packagedElement xmi:type="uml:Class" name="Customer" stereotype="Entity">
        <taggedValue tag="tableName" value="customers"/>
        <taggedValue tag="schema" value="public"/>
    </packagedElement>
    <packagedElement xmi:type="uml:Class" name="OpenDoor" stereotype="TimedAction">
        <taggedValue tag="min_duration" value="2.5"/>
        <taggedValue tag="max_duration" value="4.0"/>
        <taggedValue tag="time_unit" value="s"/>
    </packagedElement>
</uml:Model>'''
    
    def test_full_pipeline(self, test_profile_xml, test_model_xml, tmp_path):
        """Test vollständige Pipeline: Profil → Klassen → Modell → Code"""
        
        # Schritt 1: Schreibe XML-Dateien
        profile_file = tmp_path / "test_profile.xmi"
        profile_file.write_text(test_profile_xml)
        
        model_file = tmp_path / "test_model.xmi"
        model_file.write_text(test_model_xml)
        
        # Schritt 2: Parse Profil
        profile_parser = ProfileParser()
        profile = profile_parser.parse_profile(str(profile_file))
        
        assert profile.name == "IntegrationTestProfile"
        assert len(profile.stereotypes) == 2
        
        # Schritt 3: Generiere Klassen aus Profil
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        generated_classes_file = output_dir / "generated_classes.py"
        class_generator = ProfileClassGenerator(profile)
        class_generator.generate_classes(str(generated_classes_file))
        
        assert generated_classes_file.exists()
        
        # Schritt 4: Parse Modell
        model_parser = ModelParser()
        instances = model_parser.parse_model(str(model_file))
        
        assert len(instances) == 2
        
        # Schritt 5: Generiere Code aus Modell
        sys.path.insert(0, str(output_dir))
        
        try:
            code_generator = CodeGenerator("generated_classes")
            generated_code = code_generator.generate_code(instances)
            
            # Prüfe generierten Code
            assert "class Customer:" in generated_code
            assert "class OpenDoor:" in generated_code
            
            # Schreibe generierten Code
            code_file = output_dir / "generated_code.py"
            code_file.write_text(generated_code)
            
            assert code_file.exists()
            
        finally:
            sys.path.remove(str(output_dir))
            if "generated_classes" in sys.modules:
                del sys.modules["generated_classes"]
    
    def test_elevator_example_if_available(self):
        """Test mit echten Aufzug-Beispieldateien (falls vorhanden)"""
        base_dir = Path(__file__).parent.parent
        profile_file = base_dir / "examples" / "elevator_profile.xmi"
        model_file = base_dir / "examples" / "elevator_model.xmi"
        
        if not profile_file.exists() or not model_file.exists():
            pytest.skip("Aufzug-Beispieldateien nicht vorhanden")
        
        # Parse Profil
        profile_parser = ProfileParser()
        profile = profile_parser.parse_profile(str(profile_file))
        
        assert profile.name == "ElevatorTimingProfile"
        
        # Generiere Klassen
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generated_classes_file = tmp_path / "elevator_classes.py"
            
            class_generator = ProfileClassGenerator(profile)
            class_generator.generate_classes(str(generated_classes_file))
            
            # Parse Modell
            model_parser = ModelParser()
            instances = model_parser.parse_model(str(model_file))
            
            # Generiere Code
            sys.path.insert(0, str(tmp_path))
            
            try:
                code_generator = CodeGenerator("elevator_classes")
                code = code_generator.generate_code(instances)
                
                # Prüfe dass Aufzug-Aktionen vorhanden sind
                assert "OpenDoor" in code or "Door" in code
                
            finally:
                sys.path.remove(str(tmp_path))
                if "elevator_classes" in sys.modules:
                    del sys.modules["elevator_classes"]


class TestErrorHandling:
    """Tests für Fehlerbehandlung"""
    
    def test_invalid_profile_xml(self, tmp_path):
        """Test mit ungültigem Profil-XML"""
        profile_file = tmp_path / "invalid.xmi"
        profile_file.write_text("This is not valid XML")
        
        parser = ProfileParser()
        with pytest.raises(Exception):
            parser.parse_profile(str(profile_file))
    
    def test_invalid_model_xml(self, tmp_path):
        """Test mit ungültigem Modell-XML"""
        model_file = tmp_path / "invalid.xmi"
        model_file.write_text("This is not valid XML")
        
        parser = ModelParser()
        with pytest.raises(Exception):
            parser.parse_model(str(model_file))
    
    def test_missing_files(self):
        """Test mit fehlenden Dateien"""
        profile_parser = ProfileParser()
        with pytest.raises(FileNotFoundError):
            profile_parser.parse_profile("/nonexistent/profile.xmi")
        
        model_parser = ModelParser()
        with pytest.raises(FileNotFoundError):
            model_parser.parse_model("/nonexistent/model.xmi")


class TestDataConsistency:
    """Tests für Datenkonsistenz"""
    
    def test_profile_to_classes_consistency(self, tmp_path):
        """Test Konsistenz zwischen Profil und generierten Klassen"""
        from src.uml_profile_base import UMLProfile, Stereotype, TaggedValue
        
        # Erstelle Profil
        tv1 = TaggedValue(name="attr1", type="String", default_value="test")
        tv2 = TaggedValue(name="attr2", type="Integer", default_value=42)
        
        stereotype = Stereotype(
            name="TestSt",
            base_class="Class",
            tagged_values=[tv1, tv2]
        )
        
        profile = UMLProfile(name="ConsistencyTest", stereotypes=[stereotype])
        
        # Generiere Klassen
        output_file = tmp_path / "consistency.py"
        generator = ProfileClassGenerator(profile)
        generator.generate_classes(str(output_file))
        
        # Prüfe Inhalt
        content = output_file.read_text()
        
        # Stereotype-Klasse sollte vorhanden sein
        assert "class TestStStereotype:" in content
        
        # Alle Tagged Values sollten als Felder vorhanden sein
        assert "attr1: str = \"test\"" in content
        assert "attr2: int = 42" in content
    
    def test_model_to_code_consistency(self, tmp_path):
        """Test Konsistenz zwischen Modell und generiertem Code"""
        # Erstelle Test-Modul
        module_content = '''
from dataclasses import dataclass

@dataclass
class EntityStereotype:
    name: str
    value: str = "default"

    def generate_code(self) -> str:
        return f"class {self.name}: pass"
'''
        module_file = tmp_path / "test_mod.py"
        module_file.write_text(module_content)
        
        sys.path.insert(0, str(tmp_path))
        
        try:
            from src.model_parser import ModelInstance
            from src.code_generator import CodeGenerator
            
            generator = CodeGenerator("test_mod")
            
            instances = [
                ModelInstance("Entity", "Class1", {"value": "v1"}),
                ModelInstance("Entity", "Class2", {"value": "v2"}),
            ]
            
            code = generator.generate_code(instances)
            
            # Beide Klassen sollten im Code sein
            assert "class Class1:" in code
            assert "class Class2:" in code
            
        finally:
            sys.path.remove(str(tmp_path))
            if "test_mod" in sys.modules:
                del sys.modules["test_mod"]
