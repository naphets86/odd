import xml.etree.ElementTree as ET
from typing import List, Dict, Any

class ModelInstance:
    """ReprÃ¤sentiert eine Instanz eines Stereotypes im Modell"""
    def __init__(self, stereotype_name: str, element_name: str, tagged_values: Dict[str, Any]):
        self.stereotype_name = stereotype_name
        self.element_name = element_name
        self.tagged_values = tagged_values

class ModelParser:
    """Parst UML-Modelle mit angewendeten Profilen"""
    
    def parse_model(self, xmi_file: str) -> List[ModelInstance]:
        """Parst ein UML-Modell und extrahiert Stereotype-Instanzen"""
        tree = ET.parse(xmi_file)
        root = tree.getroot()
        
        instances = []
        
        # Suche nach Elementen mit angewendeten Stereotypes
        for elem in root.iter():
            if 'stereotype' in elem.attrib:
                instance = self._parse_instance(elem)
                instances.append(instance)
        
        return instances
    
    def _parse_instance(self, elem: ET.Element) -> ModelInstance:
        """Parst eine Stereotype-Instanz"""
        stereotype_name = elem.get('stereotype')
        element_name = elem.get('name', 'unnamed')
        
        tagged_values = {}
        for child in elem:
            if child.tag.startswith('taggedValue'):
                tag_name = child.get('tag')
                tag_value = child.get('value')
                tagged_values[tag_name] = tag_value
        
        return ModelInstance(stereotype_name, element_name, tagged_values)