from typing import List, Dict, Type
from src.model_parser import ModelInstance
import importlib

class CodeGenerator:
    """Generiert Python-Code aus Modell-Instanzen"""
    
    def __init__(self, generated_classes_module: str):
        """
        Args:
            generated_classes_module: Name des Moduls mit generierten Klassen
        """
        self.module = importlib.import_module(generated_classes_module)
    
    def generate_code(self, instances: List[ModelInstance]) -> str:
        """Generiert Python-Code aus Modell-Instanzen"""
        code_parts = []
        
        code_parts.append('"""Auto-generated code from UML model"""')
        code_parts.append('')
        
        for instance in instances:
            code = self._generate_instance_code(instance)
            code_parts.append(code)
            code_parts.append('')
        
        return '\n'.join(code_parts)
    
    def _generate_instance_code(self, instance: ModelInstance) -> str:
        """Generiert Code für eine einzelne Instanz"""
        # Finde die entsprechende generierte Klasse
        class_name = f"{instance.stereotype_name}Stereotype"
        
        try:
            stereotype_class = getattr(self.module, class_name)
        except AttributeError:
            return f"# Error: Class {class_name} not found"
        
        # Erstelle Instanz mit Tagged Values
        kwargs = {'name': instance.element_name}
        kwargs.update(instance.tagged_values)
        
        stereotype_instance = stereotype_class(**kwargs)
        
        # Generiere Code
        return stereotype_instance.generate_code()