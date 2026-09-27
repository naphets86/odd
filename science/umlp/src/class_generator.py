from typing import TextIO, Any
from src.uml_profile_base import UMLProfile, Stereotype, TaggedValue

class ProfileClassGenerator:
    """Generiert Python-Klassen aus UML-Profilen"""
    
    def __init__(self, profile: UMLProfile):
        self.profile = profile
    
    def generate_classes(self, output_file: str):
        """Generiert Python-Klassen für alle Stereotypes"""
        with open(output_file, 'w') as f:
            self._write_header(f)
            
            for stereotype in self.profile.stereotypes:
                self._generate_stereotype_class(f, stereotype)
    
    def _write_header(self, f: TextIO):
        """Schreibt Import-Header"""
        f.write('"""Auto-generated classes from UML Profile"""\n')
        f.write('from dataclasses import dataclass, field\n')
        f.write('from typing import Optional, List, Any\n')
        f.write('from abc import ABC\n')
        f.write('import time\n')
        f.write('import warnings\n\n')
    
    def _generate_stereotype_class(self, f: TextIO, stereotype: Stereotype):
        """Generiert eine Klasse für einen Stereotype"""
        class_name = f"{stereotype.name}Stereotype"
        
        f.write(f'@dataclass\n')
        f.write(f'class {class_name}:\n')
        f.write(f'    """Stereotype: {stereotype.name} (extends {stereotype.base_class})"""\n')
        f.write(f'    name: str\n')
        
        # Generate fields for tagged values
        for tv in stereotype.tagged_values:
            python_type = self._map_uml_type_to_python(tv.type)
            default = self._format_default_value(tv.default_value, python_type)
            
            f.write(f'    {tv.name}: {python_type}')
            if default:
                f.write(f' = {default}')
            f.write('\n')
        
        # Generate timing constraint fields if present
        if stereotype.timing_constraint:
            tc = stereotype.timing_constraint
            if tc.min_duration is not None:
                f.write(f'    min_duration: float = {tc.min_duration}\n')
            if tc.max_duration is not None:
                f.write(f'    max_duration: float = {tc.max_duration}\n')
            if tc.typical_duration is not None:
                f.write(f'    typical_duration: float = {tc.typical_duration}\n')
            if tc.timeout is not None:
                f.write(f'    timeout: float = {tc.timeout}\n')
            if tc.deadline is not None:
                f.write(f'    deadline: float = {tc.deadline}\n')
        
        f.write('\n')
        
        # Generate code generation method
        self._generate_code_method(f, stereotype)
        f.write('\n\n')
    
    def _generate_code_method(self, f: TextIO, stereotype: Stereotype):
        """Generiert die generate_code Methode"""
        f.write('    def generate_code(self) -> str:\n')
        f.write('        """Generiert Python-Code aus diesem Stereotype"""\n')
        f.write('        code = []\n')
        
        if stereotype.base_class == "Class":
            f.write('        code.append(f"class {self.name}:")\n')
            f.write('        code.append("    pass")\n')
        elif stereotype.base_class == "Property":
            f.write('        code.append(f"    {self.name}: Any")\n')
        elif stereotype.base_class == "Operation":
            f.write('        code.append(f"    def {self.name}(self):")\n')
            
            # Add timing decorator if timing constraint exists
            if stereotype.timing_constraint:
                tc = stereotype.timing_constraint
                f.write('        code.append("        start_time = time.time()")\n')
                f.write('        code.append("        # TODO: Implement operation")\n')
                f.write('        code.append("        pass")\n')
                f.write('        code.append("        duration = time.time() - start_time")\n')
                
                if tc.min_duration is not None:
                    f.write(f'        code.append("        if duration < {tc.min_duration}:")\n')
                    f.write('        code.append("            warnings.warn(f\\"Operation {{self.name}} completed too fast: {{duration}}s\\")")\n')
                
                if tc.max_duration is not None:
                    f.write(f'        code.append("        if duration > {tc.max_duration}:")\n')
                    f.write('        code.append("            warnings.warn(f\\"Operation {{self.name}} exceeded max duration: {{duration}}s\\")")\n')
            else:
                f.write('        code.append("        pass")\n')
        
        f.write('        return "\\n".join(code)\n')
    
    def _map_uml_type_to_python(self, uml_type: str) -> str:
        """Mappt UML-Typen auf Python-Typen"""
        type_mapping = {
            'String': 'str',
            'Integer': 'int',
            'Boolean': 'bool',
            'Real': 'float',
            'Duration': 'float',
            'Time': 'float',  # <-- Hinzugefügt
            'UnlimitedNatural': 'int'
        }
        return type_mapping.get(uml_type, 'Any')
    
    def _format_default_value(self, value: Any, python_type: str) -> str:
        """Formatiert Default-Werte für Python"""
        if value is None:
            return 'None'
        if python_type == 'str':
            return f'"{value}"'
        if python_type == 'bool':
            # Boolean-Werte müssen in Python großgeschrieben werden
            if isinstance(value, str):
                return 'True' if value.lower() in ('true', '1', 'yes') else 'False'
            return 'True' if value else 'False'
        return str(value)
