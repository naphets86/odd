from jinja2 import Template

class TemplateBasedGenerator:
    """Erweiterte Code-Generierung mit Templates"""
    
    def __init__(self):
        self.templates = {}
    
    def register_template(self, stereotype_name: str, template_str: str):
        """Registriert ein Template fÃ¼r einen Stereotype"""
        self.templates[stereotype_name] = Template(template_str)
    
    def generate(self, instance: ModelInstance) -> str:
        """Generiert Code mit Template"""
        template = self.templates.get(instance.stereotype_name)
        if not template:
            return f"# No template for {instance.stereotype_name}"
        
        context = {
            'name': instance.element_name,
            **instance.tagged_values
        }
        
        return template.render(context)

# Beispiel-Verwendung:
gen = TemplateBasedGenerator()

gen.register_template('Entity', '''
@dataclass
class {{ name }}:
    """Entity mapped to table {{ tableName }}"""
    __tablename__ = "{{ tableName }}"
    __table_args__ = {'schema': '{{ schema }}'}
''')