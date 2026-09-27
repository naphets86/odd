import xml.etree.ElementTree as ET
from typing import Dict, Optional
from src.uml_profile_base import UMLProfile, Stereotype, TaggedValue, TimingConstraint, TimeUnit

class ProfileParser:
    """Parst UML-Profile aus XMI-Dateien"""
    
    def __init__(self):
        self.namespaces = {
            'uml': 'http://www.omg.org/spec/UML/20131001',
            'xmi': 'http://www.omg.org/spec/XMI/20131001'
        }
    
    def parse_profile(self, xmi_file: str) -> UMLProfile:
        """Parst ein UML-Profil aus einer XMI-Datei"""
        tree = ET.parse(xmi_file)
        root = tree.getroot()
        
        profile_name = root.get('name', 'UnnamedProfile')
        profile = UMLProfile(name=profile_name)
        
        # Parse Stereotypes
        for stereotype_elem in root.findall('.//stereotype', self.namespaces):
            stereotype = self._parse_stereotype(stereotype_elem)
            profile.stereotypes.append(stereotype)
        
        return profile
    
    def _parse_stereotype(self, elem: ET.Element) -> Stereotype:
        """Parst einen einzelnen Stereotype"""
        name = elem.get('name', 'UnnamedStereotype')
        base_class = elem.get('baseClass', 'Class')
        
        stereotype = Stereotype(name=name, base_class=base_class)
        
        # Parse Tagged Values
        for attr_elem in elem.findall('.//ownedAttribute', self.namespaces):
            tagged_value = self._parse_tagged_value(attr_elem)
            stereotype.tagged_values.append(tagged_value)
        
        # Parse Timing Constraint (falls vorhanden)
        timing_elem = elem.find('.//timingConstraint', self.namespaces)
        if timing_elem is not None:
            stereotype.timing_constraint = self._parse_timing_constraint(timing_elem)
        
        return stereotype
    
    def _parse_tagged_value(self, elem: ET.Element) -> TaggedValue:
        """Parst einen Tagged Value"""
        name = elem.get('name', 'unnamed')
        type_ref = elem.get('type', 'String')
        default = elem.get('defaultValue')
        
        return TaggedValue(
            name=name,
            type=type_ref,
            default_value=default
        )
    
    def _parse_timing_constraint(self, elem: ET.Element) -> TimingConstraint:
        """Parst ein Timing Constraint"""
        min_dur_str = elem.get('minDuration')
        max_dur_str = elem.get('maxDuration')
        typical_dur_str = elem.get('typicalDuration')
        timeout_str = elem.get('timeout')
        deadline_str = elem.get('deadline')
        time_unit_str = elem.get('timeUnit', 's')
        
        # Konvertiere zu float oder None
        min_dur = float(min_dur_str) if min_dur_str else None
        max_dur = float(max_dur_str) if max_dur_str not in (None, 'inf') else None
        typical_dur = float(typical_dur_str) if typical_dur_str else None
        timeout = float(timeout_str) if timeout_str else None
        deadline = float(deadline_str) if deadline_str else None
        
        # Map string to TimeUnit enum
        time_unit_map = {
            'ns': TimeUnit.NANOSECONDS,
            'us': TimeUnit.MICROSECONDS,
            'ms': TimeUnit.MILLISECONDS,
            's': TimeUnit.SECONDS,
            'sec': TimeUnit.SECONDS,
            'seconds': TimeUnit.SECONDS,
            'min': TimeUnit.MINUTES,
            'minutes': TimeUnit.MINUTES,
            'h': TimeUnit.HOURS,
            'hours': TimeUnit.HOURS
        }
        time_unit = time_unit_map.get(time_unit_str.lower(), TimeUnit.SECONDS)
        
        return TimingConstraint(
            min_duration=min_dur,
            max_duration=max_dur,
            typical_duration=typical_dur,
            timeout=timeout,
            deadline=deadline,
            time_unit=time_unit,
            unit=time_unit
        )


