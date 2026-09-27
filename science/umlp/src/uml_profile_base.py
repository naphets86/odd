from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

class TimeUnit(Enum):
    """Zeiteinheiten für Timing-Constraints"""
    NANOSECONDS = "ns"
    MICROSECONDS = "us"
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"

@dataclass
class TimingConstraint:
    """Repräsentiert zeitliche Constraints für Stereotypes"""
    min_duration: Optional[float] = None 
    max_duration: Optional[float] = None 
    typical_duration: Optional[float] = None 
    timeout: Optional[float] = None 
    deadline: Optional[float] = None 
    time_unit: TimeUnit = TimeUnit.MILLISECONDS
    unit: Optional[TimeUnit] = None 
    
    def __post_init__(self):
        """Synchronisiere unit und time_unit"""
        if self.unit is None:
            self.unit = self.time_unit
        elif self.unit != self.time_unit:
            self.time_unit = self.unit
    
    def to_seconds(self, value: float) -> float:
        """Konvertiert einen Wert in Sekunden"""
        conversions = {
            TimeUnit.NANOSECONDS: 1e-9,
            TimeUnit.MICROSECONDS: 1e-6,
            TimeUnit.MILLISECONDS: 1e-3,
            TimeUnit.SECONDS: 1.0,
            TimeUnit.MINUTES: 60.0,
            TimeUnit.HOURS: 3600.0
        }
        return value * conversions[self.time_unit]

@dataclass
class TaggedValue:
    """Repräsentiert einen Tagged Value in einem UML-Profil"""
    name: str
    type: str
    default_value: Any = None
    multiplicity: str = "1"
    
@dataclass
class Stereotype:
    """Repräsentiert einen Stereotype in einem UML-Profil"""
    name: str
    base_class: str  # z.B. "Class", "Property", "Operation"
    tagged_values: List[TaggedValue] = field(default_factory=list)
    parent_stereotype: Optional[str] = None
    timing_constraint: Optional[TimingConstraint] = None
    
@dataclass
class UMLProfile:
    """Repräsentiert ein vollständiges UML-Profil"""
    name: str
    stereotypes: List[Stereotype] = field(default_factory=list)
    
    def get_stereotype(self, name: str) -> Optional[Stereotype]:
        return next((s for s in self.stereotypes if s.name == name), None)
