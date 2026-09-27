import pytest
from src.uml_profile_base import (
    TaggedValue, 
    Stereotype, 
    UMLProfile,
    TimingConstraint,
    TimeUnit
)

class TestTimeUnit:
    """Tests für TimeUnit Enum"""
    
    def test_time_unit_values(self):
        """Test dass TimeUnit die richtigen Werte hat"""
        assert TimeUnit.NANOSECONDS.value == "ns"
        assert TimeUnit.MICROSECONDS.value == "us"
        assert TimeUnit.MILLISECONDS.value == "ms"
        assert TimeUnit.SECONDS.value == "s"
        assert TimeUnit.MINUTES.value == "min"
        assert TimeUnit.HOURS.value == "h"
    
    def test_time_unit_count(self):
        """Test dass alle TimeUnits vorhanden sind"""
        assert len(TimeUnit) == 6


class TestTimingConstraint:
    """Tests für TimingConstraint"""
    
    def test_timing_constraint_creation_minimal(self):
        """Test Erstellung mit minimalen Parametern"""
        tc = TimingConstraint()
        assert tc.min_duration is None
        assert tc.max_duration is None
        assert tc.typical_duration is None
        assert tc.unit == TimeUnit.MILLISECONDS
        assert tc.timeout is None
        assert tc.deadline is None
    
    def test_timing_constraint_creation_full(self):
        """Test Erstellung mit allen Parametern"""
        tc = TimingConstraint(
            min_duration=100.0,
            max_duration=5000.0,
            typical_duration=2000.0,
            unit=TimeUnit.SECONDS,
            timeout=10.0,
            deadline=15.0
        )
        assert tc.min_duration == 100.0
        assert tc.max_duration == 5000.0
        assert tc.typical_duration == 2000.0
        assert tc.unit == TimeUnit.SECONDS
        assert tc.timeout == 10.0
        assert tc.deadline == 15.0


class TestTaggedValue:
    """Tests für TaggedValue"""
    
    def test_tagged_value_creation_minimal(self):
        """Test Erstellung mit minimalen Parametern"""
        tv = TaggedValue(name="testTag", type="String")
        assert tv.name == "testTag"
        assert tv.type == "String"
        assert tv.default_value is None
        assert tv.multiplicity == "1"
    
    def test_tagged_value_creation_with_default(self):
        """Test Erstellung mit Default-Wert"""
        tv = TaggedValue(
            name="tableName", 
            type="String", 
            default_value="users"
        )
        assert tv.name == "tableName"
        assert tv.default_value == "users"
    
    def test_tagged_value_different_types(self):
        """Test mit verschiedenen Typen"""
        tv_int = TaggedValue(name="count", type="Integer", default_value=42)
        tv_bool = TaggedValue(name="active", type="Boolean", default_value=True)
        tv_float = TaggedValue(name="rate", type="Real", default_value=3.14)
        
        assert tv_int.type == "Integer"
        assert tv_bool.type == "Boolean"
        assert tv_float.type == "Real"


class TestStereotype:
    """Tests für Stereotype"""
    
    def test_stereotype_creation_minimal(self):
        """Test Erstellung mit minimalen Parametern"""
        st = Stereotype(name="Entity", base_class="Class")
        assert st.name == "Entity"
        assert st.base_class == "Class"
        assert len(st.tagged_values) == 0
        assert st.parent_stereotype is None
        assert st.timing_constraint is None
    
    def test_stereotype_with_tagged_values(self):
        """Test Stereotype mit Tagged Values"""
        tv1 = TaggedValue(name="tableName", type="String")
        tv2 = TaggedValue(name="schema", type="String", default_value="public")
        
        st = Stereotype(
            name="Entity", 
            base_class="Class",
            tagged_values=[tv1, tv2]
        )
        
        assert len(st.tagged_values) == 2
        assert st.tagged_values[0].name == "tableName"
        assert st.tagged_values[1].default_value == "public"
    
    def test_stereotype_with_timing_constraint(self):
        """Test Stereotype mit Timing Constraint"""
        tc = TimingConstraint(
            min_duration=100.0,
            max_duration=3000.0,
            unit=TimeUnit.MILLISECONDS
        )
        
        st = Stereotype(
            name="DoorOperation",
            base_class="Operation",
            timing_constraint=tc
        )
        
        assert st.timing_constraint is not None
        assert st.timing_constraint.min_duration == 100.0
        assert st.timing_constraint.max_duration == 3000.0
    
    def test_stereotype_with_parent(self):
        """Test Stereotype mit Parent"""
        st = Stereotype(
            name="SpecialEntity",
            base_class="Class",
            parent_stereotype="Entity"
        )
        assert st.parent_stereotype == "Entity"


class TestUMLProfile:
    """Tests für UMLProfile"""
    
    def test_uml_profile_creation(self):
        """Test Erstellung eines leeren Profils"""
        profile = UMLProfile(name="TestProfile")
        assert profile.name == "TestProfile"
        assert len(profile.stereotypes) == 0
    
    def test_uml_profile_with_stereotypes(self):
        """Test Profil mit Stereotypes"""
        st1 = Stereotype(name="Entity", base_class="Class")
        st2 = Stereotype(name="Column", base_class="Property")
        
        profile = UMLProfile(
            name="DatabaseProfile",
            stereotypes=[st1, st2]
        )
        
        assert len(profile.stereotypes) == 2
        assert profile.stereotypes[0].name == "Entity"
        assert profile.stereotypes[1].name == "Column"
    
    def test_get_stereotype_found(self):
        """Test get_stereotype wenn Stereotype existiert"""
        st1 = Stereotype(name="Entity", base_class="Class")
        st2 = Stereotype(name="Column", base_class="Property")
        
        profile = UMLProfile(
            name="TestProfile",
            stereotypes=[st1, st2]
        )
        
        found = profile.get_stereotype("Entity")
        assert found is not None
        assert found.name == "Entity"
    
    def test_get_stereotype_not_found(self):
        """Test get_stereotype wenn Stereotype nicht existiert"""
        profile = UMLProfile(name="TestProfile")
        found = profile.get_stereotype("NonExistent")
        assert found is None
    
    def test_get_stereotype_from_multiple(self):
        """Test get_stereotype bei mehreren Stereotypes"""
        stereotypes = [
            Stereotype(name=f"Stereotype{i}", base_class="Class")
            for i in range(10)
        ]
        
        profile = UMLProfile(name="TestProfile", stereotypes=stereotypes)
        
        found = profile.get_stereotype("Stereotype5")
        assert found is not None
        assert found.name == "Stereotype5"
