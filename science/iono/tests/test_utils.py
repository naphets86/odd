"""Tests für irf.utils"""

import logging
import math
import pytest

from src.utils import (
    Logger,
    PhysicalConstants,
    UnitConverter,
    ValidationError,
    CONST,
)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------

class TestValidationError:
    def test_attributes(self):
        err = ValidationError("foo", 42, "> 0")
        assert err.parameter == "foo"
        assert err.value == 42
        assert err.constraint == "> 0"

    def test_message_contains_all_parts(self):
        err = ValidationError("voltage", -1.5, ">= 0")
        msg = str(err)
        assert "voltage" in msg
        assert "-1.5" in msg
        assert ">= 0" in msg

    def test_is_value_error(self):
        with pytest.raises(ValueError):
            raise ValidationError("x", 0, "!= 0")


# ---------------------------------------------------------------------------
# PhysicalConstants
# ---------------------------------------------------------------------------

class TestPhysicalConstants:
    def test_singleton_values(self):
        c = PhysicalConstants()
        assert c.epsilon_0 == pytest.approx(8.854187817e-12, rel=1e-6)
        assert c.k_B == pytest.approx(1.380649e-23, rel=1e-6)
        assert c.e == pytest.approx(1.602176634e-19, rel=1e-6)
        assert c.F == pytest.approx(96485.33212, rel=1e-6)
        assert c.N_A == pytest.approx(6.02214076e23, rel=1e-6)

    def test_water_properties(self):
        c = PhysicalConstants()
        assert 0.06 < c.gamma_water < 0.08
        assert c.rho_water == pytest.approx(997.0)
        assert c.epsilon_r_water == pytest.approx(78.4)

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            CONST.epsilon_0 = 0.0

    def test_const_module_level(self):
        assert isinstance(CONST, PhysicalConstants)


# ---------------------------------------------------------------------------
# UnitConverter
# ---------------------------------------------------------------------------

class TestUnitConverter:
    # --- to_meters ---
    def test_meters(self):
        assert UnitConverter.to_meters(1.0, "m") == pytest.approx(1.0)

    def test_micrometers(self):
        assert UnitConverter.to_meters(1.0, "um") == pytest.approx(1e-6)
        assert UnitConverter.to_meters(1.0, "µm") == pytest.approx(1e-6)

    def test_nanometers(self):
        assert UnitConverter.to_meters(10.0, "nm") == pytest.approx(10e-9)

    def test_centimeters(self):
        assert UnitConverter.to_meters(100.0, "cm") == pytest.approx(1.0)

    def test_millimeters(self):
        assert UnitConverter.to_meters(1000.0, "mm") == pytest.approx(1.0)

    def test_picometers(self):
        assert UnitConverter.to_meters(1.0, "pm") == pytest.approx(1e-12)

    def test_invalid_length_unit(self):
        with pytest.raises(ValidationError):
            UnitConverter.to_meters(1.0, "km")

    # --- to_volts ---
    def test_volts(self):
        assert UnitConverter.to_volts(1.0, "V") == pytest.approx(1.0)

    def test_millivolts(self):
        assert UnitConverter.to_volts(1000.0, "mV") == pytest.approx(1.0)

    def test_microvolts(self):
        assert UnitConverter.to_volts(1.0, "uV") == pytest.approx(1e-6)
        assert UnitConverter.to_volts(1.0, "µV") == pytest.approx(1e-6)

    def test_kilovolts(self):
        assert UnitConverter.to_volts(1.0, "kV") == pytest.approx(1000.0)

    def test_invalid_voltage_unit(self):
        with pytest.raises(ValidationError):
            UnitConverter.to_volts(1.0, "mA")

    # --- to_cubic_meters ---
    def test_liters_via_ul(self):
        assert UnitConverter.to_cubic_meters(1e9, "ul") == pytest.approx(1.0)

    def test_nanoliters(self):
        assert UnitConverter.to_cubic_meters(1.0, "nl") == pytest.approx(1e-12)

    def test_picoliters(self):
        assert UnitConverter.to_cubic_meters(1.0, "pl") == pytest.approx(1e-15)

    def test_femtoliters(self):
        assert UnitConverter.to_cubic_meters(1.0, "fl") == pytest.approx(1e-18)

    def test_cubic_meters(self):
        assert UnitConverter.to_cubic_meters(1.0, "m3") == pytest.approx(1.0)

    def test_cubic_centimeters(self):
        assert UnitConverter.to_cubic_meters(1.0, "cm3") == pytest.approx(1e-6)

    def test_cubic_millimeters(self):
        assert UnitConverter.to_cubic_meters(1.0, "mm3") == pytest.approx(1e-9)

    def test_microliter_alt(self):
        assert UnitConverter.to_cubic_meters(1.0, "µl") == pytest.approx(1e-9)

    def test_invalid_volume_unit(self):
        with pytest.raises(ValidationError):
            UnitConverter.to_cubic_meters(1.0, "gallons")

    # --- temperature ---
    def test_celsius_to_kelvin(self):
        assert UnitConverter.celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert UnitConverter.celsius_to_kelvin(100.0) == pytest.approx(373.15)
        assert UnitConverter.celsius_to_kelvin(-273.15) == pytest.approx(0.0)

    def test_kelvin_to_celsius(self):
        assert UnitConverter.kelvin_to_celsius(273.15) == pytest.approx(0.0)
        assert UnitConverter.kelvin_to_celsius(373.15) == pytest.approx(100.0)

    def test_kelvin_to_celsius_negative_raises(self):
        with pytest.raises(ValidationError):
            UnitConverter.kelvin_to_celsius(-1.0)

    # --- angles ---
    def test_degrees_to_radians(self):
        assert UnitConverter.degrees_to_radians(180.0) == pytest.approx(math.pi)
        assert UnitConverter.degrees_to_radians(90.0) == pytest.approx(math.pi / 2)

    def test_radians_to_degrees(self):
        assert UnitConverter.radians_to_degrees(math.pi) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TestLogger:
    def setup_method(self):
        Logger.reset()

    def test_returns_logger(self):
        logger = Logger.get("test_irf")
        assert isinstance(logger, logging.Logger)

    def test_singleton_per_name(self):
        l1 = Logger.get("abc")
        l2 = Logger.get("abc")
        assert l1 is l2

    def test_different_names_different_instances(self):
        l1 = Logger.get("x1")
        l2 = Logger.get("x2")
        assert l1 is not l2

    def test_set_level(self):
        Logger.get("lvl_test")
        Logger.set_level("lvl_test", logging.DEBUG)
        assert Logger.get("lvl_test").level == logging.DEBUG

    def test_reset_clears_all(self):
        Logger.get("a")
        Logger.get("b")
        Logger.reset()
        # Nach reset: neue Instanzen
        l = Logger.get("a")
        assert isinstance(l, logging.Logger)

    def test_default_level(self):
        logger = Logger.get("default_level_test", logging.WARNING)
        assert logger.level == logging.WARNING
