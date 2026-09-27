"""
irf.utils - Hilfsfunktionen, Konstanten, Logging und Validierung
================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Ausnahmen
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Wird ausgelöst, wenn ein Parameterwert außerhalb des gültigen Bereichs liegt."""

    def __init__(self, parameter: str, value: Any, constraint: str) -> None:
        self.parameter = parameter
        self.value = value
        self.constraint = constraint
        super().__init__(
            f"Ungültiger Wert für '{parameter}': {value!r} – Bedingung: {constraint}"
        )


# ---------------------------------------------------------------------------
# Physikalische Konstanten
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalConstants:
    """Unveränderliche physikalische Konstanten (SI-Einheiten)."""

    # Fundamentalkonstanten
    epsilon_0: float = 8.854187817e-12   # Vakuumpermittivität [F/m]
    k_B: float = 1.380649e-23            # Boltzmann-Konstante [J/K]
    e: float = 1.602176634e-19           # Elementarladung [C]
    F: float = 96485.33212               # Faraday-Konstante [C/mol]
    N_A: float = 6.02214076e23           # Avogadro-Konstante [1/mol]

    # Wassereigenschaften (25 °C)
    gamma_water: float = 0.0728          # Oberflächenspannung Wasser [N/m]
    eta_water: float = 8.90e-4           # Dynamische Viskosität Wasser [Pa·s]
    rho_water: float = 997.0             # Dichte Wasser [kg/m³]
    epsilon_r_water: float = 78.4        # Relative Permittivität Wasser

    # Referenztemperatur
    T_ref: float = 298.15                # Referenztemperatur [K]


CONST = PhysicalConstants()


# ---------------------------------------------------------------------------
# Einheitenkonvertierung
# ---------------------------------------------------------------------------

class UnitConverter:
    """Konvertierung zwischen häufig verwendeten physikalischen Einheiten."""

    # Längenpräfixe → Meter
    _LENGTH = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }

    # Spannungspräfixe → Volt
    _VOLTAGE = {
        "V": 1.0,
        "mV": 1e-3,
        "uV": 1e-6,
        "µV": 1e-6,
        "kV": 1e3,
    }

    # Volumen → m³
    _VOLUME = {
        "m3": 1.0,
        "cm3": 1e-6,
        "mm3": 1e-9,
        "ul": 1e-9,    # Mikroliter
        "µl": 1e-9,
        "nl": 1e-12,   # Nanoliter
        "pl": 1e-15,   # Pikoliter
        "fl": 1e-18,   # Femtoliter
    }

    @classmethod
    def to_meters(cls, value: float, unit: str) -> float:
        """Konvertiert Länge in Meter."""
        if unit not in cls._LENGTH:
            raise ValidationError("unit", unit, f"muss eines von {list(cls._LENGTH)} sein")
        return value * cls._LENGTH[unit]

    @classmethod
    def to_volts(cls, value: float, unit: str) -> float:
        """Konvertiert Spannung in Volt."""
        if unit not in cls._VOLTAGE:
            raise ValidationError("unit", unit, f"muss eines von {list(cls._VOLTAGE)} sein")
        return value * cls._VOLTAGE[unit]

    @classmethod
    def to_cubic_meters(cls, value: float, unit: str) -> float:
        """Konvertiert Volumen in m³."""
        if unit not in cls._VOLUME:
            raise ValidationError("unit", unit, f"muss eines von {list(cls._VOLUME)} sein")
        return value * cls._VOLUME[unit]

    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Konvertiert Celsius in Kelvin."""
        return celsius + 273.15

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Konvertiert Kelvin in Celsius."""
        if kelvin < 0:
            raise ValidationError("kelvin", kelvin, ">= 0")
        return kelvin - 273.15

    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Konvertiert Grad in Bogenmass."""
        return math.radians(degrees)

    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """Konvertiert Bogenmass in Grad."""
        return math.degrees(radians)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    """Zentrales Logging-Subsystem für das IRF-Framework."""

    _instances: dict[str, logging.Logger] = {}

    @classmethod
    def get(cls, name: str = "irf", level: int = logging.INFO) -> logging.Logger:
        """Gibt einen konfigurierten Logger zurück (Singleton pro Name)."""
        if name not in cls._instances:
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                fmt = logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
                    datefmt="%H:%M:%S",
                )
                handler.setFormatter(fmt)
                logger.addHandler(handler)
            logger.setLevel(level)
            cls._instances[name] = logger
        return cls._instances[name]

    @classmethod
    def set_level(cls, name: str, level: int) -> None:
        """Setzt den Log-Level eines bestehenden Loggers."""
        logger = cls.get(name)
        logger.setLevel(level)

    @classmethod
    def reset(cls) -> None:
        """Setzt alle Logger-Instanzen zurück (nützlich für Tests)."""
        cls._instances.clear()
