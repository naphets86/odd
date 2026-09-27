"""
irf.fabrication - Fertigungsprozessmodell und Parametervalidierung
==================================================================

Implementiert:
  - TechnologyNode    Enum der Technologiegenerationen (IRF-Roadmap)
  - ProcessStep       Einzelner Fertigungsschritt
  - ProcessValidator  Validierung von Prozessparametern
  - FabricationProcess Vollständiger Fertigungsprozess
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Technologiegenerationen
# ---------------------------------------------------------------------------

class TechnologyNode(Enum):
    """IRF-Technologiegenerationen nach der Roadmap."""
    IRF_1 = "IRF-1 (Labor, 2026)"        # 5 µm Pitch
    IRF_2 = "IRF-2 (Demo, 2027)"          # 1 µm Pitch
    IRF_3 = "IRF-3 (Pilot, 2029)"         # 500 nm Pitch
    IRF_4 = "IRF-4 (Produkt, 2031)"       # 200 nm Pitch
    IRF_5 = "IRF-5 (Erweitert, 2033)"     # 100 nm Pitch
    IRF_6 = "IRF-6 (Reif, 2035)"          # 50 nm Pitch


# Technologieparameter pro Generation
TECHNOLOGY_PARAMETERS: dict[TechnologyNode, dict[str, float]] = {
    TechnologyNode.IRF_1: {
        "electrode_pitch_m": 5e-6,
        "dielectric_thickness_m": 10e-9,
        "channel_height_m": 50e-6,
        "switching_voltage_V": 0.12,
        "switching_frequency_Hz": 1e3,
        "energy_per_switch_J": 7e-17,
    },
    TechnologyNode.IRF_2: {
        "electrode_pitch_m": 1e-6,
        "dielectric_thickness_m": 8e-9,
        "channel_height_m": 10e-6,
        "switching_voltage_V": 0.08,
        "switching_frequency_Hz": 1e4,
        "energy_per_switch_J": 2e-17,
    },
    TechnologyNode.IRF_3: {
        "electrode_pitch_m": 500e-9,
        "dielectric_thickness_m": 6e-9,
        "channel_height_m": 5e-6,
        "switching_voltage_V": 0.06,
        "switching_frequency_Hz": 1e5,
        "energy_per_switch_J": 5e-18,
    },
    TechnologyNode.IRF_4: {
        "electrode_pitch_m": 200e-9,
        "dielectric_thickness_m": 5e-9,
        "channel_height_m": 2e-6,
        "switching_voltage_V": 0.05,
        "switching_frequency_Hz": 1e6,
        "energy_per_switch_J": 1e-18,
    },
    TechnologyNode.IRF_5: {
        "electrode_pitch_m": 100e-9,
        "dielectric_thickness_m": 4e-9,
        "channel_height_m": 1e-6,
        "switching_voltage_V": 0.04,
        "switching_frequency_Hz": 1e7,
        "energy_per_switch_J": 2e-19,
    },
    TechnologyNode.IRF_6: {
        "electrode_pitch_m": 50e-9,
        "dielectric_thickness_m": 3e-9,
        "channel_height_m": 500e-9,
        "switching_voltage_V": 0.03,
        "switching_frequency_Hz": 1e8,
        "energy_per_switch_J": 5e-20,
    },
}


# ---------------------------------------------------------------------------
# Einzelner Fertigungsschritt
# ---------------------------------------------------------------------------

@dataclass
class ProcessStep:
    """
    Einzelner Schritt im IRF-Fertigungsprozess.

    Attributes:
        step_number:    Schrittnummer (1-basiert)
        name:           Bezeichnung des Schritts
        description:    Detaillierte Beschreibung
        temperature_K:  Prozesstemperatur [K]
        duration_s:     Prozessdauer [s]
        critical:       True wenn Fehler in diesem Schritt zum Ausschuss führt
        yield_factor:   Ausbeute dieses Schritts [0, 1]
    """

    step_number: int
    name: str
    description: str
    temperature_K: float
    duration_s: float
    critical: bool = True
    yield_factor: float = 0.99

    def __post_init__(self) -> None:
        if self.step_number < 1:
            raise ValidationError("step_number", self.step_number, ">= 1")
        if self.temperature_K <= 0:
            raise ValidationError("temperature_K", self.temperature_K, "> 0")
        if self.duration_s <= 0:
            raise ValidationError("duration_s", self.duration_s, "> 0")
        if not (0.0 < self.yield_factor <= 1.0):
            raise ValidationError("yield_factor", self.yield_factor, "0 < y <= 1")


# ---------------------------------------------------------------------------
# Prozessvalidierung
# ---------------------------------------------------------------------------

class ProcessValidator:
    """
    Validiert Fertigungsparameter auf Konsistenz und Machbarkeit.
    """

    # Physikalische Grenzen
    MIN_DIELECTRIC_THICKNESS_M = 2e-9   # 2 nm (ALD-Limit)
    MAX_DIELECTRIC_THICKNESS_M = 100e-9 # 100 nm
    MIN_ELECTRODE_PITCH_M = 20e-9       # 20 nm (Nanolithographie-Limit)
    MAX_SWITCHING_VOLTAGE_V = 5.0       # Hydrolysegrenze Wasser
    MIN_SWITCHING_VOLTAGE_V = 0.01      # Untergrenze für zuverlässiges Schalten

    @classmethod
    def validate_dielectric(
        cls,
        thickness_m: float,
        epsilon_r: float,
    ) -> list[str]:
        """
        Validiert Dielektrikumsparameter.

        Returns:
            Liste von Warnungen (leer = alles OK)
        """
        warnings = []
        if thickness_m < cls.MIN_DIELECTRIC_THICKNESS_M:
            warnings.append(
                f"Dielektrikumdicke {thickness_m*1e9:.1f} nm unterschreitet ALD-Limit "
                f"({cls.MIN_DIELECTRIC_THICKNESS_M*1e9:.0f} nm)"
            )
        if thickness_m > cls.MAX_DIELECTRIC_THICKNESS_M:
            warnings.append(
                f"Dielektrikumdicke {thickness_m*1e9:.1f} nm überschreitet Optimum "
                f"({cls.MAX_DIELECTRIC_THICKNESS_M*1e9:.0f} nm)"
            )
        if epsilon_r < 1.0:
            warnings.append(f"Relative Permittivität {epsilon_r} < 1 ist physikalisch unmöglich")
        if epsilon_r > 100:
            warnings.append(f"Relative Permittivität {epsilon_r} > 100 ist ungewöhnlich hoch")
        return warnings

    @classmethod
    def validate_electrode(cls, pitch_m: float, size_m: float) -> list[str]:
        """Validiert Elektrodenparameter."""
        warnings = []
        if pitch_m < cls.MIN_ELECTRODE_PITCH_M:
            warnings.append(
                f"Elektrodenpitch {pitch_m*1e9:.1f} nm unterschreitet Lithographie-Limit"
            )
        if size_m >= pitch_m:
            warnings.append(
                "Elektrodengröße muss kleiner als der Pitch sein (kein Zwischenraum)"
            )
        if size_m <= 0:
            warnings.append("Elektrodengröße muss positiv sein")
        return warnings

    @classmethod
    def validate_voltage(cls, voltage_V: float) -> list[str]:
        """Validiert Schaltspannung."""
        warnings = []
        if voltage_V > cls.MAX_SWITCHING_VOLTAGE_V:
            warnings.append(
                f"Schaltspannung {voltage_V:.2f} V überschreitet Hydrolyse-Grenze "
                f"({cls.MAX_SWITCHING_VOLTAGE_V} V)"
            )
        if voltage_V < cls.MIN_SWITCHING_VOLTAGE_V:
            warnings.append(
                f"Schaltspannung {voltage_V:.3f} V möglicherweise zu niedrig für "
                f"zuverlässiges Schalten"
            )
        return warnings

    @classmethod
    def validate_technology_node(cls, node: TechnologyNode) -> list[str]:
        """Validiert alle Parameter einer Technologiegeneration."""
        params = TECHNOLOGY_PARAMETERS[node]
        warnings = []
        warnings += cls.validate_dielectric(
            params["dielectric_thickness_m"], 9.0
        )
        warnings += cls.validate_electrode(
            params["electrode_pitch_m"],
            params["electrode_pitch_m"] * 0.7,  # 70% fill factor
        )
        warnings += cls.validate_voltage(params["switching_voltage_V"])
        return warnings


# ---------------------------------------------------------------------------
# Vollständiger Fertigungsprozess
# ---------------------------------------------------------------------------

class FabricationProcess:
    """
    Modelliert den vollständigen IRF-Fertigungsprozess mit allen 7 Schritten.
    """

    STANDARD_STEPS = [
        ProcessStep(
            step_number=1,
            name="Substrat-Präparation",
            description=(
                "Piranha-Ätzung (H₂SO₄/H₂O₂ 3:1), RCA-Reinigung, UV-Ozon-Aktivierung. "
                "Ziel: vollständig hydrophile, ionenreine Oberfläche (θ < 5°)."
            ),
            temperature_K=393.15,  # 120 °C
            duration_s=3600.0,
            critical=True,
            yield_factor=0.999,
        ),
        ProcessStep(
            step_number=2,
            name="Gate-Elektroden-Deposition",
            description=(
                "Reaktives Magnetronsputtern von ITO bei 200 °C. "
                "Strukturierung durch Photolithographie + nasschemisches Ätzen."
            ),
            temperature_K=473.15,
            duration_s=7200.0,
            critical=True,
            yield_factor=0.995,
        ),
        ProcessStep(
            step_number=3,
            name="Dielektrikum-ALD",
            description=(
                "Atomlagenabscheidung von Al₂O₃ (TMA + H₂O) bei 100 °C. "
                "Zieldicke: 10 nm, Defektdichte < 10¹⁰ cm⁻²."
            ),
            temperature_K=373.15,
            duration_s=14400.0,
            critical=True,
            yield_factor=0.998,
        ),
        ProcessStep(
            step_number=4,
            name="Selektive Oberflächenfunktionalisierung",
            description=(
                "Globale PTFE/CYTOP-Hydrophobisierung + selektive UV-Ozon-Lithographie "
                "für Pinning-Wells und Kanalverbindungen. Selbstjustierung durch SAMs."
            ),
            temperature_K=298.15,
            duration_s=3600.0,
            critical=True,
            yield_factor=0.99,
        ),
        ProcessStep(
            step_number=5,
            name="Ionische Flüssigkeitsbefüllung",
            description=(
                "Nano-Tintenstrahldruck von KCl-Pufferlösung (0.1 mol/L, pH 7.4) "
                "auf hydrophile Pinning-Wells. Anschließend Silikon-Öl-Überschichtung."
            ),
            temperature_K=298.15,
            duration_s=1800.0,
            critical=True,
            yield_factor=0.995,
        ),
        ProcessStep(
            step_number=6,
            name="Hermetische Versiegelung",
            description=(
                "ITO-Deckplatte mit SU-8 Spacer (30-100 µm) auflaminieren. "
                "UV-Härtung 60 min. Permeabilität < 10⁻¹⁰ g/(cm²·s)."
            ),
            temperature_K=298.15,
            duration_s=3600.0,
            critical=True,
            yield_factor=0.997,
        ),
        ProcessStep(
            step_number=7,
            name="CMOS-Peripherie-Integration",
            description=(
                "Flip-Chip-Bonding des CMOS-Steuerchips. Wire-Bonding der "
                "Spannungsversorgung. Funktionstest und Kalibrierung."
            ),
            temperature_K=298.15,
            duration_s=7200.0,
            critical=False,
            yield_factor=0.99,
        ),
    ]

    def __init__(
        self,
        node: TechnologyNode = TechnologyNode.IRF_1,
        custom_steps: Optional[list[ProcessStep]] = None,
    ) -> None:
        self.node = node
        self.steps: list[ProcessStep] = custom_steps or list(self.STANDARD_STEPS)
        self.validator = ProcessValidator()
        self._completed_steps: list[int] = []

    @property
    def num_steps(self) -> int:
        """Anzahl der Prozessschritte."""
        return len(self.steps)

    @property
    def total_duration_s(self) -> float:
        """Gesamtprozessdauer [s]."""
        return sum(s.duration_s for s in self.steps)

    @property
    def total_yield(self) -> float:
        """Gesamtausbeute als Produkt aller Schritt-Ausbeuten."""
        result = 1.0
        for s in self.steps:
            result *= s.yield_factor
        return result

    def execute_step(self, step_number: int) -> dict[str, object]:
        """
        Simuliert die Ausführung eines Prozessschritts.

        Args:
            step_number: 1-basierte Schrittnummer

        Returns:
            Ergebnisdictionary mit Status und Metriken
        """
        step = next((s for s in self.steps if s.step_number == step_number), None)
        if step is None:
            raise ValidationError("step_number", step_number, f"in 1..{self.num_steps}")

        self._completed_steps.append(step_number)
        return {
            "step_number": step.step_number,
            "name": step.name,
            "status": "completed",
            "yield_factor": step.yield_factor,
            "duration_s": step.duration_s,
        }

    def execute_all(self) -> list[dict[str, object]]:
        """Führt alle Prozessschritte aus."""
        return [self.execute_step(s.step_number) for s in self.steps]

    def validate(self) -> list[str]:
        """
        Validiert den gesamten Prozess für die gewählte Technologiegeneration.

        Returns:
            Liste aller Warnungen (leer = prozessbereit)
        """
        return ProcessValidator.validate_technology_node(self.node)

    @property
    def technology_parameters(self) -> dict[str, float]:
        """Parameter der gewählten Technologiegeneration."""
        return dict(TECHNOLOGY_PARAMETERS[self.node])

    def is_step_completed(self, step_number: int) -> bool:
        """Gibt True zurück, wenn der Schritt ausgeführt wurde."""
        return step_number in self._completed_steps
