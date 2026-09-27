"""
irf.transistor - Ionotronic Field-Effect Transistor (IoFET)
============================================================

Implementiert:
  - IoFETState           Betriebszustand des Transistors (Enum)
  - TransferCharacteristic  Vollständige I-V-Kennlinie
  - IoFET                Haupttransistormodell
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.physics import EWODModel, IonicConductivity
from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Betriebszustand
# ---------------------------------------------------------------------------

class IoFETState(Enum):
    """Betriebszustände des IoFET-Transistors."""
    OFF = auto()         # Gesperrt: kein leitender Kanal
    SUBTHRESHOLD = auto() # Schwellwertbereich: kleiner exponentieller Strom
    LINEAR = auto()      # Linearer Bereich: V_DS ≪ V_GS - V_T
    SATURATION = auto()  # Sättigungsbereich: vollständige Benetzung


# ---------------------------------------------------------------------------
# Übertragungskennlinie
# ---------------------------------------------------------------------------

@dataclass
class TransferCharacteristic:
    """
    Vollständige Übertragungskennlinie eines IoFET.

    Enthält alle Betriebspunkte (V_GS, I_DS)-Paare.
    """

    vgs_values: list[float] = field(default_factory=list)
    ids_values: list[float] = field(default_factory=list)
    threshold_voltage_V: float = 0.0

    def add_point(self, vgs: float, ids: float) -> None:
        """Fügt einen Messpunkt zur Kennlinie hinzu."""
        self.vgs_values.append(vgs)
        self.ids_values.append(ids)

    @property
    def on_off_ratio(self) -> float:
        """
        Ein/Aus-Verhältnis: max(I_DS) / min(I_DS).
        Gibt inf zurück, wenn min(I_DS) == 0.
        """
        if not self.ids_values:
            return 0.0
        i_max = max(self.ids_values)
        i_min = min(i for i in self.ids_values if i > 0) if any(
            i > 0 for i in self.ids_values
        ) else 0.0
        if i_min == 0.0:
            return float("inf")
        return i_max / i_min

    @property
    def num_points(self) -> int:
        """Anzahl der Messpunkte."""
        return len(self.vgs_values)


# ---------------------------------------------------------------------------
# IoFET – Haupttransistormodell
# ---------------------------------------------------------------------------

class IoFET:
    """
    Ionotronic Field-Effect Transistor (IoFET).

    Verhält sich analog zum MOSFET, aber der Schaltmechanismus basiert
    auf EWOD-gesteuerter Benetzung statt auf Halbleiterphysik.

    Kennliniengleichungen:
      Linear:     I_DS = G₀ · η_EWOD · (V_GS - V_T) · V_DS
      Sättigung:  I_DS = (G₀ · η_EWOD / 2) · (V_GS - V_T)²
    """

    def __init__(
        self,
        ewod: EWODModel,
        conductivity_model: IonicConductivity,
        channel_length_m: float,
        channel_width_m: float,
        channel_height_m: float,
        threshold_angle_deg: float = 90.0,
    ) -> None:
        """
        Args:
            ewod:               EWOD-Modell des Gate-Dielektrikums
            conductivity_model: Ionenleitfähigkeitsmodell für den Kanal
            channel_length_m:   Kanallänge [m]
            channel_width_m:    Kanalbreite [m]
            channel_height_m:   Kanalhöhe [m]
            threshold_angle_deg: Kritischer Kontaktwinkel für Schaltung [°]
        """
        if channel_length_m <= 0:
            raise ValidationError("channel_length_m", channel_length_m, "> 0")
        if channel_width_m <= 0:
            raise ValidationError("channel_width_m", channel_width_m, "> 0")
        if channel_height_m <= 0:
            raise ValidationError("channel_height_m", channel_height_m, "> 0")
        if not (0.0 < threshold_angle_deg < 180.0):
            raise ValidationError(
                "threshold_angle_deg", threshold_angle_deg, "0 < θ_c < 180"
            )

        self.ewod = ewod
        self.conductivity_model = conductivity_model
        self.channel_length_m = channel_length_m
        self.channel_width_m = channel_width_m
        self.channel_height_m = channel_height_m
        self.threshold_angle_deg = threshold_angle_deg

        # Interne Zustandsvariablen
        self._vgs: float = 0.0
        self._vds: float = 0.0

    # -----------------------------------------------------------------------
    # Abgeleitete Eigenschaften
    # -----------------------------------------------------------------------

    @property
    def threshold_voltage_V(self) -> float:
        """Schwellspannung V_T [V]: Spannung bei der der Kanal öffnet."""
        return self.ewod.threshold_voltage(self.threshold_angle_deg)

    @property
    def channel_cross_section_m2(self) -> float:
        """Kanalquerschnitt A = w · h [m²]."""
        return self.channel_width_m * self.channel_height_m

    @property
    def g0(self) -> float:
        """Grundleitfähigkeit G₀ = σ A / L [S]."""
        sigma = self.conductivity_model.conductivity_S_m()
        return sigma * self.channel_cross_section_m2 / self.channel_length_m

    @property
    def eta_ewod(self) -> float:
        """EWOD-Effizienz η_EWOD [V⁻²]."""
        return self.ewod.ewod_efficiency

    # -----------------------------------------------------------------------
    # Betriebszustand
    # -----------------------------------------------------------------------

    def state(self, vgs: float, vds: float) -> IoFETState:
        """
        Bestimmt den Betriebszustand des IoFET.

        Args:
            vgs: Gate-Source-Spannung [V]
            vds: Drain-Source-Spannung [V]

        Returns:
            IoFETState Enum-Wert
        """
        vt = self.threshold_voltage_V
        if vgs < vt * 0.5:
            return IoFETState.OFF
        if vgs < vt:
            return IoFETState.SUBTHRESHOLD
        if vds < (vgs - vt):
            return IoFETState.LINEAR
        return IoFETState.SATURATION

    # -----------------------------------------------------------------------
    # Strom-Kennlinie
    # -----------------------------------------------------------------------

    def drain_current_A(self, vgs: float, vds: float) -> float:
        """
        Berechnet den Drain-Source-Strom I_DS.

        Args:
            vgs: Gate-Source-Spannung [V]
            vds: Drain-Source-Spannung [V]

        Returns:
            Drainstrom I_DS [A]
        """
        self._vgs = vgs
        self._vds = vds
        vt = self.threshold_voltage_V
        s = self.state(vgs, vds)

        if s == IoFETState.OFF:
            return 0.0

        if s == IoFETState.SUBTHRESHOLD:
            # Exponentieller Untergrenzstrom
            n = 1.5  # Subthreshold-Idealitätsfaktor
            vth = 0.026  # k_B T / e bei 300 K [V]
            i_sub = self.g0 * vds * math.exp((vgs - vt) / (n * vth))
            return max(0.0, i_sub)

        if s == IoFETState.LINEAR:
            return self.g0 * self.eta_ewod * (vgs - vt) * vds

        # SATURATION
        return 0.5 * self.g0 * self.eta_ewod * (vgs - vt) ** 2

    def transconductance_S(self, vgs: float, vds: float) -> float:
        """
        Transkonduktanz g_m = ∂I_DS / ∂V_GS [S].

        Args:
            vgs: Gate-Source-Spannung [V]
            vds: Drain-Source-Spannung [V]

        Returns:
            Transkonduktanz [S]
        """
        s = self.state(vgs, vds)
        vt = self.threshold_voltage_V

        if s in (IoFETState.OFF, IoFETState.SUBTHRESHOLD):
            return 0.0
        if s == IoFETState.LINEAR:
            return self.g0 * self.eta_ewod * vds
        # Saturation
        return self.g0 * self.eta_ewod * (vgs - vt)

    def switching_energy_J(self, vgs_target: float) -> float:
        """
        Energie für einen Schaltvorgang auf V_GS = vgs_target [J].

        Args:
            vgs_target: Ziel-Gate-Spannung [V]

        Returns:
            Schaltenergie [J]
        """
        gate_area = self.channel_width_m * self.channel_length_m
        return self.ewod.switching_energy_J(vgs_target, gate_area)

    # -----------------------------------------------------------------------
    # Kennlinien-Sweep
    # -----------------------------------------------------------------------

    def transfer_characteristic(
        self,
        vds: float,
        vgs_start: float = 0.0,
        vgs_stop: float = 1.0,
        n_points: int = 100,
    ) -> TransferCharacteristic:
        """
        Berechnet die Übertragungskennlinie (I_DS über V_GS).

        Args:
            vds:       Feste Drain-Source-Spannung [V]
            vgs_start: Startspannung [V]
            vgs_stop:  Endspannung [V]
            n_points:  Anzahl der Punkte

        Returns:
            TransferCharacteristic-Objekt
        """
        if n_points < 2:
            raise ValidationError("n_points", n_points, ">= 2")
        tc = TransferCharacteristic(threshold_voltage_V=self.threshold_voltage_V)
        step = (vgs_stop - vgs_start) / (n_points - 1)
        for i in range(n_points):
            vgs = vgs_start + i * step
            ids = self.drain_current_A(vgs, vds)
            tc.add_point(vgs, ids)
        return tc

    def output_characteristic(
        self,
        vgs: float,
        vds_start: float = 0.0,
        vds_stop: float = 1.0,
        n_points: int = 100,
    ) -> list[tuple[float, float]]:
        """
        Berechnet die Ausgangskennlinie (I_DS über V_DS).

        Returns:
            Liste von (V_DS, I_DS)-Tupeln
        """
        if n_points < 2:
            raise ValidationError("n_points", n_points, ">= 2")
        result = []
        step = (vds_stop - vds_start) / (n_points - 1)
        for i in range(n_points):
            vds = vds_start + i * step
            ids = self.drain_current_A(vgs, vds)
            result.append((vds, ids))
        return result
