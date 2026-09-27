"""
irf.physics - Physikalische Grundmodelle
========================================

Implementiert:
  - EWODModel              Young-Lippmann-Gleichung und Schaltparameter
  - IonicConductivity      Ionenleitfähigkeit nach Kohlrausch
  - SurfaceTension         Temperaturabhängige Oberflächenspannung
  - ElectrokineticTransport Elektromigraton, Diffusion, Elektroosmose
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.utils import CONST, PhysicalConstants, ValidationError


# ---------------------------------------------------------------------------
# EWOD-Modell
# ---------------------------------------------------------------------------

@dataclass
class EWODModel:
    """
    Elektrowetting-on-Dielectric (EWOD) Modell.

    Implementiert die Young-Lippmann-Gleichung:
        cos θ(V) = cos θ₀ + η_EWOD · V²

    mit  η_EWOD = ε₀ ε_r / (2 γ_LG d)

    Attributes:
        dielectric_thickness_m:  Dielektrikumdicke [m]
        dielectric_epsilon_r:    Relative Permittivität des Dielektrikums
        contact_angle_0_deg:     Gleichgewichtskontaktwinkel ohne Spannung [°]
        surface_tension_N_m:     Grenzflächenspannung Flüssigkeit-Gas [N/m]
    """

    dielectric_thickness_m: float
    dielectric_epsilon_r: float
    contact_angle_0_deg: float = 110.0
    surface_tension_N_m: float = CONST.gamma_water

    def __post_init__(self) -> None:
        if self.dielectric_thickness_m <= 0:
            raise ValidationError(
                "dielectric_thickness_m", self.dielectric_thickness_m, "> 0"
            )
        if self.dielectric_epsilon_r <= 0:
            raise ValidationError(
                "dielectric_epsilon_r", self.dielectric_epsilon_r, "> 0"
            )
        if not (0.0 < self.contact_angle_0_deg < 180.0):
            raise ValidationError(
                "contact_angle_0_deg", self.contact_angle_0_deg, "0 < θ₀ < 180"
            )
        if self.surface_tension_N_m <= 0:
            raise ValidationError(
                "surface_tension_N_m", self.surface_tension_N_m, "> 0"
            )

    @property
    def ewod_efficiency(self) -> float:
        """η_EWOD = ε₀ ε_r / (2 γ d) [V⁻²]"""
        return (
            CONST.epsilon_0
            * self.dielectric_epsilon_r
            / (2.0 * self.surface_tension_N_m * self.dielectric_thickness_m)
        )

    def contact_angle_deg(self, voltage_V: float) -> float:
        """
        Berechnet den Kontaktwinkel bei angelegter Spannung.

        Args:
            voltage_V: Angelegte Spannung [V]

        Returns:
            Kontaktwinkel [°], geklemmt auf [0, 180]
        """
        cos_0 = math.cos(math.radians(self.contact_angle_0_deg))
        cos_v = cos_0 + self.ewod_efficiency * voltage_V ** 2
        # cos ist auf [-1, 1] begrenzt
        cos_v = max(-1.0, min(1.0, cos_v))
        return math.degrees(math.acos(cos_v))

    def threshold_voltage(self, target_angle_deg: float) -> float:
        """
        Berechnet die Spannung, die für einen Zielkontaktwinkel nötig ist.

        Args:
            target_angle_deg: Gewünschter Kontaktwinkel [°]

        Returns:
            Benötigte Spannung [V] (immer ≥ 0)
        """
        if not (0.0 < target_angle_deg < 180.0):
            raise ValidationError("target_angle_deg", target_angle_deg, "0 < θ < 180")
        cos_0 = math.cos(math.radians(self.contact_angle_0_deg))
        cos_t = math.cos(math.radians(target_angle_deg))
        delta_cos = cos_t - cos_0
        if delta_cos <= 0:
            return 0.0
        return math.sqrt(delta_cos / self.ewod_efficiency)

    def switching_energy_J(
        self,
        voltage_V: float,
        electrode_area_m2: float,
    ) -> float:
        """
        Schaltenergie E = ½ C V² = ½ (ε₀ ε_r A / d) V² [J].

        Args:
            voltage_V:        Schaltspannung [V]
            electrode_area_m2: Elektrodenfläche [m²]

        Returns:
            Schaltenergie [J]
        """
        if electrode_area_m2 <= 0:
            raise ValidationError("electrode_area_m2", electrode_area_m2, "> 0")
        capacitance = (
            CONST.epsilon_0
            * self.dielectric_epsilon_r
            * electrode_area_m2
            / self.dielectric_thickness_m
        )
        return 0.5 * capacitance * voltage_V ** 2

    def is_conducting(self, voltage_V: float, critical_angle_deg: float = 90.0) -> bool:
        """
        Gibt True zurück, wenn der Kanal bei gegebener Spannung leitend ist.

        Ein Kanal gilt als leitend, wenn θ(V) < θ_kritisch (Flüssigkeit bedeckt Elektrode).
        """
        return self.contact_angle_deg(voltage_V) < critical_angle_deg


# ---------------------------------------------------------------------------
# Ionische Leitfähigkeit
# ---------------------------------------------------------------------------

@dataclass
class IonSpecies:
    """Beschreibung einer Ionenspezies in Lösung."""

    name: str
    valence: int                  # Wertigkeit z
    mobility_m2_Vs: float         # Elektrophoretische Beweglichkeit [m²/(V·s)]
    concentration_mol_m3: float   # Molare Konzentration [mol/m³]

    def __post_init__(self) -> None:
        if self.valence == 0:
            raise ValidationError("valence", self.valence, "!= 0")
        if self.mobility_m2_Vs <= 0:
            raise ValidationError("mobility_m2_Vs", self.mobility_m2_Vs, "> 0")
        if self.concentration_mol_m3 < 0:
            raise ValidationError("concentration_mol_m3", self.concentration_mol_m3, ">= 0")

    @property
    def diffusion_coefficient_m2_s(self) -> float:
        """D = μ k_B T / (|z| e) – Einstein-Stokes-Relation [m²/s]."""
        return (
            self.mobility_m2_Vs
            * CONST.k_B
            * CONST.T_ref
            / (abs(self.valence) * CONST.e)
        )


class IonicConductivity:
    """
    Berechnung der elektrischen Leitfähigkeit ionischer Lösungen.

    σ = Σ_i |z_i| μ_i c_i F
    """

    # Vordefinierte Ionenspezies (KCl bei 25 °C)
    K_PLUS = IonSpecies("K+", +1, 7.62e-8, 0.0)
    CL_MINUS = IonSpecies("Cl-", -1, 7.91e-8, 0.0)

    def __init__(self, ions: Optional[list[IonSpecies]] = None) -> None:
        self.ions: list[IonSpecies] = ions if ions is not None else []

    def add_ion(self, ion: IonSpecies) -> None:
        """Fügt eine Ionenspezies hinzu."""
        self.ions.append(ion)

    def conductivity_S_m(self) -> float:
        """
        Berechnet die Gesamtleitfähigkeit [S/m].

        σ = Σ |z_i| μ_i c_i F
        """
        return sum(
            abs(ion.valence) * ion.mobility_m2_Vs * ion.concentration_mol_m3 * CONST.F
            for ion in self.ions
        )

    def channel_resistance_Ohm(
        self,
        length_m: float,
        cross_section_m2: float,
    ) -> float:
        """
        Kanalwiderstand R = L / (σ A) [Ω].

        Args:
            length_m:         Kanallänge [m]
            cross_section_m2: Kanalquerschnitt [m²]

        Returns:
            Elektrischer Widerstand [Ω]
        """
        if length_m <= 0:
            raise ValidationError("length_m", length_m, "> 0")
        if cross_section_m2 <= 0:
            raise ValidationError("cross_section_m2", cross_section_m2, "> 0")
        sigma = self.conductivity_S_m()
        if sigma == 0.0:
            return float("inf")
        return length_m / (sigma * cross_section_m2)

    @classmethod
    def from_kcl(cls, concentration_mol_per_L: float) -> "IonicConductivity":
        """
        Erstellt ein IonicConductivity-Objekt für eine KCl-Lösung.

        Args:
            concentration_mol_per_L: KCl-Konzentration [mol/L]

        Returns:
            Konfiguriertes IonicConductivity-Objekt
        """
        if concentration_mol_per_L < 0:
            raise ValidationError(
                "concentration_mol_per_L", concentration_mol_per_L, ">= 0"
            )
        c_mol_m3 = concentration_mol_per_L * 1000.0  # mol/L → mol/m³
        k = IonSpecies("K+", +1, 7.62e-8, c_mol_m3)
        cl = IonSpecies("Cl-", -1, 7.91e-8, c_mol_m3)
        return cls([k, cl])


# ---------------------------------------------------------------------------
# Oberflächenspannung
# ---------------------------------------------------------------------------

class SurfaceTension:
    """
    Temperaturabhängige Oberflächenspannung von Wasser.

    γ(T) = γ₀ · (1 - (T - T₀)/T_c)^μ
    """

    _GAMMA_0 = 0.07275   # N/m bei T₀ = 273.15 K
    _T_0 = 273.15        # K
    _T_C = 647.0         # Kritische Temperatur [K]
    _MU = 1.256

    @classmethod
    def at_temperature(cls, temperature_K: float) -> float:
        """
        Oberflächenspannung bei gegebener Temperatur [N/m].

        Args:
            temperature_K: Temperatur in Kelvin

        Returns:
            Oberflächenspannung [N/m]
        """
        if temperature_K <= 0:
            raise ValidationError("temperature_K", temperature_K, "> 0")
        if temperature_K >= cls._T_C:
            return 0.0
        ratio = 1.0 - (temperature_K - cls._T_0) / cls._T_C
        ratio = max(0.0, ratio)
        return cls._GAMMA_0 * ratio ** cls._MU

    @classmethod
    def at_celsius(cls, temperature_C: float) -> float:
        """
        Oberflächenspannung bei gegebener Temperatur in Celsius [N/m].
        """
        return cls.at_temperature(temperature_C + 273.15)

    @classmethod
    def droplet_merge_energy_J(
        cls,
        radius_1_m: float,
        radius_2_m: float,
        temperature_K: float = 298.15,
    ) -> float:
        """
        Freigesetzte Energie bei Tropfenverschmelzung [J] (immer ≥ 0).

        ΔE = γ · 4π · (r₁² + r₂² - r_f²)
        """
        if radius_1_m <= 0:
            raise ValidationError("radius_1_m", radius_1_m, "> 0")
        if radius_2_m <= 0:
            raise ValidationError("radius_2_m", radius_2_m, "> 0")
        gamma = cls.at_temperature(temperature_K)
        r_f = (radius_1_m ** 3 + radius_2_m ** 3) ** (1.0 / 3.0)
        delta_area = (
            4.0 * math.pi * (radius_1_m ** 2 + radius_2_m ** 2)
            - 4.0 * math.pi * r_f ** 2
        )
        return gamma * delta_area


# ---------------------------------------------------------------------------
# Elektrokinetischer Transport
# ---------------------------------------------------------------------------

@dataclass
class ElectrokineticTransport:
    """
    Modell für elektrokinetischen Transport in Mikrokanälen.

    Kombiniert Elektromigration, Diffusion und Elektroosmose.
    """

    ion: IonSpecies
    temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if self.temperature_K <= 0:
            raise ValidationError("temperature_K", self.temperature_K, "> 0")

    @property
    def diffusion_coefficient_m2_s(self) -> float:
        """Diffusionskoeffizient nach Einstein-Stokes [m²/s]."""
        return (
            self.ion.mobility_m2_Vs
            * CONST.k_B
            * self.temperature_K
            / (abs(self.ion.valence) * CONST.e)
        )

    def drift_velocity_m_s(self, electric_field_V_m: float) -> float:
        """
        Driftgeschwindigkeit v = μ · E [m/s].

        Args:
            electric_field_V_m: Elektrisches Feld [V/m]

        Returns:
            Driftgeschwindigkeit [m/s] (Vorzeichen entspricht Bewegungsrichtung)
        """
        return self.ion.mobility_m2_Vs * electric_field_V_m

    def diffusion_flux_mol_m2s(
        self,
        concentration_gradient_mol_m4: float,
    ) -> float:
        """
        Diffusionsfluss nach Fick: J = -D · (dc/dx) [mol/(m²·s)].

        Args:
            concentration_gradient_mol_m4: dc/dx [mol/m⁴]

        Returns:
            Molarer Fluss [mol/(m²·s)]
        """
        return -self.diffusion_coefficient_m2_s * concentration_gradient_mol_m4

    def electroosmotic_velocity_m_s(
        self,
        electric_field_V_m: float,
        zeta_potential_V: float = -0.05,
    ) -> float:
        """
        Elektroosmotische Fließgeschwindigkeit (Helmholtz-Smoluchowski).

        v_eo = -ε₀ ε_r ζ E / η

        Args:
            electric_field_V_m: Elektrisches Feld [V/m]
            zeta_potential_V:   Zeta-Potential der Kanalwand [V]

        Returns:
            Elektroosmotische Geschwindigkeit [m/s]
        """
        return (
            -CONST.epsilon_0
            * CONST.epsilon_r_water
            * zeta_potential_V
            * electric_field_V_m
            / CONST.eta_water
        )
