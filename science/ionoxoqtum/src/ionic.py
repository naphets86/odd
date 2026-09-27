"""
ionic.py
=============
Ionische Leitfähigkeit, Elektrokinetik und Ionentransport in wässrigen Lösungen.

Physikalische Grundlage: Abschnitt 2 der Arbeit.

Gleichungen:
  - Ionische Leitfähigkeit: σ = Σ z_i μ_i c_i F  (Gl. 1)
  - Einstein-Stokes:         D_i = μ_i k_B T / (z_i e)  (Gl. 2)
  - Debye-Abschirmlänge:    λ_D = sqrt(ε ε_0 k_B T / (2 N_A e² c_0))
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── Physikalische Konstanten ──────────────────────────────────────────────────
FARADAY       = 96_485.332_12   # C/mol
BOLTZMANN     = 1.380_649e-23   # J/K
ELEMENTARY_Q  = 1.602_176_634e-19  # C
AVOGADRO      = 6.022_140_76e23  # 1/mol
EPSILON_0     = 8.854_187_817e-12  # F/m
EPSILON_WATER = 78.4             # relative permittivity of water at 25 °C
T_STANDARD    = 298.15           # K  (25 °C)


@dataclass
class IonSpecies:
    """Beschreibt eine einzelne Ionenspezies in Lösung.

    Parameters
    ----------
    name : str
        Bezeichnung (z.B. 'K+', 'Cl-')
    valence : int
        Wertigkeit z_i (positiv oder negativ)
    mobility : float
        Elektrophoretische Beweglichkeit μ_i  [m²/(V·s)]
    concentration : float
        Molare Konzentration c_i  [mol/m³]  (1 mol/l = 1000 mol/m³)
    """
    name: str
    valence: int
    mobility: float       # m²/(V·s)
    concentration: float  # mol/m³

    @property
    def diffusion_coefficient(self, T: float = T_STANDARD) -> float:
        """Diffusionskoeffizient via Einstein-Stokes  D = μ k_B T / (|z| e)."""
        return self.mobility * BOLTZMANN * T / (abs(self.valence) * ELEMENTARY_Q)


# ── Standardparameter für KCl ─────────────────────────────────────────────────
KCL_POTASSIUM = IonSpecies(
    name="K+",
    valence=+1,
    mobility=7.62e-8,   # m²/(V·s)
    concentration=0.0,
)
KCL_CHLORIDE = IonSpecies(
    name="Cl-",
    valence=-1,
    mobility=7.91e-8,   # m²/(V·s)
    concentration=0.0,
)


class IonicSolution:
    """Modelliert eine wässrige Ionenlösung.

    Parameters
    ----------
    ions : list[IonSpecies]
        Liste der Ionenspezies in der Lösung.
    temperature : float
        Temperatur in Kelvin. Standard: 298.15 K.
    epsilon_r : float
        Relative Permittivität des Lösungsmittels. Standard: 78.4 (Wasser).
    """

    def __init__(
        self,
        ions: List[IonSpecies],
        temperature: float = T_STANDARD,
        epsilon_r: float = EPSILON_WATER,
    ) -> None:
        self.ions = ions
        self.temperature = temperature
        self.epsilon_r = epsilon_r

    # ── Leitfähigkeit ─────────────────────────────────────────────────────────
    def conductivity(self) -> float:
        """Ionische Leitfähigkeit σ = Σ |z_i| μ_i c_i F  [S/m].

        Gleichung (1) der Arbeit.

        Returns
        -------
        float
            Leitfähigkeit in S/m.
        """
        sigma = 0.0
        for ion in self.ions:
            sigma += abs(ion.valence) * ion.mobility * ion.concentration * FARADAY
        return sigma

    # ── Diffusionskoeffizienten ───────────────────────────────────────────────
    def diffusion_coefficients(self) -> Dict[str, float]:
        """Einstein-Stokes-Diffusionskoeffizient für alle Ionen.

        D_i = μ_i k_B T / (|z_i| e)

        Returns
        -------
        dict
            {ion_name: D_i [m²/s]}
        """
        result = {}
        for ion in self.ions:
            D = ion.mobility * BOLTZMANN * self.temperature / (
                abs(ion.valence) * ELEMENTARY_Q
            )
            result[ion.name] = D
        return result

    # ── Debye-Abschirmlänge ───────────────────────────────────────────────────
    def debye_length(self) -> float:
        """Debye-Abschirmlänge λ_D [m].

        λ_D = sqrt(ε ε_0 k_B T / (2 N_A e² I))
        mit Ionenstärke I = 0.5 Σ z_i² c_i  [mol/m³].

        Returns
        -------
        float
            Debye-Länge in Metern.
        """
        ionic_strength = 0.5 * sum(
            ion.valence**2 * ion.concentration for ion in self.ions
        )
        if ionic_strength <= 0:
            return np.inf
        numerator = self.epsilon_r * EPSILON_0 * BOLTZMANN * self.temperature
        denominator = 2 * AVOGADRO * ELEMENTARY_Q**2 * ionic_strength
        return np.sqrt(numerator / denominator)

    # ── Kanalwiderstand ───────────────────────────────────────────────────────
    def channel_resistance(
        self,
        length: float,
        cross_section: float,
    ) -> float:
        """Ohmscher Widerstand eines Ionenkanals R = L / (σ A)  [Ω].

        Parameters
        ----------
        length : float
            Kanallänge L [m].
        cross_section : float
            Kanalquerschnitt A [m²].

        Returns
        -------
        float
            Widerstand in Ohm.
        """
        sigma = self.conductivity()
        if sigma == 0:
            return np.inf
        return length / (sigma * cross_section)

    # ── Drift-Geschwindigkeit ─────────────────────────────────────────────────
    def drift_velocity(self, E_field: float, ion_name: str) -> float:
        """Driftgeschwindigkeit v_d = μ_i E  [m/s].

        Parameters
        ----------
        E_field : float
            Elektrisches Feld [V/m].
        ion_name : str
            Name der Ionenspezies.

        Returns
        -------
        float
            Driftgeschwindigkeit in m/s.
        """
        for ion in self.ions:
            if ion.name == ion_name:
                return ion.mobility * E_field
        raise ValueError(f"Ion '{ion_name}' nicht in Lösung gefunden.")

    def __repr__(self) -> str:
        return (
            f"IonicSolution(T={self.temperature:.1f}K, "
            f"σ={self.conductivity():.4f} S/m, "
            f"λ_D={self.debye_length()*1e9:.2f} nm)"
        )


def kcl_solution(concentration_mol_per_l: float) -> IonicSolution:
    """Erstellt eine KCl-Lösung mit gegebener Konzentration.

    Parameters
    ----------
    concentration_mol_per_l : float
        Konzentration in mol/l.

    Returns
    -------
    IonicSolution
        Fertig konfigurierte KCl-Lösung.
    """
    c_si = concentration_mol_per_l * 1000.0  # mol/l → mol/m³
    ions = [
        IonSpecies("K+",  +1, 7.62e-8, c_si),
        IonSpecies("Cl-", -1, 7.91e-8, c_si),
    ]
    return IonicSolution(ions)
