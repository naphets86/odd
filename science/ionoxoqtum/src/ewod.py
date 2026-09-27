"""
ewod.py
============
Electrowetting-on-Dielectric (EWOD) – Young-Lippmann-Modell.

Gleichungen:
  - Young-Lippmann:  cos θ(V) = cos θ_0 + η_EWOD · V²
  - EWOD-Effizienz:  η = ε_0 ε_r / (2 γ_LG d)
  - Schaltspannung:  V_schalt = sqrt((cos θ_ziel − cos θ_0) / η)
  - Schaltenergie:   E = ½ C V²  mit C = ε_0 ε_r A / d
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.ionic import EPSILON_0


@dataclass
class EWODDielectric:
    """Parametriert die dielektrische Schicht eines EWOD-Systems.

    Parameters
    ----------
    thickness : float
        Schichtdicke d [m]. Typisch: 10 nm = 10e-9 m.
    epsilon_r : float
        Relative Permittivität. Al₂O₃: 9, SiO₂: 3.9, HfO₂: 25.
    """
    thickness: float   # m
    epsilon_r: float   # dimensionslos

    @property
    def capacitance_per_area(self) -> float:
        """Flächenkapazität C/A = ε_0 ε_r / d  [F/m²]."""
        return EPSILON_0 * self.epsilon_r / self.thickness


class EWODSystem:
    """Young-Lippmann-Modell für ein EWOD-System.

    Parameters
    ----------
    dielectric : EWODDielectric
        Parametrierung der dielektrischen Schicht.
    contact_angle_0 : float
        Gleichgewichtskontaktwinkel ohne Spannung θ₀ [Grad].
    surface_tension_lg : float
        Grenzflächenspannung Flüssigkeit-Gas γ_LG [N/m].
        Wasser: 0.072 N/m bei 25 °C.
    """

    def __init__(
        self,
        dielectric: EWODDielectric,
        contact_angle_0: float = 110.0,
        surface_tension_lg: float = 0.072,
    ) -> None:
        self.dielectric = dielectric
        self.theta0_deg = contact_angle_0
        self.gamma_lg = surface_tension_lg

    @property
    def theta0_rad(self) -> float:
        """θ₀ in Radiant."""
        return np.radians(self.theta0_deg)

    @property
    def ewod_efficiency(self) -> float:
        """EWOD-Effizienz η = ε₀ ε_r / (2 γ_LG d)  [V⁻²].

        Gleichung (3) der Arbeit.
        """
        return (EPSILON_0 * self.dielectric.epsilon_r) / (
            2.0 * self.gamma_lg * self.dielectric.thickness
        )

    def contact_angle(self, voltage: float) -> float:
        """Spannungsabhängiger Kontaktwinkel θ(V) in Grad.

        Young-Lippmann: cos θ(V) = cos θ₀ + η · V²

        Parameters
        ----------
        voltage : float
            Angelegte Spannung V [V].

        Returns
        -------
        float
            Kontaktwinkel in Grad.
        """
        cos_theta = np.cos(self.theta0_rad) + self.ewod_efficiency * voltage**2
        # Begrenze auf physikalisch sinnvollen Bereich [−1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def switching_voltage(self, target_angle_deg: float) -> float:
        """Benötigte Spannung für Zielkontaktwinkel θ_ziel.

        V = sqrt((cos θ_ziel − cos θ₀) / η)

        Parameters
        ----------
        target_angle_deg : float
            Zielkontaktwinkel [Grad].

        Returns
        -------
        float
            Schaltspannung [V]. NaN wenn physikalisch unerreichbar.
        """
        delta_cos = np.cos(np.radians(target_angle_deg)) - np.cos(self.theta0_rad)
        if delta_cos / self.ewod_efficiency < 0:
            return np.nan
        return np.sqrt(delta_cos / self.ewod_efficiency)

    def switching_energy(
        self,
        voltage: float,
        electrode_area: float,
    ) -> float:
        """Schaltenergie E = ½ C V²  [J].

        Parameters
        ----------
        voltage : float
            Schaltspannung [V].
        electrode_area : float
            Elektrodenfläche A [m²].

        Returns
        -------
        float
            Schaltenergie in Joule.
        """
        C = self.dielectric.capacitance_per_area * electrode_area
        return 0.5 * C * voltage**2

    def voltage_sweep(
        self,
        v_max: float = 2.0,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet θ(V) für einen Spannungssweep von 0 bis v_max.

        Parameters
        ----------
        v_max : float
            Maximale Spannung [V].
        n_points : int
            Anzahl der Stützstellen.

        Returns
        -------
        (voltages, angles) : tuple[ndarray, ndarray]
        """
        voltages = np.linspace(0, v_max, n_points)
        angles = np.array([self.contact_angle(v) for v in voltages])
        return voltages, angles

    def saturation_voltage(self, sat_angle_deg: float = 55.0) -> float:
        """Sättigungsspannung (Kontaktwinkel-Sättigung).

        Parameters
        ----------
        sat_angle_deg : float
            Sättigungskontaktwinkel [Grad]. Typisch: 55°.

        Returns
        -------
        float
            Sättigungsspannung [V].
        """
        return self.switching_voltage(sat_angle_deg)

    def __repr__(self) -> str:
        return (
            f"EWODSystem(θ₀={self.theta0_deg}°, "
            f"η={self.ewod_efficiency:.1f} V⁻², "
            f"d={self.dielectric.thickness*1e9:.1f} nm)"
        )


def standard_al2o3_system(
    contact_angle_0: float = 110.0,
) -> EWODSystem:
    """Erstellt ein Standard-EWOD-System mit Al₂O₃-Dielektrikum (10 nm).

    Parameters
    ----------
    contact_angle_0 : float
        Anfangskontaktwinkel [Grad].

    Returns
    -------
    EWODSystem
    """
    diel = EWODDielectric(thickness=10e-9, epsilon_r=9.0)
    return EWODSystem(diel, contact_angle_0=contact_angle_0)
