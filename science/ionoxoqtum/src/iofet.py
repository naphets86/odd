"""
iofet.py
=============
Ionotronic Field-Effect Transistor (IoFET) – Transistormodell.

Physikalisches Modell analog zum MOSFET (Shockley-Gleichungen), aber mit
EWOD-gesteuertem Flüssigkeitskanal statt Halbleiter-Inversionskanal.

Gleichungen:
  - Linearer Bereich:     I_DS = G_0 · η · (V_GS − V_T) · V_DS
  - Sättigungsbereich:    I_DS = (G_0 · η / 2) · (V_GS − V_T)²
  - Grundleitfähigkeit:   G_0 = σ · A / L
  - Subthreshold-Swing:   SS = ∂V_G / ∂ log₁₀(I_DS)
  - Schaltenergie:        E = ½ C V²
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .ewod import EWODSystem, EWODDielectric, standard_al2o3_system


@dataclass
class IoFETGeometry:
    """Geometrieparameter des IoFET-Kanals.

    Parameters
    ----------
    channel_length : float
        Kanallänge L [m].
    channel_width : float
        Kanalbreite W [m].
    channel_height : float
        Kanalhöhe H [m] (Wassertiefe).
    """
    channel_length: float   # m
    channel_width: float    # m
    channel_height: float   # m

    @property
    def cross_section_area(self) -> float:
        """Kanalquerschnitt A = W · H  [m²]."""
        return self.channel_width * self.channel_height

    @property
    def electrode_area(self) -> float:
        """Gate-Elektrodenfläche A = L · W  [m²]."""
        return self.channel_length * self.channel_width


class IoFET:
    """Ionotronic Field-Effect Transistor.

    Parameters
    ----------
    conductivity : float
        Ionische Leitfähigkeit σ [S/m]. Typisch: 1–10 S/m für KCl.
    geometry : IoFETGeometry
        Kanalgeometrie.
    ewod : EWODSystem
        EWOD-System für Gate-Steuerung.
    threshold_voltage : float
        Schwellspannung V_T [V].
    subthreshold_swing : float
        Subthreshold-Swing SS [mV/dec]. Typisch für IoFET: < 10 mV/dec.
    """

    def __init__(
        self,
        conductivity: float = 1.0,
        geometry: Optional[IoFETGeometry] = None,
        ewod: Optional[EWODSystem] = None,
        threshold_voltage: float = 0.05,
        subthreshold_swing: float = 5.0,
    ) -> None:
        self.sigma = conductivity
        self.geometry = geometry or IoFETGeometry(
            channel_length=1e-6,
            channel_width=100e-9,
            channel_height=100e-9,
        )
        self.ewod = ewod or standard_al2o3_system()
        self.V_T = threshold_voltage
        self.SS = subthreshold_swing  # mV/dec

    @property
    def base_conductance(self) -> float:
        """Grundleitfähigkeit G_0 = σ · A / L  [S]."""
        return (
            self.sigma
            * self.geometry.cross_section_area
            / self.geometry.channel_length
        )

    def drain_current(
        self,
        V_gs: float,
        V_ds: float,
    ) -> float:
        """Drain-Source-Strom I_DS  [A].

        Unterscheidet drei Regime:
          - Subthreshold (V_GS < V_T): exponentieller Strom
          - Linearer Bereich (V_DS < V_GS − V_T): linear
          - Sättigung (V_DS ≥ V_GS − V_T): quadratisch

        Parameters
        ----------
        V_gs : float
            Gate-Source-Spannung [V].
        V_ds : float
            Drain-Source-Spannung [V].

        Returns
        -------
        float
            I_DS in Ampere.
        """
        eta = self.ewod.ewod_efficiency
        G0 = self.base_conductance
        overdrive = V_gs - self.V_T

        if overdrive <= 0:
            # Subthreshold: I ∝ 10^(V_GS / SS)
            return self._subthreshold_current(V_gs, V_ds)

        if V_ds < overdrive:
            # Linearer Bereich
            return G0 * eta * overdrive * V_ds
        else:
            # Sättigungsbereich
            return 0.5 * G0 * eta * overdrive**2

    def _subthreshold_current(self, V_gs: float, V_ds: float) -> float:
        """Subthreshold-Strom I = I_0 · 10^(V_GS / SS)  [A]."""
        SS_V = self.SS * 1e-3  # mV → V
        I_0 = self.base_conductance * 1e-6  # Leckstrom-Referenz
        return I_0 * np.power(10.0, (V_gs - self.V_T) / SS_V) * min(V_ds / 0.01, 1.0)

    def transfer_characteristics(
        self,
        V_ds: float = 0.1,
        V_gs_range: tuple[float, float] = (-0.1, 1.0),
        n_points: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transferkennlinie I_DS(V_GS) bei konstantem V_DS.

        Parameters
        ----------
        V_ds : float
            Drain-Source-Spannung [V].
        V_gs_range : tuple
            (V_min, V_max) für Gate-Spannung [V].
        n_points : int
            Anzahl Stützstellen.

        Returns
        -------
        (V_gs_arr, I_ds_arr)
        """
        V_gs_arr = np.linspace(V_gs_range[0], V_gs_range[1], n_points)
        I_ds_arr = np.array([self.drain_current(v, V_ds) for v in V_gs_arr])
        return V_gs_arr, I_ds_arr

    def output_characteristics(
        self,
        V_gs_values: list[float],
        V_ds_range: tuple[float, float] = (0.0, 1.0),
        n_points: int = 300,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Ausgangskennlinienfeld I_DS(V_DS) für mehrere V_GS.

        Parameters
        ----------
        V_gs_values : list[float]
            Liste der Gate-Spannungen [V].
        V_ds_range : tuple
            Bereich für V_DS [V].
        n_points : int
            Anzahl Stützstellen.

        Returns
        -------
        dict : {V_gs: (V_ds_arr, I_ds_arr)}
        """
        V_ds_arr = np.linspace(V_ds_range[0], V_ds_range[1], n_points)
        result = {}
        for V_gs in V_gs_values:
            I_ds_arr = np.array([self.drain_current(V_gs, v) for v in V_ds_arr])
            result[V_gs] = (V_ds_arr, I_ds_arr)
        return result

    def switching_energy(self) -> float:
        """Schaltenergie des IoFET  E = ½ C V_schalt²  [J]."""
        V_schalt = self.ewod.switching_voltage(target_angle_deg=60.0)
        if np.isnan(V_schalt):
            return np.nan
        return self.ewod.switching_energy(V_schalt, self.geometry.electrode_area)

    def on_off_ratio(self, V_gs_on: float = 0.5, V_ds: float = 0.1) -> float:
        """Ein/Aus-Verhältnis bei gegebenen Spannungen.

        Parameters
        ----------
        V_gs_on : float
            Gate-Spannung im EIN-Zustand [V].
        V_ds : float
            Drain-Spannung [V].

        Returns
        -------
        float
            Dimensionsloses Verhältnis I_ON / I_OFF.
        """
        I_on = self.drain_current(V_gs_on, V_ds)
        I_off = self.drain_current(0.0, V_ds)
        if I_off == 0:
            return np.inf
        return I_on / I_off

    def transistor_density_3d(self, pitch: float) -> float:
        """Theoretische IoFET-Dichte für 3D-Stapelsystem.

        Parameters
        ----------
        pitch : float
            Abstand zwischen Transistoren [m].

        Returns
        -------
        float
            Transistoren pro cm³.
        """
        return 1.0 / (pitch**3) * 1e-6  # 1/m³ → 1/cm³

    def __repr__(self) -> str:
        return (
            f"IoFET(σ={self.sigma} S/m, "
            f"L={self.geometry.channel_length*1e6:.2f} µm, "
            f"G_0={self.base_conductance:.2e} S, "
            f"V_T={self.V_T} V)"
        )
