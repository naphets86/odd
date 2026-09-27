"""
fabry_perot.py
=========================
Fabry-Pérot-Resonator als aktiver Farbfilter für Hydrogel-Systeme.

Gleichungen:
  - Reflexionsamplitude:   r_FP = (r1 + r2·e^(2iδ)) / (1 + r1·r2·e^(2iδ))
  - Reflexionsintensität:  R_FP = |r_FP|²
  - Resonanzbedingung:     2n·d·cos θ = m·λ
  - Finesse:               F = π (R1·R2)^(1/4) / (1 − √(R1·R2))
  - Quellung-Durchstimmbarkeit: λ_res(Q) ≈ 2·d₀·Q
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Sichtbares Spektrum [nm]
VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 780.0

# Standard CIE-Farbmatchingfunktionen (vereinfachte Gauss-Näherung)
def _cie_x_bar(lam_nm: np.ndarray) -> np.ndarray:
    """CIE x̄(λ) – Näherung durch zwei Gaussfunktionen."""
    return (1.056 * np.exp(-0.5 * ((lam_nm - 600) / 37.0)**2)
            + 0.362 * np.exp(-0.5 * ((lam_nm - 450) / 20.0)**2))

def _cie_y_bar(lam_nm: np.ndarray) -> np.ndarray:
    """CIE ȳ(λ) – Luminositätsfunktion."""
    return np.exp(-0.5 * ((lam_nm - 555) / 40.0)**2)

def _cie_z_bar(lam_nm: np.ndarray) -> np.ndarray:
    """CIE z̄(λ)."""
    return 1.217 * np.exp(-0.5 * ((lam_nm - 445) / 25.0)**2)


@dataclass
class FabryPerotLayer:
    """Einzelne Schicht in einem Fabry-Pérot-System.

    Parameters
    ----------
    n : float | complex
        Brechungsindex (komplex für absorbierende Medien).
    d : float
        Schichtdicke [m].
    label : str
        Beschriftung.
    """
    n: complex
    d: float
    label: str = ""


class FabryPerotResonator:
    """Fabry-Pérot-Resonator mit Hydrogel-Kavität.

    Parameters
    ----------
    d0 : float
        Trockene Kavitätsdicke [m].
    n0 : float
        Brechungsindex des trockenen Polymers.
    R1, R2 : float
        Intensitätsreflexivitäten der beiden Spiegel (0 < R < 1).
    theta_i : float
        Einfallswinkel [Grad]. Default: 0° (senkrechter Einfall).
    """

    def __init__(
        self,
        d0: float = 250e-9,
        n0: float = 1.40,
        R1: float = 0.5,
        R2: float = 0.5,
        theta_i: float = 0.0,
    ) -> None:
        self.d0 = d0
        self.n0 = n0
        self.R1 = R1
        self.R2 = R2
        self.theta_i = np.radians(theta_i)

    # ── Optische Parameter ─────────────────────────────────────────────────────

    def cavity_thickness(self, Q: float = 1.0) -> float:
        """Kavitätsdicke d(Q) = Q · d₀  [m].

        Parameters
        ----------
        Q : float
            Quellungsgrad (1 = trocken).
        """
        return Q * self.d0

    def effective_index(self, Q: float = 1.0) -> float:
        """Effektiver Brechungsindex n_eff(Q) ≈ 1 + (n₀ − 1) / Q.

        Gleichung (n-quellung) der Arbeit.
        """
        return 1.0 + (self.n0 - 1.0) / Q

    def resonance_wavelength(self, Q: float = 1.0, order: int = 1) -> float:
        """Resonanzwellenlänge λ_res = 2 n_eff d / m  [m].

        Parameters
        ----------
        Q : float
            Quellungsgrad.
        order : int
            Beugungsordnung m.

        Returns
        -------
        float
            Resonanzwellenlänge in Metern.
        """
        n = self.effective_index(Q)
        d = self.cavity_thickness(Q)
        cos_t = np.cos(self.theta_i)
        return 2.0 * n * d * cos_t / order

    # ── Spektrum ───────────────────────────────────────────────────────────────

    def phase_thickness(self, wavelength: float, Q: float = 1.0) -> float:
        """Phasendicke δ = 2π n d cos θ / λ.

        Parameters
        ----------
        wavelength : float
            Wellenlänge [m].
        Q : float
            Quellungsgrad.

        Returns
        -------
        float
            Phasendicke in Radiant.
        """
        n = self.effective_index(Q)
        d = self.cavity_thickness(Q)
        cos_t = np.cos(self.theta_i)
        return 2.0 * np.pi * n * d * cos_t / wavelength

    def reflection_amplitude(self, wavelength: float, Q: float = 1.0) -> complex:
        """Reflexionsamplitude r_FP.

        r_FP = (r1 + r2·e^(2iδ)) / (1 + r1·r2·e^(2iδ))
        """
        r1 = np.sqrt(self.R1)
        r2 = np.sqrt(self.R2)
        delta = self.phase_thickness(wavelength, Q)
        exp_2d = np.exp(2j * delta)
        return (r1 + r2 * exp_2d) / (1.0 + r1 * r2 * exp_2d)

    def reflection_intensity(self, wavelength: float, Q: float = 1.0) -> float:
        """Reflexionsintensität R_FP = |r_FP|²."""
        return abs(self.reflection_amplitude(wavelength, Q)) ** 2

    def reflection_spectrum(
        self,
        Q: float = 1.0,
        lam_min_nm: float = VISIBLE_MIN_NM,
        lam_max_nm: float = VISIBLE_MAX_NM,
        n_points: int = 400,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reflexionsspektrum R(λ) für gegebenen Quellungsgrad.

        Parameters
        ----------
        Q : float
            Quellungsgrad.
        lam_min_nm, lam_max_nm : float
            Spektraler Bereich [nm].
        n_points : int
            Anzahl Stützstellen.

        Returns
        -------
        (wavelengths_nm, R) : tuple[ndarray, ndarray]
        """
        lam_nm = np.linspace(lam_min_nm, lam_max_nm, n_points)
        lam_m = lam_nm * 1e-9
        R = np.array([self.reflection_intensity(l, Q) for l in lam_m])
        return lam_nm, R

    def tunable_spectra(
        self,
        Q_values: list[float],
        **kwargs,
    ) -> dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Reflexionsspektren für eine Liste von Quellungsgraden.

        Parameters
        ----------
        Q_values : list[float]
            Liste der Quellungsgrade.

        Returns
        -------
        dict : {Q: (lam_nm, R)}
        """
        return {Q: self.reflection_spectrum(Q=Q, **kwargs) for Q in Q_values}

    # ── Finesse und Bandbreite ─────────────────────────────────────────────────

    @property
    def finesse(self) -> float:
        """Finesse F = π (R1·R2)^(1/4) / (1 − √(R1·R2))."""
        sqrt_r1r2 = np.sqrt(self.R1 * self.R2)
        return np.pi * (self.R1 * self.R2) ** 0.25 / (1.0 - sqrt_r1r2)

    def fwhm_nm(self, Q: float = 1.0, order: int = 1) -> float:
        """Halbwertsbreite FWHM = FSR / Finesse  [nm].

        Parameters
        ----------
        Q : float
            Quellungsgrad.
        order : int
            Beugungsordnung.

        Returns
        -------
        float
            FWHM in nm.
        """
        lam_res = self.resonance_wavelength(Q, order)
        fsr = lam_res / order  # Free Spectral Range für m=1
        return (fsr / self.finesse) * 1e9  # m → nm

    # ── Kolorimetrie ───────────────────────────────────────────────────────────

    def cie_xy(self, Q: float = 1.0) -> Tuple[float, float]:
        """CIE-Farbkoordinaten (x, y) für gegebenen Quellungsgrad.

        Parameters
        ----------
        Q : float
            Quellungsgrad.

        Returns
        -------
        (x, y) : tuple[float, float]
            CIE 1931-Farbkoordinaten.
        """
        lam_nm, R = self.reflection_spectrum(Q=Q)
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        X = _trapz(R * _cie_x_bar(lam_nm), lam_nm)
        Y = _trapz(R * _cie_y_bar(lam_nm), lam_nm)
        Z = _trapz(R * _cie_z_bar(lam_nm), lam_nm)
        total = X + Y + Z
        if total == 0:
            return 0.333, 0.333
        return X / total, Y / total

    def color_gamut(
        self,
        Q_range: Tuple[float, float] = (1.0, 4.0),
        n_Q: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CIE-Farbgamut als Funktion des Quellungsgrades.

        Returns
        -------
        (Q_arr, x_arr, y_arr)
        """
        Q_arr = np.linspace(Q_range[0], Q_range[1], n_Q)
        xy = np.array([self.cie_xy(Q) for Q in Q_arr])
        return Q_arr, xy[:, 0], xy[:, 1]

    def __repr__(self) -> str:
        return (
            f"FabryPerotResonator(d₀={self.d0*1e9:.0f} nm, "
            f"n₀={self.n0:.2f}, R1={self.R1:.2f}, R2={self.R2:.2f}, "
            f"F={self.finesse:.2f})"
        )
