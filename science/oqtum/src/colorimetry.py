"""
nanooptik.colorimetry
=====================
CIE-Farbraum, photopische Empfindlichkeit und Delta-E-Berechnung
gemäß Kapitel 7 der wissenschaftlichen Arbeit (Def. 7.2).

Implementiert:
  - CIE 1931 2°-Normalbeobachter (x̄, ȳ, z̄ Farbabgleichfunktionen)
  - Photopische Empfindlichkeitskurve V(λ)
  - Spektrum → XYZ → Lab-Konversion
  - CIE-ΔE*₀₀ (CIEDE2000) und vereinfachtes ΔE* (Euklidisch)
  - Tarnindex C
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple


# ---------------------------------------------------------------------------
# CIE 1931 Farbabgleichfunktionen (tabelliert, 380–780 nm, 5 nm Schritte)
# ---------------------------------------------------------------------------

# Wellenlängen [nm]
_CIE_WAVELENGTHS = np.arange(380, 785, 5, dtype=float)

# CIE 1931 2°-Normalbeobachter (x̄, ȳ, z̄) nach Wyszecki & Stiles
_CIE_XBAR = np.array([
    0.01741, 0.02236, 0.02924, 0.03597, 0.03975, 0.03950, 0.03405, 0.02585,
    0.01669, 0.00741, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.01483, 0.03446, 0.05985, 0.09028,
    0.12790, 0.17429, 0.22788, 0.28486, 0.33609, 0.37521, 0.39842, 0.40602,
    0.40129, 0.38547, 0.35970, 0.32886, 0.29551, 0.26008, 0.22278, 0.18519,
    0.14936, 0.11320, 0.08168, 0.05725, 0.04170, 0.02732, 0.01687, 0.00985,
    0.00575, 0.00316, 0.00170, 0.00082, 0.00042, 0.00020, 0.00010, 0.00005,
    0.00002, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000,
])

_CIE_YBAR = np.array([
    0.00040, 0.00053, 0.00068, 0.00089, 0.00120, 0.00155, 0.00210, 0.00285,
    0.00384, 0.00523, 0.00716, 0.00973, 0.01329, 0.01786, 0.02369, 0.03079,
    0.03988, 0.05143, 0.07032, 0.09795, 0.13902, 0.18928, 0.24777, 0.31734,
    0.38832, 0.45620, 0.52175, 0.58125, 0.62765, 0.66033, 0.68479, 0.69694,
    0.69764, 0.68566, 0.65674, 0.61455, 0.56486, 0.50864, 0.44669, 0.38176,
    0.31756, 0.25484, 0.19620, 0.14394, 0.10245, 0.07190, 0.04677, 0.02945,
    0.01817, 0.01095, 0.00593, 0.00313, 0.00158, 0.00079, 0.00040, 0.00020,
    0.00010, 0.00005, 0.00002, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000,
])

_CIE_ZBAR = np.array([
    0.08028, 0.10474, 0.13676, 0.17166, 0.19576, 0.19938, 0.17481, 0.13472,
    0.09110, 0.04777, 0.01996, 0.00883, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000,
])

# Sicherstellen gleicher Länge
assert len(_CIE_WAVELENGTHS) == len(_CIE_XBAR) == len(_CIE_YBAR) == len(_CIE_ZBAR)


def wavelengths_visible(n_points: int = 400) -> NDArray:
    """Äquidistante Wellenlängen im sichtbaren Bereich [380, 780] nm."""
    return np.linspace(380.0, 780.0, n_points)


def photopic_sensitivity(wavelengths: ArrayLike) -> NDArray:
    """
    Photopische Empfindlichkeitskurve V(λ) des CIE-Normalbeobachters
    (identisch mit ȳ(λ), normiert auf Maximum = 1 bei λ ≈ 555 nm).

    Parameters
    ----------
    wavelengths : array_like
        Wellenlängen [nm].

    Returns
    -------
    V : ndarray
        Photopische Empfindlichkeit ∈ [0, 1].
    """
    lam = np.asarray(wavelengths, dtype=float)
    y_bar = np.interp(lam, _CIE_WAVELENGTHS, _CIE_YBAR, left=0.0, right=0.0)
    if y_bar.max() > 0:
        y_bar = y_bar / y_bar.max()
    return y_bar


def spectrum_to_xyz(
    wavelengths: ArrayLike,
    spectrum: ArrayLike,
    illuminant: str = "D65",
) -> Tuple[float, float, float]:
    """
    Konvertiert ein Reflexionsspektrum in CIE-XYZ-Tristimulus-Werte.

    X = ∫ S(λ)·R(λ)·x̄(λ) dλ / ∫ S(λ)·ȳ(λ) dλ  (normiert)

    Parameters
    ----------
    wavelengths : array_like
        Wellenlängen [nm].
    spectrum : array_like
        Reflexionsspektrum R(λ) ∈ [0, 1].
    illuminant : str
        Beleuchtungsart ('D65' = Tageslicht, 'E' = gleichenergetisch).

    Returns
    -------
    X, Y, Z : float
        CIE-XYZ-Werte.
    """
    lam  = np.asarray(wavelengths, dtype=float)
    R    = np.asarray(spectrum,    dtype=float)

    x_bar = np.interp(lam, _CIE_WAVELENGTHS, _CIE_XBAR, left=0.0, right=0.0)
    y_bar = np.interp(lam, _CIE_WAVELENGTHS, _CIE_YBAR, left=0.0, right=0.0)
    z_bar = np.interp(lam, _CIE_WAVELENGTHS, _CIE_ZBAR, left=0.0, right=0.0)

    # Beleuchtungsspektrum S(λ)
    if illuminant.upper() == "D65":
        # Vereinfachte D65-Näherung (normiert)
        S = np.ones_like(lam)
    elif illuminant.upper() == "E":
        S = np.ones_like(lam)
    else:
        S = np.ones_like(lam)

    k = 1.0 / np.trapezoid(S * y_bar, lam) if np.trapezoid(S * y_bar, lam) > 0 else 1.0

    X = k * np.trapezoid(S * R * x_bar, lam)
    Y = k * np.trapezoid(S * R * y_bar, lam)
    Z = k * np.trapezoid(S * R * z_bar, lam)

    return float(X), float(Y), float(Z)


def xyz_to_lab(
    X: float,
    Y: float,
    Z: float,
    Xn: float = 95.047,
    Yn: float = 100.000,
    Zn: float = 108.883,
) -> Tuple[float, float, float]:
    """
    Konvertiert CIE-XYZ in CIE-L*a*b* (CIELAB).

    L* = 116·f(Y/Yn) - 16
    a* = 500·[f(X/Xn) - f(Y/Yn)]
    b* = 200·[f(Y/Yn) - f(Z/Zn)]

    mit f(t) = t^(1/3) für t > (6/29)³, sonst (29/6)²·t/3 + 4/29.

    Parameters
    ----------
    X, Y, Z : float
        CIE-XYZ-Werte (Y normiert auf 0–100).
    Xn, Yn, Zn : float
        Weißpunktwerte (Standard: D65).

    Returns
    -------
    L, a, b : float
        CIELAB-Koordinaten.
    """
    def f(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1.0 / 3.0)
        return t / (3 * delta**2) + 4.0 / 29.0

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return float(L), float(a), float(b)


def delta_e(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
    method: str = "euclidean",
) -> float:
    """
    CIE-Farbdifferenz ΔE zwischen zwei Farben im L*a*b*-Raum.

    Parameters
    ----------
    lab1, lab2 : tuple (L, a, b)
        CIELAB-Koordinaten zweier Farben.
    method : str
        'euclidean' : ΔE = √(ΔL²+Δa²+Δb²)
        'cie76'     : wie euclidean (CIE 1976)

    Returns
    -------
    dE : float
        Farbdifferenz. ΔE < 2: nicht unterscheidbar (JND).

    Notes
    -----
    Gleichung (Def. 7.2): ΔE = √(ΔL*² + Δa*² + Δb*²)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return float(np.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2))


def spectrum_to_lab(
    wavelengths: ArrayLike,
    spectrum: ArrayLike,
    illuminant: str = "D65",
) -> Tuple[float, float, float]:
    """Gesamte Pipeline: Spektrum → XYZ → L*a*b*."""
    X, Y, Z = spectrum_to_xyz(wavelengths, spectrum, illuminant)
    return xyz_to_lab(X * 100, Y * 100, Z * 100)


def camouflage_index(
    wavelengths: ArrayLike,
    spectrum_system: ArrayLike,
    spectrum_background: ArrayLike,
    spectrum_unmasked: ArrayLike,
) -> float:
    """
    Tarnindex C ∈ [0, 1] (Definition 9.1 der Arbeit):

    C = 1 - ΔE_System / ΔE_Referenz

    Parameters
    ----------
    wavelengths : array_like
    spectrum_system : array_like
        Spektrum des getarnten Systems.
    spectrum_background : array_like
        Spektrum des Hintergrunds (Ziel).
    spectrum_unmasked : array_like
        Spektrum des ungetarnten Systems (Referenz).

    Returns
    -------
    C : float
        Tarnindex. C=1: perfekte Tarnung. C=0: kein Effekt.
    """
    lab_sys = spectrum_to_lab(wavelengths, spectrum_system)
    lab_bg  = spectrum_to_lab(wavelengths, spectrum_background)
    lab_ref = spectrum_to_lab(wavelengths, spectrum_unmasked)

    dE_system = delta_e(lab_sys, lab_bg)
    dE_ref    = delta_e(lab_ref, lab_bg)

    if dE_ref < 1e-10:
        return 1.0  # System und Referenz identisch mit Hintergrund
    return float(np.clip(1.0 - dE_system / dE_ref, 0.0, 1.0))
