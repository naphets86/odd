"""
viscosity.py
============
Viskositätsmodelle für hydraulische Flüssigkeiten.

Gleichungen aus Epp (2026):
  Gl.  (7)  Newton'sches Fluid   τ = η · γ̇
  Gl. (14)  Relaxationszeit      τ_R = η / (ρ c_s²)
  Gl. (15)  Relaxationsfrequenz  f_R = ρ c_s² / (2π η)
  Barus     η(P) = η_0 * exp(α_P * P)
  Gl. (18)  VII-Viskosität
  Gl. (27)  VII-Scherabbau

SI-Einheiten:
  Viskosität  [Pa·s]
  Druck       [Pa]
  Temperatur  [K]
  Zeit        [s]
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Newton'sches Fluid
# ---------------------------------------------------------------------------

def shear_stress(eta: float, gamma_dot: float) -> float:
    """Schubspannung nach Newton: τ = η · γ̇.   (Gl. 7)

    Parameters
    ----------
    eta:       Dynamische Viskosität [Pa·s]
    gamma_dot: Scherrate [1/s]

    Returns
    -------
    τ [Pa]
    """
    return eta * gamma_dot


# ---------------------------------------------------------------------------
# Relaxationszeit und -frequenz
# ---------------------------------------------------------------------------

def relaxation_time(eta: float, rho: float, c_s: float) -> float:
    """Charakteristische Relaxationszeit τ_R = η / (ρ · c_s²).   (Gl. 14)

    Parameters
    ----------
    eta: Dynamische Viskosität [Pa·s]
    rho: Dichte [kg/m³]
    c_s: Schallgeschwindigkeit [m/s]

    Returns
    -------
    τ_R [s]
    """
    if rho <= 0:
        raise ValueError("rho muss positiv sein.")
    if c_s <= 0:
        raise ValueError("c_s muss positiv sein.")
    return eta / (rho * c_s**2)


def relaxation_frequency(rho: float, c_s: float, eta: float) -> float:
    """Relaxationsfrequenz f_R = ρ c_s² / (2π η).   (Gl. 15)

    Viskoelastische Effekte werden erst ab f > f_R relevant.

    Parameters
    ----------
    rho: Dichte [kg/m³]
    c_s: Schallgeschwindigkeit [m/s]
    eta: Dynamische Viskosität [Pa·s]

    Returns
    -------
    f_R [Hz]
    """
    if eta <= 0:
        raise ValueError("eta muss positiv sein.")
    return (rho * c_s**2) / (2.0 * math.pi * eta)


# ---------------------------------------------------------------------------
# Barus-Gleichung (Druckabhängigkeit der Viskosität)
# ---------------------------------------------------------------------------

def barus_viscosity(eta_0: float, alpha_P: float, P: float) -> float:
    """Druckabhängige Viskosität nach Barus.

       η(P) = η_0 * exp(α_P * P)

    Parameters
    ----------
    eta_0:   Viskosität bei Atmosphärendruck [Pa·s]
    alpha_P: Druckviskositätskoeffizient [1/Pa]
             (typ. 2e-8 Pa⁻¹ für Mineralöl)
    P:       Absolutdruck [Pa]

    Returns
    -------
    η(P) [Pa·s]
    """
    return eta_0 * math.exp(alpha_P * P)


# ---------------------------------------------------------------------------
# VII-bedingte Viskositätserhöhung
# ---------------------------------------------------------------------------

def vii_viscosity(
    eta_basis: float,
    c_vii: float,
    rho_vii: float,
    M_c: float,
    M_e: float,
) -> float:
    """Viskosität mit VII-Additiv-Beitrag.

       η_VII = η_basis * exp(c_VII / ρ_VII * M_c / M_e)   (Gl. 18)

    Parameters
    ----------
    eta_basis: Viskosität des Basisöls (ohne VII) [Pa·s]
    c_vii:     Massenkonzentration der VII-Polymere [kg/m³]
    rho_vii:   Dichte der VII-Polymere [kg/m³]
    M_c:       Molmasse der VII-Polymerkette [g/mol]
    M_e:       Verschlaufungsmolmasse [g/mol]

    Returns
    -------
    η_VII [Pa·s]
    """
    if rho_vii <= 0:
        raise ValueError("rho_vii muss positiv sein.")
    if M_e <= 0:
        raise ValueError("M_e muss positiv sein.")
    exponent = (c_vii / rho_vii) * (M_c / M_e)
    return eta_basis * math.exp(exponent)


# ---------------------------------------------------------------------------
# VII-Scherabbau  (permanente Scherverdünnung)
# ---------------------------------------------------------------------------

def shear_degradation_viscosity(
    eta_0: float,
    eta_basis: float,
    t: float,
    tau_sch: float,
) -> float:
    """Zeitlicher Viskositätsabfall durch VII-Scherabbau.

       η_Sch(t) = η_0 − (η_0 − η_basis) * (1 − exp(−t / τ_Sch))   (Gl. 27)

    Parameters
    ----------
    eta_0:     Anfangsviskosität (frisches Öl) [Pa·s]
    eta_basis: Basisöl-Viskosität ohne VII [Pa·s]
    t:         Betriebszeit [s]
    tau_sch:   Zeitkonstante des Scherabbaus [s]

    Returns
    -------
    η_Sch(t) [Pa·s]
    """
    if tau_sch <= 0:
        raise ValueError("tau_sch muss positiv sein.")
    return eta_0 - (eta_0 - eta_basis) * (1.0 - math.exp(-t / tau_sch))
