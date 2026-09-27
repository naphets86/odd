"""
surface.py
==========
Oberflächenspannung, Laplace-Druck und Kavitation.

Gleichungen aus Epp (2026):
  Gl. (13)  Laplace-Druck      ΔP_L = 2γ / r
  Gl. (16)  Kavitationsdruck   P_Kav = P_Dampf + 2γ / r_Keim
  Gl. (10)  Henry-Gesetz       c_gas = H · P
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Laplace-Druck
# ---------------------------------------------------------------------------

def laplace_pressure(gamma: float, r: float) -> float:
    """Laplace-Innendruck einer sphärischen Blase.

       ΔP_L = 2γ / r   (Gl. 13)

    Parameters
    ----------
    gamma: Oberflächenspannung [N/m]
    r:     Blasenradius [m]

    Returns
    -------
    ΔP_L [Pa]
    """
    if r <= 0:
        raise ValueError("Blasenradius r muss positiv sein.")
    if gamma < 0:
        raise ValueError("Oberflächenspannung gamma muss nicht-negativ sein.")
    return 2.0 * gamma / r


# ---------------------------------------------------------------------------
# Kavitationsdruck
# ---------------------------------------------------------------------------

def cavitation_pressure(
    P_vapor: float,
    gamma: float,
    r_keim: float,
) -> float:
    """Schwellendruck für Kavitationseinsetzen.

       P_Kav = P_Dampf + 2γ / r_Keim   (Gl. 16)

    Parameters
    ----------
    P_vapor: Dampfdruck des Öls [Pa]
    gamma:   Oberflächenspannung [N/m]
    r_keim:  Keimradius (Rauheitsspitze, Kleinstblase) [m]

    Returns
    -------
    P_Kav [Pa]
    """
    return P_vapor + laplace_pressure(gamma, r_keim)


# ---------------------------------------------------------------------------
# Henry-Gesetz
# ---------------------------------------------------------------------------

def henry_gas_concentration(H: float, P: float) -> float:
    """Gelöster Gasgehalt nach Henry'schem Gesetz.

       c_gas = H · P   (Gl. 10)

    Parameters
    ----------
    H: Henry-Konstante [vol%/Pa]  (für Luft in Mineralöl ≈ 9.5e-6 vol%/Pa
       bei 40 °C, wenn P in Pa; oder 9.5 vol%/bar mit P in bar)
    P: Absolutdruck [Pa]  (oder bar, konsistent mit H)

    Returns
    -------
    c_gas [vol% oder andere von H abhängige Einheit]
    """
    return H * P


# ---------------------------------------------------------------------------
# Mindestdruck zur Kavitationsvermeidung (Empfehlung)
# ---------------------------------------------------------------------------

def min_safe_pressure(P_vapor: float, safety_factor: float = 3.0) -> float:
    """Empfohlener Mindestbetriebsdruck zur Kavitationsvermeidung.

       P_min = safety_factor · P_Dampf   (Tabelle optimale Betriebsparameter)

    Parameters
    ----------
    P_vapor:       Dampfdruck des Hydrauliköls [Pa]
    safety_factor: Sicherheitsfaktor (default 3, gem. Tab. 9.1)

    Returns
    -------
    P_min [Pa]
    """
    return safety_factor * P_vapor
