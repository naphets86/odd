"""
hydraulics.py
=============
Hydraulische Systemdynamik, Positioniergenauigkeit und Sicherheit.

Gleichungen aus Epp (2026):
  Gl. (23)  Kompressibilitätsfehler δx
  Gl. (24)  Hydraulische Eigenfrequenz ω_h / f_h
  Gl. (25)  SEA-Steifigkeit K_SEA
  Gl. (26)  Kompressionsenergie E_Kompr
  Gl. (29)  Max. Kontaktkraft F_max (Cobot-Sicherheit)
  Gl. (30)  Akkumulatorkonstante K_Acc

SI-Einheiten durchgängig.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Kompressibilitätsfehler
# ---------------------------------------------------------------------------

def compressibility_error(P: float, V: float, K_eff: float, A: float) -> float:
    """Positionsfehler durch Ölkompressibilität in einem Linearantrieb.

       δx = P · V / (K_eff · A)   (Gl. 23)

    Parameters
    ----------
    P:     Betriebsdruck [Pa]
    V:     Zylinder-Nutzvolumen [m³]
    K_eff: Effektiver Kompressionsmodul [Pa]
    A:     Kolbenfläche [m²]

    Returns
    -------
    δx [m]
    """
    if K_eff <= 0:
        raise ValueError("K_eff muss positiv sein.")
    if A <= 0:
        raise ValueError("A muss positiv sein.")
    return P * V / (K_eff * A)


# ---------------------------------------------------------------------------
# Hydraulische Eigenfrequenz
# ---------------------------------------------------------------------------

def hydraulic_eigenfrequency(
    K_eff: float,
    A: float,
    m: float,
    V: float,
) -> float:
    """Hydraulische Eigenfrequenz ω_h eines Zylinderantriebs.

       ω_h = sqrt(K_eff · A² / (m · V))   (Gl. 24)

    Parameters
    ----------
    K_eff: Effektiver Kompressionsmodul [Pa]
    A:     Kolbenfläche [m²]
    m:     Bewegte Masse [kg]
    V:     Gesamtvolumen unter Druck [m³]

    Returns
    -------
    ω_h [rad/s]
    """
    if K_eff <= 0 or A <= 0 or m <= 0 or V <= 0:
        raise ValueError("K_eff, A, m und V müssen positiv sein.")
    return math.sqrt(K_eff * A**2 / (m * V))


def hydraulic_natural_frequency(
    K_eff: float,
    A: float,
    m: float,
    V: float,
) -> float:
    """Hydraulische Eigenfrequenz f_h = ω_h / (2π)  [Hz].

    Parameters
    ----------
    K_eff: Effektiver Kompressionsmodul [Pa]
    A:     Kolbenfläche [m²]
    m:     Bewegte Masse [kg]
    V:     Gesamtvolumen unter Druck [m³]

    Returns
    -------
    f_h [Hz]
    """
    return hydraulic_eigenfrequency(K_eff, A, m, V) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Series Elastic Actuation (SEA)
# ---------------------------------------------------------------------------

def sea_stiffness(K_eff: float, A: float, V_comp: float) -> float:
    """Virtuelle Gelenksteifigkeit beim Series-Elastic-Actuation-Prinzip.

       K_SEA = K_eff · A² / V_comp   (Gl. 25)

    Parameters
    ----------
    K_eff:   Effektiver Kompressionsmodul [Pa]
    A:       Kolbenfläche [m²]
    V_comp:  Kompressionsvolumen (Akkumulator) [m³]

    Returns
    -------
    K_SEA [N/m]
    """
    if K_eff <= 0 or A <= 0 or V_comp <= 0:
        raise ValueError("K_eff, A und V_comp müssen positiv sein.")
    return K_eff * A**2 / V_comp


# ---------------------------------------------------------------------------
# Maximale Kontaktkraft (Cobot-Sicherheit)
# ---------------------------------------------------------------------------

def max_contact_force(v0: float, K_SEA: float, m: float) -> float:
    """Maximale Kontaktkraft bei elastischem Anprall.

       F_max = v0 · sqrt(K_SEA · m)   (Gl. 29, Satz 2)

    Parameters
    ----------
    v0:    Annäherungsgeschwindigkeit [m/s]
    K_SEA: SEA-Steifigkeit [N/m]
    m:     Bewegte Masse [kg]

    Returns
    -------
    F_max [N]
    """
    if K_SEA < 0 or m < 0:
        raise ValueError("K_SEA und m müssen nicht-negativ sein.")
    return v0 * math.sqrt(K_SEA * m)


# ---------------------------------------------------------------------------
# Kompressionsenergie
# ---------------------------------------------------------------------------

def compression_energy(P: float, V: float, K_eff: float) -> float:
    """Gespeicherte / verlorene Energie durch Ölkompression.

       E_Kompr = P² · V / (2 · K_eff)   (Gl. 26)

    Parameters
    ----------
    P:     Betriebsdruck [Pa]
    V:     Komprimiertes Volumen [m³]
    K_eff: Effektiver Kompressionsmodul [Pa]

    Returns
    -------
    E_Kompr [J]
    """
    if K_eff <= 0:
        raise ValueError("K_eff muss positiv sein.")
    return P**2 * V / (2.0 * K_eff)


# ---------------------------------------------------------------------------
# Akkumulatorkonstante (adiabat, Blasenakkumulator)
# ---------------------------------------------------------------------------

def accumulator_constant(
    V0: float,
    P0: float,
    V: float,
    kappa: float = 1.4,
) -> float:
    """Steifigkeit (Federkonstante) eines Blasenakkumulators.

       K_Acc = V0 · P0^κ / V^(κ+1)   (Gl. 30)

    Parameters
    ----------
    V0:    Gasvolumen beim Vorspanndruck [m³]
    P0:    Vorspanndruck [Pa]
    V:     Aktuelles Gasvolumen [m³]
    kappa: Adiabatenexponent (Luft ≈ 1.4)

    Returns
    -------
    K_Acc [N/m³] (druckbezogene Steifigkeit)
    """
    if V0 <= 0 or P0 <= 0 or V <= 0:
        raise ValueError("V0, P0 und V müssen positiv sein.")
    return V0 * P0**kappa / V ** (kappa + 1.0)


# ---------------------------------------------------------------------------
# Optimales Betriebsdruckfenster (Satz 1)
# ---------------------------------------------------------------------------

def max_operating_pressure(P_N: float, factor: float = 0.7) -> float:
    """Obere Grenze des empfohlenen Betriebsdruckfensters.

       P_max_empf = factor · P_N   (Satz 1, Gl. Betriebsdruckfenster)

    Parameters
    ----------
    P_N:    Nenndruck des Systems [Pa]
    factor: Sicherheitsfaktor (default 0.70)

    Returns
    -------
    Empfohlener Maximaldruck [Pa]
    """
    return factor * P_N


# ---------------------------------------------------------------------------
# Lager-Grenzfrequenz
# ---------------------------------------------------------------------------

def bearing_cutoff_frequency(
    c_L: float,
    A: float,
    rho: float,
    c_s: float,
) -> float:
    """Grenzfrequenz eines hydrostatischen Lagers.

       ω_L = c_L / (A · ρ · c_s)   (Gl. aus Abschnitt 6.1)

    Parameters
    ----------
    c_L: Statische Lagersteifigkeit [N/m]
    A:   Lagerfläche [m²]
    rho: Dichte der Flüssigkeit [kg/m³]
    c_s: Schallgeschwindigkeit [m/s]

    Returns
    -------
    ω_L [rad/s]
    """
    if A <= 0 or rho <= 0 or c_s <= 0:
        raise ValueError("A, rho und c_s müssen positiv sein.")
    return c_L / (A * rho * c_s)
