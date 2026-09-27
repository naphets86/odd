"""
bulk_modulus.py
===============
Kompressionsmodul hydraulischer Flüssigkeiten.

Alle Gleichungsreferenzen beziehen sich auf *Elastizität hydraulischer Öle
und Flüssigkeiten im industriellen Maschinenbau* (Epp, 2026).

SI-Einheiten durchgängig:
  Druck / Modul  [Pa]
  Temperatur     [K]
  Dichte         [kg/m³]
  spez. Wärme    [J/(kg·K)]
  therm. Ausdehnung [1/K]
  Schallgeschw.  [m/s]
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Gl. (2) – Isentroper Modul
# ---------------------------------------------------------------------------

def isentropic_modulus(
    K_T: float,
    alpha_V: float,
    T: float,
    rho: float,
    c_p: float,
) -> float:
    """Isentroper (adiabatischer) Kompressionsmodul K_s.

    K_s = K_T * (1 + alpha_V² * T * K_T / (rho * c_p))   (Gl. 3)

    Parameters
    ----------
    K_T:     Isothermer Kompressionsmodul [Pa]
    alpha_V: Thermischer Volumenausdehnungskoeffizient [1/K]
    T:       Absolute Temperatur [K]
    rho:     Dichte [kg/m³]
    c_p:     Spezifische Wärmekapazität [J/(kg·K)]

    Returns
    -------
    K_s [Pa]
    """
    return K_T * (1.0 + (alpha_V**2 * T * K_T) / (rho * c_p))


# ---------------------------------------------------------------------------
# Gl. (1) – Schallgeschwindigkeit
# ---------------------------------------------------------------------------

def sound_speed(K_s: float, rho: float) -> float:
    """Isentrope Schallgeschwindigkeit c_s = sqrt(K_s / rho).   (Gl. 2)

    Parameters
    ----------
    K_s: Isentroper Kompressionsmodul [Pa]
    rho: Dichte [kg/m³]

    Returns
    -------
    c_s [m/s]
    """
    if K_s < 0:
        raise ValueError("K_s muss nicht-negativ sein.")
    if rho <= 0:
        raise ValueError("Dichte rho muss positiv sein.")
    return math.sqrt(K_s / rho)


# ---------------------------------------------------------------------------
# Gl. (4) – Tait-Linearisierung  (gültig bis ~300 bar)
# ---------------------------------------------------------------------------

def bulk_modulus_tait_linear(K0: float, n: float, P: float) -> float:
    """Tangent-Bulk-Modul nach linearer Tait-Gleichung.

       K_T(P) = K0 + n * P   (Gl. 4)

    Parameters
    ----------
    K0: Modul bei Umgebungsdruck [Pa]
    n:  Dimensionsloser Druckkoeffizient (typ. 9.5–11.5)
    P:  Betriebsdruck [Pa]

    Returns
    -------
    K_T [Pa]
    """
    return K0 + n * P


# ---------------------------------------------------------------------------
# Gl. (5) – Dowson-Higginson  (für P bis ~700 bar)
# ---------------------------------------------------------------------------

def bulk_modulus_dowson_higginson(K0: float, n: float, P: float) -> float:
    """Tangent-Bulk-Modul nach Dowson-Higginson.

       K_T(P) = K0 * exp(n * P / K0)   (Gl. 5)

    Parameters
    ----------
    K0: Modul bei Umgebungsdruck [Pa]
    n:  Dimensionsloser Druckkoeffizient
    P:  Betriebsdruck [Pa]

    Returns
    -------
    K_T [Pa]
    """
    if K0 <= 0:
        raise ValueError("K0 muss positiv sein.")
    return K0 * math.exp(n * P / K0)


# ---------------------------------------------------------------------------
# Gl. (6) – Temperaturabhängigkeit
# ---------------------------------------------------------------------------

def bulk_modulus_temperature(
    K_ref: float,
    beta_K: float,
    T: float,
    T_ref: float,
) -> float:
    """Temperaturabhängiger Kompressionsmodul.

       K0(T) = K_ref * [1 - beta_K * (T - T_ref)]   (Gl. 6)

    Parameters
    ----------
    K_ref:  Referenzmodul [Pa]
    beta_K: Thermischer Kompressibilitätskoeffizient [1/K]
            (typ. 3.5e-3 K⁻¹ für Mineralöle)
    T:      Betriebstemperatur [K]
    T_ref:  Referenztemperatur [K]

    Returns
    -------
    K0(T) [Pa]
    """
    return K_ref * (1.0 - beta_K * (T - T_ref))


# ---------------------------------------------------------------------------
# Gl. (11, 12) – Effektiver Kompressionsmodul bei Gaseinschlüssen
# ---------------------------------------------------------------------------

def effective_bulk_modulus(K_oil: float, P: float, alpha: float) -> float:
    """Effektiver Kompressionsmodul einer Öl-Gas-Mischung.

       1/K_eff = (1-alpha)/K_oil + alpha/K_gas
    mit K_gas ≈ P (ideales Gas) vereinfacht zu
       K_eff ≈ K_oil * P / (K_oil * alpha + P)   (Gl. 12)

    Parameters
    ----------
    K_oil: Kompressionsmodul des reinen Öls [Pa]
    P:     Betriebsdruck [Pa]  (entspricht K_gas bei idealem Gas)
    alpha: Freier Gasvolumenanteil (0 ≤ alpha < 1)  [dimensionslos]

    Returns
    -------
    K_eff [Pa]
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha muss im Bereich [0, 1) liegen.")
    if P <= 0:
        raise ValueError("P muss positiv sein.")
    if K_oil <= 0:
        raise ValueError("K_oil muss positiv sein.")
    return (K_oil * P) / (K_oil * alpha + P)
