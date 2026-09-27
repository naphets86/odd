"""
lifetime.py
===========
Lebensdauerprognose und Alterungsmodelle für Hydrauliköle.

Gleichungen aus Epp (2026):
  Gl. (28)  Hydraulische Lebensdauerformel (Coffin-Manson-Analogie)
  Gl. (29)  Druckkorrektur der Aktivierungsenergie
  Gl. (30)  Viskositätsdrift und Kompressionsmodul-Alterung

SI-Einheiten:
  Zeit              [s] (intern), Ausgabe wahlweise [h]
  Temperatur        [K]
  Energie/Aktivierung [eV] oder [J]
  Boltzmann k_B     1.380649e-23 J/K  bzw.  8.617333e-5 eV/K
"""

from __future__ import annotations

import math

# Boltzmann-Konstante
K_B_EV: float = 8.617333e-5   # eV/K
K_B_J: float = 1.380649e-23   # J/K


# ---------------------------------------------------------------------------
# Lebensdauerformel
# ---------------------------------------------------------------------------

def oil_lifetime(
    C_hyd: float,
    delta_eta_irrev: float,
    eta_0: float,
    beta: float,
    E_a: float,
    T: float,
    k_B: float = K_B_EV,
) -> float:
    """Hydraulische Lebensdauerformel (Coffin-Manson-Analogie).

       t_L = C_hyd · (Δη_irrev / η_0)^(−β) · exp(E_a / (k_B · T))   (Gl. 28)

    Parameters
    ----------
    C_hyd:           Flüssigkeitsspez. Konstante [h]
    delta_eta_irrev: Irreversible Viskositätsänderung/h [Pa·s/h]
                     (durch Oxidation, Thermospaltung, Additivverbrauch)
    eta_0:           Nenn-Viskosität Frischöl bei 40 °C [Pa·s]
    beta:            Ermüdungsexponent (typ. 2.0–2.5)
    E_a:             Aktivierungsenergie der Oxidation [eV] (wenn k_B in eV/K)
    T:               Betriebstemperatur [K]
    k_B:             Boltzmann-Konstante [eV/K] oder [J/K] – muss zu E_a passen.

    Returns
    -------
    t_L [h]
    """
    if delta_eta_irrev <= 0:
        raise ValueError("delta_eta_irrev muss positiv sein.")
    if eta_0 <= 0:
        raise ValueError("eta_0 muss positiv sein.")
    if T <= 0:
        raise ValueError("T muss positiv sein (absolute Temperatur).")
    ratio = delta_eta_irrev / eta_0
    return C_hyd * ratio ** (-beta) * math.exp(E_a / (k_B * T))


# ---------------------------------------------------------------------------
# Druckkorrektur der Aktivierungsenergie
# ---------------------------------------------------------------------------

def activation_energy_pressure(
    E_a0: float,
    P: float,
    P_star: float = 1e8,  # ≈ 1000 bar in Pa
) -> float:
    """Druckabhängige Aktivierungsenergie der Oxidationsreaktion.

       E_a(P) = E_a0 · (1 − P / P*)   (Gl. aus Bemerkung Druckkorrektur)

    Parameters
    ----------
    E_a0:   Aktivierungsenergie bei Umgebungsdruck [eV oder J]
    P:      Betriebsdruck [Pa]
    P_star: Referenzdruck (typ. 1000 bar = 1e8 Pa)

    Returns
    -------
    E_a(P)  [gleiche Einheit wie E_a0]
    """
    if P_star <= 0:
        raise ValueError("P_star muss positiv sein.")
    return E_a0 * (1.0 - P / P_star)


# ---------------------------------------------------------------------------
# Alterungsbedingter Rückgang des Kompressionsmoduls
# ---------------------------------------------------------------------------

def bulk_modulus_aging(
    K0: float,
    delta_eta: float,
    eta_0: float,
    delta_K_coeff: float = 0.15,
) -> float:
    """Reduzierter Kompressionsmodul infolge Ölalterung.

       K_alt / K0 ≈ 1 − δ_K · (Δη / η_0)^0.5   (Gl. 30)

    Parameters
    ----------
    K0:             Frischöl-Kompressionsmodul [Pa]
    delta_eta:      Kumulative Viskositätsänderung [Pa·s]
                    (positiv = Viskositätszunahme)
    eta_0:          Nenn-Viskosität Frischöl [Pa·s]
    delta_K_coeff:  Alterungskoeffizient δ_K (typ. 0.15 für Mineralöl)

    Returns
    -------
    K_alt [Pa]
    """
    if eta_0 <= 0:
        raise ValueError("eta_0 muss positiv sein.")
    factor = 1.0 - delta_K_coeff * math.sqrt(abs(delta_eta) / eta_0)
    return K0 * factor


# ---------------------------------------------------------------------------
# Versagensraten-Modelle (drei Mechanismen)
# ---------------------------------------------------------------------------

def failure_rate_oxidation(T: float, T_opt: float, T0: float, A1: float = 1.0) -> float:
    """Versagensrate durch thermisch-oxidative Degradation.

       Ṅ_ox(T) = A1 · exp((T − T_opt) / T0)   (Gl. Versagen Oxidation)

    Parameters
    ----------
    T:     Betriebstemperatur [K]
    T_opt: Optimale Betriebstemperatur [K]
    T0:    Charakteristische Temperatur [K]
    A1:    Vorfaktor [1/h]

    Returns
    -------
    Ṅ_ox [1/h]
    """
    if T0 <= 0:
        raise ValueError("T0 muss positiv sein.")
    return A1 * math.exp((T - T_opt) / T0)


def failure_rate_friction(T: float, T_min: float, T0: float, A2: float = 1.0) -> float:
    """Versagensrate durch Viskositätsmangel / Mischreibung.

       Ṅ_Reib(T) = A2 · exp(−(T − T_min) / T0)   (Gl. Versagen Reibung)

    Parameters
    ----------
    T:     Betriebstemperatur [K]
    T_min: Mindesttemperatur für ausreichende Schmierfähigkeit [K]
    T0:    Charakteristische Temperatur [K]
    A2:    Vorfaktor [1/h]

    Returns
    -------
    Ṅ_Reib [1/h]
    """
    if T0 <= 0:
        raise ValueError("T0 muss positiv sein.")
    return A2 * math.exp(-(T - T_min) / T0)


def failure_rate_cavitation(
    P_min: float,
    P_vapor: float,
    P0: float,
    A3: float = 1.0,
) -> float:
    """Versagensrate durch Kavitationserosion.

       Ṅ_Kav(P) = A3 · exp(−(P_min − P_Dampf) / P0)   (Gl. Versagen Kavitation)

    Parameters
    ----------
    P_min:   Mindestdruck im System [Pa]
    P_vapor: Dampfdruck des Öls [Pa]
    P0:      Charakteristischer Druckparameter [Pa]
    A3:      Vorfaktor [1/h]

    Returns
    -------
    Ṅ_Kav [1/h]
    """
    if P0 <= 0:
        raise ValueError("P0 muss positiv sein.")
    return A3 * math.exp(-(P_min - P_vapor) / P0)


def total_failure_rate(
    T: float,
    T_opt: float,
    T_min: float,
    T0: float,
    P_min: float,
    P_vapor: float,
    P0: float,
    A1: float = 1.0,
    A2: float = 1.0,
    A3: float = 1.0,
) -> float:
    """Gesamtversagensrate aus drei Mechanismen.

       Ṅ_ges = Ṅ_ox + Ṅ_Reib + Ṅ_Kav

    Returns
    -------
    Ṅ_ges [1/h]
    """
    return (
        failure_rate_oxidation(T, T_opt, T0, A1)
        + failure_rate_friction(T, T_min, T0, A2)
        + failure_rate_cavitation(P_min, P_vapor, P0, A3)
    )
