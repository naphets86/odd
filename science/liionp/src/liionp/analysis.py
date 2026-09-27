"""Analytische Lebensdauer- und Verbesserungsfaktorberechnungen.

Implementiert die geschlossenen Ausdrücke aus *liionp.tex*:
  - Thermischer Verbesserungsfaktor Φ_T (Proposition 2, Gl. 29)
  - Spannungsbedingter Verbesserungsfaktor Φ_V (Proposition 2, Gl. 30)
  - Kombinierter Verbesserungsfaktor Φ (Proposition 2, Gl. 31)
  - Erweiterter Faktor Φ_ges mit Plattenabstandskanal (Satz 3, Gl. 58)
  - Reichweiten-Zyklen-Modell (Gl. 37–38)
  - Kapazitätsreduktion bei Kälte (Gl. 39)
  - E-Fahrzeug-Lebensdauer- und Kilometerhilfsrechnungen (Abschnitt 4)
  - Vorheizeffizienz (Abschnitt 4.4)
  - Verbesserungsfaktor aus mittleren Degradationsraten (Gl. 27)
"""
from __future__ import annotations

import numpy as np

from .constants import D_EOL, KAPPA_T, R_GAS, T_REF_COLD


# ── Improvement factors (Proposition 2) ──────────────────────────────────────

def improvement_factor_thermal(
    Ea: float,
    T_AI: float,
    T_static: float,
) -> float:
    """Thermischer Lebensdauer-Verbesserungsfaktor Φ_T (Gl. 29).

    Φ_T = exp( E_a / R · (1/T_AI − 1/T_static) )

    Args:
        Ea:       Aktivierungsenergie [J/mol].
        T_AI:     Mittlere Betriebstemperatur des KI-PMM [K].
        T_static: Mittlere Betriebstemperatur des statischen PMS [K].

    Returns:
        Φ_T > 1, wenn T_AI < T_static.
    """
    return float(np.exp(Ea / R_GAS * (1.0 / T_AI - 1.0 / T_static)))


def improvement_factor_voltage(
    eta_V: float,
    gamma_V_fraction: float,
) -> float:
    """Spannungsbedingter Lebensdauer-Verbesserungsfaktor Φ_V (Gl. 30).

    Φ_V = 1 / (1 − γ_V* · η_V)

    Args:
        eta_V:             Relative Reduktion der Spannungsdegradation ∈ [0, 1).
        gamma_V_fraction:  Relatives Gewicht der Spannungsdegradation γ_V*.

    Returns:
        Φ_V > 1.
    """
    return 1.0 / (1.0 - gamma_V_fraction * eta_V)


def improvement_factor_combined(
    phi_T: float,
    phi_V: float,
    gamma_T_fraction: float,
) -> float:
    """Kombinierter Lebensdauer-Verbesserungsfaktor Φ (Gl. 31).

    Φ = Φ_T^γ_T  ·  Φ_V^γ_V   mit  γ_T + γ_V = 1

    Args:
        phi_T:             Thermischer Verbesserungsfaktor.
        phi_V:             Spannungsbedingter Verbesserungsfaktor.
        gamma_T_fraction:  Gewicht der thermischen Degradation ∈ (0, 1).

    Returns:
        Kombinierter Verbesserungsfaktor Φ.
    """
    gamma_V_fraction = 1.0 - gamma_T_fraction
    return float(phi_T**gamma_T_fraction * phi_V**gamma_V_fraction)


def improvement_factor_total(
    phi: float,
    rho_M: float,
    eta_M: float,
) -> float:
    """Gesamter Verbesserungsfaktor mit Plattenabstandskanal (Satz 3, Gl. 58).

    Φ_ges = 1 / ( (1 − ρ_M) · Φ⁻¹  +  ρ_M · η_M )

    Args:
        phi:   Basis-Verbesserungsfaktor (ohne Plattenabstandskanal).
        rho_M: Relativer Anteil der mechanischen Degradation ∈ [0, 1].
        eta_M: Mechanischer Reduktionsfaktor η_M ∈ (0, 1).

    Returns:
        Erweiterter Verbesserungsfaktor Φ_ges.
    """
    return 1.0 / ((1.0 - rho_M) / phi + rho_M * eta_M)


def phi_factor_from_rates(
    mean_rate_static: float,
    mean_rate_ai: float,
) -> float:
    """Verbesserungsfaktor geschätzt aus mittleren Degradationsraten (Gl. 27).

    Φ ≈ r̄_static / r̄_AI

    Args:
        mean_rate_static: Mittlere Degradationsrate des statischen PMS [1/s].
        mean_rate_ai:     Mittlere Degradationsrate des KI-PMM [1/s].

    Returns:
        Φ = mean_rate_static / mean_rate_ai.
    """
    return mean_rate_static / mean_rate_ai


# ── Range model (Section 4) ───────────────────────────────────────────────────

def range_vs_cycles(
    R0: float,
    n: float,
    n_EoL: float,
    D_EoL: float = D_EOL,
) -> float:
    """Effektive Reichweite als Funktion der Zyklenzahl (Gl. 37).

    R(n) = R₀ · (1 − D_EoL · √(n / n_EoL))

    Args:
        R0:    Anfangsreichweite [km].
        n:     Aktuelle Zyklenzahl.
        n_EoL: Zyklenzahl bei Lebensende (EoL).
        D_EoL: EoL-Degradationsschwellwert (Standard: 0,20).

    Returns:
        Reichweite bei Zyklus *n* [km].
    """
    return float(R0 * (1.0 - D_EoL * np.sqrt(n / n_EoL)))


def range_loss_ratio(phi: float) -> float:
    """Relatives Reichweitenverlustverhältnis: ΔR_AI / ΔR_static (Gl. 38).

    ΔR_AI / ΔR_static = 1 / Φ

    Args:
        phi: Verbesserungsfaktor Φ.

    Returns:
        Reichweitenverlustverhältnis (< 1 bedeutet: KI-PMM verliert Reichweite langsamer).
    """
    return 1.0 / phi


# ── Cold-weather capacity model (eq. 39) ─────────────────────────────────────

def cold_capacity(
    Q0: float,
    T_amb: float,
    T_ref: float = T_REF_COLD,
    kappa_T: float = KAPPA_T,
) -> float:
    """Temperaturabhängige Kapazität bei Umgebungstemperaturen unterhalb des Referenzwerts (Gl. 39).

    Q(T_amb) = Q₀ · (1 − κ_T · max(0, T_ref − T_amb))

    Args:
        Q0:      Nennkapazität [Ah].
        T_amb:   Umgebungstemperatur [K].
        T_ref:   Referenztemperatur, oberhalb derer keine Derating stattfindet [K].
        kappa_T: Kapazitätsverlustkoeffizient [K⁻¹].

    Returns:
        Deratierte Kapazität [Ah].
    """
    delta_T = max(0.0, T_ref - T_amb)
    return float(Q0 * (1.0 - kappa_T * delta_T))


# ── EV lifetime helpers (Section 4.2) ────────────────────────────────────────

def ev_lifetime_years(cycles_to_eol: float, cycles_per_year: float) -> float:
    """Batterielebensdauer in Jahren.

    Args:
        cycles_to_eol:  Vollzyklenzahl bis Lebensende.
        cycles_per_year: Jährliche Zyklenrate [Zyklen/Jahr].

    Returns:
        Lebensdauer [Jahre].
    """
    return cycles_to_eol / cycles_per_year


def ev_cumulative_km(years: float, km_per_year: float) -> float:
    """Gesamt-Fahrkilometer bis zum Akkutausch.

    Args:
        years:       Akkulebensdauer [Jahre].
        km_per_year: Jährliche Fahrleistung [km/Jahr].

    Returns:
        Gesamte Kilometer [km].
    """
    return years * km_per_year


def preheating_efficiency(range_gain: float, energy_heat: float) -> float:
    """Vorheizeffizienz in km pro kWh (Abschnitt 4.4).

    η_heat = range_gain / energy_heat

    Args:
        range_gain:  Reichweitengewinn durch Akkuvorheizung [km].
        energy_heat: Aufgewendete Energie zum Vorheizen [kWh].

    Returns:
        Effizienz [km/kWh].
    """
    return range_gain / energy_heat
