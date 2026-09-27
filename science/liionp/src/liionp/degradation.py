"""Degradationsmodelle für Lithium-Ionen-Akkuzellen.

Implementiert die Gleichungen aus *liionp.tex*:
  - SEI-Wachstum (Gl. 3–5)
  - Arrhenius-Beschleunigung (Gl. 6–7)
  - Über- und Unterspannungsdegradation (Gl. 9)
  - Gesamtdegradationsrate (Gl. 16)
  - Kumuliertes Degradationsintegral (Gl. 17)
  - Kapazitätsdrift und Lebensdauer (Gl. 14, 18)
"""
from __future__ import annotations

import numpy as np

from .constants import (
    R_GAS,
    K_OV_DEFAULT,
    BETA_OV_DEFAULT,
    K_UV_DEFAULT,
    BETA_UV_DEFAULT,
)


# ── Arrhenius ─────────────────────────────────────────────────────────────────

def arrhenius_rate(A: float, Ea: float, T: float) -> float:
    """Arrhenius-Reaktionsrate (Gl. 7).

    k(T) = A · exp(−Ea / (R·T))

    Args:
        A:  Vorexponentieller Faktor (Einheit entsprechend dem Rückgabewert).
        Ea: Aktivierungsenergie [J/mol].
        T:  Absolute Temperatur [K].

    Returns:
        Reaktionsrate bei Temperatur *T*.
    """
    return A * np.exp(-Ea / (R_GAS * T))


def arrhenius_ratio(Ea: float, T0: float, T: float) -> float:
    """Relative Ratenbeschleunigung von *T0* nach *T* (Gl. 6).

    k(T) / k(T0) = exp( Ea/R · (1/T0 − 1/T) )

    Args:
        Ea: Aktivierungsenergie [J/mol].
        T0: Referenztemperatur [K].
        T:  Zieltemperatur [K].

    Returns:
        Ratenverhältnis k(T) / k(T0).
    """
    return np.exp(Ea / R_GAS * (1.0 / T0 - 1.0 / T))


# ── SEI growth ────────────────────────────────────────────────────────────────

def sei_growth_rate(
    k_SEI: float, Ea_SEI: float, T: float, delta_SEI: float
) -> float:
    """Momentane SEI-Schichtdicken-Wachstumsrate (Gl. 3).

    dδ/dt = (k_SEI / 2δ) · exp(−Ea_SEI / (R·T))

    Args:
        k_SEI:     Vorexponentieller SEI-Wachstumsfaktor [m²/s].
        Ea_SEI:    SEI-Aktivierungsenergie [J/mol].
        T:         Temperatur [K].
        delta_SEI: Aktuelle SEI-Schichtdicke [m]. Muss > 0 sein.

    Returns:
        SEI-Wachstumsrate [m/s].
    """
    return (k_SEI / (2.0 * delta_SEI)) * np.exp(-Ea_SEI / (R_GAS * T))


def sei_thickness(k_SEI: float, t: float, Ea_SEI: float, T: float) -> float:
    """Analytisch integrierte SEI-Schichtdicke bei *konstanter* Temperatur (Gl. 4).

    δ(t) = sqrt( 2 · k_SEI · t · exp(−Ea_SEI / (R·T)) )

    Args:
        k_SEI:  Vorexponentieller SEI-Wachstumsfaktor [m²/s].
        t:      Verstrichene Zeit [s].
        Ea_SEI: SEI-Aktivierungsenergie [J/mol].
        T:      (Konstante) Temperatur [K].

    Returns:
        SEI-Schichtdicke [m].
    """
    return np.sqrt(2.0 * k_SEI * t * np.exp(-Ea_SEI / (R_GAS * T)))


def capacity_loss_sei(alpha_SEI: float, delta_SEI: float) -> float:
    """Kapazitätsverlust proportional zur SEI-Schichtdicke (Gl. 5).

    ΔQ_SEI = α_SEI · δ_SEI

    Args:
        alpha_SEI: Proportionalitätskonstante [Ah/m].
        delta_SEI: SEI-Schichtdicke [m].

    Returns:
        Kapazitätsverlust [Ah].
    """
    return alpha_SEI * delta_SEI


# ── Capacity drift ────────────────────────────────────────────────────────────

def capacity_drift(Q: float, Q0: float) -> float:
    """Normierter Kapazitätsdrift (Definition 1).

    D(t) = 1 − Q(t) / Q0

    Das Lebensende (EoL) wird bei D = 0,20 (80 % der Nennkapazität) erreicht.

    Args:
        Q:  Aktuelle Kapazität [Ah].
        Q0: Nennkapazität [Ah].

    Returns:
        Kapazitätsdrift ∈ [0, 1].
    """
    return 1.0 - Q / Q0


# ── Voltage-related degradation ───────────────────────────────────────────────

def overvoltage_degradation_rate(
    V: float,
    V_max: float,
    k_OV: float = K_OV_DEFAULT,
    beta_OV: float = BETA_OV_DEFAULT,
) -> float:
    """Degradationsrate durch Überspannung (Gl. 9).

    r_OV(V) =  k_OV · (exp(β_OV · (V − V_max)) − 1)   falls V > V_max
               0                                         andernfalls

    Args:
        V:      Zellspannung [V].
        V_max:  Obere Spannungsgrenze [V].
        k_OV:   Überspannungs-Degradationskoeffizient.
        beta_OV: Exponentieller Schärfefaktor [1/V].

    Returns:
        Überspannungs-Degradationsrate (≥ 0).
    """
    if V <= V_max:
        return 0.0
    return k_OV * (np.exp(beta_OV * (V - V_max)) - 1.0)


def undervoltage_degradation_rate(
    V: float,
    V_min: float,
    k_UV: float = K_UV_DEFAULT,
    beta_UV: float = BETA_UV_DEFAULT,
) -> float:
    """Degradationsrate durch Unterspannung (symmetrisch zu Gl. 9).

    r_UV(V) =  k_UV · (exp(β_UV · (V_min − V)) − 1)   falls V < V_min
               0                                         andernfalls

    Args:
        V:      Zellspannung [V].
        V_min:  Untere Spannungsgrenze [V].
        k_UV:   Unterspannungs-Degradationskoeffizient.
        beta_UV: Exponentieller Schärfefaktor [1/V].

    Returns:
        Unterspannungs-Degradationsrate (≥ 0).
    """
    if V >= V_min:
        return 0.0
    return k_UV * (np.exp(beta_UV * (V_min - V)) - 1.0)


def voltage_degradation_rate(
    V: float,
    V_min: float,
    V_max: float,
    k_OV: float = K_OV_DEFAULT,
    beta_OV: float = BETA_OV_DEFAULT,
    k_UV: float = K_UV_DEFAULT,
    beta_UV: float = BETA_UV_DEFAULT,
) -> float:
    """Kombinierte Über- und Unterspannungs-Degradationsrate.

    Args:
        V:      Zellspannung [V].
        V_min:  Untere Spannungsgrenze [V].
        V_max:  Obere Spannungsgrenze [V].
        k_OV, beta_OV:  Überspannungsparameter.
        k_UV, beta_UV:  Unterspannungsparameter.

    Returns:
        Gesamte spannungsbedingte Degradationsrate (≥ 0).
    """
    return overvoltage_degradation_rate(
        V, V_max, k_OV, beta_OV
    ) + undervoltage_degradation_rate(V, V_min, k_UV, beta_UV)


# ── Cyclic degradation ────────────────────────────────────────────────────────

def cyclic_degradation_rate(
    I: float,
    T: float,
    k_cycle: float = 1e-5,
    Ea_cycle: float = 30_000.0,
) -> float:
    """Zyklische Degradationsrate (thermisch aktiviert, stromabhängig).

    r_cycle = k_cycle · |I| · exp(−Ea_cycle / (R·T))

    Args:
        I:        Angelegter Strom [A].
        T:        Temperatur [K].
        k_cycle:  Zyklischer Degradationskoeffizient.
        Ea_cycle: Aktivierungsenergie für zyklische Degradation [J/mol].

    Returns:
        Zyklische Degradationsrate.
    """
    return k_cycle * abs(I) * np.exp(-Ea_cycle / (R_GAS * T))


# ── Total degradation rate ────────────────────────────────────────────────────

def total_degradation_rate(
    T: float,
    V: float,
    I: float,
    V_min: float,
    V_max: float,
    k_SEI: float,
    Ea_SEI: float,
    gamma_T: float,
    gamma_V: float,
    gamma_C: float,
    k_OV: float = K_OV_DEFAULT,
    beta_OV: float = BETA_OV_DEFAULT,
    k_UV: float = K_UV_DEFAULT,
    beta_UV: float = BETA_UV_DEFAULT,
    k_cycle: float = 1e-5,
    Ea_cycle: float = 30_000.0,
) -> float:
    """Gesamtdegradationsrate (Gl. 16).

    D̊(t) = γ_T · k_SEI(T)  +  γ_V · r_volt(V)  +  γ_C · r_cycle(I, T)

    Args:
        T, V, I:           Temperatur [K], Spannung [V], Strom [A].
        V_min, V_max:      Spannungsfenster [V].
        k_SEI, Ea_SEI:     SEI-Arrhenius-Parameter.
        gamma_T, gamma_V, gamma_C: Degradationsgewichte.
        k_OV, beta_OV:     Überspannungsmodellparameter.
        k_UV, beta_UV:     Unterspannungsmodellparameter.
        k_cycle, Ea_cycle: Zyklische Modellparameter.

    Returns:
        Gesamtdegradationsrate [1/s].
    """
    r_T = arrhenius_rate(k_SEI, Ea_SEI, T)
    r_V = voltage_degradation_rate(V, V_min, V_max, k_OV, beta_OV, k_UV, beta_UV)
    r_C = cyclic_degradation_rate(I, T, k_cycle, Ea_cycle)
    return gamma_T * r_T + gamma_V * r_V + gamma_C * r_C


# ── Integral / lifetime ───────────────────────────────────────────────────────

def cumulative_degradation(degradation_rates: np.ndarray, dt: float) -> float:
    """Kumuliertes Degradationsintegral Ψ(t) (Gl. 17).

    Ψ(t) = ∫₀ᵗ D̊(τ) dτ  ≈  Σᵢ D̊ᵢ · Δt

    Args:
        degradation_rates: Eindimensionales Array der momentanen Degradationsraten.
        dt:                Zeitschritt [s].

    Returns:
        Kumulierte Degradation (dimensionslos).
    """
    return float(np.sum(degradation_rates) * dt)


def lifetime_from_rate(D_EoL: float, mean_rate: float) -> float:
    """Erwartete Lebensdauer bei konstanter mittlerer Degradationsrate (aus Gl. 18).

    L = D_EoL / mean_rate

    Args:
        D_EoL:     Degradationsschwellwert am Lebensende (EoL).
        mean_rate: Mittlere Degradationsrate [1/s].

    Returns:
        Lebensdauer [s].
    """
    return D_EoL / mean_rate
