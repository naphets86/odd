"""Plattenabstands-Mechanikmodell (Abschnitt 6 aus liionp.tex).

Modelliert den dynamischen Separatorabstandskanal des KI-PMM:
  - SoC-abhängige Elektrodenausdehnung (Gl. 47)
  - Tortuosität und ionischer Widerstand vs. Spaltbreite (Gl. 49–50)
  - Mechanische Spannung und Degradationsrate (Gl. 42, 52)
  - Mechanischer Degradationsreduktionsfaktor η_M (Gl. 57)
  - MPC-Referenztrajektoriengenerator für d_sep (Gl. 53)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    D_SEP_0,
    D_SEP_MIN,
    D_SEP_MAX,
    EPS_SEP,
    EPS_VOL_COEFFS,
    GAMMA_TAU,
    K_MECH_DEFAULT,
    M_MECH_DEFAULT,
    SIGMA_EL,
    SIGMA_YIELD_DEFAULT,
    TAU_0,
)


# ── Swelling ──────────────────────────────────────────────────────────────────

def swelling_strain(
    SoC: float,
    coeffs: tuple[float, float, float, float] = EPS_VOL_COEFFS,
) -> float:
    """Volumetrische Dehnung der Graphitanode als Funktion des SoC (Gl. 47).

    ε_vol(SoC) = a₀ + a₁·SoC + a₂·SoC² + a₃·SoC³

    Args:
        SoC:    Ladezustand ∈ [0, 1].
        coeffs: Polynomkoeffizienten (a₀, a₁, a₂, a₃).

    Returns:
        Volumetrische Dehnung (dimensionslos).
    """
    a0, a1, a2, a3 = coeffs
    return a0 + a1 * SoC + a2 * SoC**2 + a3 * SoC**3


# ── Tortuosity & ionic resistance ─────────────────────────────────────────────

def tortuosity(
    d_sep: float,
    d_sep_0: float = D_SEP_0,
    tau_0: float = TAU_0,
    gamma_tau: float = GAMMA_TAU,
) -> float:
    """Tortuosität als Funktion der Separatorkompression (Gl. 50).

    τ(d) = τ₀ · (d₀ / d)^γ_τ

    Die Tortuosität steigt, wenn der Separator komprimiert wird (d < d₀).

    Args:
        d_sep:     Aktuelle Separatordicke [m].
        d_sep_0:   Nominale (unkomprimierte) Dicke [m].
        tau_0:     Referenztortuosität bei d = d₀.
        gamma_tau: Potenzgesetz-Exponent.

    Returns:
        Tortuosität (dimensionslos, ≥ 1 für d ≤ d₀).
    """
    return tau_0 * (d_sep_0 / d_sep) ** gamma_tau


def ionic_resistance(
    d_sep: float,
    A_cell: float,
    d_sep_0: float = D_SEP_0,
    tau_0: float = TAU_0,
    gamma_tau: float = GAMMA_TAU,
    eps_sep: float = EPS_SEP,
    sigma_el: float = SIGMA_EL,
) -> float:
    """Ionischer Widerstand durch den Separator (Gl. 49).

    R_ion = τ(d) · d / (σ_el · ε_sep · A_cell)

    Args:
        d_sep:     Separatordicke [m].
        A_cell:    Aktive Zellfläche [m²].
        d_sep_0:   Nominale Dicke [m].
        tau_0:     Referenztortuosität.
        gamma_tau: Tortuositätsexponent.
        eps_sep:   Separatorporosität.
        sigma_el:  Elektrolytleitfähigkeit [S/m].

    Returns:
        Ionischer Widerstand [Ω].
    """
    tau = tortuosity(d_sep, d_sep_0, tau_0, gamma_tau)
    return tau * d_sep / (sigma_el * eps_sep * A_cell)


# ── Mechanical stress & degradation ──────────────────────────────────────────

def mechanical_stress(
    d_sep_dot: float,
    eps_vol_dot: float,
    k_proportional: float = 1.0,
) -> float:
    """Mechanische Spannung aus der Fehlanpassung von Plattenabstand- und Quellraten (Gl. 42).

    σ_mech ∝ |ṓ_sep − ε̇_vol|

    Args:
        d_sep_dot:      Änderungsrate des Separatorabstands [m/s].
        eps_vol_dot:    Änderungsrate der volumetrischen Dehnung der Elektroden [1/s].
        k_proportional: Proportionalitätskonstante.

    Returns:
        Mechanische Spannung (willkürliche Einheiten, passend zu sigma_yield).
    """
    return k_proportional * abs(d_sep_dot - eps_vol_dot)


def mechanical_degradation_rate(
    sigma_mech: float,
    sigma_yield: float = SIGMA_YIELD_DEFAULT,
    k_mech: float = K_MECH_DEFAULT,
    m_mech: float = M_MECH_DEFAULT,
) -> float:
    """Partikelriss-Degradationsrate durch mechanische Spannung (Gl. 52).

    r_mech = k_mech · max(0, |σ_mech| − σ_yield)^m_mech

    Args:
        sigma_mech:  Mechanische Spannung (w.E.).
        sigma_yield: Spannungsschwelle, unterhalb derer keine Schädigung erfolgt.
        k_mech:      Mechanischer Degradationskoeffizient.
        m_mech:      Exponent (≈ 2,5 für NMC/Graphit).

    Returns:
        Mechanische Degradationsrate (≥ 0).
    """
    excess = max(0.0, abs(sigma_mech) - sigma_yield)
    return k_mech * excess**m_mech


def mechanical_reduction_factor(
    sigma_amp_ratio: float,
    m_mech: float = M_MECH_DEFAULT,
) -> float:
    """Mechanischer Degradationsreduktionsfaktor η_M (Gl. 57).

    η_M = (1 / σ_amp_ratio)^m_mech

    Für NMC/Graphit: σ_amp / σ_yield ≈ 1,8  →  η_M ≈ 0,23

    Args:
        sigma_amp_ratio: Verhältnis σ̂_amp / σ_yield (unkontrollierte Amplitude).
        m_mech:          Potenzgesetz-Exponent.

    Returns:
        Reduktionsfaktor η_M ∈ (0, 1).
    """
    return (1.0 / sigma_amp_ratio) ** m_mech


# ── Plate-distance MPC reference generator ───────────────────────────────────

@dataclass
class PlateDistanceMPCController:
    """Referenztrajektoriengenerator für den Plattenabstands-MPC (Gl. 53).

    Berechnet d_ref(SoC) so, dass der Separatorspalt die Elektrodenvolumenänderungen
    kompensiert und die mechanische Spannung σ_mech minimiert.

    Attributes:
        d_min:      Minimaler zulässiger Separatorabstand [m].
        d_max:      Maximaler zulässiger Separatorabstand [m].
        lambda_d:   Gewichtung der Abstandsführung.
        lambda_sigma: Gewichtung der mechanischen Spannung.
        lambda_F:   Gewichtung der Aktuatorkraft (Energiekosten).
    """

    d_min: float = D_SEP_MIN
    d_max: float = D_SEP_MAX
    lambda_d: float = 1.0
    lambda_sigma: float = 2.0
    lambda_F: float = 0.1

    def reference(self, SoC: float) -> float:
        """Berechnet den Referenz-Separatorabstand für den gegebenen SoC.

        Die Referenz folgt der Nenndicke, kompensiert um die Elektrodenquellungsdehnung:

            d_ref(SoC) = d₀ · (1 + ε_vol(SoC))

        begrenzt auf [d_min, d_max].

        Args:
            SoC: Ladezustand ∈ [0, 1].

        Returns:
            Referenz-Plattenabstand [m].
        """
        strain = swelling_strain(SoC)
        d_ref = D_SEP_0 * (1.0 + strain)
        return float(np.clip(d_ref, self.d_min, self.d_max))
