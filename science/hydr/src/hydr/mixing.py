"""
mixing.py
=========
Mischungsregeln für Kompressionsmoduln gemischter Hydraulikflüssigkeiten.

Gleichungen aus Epp (2026):
  Gl. (11)  Mischungsregel Öl-Gas   1/K_eff = (1-α)/K_öl + α/K_gas
  Gl. (19)  Maxwell-Mischungsregel  HFAE-Emulsionen
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Allgemeine Reuss-Mischungsregel
# ---------------------------------------------------------------------------

def reuss_modulus(K1: float, K2: float, phi2: float) -> float:
    """Effektiver Modul nach Reuss (Reihenschaltung / untere Schranke).

       1/K_eff = (1 − φ₂)/K₁ + φ₂/K₂

    Parameters
    ----------
    K1:   Modul der Phase 1 (Matrix) [Pa]
    K2:   Modul der Phase 2 (Einschluss) [Pa]
    phi2: Volumenanteil der Phase 2 (0 ≤ φ₂ ≤ 1)

    Returns
    -------
    K_eff [Pa]
    """
    if not (0.0 <= phi2 <= 1.0):
        raise ValueError("phi2 muss im Bereich [0, 1] liegen.")
    if K1 <= 0 or K2 <= 0:
        raise ValueError("K1 und K2 müssen positiv sein.")
    return 1.0 / ((1.0 - phi2) / K1 + phi2 / K2)


# ---------------------------------------------------------------------------
# Maxwell-Mischungsregel für Emulsionen (HFAE, HFAS)
# ---------------------------------------------------------------------------

def maxwell_emulsion_modulus(
    K_W: float,
    K_O: float,
    phi: float,
) -> float:
    """Effektiver Kompressionsmodul einer Öl-in-Wasser-Emulsion (Maxwell).

       K_HFAE = K_W · [2K_W + K_O − 2φ(K_W − K_O)]
                     / [2K_W + K_O + φ(K_W − K_O)]   (Gl. 19)

    Parameters
    ----------
    K_W: Kompressionsmodul der Wasserphase [Pa]
    K_O: Kompressionsmodul der Ölphase [Pa]
    phi: Ölvolumenanteil (0 ≤ φ ≤ 1)

    Returns
    -------
    K_HFAE [Pa]
    """
    if not (0.0 <= phi <= 1.0):
        raise ValueError("phi muss im Bereich [0, 1] liegen.")
    if K_W <= 0 or K_O <= 0:
        raise ValueError("K_W und K_O müssen positiv sein.")
    numerator = 2.0 * K_W + K_O - 2.0 * phi * (K_W - K_O)
    denominator = 2.0 * K_W + K_O + phi * (K_W - K_O)
    return K_W * numerator / denominator


# ---------------------------------------------------------------------------
# Voigt-Mischungsregel (obere Schranke, vollständige Kopplung)
# ---------------------------------------------------------------------------

def voigt_modulus(K1: float, K2: float, phi2: float) -> float:
    """Effektiver Modul nach Voigt (Parallelschaltung / obere Schranke).

       K_eff = (1 − φ₂) · K₁ + φ₂ · K₂

    Parameters
    ----------
    K1:   Modul der Phase 1 [Pa]
    K2:   Modul der Phase 2 [Pa]
    phi2: Volumenanteil der Phase 2 (0 ≤ φ₂ ≤ 1)

    Returns
    -------
    K_eff [Pa]
    """
    if not (0.0 <= phi2 <= 1.0):
        raise ValueError("phi2 muss im Bereich [0, 1] liegen.")
    return (1.0 - phi2) * K1 + phi2 * K2
