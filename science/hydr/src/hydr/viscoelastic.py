"""
viscoelastic.py
===============
Zener-Modell (Standardkörper) für hydraulische Flüssigkeiten.

Frequenzantwort komplexer Kompressionsmodul:
  K*(ω) = K'(ω) + i·K''(ω)

Gleichungen aus Epp (2026):
  Gl. (8)   Zener-Differentialgleichung
  Gl. (20)  Komplexer Modul
  Gl. (21)  Speichermodul  K'(ω)
  Gl. (22)  Verlustmodul   K''(ω)
  tan δ     = K'' / K'

SI-Einheiten:
  Modul  [Pa]
  ω      [rad/s]
  τ_R    [s]
"""

from __future__ import annotations

import math


def storage_modulus(
    E0: float,
    E_inf: float,
    omega: float,
    tau_R: float,
) -> float:
    """Speichermodul (elastischer Anteil) K'(ω).

       K'(ω) = E0 + (E_inf − E0) · (ω τ_R)² / [1 + (ω τ_R)²]   (Gl. 21)

    Parameters
    ----------
    E0:    Statischer Kompressionsmodul (Gleichgewicht) [Pa]
    E_inf: Instantaner (Hochfrequenz-)Modul [Pa]
    omega: Kreisfrequenz [rad/s]
    tau_R: Relaxationszeit [s]

    Returns
    -------
    K' [Pa]
    """
    wt = omega * tau_R
    return E0 + (E_inf - E0) * wt**2 / (1.0 + wt**2)


def loss_modulus(
    E0: float,
    E_inf: float,
    omega: float,
    tau_R: float,
) -> float:
    """Verlustmodul (viskoser Anteil) K''(ω).

       K''(ω) = (E_inf − E0) · ω τ_R / [1 + (ω τ_R)²]   (Gl. 22)

    Parameters
    ----------
    E0:    Statischer Kompressionsmodul [Pa]
    E_inf: Instantaner Hochfrequenz-Modul [Pa]
    omega: Kreisfrequenz [rad/s]
    tau_R: Relaxationszeit [s]

    Returns
    -------
    K'' [Pa]
    """
    wt = omega * tau_R
    return (E_inf - E0) * wt / (1.0 + wt**2)


def loss_factor(
    E0: float,
    E_inf: float,
    omega: float,
    tau_R: float,
) -> float:
    """Verlustfaktor tan δ = K'' / K'.

    Parameters
    ----------
    E0:    Statischer Kompressionsmodul [Pa]
    E_inf: Instantaner Hochfrequenz-Modul [Pa]
    omega: Kreisfrequenz [rad/s]
    tau_R: Relaxationszeit [s]

    Returns
    -------
    tan δ  [dimensionslos]

    Raises
    ------
    ZeroDivisionError falls K'(ω) = 0.
    """
    kp = storage_modulus(E0, E_inf, omega, tau_R)
    kpp = loss_modulus(E0, E_inf, omega, tau_R)
    if kp == 0.0:
        raise ZeroDivisionError("Speichermodul K' ist null – tan δ nicht definiert.")
    return kpp / kp


def complex_modulus_magnitude(
    E0: float,
    E_inf: float,
    omega: float,
    tau_R: float,
) -> float:
    """|K*(ω)| = sqrt(K'² + K''²).

    Parameters
    ----------
    E0:    Statischer Kompressionsmodul [Pa]
    E_inf: Instantaner Hochfrequenz-Modul [Pa]
    omega: Kreisfrequenz [rad/s]
    tau_R: Relaxationszeit [s]

    Returns
    -------
    |K*| [Pa]
    """
    kp = storage_modulus(E0, E_inf, omega, tau_R)
    kpp = loss_modulus(E0, E_inf, omega, tau_R)
    return math.hypot(kp, kpp)
