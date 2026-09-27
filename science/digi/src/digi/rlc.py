"""
rlc.py – RLC series circuit as electro-mechanical membrane analogue
===================================================================
Based on Sections 7 and 8 of "Von der Diskretion".

Covers:
  • Resonance frequency  ω₀ = 1/√(LC)
  • Quality factor  Q = (1/R)√(L/C)
  • Complex impedance  Z(ω)
  • Step response (u_C, i) – free damped oscillation
  • Frequency response (bandpass current transfer function)
  • Phase angle between voltage and current
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Basic parameters
# ---------------------------------------------------------------------------


def resonance_frequency(L: float, C: float) -> tuple[float, float]:
    """Return the RLC resonance angular frequency and frequency.

    .. math::
        \\omega_0 = \\frac{1}{\\sqrt{LC}}, \\qquad
        f_0 = \\frac{1}{2\\pi\\sqrt{LC}}

    Parameters
    ----------
    L:
        Inductance [H].
    C:
        Capacitance [F].

    Returns
    -------
    omega_0 : float
        Angular resonance frequency [rad/s].
    f_0 : float
        Resonance frequency [Hz].

    Raises
    ------
    ValueError
        If *L* or *C* are not positive.
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    omega_0 = 1.0 / np.sqrt(L * C)
    f_0 = omega_0 / (2.0 * np.pi)
    return float(omega_0), float(f_0)


def quality_factor(R: float, L: float, C: float) -> float:
    """Return the Q-factor (quality factor) of the RLC series circuit.

    .. math::
        Q = \\frac{1}{R}\\sqrt{\\frac{L}{C}}
          = \\frac{\\omega_0 L}{R}
          = \\frac{1}{\\omega_0 C R}

    Parameters
    ----------
    R:
        Resistance [Ω].
    L:
        Inductance [H].
    C:
        Capacitance [F].

    Returns
    -------
    float
        Quality factor *Q* (dimensionless).
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    return float((1.0 / R) * np.sqrt(L / C))


def damping_coefficient(R: float, L: float) -> float:
    """Return the damping coefficient α = R/(2L) [1/s]."""
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    return float(R / (2.0 * L))


def damping_ratio_rlc(R: float, L: float, C: float) -> float:
    """Return the normalised damping ratio D = 1/(2Q) for the RLC circuit.

    Corresponds to the mechanical damping ratio *D* via the analogy.
    """
    Q = quality_factor(R, L, C)
    return float(1.0 / (2.0 * Q))


# ---------------------------------------------------------------------------
# Impedance
# ---------------------------------------------------------------------------


def impedance(
    omega: ArrayLike,
    R: float,
    L: float,
    C: float,
) -> NDArray[np.complex128]:
    """Complex impedance of the RLC series circuit.

    .. math::
        \\underline{Z}(\\omega) = R + j\\omega L + \\frac{1}{j\\omega C}

    Parameters
    ----------
    omega:
        Angular frequency [rad/s].
    R:
        Resistance [Ω].
    L:
        Inductance [H].
    C:
        Capacitance [F].

    Returns
    -------
    NDArray[complex]
        Complex impedance Z(ω).
    """
    omega = np.asarray(omega, dtype=float)
    return R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-300)


def impedance_magnitude(
    omega: ArrayLike,
    R: float,
    L: float,
    C: float,
) -> NDArray[np.float64]:
    """Magnitude |Z(ω)| of the RLC series impedance.

    .. math::
        |\\underline{Z}(\\omega)|
        = \\sqrt{R^2 + \\left(\\omega L - \\tfrac{1}{\\omega C}\\right)^2}
    """
    omega = np.asarray(omega, dtype=float)
    return np.sqrt(R**2 + (omega * L - 1.0 / (omega * C + 1e-300)) ** 2)


def impedance_phase(
    omega: ArrayLike,
    R: float,
    L: float,
    C: float,
) -> NDArray[np.float64]:
    """Phase angle φ(ω) = arg Z(ω) [rad] for the RLC series impedance."""
    return np.angle(impedance(omega, R, L, C))


# ---------------------------------------------------------------------------
# Current frequency response (bandpass)
# ---------------------------------------------------------------------------


def current_transfer(
    omega: ArrayLike,
    R: float,
    L: float,
    C: float,
) -> NDArray[np.complex128]:
    """Current transfer function H_I(ω) = I/U for a voltage-driven RLC circuit.

    At resonance |H_I(ω₀)| = 1/R (maximum current), showing bandpass behaviour.

    .. math::
        H_I(\\omega) = \\frac{1}{Z(\\omega)}
    """
    return 1.0 / impedance(omega, R, L, C)


def bandpass_magnitude(
    omega: ArrayLike,
    R: float,
    L: float,
    C: float,
) -> NDArray[np.float64]:
    """Bandpass magnitude |H_I(ω)| = 1/|Z(ω)|, normalised to peak = 1."""
    H = np.abs(current_transfer(omega, R, L, C))
    return H / np.max(H) if np.max(H) > 0 else H


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------


def step_response(
    t: ArrayLike,
    R: float,
    L: float,
    C: float,
    U0: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RLC series step response: capacitor voltage u_C(t) and current i(t).

    Underdamped case (D < 1):

    .. math::
        u_C(t) = U_0 \\left(1 - e^{-\\alpha t}
                 \\bigl[\\cos(\\omega_d t)
                 + \\tfrac{\\alpha}{\\omega_d}\\sin(\\omega_d t)\\bigr]\\right)

    with :math:`\\alpha = R/(2L)` and
    :math:`\\omega_d = \\sqrt{\\omega_0^2 - \\alpha^2}`.

    Parameters
    ----------
    t:
        Time vector [s] (≥ 0).
    R:
        Resistance [Ω].
    L:
        Inductance [H].
    C:
        Capacitance [F].
    U0:
        Step voltage amplitude [V].

    Returns
    -------
    u_C : NDArray
        Capacitor voltage [V].
    i : NDArray
        Current i(t) = C · du_C/dt, computed via finite differences.
    """
    t = np.asarray(t, dtype=float)
    omega_0, _ = resonance_frequency(L, C)
    alpha = damping_coefficient(R, L)

    if alpha < omega_0:  # underdamped
        omega_d = float(np.sqrt(omega_0**2 - alpha**2))
        u_C = U0 * (
            1.0
            - np.exp(-alpha * t) * (
                np.cos(omega_d * t) + (alpha / omega_d) * np.sin(omega_d * t)
            )
        )
    elif abs(alpha - omega_0) < 1e-12 * omega_0:  # critically damped
        u_C = U0 * (1.0 - np.exp(-alpha * t) * (1.0 + alpha * t))
    else:  # overdamped
        s1 = -alpha + float(np.sqrt(alpha**2 - omega_0**2))
        s2 = -alpha - float(np.sqrt(alpha**2 - omega_0**2))
        A = U0 * s2 / (s2 - s1)
        B = U0 * s1 / (s1 - s2)
        u_C = A * np.exp(s1 * t) + B * np.exp(s2 * t) + U0

    # Current: i = C du_C/dt  (needs at least two points for gradient)
    if len(t) > 1:
        i = C * np.gradient(u_C, t)
    else:
        i = np.zeros_like(u_C)

    return u_C, i


# ---------------------------------------------------------------------------
# Bandwidth
# ---------------------------------------------------------------------------


def bandwidth_3db(R: float, L: float, C: float) -> float:
    """Return the 3 dB bandwidth of the RLC bandpass.

    .. math::
        \\Delta f_{3\\,\\text{dB}} = \\frac{f_0}{Q} = \\frac{R}{2\\pi L}
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    return float(R / (2.0 * np.pi * L))
