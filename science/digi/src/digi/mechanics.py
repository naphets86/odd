"""
mechanics.py – Spring-mass-damper membrane model
================================================
Based on Sections 6 and 7 of "Von der Diskretion".

Covers:
  • Natural (eigen) frequency of undamped oscillator
  • Damping ratio D
  • Damped natural frequency
  • Free damped oscillation (step response of the mechanical system)
  • Classification of damping regime
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Frequency and damping
# ---------------------------------------------------------------------------


def natural_frequency(k: float, m: float) -> tuple[float, float]:
    """Return the undamped natural angular frequency and frequency.

    .. math::
        \\omega_0 = \\sqrt{\\frac{k}{m}}, \\qquad f_0 = \\frac{\\omega_0}{2\\pi}

    Parameters
    ----------
    k:
        Spring constant [N/m].
    m:
        Mass [kg].

    Returns
    -------
    omega_0 : float
        Angular frequency [rad/s].
    f_0 : float
        Frequency [Hz].

    Raises
    ------
    ValueError
        If *k* or *m* are not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    omega_0 = float(np.sqrt(k / m))
    f_0 = omega_0 / (2.0 * np.pi)
    return omega_0, f_0


def damping_ratio(r: float, m: float, k: float) -> float:
    """Compute the dimensionless damping ratio D.

    .. math::
        D = \\frac{r}{2\\sqrt{mk}}

    Parameters
    ----------
    r:
        Damping coefficient [N·s/m].
    m:
        Mass [kg].
    k:
        Spring constant [N/m].

    Returns
    -------
    float
        Damping ratio *D*.

    Raises
    ------
    ValueError
        If *m* or *k* are not positive, or *r* < 0.
    """
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return float(r / (2.0 * np.sqrt(m * k)))


def damped_frequency(omega_0: float, D: float) -> float:
    """Return the damped natural angular frequency.

    .. math::
        \\omega_d = \\omega_0 \\sqrt{1 - D^2}

    Valid only for the underdamped case *D* < 1.

    Parameters
    ----------
    omega_0:
        Undamped natural angular frequency [rad/s].
    D:
        Damping ratio.

    Returns
    -------
    float
        Damped angular frequency ωd [rad/s].  Returns 0 for *D* ≥ 1.
    """
    if D >= 1.0:
        return 0.0
    return float(omega_0 * np.sqrt(1.0 - D**2))


def damping_regime(D: float) -> str:
    """Classify the damping regime.

    Parameters
    ----------
    D:
        Damping ratio.

    Returns
    -------
    str
        One of ``'underdamped'``, ``'critically_damped'``, ``'overdamped'``.
    """
    if D < 1.0:
        return "underdamped"
    elif D == 1.0:
        return "critically_damped"
    else:
        return "overdamped"


# ---------------------------------------------------------------------------
# Free oscillation (impulse / step response)
# ---------------------------------------------------------------------------


def free_oscillation(
    t: ArrayLike,
    omega_0: float,
    D: float,
    x0: float = 1.0,
    v0: float = 0.0,
) -> NDArray[np.float64]:
    """Displacement of a free damped oscillator (underdamped).

    Solves the homogeneous ODE
    :math:`m\\ddot{x} + r\\dot{x} + k\\,x = 0`
    with initial displacement *x0* and velocity *v0*.

    .. math::
        x(t) = e^{-\\alpha t}
               \\left[x_0 \\cos(\\omega_d t)
               + \\frac{v_0 + \\alpha x_0}{\\omega_d} \\sin(\\omega_d t)\\right]

    with :math:`\\alpha = D \\omega_0`.

    Parameters
    ----------
    t:
        Time vector [s] (must be ≥ 0).
    omega_0:
        Undamped natural angular frequency [rad/s].
    D:
        Damping ratio (must be < 1 for oscillation).
    x0:
        Initial displacement.
    v0:
        Initial velocity.

    Returns
    -------
    NDArray
        Displacement x(t).
    """
    t = np.asarray(t, dtype=float)
    alpha = D * omega_0
    omega_d = damped_frequency(omega_0, D)
    if omega_d == 0.0:
        # Critically or overdamped – return pure exponential decay
        return x0 * np.exp(-alpha * t)
    envelope = np.exp(-alpha * t)
    oscillation = x0 * np.cos(omega_d * t) + (v0 + alpha * x0) / omega_d * np.sin(omega_d * t)
    return envelope * oscillation


def step_response_membrane(
    t: ArrayLike,
    omega_0: float,
    D: float,
    U0: float = 1.0,
) -> NDArray[np.float64]:
    """Voltage step response of the mechanical membrane (capacitor voltage analogy).

    Underdamped response:

    .. math::
        u_C(t) = U_0 \\left(1 - e^{-\\alpha t}
                 \\left[\\cos(\\omega_d t) + \\frac{\\alpha}{\\omega_d}\\sin(\\omega_d t)\\right]\\right)

    Parameters
    ----------
    t:
        Time vector [s].
    omega_0:
        Natural angular frequency [rad/s].
    D:
        Damping ratio (< 1 for underdamped).
    U0:
        Step amplitude.

    Returns
    -------
    NDArray
        Step response u_C(t).
    """
    t = np.asarray(t, dtype=float)
    alpha = D * omega_0
    omega_d = damped_frequency(omega_0, D)
    if omega_d == 0.0:
        # Critically damped: u_C(t) = U0 * (1 - e^{-alpha t}*(1 + alpha*t))
        return U0 * (1.0 - np.exp(-alpha * t) * (1.0 + alpha * t))
    return U0 * (
        1.0 - np.exp(-alpha * t) * (
            np.cos(omega_d * t) + (alpha / omega_d) * np.sin(omega_d * t)
        )
    )


# ---------------------------------------------------------------------------
# Mechanical impedance (analogous to electrical)
# ---------------------------------------------------------------------------


def mechanical_impedance(omega: ArrayLike, m: float, r: float, k: float) -> NDArray[np.complex128]:
    """Return the mechanical impedance Z_m(ω) of the spring-mass-damper system.

    .. math::
        Z_m(\\omega) = r + j\\omega m + \\frac{k}{j\\omega}

    (Analogous to the RLC electrical impedance with Force↔Voltage, Velocity↔Current.)

    Parameters
    ----------
    omega:
        Angular frequency vector [rad/s].
    m:
        Mass [kg].
    r:
        Damping [N·s/m].
    k:
        Spring constant [N/m].
    """
    omega = np.asarray(omega, dtype=float)
    return r + 1j * omega * m + k / (1j * omega + 1e-300)
