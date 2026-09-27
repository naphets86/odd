"""
acoustics.py – Acoustic pressure waves and sound pressure level
===============================================================
Based on Section 5 of "Von der Diskretion".

Covers:
  • Plane acoustic pressure wave
  • Sound intensity
  • Sound pressure level (SPL) in dB
  • Reference quantities (ISO standard)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Standard reference sound pressure [Pa]  (threshold of hearing at 1 kHz)
P_REF: float = 20e-6  # 20 µPa

#: Approximate speed of sound in air at 20 °C [m/s]
C_SOUND_AIR: float = 343.0  # m/s

#: Approximate air density at 20 °C, 1 atm [kg/m³]
RHO_AIR: float = 1.2  # kg/m³


# ---------------------------------------------------------------------------
# Wave functions
# ---------------------------------------------------------------------------


def pressure_wave(
    x: ArrayLike,
    t: ArrayLike,
    p0: float,
    frequency: float,
    c: float = C_SOUND_AIR,
    p_ambient: float = 101_325.0,
) -> NDArray[np.float64]:
    """Return the instantaneous acoustic pressure of a plane travelling wave.

    .. math::
        p(x, t) = p_0 \\sin(\\omega t - kx) + p_{\\text{amb}}

    with wave number :math:`k = \\omega / c`.

    Parameters
    ----------
    x:
        Spatial positions [m] (can be scalar or array).
    t:
        Time [s] (can be scalar or array).
    p0:
        Pressure amplitude [Pa].
    frequency:
        Frequency [Hz].
    c:
        Speed of sound [m/s].
    p_ambient:
        Ambient (static) pressure [Pa].

    Returns
    -------
    NDArray
        Pressure values p(x,t) [Pa].  Shape is broadcast(x, t).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi * frequency
    k = omega / c
    return p0 * np.sin(omega * t - k * x) + p_ambient


def wave_number(frequency: float, c: float = C_SOUND_AIR) -> float:
    """Compute the wave number *k = ω/c = 2πf/c* [1/m].

    Parameters
    ----------
    frequency:
        Frequency [Hz].
    c:
        Speed of sound [m/s].
    """
    return 2.0 * np.pi * frequency / c


def wavelength(frequency: float, c: float = C_SOUND_AIR) -> float:
    """Compute the wavelength *λ = c/f* [m].

    Parameters
    ----------
    frequency:
        Frequency [Hz].
    c:
        Speed of sound [m/s].
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    return c / frequency


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------


def sound_intensity(
    p0: float,
    rho: float = RHO_AIR,
    c: float = C_SOUND_AIR,
) -> float:
    """Return the time-averaged intensity of a plane acoustic wave.

    .. math::
        I = \\frac{p_0^2}{2 \\rho c}

    Parameters
    ----------
    p0:
        Pressure amplitude [Pa].
    rho:
        Air density [kg/m³].
    c:
        Speed of sound [m/s].

    Returns
    -------
    float
        Intensity [W/m²].
    """
    return p0**2 / (2.0 * rho * c)


# ---------------------------------------------------------------------------
# Sound pressure level
# ---------------------------------------------------------------------------


def spl(p: ArrayLike, p_ref: float = P_REF) -> NDArray[np.float64]:
    """Convert RMS sound pressure to Sound Pressure Level (SPL) in dB.

    .. math::
        L_p = 20 \\log_{10}\\!\\left(\\frac{p}{p_0}\\right) \\; [\\text{dB SPL}]

    Parameters
    ----------
    p:
        RMS sound pressure [Pa].  Values ≤ 0 are replaced by *p_ref*.
    p_ref:
        Reference pressure [Pa] (default: 20 µPa, ISO 1683).

    Returns
    -------
    NDArray
        SPL values in dB SPL.
    """
    p = np.asarray(p, dtype=float)
    p = np.where(p > 0, p, p_ref)
    return 20.0 * np.log10(p / p_ref)


def spl_to_pressure(l_p: ArrayLike, p_ref: float = P_REF) -> NDArray[np.float64]:
    """Convert SPL in dB back to RMS sound pressure [Pa].

    .. math::
        p = p_0 \\cdot 10^{L_p / 20}

    Parameters
    ----------
    l_p:
        Sound pressure level [dB SPL].
    p_ref:
        Reference pressure [Pa].
    """
    l_p = np.asarray(l_p, dtype=float)
    return p_ref * 10.0 ** (l_p / 20.0)


def amplitude_to_rms(amplitude: float) -> float:
    """Convert peak amplitude to RMS for a pure sinusoid: p_rms = p0/√2."""
    return amplitude / np.sqrt(2.0)
