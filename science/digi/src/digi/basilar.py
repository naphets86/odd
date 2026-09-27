"""
basilar.py – Basilar membrane model: tonotopy, filterbank, cochlear amplifier
==============================================================================
Based on Sections 10 and 11 of "Von der Diskretion".

Covers:
  • Tonotopic frequency mapping  f₀(x) = f_apex · exp(β x)
  • Inverse tonotopic mapping (frequency → position)
  • Gammatone filter (impulse response and envelope)
  • Bandwidth efficiency η_B = B / L
  • Active Q via cochlear amplifier (Prestin motor)
  • Noise figure improvement through active gain
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Tonotopic mapping
# ---------------------------------------------------------------------------


def tonotopic_frequency(
    x: ArrayLike,
    f_apex: float,
    beta: float,
) -> NDArray[np.float64]:
    """Return the characteristic frequency at basilar membrane position *x*.

    .. math::
        f_0(x) = f_{\\text{apex}} \\cdot e^{\\beta x}, \\quad 0 \\leq x \\leq L

    The cochlear apex corresponds to *x* = 0 (lowest frequency), the base
    to *x* = *L* (highest frequency).

    Parameters
    ----------
    x:
        Position along the membrane [m] or [mm] – units must match *beta*.
    f_apex:
        Characteristic frequency at the apex [Hz].
    beta:
        Tonotopic gradient [1/m] or [1/mm].

    Returns
    -------
    NDArray
        Characteristic frequencies [Hz].
    """
    x = np.asarray(x, dtype=float)
    return f_apex * np.exp(beta * x)


def tonotopic_position(
    f: ArrayLike,
    f_apex: float,
    beta: float,
) -> NDArray[np.float64]:
    """Return the basilar membrane position for a given frequency.

    Inverse of :func:`tonotopic_frequency`:

    .. math::
        x(f) = \\frac{1}{\\beta} \\ln\\!\\left(\\frac{f}{f_{\\text{apex}}}\\right)

    Parameters
    ----------
    f:
        Frequency [Hz].
    f_apex:
        Apex frequency [Hz].
    beta:
        Tonotopic gradient [1/m] or [1/mm].

    Returns
    -------
    NDArray
        Positions [m] (or [mm] if *beta* is in 1/mm).
    """
    f = np.asarray(f, dtype=float)
    if np.any(f <= 0):
        raise ValueError("All frequencies must be positive")
    return (1.0 / beta) * np.log(f / f_apex)


def tonotopic_gradient(f_apex: float, f_base: float, L: float) -> float:
    """Compute the tonotopic gradient β from known apex/base frequencies and length.

    .. math::
        \\beta = \\frac{1}{L} \\ln\\!\\left(\\frac{f_{\\text{base}}}{f_{\\text{apex}}}\\right)

    Parameters
    ----------
    f_apex:
        Characteristic frequency at the apex [Hz].
    f_base:
        Characteristic frequency at the base [Hz].
    L:
        Total membrane length [m] or [mm].
    """
    if f_base <= f_apex:
        raise ValueError("f_base must be greater than f_apex")
    return float(np.log(f_base / f_apex) / L)


# ---------------------------------------------------------------------------
# Bandwidth efficiency
# ---------------------------------------------------------------------------


def bandwidth_efficiency(B: float, L: float) -> float:
    """Bandwidth per unit membrane length η_B = B/L.

    .. math::
        \\eta_B = \\frac{B}{L}

    Parameters
    ----------
    B:
        Audible bandwidth [Hz].
    L:
        Membrane length [m] or [mm].

    Returns
    -------
    float
        η_B [Hz/m] (or [Hz/mm]).
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    return float(B / L)


# ---------------------------------------------------------------------------
# Q factor and cochlear amplifier
# ---------------------------------------------------------------------------


def q10db(f0: float, delta_f_10db: float) -> float:
    """Compute the Q₁₀dB selectivity metric.

    .. math::
        Q_{10\\,\\text{dB}} = \\frac{f_0}{\\Delta f_{10\\,\\text{dB}}}

    Parameters
    ----------
    f0:
        Characteristic frequency [Hz].
    delta_f_10db:
        Bandwidth 10 dB below the peak [Hz].
    """
    if delta_f_10db <= 0:
        raise ValueError("delta_f_10db must be positive")
    return float(f0 / delta_f_10db)


def active_quality_factor(
    Q_passive: float,
    g_motor: float,
    R_passive: float,
) -> float:
    """Return the effective Q factor after active cochlear amplification.

    The Prestin motor protein acts as a negative resistance source,
    reducing the effective damping:

    .. math::
        Q_{\\text{aktiv}} = \\frac{1}{R_{\\text{eff}}} \\sqrt{\\frac{L_m}{C_m}}
        = Q_{\\text{passiv}} \\cdot \\frac{R_{\\text{passiv}}}{R_{\\text{passiv}} - g_{\\text{motor}}}

    Parameters
    ----------
    Q_passive:
        Passive Q factor of the membrane segment.
    g_motor:
        Motor conductance gain [same unit as R_passive].
    R_passive:
        Passive damping resistance [same unit as g_motor].

    Returns
    -------
    float
        Active Q factor.

    Raises
    ------
    ValueError
        If *g_motor* ≥ *R_passive* (system becomes unstable/oscillates).
    """
    R_eff = R_passive - g_motor
    if R_eff <= 0:
        raise ValueError(
            f"g_motor ({g_motor}) must be < R_passive ({R_passive}) for stable operation. "
            "At g_motor = R_passive the system becomes an oscillator (oto-acoustic emission)."
        )
    return float(Q_passive * R_passive / R_eff)


def cochlear_amplifier_gain(R_passive: float, g_motor: float) -> float:
    """Signal gain factor G of the cochlear amplifier.

    .. math::
        G = \\frac{R_{\\text{passiv}}}{R_{\\text{eff}}}
          = \\frac{R_{\\text{passiv}}}{R_{\\text{passiv}} - g_{\\text{motor}}}

    Parameters
    ----------
    R_passive:
        Passive resistance.
    g_motor:
        Motor conductance (must be < R_passive).
    """
    R_eff = R_passive - g_motor
    if R_eff <= 0:
        raise ValueError("g_motor must be < R_passive")
    return float(R_passive / R_eff)


def snr_gain_db(Q_cat: float, Q_dog: float) -> float:
    """SNR gain of cat over dog, from Q factor ratio.

    .. math::
        \\Delta\\text{SNR} = 20\\log_{10}\\!\\left(\\frac{Q_{\\text{cat}}}{Q_{\\text{dog}}}\\right)
    """
    return float(20.0 * np.log10(Q_cat / Q_dog))


# ---------------------------------------------------------------------------
# Gammatone filter
# ---------------------------------------------------------------------------


def gammatone_impulse_response(
    t: ArrayLike,
    f_center: float,
    n: int = 4,
    b: float | None = None,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Gammatone filter impulse response.

    .. math::
        g(t, f, n) = t^{n-1} \\cdot e^{-2\\pi b t} \\cos(2\\pi f t + \\varphi)

    The bandwidth parameter *b* models the equivalent rectangular bandwidth (ERB)
    of the basilar membrane segment.

    Parameters
    ----------
    t:
        Time vector [s] (should start at 0).
    f_center:
        Centre frequency [Hz].
    n:
        Filter order (typically 4 for auditory models).
    b:
        Bandwidth parameter [Hz].  If *None*, uses ERB approximation:
        ``b ≈ 1.019 * ERB(f_center)`` where ``ERB ≈ 24.7*(4.37e-3*f + 1)``.
    phase:
        Carrier phase [rad].

    Returns
    -------
    NDArray
        Impulse response values.
    """
    t = np.asarray(t, dtype=float)
    if b is None:
        # Glasberg & Moore (1990) ERB approximation
        erb = 24.7 * (4.37e-3 * f_center + 1.0)
        b = 1.019 * erb

    t_pos = np.where(t >= 0, t, 0.0)
    return (t_pos ** (n - 1)) * np.exp(-2.0 * np.pi * b * t_pos) * np.cos(
        2.0 * np.pi * f_center * t_pos + phase
    )


def gammatone_envelope(
    t: ArrayLike,
    f_center: float,
    n: int = 4,
    b: float | None = None,
) -> NDArray[np.float64]:
    """Gammatone filter envelope (without carrier oscillation).

    .. math::
        E(t) = t^{n-1} \\cdot e^{-2\\pi b t}
    """
    t = np.asarray(t, dtype=float)
    if b is None:
        erb = 24.7 * (4.37e-3 * f_center + 1.0)
        b = 1.019 * erb
    t_pos = np.where(t >= 0, t, 0.0)
    return (t_pos ** (n - 1)) * np.exp(-2.0 * np.pi * b * t_pos)
