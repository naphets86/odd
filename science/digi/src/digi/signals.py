"""
signals.py – Continuous and discrete signal fundamentals
=========================================================
Based on Section 2 of "Von der Diskretion".

Covers:
  • Harmonic (sinusoidal) signal generation
  • Signal energy and power
  • Fourier transform (via numpy.fft)
  • Derivative of a harmonic signal
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def harmonic(
    t: ArrayLike,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Return a harmonic (sinusoidal) signal.

    .. math::
        x(t) = A \\sin(2\\pi f t + \\varphi_0)

    Parameters
    ----------
    t:
        Time vector [s].
    amplitude:
        Peak amplitude *A*.
    frequency:
        Frequency *f* [Hz].
    phase:
        Initial phase *φ₀* [rad].

    Returns
    -------
    NDArray
        Signal values with the same shape as *t*.
    """
    t = np.asarray(t, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def harmonic_derivative(
    t: ArrayLike,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Return the analytical derivative of :func:`harmonic`.

    .. math::
        \\dot{x}(t) = A\\,\\omega\\,\\cos(\\omega t + \\varphi_0),
        \\quad \\omega = 2\\pi f

    Parameters follow :func:`harmonic`.
    """
    t = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi * frequency
    return amplitude * omega * np.cos(omega * t + phase)


# ---------------------------------------------------------------------------
# Energy and power
# ---------------------------------------------------------------------------


def signal_energy(x: ArrayLike, dt: float) -> float:
    """Compute the discrete approximation of signal energy.

    .. math::
        E = \\int_{-\\infty}^{+\\infty} |x(t)|^2\\,dt
          \\approx \\sum_n |x[n]|^2 \\cdot \\Delta t

    Parameters
    ----------
    x:
        Signal samples.
    dt:
        Sampling interval [s].

    Returns
    -------
    float
        Energy [unit² · s].
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2) * dt)


def signal_power(x: ArrayLike, dt: float) -> float:
    """Compute the mean signal power.

    .. math::
        P = \\frac{E}{T} = \\frac{1}{N}\\sum_n |x[n]|^2

    Parameters follow :func:`signal_energy`.

    Returns
    -------
    float
        Mean power [unit²].
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean(x**2))


def rms(x: ArrayLike) -> float:
    """Root-mean-square value of a signal.

    Parameters
    ----------
    x:
        Signal samples.

    Returns
    -------
    float
        RMS value.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


# ---------------------------------------------------------------------------
# Fourier transform helpers
# ---------------------------------------------------------------------------


def fourier_transform(
    x: ArrayLike, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Compute the one-sided discrete Fourier transform.

    .. math::
        X(f) = \\mathcal{F}\\{x(t)\\} = \\int_{-\\infty}^{+\\infty} x(t)\\,e^{-j2\\pi ft}\\,dt

    Parameters
    ----------
    x:
        Signal samples.
    dt:
        Sampling interval [s].

    Returns
    -------
    freqs : NDArray
        Frequency vector [Hz] (one-sided, 0 … fs/2).
    X : NDArray[complex]
        Complex spectrum (one-sided, normalised by *N*).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    X = np.fft.rfft(x) / n
    return freqs, X


def spectrum_magnitude(
    x: ArrayLike, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the magnitude spectrum of a signal.

    Parameters follow :func:`fourier_transform`.

    Returns
    -------
    freqs : NDArray
        Frequency vector [Hz].
    magnitude : NDArray
        ``|X(f)|`` (double-sided power folded to one side, first bin excluded).
    """
    freqs, X = fourier_transform(x, dt)
    magnitude = np.abs(X)
    # Compensate for one-sided (double the AC bins, keep DC/Nyquist as is).
    n = 2 * (len(freqs) - 1)
    mag = magnitude.copy()
    mag[1:-1] *= 2.0
    return freqs, mag


# ---------------------------------------------------------------------------
# Angular frequency helpers
# ---------------------------------------------------------------------------


def angular_frequency(f: float) -> float:
    """Convert frequency to angular frequency: ω = 2π f."""
    return 2.0 * np.pi * f


def frequency_from_angular(omega: float) -> float:
    """Convert angular frequency to frequency: f = ω / (2π)."""
    return omega / (2.0 * np.pi)
