"""
quantization.py – Amplitude quantisation (A/D conversion)
==========================================================
Based on Sections 3.3 and 9 of "Von der Diskretion".

Covers:
  • N-bit mid-tread quantiser
  • Quantisation step size Δu
  • Quantisation error / noise
  • Maximum SNR formula:  SNR_max ≈ 6.02 N + 1.76 dB
  • Number of quantisation levels
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Step size and levels
# ---------------------------------------------------------------------------


def quantization_step(n_bits: int, x_min: float = -1.0, x_max: float = 1.0) -> float:
    """Return the quantisation step size Δu for an N-bit converter.

    .. math::
        \\Delta u = \\frac{u_{\\max} - u_{\\min}}{2^N}

    Parameters
    ----------
    n_bits:
        Number of bits *N* ≥ 1.
    x_min:
        Minimum of the input range.
    x_max:
        Maximum of the input range.

    Returns
    -------
    float
        Step size Δu.

    Raises
    ------
    ValueError
        If *n_bits* < 1 or *x_max* ≤ *x_min*.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be ≥ 1, got {n_bits}")
    if x_max <= x_min:
        raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")
    return (x_max - x_min) / (2**n_bits)


def num_levels(n_bits: int) -> int:
    """Return the number of quantisation levels *2^N*.

    Parameters
    ----------
    n_bits:
        Number of bits *N* ≥ 1.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be ≥ 1, got {n_bits}")
    return 2**n_bits


# ---------------------------------------------------------------------------
# Quantiser
# ---------------------------------------------------------------------------


def quantize(
    x: ArrayLike,
    n_bits: int,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Apply uniform mid-tread quantisation to signal *x*.

    Values outside [*x_min*, *x_max*] are clipped before quantisation.

    Parameters
    ----------
    x:
        Input signal (continuous amplitude).
    n_bits:
        Resolution in bits.
    x_min:
        Lower bound of the quantiser input range.
    x_max:
        Upper bound of the quantiser input range.

    Returns
    -------
    NDArray
        Quantised signal (amplitudes are quantiser output levels, not integer codes).
    """
    x = np.asarray(x, dtype=float)
    delta = quantization_step(n_bits, x_min, x_max)
    x_clipped = np.clip(x, x_min, x_max - delta)
    # Integer code
    codes = np.floor((x_clipped - x_min) / delta).astype(int)
    # Back to amplitude: centre of each step
    return x_min + (codes + 0.5) * delta


def quantization_error(x: ArrayLike, x_quantized: ArrayLike) -> NDArray[np.float64]:
    """Return the quantisation error *e[n] = x_q[n] - x[n]*.

    Parameters
    ----------
    x:
        Original (ideal continuous) signal.
    x_quantized:
        Quantised signal.
    """
    return np.asarray(x_quantized, dtype=float) - np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------


def snr_max_db(n_bits: int) -> float:
    """Maximum achievable Signal-to-Noise Ratio for an N-bit converter.

    .. math::
        \\text{SNR}_{\\max} \\approx 6.02\\,N + 1.76 \\; [\\text{dB}]

    This formula assumes a full-scale sinusoidal input and uniformly distributed
    quantisation noise.

    Parameters
    ----------
    n_bits:
        Resolution in bits.

    Returns
    -------
    float
        SNR_max in dB.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be ≥ 1, got {n_bits}")
    return 6.02 * n_bits + 1.76


def snr_db(signal: ArrayLike, noise: ArrayLike) -> float:
    """Compute the actual SNR in dB from signal and noise arrays.

    .. math::
        \\text{SNR} = 10\\log_{10}\\!\\left(\\frac{\\sum x^2}{\\sum e^2}\\right)

    Parameters
    ----------
    signal:
        Original signal samples.
    noise:
        Noise (or error) samples.
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    power_signal = np.sum(signal**2)
    power_noise = np.sum(noise**2)
    if power_noise == 0.0:
        return float("inf")
    return float(10.0 * np.log10(power_signal / power_noise))


def snr_linear_to_db(snr_linear: float) -> float:
    """Convert a linear SNR (power ratio) to dB."""
    return 10.0 * np.log10(snr_linear)


def snr_db_to_linear(snr_db_val: float) -> float:
    """Convert an SNR in dB to a linear (power) ratio."""
    return 10.0 ** (snr_db_val / 10.0)
