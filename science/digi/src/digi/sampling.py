"""
sampling.py – Nyquist-Shannon sampling theorem
===============================================
Based on Section 3 of "Von der Diskretion".

Covers:
  • Nyquist (minimum) sampling frequency
  • Discrete sampling of a continuous signal
  • Aliasing frequency calculation
  • Sinc-based ideal reconstruction (Whittaker–Shannon interpolation)
  • Anti-aliasing (ideal lowpass) check
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Nyquist criterion
# ---------------------------------------------------------------------------


def nyquist_rate(f_max: float) -> float:
    """Return the minimum (Nyquist) sampling rate for a given signal bandwidth.

    .. math::
        f_N = 2 f_{\\max}

    Parameters
    ----------
    f_max:
        Highest frequency component of the signal [Hz].

    Returns
    -------
    float
        Nyquist rate *f_N* [Hz].

    Raises
    ------
    ValueError
        If *f_max* ≤ 0.
    """
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    return 2.0 * f_max


def nyquist_satisfied(f_max: float, fs: float) -> bool:
    """Return *True* iff the Nyquist criterion is met (*fs* ≥ 2 *f_max*).

    Parameters
    ----------
    f_max:
        Maximum signal frequency [Hz].
    fs:
        Actual sampling frequency [Hz].
    """
    return fs >= nyquist_rate(f_max)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample(
    x_func: object,
    t_start: float,
    t_end: float,
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a callable signal at regular intervals.

    Parameters
    ----------
    x_func:
        Callable ``x(t) -> float`` representing the continuous signal.
    t_start:
        Start of sampling window [s].
    t_end:
        End of sampling window [s].
    fs:
        Sampling frequency [Hz].

    Returns
    -------
    t_samples : NDArray
        Sample time instants [s].
    x_samples : NDArray
        Sampled values *x[n] = x(n · Ts)*.
    """
    Ts = 1.0 / fs
    t_samples = np.arange(t_start, t_end + Ts / 2, Ts)
    x_samples = np.array([x_func(ti) for ti in t_samples], dtype=float)
    return t_samples, x_samples


def sample_array(
    x: ArrayLike,
    t: ArrayLike,
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a given waveform by picking values closest to the sample grid.

    This uses the time vector *t* (assumed to be fine-grained) and an
    integer decimation so that the output spacing approximates 1/*fs*.

    Parameters
    ----------
    x:
        Signal values.
    t:
        Corresponding time axis [s].
    fs:
        Desired sampling frequency [Hz].

    Returns
    -------
    t_s, x_s : NDArrays
        Sample instants and values.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = t[1] - t[0]
    step = max(1, int(round(1.0 / (fs * dt))))
    return t[::step].copy(), x[::step].copy()


# ---------------------------------------------------------------------------
# Aliasing
# ---------------------------------------------------------------------------


def alias_frequency(f_signal: float, fs: float) -> float:
    """Compute the aliased frequency when *f_signal* > *fs*/2.

    .. math::
        f_{\\text{alias}} = \\bigl|f_{\\text{signal}} - k \\cdot f_s\\bigr|

    where *k* is chosen to minimise the result (i.e. fold back to baseband).

    Parameters
    ----------
    f_signal:
        True signal frequency [Hz].
    fs:
        Sampling frequency [Hz].

    Returns
    -------
    float
        Apparent (aliased) frequency [Hz].
    """
    # Reduce to range [0, fs)
    f_mod = f_signal % fs
    # Fold to [0, fs/2]
    if f_mod > fs / 2.0:
        f_mod = fs - f_mod
    return float(f_mod)


# ---------------------------------------------------------------------------
# Sinc (Whittaker-Shannon) interpolation
# ---------------------------------------------------------------------------


def sinc_reconstruct(
    x_samples: ArrayLike,
    t_samples: ArrayLike,
    t_out: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct a bandlimited signal from its samples using sinc interpolation.

    .. math::
        x(t) = \\sum_{n=-\\infty}^{+\\infty}
        x[n] \\cdot \\operatorname{sinc}\\!\\left(\\frac{t - nT_s}{T_s}\\right)

    with ``sinc(u) = sin(π u) / (π u)`` (numpy convention: normalised sinc).

    Parameters
    ----------
    x_samples:
        Discrete sample values.
    t_samples:
        Sample time instants [s].
    t_out:
        Output time vector [s].

    Returns
    -------
    NDArray
        Reconstructed signal at *t_out*.
    """
    x_samples = np.asarray(x_samples, dtype=float)
    t_samples = np.asarray(t_samples, dtype=float)
    t_out = np.asarray(t_out, dtype=float)

    Ts = t_samples[1] - t_samples[0]

    # Outer subtraction: shape (len(t_out), len(t_samples))
    u = (t_out[:, None] - t_samples[None, :]) / Ts
    # np.sinc uses normalised sinc: sinc(u) = sin(π u)/(π u)
    return (x_samples[None, :] * np.sinc(u)).sum(axis=1)


# ---------------------------------------------------------------------------
# Sampling interval / frequency helpers
# ---------------------------------------------------------------------------


def sampling_interval(fs: float) -> float:
    """Return the sampling interval *Ts = 1/fs* [s]."""
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return 1.0 / fs


def max_representable_frequency(fs: float) -> float:
    """Return the maximum representable frequency for a given *fs*.

    .. math::
        f_{\\max} = f_s / 2   \\quad (\\text{Nyquist frequency})
    """
    return fs / 2.0
