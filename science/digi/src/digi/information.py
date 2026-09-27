"""
information.py – Information theory, thermal noise and quantum limits
=====================================================================
Based on Sections 3.2, 4, 12.2 and 12.3 of "Von der Diskretion".

Covers:
  • Shannon information content and entropy
  • Shannon channel capacity (AWGN)
  • Thermal (Johnson-Nyquist) noise voltage
  • Standard Quantum Limit (SQL) for a harmonic oscillator
  • Noise figure: Friis passive limit, quantum SQL limit
  • Bekenstein bound
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Boltzmann constant [J/K]
K_B: float = 1.380_649e-23

#: Planck constant [J·s]
H_PLANCK: float = 6.626_070_15e-34

#: Reduced Planck constant ℏ = h/(2π) [J·s]
H_BAR: float = H_PLANCK / (2.0 * np.pi)

#: ISO reference temperature for noise figure [K]  (290 K ≈ 16.85 °C)
T_0_NOISE: float = 290.0

#: Human body temperature [K]
T_BODY: float = 310.15


# ---------------------------------------------------------------------------
# Shannon information theory
# ---------------------------------------------------------------------------


def information_content(p: ArrayLike) -> NDArray[np.float64]:
    """Shannon information content of an event with probability *p*.

    .. math::
        I = -\\log_2(p) \\; [\\text{bit}]

    Parameters
    ----------
    p:
        Probability (0 < p ≤ 1).

    Returns
    -------
    NDArray
        Information in bits.

    Raises
    ------
    ValueError
        If any probability is ≤ 0 or > 1.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("All probabilities must be in (0, 1]")
    return -np.log2(p)


def shannon_entropy(probabilities: ArrayLike) -> float:
    """Shannon entropy of a discrete probability distribution.

    .. math::
        H = -\\sum_i p_i \\log_2 p_i \\; [\\text{bit}]

    Parameters
    ----------
    probabilities:
        Array of probabilities that must sum to 1.

    Returns
    -------
    float
        Entropy in bits.
    """
    p = np.asarray(probabilities, dtype=float)
    if abs(p.sum() - 1.0) > 1e-9:
        raise ValueError(f"Probabilities must sum to 1, got {p.sum()}")
    # Handle p=0 (0 log 0 = 0 by convention)
    nonzero = p > 0
    return float(-np.sum(p[nonzero] * np.log2(p[nonzero])))


def shannon_capacity(B: float, snr_linear: float) -> float:
    """AWGN Shannon channel capacity.

    .. math::
        C = B \\log_2\\!\\left(1 + \\frac{S}{N}\\right) \\; [\\text{bit/s}]

    Parameters
    ----------
    B:
        Bandwidth [Hz].
    snr_linear:
        Signal-to-noise ratio (linear power ratio, not dB).

    Returns
    -------
    float
        Capacity [bit/s].

    Raises
    ------
    ValueError
        If *B* ≤ 0 or *snr_linear* < 0.
    """
    if B <= 0:
        raise ValueError(f"Bandwidth B must be positive, got {B}")
    if snr_linear < 0:
        raise ValueError(f"SNR must be non-negative, got {snr_linear}")
    return float(B * np.log2(1.0 + snr_linear))


def shannon_capacity_db(B: float, snr_db: float) -> float:
    """Shannon capacity with SNR given in dB.

    Parameters
    ----------
    B:
        Bandwidth [Hz].
    snr_db:
        SNR [dB].
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return shannon_capacity(B, snr_linear)


def capacity_over_bandwidth(snr_db_values: ArrayLike) -> NDArray[np.float64]:
    """Spectral efficiency C/B = log₂(1 + SNR) [bit/s/Hz] for an array of SNR values.

    Parameters
    ----------
    snr_db_values:
        SNR values [dB].
    """
    snr_db_values = np.asarray(snr_db_values, dtype=float)
    snr_linear = 10.0 ** (snr_db_values / 10.0)
    return np.log2(1.0 + snr_linear)


# ---------------------------------------------------------------------------
# Thermal noise
# ---------------------------------------------------------------------------


def thermal_noise_voltage(
    T: float,
    R: float,
    delta_f: float,
) -> float:
    """Johnson-Nyquist RMS noise voltage.

    .. math::
        U_N = \\sqrt{4\\,k_B\\,T\\,R\\,\\Delta f}

    Parameters
    ----------
    T:
        Temperature [K].
    R:
        Resistance [Ω].
    delta_f:
        Noise bandwidth [Hz].

    Returns
    -------
    float
        RMS noise voltage [V].

    Raises
    ------
    ValueError
        If any argument is negative.
    """
    if T < 0:
        raise ValueError(f"Temperature must be non-negative, got {T}")
    if R < 0:
        raise ValueError(f"Resistance must be non-negative, got {R}")
    if delta_f < 0:
        raise ValueError(f"Bandwidth must be non-negative, got {delta_f}")
    return float(np.sqrt(4.0 * K_B * T * R * delta_f))


def thermal_noise_power(T: float, delta_f: float) -> float:
    """Available thermal noise power (kT bandwidth) [W].

    .. math::
        P_N = k_B T \\Delta f

    Parameters
    ----------
    T:
        Temperature [K].
    delta_f:
        Bandwidth [Hz].
    """
    return float(K_B * T * delta_f)


# ---------------------------------------------------------------------------
# Standard Quantum Limit (SQL)
# ---------------------------------------------------------------------------


def sql_displacement_psd(m: float, omega_0: float) -> float:
    """Standard Quantum Limit position noise PSD for a harmonic oscillator.

    .. math::
        S_{xx}^{\\text{SQL}}(\\omega) = \\frac{\\hbar}{m\\,\\omega_0} \\; [\\text{m}^2/\\text{Hz}]

    Parameters
    ----------
    m:
        Effective mass of the oscillator [kg].
    omega_0:
        Natural angular frequency [rad/s].

    Returns
    -------
    float
        SQL position PSD [m²/Hz].
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if omega_0 <= 0:
        raise ValueError(f"omega_0 must be positive, got {omega_0}")
    return float(H_BAR / (m * omega_0))


def sql_displacement_rms(m: float, omega_0: float) -> float:
    """Standard Quantum Limit RMS displacement (zero-point fluctuation).

    .. math::
        x_{\\text{SQL}} = \\sqrt{S_{xx}^{\\text{SQL}}}
                         = \\sqrt{\\frac{\\hbar}{m\\,\\omega_0}} \\; [\\text{m}/\\sqrt{\\text{Hz}}]

    Parameters
    ----------
    m:
        Effective mass [kg].
    omega_0:
        Natural angular frequency [rad/s].
    """
    return float(np.sqrt(sql_displacement_psd(m, omega_0)))


# ---------------------------------------------------------------------------
# Noise figure
# ---------------------------------------------------------------------------


def noise_figure_passive_min_db() -> float:
    """Minimum noise figure of any passive system: 0 dB (Friis limit).

    .. math::
        \\text{NF} \\geq 0\\,\\text{dB} \\quad (\\text{passive system})
    """
    return 0.0


def noise_figure_sql_db(f0: float, T: float = T_BODY) -> float:
    """Quantum noise limit noise figure for an active resonator.

    .. math::
        \\text{NF}_{\\text{SQL}} = \\frac{\\hbar\\omega_0}{2\\,k_B\\,T}
        \\quad [\\text{linear}] \\;\\Rightarrow\\; \\text{NF}_{\\text{SQL, dB}}

    Parameters
    ----------
    f0:
        Resonance frequency [Hz].
    T:
        Temperature [K].

    Returns
    -------
    float
        SQL noise figure [dB].
    """
    omega_0 = 2.0 * np.pi * f0
    nf_linear = H_BAR * omega_0 / (2.0 * K_B * T)
    return float(10.0 * np.log10(nf_linear))


def noise_figure_margin_db(nf_system_db: float, f0: float, T: float = T_BODY) -> float:
    """How many dB above the SQL the system operates.

    A positive result means the system is above the SQL.
    A very large positive result (e.g. passive) means far from the SQL.
    The cat cochlea is ~63 dB above the SQL (but −18 dB vs passive limit).

    Parameters
    ----------
    nf_system_db:
        System noise figure [dB].
    f0:
        Resonance frequency [Hz].
    T:
        Temperature [K].
    """
    return float(nf_system_db - noise_figure_sql_db(f0, T))


# ---------------------------------------------------------------------------
# Bekenstein bound (information content of a physical region)
# ---------------------------------------------------------------------------


def bekenstein_bound_bits(energy_J: float, radius_m: float) -> float:
    """Maximum information content of a physical system (Bekenstein bound).

    .. math::
        I \\leq \\frac{2\\pi\\,R\\,E}{\\hbar\\,c\\,\\ln 2} \\; [\\text{bit}]

    Parameters
    ----------
    energy_J:
        Total energy of the system [J].
    radius_m:
        Characteristic radius of the system [m].

    Returns
    -------
    float
        Maximum information in bits.
    """
    C_LIGHT = 2.997_924_58e8  # m/s
    return float(
        2.0 * np.pi * radius_m * energy_J / (H_BAR * C_LIGHT * np.log(2.0))
    )
