"""
neuron.py – Biological membrane: neuron models and action potential
===================================================================
Based on Sections 8 and 9 of "Von der Diskretion".

Covers:
  • Simplified Hodgkin-Huxley ODE (4-state model)
  • Leaky integrate-and-fire (LIF) neuron
  • Action potential waveform template
  • Threshold circuit response (all-or-nothing)
  • Refractory period and spike rate encoding

Note
----
The full HH model requires numerical ODE integration.  This module implements
a forward-Euler solver that requires only NumPy (no SciPy dependency).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Physical / biophysical constants
# ---------------------------------------------------------------------------

#: Membrane capacitance [µF/cm²]  (Hodgkin-Huxley canonical value)
C_M_DEFAULT: float = 1.0  # µF/cm²

#: Resting membrane potential [mV]
V_REST: float = -70.0  # mV

#: Threshold potential [mV]
V_THRESH: float = -55.0  # mV

#: Peak action potential [mV]
V_PEAK: float = 40.0  # mV


# ---------------------------------------------------------------------------
# Threshold (all-or-nothing) model
# ---------------------------------------------------------------------------


def threshold_response(
    u_in: ArrayLike,
    u_threshold: float,
    u_high: float = 1.0,
    u_low: float = 0.0,
) -> NDArray[np.float64]:
    """Binary all-or-nothing threshold circuit.

    Implements the biological analogue of a diode threshold circuit:
    the output fires if and only if ``u_in > u_threshold``.

    .. math::
        y[n] = \\begin{cases}
            u_{\\text{high}} & \\text{if } u_{\\text{in}}[n] > u_{\\text{th}} \\\\
            u_{\\text{low}} & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    u_in:
        Input signal samples.
    u_threshold:
        Threshold voltage.
    u_high:
        Output level when threshold is exceeded (action potential peak).
    u_low:
        Output level below threshold (resting state).

    Returns
    -------
    NDArray
        Binary output signal.
    """
    u_in = np.asarray(u_in, dtype=float)
    return np.where(u_in > u_threshold, u_high, u_low)


def exceeds_threshold(u: float, u_threshold: float) -> bool:
    """Return *True* iff the membrane potential exceeds the threshold."""
    return float(u) > float(u_threshold)


# ---------------------------------------------------------------------------
# Action potential waveform template
# ---------------------------------------------------------------------------


def action_potential_waveform(
    t: ArrayLike,
    t_spike: float = 0.0,
    v_rest: float = V_REST,
    v_thresh: float = V_THRESH,
    v_peak: float = V_PEAK,
    tau_rise: float = 0.2e-3,
    tau_fall: float = 0.5e-3,
    tau_ahp: float = 2.0e-3,
    v_ahp: float = -80.0,
) -> NDArray[np.float64]:
    """Generate a schematic action potential waveform.

    The template uses three exponential phases:
    1. Fast depolarisation (rise).
    2. Repolarisation (fall to resting).
    3. After-hyperpolarisation (AHP).

    Parameters
    ----------
    t:
        Time vector [s].
    t_spike:
        Time of spike initiation [s].
    v_rest:
        Resting membrane potential [mV].
    v_thresh:
        Threshold potential [mV].
    v_peak:
        Peak action potential voltage [mV].
    tau_rise:
        Time constant of depolarisation [s].
    tau_fall:
        Time constant of repolarisation [s].
    tau_ahp:
        Time constant of after-hyperpolarisation [s].
    v_ahp:
        Minimum voltage during AHP [mV].

    Returns
    -------
    NDArray
        Membrane voltage [mV] at each time point.
    """
    t = np.asarray(t, dtype=float)
    v = np.full_like(t, v_rest)
    tau = t - t_spike

    # Phase 1: rise (depolarisation) for τ ≥ 0
    mask_rise = tau >= 0
    v[mask_rise] = (
        v_rest
        + (v_peak - v_rest) * (1 - np.exp(-tau[mask_rise] / tau_rise))
    )

    # Phase 2 & 3: repolarisation + AHP only after the peak has been reached
    # Rough threshold: after 3*tau_rise
    mask_fall = tau >= 3.0 * tau_rise
    tau_f = tau[mask_fall] - 3.0 * tau_rise
    v[mask_fall] = (
        v_rest
        + (v_peak - v_rest) * np.exp(-tau_f / tau_fall)
        + (v_ahp - v_rest) * (1 - np.exp(-tau_f / tau_ahp))
        * np.exp(-tau_f / tau_ahp)
    )

    return v


# ---------------------------------------------------------------------------
# Leaky integrate-and-fire (LIF) neuron
# ---------------------------------------------------------------------------


def leaky_integrate_and_fire(
    I_ext: ArrayLike,
    dt: float,
    C_m: float = 100e-12,   # membrane capacitance [F]
    R_m: float = 10e6,       # membrane resistance [Ω]
    V_rest: float = -70e-3,  # resting potential [V]
    V_thresh: float = -55e-3,  # spike threshold [V]
    V_reset: float = -75e-3,   # reset potential after spike [V]
    t_refrac: float = 2e-3,    # refractory period [s]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a leaky integrate-and-fire neuron.

    The membrane equation:
    :math:`\\tau_m \\dot{V} = -(V - V_{\\text{rest}}) + R_m I_{\\text{ext}}`

    with time constant :math:`\\tau_m = R_m C_m`.

    Parameters
    ----------
    I_ext:
        External current [A] for each time step.
    dt:
        Time step [s].
    C_m:
        Membrane capacitance [F].
    R_m:
        Membrane resistance [Ω].
    V_rest:
        Resting potential [V].
    V_thresh:
        Threshold potential [V].
    V_reset:
        Reset potential following a spike [V].
    t_refrac:
        Absolute refractory period [s].

    Returns
    -------
    V : NDArray
        Membrane potential time series [V].
    spikes : NDArray
        Boolean array, *True* at each spike time step.
    """
    I_ext = np.asarray(I_ext, dtype=float)
    n = len(I_ext)
    V = np.full(n, V_rest)
    spikes = np.zeros(n, dtype=bool)
    tau_m = R_m * C_m
    refrac_steps = int(t_refrac / dt)
    refractory_counter = 0

    for i in range(1, n):
        if refractory_counter > 0:
            V[i] = V_reset
            refractory_counter -= 1
        else:
            dV = dt / tau_m * (-(V[i - 1] - V_rest) + R_m * I_ext[i - 1])
            V[i] = V[i - 1] + dV
            if V[i] >= V_thresh:
                V[i] = V_PEAK * 1e-3  # display spike height [V]
                spikes[i] = True
                refractory_counter = refrac_steps

    return V, spikes


# ---------------------------------------------------------------------------
# Simplified Hodgkin-Huxley model
# ---------------------------------------------------------------------------

# HH default parameters (Hodgkin & Huxley 1952, squid giant axon)
# Potentials are in the original HH convention: V = 0 at rest,
# depolarisation is positive.
_HH_DEFAULTS = dict(
    C_m=1.0,       # µF/cm²
    g_Na=120.0,    # mS/cm²
    g_K=36.0,      # mS/cm²
    g_L=0.3,       # mS/cm²
    E_Na=115.0,    # mV  (HH 1952: relative to resting potential)
    E_K=-12.0,     # mV
    E_L=10.613,    # mV
)


def hodgkin_huxley(
    I_ext: ArrayLike,
    dt: float = 0.01e-3,
    **kwargs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate the Hodgkin-Huxley neuron model with forward Euler integration.

    Membrane equation:

    .. math::
        C_m \\frac{du_m}{dt} =
        -g_{\\text{Na}}\\,m^3 h\\,(u_m - E_{\\text{Na}})
        -g_K\\,n^4\\,(u_m - E_K)
        -g_L\\,(u_m - E_L)
        + I_{\\text{ext}}

    Parameters
    ----------
    I_ext:
        External current [µA/cm²] for each time step.
    dt:
        Integration time step [s].  Use ≤ 0.01 ms for numerical stability.
    **kwargs:
        Override any default HH parameters (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L).

    Returns
    -------
    V : NDArray
        Membrane potential [mV].
    t : NDArray
        Time vector [s].
    """
    p = {**_HH_DEFAULTS, **kwargs}
    I_ext = np.asarray(I_ext, dtype=float)
    n_steps = len(I_ext)
    t = np.arange(n_steps) * dt

    V = np.zeros(n_steps)
    m_var = np.zeros(n_steps)
    h_var = np.zeros(n_steps)
    n_var = np.zeros(n_steps)

    # Initial conditions (resting, V=0 convention in HH paper)
    V[0] = 0.0
    m_var[0] = _hh_m_inf(V[0])
    h_var[0] = _hh_h_inf(V[0])
    n_var[0] = _hh_n_inf(V[0])

    for i in range(1, n_steps):
        v = V[i - 1]
        m = m_var[i - 1]
        h = h_var[i - 1]
        n = n_var[i - 1]

        I_Na = p["g_Na"] * m**3 * h * (v - p["E_Na"])
        I_K = p["g_K"] * n**4 * (v - p["E_K"])
        I_L = p["g_L"] * (v - p["E_L"])

        dV = (I_ext[i - 1] - I_Na - I_K - I_L) / p["C_m"]

        dm = _hh_alpha_m(v) * (1 - m) - _hh_beta_m(v) * m
        dh = _hh_alpha_h(v) * (1 - h) - _hh_beta_h(v) * h
        dn = _hh_alpha_n(v) * (1 - n) - _hh_beta_n(v) * n

        V[i] = v + dV * dt * 1e3   # dt in s, dV in mV/ms → scale
        m_var[i] = np.clip(m + dm * dt * 1e3, 0.0, 1.0)
        h_var[i] = np.clip(h + dh * dt * 1e3, 0.0, 1.0)
        n_var[i] = np.clip(n + dn * dt * 1e3, 0.0, 1.0)

    return V, t


# ---------------------------------------------------------------------------
# HH gating variable helpers
# V in mV, HH 1952 original convention: V = 0 at rest, positive = depolarised.
# ---------------------------------------------------------------------------

def _hh_alpha_m(V: float) -> float:
    """Rate constant α_m [ms⁻¹] (HH 1952)."""
    dv = 25.0 - V
    if abs(dv) < 1e-7:
        return 1.0
    return 0.1 * dv / (np.exp(dv / 10.0) - 1.0)


def _hh_beta_m(V: float) -> float:
    """Rate constant β_m [ms⁻¹] (HH 1952)."""
    return 4.0 * np.exp(-V / 18.0)


def _hh_alpha_h(V: float) -> float:
    """Rate constant α_h [ms⁻¹] (HH 1952)."""
    return 0.07 * np.exp(-V / 20.0)


def _hh_beta_h(V: float) -> float:
    """Rate constant β_h [ms⁻¹] (HH 1952)."""
    return 1.0 / (1.0 + np.exp((30.0 - V) / 10.0))


def _hh_alpha_n(V: float) -> float:
    """Rate constant α_n [ms⁻¹] (HH 1952)."""
    dv = 10.0 - V
    if abs(dv) < 1e-7:
        return 0.1
    return 0.01 * dv / (np.exp(dv / 10.0) - 1.0)


def _hh_beta_n(V: float) -> float:
    """Rate constant β_n [ms⁻¹] (HH 1952)."""
    return 0.125 * np.exp(-V / 80.0)


def _hh_m_inf(V: float) -> float:
    return _hh_alpha_m(V) / (_hh_alpha_m(V) + _hh_beta_m(V))


def _hh_h_inf(V: float) -> float:
    return _hh_alpha_h(V) / (_hh_alpha_h(V) + _hh_beta_h(V))


def _hh_n_inf(V: float) -> float:
    return _hh_alpha_n(V) / (_hh_alpha_n(V) + _hh_beta_n(V))
