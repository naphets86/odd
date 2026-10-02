"""
Ship Freight Module
Based on: Band der Arbeiten zur Wirtschaft - Kapitel "Schiffsfracht im
wirtschaftlichen Gleichgewicht: Die gerade lohnende Ladung"
Author: Stephan Epp
Adapted for Python: 2026

Implements the results of the ship-freight chapter:
- Demand, land-transport gap and the period gap G (Lemma 1)
- Balance theorem N*L = G and overflow/shortage times (Satz 1)
- Buffer requirement of a single cargo (Satz 2)
- Profitability: unit margin, EOQ, viable loads, safety margin (Sätze 3-4)
- Market equilibrium and the "just worthwhile" load (Hauptsatz)
- Price disturbance from periodic overflow (Satz 6)
- Distribution over economic cells (Satz 7)
- Sea freight and bacterial development of the cargo (chapter "Seefracht und
  bakterielle Entwicklung", labels sb-*): growth work and shelf-life budget,
  temperature/atmosphere as local information, vehicle comparison, pulse age
  in the port store, freshness limit L_F and the double clamp of the voyage
  duration, correlated failure risk (see the last section of this module)

Sign convention: the price dynamics are the restoring form
dP/dt = -alpha (P - P0) + eta sin(2 pi t / T)  (see Bemerkung 1 of the chapter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _nonnegative(name: str, value: float) -> None:
    if not value >= 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _check_demand(d0: float, d_hat: float, T: float) -> None:
    """Definition 1: d0 > d_hat > 0 and T > 0."""
    _positive("d_hat", d_hat)
    _positive("T", T)
    if not d0 > d_hat:
        raise ValueError(f"d0 must be > d_hat, got d0={d0}, d_hat={d_hat}")


# ============================================================================
# DEMAND AND LAND-TRANSPORT GAP (Definitions 1-3, Lemma 1)
# ============================================================================

def demand_rate(t: float, d0: float, d_hat: float, T: float) -> float:
    """Justified transport demand d(t) = d0 + d_hat sin(2 pi t / T)."""
    _check_demand(d0, d_hat, T)
    return d0 + d_hat * math.sin(2.0 * math.pi * t / T)


def gap_rate(t: float, d0: float, d_hat: float, K_L: float, T: float) -> float:
    """Demand not served by land transport: u(t) = (d(t) - K_L)^+."""
    _nonnegative("K_L", K_L)
    return max(demand_rate(t, d0, d_hat, T) - K_L, 0.0)


def period_gap(d0: float, d_hat: float, K_L: float, T: float) -> float:
    """
    Period gap G = integral of u over one period (closed form, Lemma 1).

    With m = d0 - K_L and c = m / d_hat:
        c <= -1: G = 0;  |c| < 1: arcsine formula;  c >= 1: G = m T.
    """
    _check_demand(d0, d_hat, T)
    _nonnegative("K_L", K_L)
    m = d0 - K_L
    c = m / d_hat
    if c <= -1.0:
        return 0.0
    if c >= 1.0:
        return m * T
    return d_hat * T / math.pi * (
        c * (math.pi / 2.0 + math.asin(c)) + math.sqrt(1.0 - c * c)
    )


def _positive_intervals(m: float, d_hat: float) -> list:
    """Phase intervals in [0, 2 pi) where m + d_hat sin(theta) > 0."""
    c = m / d_hat
    if c >= 1.0:
        return [(0.0, 2.0 * math.pi)]
    if c <= -1.0:
        return []
    alpha = math.asin(c)
    if alpha >= 0.0:
        return [(0.0, math.pi + alpha), (2.0 * math.pi - alpha, 2.0 * math.pi)]
    return [(-alpha, math.pi + alpha)]


def cumulative_gap(t: float, d0: float, d_hat: float, K_L: float, T: float) -> float:
    """Exact U(t) = integral_0^t u(s) ds for t >= 0."""
    _nonnegative("t", t)
    G = period_gap(d0, d_hat, K_L, T)
    m = d0 - K_L
    n = math.floor(t / T)
    theta = 2.0 * math.pi * (t - n * T) / T
    partial = 0.0
    for a, b in _positive_intervals(m, d_hat):
        hi = min(b, theta)
        if hi > a:
            partial += m * (hi - a) + d_hat * (math.cos(a) - math.cos(hi))
    return n * G + partial * T / (2.0 * math.pi)


# ============================================================================
# BALANCE THEOREM (Satz 1)
# ============================================================================

def load_for_departures(G: float, N: int) -> float:
    """Korollar 1: L = G / N."""
    _positive("G", G)
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    return G / N


def departures_for_load(G: float, L: float) -> float:
    """Korollar 1: N = G / L (real-valued mean departure count)."""
    _positive("G", G)
    _positive("L", L)
    return G / L


def period_imbalance(N: float, L: float, G: float) -> float:
    """Q - G with Q = N L; zero iff the operation can be admissible."""
    return N * L - G


def _check_stock(B: float, I0: float) -> None:
    _positive("B", B)
    if not 0.0 <= I0 <= B:
        raise ValueError(f"I0 must be in [0, B], got I0={I0}, B={B}")


def overflow_time_estimate(B: float, I0: float, Q: float, G: float, T: float) -> float:
    """t* = T (B - I0) / (Q - G) for Q > G (Satz 1.2)."""
    _check_stock(B, I0)
    _positive("T", T)
    if not Q > G:
        raise ValueError("overflow requires Q > G")
    return T * (B - I0) / (Q - G)


def overflow_time_tolerance(Q: float, G: float, T: float) -> float:
    """Bound T Q / (Q - G) on |t_B - t*| (Satz 1.2)."""
    _positive("T", T)
    if not Q > G:
        raise ValueError("overflow requires Q > G")
    return T * Q / (Q - G)


def shortage_time_estimate(B: float, I0: float, Q: float, G: float, T: float) -> float:
    """t*_F = T I0 / (G - Q) for Q < G (Satz 1.3)."""
    _check_stock(B, I0)
    _positive("T", T)
    if not Q < G:
        raise ValueError("shortage requires Q < G")
    return T * I0 / (G - Q)


def shortage_time_tolerance(Q: float, G: float, T: float) -> float:
    """Bound T G / (G - Q) on |t_F - t*_F| (Satz 1.3)."""
    _positive("T", T)
    if not Q < G:
        raise ValueError("shortage requires Q < G")
    return T * G / (G - Q)


@dataclass
class StockTrajectory:
    """Free stock X(t) evaluated immediately before/after every call."""
    times: np.ndarray
    before: np.ndarray
    after: np.ndarray
    G: float
    L: float


def stock_at_calls(N: int, L: float, d0: float, d_hat: float, K_L: float, T: float,
                   I0: float, t0: float = 0.0, n_periods: int = 1) -> StockTrajectory:
    """
    Exact stock X(t) = I0 + A(t) - U(t) at the call times t_k = t0 + k T / N
    (Definitions 4-5). X is non-increasing between calls and jumps by +L at
    each call, so suprema/infima are attained at these points.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    _positive("L", L)
    _positive("T", T)
    if not 0.0 <= t0 < T / N:
        raise ValueError("t0 must be in [0, T/N)")
    G = period_gap(d0, d_hat, K_L, T)
    k = np.arange(N * n_periods)
    times = t0 + k * T / N
    U = np.array([cumulative_gap(float(t), d0, d_hat, K_L, T) for t in times])
    before = I0 + k * L - U
    return StockTrajectory(times=times, before=before, after=before + L, G=G, L=L)


def first_overflow_time(traj: StockTrajectory, B: float) -> Optional[float]:
    """t_B = inf{t : X(t) > B}; attained at a call time. None if not reached."""
    hits = np.nonzero(traj.after > B)[0]
    if hits.size == 0:
        return None
    return float(traj.times[hits[0]])


def first_shortage_time(traj: StockTrajectory, I0: float, d0: float, d_hat: float,
                        K_L: float, T: float) -> Optional[float]:
    """t_F = inf{t : X(t) < 0}, located by root finding between calls."""
    hits = np.nonzero(traj.before < 0.0)[0]
    if hits.size == 0:
        return None
    k = int(hits[0])
    if k == 0:
        t_prev, x_prev = 0.0, I0
    else:
        t_prev, x_prev = float(traj.times[k - 1]), float(traj.after[k - 1])
    t_next = float(traj.times[k])
    U_prev = cumulative_gap(t_prev, d0, d_hat, K_L, T)

    def x_of(t: float) -> float:
        return x_prev - (cumulative_gap(t, d0, d_hat, K_L, T) - U_prev)

    return float(brentq(x_of, t_prev, t_next))


# ============================================================================
# BUFFER REQUIREMENT OF A SINGLE CARGO (Satz 2)
# ============================================================================

def season_buffer(d_hat: float, T: float) -> float:
    """Seasonal buffer d_hat T / pi (independent of ship size)."""
    _positive("d_hat", d_hat)
    _positive("T", T)
    return d_hat * T / math.pi


def _relative_stock_levels(N: int, d0: float, d_hat: float, K_L: float, T: float,
                           t0: float) -> Tuple[float, np.ndarray]:
    """Return L = G/N and y_k = X(t_k^-) - I0 for k = 0..N-1 (Q = G)."""
    G = period_gap(d0, d_hat, K_L, T)
    L = load_for_departures(G, N)
    if not 0.0 <= t0 < T / N:
        raise ValueError("t0 must be in [0, T/N)")
    edges = np.array([cumulative_gap(t0 + k * T / N, d0, d_hat, K_L, T)
                      for k in range(N + 1)])
    consumed = np.diff(edges)
    y = np.concatenate(([-edges[0]], -edges[0] + np.cumsum(L - consumed)))[:N]
    return L, y


def buffer_requirement(N: int, d0: float, d_hat: float, K_L: float, T: float,
                       t0: float = 0.0) -> float:
    """B_min(L) = L + max_k y_k - min_k y_k (Satz 2.1)."""
    L, y = _relative_stock_levels(N, d0, d_hat, K_L, T, t0)
    return float(L + y.max() - y.min())


def admissible_initial_stock(N: int, B: float, d0: float, d_hat: float, K_L: float,
                             T: float, t0: float = 0.0) -> Optional[Tuple[float, float]]:
    """Interval [-min y, B - L - max y] of admissible I0, or None (Satz 2.1)."""
    L, y = _relative_stock_levels(N, d0, d_hat, K_L, T, t0)
    lo = float(-y.min())
    hi = float(B - L - y.max())
    if lo > hi:
        return None
    return lo, hi


def buffer_upper_bound(L: float, d_hat: float, T: float) -> float:
    """L + d_hat T / pi: sufficient buffer in the year-round-gap case (Satz 2.3)."""
    _positive("L", L)
    return L + season_buffer(d_hat, T)


def buffer_limit_sufficient(B: float, d_hat: float, T: float) -> float:
    """L_B = B - d_hat T / pi: sufficient largest cargo (Gleichung LB)."""
    _positive("B", B)
    return B - season_buffer(d_hat, T)


# ============================================================================
# PROFITABILITY (Definition 6, Lemma 2, Sätze 3-5)
# ============================================================================

def storage_cost_coefficient(h: float, T: float, G: float) -> float:
    """kappa = h T / (2 G): storage cost per departure is kappa L^2."""
    _positive("h", h)
    _positive("T", T)
    _positive("G", G)
    return h * T / (2.0 * G)


def unit_margin(L: float, a0: float, Cf: float, kappa: float) -> float:
    """m(L) = a0 - Cf / L - kappa L."""
    _positive("L", L)
    _positive("Cf", Cf)
    _positive("kappa", kappa)
    return a0 - Cf / L - kappa * L


def optimal_load(Cf: float, kappa: float) -> float:
    """EOQ load L_E = sqrt(Cf / kappa), the maximiser of m."""
    _positive("Cf", Cf)
    _positive("kappa", kappa)
    return math.sqrt(Cf / kappa)


def max_margin(a0: float, Cf: float, kappa: float) -> float:
    """m_max = a0 - 2 sqrt(kappa Cf)."""
    _positive("Cf", Cf)
    _positive("kappa", kappa)
    return a0 - 2.0 * math.sqrt(kappa * Cf)


def viable_load_interval(a0: float, eps: float, Cf: float,
                         kappa: float) -> Optional[Tuple[float, float]]:
    """
    Viable loads V_eps = {L : m(L) >= eps} = [L-, L+] (Sätze 3 and 4), or None
    if eps > m_max. For eps = 0 this is the profit interval of Satz 3.
    """
    _nonnegative("eps", eps)
    if eps > max_margin(a0, Cf, kappa):
        return None
    disc = max((a0 - eps) ** 2 - 4.0 * kappa * Cf, 0.0)
    root = math.sqrt(disc)
    return ((a0 - eps) - root) / (2.0 * kappa), ((a0 - eps) + root) / (2.0 * kappa)


def degradation_exponential(tau: float, delta_max: float, lam: float) -> float:
    """delta(tau) = 1 + (delta_max - 1)(1 - exp(-lam tau))."""
    _nonnegative("tau", tau)
    _positive("lam", lam)
    if not delta_max >= 1.0:
        raise ValueError(f"delta_max must be >= 1, got {delta_max}")
    return 1.0 + (delta_max - 1.0) * (1.0 - math.exp(-lam * tau))


def safety_margin(sigma_m: float, delta: float, p: float) -> float:
    """eps = z_p sigma_m sqrt(delta) with z_p = Phi^{-1}(1 - p), p in (0, 1/2]."""
    _positive("sigma_m", sigma_m)
    _positive("delta", delta)
    if not 0.0 < p <= 0.5:
        raise ValueError(f"p must be in (0, 0.5], got {p}")
    return float(norm.isf(p)) * sigma_m * math.sqrt(delta)


def critical_delta(m_max: float, sigma_m: float, p: float) -> float:
    """delta_crit = (m_max / (z_p sigma_m))^2 (Satz 4.4)."""
    _positive("m_max", m_max)
    _positive("sigma_m", sigma_m)
    if not 0.0 < p < 0.5:
        raise ValueError(f"p must be in (0, 0.5), got {p}")
    return (m_max / (float(norm.isf(p)) * sigma_m)) ** 2


def critical_horizon(delta_crit: float, delta_max: float,
                     lam: float) -> Optional[float]:
    """
    Critical planning horizon tau_crit (Gleichung taucrit).
    Returns math.inf if delta_crit >= delta_max (always viable), and None if
    delta_crit < 1 (not viable even at tau = 0).
    """
    _positive("lam", lam)
    if not delta_max > 1.0:
        raise ValueError(f"delta_max must be > 1, got {delta_max}")
    if delta_crit < 1.0:
        return None
    if delta_crit >= delta_max:
        return math.inf
    return -math.log(1.0 - (delta_crit - 1.0) / (delta_max - 1.0)) / lam


def power_law_fixed_cost(K: float, phi: float, beta: float) -> float:
    """Fixed cost C_f(K) = phi K^beta (economies of scale for beta < 1)."""
    _positive("K", K)
    _positive("phi", phi)
    _positive("beta", beta)
    return phi * K ** beta


def ship_unit_margin(L: float, K: float, a0: float, kappa: float,
                     fixed_cost: Callable[[float], float]) -> float:
    """m(L; K) = a0 - C_f(K)/L - kappa L for a ship of capacity K >= L (Satz 5)."""
    _positive("L", L)
    if K < L:
        raise ValueError("capacity K must be >= load L")
    return unit_margin(L, a0, fixed_cost(K), kappa)


# ============================================================================
# MARKET EQUILIBRIUM AND THE JUST-WORTHWHILE LOAD (Satz 6 / Hauptsatz)
# ============================================================================

def equilibrium_freight(m_L: float, eps: float, gamma: float, G: float) -> float:
    """Q_eq = 0 if m(L) < eps, else G + (m(L) - eps) / gamma."""
    _positive("gamma", gamma)
    _positive("G", G)
    if m_L < eps:
        return 0.0
    return G + (m_L - eps) / gamma


def equilibrium_oversupply(m_L: float, eps: float, gamma: float) -> float:
    """Delta_eq = (m(L) - eps)^+ / gamma."""
    _positive("gamma", gamma)
    return max(m_L - eps, 0.0) / gamma


def max_oversupply(a0: float, Cf: float, kappa: float, eps: float,
                   gamma: float) -> float:
    """Delta_max = (m_max - eps)^+ / gamma, attained at L_E (Hauptsatz 2)."""
    return equilibrium_oversupply(max_margin(a0, Cf, kappa), eps, gamma)


def just_worth_load(a0: float, eps: float, Cf: float, kappa: float) -> Optional[float]:
    """Smallest viable load L_eps^-: the just-worthwhile load (Korollar 2)."""
    interval = viable_load_interval(a0, eps, Cf, kappa)
    if interval is None:
        return None
    return interval[0]


def tolerance_band(a0: float, eps: float, Cf: float, kappa: float, gamma: float,
                   delta_tol: float) -> Optional[Tuple[Tuple[float, float],
                                                       Optional[Tuple[float, float]]]]:
    """
    Korollar 3: loads with m(L) >= eps and Delta_eq <= delta_tol.
    Returns (lower_branch, upper_branch); upper_branch is None if eps' > m_max
    (then the full viable interval is compatible and is returned as lower_branch).
    Returns None if no load is viable.
    """
    _nonnegative("delta_tol", delta_tol)
    _positive("gamma", gamma)
    viable = viable_load_interval(a0, eps, Cf, kappa)
    if viable is None:
        return None
    eps_prime = eps + gamma * delta_tol
    inner = viable_load_interval(a0, eps_prime, Cf, kappa)
    if inner is None:
        return viable, None
    return (viable[0], inner[0]), (inner[1], viable[1])


@dataclass
class IntegerDeparture:
    """Result of Korollar 4 (integer number of departures)."""
    N: int
    load: float
    margin: float
    oversupply: float
    beyond_eoq: bool


def integer_departures(G: float, a0: float, eps: float, Cf: float, kappa: float,
                       gamma: float) -> Optional[IntegerDeparture]:
    """
    N* = floor(G / L_eps^-): the integer departure count with the smallest
    oversupply among viable loads G/N (Korollar 4). None if no load is viable
    or fewer than one departure fits.
    """
    _positive("G", G)
    L_minus = just_worth_load(a0, eps, Cf, kappa)
    if L_minus is None:
        return None
    N = math.floor(G / L_minus)
    if N < 1:
        return None
    load = G / N
    m = unit_margin(load, a0, Cf, kappa)
    return IntegerDeparture(N=N, load=load, margin=m,
                            oversupply=equilibrium_oversupply(m, eps, gamma),
                            beyond_eoq=load > optimal_load(Cf, kappa))


# ============================================================================
# PRICE DISTURBANCE FROM PERIODIC OVERFLOW (Lemma 4, Satz 6)
# ============================================================================

def price_amplitude(alpha: float, eta: float, T: float) -> float:
    """A = eta / sqrt(alpha^2 + (2 pi / T)^2) of the limit cycle."""
    _positive("alpha", alpha)
    _positive("eta", eta)
    _positive("T", T)
    return eta / math.sqrt(alpha ** 2 + (2.0 * math.pi / T) ** 2)


def price_phase(alpha: float, T: float) -> float:
    """phi = -arctan(2 pi / (alpha T)) of the limit cycle."""
    _positive("alpha", alpha)
    _positive("T", T)
    return -math.atan(2.0 * math.pi / (alpha * T))


def disturbance_profile(s: float, beta: float, o: float, alpha: float,
                        T_s: float) -> float:
    """x_Delta(s) = -beta o e^{-alpha s} / (1 - e^{-alpha T_s}), s in [0, T_s)."""
    _positive("beta", beta)
    _positive("o", o)
    _positive("alpha", alpha)
    _positive("T_s", T_s)
    if not 0.0 <= s < T_s:
        raise ValueError("s must be in [0, T_s)")
    return -beta * o * math.exp(-alpha * s) / (1.0 - math.exp(-alpha * T_s))


def peak_disturbance(beta: float, delta: float, alpha: float, T: float, N: float) -> float:
    """sup |x_Delta| = beta o / (1 - e^{-alpha T / N}) with o = delta / N."""
    _positive("beta", beta)
    _positive("delta", delta)
    _positive("alpha", alpha)
    _positive("T", T)
    _positive("N", N)
    return beta * (delta / N) / (1.0 - math.exp(-alpha * T / N))


def mean_disturbance(beta: float, delta: float, alpha: float, T: float) -> float:
    """Time average of x_Delta: -beta Delta / (alpha T)."""
    _positive("beta", beta)
    _positive("delta", delta)
    _positive("alpha", alpha)
    _positive("T", T)
    return -beta * delta / (alpha * T)


def oversupply_tolerance(N: float, kappa_p: float, A: float, beta: float,
                         alpha: float, T: float) -> float:
    """Delta_tol(N) = kappa_p A / beta * N (1 - e^{-alpha T / N}) (Gleichung Dtol)."""
    _positive("N", N)
    _positive("kappa_p", kappa_p)
    _positive("A", A)
    _positive("beta", beta)
    _positive("alpha", alpha)
    _positive("T", T)
    return kappa_p * A / beta * N * (1.0 - math.exp(-alpha * T / N))


# ============================================================================
# DISTRIBUTION OVER ECONOMIC CELLS (Satz 7)
# ============================================================================

def _check_cells(gaps: Sequence[float], buffers: Sequence[float]) -> None:
    if len(gaps) == 0 or len(gaps) != len(buffers):
        raise ValueError("gaps and buffers must be non-empty and of equal length")
    for g, b in zip(gaps, buffers):
        _positive("cell gap", g)
        _positive("cell buffer", b)


def cell_shares(gaps: Sequence[float]) -> np.ndarray:
    """Unloading shares s_j = G_j / G forced by the cell balance."""
    arr = np.asarray(gaps, dtype=float)
    if arr.size == 0 or not np.all(arr > 0):
        raise ValueError("gaps must be non-empty and positive")
    return arr / arr.sum()


def cell_load_limit(gaps: Sequence[float], buffers: Sequence[float]) -> float:
    """L_cells = G min_j B_j / G_j: the weakest cell limits the cargo (Satz 7.2)."""
    _check_cells(gaps, buffers)
    g = np.asarray(gaps, dtype=float)
    b = np.asarray(buffers, dtype=float)
    return float(g.sum() * np.min(b / g))


def pool_load_limit(gaps: Sequence[float], buffers: Sequence[float]) -> float:
    """L_pool = sum_j B_j under free redistribution (Satz 7.3)."""
    _check_cells(gaps, buffers)
    return float(np.sum(buffers))


# ============================================================================
# PARAMETER SET AND COMBINED ANALYSIS
# ============================================================================

@dataclass(frozen=True)
class ShipMarketParams:
    """Full parameter set; defaults are the illustrative example of the chapter."""
    d0: float = 4000.0
    d_hat: float = 1000.0
    T: float = 1.0
    K_L: float = 2500.0
    B: float = 500.0
    a0: float = 40.0
    Cf: float = 1200.0
    h: float = 120.0
    gamma: float = 0.02
    sigma_m: float = 10.0
    delta_max: float = 2.0
    lam: float = 0.1
    tau: float = 6.0
    p: float = 0.05
    alpha: float = 4.0
    eta: float = 6.0
    beta: float = 0.02
    kappa_p: float = 1.0
    delta_tol: float = 0.0
    cell_gaps: Optional[Tuple[float, ...]] = None
    cell_buffers: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _check_demand(self.d0, self.d_hat, self.T)
        _nonnegative("K_L", self.K_L)
        for name in ("B", "a0", "Cf", "h", "gamma", "sigma_m", "lam",
                     "alpha", "eta", "beta", "kappa_p"):
            _positive(name, getattr(self, name))
        _nonnegative("tau", self.tau)
        _nonnegative("delta_tol", self.delta_tol)
        if not self.delta_max >= 1.0:
            raise ValueError("delta_max must be >= 1")
        if not 0.0 < self.p <= 0.5:
            raise ValueError("p must be in (0, 0.5]")
        _positive("G (period gap)", self.G)
        if (self.cell_gaps is None) != (self.cell_buffers is None):
            raise ValueError("cell_gaps and cell_buffers must be given together")
        if self.cell_gaps is not None:
            _check_cells(self.cell_gaps, self.cell_buffers)
            if not math.isclose(sum(self.cell_gaps), self.G, rel_tol=1e-9):
                raise ValueError("cell gaps must sum to the period gap G")

    @property
    def G(self) -> float:
        return period_gap(self.d0, self.d_hat, self.K_L, self.T)

    @property
    def kappa(self) -> float:
        return storage_cost_coefficient(self.h, self.T, self.G)

    @property
    def L_E(self) -> float:
        return optimal_load(self.Cf, self.kappa)

    @property
    def m_max(self) -> float:
        return max_margin(self.a0, self.Cf, self.kappa)

    @property
    def eps(self) -> float:
        delta = degradation_exponential(self.tau, self.delta_max, self.lam)
        return safety_margin(self.sigma_m, delta, self.p)

    @property
    def A(self) -> float:
        return price_amplitude(self.alpha, self.eta, self.T)


def price_margin(L: float, params: ShipMarketParams) -> float:
    """Delta_tol(G/L) - Delta_eq(L): >= 0 iff the price band holds (Satz vertr)."""
    P = params
    m = unit_margin(L, P.a0, P.Cf, P.kappa)
    tolerance = oversupply_tolerance(P.G / L, P.kappa_p, P.A, P.beta, P.alpha, P.T)
    return tolerance - equilibrium_oversupply(m, P.eps, P.gamma)


def is_price_compatible(L: float, params: ShipMarketParams) -> bool:
    """True iff the equilibrium oversupply respects the price band."""
    return price_margin(L, params) >= 0.0


def max_price_compatible_load(params: ShipMarketParams) -> Optional[float]:
    """
    Largest load on the lower branch [L-, L_E] with a compatible price band.
    On this branch oversupply increases and the tolerance decreases with L
    (Hauptsatz 2, Korollar 6), so price_margin is strictly decreasing and has
    at most one root. Returns None if no load is viable.
    """
    L_minus = just_worth_load(params.a0, params.eps, params.Cf, params.kappa)
    if L_minus is None:
        return None
    if is_price_compatible(params.L_E, params):
        return params.L_E
    return float(brentq(lambda L: price_margin(L, params), L_minus, params.L_E))


def tolerance_upper_bound(params: ShipMarketParams) -> Optional[float]:
    """L_eps'^- (or L_eps^+ if eps' > m_max), the upper limit of Korollar 3."""
    band = tolerance_band(params.a0, params.eps, params.Cf, params.kappa,
                          params.gamma, params.delta_tol)
    if band is None:
        return None
    return band[0][1]


def admissible_load_interval(params: ShipMarketParams) -> Optional[Tuple[float, float]]:
    """
    Boxed result of the chapter: [L_eps^-, min{L_eps'^-, B - d_hat T/pi,
    G min_j B_j/G_j}]. None if profitable and compatible loads do not exist.
    """
    lower = just_worth_load(params.a0, params.eps, params.Cf, params.kappa)
    if lower is None:
        return None
    upper = min(tolerance_upper_bound(params),
                buffer_limit_sufficient(params.B, params.d_hat, params.T))
    if params.cell_gaps is not None:
        upper = min(upper, cell_load_limit(params.cell_gaps, params.cell_buffers))
    if upper < lower:
        return None
    return lower, upper


@dataclass
class ShipFreightAnalysis:
    """Summary of the ship-freight analysis for one parameter set."""
    G: float
    kappa: float
    L_E: float
    m_max: float
    eps: float
    viable: Optional[Tuple[float, float]]
    just_worth_load: Optional[float]
    departures_at_just_worth: Optional[float]
    buffer_limit: float
    tolerance_upper: Optional[float]
    price_compatible_max: Optional[float]
    admissible: Optional[Tuple[float, float]]


def analyze(params: ShipMarketParams) -> ShipFreightAnalysis:
    """Evaluate all chapter results for one parameter set."""
    P = params
    viable = viable_load_interval(P.a0, P.eps, P.Cf, P.kappa)
    L_minus = None if viable is None else viable[0]
    return ShipFreightAnalysis(
        G=P.G, kappa=P.kappa, L_E=P.L_E, m_max=P.m_max, eps=P.eps,
        viable=viable,
        just_worth_load=L_minus,
        departures_at_just_worth=None if L_minus is None else P.G / L_minus,
        buffer_limit=buffer_limit_sufficient(P.B, P.d_hat, P.T),
        tolerance_upper=tolerance_upper_bound(P),
        price_compatible_max=max_price_compatible_load(P),
        admissible=admissible_load_interval(P),
    )


# ============================================================================
# SEA FREIGHT AND BACTERIAL DEVELOPMENT
# (chapter "Seefracht und bakterielle Entwicklung: Die Frische der Ladung auf
#  dem Meer", labels sb-*)
# ============================================================================
#
# Units: time in days, temperature in degrees Celsius, CO2 in per cent.
# (The economic part of this module measures time in years, T = 1; the
# conversion is explicit where both meet, see critical_voyage_duration.)
#
# Central objects:
#   mu      growth rate of an article in the local environment (eq. sb-mu)
#   Omega   growth work  = integral of mu (Definition sb-omega)
#   g_h0    log growth   y - y0 = g_h0(Omega) (Satz sb-loesung)
#   Omega_s shelf-life budget = g_h0^{-1}(Delta_s) (Definition sb-budget)
# The transport vehicle enters only through the local environment E(t)
# (Satz sb-invarianz); everything below is therefore a function of E(t).


def _check_chi(chi: float) -> None:
    if not 0.0 < chi <= 1.0:
        raise ValueError(f"chi must be in (0, 1], got {chi}")


def _check_prob(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


# ----------------------------------------------------------------------------
# Article parameters and growth rate (Definitions sb-artikel, sb-rate)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ArticleParams:
    """
    Article a with dominating spoilage flora (Definition sb-artikel):
    Ratkowsky coefficient b (d^-1/2 degC^-1), minimum temperature theta_min
    (degC), adaptation work h0 >= 0, admissible log increase delta_s > 0 and
    CO2 sensitivity kappa >= 0 (1/%).
    """
    b: float
    theta_min: float
    h0: float
    delta_s: float
    kappa: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        _positive("b", self.b)
        _nonnegative("h0", self.h0)
        _positive("delta_s", self.delta_s)
        _nonnegative("kappa", self.kappa)

    @property
    def omega_s(self) -> float:
        """Shelf-life budget Omega_s of the article (eq. sb-Os)."""
        return shelf_life_budget(self.delta_s, self.h0)


# Illustrative articles of Beispiele sb-modi and sb-klassen (chosen, not
# estimated): initial count 1e3, limit count 1e7, i.e. delta_s = ln(1e4).
ARTICLE_CLASS_C = ArticleParams(b=0.15, theta_min=-5.0, h0=1.5,
                                delta_s=math.log(1.0e4), kappa=0.3, name="C")
ARTICLE_CLASS_L = ArticleParams(b=0.10, theta_min=0.0, h0=3.0,
                                delta_s=math.log(1.0e4), kappa=0.0, name="L")


def atmosphere_factor(c: float, kappa: float) -> float:
    """chi(c) = 1 / (1 + kappa c) in (0, 1]: inhibition by CO2 (eq. sb-mu)."""
    _nonnegative("c", c)
    _nonnegative("kappa", kappa)
    return 1.0 / (1.0 + kappa * c)


def bacterial_growth_rate(theta: float, c: float, article: ArticleParams) -> float:
    """mu_a(theta, c) = chi_a(c) b^2 ((theta - theta_min)^+)^2 (eq. sb-mu)."""
    chi = atmosphere_factor(c, article.kappa)
    excess = max(theta - article.theta_min, 0.0)
    return chi * article.b ** 2 * excess ** 2


# ----------------------------------------------------------------------------
# Population model with adaptation phase (Satz sb-loesung, Lemma sb-g)
# ----------------------------------------------------------------------------

def log_growth(omega: float, h0: float) -> float:
    """
    g_h0(Omega) = ln(1 + (e^Omega - 1) e^-h0): log increase of the count after
    growth work Omega (eq. sb-lsg); g = Omega for h0 = 0. Evaluated in the
    overflow-safe form of Lemma sb-g (3) for Omega >= h0.
    """
    _nonnegative("omega", omega)
    _nonnegative("h0", h0)
    if h0 == 0.0:
        return omega
    if omega >= h0:
        return omega - h0 + math.log1p(math.expm1(h0) * math.exp(-omega))
    return math.log1p(math.expm1(omega) * math.exp(-h0))


def shelf_life_budget(delta_s: float, h0: float) -> float:
    """
    Omega_s = g_h0^{-1}(Delta_s) = ln(1 + (e^Delta_s - 1) e^h0) (eq. sb-Os),
    written in an overflow-safe form.
    """
    _positive("delta_s", delta_s)
    _nonnegative("h0", h0)
    return delta_s + h0 + math.log1p(-math.exp(-delta_s) * (1.0 - math.exp(-h0)))


def growth_bounds(omega: float, h0: float) -> Tuple[float, float]:
    """
    Two-regime bounds of Lemma sb-g (2):
    max{0, Omega - h0} <= g(Omega) <= (Omega - h0)^+ + ln(2 - e^-h0).
    """
    _nonnegative("omega", omega)
    _nonnegative("h0", h0)
    lower = max(0.0, omega - h0)
    upper = max(omega - h0, 0.0) + math.log(2.0 - math.exp(-h0))
    return lower, upper


def in_adaptation_regime(omega: float, h0: float) -> bool:
    """True iff Omega <= h0: the population does not double (Satz sb-regime)."""
    _nonnegative("omega", omega)
    _nonnegative("h0", h0)
    return omega <= h0


def population_ode(mu: Callable[[float], float], t_end: float, h0: float,
                   y0: float = 0.0, n_points: int = 201,
                   max_step: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerical solution of the population model (eq. sb-pop),
    y' = mu Q/(1+Q), Q' = mu Q, Q(0) = 1/(e^h0 - 1), y(0) = y0 (h0 = 0: y' = mu).
    Used to check Satz sb-loesung against the closed form; returns (t, y).
    """
    _positive("t_end", t_end)
    _nonnegative("h0", h0)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    t_eval = np.linspace(0.0, t_end, n_points)
    if h0 == 0.0:
        def rhs(t, state):
            return [mu(t)]
        state0 = [y0]
    else:
        def rhs(t, state):
            q = state[1]
            return [mu(t) * q / (1.0 + q), mu(t) * q]
        state0 = [y0, 1.0 / math.expm1(h0)]
    sol = solve_ivp(rhs, (0.0, t_end), state0, t_eval=t_eval, max_step=max_step,
                    rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0]


# ----------------------------------------------------------------------------
# Temperature as local information (Sätze sb-jensen, sb-lp, sb-exp)
# ----------------------------------------------------------------------------

def variance_surcharge(theta_bar: float, variance: float, theta_min: float) -> float:
    """rho = 1 + v / (theta_bar - theta_min)^2 >= 1 (Satz sb-jensen 2)."""
    _nonnegative("variance", variance)
    excess = theta_bar - theta_min
    _positive("theta_bar - theta_min", excess)
    return 1.0 + variance / excess ** 2


def segment_work(length: float, theta_bar: float, variance: float,
                 article: ArticleParams, chi: float = 1.0,
                 exposure: Optional[float] = None) -> float:
    """
    Growth work of one segment, chi b^2 l ((theta_bar - theta_min)^2 + v)
    (eq. sb-Om). If `exposure` = integral of chi dt is given (e.g. from
    atmosphere_exposure) it replaces chi * length. A constant segment at or
    below theta_min (frozen cargo) costs nothing (Korollar sb-gefroren); with
    variance below theta_min the formula does not apply.
    """
    _nonnegative("length", length)
    _nonnegative("variance", variance)
    _check_chi(chi)
    if theta_bar < article.theta_min:
        if variance > 0.0:
            raise ValueError("variance formula requires theta >= theta_min")
        return 0.0
    x = chi * length if exposure is None else exposure
    _nonnegative("exposure", x)
    return article.b ** 2 * x * ((theta_bar - article.theta_min) ** 2 + variance)


def jensen_lower_bound(length: float, theta_bar: float, article: ArticleParams,
                       chi: float = 1.0) -> float:
    """l chi b^2 ((theta_bar - theta_min)^+)^2: fluctuation never lowers Omega."""
    _nonnegative("length", length)
    _check_chi(chi)
    excess = max(theta_bar - article.theta_min, 0.0)
    return length * chi * article.b ** 2 * excess ** 2


def lowpass_amplitude(A: float, omega: float, tau_p: float) -> float:
    """Amplitude A / sqrt(1 + omega^2 tau_p^2) of the product temperature."""
    _nonnegative("A", A)
    _nonnegative("omega", omega)
    _positive("tau_p", tau_p)
    return A / math.sqrt(1.0 + (omega * tau_p) ** 2)


def lowpass_variance(A: float, omega: float, tau_p: float) -> float:
    """v_p = A^2 / (2 (1 + omega^2 tau_p^2)) (Satz sb-lp)."""
    return lowpass_amplitude(A, omega, tau_p) ** 2 / 2.0


def lowpass_phase(omega: float, tau_p: float) -> float:
    """Phase lag arctan(omega tau_p) of the product temperature (Satz sb-lp)."""
    _nonnegative("omega", omega)
    _positive("tau_p", tau_p)
    return math.atan(omega * tau_p)


def exponential_work(length: float, theta_inf: float, delta0: float, tau_p: float,
                     article: ArticleParams, chi: float = 1.0) -> float:
    """
    Growth work for theta(t) = theta_inf + delta0 e^{-t/tau_p} on [0, l]
    (eq. sb-expform): cooling of warm cargo (delta0 > 0), power failure
    (theta_inf = ambient, delta0 = set - ambient < 0). Requires theta(t) >=
    theta_min on the whole interval.
    """
    _nonnegative("length", length)
    _positive("tau_p", tau_p)
    _check_chi(chi)
    c_inf = theta_inf - article.theta_min
    if not c_inf > 0.0 or c_inf + delta0 < 0.0:
        raise ValueError("exponential formula requires theta(t) >= theta_min")
    bracket = (c_inf ** 2 * length
               + 2.0 * c_inf * delta0 * tau_p * (1.0 - math.exp(-length / tau_p))
               + delta0 ** 2 * tau_p / 2.0 * (1.0 - math.exp(-2.0 * length / tau_p)))
    return chi * article.b ** 2 * bracket


def exponential_fixed_cost(theta_inf: float, delta0: float, tau_p: float,
                           article: ArticleParams, chi: float = 1.0) -> float:
    """
    Duration-independent extra cost of the transient, the l -> infinity limit
    of exponential_work minus steady operation (Satz sb-exp):
    chi b^2 (2 c_inf delta0 tau_p + delta0^2 tau_p / 2).
    """
    _positive("tau_p", tau_p)
    _check_chi(chi)
    c_inf = theta_inf - article.theta_min
    _positive("theta_inf - theta_min", c_inf)
    return chi * article.b ** 2 * (2.0 * c_inf * delta0 * tau_p
                                   + delta0 ** 2 * tau_p / 2.0)


def ramp_work(length: float, theta0: float, rate: float, article: ArticleParams,
              chi: float = 1.0) -> float:
    """
    Growth work for the ramp theta(t) = theta0 + rate t, rate >= 0
    (eq. sb-rampe), e.g. handling between cold stores.
    """
    _nonnegative("length", length)
    _nonnegative("rate", rate)
    _check_chi(chi)
    c0 = theta0 - article.theta_min
    _nonnegative("theta0 - theta_min", c0)
    return chi * article.b ** 2 * (c0 ** 2 * length + c0 * rate * length ** 2
                                   + rate ** 2 * length ** 3 / 3.0)


# ----------------------------------------------------------------------------
# Shelf-life budget (Satz sb-budget, sb-regime, sb-grenzdauer)
# ----------------------------------------------------------------------------

def freshness_reserve(omega_total: float, omega_s: float) -> float:
    """F = Omega_s - Omega (Definition sb-budget)."""
    _nonnegative("omega_total", omega_total)
    _positive("omega_s", omega_s)
    return omega_s - omega_total


def freshness_degree(omega_total: float, omega_s: float) -> float:
    """phi = max{0, F} / Omega_s in [0, 1] (Definition sb-budget)."""
    return max(0.0, freshness_reserve(omega_total, omega_s)) / omega_s


def is_fresh(omega_total: float, omega_s: float) -> bool:
    """y <= y_s  iff  Omega <= Omega_s (Satz sb-budget 1)."""
    return freshness_reserve(omega_total, omega_s) >= 0.0


def exhaustion_time(omega_s: float, mu: float) -> float:
    """t* = Omega_s / mu at constant rate (Satz sb-budget 3); inf for mu = 0."""
    _positive("omega_s", omega_s)
    _nonnegative("mu", mu)
    if mu == 0.0:
        return math.inf
    return omega_s / mu


def residual_life(f_arrival: float, mu_hold: float) -> float:
    """R = F_ank / mu_R (Satz sb-budget 4); 0 for a spoiled arrival."""
    _nonnegative("mu_hold", mu_hold)
    if f_arrival <= 0.0:
        return 0.0
    if mu_hold == 0.0:
        return math.inf
    return f_arrival / mu_hold


def limit_duration(omega_s: float, omega_before: float, omega_after: float,
                   mu_sea: float) -> float:
    """
    Limit duration tau_S* = (Omega_s - Omega_vor - Omega_nach)^+ / mu_S
    (eq. sb-tauS); infinite for mu_S = 0 (frozen cargo, Korollar sb-gefroren).
    """
    _positive("omega_s", omega_s)
    _nonnegative("omega_before", omega_before)
    _nonnegative("omega_after", omega_after)
    _nonnegative("mu_sea", mu_sea)
    if mu_sea == 0.0:
        return math.inf
    return max(omega_s - omega_before - omega_after, 0.0) / mu_sea


# ----------------------------------------------------------------------------
# Transport profiles and vehicle comparison (Definition sb-profil, Satz sb-vergleich)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransportSegment:
    """
    One segment (l_j, theta_bar_j, v_j, chi_j) of a transport profile
    (Definition sb-profil). `exposure` optionally replaces chi * l by the
    integral of chi dt of a closing atmosphere (atmosphere_exposure).
    """
    length: float
    theta_bar: float
    variance: float = 0.0
    chi: float = 1.0
    exposure: Optional[float] = None

    def __post_init__(self) -> None:
        _positive("length", self.length)
        _nonnegative("variance", self.variance)
        _check_chi(self.chi)
        if self.exposure is not None:
            _nonnegative("exposure", self.exposure)


def profile_duration(segments: Sequence[TransportSegment]) -> float:
    """tau_m = sum of the segment durations."""
    if len(segments) == 0:
        raise ValueError("a profile needs at least one segment")
    return float(sum(s.length for s in segments))


def profile_work(segments: Sequence[TransportSegment], article: ArticleParams) -> float:
    """Omega_m = sum of the segment works (eq. sb-Om, additive: Satz sb-budget 2)."""
    if len(segments) == 0:
        raise ValueError("a profile needs at least one segment")
    return float(sum(segment_work(s.length, s.theta_bar, s.variance, article,
                                  s.chi, s.exposure) for s in segments))


def mean_rate(segments: Sequence[TransportSegment], article: ArticleParams) -> float:
    """Mean rate mu_m = Omega_m / tau_m."""
    return profile_work(segments, article) / profile_duration(segments)


def sea_temperature_threshold(theta_air: float, tau_air: float, tau_sea: float,
                              theta_min: float) -> float:
    """
    Highest constant sea temperature with Omega_S <= Omega_A against a constant
    air chain (eq. sb-gleich): theta_min + sqrt(tau_A / tau_S) (theta_A - theta_min).
    Independent of b, h0 and delta_s.
    """
    _positive("tau_air", tau_air)
    _positive("tau_sea", tau_sea)
    _positive("theta_air - theta_min", theta_air - theta_min)
    return theta_min + math.sqrt(tau_air / tau_sea) * (theta_air - theta_min)


def sea_not_worse_than_air(omega_sea: float, omega_air: float) -> bool:
    """Omega_S <= Omega_A (Satz sb-vergleich 1); then y_S <= y_A in every regime."""
    _nonnegative("omega_sea", omega_sea)
    _nonnegative("omega_air", omega_air)
    return omega_sea <= omega_air


def sea_suitability(omega_s: float, mu_sea: float, tau_sea: float) -> float:
    """
    sigma_a = Omega_s / (mu_a tau_S) (Satz sb-artikel); the voyage alone is fresh
    iff sigma_a >= 1, and sigma_a = inf for frozen/dry cargo (mu_a = 0).
    """
    _positive("omega_s", omega_s)
    _nonnegative("mu_sea", mu_sea)
    _positive("tau_sea", tau_sea)
    if mu_sea == 0.0:
        return math.inf
    return omega_s / (mu_sea * tau_sea)


def sea_voyage_feasible(sigma: float) -> bool:
    """The voyage can be made fresh (without pre/post chain) iff sigma_a >= 1."""
    return sigma >= 1.0


# ----------------------------------------------------------------------------
# Closed atmosphere and dominance change (Lemma sb-gas, Sätze sb-chi, sb-dominanz)
# ----------------------------------------------------------------------------

def gas_equilibrium(c_amb: float, source: float, leak: float) -> float:
    """c_inf = c_amb + s / k (Lemma sb-gas)."""
    _nonnegative("c_amb", c_amb)
    _nonnegative("source", source)
    _positive("leak", leak)
    return c_amb + source / leak


def gas_concentration(t: float, c0: float, c_inf: float, leak: float) -> float:
    """c(t) = c_inf + (c0 - c_inf) e^{-k t} (Lemma sb-gas)."""
    _nonnegative("t", t)
    _nonnegative("c0", c0)
    _nonnegative("c_inf", c_inf)
    _positive("leak", leak)
    return c_inf + (c0 - c_inf) * math.exp(-leak * t)


def equilibrium_fraction(leak: float, duration: float) -> float:
    """Reached fraction 1 - e^{-k t} of the equilibrium atmosphere (Korollar sb-zeitskala)."""
    _positive("leak", leak)
    _nonnegative("duration", duration)
    return 1.0 - math.exp(-leak * duration)


def atmosphere_exposure(length: float, c0: float, c_inf: float, leak: float,
                        kappa: float) -> float:
    """
    X(l) = integral_0^l chi(c(t)) dt = ln((A e^{kl} + B) / (A + B)) / (A k)
    with A = 1 + kappa c_inf, B = kappa (c0 - c_inf) (eq. sb-X). Then
    Omega = mu_0 X(l) at constant product temperature.
    """
    _nonnegative("length", length)
    _nonnegative("c0", c0)
    _nonnegative("c_inf", c_inf)
    _positive("leak", leak)
    _nonnegative("kappa", kappa)
    a = 1.0 + kappa * c_inf
    b = kappa * (c0 - c_inf)
    # ln(A e^{kl} + B) = kl + ln(A + B e^{-kl}): no overflow for long voyages
    return (leak * length + math.log(a + b * math.exp(-leak * length))
            - math.log(a + b)) / (a * leak)


def rate_ratio(c: float, rho0: float, kappa1: float, kappa2: float) -> float:
    """rho(c) = rho0 (1 + kappa2 c) / (1 + kappa1 c): ratio mu_1 / mu_2 (Satz sb-dominanz 1)."""
    _nonnegative("c", c)
    _positive("rho0", rho0)
    _nonnegative("kappa1", kappa1)
    _nonnegative("kappa2", kappa2)
    return rho0 * (1.0 + kappa2 * c) / (1.0 + kappa1 * c)


def critical_co2(rho0: float, kappa1: float, kappa2: float) -> Optional[float]:
    """
    c* = (rho0 - 1) / (kappa1 - rho0 kappa2) (eq. sb-cstar), where the
    dominating group changes; None if group 1 dominates for every c
    (kappa1 <= rho0 kappa2). Requires rho0 > 1 and kappa1 > kappa2.
    """
    if not rho0 > 1.0:
        raise ValueError(f"rho0 must be > 1, got {rho0}")
    _nonnegative("kappa2", kappa2)
    if not kappa1 > kappa2:
        raise ValueError(f"kappa1 must be > kappa2, got {kappa1}, {kappa2}")
    if kappa1 <= rho0 * kappa2:
        return None
    return (rho0 - 1.0) / (kappa1 - rho0 * kappa2)


def dominance_gap(t: float, y10: float, y20: float, h01: float, h02: float,
                  mu1: float, mu2: float) -> float:
    """(y1 - y2)(t) for constant rates, with the exact growth function (Satz sb-dominanz 3)."""
    _nonnegative("t", t)
    _nonnegative("mu1", mu1)
    _nonnegative("mu2", mu2)
    return (y10 + log_growth(mu1 * t, h01)) - (y20 + log_growth(mu2 * t, h02))


def dominance_certain(t: float, y10: float, y20: float, h01: float, h02: float,
                      mu1: float, mu2: float) -> bool:
    """
    Sufficient condition (eq. sb-dom4) for N_2(t) > N_1(t):
    (y20 - y10) + mu2 t - h02 > (mu1 t - h01)^+ + ln 2.
    """
    _nonnegative("t", t)
    _nonnegative("mu1", mu1)
    _nonnegative("mu2", mu2)
    _nonnegative("h01", h01)
    _nonnegative("h02", h02)
    return (y20 - y10) + mu2 * t - h02 > max(mu1 * t - h01, 0.0) + math.log(2.0)


# ----------------------------------------------------------------------------
# Pulse structure: age in the port store and freshness limit (Sätze sb-alter,
# sb-LF, sb-chance, sb-klemme)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AgeDistribution:
    """Uniform storage-age distribution of one cargo (Satz sb-alter)."""
    a_min: float
    a_max: float
    mean: float
    arrival_interval: float


def age_distribution(load: float, u_bar: float, y: float = 0.0) -> AgeDistribution:
    """
    FIFO storage age of the units of a cargo L released at rate u_bar with stock
    y before the call: uniform on [y/u_bar, (y + L)/u_bar], width T_s = L/u_bar
    (Satz sb-alter). For y = 0 the mean is T_s / 2.
    """
    _positive("load", load)
    _positive("u_bar", u_bar)
    _nonnegative("y", y)
    a_min = y / u_bar
    a_max = (y + load) / u_bar
    return AgeDistribution(a_min=a_min, a_max=a_max, mean=(a_min + a_max) / 2.0,
                           arrival_interval=load / u_bar)


def effective_budget(omega_s: float, sigma_omega: float, delta: float,
                     p: float) -> float:
    """
    Omega_s^eff = Omega_s - z_p sigma_Omega sqrt(delta(tau)) (eq. sb-Oseff),
    z_p = Phi^-1(1 - p): budget reduced by the safety reserve.
    """
    _positive("omega_s", omega_s)
    _nonnegative("sigma_omega", sigma_omega)
    _positive("delta", delta)
    if not 0.0 < p <= 0.5:
        raise ValueError(f"p must be in (0, 0.5], got {p}")
    return omega_s - float(norm.isf(p)) * sigma_omega * math.sqrt(delta)


def admissible_storage_age(omega_s_eff: float, omega_before: float,
                           omega_after: float, mu_store: float) -> float:
    """
    a_ok = (Omega_s^eff - Omega_vor - Omega_nach) / mu_st (eq. sb-aok). The value
    is not truncated: a negative a_ok means that no unit can stay fresh.
    """
    _nonnegative("omega_before", omega_before)
    _nonnegative("omega_after", omega_after)
    _positive("mu_store", mu_store)
    return (omega_s_eff - omega_before - omega_after) / mu_store


def freshness_load_limit(u_bar: float, a_ok: float, y: float = 0.0) -> float:
    """L_F = u_bar a_ok - y (eq. sb-LF): largest cargo of which every unit stays fresh."""
    _positive("u_bar", u_bar)
    _nonnegative("y", y)
    return u_bar * a_ok - y


def spoiled_fraction(load: float, l_f: float) -> float:
    """w(L) = min{1, (1 - L_F / L)^+} (eq. sb-w); 1 if L_F < 0."""
    _positive("load", load)
    if l_f < 0.0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - l_f / load))


def min_departures_for_freshness(T: float, a_ok: float, u_bar: float,
                                 y: float = 0.0) -> Optional[float]:
    """
    N_F = T / (a_ok - y/u_bar) (Satz sb-LF 3): perishable cargo forces more,
    smaller calls. None if a_ok <= y/u_bar (no call frequency is fresh).
    """
    _positive("T", T)
    _positive("u_bar", u_bar)
    _nonnegative("y", y)
    spare = a_ok - y / u_bar
    if spare <= 0.0:
        return None
    return T / spare


def fresh_and_worthwhile(l_f: float, just_worth: Optional[float]) -> bool:
    """
    Korollar sb-F-wirtschaft: a cargo that is fresh and just worthwhile exists
    iff 0 < L_F and L_eps^- <= L_F (and L_eps^- exists).
    """
    if just_worth is None:
        return False
    return l_f > 0.0 and just_worth <= l_f


def critical_voyage_duration(params: ShipMarketParams, mu_sea: float, mu_store: float,
                             omega_s: float, omega_fixed: float, tau0_days: float = 0.0,
                             y: float = 0.0, sigma_omega: float = 0.0,
                             days_per_horizon_unit: float = 365.0) -> Optional[float]:
    """
    Critical voyage duration tau_S^krit of the double clamp (Satz sb-klemme):
    the largest tau_S (days) with L_eps(tau)^- <= L_F(tau_S), where
        tau = tau0 + tau_S                      (planning horizon),
        Omega_vor + Omega_nach = omega_fixed + mu_S tau_S
                                                (omega_fixed = all work that does
                                                 not scale with the voyage),
        L_F = u_bar a_ok - y,  u_bar = G / T  (kt per day; T converted to days),
    and eps(tau) from the degradation function of `params`; L_eps^- uses the
    economic parameters of `params`. `days_per_horizon_unit` converts days to
    the time unit of tau (the economic chapter uses years, hence 365).
    Returns None if no duration is feasible (already tau_S = 0 fails, or no
    load is viable). With sigma_omega = 0 no uncertainty discount is applied.
    """
    _positive("mu_store", mu_store)
    _nonnegative("mu_sea", mu_sea)
    _positive("omega_s", omega_s)
    _nonnegative("omega_fixed", omega_fixed)
    _nonnegative("tau0_days", tau0_days)
    _nonnegative("y", y)
    _nonnegative("sigma_omega", sigma_omega)
    _positive("days_per_horizon_unit", days_per_horizon_unit)
    P = params
    u_bar = P.G / (P.T * days_per_horizon_unit)  # kt per day (T is in horizon units)

    def gap_to_economics(tau_s: float) -> float:
        horizon = (tau0_days + tau_s) / days_per_horizon_unit
        delta = degradation_exponential(horizon, P.delta_max, P.lam)
        eps = safety_margin(P.sigma_m, delta, P.p)
        l_minus = just_worth_load(P.a0, eps, P.Cf, P.kappa)
        if l_minus is None:
            return -math.inf
        budget = omega_s
        if sigma_omega > 0.0:
            budget = effective_budget(omega_s, sigma_omega, delta, P.p)
        a_ok = admissible_storage_age(budget, omega_fixed + mu_sea * tau_s, 0.0,
                                      mu_store)
        return freshness_load_limit(u_bar, a_ok, y) - l_minus

    if gap_to_economics(0.0) < 0.0:
        return None
    hi = 1.0
    while gap_to_economics(hi) >= 0.0:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap_to_economics(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------
# Correlated failure risk (Lemma sb-dstar, Satz sb-risiko)
# ----------------------------------------------------------------------------

def failure_work(duration: float, theta_set: float, theta_amb: float, tau_p: float,
                 article: ArticleParams, chi: float = 1.0) -> float:
    """
    Growth work of a cooling failure of length D: eq. sb-expform with
    theta_inf = theta_amb and delta0 = theta_set - theta_amb (Lemma sb-dstar).
    """
    return exponential_work(duration, theta_amb, theta_set - theta_amb, tau_p,
                            article, chi)


def failure_tolerance(reserve: float, theta_set: float, theta_amb: float, tau_p: float,
                      article: ArticleParams, chi: float = 1.0) -> float:
    """
    Tolerance duration D*(F) with Omega_fail(D*) = F (Lemma sb-dstar); strictly
    increasing in the reserve F > 0, so it falls during the voyage (Korollar
    sb-verwund).
    """
    _positive("reserve", reserve)
    hi = tau_p
    while failure_work(hi, theta_set, theta_amb, tau_p, article, chi) < reserve:
        hi *= 2.0
    return float(brentq(
        lambda d: failure_work(d, theta_set, theta_amb, tau_p, article, chi) - reserve,
        0.0, hi))


def ship_failure_mean(p_s: float, p_c: float) -> float:
    """E[X] = p_s + (1 - p_s) p_c (eq. sb-EX)."""
    _check_prob("p_s", p_s)
    _check_prob("p_c", p_c)
    return p_s + (1.0 - p_s) * p_c


def ship_failure_variance(p_s: float, p_c: float, n: int) -> float:
    """Var X = (1-p_s)(1-p_c)[p_s(1-p_c) + p_c/n] (eq. sb-VarX)."""
    _check_prob("p_s", p_s)
    _check_prob("p_c", p_c)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return (1.0 - p_s) * (1.0 - p_c) * (p_s * (1.0 - p_c) + p_c / n)


def ship_failure_variance_limit(p_s: float, p_c: float) -> float:
    """Non-diversifiable rest p_s (1 - p_s)(1 - p_c)^2 for n -> infinity (eq. sb-VarLim)."""
    _check_prob("p_s", p_s)
    _check_prob("p_c", p_c)
    return p_s * (1.0 - p_s) * (1.0 - p_c) ** 2


# ----------------------------------------------------------------------------
# Combined evaluation of one chain
# ----------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Evaluation of one transport chain for one article (Beispiel sb-modi)."""
    duration: float
    omega: float
    log_increase: float
    growth_factor: float
    budget_ratio: float
    fresh: bool
    residual_life: float


def evaluate_chain(segments: Sequence[TransportSegment], article: ArticleParams,
                   mu_hold: float) -> ChainResult:
    """
    Growth work, log increase y - y0 = g_h0(Omega), factor N/N0, Omega/Omega_s and
    the residual life R = (Omega_s - Omega)^+ / mu_hold of a chain.
    """
    _nonnegative("mu_hold", mu_hold)
    omega = profile_work(segments, article)
    omega_s = article.omega_s
    y = log_growth(omega, article.h0)
    return ChainResult(
        duration=profile_duration(segments),
        omega=omega,
        log_increase=y,
        growth_factor=math.exp(y),
        budget_ratio=omega / omega_s,
        fresh=is_fresh(omega, omega_s),
        residual_life=residual_life(omega_s - omega, mu_hold),
    )
