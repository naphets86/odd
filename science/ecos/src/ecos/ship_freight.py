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

Sign convention: the price dynamics are the restoring form
dP/dt = -alpha (P - P0) + eta sin(2 pi t / T)  (see Bemerkung 1 of the chapter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
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
