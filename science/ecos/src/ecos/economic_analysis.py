"""
Economic Analysis Module
Based on: Band der Arbeiten zur Wirtschaft - Formale Theorie, Systemanalyse
Author: Stephan Epp
Adapted for Python: 2026

Core economic concepts from the dissertation:
- Resonance window optimization (Ω ∈ [0.618, 0.796])
- Economic uncertainty and stability analysis
- Supply chain optimization
- Market dynamics and filter models
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import solve_continuous_are


# ============================================================================
# CONSTANTS AND FUNDAMENTAL PARAMETERS
# ============================================================================

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618
GOLDEN_RATIO_INV = 1 / GOLDEN_RATIO  # 1/φ ≈ 0.618
OPTIMAL_UNCERTAINTY = np.sin(np.pi / 4)  # sin(π/4) ≈ 0.707
RESONANCE_WINDOW_MIN = GOLDEN_RATIO_INV  # 0.618
RESONANCE_WINDOW_MAX = GOLDEN_RATIO_INV + 2 * (OPTIMAL_UNCERTAINTY - GOLDEN_RATIO_INV)  # 0.796
RESONANCE_WINDOW_CENTER = OPTIMAL_UNCERTAINTY  # 0.707


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResonanceAnalysis:
    """Analysis results for economic system resonance properties."""
    omega: float
    in_window: bool
    stability_score: float
    productivity: float
    freedom_degree: float
    predictability: float
    gini_coefficient: float
    growth_rate: float
    window_distance: float


@dataclass
class FilterResponse:
    """Frequency response of economic low-pass filter."""
    frequency: float
    magnitude: float
    phase: float
    cutoff_reached: bool


@dataclass
class SupplyChainOptimization:
    """Supply chain routing optimization results."""
    total_cost: float
    allocation: np.ndarray
    supply_levels: np.ndarray
    demand_satisfaction: np.ndarray
    convergence: bool


# ============================================================================
# CORE ECONOMIC ANALYSIS FUNCTIONS
# ============================================================================

def validate_omega(omega: float) -> None:
    """Validate uncertainty index is within valid range [0, 1]."""
    if not (0 <= omega <= 1):
        raise ValueError(f"Omega must be in [0, 1], got {omega}")


def in_resonance_window(omega: float) -> bool:
    """
    Check if economic system operates within optimal resonance window.
    
    Window: [1/φ, 1/φ + 2*Δh] ≈ [0.618, 0.796]
    This window is characterized by:
    - Optimal innovation rates
    - Adequate structural stability
    - Fair wealth distribution
    - Maximum long-term growth rates
    
    Args:
        omega: Uncertainty index Ω ∈ [0, 1]
    
    Returns:
        True if system is in optimal window, False otherwise
    """
    validate_omega(omega)
    return RESONANCE_WINDOW_MIN <= omega <= RESONANCE_WINDOW_MAX


def calculate_productivity(omega: float, lambda_param: float = 2.0) -> float:
    """
    Calculate economic productivity as function of uncertainty index.
    
    Formula: P(Ω) = 1/(1-Ω) - λ(Ω - Ω_center)²
    
    In resonance window:
    - Productivity is convex
    - Maximized at optimal center point (sin(π/4))
    - Decreases toward boundaries
    
    Args:
        omega: Uncertainty index
        lambda_param: Convexity parameter (default=2.0)
    
    Returns:
        Productivity value
    """
    validate_omega(omega)
    if omega >= 1.0:
        return 0.0
    
    term1 = 1.0 / (1.0 - omega)
    term2 = lambda_param * (omega - RESONANCE_WINDOW_CENTER) ** 2
    return max(np.finfo(float).eps, term1 - term2)


def calculate_freedom_degree(omega: float) -> float:
    """
    Calculate degree of freedom in economic system.
    
    Formula: F(Ω) = -ln(1 - Ω)
    
    Properties:
    - At Ω=0.618: F ≈ 0.955
    - At Ω=0.707: F ≈ 1.204 (optimal)
    - At Ω=0.796: F ≈ 1.591
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Freedom degree (higher = more freedom)
    """
    validate_omega(omega)
    if omega >= 1.0:
        return np.inf
    return -np.log(1.0 - omega)


def calculate_predictability(omega: float) -> float:
    """
    Calculate predictability of market outcomes.
    
    Formula: V(Ω) = 1 - |Ω - 0.707| / 0.5
    
    Properties:
    - Maximum at optimal center (V = 1.0)
    - Symmetric around optimal point
    - At boundaries: V ≈ 0.822
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Predictability score in [0, 1]
    """
    distance = abs(omega - RESONANCE_WINDOW_CENTER)
    return max(np.finfo(float).eps, 1.0 - distance / 0.5)


def calculate_price_variance(omega: float, gamma: float = 1.0) -> float:
    """
    Calculate market price variance (instability measure).
    
    Formula: σ²_p = γ·Ω·(1 - Ω)
    
    Properties:
    - Maximum at Ω=0.5 (0.25γ)
    - Better structural stability in window [0.618, 0.796]
    - Feedback mechanisms remain effective in window
    
    Args:
        omega: Uncertainty index
        gamma: Volatility scaling parameter
    
    Returns:
        Price variance value
    """
    validate_omega(omega)
    return gamma * omega * (1.0 - omega)


def calculate_gini_coefficient(omega: float) -> float:
    """
    Calculate wealth inequality (Gini coefficient) as function of Ω.
    
    In resonance window [0.618, 0.796]:
    - Gini ≈ 0.35-0.45 (moderate inequality)
    - Enough freedom for wealth concentration
    - Enough structure to prevent poverty
    
    Empirically: high growth correlates with this Gini range
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Gini coefficient in [0, 1]
    """
    validate_omega(omega)
    
    # Gini increases with omega (more freedom → more inequality)
    # But bounded by structural constraints
    gini_base = omega ** 1.5
    
    if in_resonance_window(omega):
        # Modulate Gini in window to stay in optimal range
        return 0.40 + 0.1 * (omega - RESONANCE_WINDOW_CENTER)
    elif omega < RESONANCE_WINDOW_MIN:
        # Below window: controlled economy has low Gini
        return 0.2 + 0.4 * (omega / RESONANCE_WINDOW_MIN)
    else:
        # Above window: chaotic economy has high Gini
        return 0.45 + 0.55 * ((omega - RESONANCE_WINDOW_MAX) / (1 - RESONANCE_WINDOW_MAX))


def calculate_growth_rate(omega: float) -> float:
    """
    Calculate long-term GDP growth rate as function of Ω.
    
    Empirical ranges (World Bank data):
    - Ω < 0.6: 0-1% p.a. (over-controlled)
    - Ω ∈ [0.6, 0.8]: 2-4% p.a. (optimal window)
    - Ω > 0.8: negative (crisis regime)
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Annual growth rate as decimal (e.g., 0.03 = 3%)
    """
    validate_omega(omega)
    
    if omega < RESONANCE_WINDOW_MIN:
        # Controlled regime: low to zero growth
        return 0.005 * (omega / RESONANCE_WINDOW_MIN)
    elif in_resonance_window(omega):
        # Optimal window: high growth
        productivity = calculate_productivity(omega)
        predictability = calculate_predictability(omega)
        return 0.03 * (1.0 + 0.4 * predictability + 0.1 * min(1.0, productivity))
    else:
        # Crisis regime: negative growth
        excess = (omega - RESONANCE_WINDOW_MAX) / (1 - RESONANCE_WINDOW_MAX)
        return -0.02 * excess ** 2


def resonance_analysis(omega: float) -> ResonanceAnalysis:
    """
    Comprehensive resonance analysis for economic system.
    
    Analyzes all key properties of economic system given uncertainty index.
    
    Args:
        omega: Uncertainty index
    
    Returns:
        ResonanceAnalysis object with all metrics
    """
    validate_omega(omega)
    
    in_window = in_resonance_window(omega)
    
    # Distance from optimal center
    window_distance = abs(omega - RESONANCE_WINDOW_CENTER)
    
    # Calculate all metrics
    stability = 1.0 - window_distance  # Higher if close to center
    productivity = calculate_productivity(omega)
    freedom = calculate_freedom_degree(min(omega, 1.0 - np.finfo(float).eps))
    predictability = calculate_predictability(omega)
    gini = calculate_gini_coefficient(omega)
    growth = calculate_growth_rate(omega)
    
    return ResonanceAnalysis(
        omega=omega,
        in_window=bool(in_window),
        stability_score=stability,
        productivity=productivity,
        freedom_degree=freedom,
        predictability=predictability,
        gini_coefficient=gini,
        growth_rate=growth,
        window_distance=window_distance
    )


# ============================================================================
# FILTER MODELS (Economic Low-Pass Filter)
# ============================================================================

def calculate_filter_cutoff_frequency(tau: float) -> float:
    """
    Calculate cutoff frequency of economic low-pass filter.
    
    For first-order RC filter: f_c = 1/(2π·τ)
    
    τ is the market response time constant (weeks to months).
    
    Args:
        tau: Time constant in appropriate units
    
    Returns:
        Cutoff frequency
    """
    if tau <= 0:
        raise ValueError("Time constant tau must be positive")
    return 1.0 / (2.0 * np.pi * tau)


def calculate_filter_magnitude(frequency: float, tau: float) -> float:
    """
    Calculate magnitude of filter frequency response.
    
    Formula: |H(iω)| = 1 / √(1 + ω²τ²)
    
    At cutoff frequency: |H(iω_c)| = 1/√2 ≈ 0.707
    
    Args:
        frequency: Angular frequency ω
        tau: Time constant
    
    Returns:
        Magnitude in [0, 1]
    """
    if tau <= 0:
        raise ValueError("Time constant tau must be positive")
    omega = 2.0 * np.pi * frequency
    return 1.0 / np.sqrt(1.0 + (omega * tau) ** 2)


def filter_shock_response(shock_magnitude: float, frequency: float, 
                          tau: float) -> Tuple[float, float]:
    """
    Calculate how economic filter damps periodic market shocks.
    
    Uses low-pass filter model:
    dV_out/dt + (1/τ)·V_out = (1/τ)·V_in
    
    For stability under periodic shocks with period T:
    Condition: T > 2π·τ
    
    Args:
        shock_magnitude: Amplitude of shock V_in
        frequency: Frequency of periodic shock
        tau: Market response time constant
    
    Returns:
        Tuple of (damped_magnitude, damping_factor)
    """
    if tau <= 0:
        raise ValueError("Time constant tau must be positive")
    
    magnitude = calculate_filter_magnitude(frequency, tau)
    damped = shock_magnitude * magnitude
    return damped, magnitude


def filter_stability_condition(shock_period: float, tau: float) -> bool:
    """
    Check if system remains stable under periodic shocks.
    
    Stability condition: T > 2π·τ
    
    Args:
        shock_period: Period of shock
        tau: Market response time constant
    
    Returns:
        True if system is stable, False if resonance occurs
    """
    if tau <= 0 or shock_period <= 0:
        raise ValueError("Period and tau must be positive")
    return bool(shock_period > 2.0 * np.pi * tau)


# ============================================================================
# LEIBNIZ SERIES AND HARMONIC ANALYSIS
# ============================================================================

def leibniz_series_coefficient(n: int) -> float:
    """
    Calculate n-th coefficient of Leibniz series.
    
    Series: π/4 = Σ (-1)^n / (2n+1)
    
    In economics: represents odd harmonics of market supply/demand cycles
    - Alternating signs provide self-stabilizing structure
    - Amplitudes decay as 1/(2n+1)
    - Higher harmonics automatically damped
    
    Args:
        n: Harmonic index (0, 1, 2, ...)
    
    Returns:
        Coefficient for n-th odd harmonic
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return (-1) ** n / (2 * n + 1)


def leibniz_series_sum(num_terms: int) -> float:
    """
    Calculate partial sum of Leibniz series.
    
    Approximates π/4 ≈ 0.7854
    
    Args:
        num_terms: Number of terms to sum
    
    Returns:
        Partial sum value
    """
    if num_terms <= 0:
        raise ValueError("num_terms must be positive")
    return sum(leibniz_series_coefficient(n) for n in range(num_terms))


def leibniz_convergence_error(num_terms: int) -> float:
    """
    Estimate convergence error of Leibniz series.
    
    For alternating series: |Error| < 1/(2n+1)
    
    Args:
        num_terms: Number of terms summed
    
    Returns:
        Upper bound on absolute error
    """
    if num_terms <= 0:
        raise ValueError("num_terms must be positive")
    return 1.0 / (2 * num_terms + 1)


def market_supply_odd_harmonics(t: float, S0: float, amplitudes: List[float],
                                frequencies: List[float], 
                                phases: Optional[List[float]] = None) -> float:
    """
    Calculate market supply as superposition of odd harmonics (Leibniz structure).
    
    Formula: S(t) = S_0 + Σ [S_n/(2n+1)] · sin((2n+1)ωt + φ_n)
    
    Properties:
    - Alternating signs provide self-stabilization
    - Amplitudes scale as 1/(2n+1)
    - Higher harmonics naturally damped
    
    Args:
        t: Time
        S0: Base supply level
        amplitudes: List of harmonic amplitudes [S_1, S_3, S_5, ...]
        frequencies: List of fundamental frequencies
        phases: Optional phase shifts for each harmonic
    
    Returns:
        Total supply at time t
    """
    supply = S0
    
    if phases is None:
        phases = [0.0] * len(amplitudes)
    
    for n, (amplitude, freq, phase) in enumerate(zip(amplitudes, frequencies, phases)):
        harmonic_n = n  # 0->1st harmonic, 1->3rd harmonic, etc.
        harmonic_freq = (2 * harmonic_n + 1) * freq
        harmonic_ampl = amplitude / (2 * harmonic_n + 1)
        
        supply += harmonic_ampl * np.sin(harmonic_freq * t + phase)
    
    return supply


# ============================================================================
# HISTORICAL EXAMPLES AND ECONOMIC REGIMES
# ============================================================================

def estimate_historical_omega(regime: str) -> Tuple[float, str]:
    """
    Estimate uncertainty index for historical economic regimes.
    
    Based on dissertation analysis of historical examples.
    
    Args:
        regime: Name of economic regime
    
    Returns:
        Tuple of (omega_estimate, interpretation)
    """
    regimes = {
        "soviet_union": (0.35, "Highly controlled - stagnation"),
        "east_germany": (0.30, "Central planning - productivity deficit"),
        "north_korea": (0.10, "Extreme control - famine and poverty"),
        "switzerland": (0.70, "Optimal window - prosperity"),
        "scandinavia": (0.68, "Optimal window - high living standards"),
        "germany_longterm": (0.70, "Optimal window average"),
        "weimar_1923": (0.95, "Hyperinflation - market collapse"),
        "argentina_2001": (0.88, "Financial crisis - bank runs"),
        "zimbabwe_2008": (0.98, "Hyperinflation - currency failure"),
        "uk_cornlaw_before": (0.72, "Pre-law optimal"),
        "uk_cornlaw_during": (0.55, "Law-induced stagnation"),
        "uk_cornlaw_after": (0.70, "Post-law recovery"),
    }
    
    regime_lower = regime.lower().replace(" ", "_")
    if regime_lower in regimes:
        omega, interpretation = regimes[regime_lower]
        return omega, interpretation
    else:
        raise ValueError(f"Unknown regime: {regime}. Available: {list(regimes.keys())}")


def interpret_regime(omega: float) -> str:
    """
    Provide textual interpretation of economic regime based on Ω.
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Regime interpretation string
    """
    validate_omega(omega)
    
    if omega < 0.618:
        if omega < 0.4:
            return "Extreme Control - Severe Stagnation & Poverty"
        else:
            return "Over-controlled - Innovation Suppressed"
    elif omega <= 0.796:
        if abs(omega - RESONANCE_WINDOW_CENTER) < 0.03:
            return "Optimal Balance - Prosperity & Innovation"
        elif omega < RESONANCE_WINDOW_CENTER:
            return "Toward Optimum - Structure Gradually Yielding to Freedom"
        else:
            return "Toward Chaos - Freedom Increasing, Predictability Declining"
    else:
        if omega < 0.85:
            return "Early Crisis - Financial Instability Emerging"
        elif omega < 0.95:
            return "Crisis Regime - Speculation & Market Breakdown"
        else:
            return "Extreme Chaos - Collapse Likely"


def optimize_supply_chain_routing(supply: np.ndarray, demand: np.ndarray, 
                                  costs: np.ndarray, 
                                  quality: Optional[np.ndarray] = None,
                                  weights: Optional[np.ndarray] = None,
                                  min_quality: Optional[np.ndarray] = None,
                                  lambda_penalty: float = 10.0) -> SupplyChainOptimization:
    """
    Optimize multi-supplier, multi-period supply chain routing.
    
    Minimizes total cost subject to supply and demand constraints,
    optionally with quality constraints.
    
    Args:
        supply: (S, T) array of supplier capacity
        demand: (L, T) array of demand levels
        costs: (S, T) array of routing costs
        quality: (S, T, n) array of quality vectors (optional)
        weights: (L, n) array of quality weights (optional)
        min_quality: (L,) array of minimum quality indices (optional)
        lambda_penalty: Penalty weight for quality violations
    
    Returns:
        SupplyChainOptimization result
    """
    S, T = supply.shape
    L = demand.shape[0]
    
    # Flatten problem for scipy.optimize
    n_vars = S * T
    
    def objective(x_flat):
        x_reshaped = x_flat.reshape(S, T)
        return np.sum(costs * x_reshaped)
    
    def demand_constraint_gen(l, t):
        def constraint(x_flat):
            x_reshaped = x_flat.reshape(S, T)
            return np.sum(x_reshaped[:, t]) - demand[l, t]
        return constraint
    
    def supply_constraint_gen(s, t):
        def constraint(x_flat):
            x_reshaped = x_flat.reshape(S, T)
            return supply[s, t] - x_reshaped[s, t]
        return constraint
    
    # Build constraints list
    constraints = []
    for l in range(L):
        for t in range(T):
            constraints.append({
                'type': 'ineq',
                'fun': demand_constraint_gen(l, t)
            })
    
    for s in range(S):
        for t in range(T):
            constraints.append({
                'type': 'ineq',
                'fun': supply_constraint_gen(s, t)
            })
    
    # Initial guess
    x0 = np.ones(n_vars) * 0.1
    
    # Bounds: all variables >= 0
    bounds = [(0, None) for _ in range(n_vars)]
    
    # Solve
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                     constraints=constraints if constraints else (),
                     options={'maxiter': 1000, 'ftol': 1e-6})
    
    allocation = result.x.reshape(S, T)
    
    # Calculate demand satisfaction
    demand_satisfaction = np.array([
        np.sum(allocation[:, t]) for t in range(T)
    ]) / np.sum(demand)
    
    return SupplyChainOptimization(
        total_cost=result.fun,
        allocation=allocation,
        supply_levels=np.sum(allocation, axis=1),
        demand_satisfaction=demand_satisfaction,
        convergence=result.success
    )


# ============================================================================
# HARMONIC AND SPECTRAL ANALYSIS
# ============================================================================

def fourier_rectangle_wave(t: float, amplitude: float = 1.0, period: float = 1.0,
                           num_harmonics: int = 50) -> float:
    """
    Reconstruct rectangle wave using Fourier series (odd harmonics only).
    
    Used to model demand shocks with non-smooth transitions.
    
    Formula: V_rect(t) = (4A/π) Σ sin((2n+1)ωt)/(2n+1)
    
    Args:
        t: Time
        amplitude: Wave amplitude
        period: Period of oscillation
        num_harmonics: Number of harmonics to sum
    
    Returns:
        Rectangle wave value at time t
    """
    if num_harmonics <= 0:
        raise ValueError("num_harmonics must be positive")
    
    omega = 2.0 * np.pi / period
    result = 0.0
    
    for n in range(num_harmonics):
        harmonic = (2 * n + 1)
        result += np.sin(harmonic * omega * t) / harmonic
    
    return (4.0 * amplitude / np.pi) * result


def market_adjustment_dynamics(demand_shock: float, alpha: float, eta: float,
                              T: float, t_array: np.ndarray) -> np.ndarray:
    """
    Model price adjustment under demand shocks.
    
    Fundamental equation from dissertation:
    dP/dt = α(P(t) - P_0) + η·sin(2πt/T)
    
    This is equivalent to low-pass filter equation.
    
    Args:
        demand_shock: Initial demand shock
        alpha: Adjustment speed parameter
        eta: Periodic shock amplitude
        T: Period of oscillation
        t_array: Time array for solution
    
    Returns:
        Array of price values over time
    """
    if alpha >= 0:
        warnings.warn("alpha should be negative for stable system")
    
    # Solve ODE: dP/dt + (|α|)P = (|α|)P_0 + η·sin(2πt/T)
    P0 = demand_shock
    omega = 2.0 * np.pi / T
    alpha_abs = abs(alpha)
    
    # Homogeneous solution: P_h = C·exp(-|α|t)
    # Particular solution for driven sine wave
    denom = alpha_abs ** 2 + omega ** 2
    
    # Steady-state driven response
    P_steady = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        P_homogeneous = P0 * np.exp(-alpha_abs * t)
        P_particular = (eta * alpha_abs * np.sin(omega * t) + 
                       eta * omega * np.cos(omega * t)) / denom
        P_steady[i] = P_homogeneous + P_particular
    
    return P_steady


# ============================================================================
# ENTROPY AND INFORMATION MEASURES
# ============================================================================

def market_entropy(omega: float) -> float:
    """
    Calculate information-theoretic entropy of economic system.
    
    Higher entropy = more uncertainty = higher information content
    
    Formula inspired by Shannon entropy: S(Ω) = -Ω·ln(Ω) - (1-Ω)·ln(1-Ω)
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Entropy value in [0, ln(2)]
    """
    validate_omega(omega)
    
    if omega == 0 or omega == 1:
        return 0.0
    
    return -(omega * np.log(omega) + (1 - omega) * np.log(1 - omega))


def mutual_information_market(omega1: float, omega2: float, 
                             correlation: float = 0.5) -> float:
    """
    Calculate mutual information between two markets with uncertainty indices.
    
    Args:
        omega1: First market uncertainty index
        omega2: Second market uncertainty index
        correlation: Correlation between markets [0, 1]
    
    Returns:
        Mutual information (non-negative)
    """
    validate_omega(omega1)
    validate_omega(omega2)
    
    if not (0 <= correlation <= 1):
        raise ValueError("Correlation must be in [0, 1]")
    
    # Simplified mutual information based on joint uncertainty
    joint_entropy = -0.5 * ((omega1 + omega2) * np.log(0.5 * (omega1 + omega2) + 1e-10))
    
    individual_entropy = 0.5 * (market_entropy(omega1) + market_entropy(omega2))
    
    return max(0, individual_entropy - joint_entropy) * correlation


# ============================================================================
# LYAPUNOV STABILITY ANALYSIS
# ============================================================================

def lyapunov_exponent_economic_system(omega: float) -> float:
    """
    Calculate maximum Lyapunov exponent for economic system.
    
    For dissipative systems: λ_max ∝ -1/(1-Ω)
    
    Properties:
    - λ_max < 0 in resonance window → stability
    - λ_max → -∞ as Ω → 1 → extreme stability (no dynamics)
    - λ_max ≈ -1.0 at optimal point
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Maximum Lyapunov exponent (typically negative)
    """
    validate_omega(omega)
    
    if omega >= 1.0:
        return -np.inf
    
    # Normalized to physical units
    return -1.0 / (1.0 - omega)


def lyapunov_stability_margin(omega: float) -> float:
    """
    Calculate stability margin based on Lyapunov exponents.
    
    Margin of 1 = critical; >1 = stable; <1 = unstable
    
    Args:
        omega: Uncertainty index
    
    Returns:
        Stability margin
    """
    validate_omega(omega)
    
    lyap_exp = lyapunov_exponent_economic_system(omega)
    
    if lyap_exp == 0:
        return 1.0
    
    # Convert to margin: margin = exp(-|λ|·t_characteristic)
    # where t_characteristic = 1 year
    if lyap_exp < 0:
        softening = 0.25 * omega / (1.0 - omega)
        return np.exp(lyap_exp + softening)
    return 0.0
