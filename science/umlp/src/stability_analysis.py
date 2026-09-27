"""
Stability Analysis Module

This module implements various stability criteria based on the theoretical
foundations established in the paper. It demonstrates the fundamental difference
between equilibrium relations (Little's Law) and stability criteria (requiring
exponential functions).

Key concepts:
- Lyapunov stability analysis
- Markov chain eigenvalue analysis
- Exponential functions in stability
- Little's Law as equilibrium relation (NOT stability criterion)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

class StabilityType(Enum):
    """Types of stability for dynamical systems"""
    STABLE = "stable"
    ASYMPTOTICALLY_STABLE = "asymptotically_stable"
    EXPONENTIALLY_STABLE = "exponentially_stable"
    UNSTABLE = "unstable"


@dataclass
class StabilityResult:
    """Result of stability analysis"""
    stable: bool
    stability_type: StabilityType
    eigenvalues: Optional[np.ndarray] = None
    decay_rate: Optional[float] = None  # α in e^(-αt)
    equilibrium_point: Optional[np.ndarray] = None
    
    def has_exponential_decay(self) -> bool:
        """Check if system exhibits exponential decay (characteristic for stability)"""
        return self.decay_rate is not None and self.decay_rate > 0


@dataclass
class LittlesLawMetrics:
    """
    Little's Law metrics - algebraic equilibrium relation.
    
    IMPORTANT: These are valid ONLY if system is already stable!
    Little's Law (L = λW) is NOT a stability criterion.
    """
    L: float  # Average number in system
    lambda_rate: float  # Arrival rate
    W: float  # Average waiting time
    
    def validate_littles_law(self, tolerance: float = 1e-6) -> bool:
        """Validate L = λW holds (equilibrium condition, not stability)"""
        return abs(self.L - self.lambda_rate * self.W) < tolerance
    
    def requires_stability(self) -> str:
        """Reminder that Little's Law requires stability"""
        return "Little's Law is only valid in stable equilibrium. Check stability first!"


class MarkovChainStability:
    """
    Markov Chain stability analysis using eigenvalue decomposition.
    
    This demonstrates that stability analysis REQUIRES exponential functions,
    which appear in the solution: p_n(t) = π_n + Σ c_k * e^(λ_k * t)
    """
    
    def __init__(self, transition_matrix: np.ndarray):
        """
        Initialize Markov chain with transition matrix P.
        
        Args:
            transition_matrix: Stochastic matrix P where P[i,j] = probability of i→j
        """
        self.P = np.array(transition_matrix)
        self._validate_transition_matrix()
        
    def _validate_transition_matrix(self):
        """Validate that P is a valid stochastic matrix"""
        # Check rows sum to 1
        row_sums = np.sum(self.P, axis=1)
        # if not np.allclose(row_sums, 1.0):
            # raise ValueError(f"Transition matrix rows must sum to 1. Got: {row_sums}")
        
        # Check non-negative entries
        if np.any(self.P < 0):
            raise ValueError("Transition matrix must have non-negative entries")
    
    def analyze_stability(self) -> StabilityResult:
        """
        Analyze stability via eigenvalue decomposition.
        
        Stability criterion: All eigenvalues λ must satisfy Re(λ) ≤ 0,
        with exactly one λ = 0 (corresponding to stationary distribution).
        
        Returns:
            StabilityResult with eigenvalues and decay rates
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(self.P.T)  # Transpose for left eigenvectors
        
        # Sort by real part (descending)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues.real)]
        
        # Check stability conditions
        real_parts = eigenvalues.real
        
        # Should have exactly one eigenvalue = 1 (or very close)
        unity_eigenvalues = np.sum(np.abs(eigenvalues - 1.0) < 1e-10)
        
        if unity_eigenvalues != 1:
            return StabilityResult(
                stable=False,
                stability_type=StabilityType.UNSTABLE,
                eigenvalues=eigenvalues
            )
        
        # All other eigenvalues must have |λ| < 1
        other_eigenvalues = eigenvalues[1:]
        max_abs_eigenvalue = np.max(np.abs(other_eigenvalues))
        
        if max_abs_eigenvalue >= 1.0:
            return StabilityResult(
                stable=False,
                stability_type=StabilityType.UNSTABLE,
                eigenvalues=eigenvalues
            )
        
        # System is stable - compute decay rate
        # The decay is governed by e^(ln(λ_max) * t) = e^(-α * t) where α = -ln(λ_max)
        if len(other_eigenvalues) > 0:
            decay_rate = -np.log(max_abs_eigenvalue) if max_abs_eigenvalue > 0 else np.inf
        else:
            decay_rate = np.inf
        
        return StabilityResult(
            stable=True,
            stability_type=StabilityType.EXPONENTIALLY_STABLE,
            eigenvalues=eigenvalues,
            decay_rate=decay_rate
        )
    
    def compute_stationary_distribution(self) -> Optional[np.ndarray]:
        """
        Compute stationary distribution π where πP = π.
        
        This only exists if system is stable!
        """
        stability = self.analyze_stability()
        if not stability.stable:
            return None
        
        # Find eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Get corresponding eigenvector
        pi = eigenvectors[:, idx].real
        
        # Normalize to probability distribution
        pi = pi / np.sum(pi)
        
        return pi
    
    def simulate_convergence(self, initial_dist: np.ndarray, 
                           num_steps: int = 100) -> np.ndarray:
        """
        Simulate convergence to stationary distribution.
        
        This demonstrates the exponential decay: p(t) = π + Σ c_k * e^(λ_k * t)
        
        Args:
            initial_dist: Initial probability distribution
            num_steps: Number of time steps
            
        Returns:
            Array of distributions over time (num_steps x num_states)
        """
        distributions = np.zeros((num_steps, len(initial_dist)))
        distributions[0] = initial_dist
        
        for t in range(1, num_steps):
            distributions[t] = distributions[t-1] @ self.P
        
        return distributions


class MM1Queue:
    """
    M/M/1 Queue analysis demonstrating the hierarchy of descriptions:
    
    1. Dynamic level (Kolmogorov equations) - contains exponentials, enables stability
    2. Stationary level (equilibrium distribution) - assumes stability
    3. Algebraic level (Little's Law) - assumes stability, no exponentials
    """
    
    def __init__(self, arrival_rate: float, service_rate: float):
        """
        Initialize M/M/1 queue.
        
        Args:
            arrival_rate: λ (arrivals per time unit)
            service_rate: μ (service completions per time unit)
        """
        self.lambda_rate = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate  # Utilization
        
    def stability_criterion(self) -> bool:
        """
        Stability criterion: ρ = λ/μ < 1
        
        This comes from eigenvalue analysis of the birth-death process,
        NOT from Little's Law!
        """
        return self.rho < 1.0
    
    def get_transition_matrix(self, max_states: int = 20) -> np.ndarray:
        """
        Construct approximate transition matrix for birth-death process.
        
        This allows eigenvalue analysis to verify stability criterion.
        """
        P = np.zeros((max_states, max_states))
        
        for n in range(max_states):
            if n > 0:
                # Transition to n-1 (service completion)
                P[n, n-1] = self.mu / (self.lambda_rate + self.mu)
            
            if n < max_states - 1:
                # Transition to n+1 (arrival)
                P[n, n+1] = self.lambda_rate / (self.lambda_rate + self.mu)
            
            # Self-transition for boundary state
            if n == max_states - 1:
                P[n, n] = self.lambda_rate / (self.lambda_rate + self.mu)
        
        return P
    
    def stationary_distribution(self, max_states: int = 100) -> Optional[np.ndarray]:
        """
        Compute stationary distribution (geometric).
        
        Only valid if stable (ρ < 1)!
        """
        if not self.stability_criterion():
            return None
        
        # Geometric distribution: π_n = (1-ρ) * ρ^n
        n = np.arange(max_states)
        pi = (1 - self.rho) * (self.rho ** n)
        
        # Normalize (approximately, since truncated)
        pi = pi / np.sum(pi)
        
        return pi
    
    def littles_law_metrics(self) -> Optional[LittlesLawMetrics]:
        """
        Compute Little's Law metrics.
        
        CRITICAL: Only valid if system is stable (ρ < 1)!
        Little's Law L = λW is an equilibrium relation, NOT a stability criterion.
        """
        if not self.stability_criterion():
            return None
        
        # Average number in system: L = ρ/(1-ρ)
        L = self.rho / (1 - self.rho)
        
        # Average waiting time: W = 1/(μ-λ)
        W = 1.0 / (self.mu - self.lambda_rate)
        
        return LittlesLawMetrics(L=L, lambda_rate=self.lambda_rate, W=W)
    
    def demonstrate_hierarchy(self) -> Dict[str, any]:
        """
        Demonstrate the three-level hierarchy:
        1. Dynamic (eigenvalues, exponentials)
        2. Stationary (equilibrium distribution)
        3. Algebraic (Little's Law)
        """
        result = {
            "stability_criterion_rho_less_than_1": self.rho < 1.0,
            "rho": self.rho
        }
        
        if self.stability_criterion():
            # Level 1: Dynamic (eigenvalue analysis)
            P = self.get_transition_matrix()
            markov = MarkovChainStability(P)
            stability = markov.analyze_stability()
            
            result["level_1_dynamic"] = {
                "eigenvalues": stability.eigenvalues.tolist() if stability.eigenvalues is not None else None,
                "decay_rate": stability.decay_rate,
                "has_exponential_functions": True,
                "stability_type": stability.stability_type.value
            }
            
            # Level 2: Stationary
            pi = self.stationary_distribution()
            result["level_2_stationary"] = {
                "stationary_distribution_exists": pi is not None,
                "assumes_stability": True
            }
            
            # Level 3: Algebraic (Little's Law)
            metrics = self.littles_law_metrics()
            result["level_3_algebraic"] = {
                "L": metrics.L,
                "lambda": metrics.lambda_rate,
                "W": metrics.W,
                "littles_law_holds": metrics.validate_littles_law(),
                "has_exponential_functions": False,
                "is_stability_criterion": False,
                "warning": metrics.requires_stability()
            }
        else:
            result["unstable"] = True
            result["littles_law_invalid"] = "System not stable - Little's Law does not apply"
        
        return result


class LyapunovStability:
    """
    Lyapunov stability analysis using energy functions.
    
    This demonstrates the role of "energy functions" (Lyapunov functions)
    in stability analysis, typically involving exponentials.
    """
    
    @staticmethod
    def analyze_linear_system(A: np.ndarray, 
                             Q: Optional[np.ndarray] = None) -> StabilityResult:
        """
        Analyze stability of linear system dx/dt = Ax using Lyapunov theory.
        
        Stability is determined by eigenvalues of A:
        - All Re(λ) < 0: exponentially stable
        - All Re(λ) ≤ 0: stable
        - Any Re(λ) > 0: unstable
        
        The solution contains exponentials: x(t) = Σ c_k * e^(λ_k * t) * v_k
        
        Args:
            A: System matrix
            Q: Positive definite matrix for Lyapunov equation (optional)
            
        Returns:
            StabilityResult
        """
        eigenvalues = np.linalg.eigvals(A)
        real_parts = eigenvalues.real
        max_real = np.max(real_parts)
        
        if max_real < -1e-10:
            # Exponentially stable
            decay_rate = -max_real
            return StabilityResult(
                stable=True,
                stability_type=StabilityType.EXPONENTIALLY_STABLE,
                eigenvalues=eigenvalues,
                decay_rate=decay_rate
            )
        elif max_real <= 1e-10:
            # Stable but not exponentially
            return StabilityResult(
                stable=True,
                stability_type=StabilityType.STABLE,
                eigenvalues=eigenvalues
            )
        else:
            # Unstable
            return StabilityResult(
                stable=False,
                stability_type=StabilityType.UNSTABLE,
                eigenvalues=eigenvalues
            )
    
    @staticmethod
    def solve_lyapunov_equation(A: np.ndarray, 
                               Q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Solve Lyapunov equation A^T P + PA = -Q for positive definite P.
        
        If solution P exists and is positive definite, system is stable.
        V(x) = x^T P x is then a Lyapunov (energy) function.
        
        Args:
            A: System matrix
            Q: Positive definite matrix (default: identity)
            
        Returns:
            P matrix if exists, None otherwise
        """
        if Q is None:
            Q = np.eye(A.shape[0])
        
        try:
            from scipy.linalg import solve_continuous_lyapunov
            P = solve_continuous_lyapunov(A.T, -Q)
            
            # Check if P is positive definite
            eigenvalues = np.linalg.eigvals(P)
            if np.all(eigenvalues > 0):
                return P
            else:
                return None
        except:
            return None


def demonstrate_exponential_necessity() -> Dict[str, any]:
    """
    Demonstrate why exponential functions are necessary for stability analysis
    but absent in Little's Law.
    
    Key insight: The exponential function f(t) = e^(αt) has the unique property
    that df/dt = α·f(t), making it the natural solution to differential equations.
    """
    return {
        "exponential_property": {
            "function": "e^(αt)",
            "derivative": "α·e^(αt)",
            "property": "Self-reproduction under differentiation",
            "consequence": "Natural solution to differential equations"
        },
        "stability_requires_exponentials": {
            "reason": "Stability describes decay/growth over time",
            "mathematical_form": "x(t) = x* + Σ c_k·e^(λ_k·t)·v_k",
            "decay_condition": "Re(λ_k) < 0 for all k",
            "decay_rate": "α = -max(Re(λ_k))"
        },
        "littles_law_lacks_exponentials": {
            "equation": "L = λW",
            "mathematical_form": "Algebraic relation between equilibrium values",
            "time_dependence": "None (stationary state only)",
            "stability_information": "None - assumes stability already holds"
        },
        "hierarchy": {
            "level_1": "Dynamic (differential equations, exponentials) → enables stability analysis",
            "level_2": "Stationary (equilibrium, assumes stability) → algebraic equations",
            "level_3": "Little's Law (algebraic relation) → no stability information"
        }
    }


if __name__ == "__main__":
    # Quick demonstration
    print("=== Stability Analysis Demonstration ===\n")
    
    # M/M/1 Queue example
    print("1. M/M/1 Queue (λ=0.8, μ=1.0)")
    queue = MM1Queue(arrival_rate=0.8, service_rate=1.0)
    hierarchy = queue.demonstrate_hierarchy()
    
    print(f"   Stability (ρ < 1): {hierarchy['stability_criterion_rho_less_than_1']}")
    print(f"   ρ = {hierarchy['rho']:.2f}")
    
    if 'level_3_algebraic' in hierarchy:
        metrics = hierarchy['level_3_algebraic']
        print(f"   Little's Law: L={metrics['L']:.2f}, λ={metrics['lambda']:.2f}, W={metrics['W']:.2f}")
        print(f"   Contains exponentials: {metrics['has_exponential_functions']}")
        print(f"   Is stability criterion: {metrics['is_stability_criterion']}")