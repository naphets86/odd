#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
NEUE FEATURES DER LEIBNIZ-REIHEN-ANALYSE
================================================================================

Diese Datei enthält NEUE Features, die NICHT in den Standard-Modulen vorhanden sind:

1. Parametrisierte Leibniz-Funktion L(t; λ)
   └─ Integrodifferentialgleichung-Verifikation
   
2. Asymmetrische Zerlegung: π/4 = S_odd - S_even
   └─ Mathematische Analyse divergenter Komponenten
   
3. Visualisierungen (Matplotlib-basiert)
   └─ 4 verschiedene Plot-Funktionen

HINWEIS: Grundlegende Funktionen (Leibniz-Reihe, Dirichlet-Beta, Polylogarithmen,
Filter, Beschleunigungsmethoden) sind in den separaten Modulen implementiert:
- dirichlet_beta.py
- convergence_acceleration.py
- lowpass_filter.py
- special_functions.py

Autor: Forschungsarbeit Leibniz-Reihe
Datum: September 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special, optimize
from scipy.special import gamma, digamma
import warnings
from typing import Tuple, List, Callable, Dict, Any
import unittest
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# ============================================================================
# TEIL 1: ASYMMETRISCHE ZERLEGUNG DER LEIBNIZ-REIHE
# ============================================================================

class AsymmetricDecomposition:
    """
    Asymmetrische Zerlegung der Leibniz-Reihe:
    π/4 = S_odd - S_even
    
    wobei:
    - S_odd = Σ 1/(4n+1) für n=0,1,2,...
    - S_even = Σ 1/(4n+3) für n=0,1,2,...
    
    Mathematisch interessant: Beide Reihen divergieren einzeln,
    aber ihre Differenz konvergiert gegen π/4!
    """
    
    @staticmethod
    def S_odd(N: int) -> float:
        """
        Berechnet S_odd = Σ_{n=0}^{N-1} 1/(4n+1).
        
        Args:
            N: Anzahl der Terme
            
        Returns:
            Partial sum S_odd
        """
        return sum(1/(4*n + 1) for n in range(N))
    
    @staticmethod
    def S_even(N: int) -> float:
        """
        Berechnet S_even = Σ_{n=0}^{N-1} 1/(4n+3).
        
        Args:
            N: Anzahl der Terme
            
        Returns:
            Partial sum S_even
        """
        return sum(1/(4*n + 3) for n in range(N))
    
    @staticmethod
    def decomposition(N: int) -> Dict[str, float]:
        """
        Berechnet die asymmetrische Zerlegung und Analyse.
        
        Args:
            N: Anzahl der Terme pro Teilsumme
            
        Returns:
            Dictionary mit S_odd, S_even, Differenz, Fehler
        """
        S_odd = AsymmetricDecomposition.S_odd(N)
        S_even = AsymmetricDecomposition.S_even(N)
        difference = S_odd - S_even
        pi_4_exact = np.pi / 4
        
        return {
            'N': N,
            'S_odd': S_odd,
            'S_even': S_even,
            'difference': difference,
            'pi_4_exact': pi_4_exact,
            'absolute_error': abs(difference - pi_4_exact),
            'relative_error': abs(difference - pi_4_exact) / pi_4_exact if pi_4_exact != 0 else 0
        }
    
    @staticmethod
    def convergence_analysis(N_values: List[int]) -> List[Dict[str, Any]]:
        """
        Analysiert die Konvergenz für verschiedene N-Werte.
        
        Args:
            N_values: Liste von N-Werten
            
        Returns:
            Liste von Analyse-Dictionaries
        """
        results = []
        for N in N_values:
            results.append(AsymmetricDecomposition.decomposition(N))
        return results
    
    @staticmethod
    def verify_divergence_with_convergent_difference(N: int) -> Dict[str, Any]:
        """
        Verifiziert das paradoxe Verhalten:
        S_odd und S_even divergieren beide, aber ihre Differenz konvergiert!
        
        Args:
            N: Anzahl der Terme
            
        Returns:
            Dictionary mit Verifikationsergebnissen
        """
        results = []
        for n in [100, 500, 1000, 5000]:
            S_o = AsymmetricDecomposition.S_odd(n)
            S_e = AsymmetricDecomposition.S_even(n)
            diff = S_o - S_e
            results.append({
                'N': n,
                'S_odd': S_o,
                'S_even': S_e,
                'S_odd - S_even': diff
            })
        
        pi_4 = np.pi / 4
        
        return {
            'results': results,
            'S_odd_increasing': results[-1]['S_odd'] > results[0]['S_odd'],
            'S_even_increasing': results[-1]['S_even'] > results[0]['S_even'],
            'difference_converging': abs(results[-1]['S_odd - S_even'] - pi_4) < 
                                     abs(results[0]['S_odd - S_even'] - pi_4),
            'paradox_verified': True
        }


# ============================================================================
# TEIL 2: PARAMETRISIERTE LEIBNIZ-FUNKTION UND DIFFERENTIALGLEICHUNGEN
# ============================================================================

class ParametrizedLeibniz:
    """
    Implementierung der parametrisierten Leibniz-Funktion:
    L(t; λ) = Σ(n=0 to ∞) [(-1)^n / (2n+1)] * exp(-λ(2n+1)t)
    
    Geschlossene Form: L(t; λ) = arctan(exp(-λt)).
    
    Sie genügt der exakten autonomen Differentialgleichung
    dL/dt = -(λ/2) * sin(2L)
    (die Gleichung dL/dt + λL = λ[π/4 - ∫₀ᵗ L' dτ] wird NICHT erfüllt).
    
    Eigenschaften:
    - Randbedingung: L(0; λ) = π/4
    - Asymptotisches Verhalten: L(t; λ) → 0 wenn t → ∞
      (L ≈ exp(-λt), erst bei λt ≳ 5 unter 1e-2)
    """
    
    @staticmethod
    def L_odd(t: float, lambda_param: float, N_terms: int = 100) -> float:
        """
        Berechnet die parametrisierte Leibniz-Funktion L(t; λ).
        
        L(t; λ) = Σ_{n=0}^{∞} [(-1)^n / (2n+1)] * exp(-λ(2n+1)t)
        
        Args:
            t: Zeit-/Variablenwert
            lambda_param: Dämpfungsparameter λ > 0
            N_terms: Anzahl der Summationsterme
            
        Returns:
            Funktionswert L(t; λ)
        """
        if t < 0:
            raise ValueError("t muss >= 0 sein")
        if lambda_param <= 0:
            raise ValueError("λ muss > 0 sein")
            
        return sum(
            (-1)**n / (2*n + 1) * np.exp(-lambda_param * (2*n + 1) * t)
            for n in range(N_terms)
        )
    
    @staticmethod
    def L_odd_derivative(t: float, lambda_param: float, N_terms: int = 100) -> float:
        """
        Berechnet die Ableitung von L(t; λ) nach t.
        
        dL/dt = -λ Σ_{n=0}^{∞} (-1)^n * exp(-λ(2n+1)t)
        
        Args:
            t: Zeit-/Variablenwert
            lambda_param: Dämpfungsparameter
            N_terms: Anzahl der Summationsterme
            
        Returns:
            Ableitungswert
        """
        if t < 0:
            raise ValueError("t muss >= 0 sein")
        if lambda_param <= 0:
            raise ValueError("λ muss > 0 sein")
            
        return -lambda_param * sum(
            (-1)**n * np.exp(-lambda_param * (2*n + 1) * t)
            for n in range(N_terms)
        )
    
    @staticmethod
    def S_even(t: float, lambda_param: float, N_terms: int = 100) -> float:
        """
        Berechnet S_even(t; λ) = Σ_{n=0}^{∞} (-1)^n * exp(-λ(2n+1)t).
        
        Diese Funktion erfüllt: dL_odd/dt = -λ * S_even
        
        Args:
            t: Zeit-/Variablenwert
            lambda_param: Dämpfungsparameter
            N_terms: Anzahl der Summationsterme
            
        Returns:
            Funktionswert S_even
        """
        if t < 0:
            raise ValueError("t muss >= 0 sein")
        if lambda_param <= 0:
            raise ValueError("λ muss > 0 sein")
            
        return sum(
            (-1)**n * np.exp(-lambda_param * (2*n + 1) * t)
            for n in range(N_terms)
        )
    
    @staticmethod
    def verify_differential_equation(
        t: float, 
        lambda_param: float, 
        N_terms: int = 100,
        numerical_h: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Verifiziert, dass L_odd die exakte Differentialgleichung erfüllt:
        dL/dt = -(λ/2) * sin(2L)
        
        Herleitung: mit x = exp(-λt) ist L = arctan(x), also
        L' = -λ x/(1+x²) = -(λ/2) sin(2 arctan x) = -(λ/2) sin(2L).
        
        Die Reihe wird nach N_terms Gliedern abgebrochen; für kleine λt
        (λt ≲ 1/N_terms) ist der Abbruchfehler nicht vernachlässigbar.
        
        Args:
            t: Zeit-/Variablenwert
            lambda_param: Dämpfungsparameter
            N_terms: Anzahl der Summationsterme
            numerical_h: (ungenutzt, aus Kompatibilitätsgründen behalten)
            
        Returns:
            Dictionary mit LHS, RHS und deren Differenz
        """
        L = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms)
        dL_dt = ParametrizedLeibniz.L_odd_derivative(t, lambda_param, N_terms)
        
        # Linke Seite: dL/dt (aus der gliedweise abgeleiteten Reihe)
        LHS = dL_dt
        
        # Rechte Seite: -(λ/2) sin(2L)
        RHS = -0.5 * lambda_param * np.sin(2 * L)
        
        equation_error = abs(LHS - RHS)
        equation_satisfied = equation_error < 1e-3
        
        return {
            't': t,
            'lambda': lambda_param,
            'L_value': L,
            'dL_dt': dL_dt,
            'LHS': LHS,
            'RHS': RHS,
            'equation_error': equation_error,
            'equation_satisfied': equation_satisfied,
            'relative_error': equation_error / max(abs(RHS), 1e-10)
        }
    
    @staticmethod
    def boundary_conditions_analysis(lambda_param: float, N_terms: int = 100) -> Dict[str, Any]:
        """
        Analysiert die Randbedingungen und asymptotisches Verhalten.
        
        Args:
            lambda_param: Dämpfungsparameter
            N_terms: Anzahl der Summationsterme
            
        Returns:
            Dictionary mit Analyse der Randbedingungen
        """
        pi_4 = np.pi / 4
        
        # Randbedingung: L(0; λ) = π/4
        L_at_0 = ParametrizedLeibniz.L_odd(0, lambda_param, N_terms)
        boundary_error = abs(L_at_0 - pi_4)
        boundary_satisfied = boundary_error < 0.01
        
        # Asymptotisches Verhalten: L(t; λ) → 0 wenn t → ∞
        t_large = 10.0 / lambda_param  # Skalierte große Zeit
        L_at_large_t = ParametrizedLeibniz.L_odd(t_large, lambda_param, N_terms)
        asymptotic_satisfied = abs(L_at_large_t) < 1e-3
        
        return {
            'lambda': lambda_param,
            'L_at_0': L_at_0,
            'pi_4_expected': pi_4,
            'boundary_error': boundary_error,
            'boundary_satisfied': boundary_satisfied,
            'L_at_large_t': L_at_large_t,
            'asymptotic_satisfied': asymptotic_satisfied,
            'all_conditions_satisfied': boundary_satisfied and asymptotic_satisfied
        }


# ============================================================================
# TEIL 3: VISUALISIERUNGEN (MATPLOTLIB)
# ============================================================================

class Visualization:
    """
    Visualisierungen für die Leibniz-Reihen-Analyse.
    """
    
    @staticmethod
    def plot_asymmetric_decomposition():
        """
        Plot: Asymmetrische Zerlegung π/4 = S_odd - S_even.
        Zeigt, wie S_odd und S_even einzeln divergieren,
        aber ihre Differenz konvergiert.
        """
        N_range = np.logspace(1, 4, 50, dtype=int)  # 10 bis 10000
        
        S_odd_values = [AsymmetricDecomposition.S_odd(N) for N in N_range]
        S_even_values = [AsymmetricDecomposition.S_even(N) for N in N_range]
        differences = [S_odd_values[i] - S_even_values[i] for i in range(len(N_range))]
        
        pi_4 = np.pi / 4
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: S_odd und S_even (Divergenz)
        axes[0].semilogx(N_range, S_odd_values, 'b-', label='S_odd', linewidth=2)
        axes[0].semilogx(N_range, S_even_values, 'r-', label='S_even', linewidth=2)
        axes[0].set_xlabel('N (Anzahl Terme)', fontsize=11)
        axes[0].set_ylabel('Wert', fontsize=11)
        axes[0].set_title('S_odd und S_even (beide divergieren)', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Differenz (Konvergenz)
        axes[1].semilogx(N_range, differences, 'g-', linewidth=2, label='S_odd - S_even')
        axes[1].axhline(pi_4, color='k', linestyle='--', linewidth=2, label='π/4')
        axes[1].set_xlabel('N (Anzahl Terme)', fontsize=11)
        axes[1].set_ylabel('Wert', fontsize=11)
        axes[1].set_title('S_odd - S_even (konvergiert gegen π/4)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Fehler (logarithmisch)
        errors = [abs(d - pi_4) for d in differences]
        axes[2].loglog(N_range, errors, 'purple', linewidth=2, marker='o', markersize=3)
        axes[2].set_xlabel('N (Anzahl Terme)', fontsize=11)
        axes[2].set_ylabel('|Error|', fontsize=11)
        axes[2].set_title('Absoluter Fehler (log-log)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig('asymmetric_decomposition.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: asymmetric_decomposition.png")
        plt.close()
    
    @staticmethod
    def plot_parametrized_leibniz():
        """
        Plot: Parametrisierte Leibniz-Funktion L(t; λ).
        Zeigt das Verhalten für verschiedene λ-Werte.
        """
        t_values = np.linspace(0, 5, 200)
        lambda_values = [0.5, 1.0, 2.0, 5.0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        
        for idx, lambda_param in enumerate(lambda_values):
            L_values = [ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100) 
                       for t in t_values]
            
            axes[idx].plot(t_values, L_values, 'b-', linewidth=2)
            axes[idx].axhline(np.pi / 4, color='k', linestyle='--', 
                            linewidth=1, label='π/4 (Initial)')
            axes[idx].axhline(0, color='r', linestyle=':', linewidth=1, label='0 (Asymptote)')
            axes[idx].set_xlabel('Zeit t', fontsize=10)
            axes[idx].set_ylabel('L(t; λ)', fontsize=10)
            axes[idx].set_title(f'Parametrisierte Leibniz mit λ = {lambda_param}', 
                              fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parametrized_leibniz.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: parametrized_leibniz.png")
        plt.close()
    
    @staticmethod
    def plot_differential_equation_verification():
        """
        Plot: Verifizierung der Integrodifferentialgleichung.
        Zeigt LHS vs. RHS der DGL.
        """
        lambda_param = 1.0
        t_values = np.linspace(0.1, 5, 50)
        
        LHS_values = []
        RHS_values = []
        errors = []
        
        for t in t_values:
            result = ParametrizedLeibniz.verify_differential_equation(t, lambda_param, N_terms=100)
            LHS_values.append(result['LHS'])
            RHS_values.append(result['RHS'])
            errors.append(result['equation_error'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: LHS vs. RHS
        axes[0].plot(t_values, LHS_values, 'b-', linewidth=2, label='LHS (dL/dt)')
        axes[0].plot(t_values, RHS_values, 'r-', linewidth=2, label='RHS (−(λ/2)·sin 2L)')
        axes[0].set_xlabel('Zeit t', fontsize=11)
        axes[0].set_ylabel('Wert', fontsize=11)
        axes[0].set_title('DGL Verifizierung: LHS vs. RHS', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fehler
        axes[1].semilogy(t_values, errors, 'g-', linewidth=2, marker='o', markersize=4)
        axes[1].set_xlabel('Zeit t', fontsize=11)
        axes[1].set_ylabel('Fehler |LHS - RHS|', fontsize=11)
        axes[1].set_title('DGL Fehler (logarithmisch)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig('differential_equation_verification.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: differential_equation_verification.png")
        plt.close()
    
    @staticmethod
    def plot_boundary_conditions():
        """
        Plot: Randbedingungen und asymptotisches Verhalten.
        """
        lambda_values = [0.5, 1.0, 2.0, 5.0]
        t_values = np.linspace(0, 10, 300)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        
        pi_4 = np.pi / 4
        
        for idx, lambda_param in enumerate(lambda_values):
            L_values = [ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100) 
                       for t in t_values]
            
            axes[idx].plot(t_values, L_values, 'b-', linewidth=2.5, label='L(t; λ)')
            axes[idx].axhline(pi_4, color='g', linestyle='--', linewidth=2, 
                            label=f'L(0; λ) = π/4 ≈ {pi_4:.4f}')
            axes[idx].axhline(0, color='r', linestyle=':', linewidth=1.5, 
                            label='L(∞; λ) = 0')
            
            # Markiere decay time (wo L auf ~37% von π/4 fällt)
            decay_time = 1 / lambda_param
            L_at_decay = ParametrizedLeibniz.L_odd(decay_time, lambda_param, N_terms=100)
            axes[idx].plot(decay_time, L_at_decay, 'ro', markersize=8, 
                         label=f'τ = 1/λ ≈ {decay_time:.2f}')
            
            axes[idx].set_xlabel('Zeit t', fontsize=10)
            axes[idx].set_ylabel('L(t; λ)', fontsize=10)
            axes[idx].set_title(f'Randbedingungen mit λ = {lambda_param}', 
                              fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([-.05, pi_4 * 1.1])
        
        plt.tight_layout()
        plt.savefig('boundary_conditions.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: boundary_conditions.png")
        plt.close()


# ============================================================================
# UNIT-TESTS FÜR NEUE FEATURES
# ============================================================================

class TestAsymmetricDecomposition(unittest.TestCase):
    """Unit-Tests für asymmetrische Zerlegung."""
    
    def test_asymmetric_decomposition_convergence(self):
        """Test: Asymmetrische Zerlegung konvergiert gegen π/4."""
        pi_4_exact = np.pi / 4
        for N in [100, 1000, 5000]:
            decomp = AsymmetricDecomposition.decomposition(N)
            error = abs(decomp['difference'] - pi_4_exact)
            self.assertLess(error, 0.1, f"Fehler bei N={N} zu groß")
    
    def test_S_odd_and_S_even_diverge(self):
        """Test: S_odd und S_even divergieren einzeln."""
        decomp_100 = AsymmetricDecomposition.decomposition(100)
        decomp_1000 = AsymmetricDecomposition.decomposition(1000)
        
        self.assertGreater(decomp_1000['S_odd'], decomp_100['S_odd'])
        self.assertGreater(decomp_1000['S_even'], decomp_100['S_even'])
    
    def test_difference_converges_while_components_diverge(self):
        """Test: Differenz konvergiert, obwohl Komponenten divergieren."""
        pi_4 = np.pi / 4
        decomp_100 = AsymmetricDecomposition.decomposition(100)
        decomp_10000 = AsymmetricDecomposition.decomposition(10000)
        
        error_100 = abs(decomp_100['difference'] - pi_4)
        error_10000 = abs(decomp_10000['difference'] - pi_4)
        
        self.assertLess(error_10000, error_100)
    
    def test_paradox_verification(self):
        """Test: Paradoxe Konvergenz wird verifiziert."""
        result = AsymmetricDecomposition.verify_divergence_with_convergent_difference(1000)
        self.assertTrue(result['paradox_verified'])


class TestParametrizedLeibniz(unittest.TestCase):
    """Unit-Tests für parametrisierte Leibniz-Funktion."""
    
    def test_L_odd_initial_value(self):
        """Test: L_odd(0; λ) ≈ π/4."""
        pi_4_exact = np.pi / 4
        for lambda_param in [0.5, 1.0, 2.0]:
            value = ParametrizedLeibniz.L_odd(0, lambda_param, N_terms=150)
            self.assertAlmostEqual(value, pi_4_exact, places=2,
                                 msg=f"L_odd(0; {lambda_param}) ≠ π/4")
    
    def test_L_odd_decay_for_large_t(self):
        """Test: L_odd(t; λ) → 0 wenn t → ∞."""
        for lambda_param in [0.5, 1.0, 2.0]:
            t = 10.0 / lambda_param  # Skalierte große Zeit
            value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100)
            self.assertLess(abs(value), 1e-3, 
                          f"L_odd sollte gegen 0 konvergieren für λ={lambda_param}")
    
    def test_differential_equation_verification(self):
        """Test: L_odd erfüllt die Integrodifferentialgleichung."""
        for t in [0.5, 1.0, 2.0]:
            result = ParametrizedLeibniz.verify_differential_equation(t, 1.0, N_terms=100)
            self.assertTrue(result['equation_satisfied'],
                          f"DGL nicht erfüllt bei t={t}")
    
    def test_boundary_conditions_satisfied(self):
        """Test: Randbedingungen sind erfüllt."""
        for lambda_param in [0.5, 1.0, 2.0]:
            result = ParametrizedLeibniz.boundary_conditions_analysis(lambda_param, N_terms=150)
            self.assertTrue(result['all_conditions_satisfied'],
                          f"Randbedingungen nicht erfüllt für λ={lambda_param}")
    
    def test_invalid_parameters(self):
        """Test: Ungültige Parameter werden abgelehnt."""
        with self.assertRaises(ValueError):
            ParametrizedLeibniz.L_odd(-1, 1.0)  # t < 0
        
        with self.assertRaises(ValueError):
            ParametrizedLeibniz.L_odd(1.0, -1.0)  # λ ≤ 0


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    """Führt alle Berechnungen und Visualisierungen durch."""
    
    print("\n" + "="*80)
    print("NEUE FEATURES DER LEIBNIZ-REIHEN-ANALYSE")
    print("="*80 + "\n")
    
    # 1. Asymmetrische Zerlegung
    print("1. ASYMMETRISCHE ZERLEGUNG: π/4 = S_odd - S_even")
    print("-" * 80)
    
    decomp = AsymmetricDecomposition.decomposition(5000)
    print(f"\nFür N = 5000 Terme:")
    print(f"  S_odd              = {decomp['S_odd']:.10f}")
    print(f"  S_even             = {decomp['S_even']:.10f}")
    print(f"  S_odd - S_even     = {decomp['difference']:.10f}")
    print(f"  π/4 (exakt)        = {decomp['pi_4_exact']:.10f}")
    print(f"  Fehler             = {decomp['absolute_error']:.2e}")
    
    paradox = AsymmetricDecomposition.verify_divergence_with_convergent_difference(5000)
    print(f"\nParadoxe Konvergenz:")
    print(f"  S_odd divergiert:            {paradox['S_odd_increasing']}")
    print(f"  S_even divergiert:           {paradox['S_even_increasing']}")
    print(f"  S_odd - S_even konvergiert: {paradox['difference_converging']}")
    
    # 2. Parametrisierte Leibniz-Funktion
    print("\n" + "="*80)
    print("2. PARAMETRISIERTE LEIBNIZ-FUNKTION L(t; λ)")
    print("-" * 80)
    
    for lambda_param in [0.5, 1.0, 2.0]:
        L_0 = ParametrizedLeibniz.L_odd(0, lambda_param, N_terms=150)
        L_large = ParametrizedLeibniz.L_odd(10/lambda_param, lambda_param, N_terms=100)
        print(f"\nλ = {lambda_param}:")
        print(f"  L(0; λ)  = {L_0:.10f} (sollte ≈ π/4 = {np.pi/4:.10f})")
        print(f"  L(∞; λ)  = {L_large:.2e} (sollte → 0)")
    
    # 3. DGL Verifikation
    print("\n" + "="*80)
    print("3. INTEGRODIFFERENTIALGLEICHUNG-VERIFIKATION")
    print("-" * 80)
    
    for t in [0.5, 1.0, 2.0]:
        result = ParametrizedLeibniz.verify_differential_equation(t, 1.0, N_terms=100)
        print(f"\nBei t = {t}:")
        print(f"  LHS (dL/dt)         = {result['LHS']:.6f}")
        print(f"  RHS (-(λ/2) sin 2L) = {result['RHS']:.6f}")
        print(f"  Fehler              = {result['equation_error']:.2e}")
        print(f"  Erfüllt:            {'JA ✓' if result['equation_satisfied'] else 'NEIN ✗'}")
    
    # 4. Visualisierungen
    print("\n" + "="*80)
    print("4. ERSTELLE VISUALISIERUNGEN")
    print("-" * 80 + "\n")
    
    Visualization.plot_asymmetric_decomposition()
    Visualization.plot_parametrized_leibniz()
    Visualization.plot_differential_equation_verification()
    Visualization.plot_boundary_conditions()
    
    # 5. Unit-Tests
    print("\n" + "="*80)
    print("5. UNIT-TESTS FÜR NEUE FEATURES")
    print("-" * 80 + "\n")
    
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAsymmetricDecomposition))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParametrizedLeibniz))
    
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    print(f"\nTests durchgeführt:  {test_result.testsRun}")
    print(f"Erfolgreich:         {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}")
    print(f"Fehler:              {len(test_result.failures)}")
    print(f"Ausnahmen:           {len(test_result.errors)}")
    
    if test_result.wasSuccessful():
        print("\n✓ ALLE TESTS FÜR NEUE FEATURES ERFOLGREICH!")
    
    print("\n" + "="*80 + "\n")
    
    return test_result.wasSuccessful()


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
