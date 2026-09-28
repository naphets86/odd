#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
PYTEST TESTABDECKUNG: NEUE FEATURES DER LEIBNIZ-REIHEN-ANALYSE
================================================================================

Tests für die NEUEN Features, die NICHT in den Standard-Modulen vorhanden sind:
- Asymmetrische Zerlegung π/4 = S_odd - S_even
- Parametrisierte Leibniz-Funktion L(t; λ)
- Integrodifferentialgleichung-Verifikation

Ausführung:
    pytest test_leibniz_clean.py -v
    pytest test_leibniz_clean.py --cov=leibniz_analysis
    pytest test_leibniz_clean.py -v --tb=short

================================================================================
"""

import contextlib
import os
import runpy
import tempfile

import matplotlib
matplotlib.use("Agg")  # kein GUI-Backend nötig (vor dem Import von leibniz_analysis)

import pytest
import numpy as np
from typing import Dict, List, Tuple
import time
import leibniz_analysis
from leibniz_analysis import (
    AsymmetricDecomposition,
    ParametrizedLeibniz,
    Visualization
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def pi_4_exact():
    """Fixture: Exakter Wert von π/4."""
    return np.pi / 4


@pytest.fixture
def tolerance():
    """Fixture: Toleranzen für verschiedene Berechnungen."""
    return {
        'high': 1e-15,      # Höchste Genauigkeit
        'normal': 1e-10,    # Normale Genauigkeit
        'loose': 1e-3,      # Lose Toleranz
        'very_loose': 0.1   # Sehr lose Toleranz
    }


@pytest.fixture
def n_values():
    """Fixture: Verschiedene Anzahlen von Termen für Tests."""
    return [100, 500, 1000, 5000, 10000]


@pytest.fixture
def lambda_values():
    """Fixture: Verschiedene λ-Parameter."""
    return [0.1, 0.5, 1.0, 2.0, 5.0]


@pytest.fixture
def t_values():
    """Fixture: Verschiedene Zeit-Werte."""
    return [0.1, 0.5, 1.0, 2.0, 5.0]


@contextlib.contextmanager
def in_tmp_dir():
    """Wechselt in ein temporäres Verzeichnis (die Plots speichern relativ)."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old)


# ============================================================================
# PYTEST MARKER - Kategorisierung von Tests
# ============================================================================

pytestmark = [
    pytest.mark.new_features,  # Alle Tests in dieser Datei
]


# ============================================================================
# TESTS FÜR ASYMMETRISCHE ZERLEGUNG
# ============================================================================

@pytest.mark.asymmetric
class TestAsymmetricDecomposition:
    """Tests für asymmetrische Zerlegung π/4 = S_odd - S_even."""
    
    @pytest.mark.parametrize("N", [100, 1000, 5000])
    def test_asymmetric_decomposition_convergence(self, N, pi_4_exact):
        """Test: Asymmetrische Zerlegung konvergiert gegen π/4."""
        decomp = AsymmetricDecomposition.decomposition(N)
        error = abs(decomp['difference'] - pi_4_exact)
        
        # S_odd - S_even sind die ersten 2N Leibniz-Terme (Fehler ≈ 1/(8N)).
        # Leibniz-Kriterium: Fehler < erstes weggelassenes Glied 1/(4N+1).
        assert error < 1.0 / (4 * N + 1), \
            f"Asymmetrische Zerlegung nicht konvergent für N={N}"
    
    def test_S_odd_and_S_even_both_diverge(self):
        """Test: S_odd und S_even divergieren einzeln."""
        decomp_100 = AsymmetricDecomposition.decomposition(100)
        decomp_1000 = AsymmetricDecomposition.decomposition(1000)
        decomp_5000 = AsymmetricDecomposition.decomposition(5000)
        
        # S_odd sollte monoton wachsen
        assert decomp_1000['S_odd'] > decomp_100['S_odd']
        assert decomp_5000['S_odd'] > decomp_1000['S_odd']
        
        # S_even sollte monoton wachsen
        assert decomp_1000['S_even'] > decomp_100['S_even']
        assert decomp_5000['S_even'] > decomp_1000['S_even']
    
    def test_difference_converges_while_components_diverge(self, pi_4_exact):
        """Test: Differenz konvergiert, während Komponenten divergieren (Paradoxon!)."""
        decomp_100 = AsymmetricDecomposition.decomposition(100)
        decomp_5000 = AsymmetricDecomposition.decomposition(5000)
        decomp_10000 = AsymmetricDecomposition.decomposition(10000)
        
        error_100 = abs(decomp_100['difference'] - pi_4_exact)
        error_5000 = abs(decomp_5000['difference'] - pi_4_exact)
        error_10000 = abs(decomp_10000['difference'] - pi_4_exact)
        
        # Fehler sollte monoton abnehmen
        assert error_5000 < error_100
        assert error_10000 < error_5000
    
    def test_paradox_verified(self):
        """Test: Das mathematische Paradoxon wird verifiziert."""
        result = AsymmetricDecomposition.verify_divergence_with_convergent_difference(5000)
        
        assert result['S_odd_increasing'], "S_odd sollte divergieren"
        assert result['S_even_increasing'], "S_even sollte divergieren"
        assert result['difference_converging'], "Differenz sollte konvergieren"
        assert result['paradox_verified'], "Paradoxon sollte verifiziert sein"
    
    def test_asymmetric_decomposition_return_values(self):
        """Test: Decomposition gibt alle erforderlichen Werte zurück."""
        decomp = AsymmetricDecomposition.decomposition(1000)
        
        required_keys = ['N', 'S_odd', 'S_even', 'difference', 
                        'pi_4_exact', 'absolute_error', 'relative_error']
        
        for key in required_keys:
            assert key in decomp, f"Key '{key}' fehlt in Ergebnis"
        
        # Werte sollten Zahlen sein
        assert isinstance(decomp['S_odd'], (int, float))
        assert isinstance(decomp['S_even'], (int, float))
        assert isinstance(decomp['difference'], (int, float))
    
    def test_convergence_analysis(self, n_values):
        """Test: Konvergenzanalyse funktioniert für mehrere N."""
        results = AsymmetricDecomposition.convergence_analysis(n_values)
        
        assert len(results) == len(n_values)
        
        for result in results:
            assert 'absolute_error' in result
            assert 'relative_error' in result
            assert result['absolute_error'] >= 0
            assert result['relative_error'] >= 0


# ============================================================================
# TESTS FÜR PARAMETRISIERTE LEIBNIZ-FUNKTION
# ============================================================================

@pytest.mark.parametrized_leibniz
class TestParametrizedLeibniz:
    """Tests für parametrisierte Leibniz-Funktion L(t; λ)."""
    
    @pytest.mark.parametrize("lambda_param", [0.5, 1.0, 2.0])
    def test_L_odd_initial_value(self, lambda_param, pi_4_exact):
        """Test: L_odd(0; λ) = π/4 (Randbedingung)."""
        value = ParametrizedLeibniz.L_odd(0, lambda_param, N_terms=150)
        
        # Toleranz für Finite-Terme-Approximation
        assert abs(value - pi_4_exact) < 0.01, \
            f"L_odd(0; {lambda_param}) ≠ π/4"
    
    @pytest.mark.parametrize("lambda_param,t", [
        (0.5, 10.0),   # λt = 5: L ≈ e^-5 ≈ 6.7e-3 (bei λt = 2.5 wäre L ≈ 0.08)
        (1.0, 5.0),
        (1.0, 10.0),
        (2.0, 3.0),
        (2.0, 5.0)
    ])
    def test_L_odd_decay_for_large_t(self, lambda_param, t):
        """Test: L_odd(t; λ) → 0 wenn t → ∞ (asymptotisches Verhalten)."""
        value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100)
        
        # Für große t sollte L_odd sehr klein sein
        assert abs(value) < 1e-2, \
            f"L_odd({t}; {lambda_param}) sollte klein sein"
    
    @pytest.mark.parametrize("lambda_param,t", [(0.5, 0.7), (1.0, 1.0), (2.0, 0.3)])
    def test_L_odd_matches_closed_form(self, lambda_param, t):
        """Test: L(t; λ) = arctan(exp(-λt)) (geschlossene Form)."""
        value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=200)
        assert abs(value - np.arctan(np.exp(-lambda_param * t))) < 1e-10
    
    def test_S_even_calculation(self):
        """Test: S_even Berechnung ist korrekt."""
        S_even_result = ParametrizedLeibniz.S_even(1.0, 1.0, N_terms=50)
        
        # S_even sollte nicht null sein
        assert S_even_result != 0
        assert isinstance(S_even_result, (int, float))
    
    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
    def test_differential_equation_small_error(self, t):
        """Test: Integrodifferentialgleichung wird mit kleinem Fehler erfüllt."""
        result = ParametrizedLeibniz.verify_differential_equation(t, 1.0, N_terms=100)
        
        assert result['equation_satisfied'], \
            f"DGL sollte bei t={t} erfüllt sein"
        
        # Fehler sollte klein sein
        assert result['equation_error'] < 1e-3
    
    @pytest.mark.parametrize("lambda_param", [0.5, 1.0, 2.0, 5.0])
    def test_boundary_conditions_satisfied(self, lambda_param):
        """Test: Randbedingungen sind erfüllt."""
        result = ParametrizedLeibniz.boundary_conditions_analysis(lambda_param, N_terms=150)
        
        assert result['boundary_satisfied'], \
            f"Randbedingung nicht erfüllt für λ={lambda_param}"
        
        assert result['asymptotic_satisfied'], \
            f"Asymptotisches Verhalten nicht erfüllt für λ={lambda_param}"
        
        assert result['all_conditions_satisfied'], \
            f"Nicht alle Bedingungen erfüllt für λ={lambda_param}"
    
    def test_L_odd_derivative_relation(self):
        """Test: Relation zwischen L_odd und seiner Ableitung."""
        t = 1.0
        lambda_param = 1.0
        
        L = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100)
        dL_dt = ParametrizedLeibniz.L_odd_derivative(t, lambda_param, N_terms=100)
        S_even = ParametrizedLeibniz.S_even(t, lambda_param, N_terms=100)
        
        # Sollte gelten: dL/dt = -λ * S_even
        expected_dL_dt = -lambda_param * S_even
        
        assert abs(dL_dt - expected_dL_dt) < 1e-6
    
    def test_monotone_decay(self, lambda_param=1.0):
        """Test: L_odd fällt monoton ab."""
        t_values = [0.0, 0.5, 1.0, 2.0, 3.0]
        L_values = [ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100) 
                   for t in t_values]
        
        # L sollte monoton abnehmen
        for i in range(len(L_values) - 1):
            assert L_values[i] >= L_values[i+1], \
                f"L({t_values[i]}) sollte >= L({t_values[i+1]}) sein"
    
    @pytest.mark.parametrize("t,lambda_param", [
        (-1.0, 1.0),     # t < 0
        (1.0, -1.0),     # λ < 0
        (1.0, 0.0),      # λ = 0
    ])
    def test_invalid_parameters(self, t, lambda_param):
        """Test: Ungültige Parameter werden abgelehnt."""
        with pytest.raises(ValueError):
            ParametrizedLeibniz.L_odd(t, lambda_param)
    
    def test_numerical_stability_large_lambda(self):
        """Test: Numerische Stabilität bei großem λ."""
        t = 1.0
        lambda_param = 10.0  # Großes λ
        
        value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=50)
        
        # Sollte nicht NaN oder Inf sein
        assert not np.isnan(value)
        assert not np.isinf(value)
        
        # Sollte sehr klein sein (schneller Decay)
        assert abs(value) < 1e-3
    
    def test_numerical_stability_small_lambda(self):
        """Test: Numerische Stabilität bei kleinem λ."""
        t = 1.0
        lambda_param = 0.01  # Kleines λ
        
        value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100)
        
        # Sollte nicht NaN oder Inf sein
        assert not np.isnan(value)
        assert not np.isinf(value)
        
        # Sollte nicht zu weit von π/4 entfernt sein
        pi_4 = np.pi / 4
        assert abs(value) < pi_4 * 1.5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration-Tests zwischen asymmetrischer Zerlegung und parametrisierter Leibniz."""
    
    def test_asymmetric_decomposition_and_parametrized_consistency(self, pi_4_exact):
        """Test: Asymmetrische Zerlegung und parametrisierte Leibniz sind konsistent."""
        # Asymmetrische Zerlegung
        decomp = AsymmetricDecomposition.decomposition(5000)
        asymmetric_result = decomp['difference']
        
        # Parametrisierte Leibniz bei t=0
        L_at_0 = ParametrizedLeibniz.L_odd(0, 1.0, N_terms=150)
        
        # Sollten beide gegen π/4 konvergieren
        assert abs(asymmetric_result - pi_4_exact) < 0.01
        assert abs(L_at_0 - pi_4_exact) < 0.01
    
    def test_differential_equation_verification_consistency(self):
        """Test: DGL-Verifikation ist konsistent für verschiedene t."""
        lambda_param = 1.0
        
        results = []
        for t in [0.5, 1.0, 2.0]:
            result = ParametrizedLeibniz.verify_differential_equation(t, lambda_param, N_terms=100)
            results.append(result)
            assert result['equation_satisfied']
        
        # Fehler sollte klein sein für alle t
        for result in results:
            assert result['equation_error'] < 1e-3


# ============================================================================
# EDGE CASES UND NUMERISCHE STABILITÄT
# ============================================================================

@pytest.mark.edge_cases
class TestEdgeCases:
    """Tests für Edge-Cases und Grenzsituationen."""
    
    def test_asymmetric_with_zero_terms(self):
        """Test: Asymmetrische Zerlegung mit N=0."""
        S_odd = AsymmetricDecomposition.S_odd(0)
        S_even = AsymmetricDecomposition.S_even(0)
        
        assert S_odd == 0
        assert S_even == 0
    
    def test_asymmetric_with_one_term(self):
        """Test: Asymmetrische Zerlegung mit N=1."""
        decomp = AsymmetricDecomposition.decomposition(1)
        
        assert decomp['S_odd'] == 1.0  # Nur 1/(4*0+1) = 1
        assert decomp['S_even'] == 1/3  # Nur 1/(4*0+3) = 1/3
        assert abs(decomp['difference'] - 2/3) < 1e-10
    
    def test_parametrized_leibniz_at_zero(self, pi_4_exact):
        """Test: Parametrisierte Leibniz bei t=0."""
        for lambda_param in [0.5, 1.0, 2.0]:
            value = ParametrizedLeibniz.L_odd(0, lambda_param, N_terms=200)
            # Sollte sehr nah bei π/4 sein
            assert abs(value - pi_4_exact) < 0.005
    
    def test_parametrized_leibniz_very_small_t(self):
        """Test: Parametrisierte Leibniz für sehr kleine t."""
        t = 1e-10
        lambda_param = 1.0
        
        value = ParametrizedLeibniz.L_odd(t, lambda_param, N_terms=100)
        pi_4 = np.pi / 4
        
        # Sollte sehr nah bei π/4 sein (kein signifikanter Decay bei tiny t)
        assert abs(value - pi_4) < 0.01


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests für numerische Stabilität."""
    
    def test_large_N_stability(self, pi_4_exact):
        """Test: Stabilität bei großen N."""
        large_N = 100000
        decomp = AsymmetricDecomposition.decomposition(large_N)
        
        # Sollte nicht NaN oder Inf sein
        assert not np.isnan(decomp['difference'])
        assert not np.isinf(decomp['difference'])
        
        # Sollte konvergieren
        assert abs(decomp['difference'] - pi_4_exact) < 0.001
    
    def test_lambda_parameter_extremes(self):
        """Test: Stabilität bei extremen λ-Werten."""
        t = 1.0
        
        # Sehr kleines λ
        value_small_lambda = ParametrizedLeibniz.L_odd(t, 0.001, N_terms=100)
        assert not np.isnan(value_small_lambda)
        assert not np.isinf(value_small_lambda)
        
        # Sehr großes λ
        value_large_lambda = ParametrizedLeibniz.L_odd(t, 1000.0, N_terms=100)
        assert not np.isnan(value_large_lambda)
        assert not np.isinf(value_large_lambda)
    
    def test_derivative_numerical_stability(self):
        """Test: Numerische Stabilität der Ableitung."""
        t_values = np.linspace(0.1, 5, 20)
        
        for t in t_values:
            dL = ParametrizedLeibniz.L_odd_derivative(t, 1.0, N_terms=100)
            
            assert not np.isnan(dL)
            assert not np.isinf(dL)
            assert isinstance(dL, (int, float))


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance-Tests."""
    
    def test_asymmetric_decomposition_speed(self, benchmark=None):
        """Test: Asymmetrische Zerlegung ist schnell."""
        def run_decomposition():
            return AsymmetricDecomposition.decomposition(1000)
        
        # Sollte sehr schnell sein (< 10ms)
        import time
        start = time.time()
        run_decomposition()
        elapsed = time.time() - start
        
        assert elapsed < 0.01, f"Zu langsam: {elapsed}s"
    
    def test_parametrized_leibniz_speed(self):
        """Test: Parametrisierte Leibniz ist schnell."""
        import time
        
        start = time.time()
        for _ in range(10):
            ParametrizedLeibniz.L_odd(1.0, 1.0, N_terms=100)
        elapsed = time.time() - start
        
        # 10 Aufrufe sollten < 100ms sein
        assert elapsed < 0.1, f"Zu langsam: {elapsed}s"


# ============================================================================
# UNGÜLTIGE PARAMETER (Ableitung, S_even)
# ============================================================================

class TestInvalidParametersDerivatives:
    """Fehlerpfade von L_odd_derivative und S_even."""
    
    @pytest.mark.parametrize("t,lambda_param", [(-1.0, 1.0), (1.0, -1.0), (1.0, 0.0)])
    def test_derivative_invalid(self, t, lambda_param):
        with pytest.raises(ValueError):
            ParametrizedLeibniz.L_odd_derivative(t, lambda_param)
    
    @pytest.mark.parametrize("t,lambda_param", [(-1.0, 1.0), (1.0, -1.0), (1.0, 0.0)])
    def test_S_even_invalid(self, t, lambda_param):
        with pytest.raises(ValueError):
            ParametrizedLeibniz.S_even(t, lambda_param)


# ============================================================================
# VISUALISIERUNGEN UND HAUPTPROGRAMM
# ============================================================================

@pytest.mark.visualization
class TestVisualization:
    """Die Plot-Funktionen erzeugen ihre PNG-Dateien."""
    
    @pytest.mark.parametrize("func,filename", [
        ("plot_asymmetric_decomposition", "asymmetric_decomposition.png"),
        ("plot_parametrized_leibniz", "parametrized_leibniz.png"),
        ("plot_differential_equation_verification", "differential_equation_verification.png"),
        ("plot_boundary_conditions", "boundary_conditions.png"),
    ])
    def test_plot_creates_file(self, func, filename):
        with in_tmp_dir() as tmp:
            getattr(Visualization, func)()
            path = os.path.join(tmp, filename)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 1000


@pytest.mark.integration
class TestMain:
    """Hauptprogramm (inkl. eingebauter unittest-Klassen und __main__-Block)."""
    
    def test_main_returns_true(self):
        with in_tmp_dir():
            assert leibniz_analysis.main() is True
    
    def test_script_exits_with_zero(self):
        with in_tmp_dir():
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(leibniz_analysis.__file__, run_name="__main__")
            assert excinfo.value.code == 0


# ============================================================================
# HELPER FUNCTION (nicht als Test erkannt)
# ============================================================================

def _helper_function_not_a_test():
    """Helper-Funktion (wird nicht als Test erkannt)."""
    return "This is a helper function"


if __name__ == '__main__':
    # Pytest direkt ausführen
    pytest.main([__file__, '-v', '--tb=short'])
