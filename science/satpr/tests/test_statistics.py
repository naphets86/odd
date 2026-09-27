"""
Tests für das Statistik-Modul.

Tests für:
- Phase-Übergänge
- Konvergenzanalyse
- Bimodale Verteilungen
- Entropie-Analysen
- Statistische Tests
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistics import (
    PhaseTransitionAnalysis,
    ConvergenceAnalysis,
    BimodalDistributionAnalysis,
    EntropyAnalysis,
    StatisticalTests
)


class TestPhaseTransitionAnalysis:
    """Tests für Phase-Übergangs-Analyse."""
    
    def test_detect_transition_points(self):
        """Test: Erkenne Übergangspunkte."""
        # Daten mit scharfem Sprung
        data = [1, 1, 1, 1, 5, 5, 5, 5, 15, 15, 15]
        
        transitions = PhaseTransitionAnalysis.detect_transition_points(data, window_size=2)
        
        assert len(transitions) > 0
    
    def test_detect_transition_gradual_change(self):
        """Test: Erkenne graduellen Wechsel."""
        # Gradueller Anstieg statt scharfer Sprung
        data = list(np.linspace(1, 20, 20))
        
        transitions = PhaseTransitionAnalysis.detect_transition_points(data, window_size=3)
        
        # Mit gradueller Änderung sollten weniger Übergänge sein
        assert len(transitions) <= 3
    
    def test_phase_duration_analysis(self):
        """Test: Analyse der Phasendauer."""
        # Simuliere 3-Phasen-Struktur
        data = (
            [1] * 5 +           # Phase 1: klein
            [5] * 5 +           # Phase 2: mittel
            [20] * 5            # Phase 3: groß
        )
        
        analysis = PhaseTransitionAnalysis.phase_duration_analysis(data, n_variables=4)
        
        assert 'theoretical_phase1_end' in analysis
        assert 'pyramid_size' in analysis
        assert analysis['pyramid_size'] == 2 ** 4


class TestConvergenceAnalysis:
    """Tests für Konvergenz-Analyse."""
    
    def test_exponential_convergence_test_true(self):
        """Test: Erkenne exponentielle Konvergenz."""
        # Daten die exponentiell konvergieren
        target = 100
        data = [target - 50 * np.exp(-0.5 * k) for k in range(20)]
        
        result = ConvergenceAnalysis.exponential_convergence_test(data, target)
        
        assert 'convergence_rate' in result
        assert result['convergence_rate'] > 0
    
    def test_exponential_convergence_test_false(self):
        """Test: Erkenne keine exponentielle Konvergenz."""
        # Lineare Konvergenz
        target = 100
        data = list(np.linspace(90, 99, 20))
        
        result = ConvergenceAnalysis.exponential_convergence_test(data, target)
        
        assert 'convergence_rate' in result or 'is_exponential' in result
    
    def test_halving_convergence_test(self):
        """Test: Teste Halbierungs-Konvergenz."""
        # Daten mit Verdoppelung (Halbierung des Fehlers)
        data = [100 / (2 ** k) for k in range(10)]
        
        result = ConvergenceAnalysis.halving_convergence_test(data)
        
        assert 'mean_ratio' in result
        # Sollte bei ~2.0 liegen (Verdoppelung)
        assert 1.5 < result['mean_ratio'] < 2.5


class TestBimodalDistributionAnalysis:
    """Tests für bimodale Verteilungs-Analyse."""
    
    # def test_estimate_modes_clear_bimodal(self):
    #     """Test: Erkenne klare bimodale Verteilung."""
    #     # Zwei Cluster
    #     data = list([1] * 50 + [10] * 50)
        
    #     mode1, mode2, weight1, weight2 = BimodalDistributionAnalysis.estimate_modes(data)
        
    #     assert mode1 < mode2
    #     assert np.isclose(weight1, 0.5, atol=0.1)
    #     assert np.isclose(weight2, 0.5, atol=0.1)
    
    def test_estimate_modes_unimodal(self):
        """Test: Erkenne unimodale Verteilung."""
        # Ein Cluster
        data = list(np.random.normal(5, 1, 100))
        
        mode1, mode2, weight1, weight2 = BimodalDistributionAnalysis.estimate_modes(data)
        
        # Sollten nah beieinander liegen
        assert abs(mode1 - mode2) < 3
    
    def test_kullback_leibler_divergence_identical(self):
        """Test: KL-Divergenz identischer Verteilungen."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.2, 0.3, 0.5])
        
        kl = BimodalDistributionAnalysis.kullback_leibler_divergence(p, q)
        
        assert np.isclose(kl, 0.0, atol=1e-10)
    
    def test_kullback_leibler_divergence_different(self):
        """Test: KL-Divergenz verschiedener Verteilungen."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        
        kl = BimodalDistributionAnalysis.kullback_leibler_divergence(p, q)
        
        assert kl > 0
        assert np.isfinite(kl)


class TestEntropyAnalysis:
    """Tests für Entropie-Analyse."""
    
    def test_shannon_entropy_uniform(self):
        """Test: Shannon-Entropie der Gleichverteilung."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        entropy = EntropyAnalysis.shannon_entropy(probs)
        
        # H(U) = log_2(4) = 2
        assert np.isclose(entropy, 2.0)
    
    def test_shannon_entropy_deterministic(self):
        """Test: Entropie der deterministischen Verteilung."""
        probs = np.array([1.0, 0.0, 0.0])
        
        entropy = EntropyAnalysis.shannon_entropy(probs)
        
        assert np.isclose(entropy, 0.0)
    
    def test_shannon_entropy_binary(self):
        """Test: Entropie der Binärverteilung."""
        probs = np.array([0.5, 0.5])
        
        entropy = EntropyAnalysis.shannon_entropy(probs)
        
        # H(B) = 1
        assert np.isclose(entropy, 1.0)
    
    def test_renyi_entropy_coincides_shannon(self):
        """Test: Rényi-Entropie α=1 = Shannon."""
        probs = np.array([0.2, 0.3, 0.5])
        
        shannon = EntropyAnalysis.shannon_entropy(probs)
        renyi1 = EntropyAnalysis.renyi_entropy(probs, alpha=1.0)
        
        assert np.isclose(shannon, renyi1, atol=1e-10)
    
    # def test_renyi_entropy_alpha2(self):
    #     """Test: Rényi-Entropie für α=2."""
    #     probs = np.array([0.5, 0.5])
        
    #     renyi2 = EntropyAnalysis.renyi_entropy(probs, alpha=2.0)
        
    #     # H_2(U) = log_2(Σ p_i^2) = log_2(0.5) = -1
    #     expected = np.log2(0.25 + 0.25)
    #     assert np.isclose(renyi2, expected)
    
    def test_entropy_change_rate(self):
        """Test: Berechne Entropie-Änderungsrate."""
        # Entropie nimmt ab
        sequence = [2.0, 1.8, 1.6, 1.4, 1.0]
        
        analysis = EntropyAnalysis.entropy_change_rate(sequence)
        
        assert 'mean_change' in analysis
        assert analysis['is_decreasing'] == True
        assert analysis['mean_change'] < 0


class TestStatisticalTests:
    """Tests für statistische Tests."""
    
    def test_kolmogorov_smirnov_normal(self):
        """Test: KS-Test gegen Normal-Verteilung."""
        from scipy import stats as sp_stats
        
        # Generiere Normal-verteilte Daten
        data = np.random.normal(0, 1, 100)
        
        result = StatisticalTests.kolmogorov_smirnov_test(
            data, 
            sp_stats.norm.cdf
        )
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['p_value'] > 0
    
    def test_chi_square_test_independence(self):
        """Test: Chi-Quadrat Test."""
        observed = np.array([10, 15, 25])
        expected = np.array([10, 10, 30])
        
        result = StatisticalTests.chi_square_test(observed, expected)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['statistic'] >= 0
    
    def test_mann_whitney_u_test(self):
        """Test: Mann-Whitney U Test."""
        data1 = np.random.normal(0, 1, 30)
        data2 = np.random.normal(1, 1, 30)  # Verschoben
        
        result = StatisticalTests.mann_whitney_u_test(data1, data2)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1


class TestIntegration:
    """Integrationstests für statistische Analysen."""
    
    def test_phase_transition_to_convergence_pipeline(self):
        """Test: Pipeline von Phase-Erkennung zu Konvergenz-Analyse."""
        # Generiere 3-Phasen-Daten
        data = (
            list(np.random.normal(2, 0.5, 10)) +        # Phase 1
            list(np.random.normal(10, 2, 10)) +          # Phase 2
            list(np.random.normal(50, 3, 10))            # Phase 3
        )
        
        # Erkenne Übergänge
        transitions = PhaseTransitionAnalysis.detect_transition_points(data, window_size=3)
        
        # Analyse Konvergenz
        target = 50
        result = ConvergenceAnalysis.exponential_convergence_test(data, target)
        
        assert len(transitions) > 0
        assert result is not None
    
    def test_entropy_dynamics_in_phases(self):
        """Test: Entropie-Dynamik über Phasen."""
        # Phase 1: Hohe Entropie (viele mögliche Outcomes)
        phase1_dist = np.array([0.3, 0.3, 0.2, 0.2])
        
        # Phase 3: Niedrige Entropie (wenige wahrscheinliche Outcomes)
        phase3_dist = np.array([0.9, 0.05, 0.04, 0.01])
        
        entropy1 = EntropyAnalysis.shannon_entropy(phase1_dist)
        entropy3 = EntropyAnalysis.shannon_entropy(phase3_dist)
        
        assert entropy1 > entropy3
    
    def test_bimodal_to_unimodal_transition(self):
        """Test: Übergang von bimodal zu unimodal."""
        # Frühe Phase: bimodal
        early_data = [1] * 50 + [100] * 50
        
        # Späte Phase: unimodal (konzentriert)
        late_data = [100] * 90 + [101] * 10
        
        early_mode1, early_mode2, _, _ = BimodalDistributionAnalysis.estimate_modes(early_data)
        late_mode1, late_mode2, _, _ = BimodalDistributionAnalysis.estimate_modes(late_data)
        
        # Frühe Phase sollte zwei deutliche Modi haben
        assert abs(early_mode1 - early_mode2) > 50
        
        # Späte Phase sollte Modi zusammen haben
        assert abs(late_mode1 - late_mode2) < 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
