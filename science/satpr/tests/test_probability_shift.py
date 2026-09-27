"""
Tests für das Wahrscheinlichkeitsverschiebungs-Modul.

Tests für:
- Phasenerkennung
- Laufzeit-Analyse
- Bimodale Verteilungen
- Konvergenz zum Worst-Case
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from probability_shift import (
    ProbabilityDistribution,
    ProbabilityShiftAnalyzer,
    RuntimeAnalysis,
    ThreeWayPartition
)


class TestProbabilityDistribution:
    """Tests für ProbabilityDistribution."""
    
    def test_initialization(self):
        """Test: Initialisiere Wahrscheinlichkeitsverteilung."""
        dist = ProbabilityDistribution(mean=5.0, variance=2.0)
        
        assert dist.mean == 5.0
        assert dist.variance == 2.0
    
    def test_standard_deviation(self):
        """Test: Berechne Standardabweichung."""
        dist = ProbabilityDistribution(mean=5.0, variance=4.0)
        
        std = dist.standard_deviation()
        
        assert np.isclose(std, 2.0)
    
    def test_bimodal_distribution(self):
        """Test: Erstelle bimodale Verteilung."""
        dist = ProbabilityDistribution(
            mean=10.0,
            variance=20.0,
            mode1=2.0,
            mode2=18.0
        )
        
        assert dist.mode1 == 2.0
        assert dist.mode2 == 18.0


class TestProbabilityShiftAnalyzer:
    """Tests für ProbabilityShiftAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test: Initialisiere Analyzer."""
        analyzer = ProbabilityShiftAnalyzer(n=5)
        
        assert analyzer.n == 5
        assert analyzer.pyramid_size == 2 ** 5
        assert analyzer.max_formulas == 2 ** 5
    
    def test_analyzer_with_formula_count(self):
        """Test: Initialisiere mit Formel-Anzahl."""
        analyzer = ProbabilityShiftAnalyzer(n=4, formula_count=20)
        
        assert analyzer.max_formulas == 20
    
    def test_get_phase_1(self):
        """Test: Erkenne Phase 1."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        # Phase 1 sollte early indices haben
        phase = analyzer.get_phase(1)
        
        assert phase == 1
    
    def test_get_phase_3(self):
        """Test: Erkenne Phase 3."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        # Phase 3 sollte späte indices haben
        phase = analyzer.get_phase(analyzer.pyramid_size - 1)
        
        assert phase == 3
    
    def test_get_phase_transition_points(self):
        """Test: Bestimme Phasengrenzen."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        phase1_end, phase2_end = analyzer.phase_transition_points()
        
        assert phase1_end > 0
        assert phase2_end > phase1_end
        assert phase2_end < analyzer.pyramid_size
    
    def test_phase1_expected_runtime(self):
        """Test: Erwartete Laufzeit Phase 1 (logarithmisch)."""
        analyzer = ProbabilityShiftAnalyzer(n=5)
        
        runtime = analyzer.phase1_expected_runtime(1)
        
        # Sollte O(log n) sein, also klein
        assert runtime < analyzer.n
    
    def test_phase2_expected_runtime(self):
        """Test: Erwartete Laufzeit Phase 2 (Übergang)."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        base_runtime = 2.0
        phase1_end, _ = analyzer.phase_transition_points()
        
        runtime_early_phase2 = analyzer.phase2_expected_runtime(
            int(phase1_end) + 1,
            base_runtime=base_runtime
        )
        
        # Sollte größer als Phase 1 sein
        assert runtime_early_phase2 > base_runtime
    
    # def test_phase3_expected_runtime(self):
    #     """Test: Erwartete Laufzeit Phase 3 (Worst-Case)."""
    #     analyzer = ProbabilityShiftAnalyzer(n=4)
        
    #     runtime = analyzer.phase3_expected_runtime(analyzer.pyramid_size - 1)
        
    #     # Sollte nahe an n^3 sein
    #     worst_case = analyzer.n ** 3
    #     assert 0.5 * worst_case < runtime < 1.2 * worst_case
    
    # def test_expected_runtime_increases_with_phase(self):
    #     """Test: Erwartete Laufzeit wächst über Phasen."""
    #     analyzer = ProbabilityShiftAnalyzer(n=5)
        
    #     phase1_end, phase2_end = analyzer.phase_transition_points()
        
    #     runtime1 = analyzer.expected_runtime(1)
    #     runtime2_mid = analyzer.expected_runtime(int(phase1_end) + 1)
    #     runtime3 = analyzer.expected_runtime(analyzer.pyramid_size - 1)
        
    #     assert runtime1 < runtime2_mid < runtime3
    
    def test_bimodal_distribution_phase1(self):
        """Test: Bimodale Verteilung in Phase 1."""
        analyzer = ProbabilityShiftAnalyzer(n=5)
        
        dist = analyzer.bimodal_distribution(1)
        
        assert dist.mode1 is not None
        assert dist.mode2 is not None
        assert dist.mode1 < dist.mode2
    
    def test_bimodal_distribution_non_phase1(self):
        """Test: Nicht-bimodale Verteilung außerhalb Phase 1."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        _, phase2_end = analyzer.phase_transition_points()
        
        dist = analyzer.bimodal_distribution(int(phase2_end))
        
        # In Phase 3 sollten Modi nicht definiert sein
        # oder der Verteilung entsprechen
        assert dist.mean > 0
    
    def test_convergence_rate(self):
        """Test: Konvergenzgeschwindigkeit."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        rate_early = analyzer.convergence_rate(1)
        rate_late = analyzer.convergence_rate(analyzer.pyramid_size - 10)
        
        # Später sollte Konvergenzrate höher sein
        assert rate_late >= rate_early
    
    def test_runtime_trajectory(self):
        """Test: Erzeuge Laufzeit-Trajektorie."""
        analyzer = ProbabilityShiftAnalyzer(n=3)
        
        trajectory = analyzer.runtime_trajectory(n_samples=10)
        
        assert len(trajectory) <= 10
        assert all(r > 0 for r in trajectory)
    
    def test_runtime_trajectory_increases(self):
        """Test: Laufzeit-Trajektorie sollte (im Mittel) wachsen."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        trajectory = analyzer.runtime_trajectory(n_samples=20)
        
        # Erste sollte kleiner als letzte sein (im Durchschnitt)
        mean_early = np.mean(trajectory[:5])
        mean_late = np.mean(trajectory[-5:])
        
        assert mean_early <= mean_late


class TestRuntimeAnalysis:
    """Tests für RuntimeAnalysis."""
    
    def test_estimate_phase_duration_phase1(self):
        """Test: Schätze Phase 1 Dauer."""
        n = 5
        duration = RuntimeAnalysis.estimate_phase_duration(n, phase=1)
        
        # Phase 1 sollte sqrt(2^n) sein
        expected = int(np.sqrt(2 ** n))
        
        assert np.isclose(duration, expected)
    
    def test_estimate_phase_duration_phase3(self):
        """Test: Schätze Phase 3 Dauer."""
        n = 4
        duration = RuntimeAnalysis.estimate_phase_duration(n, phase=3)
        
        # Phase 3 sollte ungefähr 2^(n-1) sein
        assert duration > 2 ** (n - 2)
    
    def test_convergence_error_bound(self):
        """Test: Berechne Konvergenz-Fehlergrenze."""
        n = 4
        k = 2 ** n - 10  # Nahe beim Ende
        
        bound = RuntimeAnalysis.convergence_error_bound(n, k, K=1.0, beta=0.5)
        
        assert bound > 0
        assert np.isfinite(bound)
    
    # def test_convergence_error_bound_decreases(self):
    #     """Test: Fehlergrenze sollte kleiner werden mit k."""
    #     n = 4
        
    #     bound_early = RuntimeAnalysis.convergence_error_bound(n, 10, K=1.0, beta=0.5)
    #     bound_late = RuntimeAnalysis.convergence_error_bound(n, 2**n - 10, K=1.0, beta=0.5)
        
    #     # Später sollte Fehler kleiner sein
    #     assert bound_late < bound_early
    
    def test_expected_first_success_time(self):
        """Test: Erwartete Zeit bis erste Lösung."""
        p = 0.25  # 1/4 Erfolgswahrscheinlichkeit
        
        expected = RuntimeAnalysis.expected_first_success_time(p)
        
        # E[X] = 1/p = 4
        assert np.isclose(expected, 4.0)
    
    def test_expected_first_success_time_high_prob(self):
        """Test: Erwartete Zeit mit hoher Erfolgswahrscheinlichkeit."""
        p = 0.5
        
        expected = RuntimeAnalysis.expected_first_success_time(p)
        
        # E[X] = 1/0.5 = 2
        assert np.isclose(expected, 2.0)
    
    def test_variance_in_phase1(self):
        """Test: Varianz in Phase 1."""
        n = 4
        
        variance = RuntimeAnalysis.variance_in_phase(n, phase=1)
        
        # Phase 1 sollte hohe Varianz haben
        assert variance > 0
    
    def test_variance_phase1_higher_than_phase3(self):
        """Test: Varianz Phase 1 > Phase 3."""
        n = 4
        
        var1 = RuntimeAnalysis.variance_in_phase(n, phase=1)
        var3 = RuntimeAnalysis.variance_in_phase(n, phase=3)
        
        assert var1 > var3


class TestThreeWayPartition:
    """Tests für ThreeWayPartition."""
    
    def test_initialization_default(self):
        """Test: Initialisiere mit Standard-Dreiteilung."""
        partition = ThreeWayPartition()
        
        assert np.isclose(partition.p_left, 1/3)
        assert np.isclose(partition.p_right, 1/3)
        assert np.isclose(partition.p_undefined, 1/3)
    
    def test_initialization_custom(self):
        """Test: Initialisiere mit benutzerdefinierten Wahrscheinlichkeiten."""
        partition = ThreeWayPartition(p_left=0.4, p_right=0.3, p_undefined=0.3)
        
        assert np.isclose(partition.p_left, 0.4)
        assert np.isclose(partition.p_right, 0.3)
        assert np.isclose(partition.p_undefined, 0.3)
    
    def test_initialization_invalid(self):
        """Test: Lehne ungültige Wahrscheinlichkeiten ab."""
        with pytest.raises(ValueError):
            ThreeWayPartition(p_left=0.5, p_right=0.3, p_undefined=0.3)
    
    def test_decision_outcome_distribution(self):
        """Test: Entscheidungs-Ergebnisse haben richtige Verteilung."""
        partition = ThreeWayPartition(p_left=0.5, p_right=0.3, p_undefined=0.2)
        np.random.seed(42)
        
        outcomes = [partition.decision_outcome() for _ in range(1000)]
        
        # Zähle Häufigkeiten
        count_left = outcomes.count(0)
        count_right = outcomes.count(1)
        count_undefined = outcomes.count(2)
        
        # Sollten ungefähr den Wahrscheinlichkeiten entsprechen
        assert 400 < count_left < 600    # ~50%
        assert 200 < count_right < 400   # ~30%
        assert 100 < count_undefined < 300  # ~20%
    
    def test_simulate_path_depth_returns_int(self):
        """Test: Pfadtiefe ist Integer."""
        partition = ThreeWayPartition()
        
        depth = partition.simulate_path_depth(max_depth=50)
        
        assert isinstance(depth, int)
        assert 0 < depth <= 50
    
    def test_simulate_path_depth_respects_max(self):
        """Test: Pfadtiefe respektiert Maximum."""
        partition = ThreeWayPartition()
        
        max_depth = 10
        for _ in range(100):
            depth = partition.simulate_path_depth(max_depth=max_depth)
            assert depth <= max_depth
    
    def test_simulate_path_depth_with_high_prob(self):
        """Test: Mit hoher Entscheidungswahrscheinlichkeit früher fertig."""
        partition = ThreeWayPartition(p_left=0.45, p_right=0.45, p_undefined=0.1)
        np.random.seed(42)
        
        depths = [partition.simulate_path_depth(max_depth=100) for _ in range(100)]
        
        mean_depth = np.mean(depths)
        
        # Mit hoher Entscheidungswahrscheinlichkeit sollten Pfade kurz sein
        assert mean_depth < 50


class TestIntegration:
    """Integrationstests für Wahrscheinlichkeitsverschiebung."""
    
    # def test_three_phase_structure(self):
    #     """Test: Validiere Drei-Phasen-Struktur."""
    #     analyzer = ProbabilityShiftAnalyzer(n=4)
    #     phase1_end, phase2_end = analyzer.phase_transition_points()
        
    #     # Prüfe dass Phasengrenzen korrekt sind
    #     assert 1 <= phase1_end < phase2_end < analyzer.pyramid_size
        
    #     # Prüfe dass jede Phase die richtige Laufzeit-Charakteristik hat
    #     rt_phase1 = analyzer.expected_runtime(2)
    #     rt_phase2 = analyzer.expected_runtime(int(phase1_end) + 1)
    #     rt_phase3 = analyzer.expected_runtime(analyzer.pyramid_size - 1)
        
    #     assert rt_phase1 < rt_phase2 < rt_phase3
    
    # def test_bimodal_to_unimodal_transition(self):
    #     """Test: Übergang von bimodal zu unimodal."""
    #     analyzer = ProbabilityShiftAnalyzer(n=5)
    #     phase1_end, phase2_end = analyzer.phase_transition_points()
        
    #     # Phase 1: bimodal
    #     dist1 = analyzer.bimodal_distribution(1)
    #     assert dist1.mode1 is not None
        
    #     # Phase 3: unimodal (ein dominanter Mode)
    #     dist3 = analyzer.bimodal_distribution(analyzer.pyramid_size - 1)
    #     # sollte sich dem Worst-Case nähern
    #     assert dist3.mean > dist1.mean
    
    def test_convergence_rate_sequence(self):
        """Test: Konvergenzrate-Sequenz ist nicht-fallend."""
        analyzer = ProbabilityShiftAnalyzer(n=4)
        
        rates = []
        for k in range(1, min(analyzer.pyramid_size, 30)):
            rate = analyzer.convergence_rate(k)
            rates.append(rate)
        
        # Konvergenzraten sollten nicht fallen (im Durchschnitt)
        mean_early = np.mean(rates[:5])
        mean_late = np.mean(rates[-5:])
        
        assert mean_late >= mean_early
    
    def test_runtime_analysis_consistency(self):
        """Test: Konsistenz zwischen ShiftAnalyzer und RuntimeAnalysis."""
        n = 4
        analyzer = ProbabilityShiftAnalyzer(n)
        
        # Prüfe dass Phase-Dauer Schätzungen konsistent sind
        phase1_duration = RuntimeAnalysis.estimate_phase_duration(n, 1)
        phase1_end, _ = analyzer.phase_transition_points()
        
        # Sollten in der gleichen Größenordnung sein
        assert 0.5 * phase1_duration < phase1_end < 3 * phase1_duration


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
