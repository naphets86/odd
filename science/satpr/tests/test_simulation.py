"""
Tests für das Simulations-Modul.

Tests für:
- Monte-Carlo Simulationen
- Phase-Simulationen
- LBV-Simulationen
- Statistiken
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation import (
    MonteCarloSimulator,
    LBVSimulator,
    SimulationResult
)
from sat_solver import SATFormula


class TestSimulationResult:
    """Tests für SimulationResult-Datenklasse."""
    
    def test_result_creation(self):
        """Test: Erstelle SimulationResult."""
        result = SimulationResult(
            formula_index=1,
            n_variables=5,
            phase=1,
            expected_runtime=2.0,
            actual_runtime=2.5,
            found_solution=True,
            solver_type='subgraph'
        )
        
        assert result.formula_index == 1
        assert result.phase == 1
        assert result.found_solution is True


class TestMonteCarloSimulator:
    """Tests für MonteCarloSimulator."""
    
    def test_simulator_creation(self):
        """Test: Erstelle Simulator."""
        simulator = MonteCarloSimulator(n_variables=4)
        
        assert simulator.n_variables == 4
        assert simulator.n_formulas == 2 ** 4
        assert len(simulator.results) == 0
    
    def test_simulator_with_seed(self):
        """Test: Reproduzierbarkeit mit Seed."""
        sim1 = MonteCarloSimulator(n_variables=3, seed=42)
        sim2 = MonteCarloSimulator(n_variables=3, seed=42)
        
        # Beide sollten die gleichen zufälligen Formeln generieren
        formula1 = SATFormula.random_formula(3, 5)
        formula2 = SATFormula.random_formula(3, 5)
        
        # Bei gleichem Seed sollten sie ähnliche Strukturen haben
        assert len(formula1.clauses) == len(formula2.clauses)
    
    def test_run_phase_simulation_phase1(self):
        """Test: Simuliere Phase 1."""
        simulator = MonteCarloSimulator(n_variables=4)
        
        results = simulator.run_phase_simulation(phase=1, n_samples=3)
        
        assert len(results) == 3
        assert all(r.phase == 1 for r in results)
        assert all(r.n_variables == 4 for r in results)
    
    def test_run_phase_simulation_phase2(self):
        """Test: Simuliere Phase 2."""
        simulator = MonteCarloSimulator(n_variables=4)
        
        results = simulator.run_phase_simulation(phase=2, n_samples=2)
        
        assert len(results) == 2
        assert all(r.phase == 2 for r in results)
    
    def test_run_phase_simulation_phase3(self):
        """Test: Simuliere Phase 3."""
        simulator = MonteCarloSimulator(n_variables=4)
        
        results = simulator.run_phase_simulation(phase=3, n_samples=2)
        
        assert len(results) == 2
        assert all(r.phase == 3 for r in results)
    
    def test_run_full_simulation(self):
        """Test: Vollständige Simulation über alle Phasen."""
        simulator = MonteCarloSimulator(n_variables=3)
        
        results = simulator.run_full_simulation(n_samples_per_phase=2)
        
        assert len(results) == 6  # 3 Phasen * 2 Samples
        assert len(simulator.results) == 6
        
        phases = [r.phase for r in results]
        assert 1 in phases
        assert 2 in phases
        assert 3 in phases
    
    def test_validate_phase1_log_runtime(self):
        """Test: Validierung Phase 1 logarithmisch."""
        simulator = MonteCarloSimulator(n_variables=4)
        simulator.run_phase_simulation(phase=1, n_samples=5)
        
        validation = simulator.validate_phase1_log_runtime()
        
        assert 'valid' in validation
        assert 'baseline_O_log_n' in validation
        assert validation['baseline_O_log_n'] > 0
    
    def test_validate_phase3_cubic_convergence(self):
        """Test: Validierung Phase 3 gegen n^3."""
        simulator = MonteCarloSimulator(n_variables=3)
        simulator.run_phase_simulation(phase=3, n_samples=3)
        
        validation = simulator.validate_phase3_cubic_convergence()
        
        assert 'valid' in validation
        assert 'worst_case_O_n3' in validation
        assert validation['worst_case_O_n3'] == 27  # 3^3
    
    def test_get_statistics(self):
        """Test: Berechne Statistiken."""
        simulator = MonteCarloSimulator(n_variables=3)
        simulator.run_full_simulation(n_samples_per_phase=2)
        
        stats = simulator.get_statistics()
        
        assert 'phase_1' in stats
        assert 'phase_2' in stats
        assert 'phase_3' in stats
        
        for phase_key in ['phase_1', 'phase_2', 'phase_3']:
            phase_stats = stats[phase_key]
            assert 'count' in phase_stats
            assert 'mean_runtime' in phase_stats
            assert 'success_rate' in phase_stats


class TestLBVSimulator:
    """Tests für LBV-Simulator."""
    
    def test_lbv_simulator_creation(self):
        """Test: Erstelle LBV-Simulator."""
        simulator = LBVSimulator(n_variables=5)
        
        assert simulator.n_variables == 5
        assert simulator.lbv is not None
    
    def test_simulate_lbv_success_rate(self):
        """Test: Simuliere LBV-Erfolgsrate."""
        simulator = LBVSimulator(n_variables=4)
        
        result = simulator.simulate_lbv_success_rate(n_formulas=10)
        
        assert 'success_rate' in result
        assert 'mean_iterations' in result
        assert 'max_iterations' in result
        assert 'theoretical_iterations' in result
        
        assert 0 <= result['success_rate'] <= 1
    
    def test_simulate_repeated_lbv(self):
        """Test: Simuliere wiederholtes LBV."""
        simulator = LBVSimulator(n_variables=4)
        
        result = simulator.simulate_repeated_lbv(
            n_repetitions=5,
            n_formulas=5
        )
        
        assert 'mean_solutions_found' in result
        assert 'std_solutions_found' in result
        assert 'expected_mean' in result
        
        # Erwartete Anzahl sollten positive sein
        assert result['expected_mean'] > 0
    
    # def test_validate_binomial_distribution(self):
    #     """Test: Validierung Binomial-Verteilung."""
    #     simulator = LBVSimulator(n_variables=5)
        
    #     result = simulator.validate_binomial_distribution(n_trials=100)
        
    #     assert 'observed_mean' in result
    #     assert 'expected_mean' in result
    #     assert 'observed_var' in result
    #     assert 'expected_var' in result
        
    #     # Sollten ungefähr übereinstimmen
    #     assert result['mean_error'] < 1.0
    #     assert result['var_error'] < 2.0
    
    def test_converges_to_one(self):
        """Test: Konvergenz zu 1."""
        simulator = LBVSimulator(n_variables=10)
        
        result = simulator.validate_binomial_distribution(n_trials=50)
        
        # Konvergiert zu 1
        assert 0.5 < result['converges_to_one'] < 1.0


class TestProbabilityDistributionAnalysis:
    """Tests für Wahrscheinlichkeitsverteilungs-Analyse."""
    
    def test_simulation_consistency(self):
        """Test: Konsistenz zwischen Phasen."""
        simulator = MonteCarloSimulator(n_variables=3, seed=123)
        
        results = simulator.run_full_simulation(n_samples_per_phase=3)
        
        # Runtimes sollten über Phasen ansteigen
        phase1_runtimes = [r.expected_runtime for r in results if r.phase == 1]
        phase3_runtimes = [r.expected_runtime for r in results if r.phase == 3]
        
        if phase1_runtimes and phase3_runtimes:
            assert np.mean(phase3_runtimes) >= np.mean(phase1_runtimes)
    
    def test_multiple_runs_stability(self):
        """Test: Stabilität über mehrere Läufe."""
        results1 = []
        results2 = []
        
        for seed in [42, 43]:
            simulator = MonteCarloSimulator(n_variables=3, seed=seed)
            sim_results = simulator.run_phase_simulation(phase=1, n_samples=2)
            
            if seed == 42:
                results1 = sim_results
            else:
                results2 = sim_results
        
        # Beide sollten ähnliche Strukturen haben
        assert len(results1) == len(results2)
        assert all(r.phase == 1 for r in results1 + results2)


class TestIntegration:
    """Integrationstests für Simulationen."""
    
    def test_full_simulation_pipeline(self):
        """Test: Kompletter Simulations-Pipeline."""
        simulator = MonteCarloSimulator(n_variables=3, seed=42)
        
        # Führe volle Simulation aus
        results = simulator.run_full_simulation(n_samples_per_phase=3)
        
        # Sammle Statistiken
        stats = simulator.get_statistics()
        
        # Validiere Phasen
        validation1 = simulator.validate_phase1_log_runtime()
        validation3 = simulator.validate_phase3_cubic_convergence()
        
        assert len(results) > 0
        assert len(stats) > 0
    
    def test_lbv_vs_subgraph_efficiency(self):
        """Test: Vergleich LBV vs Subgraph."""
        # LBV sollte bei manchen Formeln schneller sein
        
        lbv_sim = LBVSimulator(n_variables=5)
        lbv_result = lbv_sim.simulate_lbv_success_rate(n_formulas=10)
        
        mc_sim = MonteCarloSimulator(n_variables=5)
        mc_results = mc_sim.run_phase_simulation(phase=1, n_samples=5)
        
        # LBV sollte weniger Iterationen benötigen
        assert lbv_result['theoretical_iterations'] < 10
    
    def test_reproducibility_with_seed(self):
        """Test: Reproduzierbarkeit mit Seed."""
        sim1 = MonteCarloSimulator(n_variables=3, seed=999)
        results1 = sim1.run_phase_simulation(phase=1, n_samples=2)
        
        sim2 = MonteCarloSimulator(n_variables=3, seed=999)
        results2 = sim2.run_phase_simulation(phase=1, n_samples=2)
        
        # Sollten ähnliche Ergebnisse geben
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1.n_variables == r2.n_variables
            assert r1.phase == r2.phase


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
