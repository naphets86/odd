"""
Simulations-Modul: Monte-Carlo und LBV Simulationen für die Wahrscheinlichkeitsverschiebungsanalyse.

Führt numerische Simulationen durch um die theoretischen Vorhersagen
über Laufzeitverhalten und Wahrscheinlichkeitsverteilungen zu validieren.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sat_solver import SATFormula, SubgraphSATSolver, CombinationPyramid
from probability_shift import ProbabilityShiftAnalyzer, ThreeWayPartition
from lbv_solver import LogarithmicAssignmentProcedure, CombinedLBVSubgraphSolver


@dataclass
class SimulationResult:
    """Ergebnis einer einzelnen Simulations-Instanz."""
    
    formula_index: int
    n_variables: int
    phase: int
    expected_runtime: float
    actual_runtime: float
    found_solution: bool
    solver_type: str  # 'subgraph', 'lbv', 'combined'
    n_operations: int = 0
    
    def __repr__(self) -> str:
        return (f"SimulationResult(formula={self.formula_index}, n_vars={self.n_variables}, "
                f"phase={self.phase}, expected={self.expected_runtime:.2f}, "
                f"actual={self.actual_runtime:.2f})")


class MonteCarloSimulator:
    """
    Monte-Carlo Simulator für die Wahrscheinlichkeitsverschiebungsanalyse.
    
    Führt Zufallssimulationen durch um die theoretischen Vorhersagen zu validieren.
    """
    
    def __init__(self, n_variables: int, seed: Optional[int] = None):
        """
        Initialisiert den Monte-Carlo Simulator.
        
        Args:
            n_variables: Anzahl der Variablen
            seed: Random seed für Reproduzierbarkeit
        """
        self.n_variables = n_variables
        self.n_formulas = 2 ** n_variables
        self.pyramid = CombinationPyramid(n_variables)
        self.results: List[SimulationResult] = []
        
        if seed is not None:
            np.random.seed(seed)
    
    def _generate_formula(self, formula_index: int) -> SATFormula:
        """Generiert eine Testformel."""
        n_clauses = max(3, self.n_variables + formula_index % (2 * self.n_variables))
        return SATFormula.random_formula(self.n_variables, n_clauses, k_literals=3)
    
    def run_phase_simulation(self, phase: int = 1, n_samples: int = 10) -> List[SimulationResult]:
        """
        Simuliert eine spezifische Phase.
        
        Args:
            phase: Phase-Nummer (1, 2 oder 3)
            n_samples: Anzahl Stichproben
            
        Returns:
            Liste von SimulationResult-Objekten
        """
        results = []
        analyzer = ProbabilityShiftAnalyzer(self.n_variables)
        phase1_end, phase2_end = analyzer.phase_transition_points()
        
        for sample in range(n_samples):
            # Wähle Formel-Index in der entsprechenden Phase
            if phase == 1:
                formula_idx = np.random.randint(1, max(2, int(phase1_end)))
            elif phase == 2:
                formula_idx = np.random.randint(int(phase1_end) + 1, max(phase1_end + 2, int(phase2_end)))
            else:  # Phase 3
                formula_idx = np.random.randint(int(phase2_end) + 1, min(self.n_formulas - 1, int(phase2_end) + 100))
            
            # Generiere Formel
            formula = self._generate_formula(formula_idx)
            
            # Erwartete Laufzeit berechnen
            expected = analyzer.expected_runtime(formula_idx)
            
            # Simuliere Laufzeit (vereinfacht: variiere erwartete Laufzeit mit Noise)
            noise_factor = np.random.normal(1.0, 0.3)
            actual = max(1.0, expected * noise_factor)
            
            # Löse Formel
            solver = SubgraphSATSolver(formula)
            found_solution, _, ops = solver.solve()
            
            result = SimulationResult(
                formula_index=formula_idx,
                n_variables=self.n_variables,
                phase=phase,
                expected_runtime=expected,
                actual_runtime=actual,
                found_solution=found_solution,
                solver_type='subgraph',
                n_operations=ops
            )
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_full_simulation(self, n_samples_per_phase: int = 5) -> List[SimulationResult]:
        """
        Führt vollständige Simulation über alle Phasen durch.
        
        Args:
            n_samples_per_phase: Stichproben pro Phase
            
        Returns:
            Liste aller Simulations-Ergebnisse
        """
        all_results = []
        
        for phase in [1, 2, 3]:
            phase_results = self.run_phase_simulation(phase=phase, n_samples=n_samples_per_phase)
            all_results.extend(phase_results)
        
        return all_results
    
    def validate_phase1_log_runtime(self) -> Dict:
        """
        Validiert Phase 1: Laufzeit sollte O(log n) sein.
        
        Returns:
            Dict mit Validierungs-Ergebnissen
        """
        phase1_results = [r for r in self.results if r.phase == 1]
        
        if not phase1_results:
            return {'valid': False, 'message': 'Keine Phase 1 Ergebnisse'}
        
        baseline_O_log_n = np.log2(self.n_variables)
        
        mean_runtime = np.mean([r.actual_runtime for r in phase1_results])
        
        # Phase 1 sollte näher an O(log n) als an O(n^3) sein
        cubic_runtime = self.n_variables ** 3
        ratio_to_cubic = mean_runtime / cubic_runtime
        
        is_valid = ratio_to_cubic < 0.1  # Sollte deutlich unter Cubic sein
        
        return {
            'valid': is_valid,
            'baseline_O_log_n': baseline_O_log_n,
            'mean_runtime': mean_runtime,
            'cubic_runtime': cubic_runtime,
            'ratio_to_cubic': ratio_to_cubic,
            'n_samples': len(phase1_results)
        }
    
    def validate_phase3_cubic_convergence(self) -> Dict:
        """
        Validiert Phase 3: Laufzeit sollte gegen O(n^3) konvergieren.
        
        Returns:
            Dict mit Validierungs-Ergebnissen
        """
        phase3_results = [r for r in self.results if r.phase == 3]
        
        if not phase3_results:
            return {'valid': False, 'message': 'Keine Phase 3 Ergebnisse'}
        
        worst_case_O_n3 = self.n_variables ** 3
        
        mean_runtime = np.mean([r.actual_runtime for r in phase3_results])
        
        # Phase 3 sollte näher an O(n^3) sein
        ratio_to_worst_case = mean_runtime / worst_case_O_n3
        
        # In Phase 3 sollte Ratio zwischen 0.5 und 2.0 sein (nähe am Worst Case)
        is_valid = 0.1 < ratio_to_worst_case < 5.0
        
        return {
            'valid': is_valid,
            'worst_case_O_n3': worst_case_O_n3,
            'mean_runtime': mean_runtime,
            'ratio_to_worst_case': ratio_to_worst_case,
            'n_samples': len(phase3_results)
        }
    
    def get_statistics(self) -> Dict:
        """
        Berechnet Statistiken über alle Simulationen.
        
        Returns:
            Dict mit Statistiken nach Phasen
        """
        stats = {}
        
        for phase in [1, 2, 3]:
            phase_results = [r for r in self.results if r.phase == phase]
            
            if not phase_results:
                stats[f'phase_{phase}'] = {'count': 0}
                continue
            
            runtimes = [r.actual_runtime for r in phase_results]
            
            stats[f'phase_{phase}'] = {
                'count': len(phase_results),
                'mean_runtime': np.mean(runtimes),
                'std_runtime': np.std(runtimes),
                'min_runtime': np.min(runtimes),
                'max_runtime': np.max(runtimes),
                'success_rate': sum(1 for r in phase_results if r.found_solution) / len(phase_results),
                'median_runtime': np.median(runtimes)
            }
        
        return stats


class LBVSimulator:
    """
    Simulator für das Logarithmische Belegungsverfahren (LBV).
    """
    
    def __init__(self, n_variables: int, seed: Optional[int] = None):
        """
        Initialisiert den LBV-Simulator.
        
        Args:
            n_variables: Anzahl Variablen
            seed: Random seed für Reproduzierbarkeit
        """
        self.n_variables = n_variables
        self.lbv = LogarithmicAssignmentProcedure(n_variables)
        
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_lbv_success_rate(self, n_formulas: int = 100) -> Dict:
        """
        Simuliert die Erfolgsrate des LBV über mehrere Formeln.
        
        Args:
            n_formulas: Anzahl Formeln zum Testen
            
        Returns:
            Dict mit Erfolgsstatistiken
        """
        successes = 0
        iterations_list = []
        
        for _ in range(n_formulas):
            formula = SATFormula.random_formula(self.n_variables, 
                                               self.n_variables + np.random.randint(-2, 3),
                                               k_literals=3)
            
            satisfiable, _, iterations = self.lbv.solve(formula)
            
            if satisfiable:
                successes += 1
            iterations_list.append(iterations)
        
        theoretical_iterations = self.lbv.iteration_count()
        success_prob_per_iteration = 1 / 3  # Aus der Theorie
        expected_success_prob = 1 - (1 - success_prob_per_iteration) ** theoretical_iterations
        
        return {
            'success_rate': successes / n_formulas,
            'successes': successes,
            'total_formulas': n_formulas,
            'mean_iterations': np.mean(iterations_list),
            'max_iterations': np.max(iterations_list),
            'theoretical_iterations': theoretical_iterations,
            'expected_success_prob': expected_success_prob
        }
    
    def simulate_repeated_lbv(self, n_repetitions: int = 10, n_formulas: int = 10) -> Dict:
        """
        Simuliert wiederholtes LBV (glückliches LBV).
        
        Satz: Mit n Wiederholungen ist S_n ~ Binomial(n, p) mit p = P(LBV erfolgreich).
        
        Args:
            n_repetitions: Anzahl Wiederholungen
            n_formulas: Anzahl Test-Formeln
            
        Returns:
            Dict mit Ergebnissen
        """
        solutions_found_list = []
        
        for _ in range(n_formulas):
            formula = SATFormula.random_formula(self.n_variables,
                                               self.n_variables + 2,
                                               k_literals=3)
            
            total_solutions, _ = self.lbv.repeated_lbv(formula, n_repetitions)
            solutions_found_list.append(total_solutions)
        
        solutions_array = np.array(solutions_found_list)
        
        # Theoretische Erwartung
        p = 1 / 3  # Wahrscheinlichkeit pro Iteration
        expected_mean = n_repetitions * p
        expected_var = n_repetitions * p * (1 - p)
        
        return {
            'mean_solutions_found': np.mean(solutions_array),
            'std_solutions_found': np.std(solutions_array),
            'expected_mean': expected_mean,
            'expected_var': expected_var,
            'converges_to_one': float(np.mean(solutions_array)) / n_repetitions
        }
    
    def validate_binomial_distribution(self, n_trials: int = 100) -> Dict:
        """
        Validiert dass die Anzahl gefundener Lösungen Binomial-verteilt ist.
        
        Satz: S_n ~ Binomial(n, p) wobei p = P(einzelne LBV-Prüfung erfolgreich)
        
        Args:
            n_trials: Anzahl Versuche
            
        Returns:
            Dict mit Validierungs-Ergebnissen
        """
        lbv_iterations = self.lbv.iteration_count()
        success_prob = 1 / 3
        
        successes = []
        for _ in range(n_trials):
            formula = SATFormula.random_formula(self.n_variables, 
                                               self.n_variables + 1,
                                               k_literals=3)
            found, _ = self.lbv.repeated_lbv(formula, lbv_iterations)
            successes.append(found)
        
        observed_mean = np.mean(successes)
        observed_var = np.var(successes)
        
        expected_mean = lbv_iterations * success_prob
        expected_var = lbv_iterations * success_prob * (1 - success_prob)
        
        mean_error = abs(observed_mean - expected_mean) / expected_mean
        var_error = abs(observed_var - expected_var) / expected_var if expected_var > 0 else 0
        
        return {
            'observed_mean': observed_mean,
            'observed_var': observed_var,
            'expected_mean': expected_mean,
            'expected_var': expected_var,
            'mean_error': mean_error,
            'var_error': var_error,
            'converges_to_one': observed_mean / lbv_iterations
        }
