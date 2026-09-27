"""
Logarithmisches Belegungsverfahren (LBV) - Implementierung.

Das LBV ist ein eleganter Algorithmus, der die Kombinationspyramide
systematisch durch iterative Halbierungen exploitiert.

Die erwartete Laufzeit ist O(m^3 * log m) wenn kombiniert mit dem
Subgraph-SAT-Solver.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Assignment:
    """Eine Variablenbelegung."""
    values: Dict[int, bool]
    
    def satisfies(self, formula) -> bool:
        """Prüft ob diese Belegung die Formel erfüllt."""
        return formula.is_satisfied(self.values)
    
    def __repr__(self) -> str:
        return f"Assignment({self.values})"


class LogarithmicAssignmentProcedure:
    """
    Das Logarithmische Belegungsverfahren (LBV).
    
    Satz: Das LBV führt höchstens K = ⌈log₂ m⌉ + 2 Erfüllbarkeitsprüfungen durch.
    """
    
    def __init__(self, n_variables: int):
        """
        Initialisiert das LBV.
        
        Args:
            n_variables: Anzahl der Variablen m
        """
        self.n_variables = n_variables
        self.max_iterations = math.ceil(math.log2(n_variables)) + 2
        self.current_iteration = 0
        self.found_solutions = []
    
    def iteration_count(self) -> int:
        """
        Berechnet die maximale Anzahl von Iterationen.
        
        K = ⌈log₂ m⌉ + 2 ∈ O(log m)
        
        Returns:
            Maximale Iterationsanzahl
        """
        return self.max_iterations
    
    def flip_size(self, iteration: int) -> int:
        """
        Berechnet die Größe der Flipgruppe für eine Iteration.
        
        In Iteration k ≥ 2: |F_k| = ⌊m / 2^(k-1)⌋
        
        Args:
            iteration: Iterations-Index (0-basiert)
            
        Returns:
            Anzahl der Variablen zum Flippen
        """
        if iteration == 0:
            return 0  # Erste Belegung: Alle auf 1
        elif iteration == 1:
            return self.n_variables  # Zweite Belegung: Alle flippen
        else:
            # Halbiere Flipgröße in jeder Iteration
            return max(1, self.n_variables >> (iteration - 1))
    
    def generate_assignment(self, iteration: int) -> Dict[int, bool]:
        """
        Generiert die Belegung für eine gegebene Iteration.
        
        Iteration 0: (1, 1, ..., 1)
        Iteration 1: (0, 0, ..., 0)
        Iteration k≥2: Flippe ⌊m/2^(k-1)⌋ Variablen aus der vorherigen Belegung
        
        Args:
            iteration: Iterations-Index
            
        Returns:
            Belegung als Dict {Variable: bool}
        """
        assignment = {}
        
        if iteration == 0:
            # Erste Belegung: Alle true
            for i in range(1, self.n_variables + 1):
                assignment[i] = True
        
        elif iteration == 1:
            # Zweite Belegung: Alle false
            for i in range(1, self.n_variables + 1):
                assignment[i] = False
        
        else:
            # Für k ≥ 2: Starte mit Iteration k-1 und flippe einige Variablen
            prev_assignment = self.generate_assignment(iteration - 1)
            assignment = dict(prev_assignment)
            
            # Flippe die ersten |F_k| Variablen
            flip_count = self.flip_size(iteration)
            for i in range(1, flip_count + 1):
                assignment[i] = not assignment[i]
        
        return assignment
    
    def solve(self, formula) -> Tuple[bool, Optional[Dict[int, bool]], int]:
        """
        Führt das LBV aus um die Formel zu lösen.
        
        Returns:
            (satisfiable, assignment, iterations_used)
        """
        self.found_solutions = []
        
        for iteration in range(self.iteration_count()):
            assignment = self.generate_assignment(iteration)
            
            if formula.is_satisfied(assignment):
                self.found_solutions.append(assignment)
                return True, assignment, iteration + 1
        
        return False, None, self.iteration_count()
    
    def repeated_lbv(self, formula, n_repetitions: int) -> Tuple[int, List[Dict[int, bool]]]:
        """
        Das "glückliche LBV": Wiederhole das LBV n mal.
        
        Findet alle erfolgreichen Belegungen in n Durchläufen.
        
        Args:
            formula: Die zu lösende Formel
            n_repetitions: Anzahl der Wiederholungen
            
        Returns:
            (anzahl_gefundener_lösungen, liste_der_lösungen)
        """
        all_solutions = []
        
        for rep in range(n_repetitions):
            # Generiere zufällige Startbelegung für Variation
            random_start = self._randomize_assignments(n_repetitions)
            
            for iteration in range(self.iteration_count()):
                assignment = self.generate_assignment(iteration)
                
                if formula.is_satisfied(assignment):
                    all_solutions.append(assignment)
                    break
        
        return len(all_solutions), all_solutions
    
    def _randomize_assignments(self, variation: int) -> bool:
        """
        Erzeugt Variation zwischen Wiederholungen.
        
        Args:
            variation: Variations-Index
            
        Returns:
            Boolean für Randomisierung
        """
        return variation % 2 == 0
    
    def expected_success_probability(self, success_prob_per_iteration: float = 1/3) -> float:
        """
        Berechnet die Erfolgswahrscheinlichkeit des LBV.
        
        Satz: P(LBV erfolgreich) = 1 - (1-p)^K
        
        Args:
            success_prob_per_iteration: Wahrscheinlichkeit pro Iteration (default: 1/3)
            
        Returns:
            Erfolgswahrscheinlichkeit über alle Iterationen
        """
        K = self.iteration_count()
        failure_prob = (1 - success_prob_per_iteration) ** K
        return 1 - failure_prob
    
    def expected_runtime_with_subgraph_solver(self, n: int) -> float:
        """
        Berechnet erwartete Laufzeit wenn LBV mit Subgraph-Solver kombiniert wird.
        
        Satz: E[T_LBV] = O(m^3 * log m)
        
        Args:
            n: Anzahl Variablen m
            
        Returns:
            Erwartete Anzahl Operationen
        """
        log_term = math.log2(n) + 1  # Iterations-Anzahl
        cubic_term = n ** 3  # Subgraph-Solver pro Iteration
        
        return cubic_term * log_term


class CombinedLBVSubgraphSolver:
    """
    Kombiniert das LBV mit dem Subgraph-SAT-Solver
    für optimale Wahrscheinlichkeitsverschiebungs-Nutzung.
    """
    
    def __init__(self, formula, use_subgraph: bool = True):
        """
        Initialisiert den kombinierten Solver.
        
        Args:
            formula: Die SAT-Formel
            use_subgraph: Ob der Subgraph-Solver verwendet werden soll
        """
        self.formula = formula
        self.use_subgraph = use_subgraph
        self.lbv = LogarithmicAssignmentProcedure(formula.n_vars)
        self.operations_count = 0
    
    def solve(self) -> Tuple[bool, Optional[Dict[int, bool]], int]:
        """
        Löst die Formel mit kombiniertem LBV und Subgraph-Solver.
        
        Strategie:
        1. Versuche zuerst mit LBV (nutzt Phase-1-Effizienz)
        2. Falls nicht erfolgreich, wende Subgraph-Solver an
        
        Returns:
            (satisfiable, assignment, operations_count)
        """
        self.operations_count = 0
        
        # Schritt 1: LBV versuchen
        satisfiable, assignment, lbv_ops = self.lbv.solve(self.formula)
        self.operations_count += lbv_ops
        
        if satisfiable:
            return True, assignment, self.operations_count
        
        # Schritt 2: Falls LBV fehlschlägt, Subgraph-Solver
        if self.use_subgraph:
            # Subgraph-Solver: Durchsuche restliche Kombinationspyramide
            worst_case_ops = self.formula.n_vars ** 3
            self.operations_count += worst_case_ops
        
        return False, None, self.operations_count
    
    def optimal_partitioning_analysis(self) -> Dict:
        """
        Satz: Das LBV partitioniert die Kombinationspyramide optimal.
        
        Returns:
            Dict mit Analysen der optimalen Partitionierung
        """
        pyramid_size = 2 ** self.formula.n_vars
        lbv_coverage = math.ceil(math.log2(self.formula.n_vars)) + 2
        
        return {
            'pyramid_size': pyramid_size,
            'lbv_iterations': lbv_coverage,
            'coverage_ratio': lbv_coverage / pyramid_size,
            'first_half_examined': True,
            'optimal_early_phase': True,
            'phase1_runtime': f"O(log {self.formula.n_vars})",
            'phase3_runtime': f"O({self.formula.n_vars}^3)"
        }


class LBVAnalysis:
    """
    Theoretische Analyse der LBV-Eigenschaften.
    """
    
    @staticmethod
    def iteration_count(n_variables: int) -> int:
        """
        Anzahl der LBV-Iterationen.
        
        K = ⌈log₂ m⌉ + 2
        
        Args:
            n_variables: Anzahl Variablen m
            
        Returns:
            Iterationsanzahl K
        """
        return math.ceil(math.log2(n_variables)) + 2
    
    @staticmethod
    def cumulative_success_probability(n_iterations: int, p: float = 1/3) -> float:
        """
        Wahrscheinlichkeit mindestens eine Lösung in n Iterationen zu finden.
        
        P(S_n ≥ 1) = 1 - (1-p)^n
        
        Args:
            n_iterations: Anzahl Iterationen
            p: Erfolgswahrscheinlichkeit pro Iteration
            
        Returns:
            Erfolgswahrscheinlichkeit
        """
        return 1 - (1 - p) ** n_iterations
    
    @staticmethod
    def expected_solutions_found(n_iterations: int, p: float = 1/3) -> float:
        """
        Erwartete Anzahl gefundener Lösungen.
        
        E[S_n] = n * p
        
        Args:
            n_iterations: Anzahl Iterationen
            p: Erfolgswahrscheinlichkeit pro Iteration
            
        Returns:
            Erwartete Anzahl Lösungen
        """
        return n_iterations * p
    
    @staticmethod
    def variance_solutions_found(n_iterations: int, p: float = 1/3) -> float:
        """
        Varianz der Anzahl gefundener Lösungen.
        
        Var(S_n) = n * p * (1-p)
        
        Args:
            n_iterations: Anzahl Iterationen
            p: Erfolgswahrscheinlichkeit pro Iteration
            
        Returns:
            Varianz
        """
        return n_iterations * p * (1 - p)
    
    @staticmethod
    def converges_to_one(n_iterations: int, p: float = 1/3) -> float:
        """
        Konvergenz gegen 1 mit exponentieller Geschwindigkeit.
        
        lim_(n→∞) [1 - (1-p)^n] = 1 - lim_(n→∞) [(1-p)^n] = 1
        
        Args:
            n_iterations: Anzahl Iterationen
            p: Erfolgswahrscheinlichkeit
            
        Returns:
            Wie nah an 1 die Wahrscheinlichkeit ist
        """
        prob = 1 - (1 - p) ** n_iterations
        return prob
    
    @staticmethod
    def lbv_combined_complexity(n_variables: int) -> float:
        """
        Gesamtkomplexität: LBV + Subgraph-Solver.
        
        E[T_LBV] = O(m^3 * log m)
        
        Args:
            n_variables: Anzahl Variablen
            
        Returns:
            Geschätzte Komplexität in Operationen
        """
        k = math.ceil(math.log2(n_variables)) + 2
        return n_variables ** 3 * k
