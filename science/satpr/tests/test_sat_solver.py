"""
Tests für das SAT-Solver Modul.

Tests für:
- Klauseln und Formeln
- SAT-Solver
- Kombinationspyramide
- Subgraph-Exploration
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from sat_solver import Clause, SATFormula, CombinationPyramid, SubgraphSATSolver


class TestClause:
    """Tests für Clause."""
    
    def test_clause_creation(self):
        """Test: Erstelle Klausel."""
        clause = Clause([1, -2, 3])
        
        assert len(clause.literals) == 3
        assert clause.literals == [1, -2, 3]
    
    def test_clause_is_satisfied_true(self):
        """Test: Klausel ist erfüllt."""
        clause = Clause([1, -2, 3])  # x1 OR NOT x2 OR x3
        assignment = {1: False, 2: True, 3: False}  # x1=F, x2=T, x3=F
        
        # NOT x2 ist erfüllt (x2=T → NOT x2 = F), aber x1=F und x3=F
        # Warte: x1=F, NOT x2 = F, x3=F → nicht erfüllt
        
        # Versuche mit erfüllter Belegung
        assignment = {1: True, 2: True, 3: False}  # x1=T, x2=T, x3=F
        
        is_satisfied = clause.is_satisfied(assignment)
        
        # x1=T erfüllt die Klausel
        assert is_satisfied
    
    def test_clause_is_satisfied_false(self):
        """Test: Klausel ist nicht erfüllt."""
        clause = Clause([1, 2, 3])  # x1 OR x2 OR x3
        assignment = {1: False, 2: False, 3: False}
        
        is_satisfied = clause.is_satisfied(assignment)
        
        assert not is_satisfied
    
    def test_clause_evaluate_true(self):
        """Test: Evaluiere erfüllte Klausel."""
        clause = Clause([1, -2])
        assignment = {1: True}  # Partiell: nur x1 zugewiesen
        
        result = clause.evaluate(assignment)
        
        # Bereits erfüllt
        assert result is True
    
    def test_clause_evaluate_false(self):
        """Test: Evaluiere nicht erfüllbare Klausel."""
        clause = Clause([1, 2])
        assignment = {1: False, 2: False}  # Vollständig
        
        result = clause.evaluate(assignment)
        
        assert result is False
    
    def test_clause_evaluate_unknown(self):
        """Test: Evaluiere Klausel mit unbekannten Variablen."""
        clause = Clause([1, 2])
        assignment = {1: False}  # Nur x1 zugewiesen
        
        result = clause.evaluate(assignment)
        
        # x1=F, x2 unbekannt → undefiniert
        assert result is None
    
    def test_clause_single_literal(self):
        """Test: Klausel mit einzelnem Literal."""
        clause = Clause([5])
        
        assert len(clause.literals) == 1
        
        # Erfüllt wenn x5=T
        assert clause.is_satisfied({5: True})
        assert not clause.is_satisfied({5: False})
    
    def test_clause_negation(self):
        """Test: Klausel mit Negationen."""
        clause = Clause([-1, -2, -3])  # NOT x1 OR NOT x2 OR NOT x3
        
        # Erfüllt wenn mindestens eine Variable false ist
        assert clause.is_satisfied({1: False, 2: True, 3: True})
        assert not clause.is_satisfied({1: True, 2: True, 3: True})


class TestSATFormula:
    """Tests für SATFormula."""
    
    def test_formula_creation(self):
        """Test: Erstelle SAT-Formel."""
        clauses = [Clause([1, 2]), Clause([-1, 3])]
        formula = SATFormula(clauses, n_vars=3)
        
        assert len(formula.clauses) == 2
        assert formula.n_vars == 3
    
    def test_formula_auto_n_vars(self):
        """Test: Bestimme n_vars automatisch."""
        clauses = [Clause([1, 5]), Clause([3, 4])]
        formula = SATFormula(clauses)
        
        assert formula.n_vars == 5
    
    def test_formula_is_satisfied_true(self):
        """Test: Formel ist erfüllt."""
        clauses = [
            Clause([1, 2]),      # x1 OR x2
            Clause([-1, 3])      # NOT x1 OR x3
        ]
        formula = SATFormula(clauses, n_vars=3)
        
        assignment = {1: True, 2: False, 3: True}
        
        is_satisfied = formula.is_satisfied(assignment)
        
        # Erste Klausel: x1=T (erfüllt)
        # Zweite Klausel: NOT x1=F, x3=T (erfüllt)
        assert is_satisfied
    
    def test_formula_is_satisfied_false(self):
        """Test: Formel ist nicht erfüllt."""
        clauses = [
            Clause([1, 2]),
            Clause([-1, -2])
        ]
        formula = SATFormula(clauses, n_vars=2)
        
        # Keine Belegung kann beide erfüllen
        # (x1 OR x2) AND (NOT x1 OR NOT x2) ist erfüllbar
        # Aber mit x1=T, x2=T: erste erfüllt, zweite nicht
        assignment = {1: True, 2: True}
        
        is_satisfied = formula.is_satisfied(assignment)
        
        # NOT x1 OR NOT x2 mit x1=T, x2=T ist nicht erfüllt
        assert not is_satisfied
    
    def test_formula_evaluate_partial(self):
        """Test: Evaluiere Formel mit partieller Belegung."""
        clauses = [Clause([1, 2, 3])]
        formula = SATFormula(clauses, n_vars=3)
        
        partial = {1: False}
        
        result = formula.evaluate(partial)
        
        # Noch undefiniert (x2, x3 unbekannt)
        assert result is None
    
    def test_formula_is_valid_assignment(self):
        """Test: Prüfe ob Belegung vollständig ist."""
        formula = SATFormula([Clause([1, 2])], n_vars=2)
        
        valid = {1: True, 2: False}
        invalid = {1: True}
        
        assert formula.is_valid_assignment(valid)
        assert not formula.is_valid_assignment(invalid)
    
    def test_formula_random_assignment(self):
        """Test: Generiere zufällige Belegung."""
        formula = SATFormula([Clause([1, 2, 3])], n_vars=3)
        
        assignment = formula.random_assignment()
        
        assert len(assignment) == 3
        assert all(isinstance(v, (bool, np.bool_)) for v in assignment.values())
    
    def test_formula_random_3sat(self):
        """Test: Generiere zufällige 3-SAT Formel."""
        n_vars = 5
        n_clauses = 10
        
        formula = SATFormula.random_formula(n_vars, n_clauses, k_literals=3)
        
        assert formula.n_vars == n_vars
        assert len(formula.clauses) == n_clauses
        
        # Prüfe dass jede Klausel 3 Literale hat
        for clause in formula.clauses:
            assert len(clause.literals) == 3
    
    def test_formula_satisfiability_random(self):
        """Test: Teste Erfüllbarkeit zufälliger Formel."""
        # Generiere kleine Formel
        formula = SATFormula.random_formula(3, 2, k_literals=2)
        
        # Versuche zufällige Belegungen
        found_solution = False
        for _ in range(100):
            assignment = formula.random_assignment()
            if formula.is_satisfied(assignment):
                found_solution = True
                break
        
        # Bei nur 3 Variablen und 2 Klauseln sollten wir schnell eine finden
        assert found_solution


class TestCombinationPyramid:
    """Tests für CombinationPyramid."""
    
    def test_pyramid_initialization(self):
        """Test: Initialisiere Kombinationspyramide."""
        pyramid = CombinationPyramid(4)
        
        assert pyramid.n_vars == 4
        assert np.isclose(pyramid.p_left, 1/3)
        assert np.isclose(pyramid.p_right, 1/3)
        assert np.isclose(pyramid.p_undefined, 1/3)
    
    def test_pyramid_level_size(self):
        """Test: Berechne Größe einer Ebene."""
        pyramid = CombinationPyramid(3)
        
        # Level k hat C(n,k) Knoten
        assert pyramid.level_size(0) == 1      # C(3,0) = 1
        assert pyramid.level_size(1) == 3      # C(3,1) = 3
        assert pyramid.level_size(2) == 3      # C(3,2) = 3
        assert pyramid.level_size(3) == 1      # C(3,3) = 1
    
    def test_pyramid_total_nodes(self):
        """Test: Berechne Gesamtzahl Knoten."""
        n = 5
        pyramid = CombinationPyramid(n)
        
        total = pyramid.total_nodes()
        
        assert total == 2 ** n
    
    def test_pyramid_leaf_nodes(self):
        """Test: Berechne Anzahl Blattknoten."""
        n = 4
        pyramid = CombinationPyramid(n)
        
        leaves = pyramid.leaf_nodes()
        
        # Blattknoten = alle möglichen Belegungen = 2^n
        assert leaves == 2 ** n
    
    def test_pyramid_three_way_split(self):
        """Test: Prüfe 1/3-Dreiteilung."""
        pyramid = CombinationPyramid(5)
        
        split = pyramid.three_way_split()
        
        assert len(split) == 3
        assert np.isclose(sum(split), 1.0)
        assert all(p == 1/3 for p in split)
    
    def test_pyramid_expected_depth(self):
        """Test: Berechne erwartete Tiefe."""
        n = 5
        pyramid = CombinationPyramid(n)
        
        depth = pyramid.expected_depth()
        
        # Bei Dreiteilung: d ≈ log_3(2^n) = n * log_3(2)
        expected_depth = n * np.log(2) / np.log(3)
        
        assert np.isclose(depth, expected_depth)
    
    def test_pyramid_custom_probabilities(self):
        """Test: Initialisiere mit benutzerdefinierten Wahrscheinlichkeiten."""
        pyramid = CombinationPyramid(3, p_left=0.4, p_right=0.3, p_undefined=0.3)
        
        split = pyramid.three_way_split()
        
        assert np.isclose(split[0], 0.4)
        assert np.isclose(split[1], 0.3)
        assert np.isclose(split[2], 0.3)
    
    def test_pyramid_invalid_probabilities(self):
        """Test: Lehne ungültige Wahrscheinlichkeiten ab."""
        with pytest.raises(ValueError):
            CombinationPyramid(3, p_left=0.5, p_right=0.3, p_undefined=0.3)


class TestSubgraphSATSolver:
    """Tests für SubgraphSATSolver."""
    
    def test_solver_initialization(self):
        """Test: Initialisiere Solver."""
        formula = SATFormula.random_formula(3, 2)
        solver = SubgraphSATSolver(formula)
        
        assert solver.formula == formula
        assert solver.operations_count == 0
        assert solver.found_solution is False
    
    def test_solver_simple_sat(self):
        """Test: Löse einfache erfüllbare Formel."""
        clauses = [Clause([1]), Clause([2])]  # x1 AND x2
        formula = SATFormula(clauses, n_vars=2)
        
        solver = SubgraphSATSolver(formula)
        satisfiable, assignment, ops = solver.solve()
        
        assert satisfiable
        assert assignment is not None
        assert ops > 0
    
    def test_solver_simple_unsat(self):
        """Test: Erkenne unerfüllbare Formel."""
        clauses = [
            Clause([1]),   # x1
            Clause([-1])   # NOT x1
        ]
        formula = SATFormula(clauses, n_vars=1)
        
        solver = SubgraphSATSolver(formula)
        satisfiable, assignment, ops = solver.solve()
        
        assert not satisfiable
        assert assignment is None
    
    def test_solver_random_formula(self):
        """Test: Löse zufällige Formel."""
        formula = SATFormula.random_formula(4, 5)
        
        solver = SubgraphSATSolver(formula)
        satisfiable, assignment, ops = solver.solve()
        
        if satisfiable:
            # Verifiziere dass Belegung tatsächlich erfüllt
            assert formula.is_satisfied(assignment)
        
        assert ops > 0
    
    def test_solver_estimated_runtime(self):
        """Test: Schätze Laufzeit."""
        n = 5
        formula = SATFormula.random_formula(n, n)
        
        solver = SubgraphSATSolver(formula)
        runtime = solver.estimated_runtime()
        
        # Sollte O(n^3) sein
        assert runtime > 0
        assert runtime <= n ** 3 * 10  # Mit etwas Overhead
    
    def test_solver_subgraph_exploration_factor(self):
        """Test: Berechne Explorations-Effizienz."""
        formula = SATFormula.random_formula(3, 5)
        
        solver = SubgraphSATSolver(formula)
        factor = solver.subgraph_exploration_factor()
        
        assert 0 <= factor <= 1
    
    def test_solver_with_timeout(self):
        """Test: Solver mit Timeout."""
        formula = SATFormula.random_formula(5, 5)
        
        solver = SubgraphSATSolver(formula)
        satisfiable, assignment, ops = solver.solve(timeout=10)
        
        # Sollte stoppen bei Timeout
        assert ops <= 10


class TestIntegration:
    """Integrationstests für SAT-Solver."""
    
    def test_formula_to_solver_pipeline(self):
        """Test: Pipeline von Formel zu Solver."""
        # Generiere Formel
        formula = SATFormula.random_formula(3, 3, k_literals=2)
        
        # Prüfe mit Solver
        solver = SubgraphSATSolver(formula)
        satisfiable, assignment, ops = solver.solve()
        
        # Wenn erfüllt, verifiziere
        if satisfiable:
            assert formula.is_satisfied(assignment)
    
    def test_pyramid_with_solver(self):
        """Test: Kombinationspyramide mit Solver."""
        n = 4
        formula = SATFormula.random_formula(n, 5)
        pyramid = CombinationPyramid(n)
        
        # Pyramide sollte konsistent mit Formel sein
        assert pyramid.total_nodes() == 2 ** n
        
        # Solver sollte auf Formel arbeitbar sein
        solver = SubgraphSATSolver(formula)
        satisfiable, _, ops = solver.solve()
        
        assert ops > 0
    
    def test_multiple_formulas_consistency(self):
        """Test: Konsistenz über mehrere Formeln."""
        n = 3
        results = []
        
        for _ in range(5):
            formula = SATFormula.random_formula(n, 4)
            solver = SubgraphSATSolver(formula)
            satisfiable, assignment, ops = solver.solve()
            
            results.append({
                'satisfiable': satisfiable,
                'operations': ops,
                'verified': not satisfiable or formula.is_satisfied(assignment)
            })
        
        # Alle Ergebnisse sollten verifizierbar sein
        assert all(r['verified'] for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
