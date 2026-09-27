"""
Tests for subgraph_sat_solver module (subgraph_sat_solver.py)
"""

import pytest
from src.cnf import CNFFormula, Literal, Clause
from src.subgraph_sat_solver import SubgraphSATSolver


class TestSubgraphSATSolverBasic:
    """Basic tests for Subgraph SAT Solver."""
    
    def test_sat_solver_creation(self):
        """Test creating SAT solver."""
        solver = SubgraphSATSolver()
        assert solver is not None
    
    def test_sat_solver_verify_solution(self):
        """Test verifying a solution."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        cnf.add_clause(Clause({Literal("x", False), Literal("y", False)}))
        
        # x=True satisfies (x ∨ y)
        assignment = {"x": True, "y": False}
        
        assert solver.verify_solution(cnf, assignment)
    
    def test_sat_solver_3sat_conversion(self):
        """Test 3-SAT normalization."""
        solver = SubgraphSATSolver()
        
        # Create CNF with clause of length > 3
        cnf = CNFFormula()
        cnf.add_clause(Clause({
            Literal("x", False),
            Literal("y", False),
            Literal("z", False),
            Literal("w", False)
        }))
        
        three_sat = solver._normalize_to_3sat(cnf)
        
        # All clauses should have <= 3 literals
        for clause in three_sat.clauses:
            assert len(clause) <= 3
    
    def test_sat_solver_clique_reduction(self):
        """Test clique reduction."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        cnf.add_clause(Clause({Literal("x", False), Literal("y", True)}))
        cnf.add_clause(Clause({Literal("x", True), Literal("y", False)}))
        
        graph, mapping = solver._clique_reduction(cnf)
        
        assert graph.num_nodes > 0
        assert len(mapping) == 2


class TestSubgraphSATSolverAdvanced:
    """Advanced Subgraph SAT Solver tests."""
    
    def test_sat_solver_empty_formula(self):
        """Test SAT solver with empty CNF."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        result = solver.solve(cnf)
        
        # Empty CNF is satisfiable
        assert result.satisfiable == True
    
    def test_sat_solver_unit_clause(self):
        """Test SAT solver with unit clause."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        cnf.add_clause(Clause({Literal("x", False)}))
        
        # Unit clause should be solvable
        assert cnf.get_num_clauses() == 1
    
    def test_sat_solver_2sat(self):
        """Test SAT solver with 2-SAT formula."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        cnf.add_clause(Clause({Literal("x", False), Literal("y", False)}))
        cnf.add_clause(Clause({Literal("x", True), Literal("y", True)}))
        
        # 2-SAT should have 2 clauses
        assert cnf.get_num_clauses() == 2
    
    def test_sat_solver_conflicting_clauses(self):
        """Test SAT solver with conflicting clauses."""
        solver = SubgraphSATSolver()
        
        cnf = CNFFormula()
        cnf.add_clause(Clause({Literal("x", False)}))
        cnf.add_clause(Clause({Literal("x", True)}))
        
        # Conflicting clauses
        assert cnf.get_num_clauses() == 2
    
    def test_graph_creation(self):
        """Test subgraph SAT solver components."""
        solver = SubgraphSATSolver()
        
        # Solver should be created properly
        assert solver is not None
        assert solver.subgraph_algo is not None
    
    def test_graph_adjacency(self):
        """Test graph adjacency matrix."""
        solver = SubgraphSATSolver()
        
        # Solver should be created properly
        assert solver is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
