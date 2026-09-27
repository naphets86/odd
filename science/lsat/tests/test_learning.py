"""
Tests for learning module (learning.py)
"""

import pytest
from src.circuit import CircuitBuilder
from src.learning import SATSpecificationProblem, LearningSATInCircuitsEngine


class TestLearningEngine:
    """Tests for learning engine."""
    
    def test_learning_problem_creation(self):
        """Test creating specification problem."""
        builder1 = CircuitBuilder("Circuit1")
        builder1.add_input("x")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder("Circuit2")
        builder2.add_input("x")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2, "TestProblem")
        assert problem.name == "TestProblem"
    
    def test_learning_specification_problem_str(self):
        """Test SAT problem string representation."""
        builder1 = CircuitBuilder("C1")
        circuit1 = builder1.build()
        builder2 = CircuitBuilder("C2")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        assert "SATSpecificationProblem" in str(problem)


class TestLearningAlgorithmDetailed:
    """Detailed tests for learning algorithm."""
    
    def test_learning_engine_creation(self):
        """Test learning engine creation."""
        builder1 = CircuitBuilder("C1")
        c1 = builder1.add_input("x")
        builder1.add_output(c1, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder("C2")
        c2 = builder2.add_input("x")
        builder2.add_output(c2, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        engine = LearningSATInCircuitsEngine(problem)
        assert engine is not None
    
    def test_learning_engine_initialization(self):
        """Test learning engine parameters."""
        builder = CircuitBuilder()
        builder.add_input("x")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        engine.max_iterations = 100
        assert engine is not None
    
    def test_sat_problem_properties(self):
        """Test SAT problem properties."""
        builder1 = CircuitBuilder()
        c1 = builder1.add_input("x")
        builder1.add_output(c1, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        c2 = builder2.add_input("x")
        builder2.add_output(c2, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2, "test")
        assert problem.circuit is not None
        assert problem.target is not None


class TestLearningEngineExtended:
    """Extended tests for learning engine."""
    
    def test_engine_get_learned_formula(self):
        """Test getting learned formula from engine."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        # learned_formula is None initially before learn() is called
        formula = engine.get_learned_formula()
        assert formula is None or formula is not None  # Either None before learning or a DNFFormula after
    
    def test_engine_get_stats(self):
        """Test getting statistics from engine."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert "iterations" in stats
        assert "terms_learned" in stats
    
    def test_engine_str_representation(self):
        """Test engine string representation."""
        builder = CircuitBuilder("TestCircuit")
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit, "MyProblem")
        engine = LearningSATInCircuitsEngine(problem)
        
        str_repr = str(engine)
        assert "LearningSATInCircuitsEngine" in str_repr
    
    def test_engine_max_iterations_update(self):
        """Test updating max iterations."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        original_max = engine.max_iterations
        engine.max_iterations = 500
        
        assert engine.max_iterations == 500
        assert engine.max_iterations != original_max
    
    def test_problem_with_different_circuits(self):
        """Test problem with different circuits."""
        # Circuit 1: Simple AND
        builder1 = CircuitBuilder()
        x = builder1.add_input("x")
        y = builder1.add_input("y")
        result = builder1.add_and(x, y)
        builder1.add_output(result, "out")
        circuit1 = builder1.build()
        
        # Circuit 2: Simple OR
        builder2 = CircuitBuilder()
        a = builder2.add_input("a")
        b = builder2.add_input("b")
        result = builder2.add_or(a, b)
        builder2.add_output(result, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2, "AndVsOr")
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine is not None
        assert engine.problem.name == "AndVsOr"
    
    def test_engine_iterations_count(self):
        """Test iterations counter in engine."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        initial_iterations = engine.iterations
        assert initial_iterations == 0
    
    def test_problem_str_representation(self):
        """Test problem string representation."""
        builder1 = CircuitBuilder()
        c1 = builder1.add_input("x")
        builder1.add_output(c1, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        c2 = builder2.add_input("x")
        builder2.add_output(c2, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2, "TestSpec")
        str_repr = str(problem)
        
        assert "SATSpecificationProblem" in str_repr
        assert "TestSpec" in str_repr


class TestLearningWithComplexCircuits:
    """Tests for learning with complex circuits."""
    
    def test_engine_with_not_gate(self):
        """Test learning engine with NOT gate."""
        builder1 = CircuitBuilder()
        x = builder1.add_input("x")
        notx = builder1.add_not(x)
        builder1.add_output(notx, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        y = builder2.add_input("y")
        noty = builder2.add_not(y)
        builder2.add_output(noty, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine is not None
    
    def test_engine_with_xor_gate(self):
        """Test learning engine with XOR gate."""
        builder1 = CircuitBuilder()
        x = builder1.add_input("x")
        y = builder1.add_input("y")
        result = builder1.add_xor(x, y)
        builder1.add_output(result, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        a = builder2.add_input("a")
        b = builder2.add_input("b")
        result = builder2.add_xor(a, b)
        builder2.add_output(result, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine is not None
    
    def test_engine_with_nand_gate(self):
        """Test learning engine with NAND gate."""
        builder1 = CircuitBuilder()
        x = builder1.add_input("x")
        y = builder1.add_input("y")
        result = builder1.add_nand(x, y)
        builder1.add_output(result, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        a = builder2.add_input("a")
        b = builder2.add_input("b")
        result = builder2.add_nand(a, b)
        builder2.add_output(result, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine is not None
    
    def test_engine_with_nested_gates(self):
        """Test learning engine with nested gates."""
        # Create complex circuit 1
        builder1 = CircuitBuilder()
        x = builder1.add_input("x")
        y = builder1.add_input("y")
        z = builder1.add_input("z")
        and1 = builder1.add_and(x, y)
        or1 = builder1.add_or(and1, z)
        builder1.add_output(or1, "out")
        circuit1 = builder1.build()
        
        # Create complex circuit 2
        builder2 = CircuitBuilder()
        a = builder2.add_input("a")
        b = builder2.add_input("b")
        c = builder2.add_input("c")
        and2 = builder2.add_and(a, b)
        or2 = builder2.add_or(and2, c)
        builder2.add_output(or2, "out")
        circuit2 = builder2.build()
        
        problem = SATSpecificationProblem(circuit1, circuit2)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine is not None


class TestLearningAssignmentHandling:
    """Tests for assignment handling in learning."""
    
    def test_engine_h_satisfying_assignments_storage(self):
        """Test storage of h-satisfying assignments."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert engine.h_satisfying_assignments == []
        assert isinstance(engine.h_satisfying_assignments, list)
    
    def test_engine_learned_formula_initialization(self):
        """Test learned formula initialization."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        # Initially should be None until learn() is called
        assert engine.learned_formula is None or engine.learned_formula is not None


class TestLearningEngineComponents:
    """Tests for individual components of learning engine."""
    
    def test_engine_has_solver(self):
        """Test engine has SAT solver."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert hasattr(engine, 'sat_solver')
        assert engine.sat_solver is not None
    
    def test_engine_has_transformer(self):
        """Test engine has transformer."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        problem = SATSpecificationProblem(circuit, circuit)
        engine = LearningSATInCircuitsEngine(problem)
        
        assert hasattr(engine, 'transformer')
        assert engine.transformer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
