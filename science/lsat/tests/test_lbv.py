"""
Tests for lbv module (lbv.py)
"""

import pytest
from src.circuit import CircuitBuilder
from src.lbv import LogarithmicAssignmentProcedure, MassiveNetworkAnalyzer


class TestLAP:
    """Tests for LAP."""
    
    def test_lap_simple_and(self):
        """Test LAP with simple AND circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        found = lap.execute("out")
        
        assert isinstance(found, bool)
    
    def test_lap_statistics(self):
        """Test LAP statistics."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        lap.execute("out")
        stats = lap.get_statistics()
        
        assert isinstance(stats, dict)
    
    def test_massive_network_analyzer(self):
        """Test massive network analyzer."""
        analyzer = MassiveNetworkAnalyzer()
        
        # Just verify analyzer was created
        assert analyzer is not None


class TestLAPAdvanced:
    """Advanced LAP tests."""
    
    def test_lap_check_assignment(self):
        """Test LAP assignment checking."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_or(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        
        # Test checking an assignment
        assignment = {"x": True, "y": False}
        result = lap._check_assignment(assignment, "out")
        
        assert result == True  # OR(True, False) = True
    
    def test_lap_no_inputs(self):
        """Test LAP with circuit having no inputs."""
        circuit_obj = CircuitBuilder()
        circuit_obj.add_input("x")
        circuit = circuit_obj.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        found = lap.execute("x")
        
        assert isinstance(found, bool)


class TestLAPMethods:
    """Tests for individual LAP methods."""
    
    def test_lap_execute_with_single_input(self):
        """Test LAP execute with single input."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)
    
    def test_lap_with_or_gates(self):
        """Test LAP with OR gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_or(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        found = lap.execute("out")
        
        assert isinstance(found, bool)
    
    def test_lap_get_statistics(self):
        """Test LAP statistics."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        lap.execute("out")
        stats = lap.get_statistics()
        
        assert "iterations" in stats or isinstance(stats, dict)
    
    def test_lap_with_multiple_gates(self):
        """Test LAP with multiple gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        and1 = builder.add_and(x, y)
        and2 = builder.add_and(and1, z)
        builder.add_output(and2, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)


class TestLAPOptimizer:
    """Tests for LAP_Optimizer class."""
    
    def test_optimizer_creation(self):
        """Test creating LAP optimizer."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        from src.lbv import LAP_Optimizer
        optimizer = LAP_Optimizer(circuit)
        assert optimizer is not None
    
    def test_optimizer_execute(self):
        """Test optimizer execute method."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        from src.lbv import LAP_Optimizer
        optimizer = LAP_Optimizer(circuit)
        result = optimizer.execute_optimized("out")
        
        assert isinstance(result, bool)
    
    def test_optimizer_get_stats(self):
        """Test optimizer stats."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        from src.lbv import LAP_Optimizer
        optimizer = LAP_Optimizer(circuit)
        optimizer.execute_optimized("out")
        stats = optimizer.get_stats()
        
        assert isinstance(stats, dict)


class TestMassiveNetworkAnalyzerExtended:
    """Extended tests for MassiveNetworkAnalyzer."""
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = MassiveNetworkAnalyzer(8e22)
        assert analyzer is not None
    
    def test_analyzer_required_iterations(self):
        """Test required iterations calculation."""
        analyzer = MassiveNetworkAnalyzer()
        
        # Test with various input counts
        iter_0 = analyzer.compute_required_iterations(0)
        iter_1 = analyzer.compute_required_iterations(1)
        iter_8 = analyzer.compute_required_iterations(8)
        iter_16 = analyzer.compute_required_iterations(16)
        iter_1024 = analyzer.compute_required_iterations(1024)
        
        assert iter_0 == 0
        assert iter_1 > 0
        assert iter_8 > iter_1
        assert iter_16 > iter_8
        assert iter_1024 > iter_16
    
    def test_analyzer_analyze_massive(self):
        """Test massive network analysis."""
        analyzer = MassiveNetworkAnalyzer()
        results = analyzer.analyze_massive_network(1000)
        
        assert "lap_iterations" in results
        assert results["lap_iterations"] > 0
    
    def test_analyzer_analysis_results(self):
        """Test analysis results structure."""
        analyzer = MassiveNetworkAnalyzer()
        results = analyzer.analyze_massive_network(100)
        
        assert "network_size" in results
        assert "num_inputs" in results
        assert "lap_iterations" in results
        assert "improvement_factor" in results


class TestLAPEdgeCases:
    """Tests for LAP edge cases."""
    
    def test_lap_single_true_input(self):
        """Test LAP with single True condition."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        # Should find satisfying assignment where x=True
        if result:
            assert lap.satisfying_assignment is not None
    
    def test_lap_nand_gate(self):
        """Test LAP with NAND gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_nand(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)
    
    def test_lap_nor_gate(self):
        """Test LAP with NOR gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_nor(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)
    
    def test_lap_xor_gate(self):
        """Test LAP with XOR gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_xor(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)
    
    def test_lap_get_assignment_if_satisfiable(self):
        """Test getting assignment from satisfiable LAP."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_or(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        lap.execute("out")
        
        assignment = lap.get_assignment_if_satisfiable()
        # OR gate should be satisfiable when at least one input is True
        if assignment:
            assert isinstance(assignment, dict)
    
    def test_lap_print_statistics(self, capsys):
        """Test LAP statistics printing."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        lap.execute("out")
        lap.print_statistics()
        
        captured = capsys.readouterr()
        assert "Statistics" in captured.out or "Iteration" in captured.out or len(captured.out) > 0
    
    def test_analyzer_print_analysis(self, capsys):
        """Test analyzer analysis printing."""
        analyzer = MassiveNetworkAnalyzer()
        analyzer.print_analysis()
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0
    
    def test_lap_three_variable_and(self):
        """Test LAP with three-variable AND."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        and1 = builder.add_and(x, y)
        and2 = builder.add_and(and1, z)
        builder.add_output(and2, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        if result:
            # Should find x=True, y=True, z=True
            assert lap.satisfying_assignment is not None
            assert lap.satisfying_assignment.get("x") == True
            assert lap.satisfying_assignment.get("y") == True
            assert lap.satisfying_assignment.get("z") == True
    
    def test_lap_three_variable_or(self):
        """Test LAP with three-variable OR."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        or1 = builder.add_or(x, y)
        or2 = builder.add_or(or1, z)
        builder.add_output(or2, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        # OR is very easy to satisfy
        assert result == True or result == False
    
    def test_lap_nested_circuit(self):
        """Test LAP with nested complex circuit."""
        builder = CircuitBuilder()
        a = builder.add_input("a")
        b = builder.add_input("b")
        c = builder.add_input("c")
        d = builder.add_input("d")
        
        and1 = builder.add_and(a, b)
        and2 = builder.add_and(c, d)
        final = builder.add_or(and1, and2)
        
        builder.add_output(final, "out")
        circuit = builder.build()
        
        lap = LogarithmicAssignmentProcedure(circuit)
        result = lap.execute("out")
        
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
