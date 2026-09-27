"""
Tests for circuit module (circuit.py)
"""

import pytest
from src.circuit import BooleanCircuit, CircuitBuilder, CircuitNode


class TestCircuit:
    """Test BooleanCircuit class."""
    
    def test_circuit_creation(self):
        """Test creating circuits."""
        circuit = BooleanCircuit("TestCircuit")
        assert circuit.name == "TestCircuit"
    
    def test_circuit_add_inputs(self):
        """Test adding inputs to circuit."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        assert len(circuit) == 2
    
    def test_circuit_and_gate(self):
        """Test AND gate evaluation."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "AND", ["x", "y"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True, "y": True})
        assert result["g1"] == True
        
        result = circuit.evaluate({"x": True, "y": False})
        assert result["g1"] == False
    
    def test_circuit_or_gate(self):
        """Test OR gate evaluation."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "OR", ["x", "y"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True, "y": False})
        assert result["g1"] == True
        
        result = circuit.evaluate({"x": False, "y": False})
        assert result["g1"] == False
    
    def test_circuit_xor_gate(self):
        """Test XOR gate evaluation."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "XOR", ["x", "y"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True, "y": False})
        assert result["g1"] == True
        
        result = circuit.evaluate({"x": True, "y": True})
        assert result["g1"] == False


class TestCircuitAdvanced:
    """Test advanced circuit operations."""
    
    def test_circuit_not_gate(self):
        """Test NOT gate."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_gate("g1", "NOT", ["x"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True})
        assert result["g1"] == False
    
    def test_circuit_nand_gate(self):
        """Test NAND gate."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "NAND", ["x", "y"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True, "y": True})
        assert result["g1"] == False
        
        result = circuit.evaluate({"x": False, "y": False})
        assert result["g1"] == True
    
    def test_circuit_nor_gate(self):
        """Test NOR gate."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "NOR", ["x", "y"])
        circuit.add_output("out", "g1")
        
        result = circuit.evaluate({"x": True, "y": True})
        assert result["g1"] == False
    
    def test_circuit_topological_order(self):
        """Test topological ordering."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "AND", ["x", "y"])
        circuit.add_gate("g2", "OR", ["g1", "x"])
        circuit.add_output("out", "g2")
        
        topo_order = circuit.get_topological_order()
        
        assert len(topo_order) > 0
    
    def test_circuit_stats(self):
        """Test circuit statistics."""
        circuit = BooleanCircuit()
        x = circuit.add_input("x")
        y = circuit.add_input("y")
        circuit.add_gate("g1", "AND", ["x", "y"])
        circuit.add_output("out", "g1")
        
        stats = circuit.get_stat_info()
        assert "total_nodes" in stats


class TestCircuitNodeAdvanced:
    """Tests for CircuitNode."""
    
    def test_circuit_node_binary_gate(self):
        """Test binary gate properties."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_gate("g1", "AND", ["x", "y"])
        
        node = circuit.get_node("g1")
        assert node.is_binary_gate()
    
    def test_circuit_node_unary_gate(self):
        """Test unary gate properties."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_gate("g1", "NOT", ["x"])
        
        node = circuit.get_node("g1")
        assert node.is_unary_gate()
    
    def test_circuit_node_arity(self):
        """Test gate arity."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        circuit.add_input("y")
        circuit.add_input("z")
        circuit.add_gate("g1", "AND", ["x", "y"])
        circuit.add_gate("g2", "NOT", ["x"])
        
        node_and = circuit.get_node("g1")
        node_not = circuit.get_node("g2")
        
        assert node_and.get_arity() == 2
        assert node_not.get_arity() == 1


class TestCircuitGates:
    """Tests for all circuit gate types."""
    
    def test_circuit_with_all_gates(self):
        """Test circuit with all gate types."""
        builder = CircuitBuilder()
        
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        and_result = builder.add_and(x, y)
        or_result = builder.add_or(x, y)
        not_result = builder.add_not(x)
        nand_result = builder.add_nand(x, y)
        nor_result = builder.add_nor(x, y)
        xor_result = builder.add_xor(x, y)
        
        builder.add_output(and_result, "and_out")
        builder.add_output(or_result, "or_out")
        builder.add_output(not_result, "not_out")
        builder.add_output(nand_result, "nand_out")
        builder.add_output(nor_result, "nor_out")
        builder.add_output(xor_result, "xor_out")
        
        circuit = builder.build()
        
        assignment = {"x": False, "y": False}
        results = circuit.evaluate(assignment)
        
        assert results["and_out"] == False
        assert results["or_out"] == False
        assert results["not_out"] == True
        assert results["nand_out"] == True
        assert results["nor_out"] == True
        assert results["xor_out"] == False
    
    def test_circuit_constants(self):
        """Test circuit with constant nodes."""
        circuit = BooleanCircuit()
        
        circuit.add_constant_node("const_true", True)
        circuit.add_constant_node("const_false", False)
        
        circuit.add_output("true_out", "const_true")
        circuit.add_output("false_out", "const_false")
        
        results = circuit.evaluate({})
        
        assert results["const_true"] == True
        assert results["const_false"] == False


class TestComplexIntegration:
    """Complex integration scenarios."""
    
    def test_nested_gates(self):
        """Test deeply nested gate structure."""
        builder = CircuitBuilder()
        
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        
        # Create (x AND y) OR (NOT z)
        and_result = builder.add_and(x, y)
        not_z = builder.add_not(z)
        or_result = builder.add_or(and_result, not_z)
        
        builder.add_output(or_result, "out")
        circuit = builder.build()
        
        # Test several assignments
        test_cases = [
            ({"x": False, "y": False, "z": False}, True),   # OR(False, True) = True
            ({"x": True, "y": False, "z": False}, True),    # OR(False, True) = True
            ({"x": True, "y": True, "z": False}, True),     # OR(True, True) = True
            ({"x": True, "y": True, "z": True}, True),      # OR(True, False) = True
            ({"x": False, "y": False, "z": True}, False),   # OR(False, False) = False
        ]
        
        for assignment, expected in test_cases:
            result = circuit.evaluate(assignment)["out"]
            assert result == expected
    
    def test_multiple_outputs_same_circuit(self):
        """Test circuit with multiple independent outputs."""
        builder = CircuitBuilder()
        
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        and_result = builder.add_and(x, y)
        or_result = builder.add_or(x, y)
        xor_result = builder.add_xor(x, y)
        
        builder.add_output(and_result, "and_out")
        builder.add_output(or_result, "or_out")
        builder.add_output(xor_result, "xor_out")
        
        circuit = builder.build()
        
        # When both False
        results = circuit.evaluate({"x": False, "y": False})
        assert results["and_out"] == False
        assert results["or_out"] == False
        assert results["xor_out"] == False
        
        # When both True
        results = circuit.evaluate({"x": True, "y": True})
        assert results["and_out"] == True
        assert results["or_out"] == True
        assert results["xor_out"] == False


class TestCircuitNodeMethods:
    """Test CircuitNode specific methods."""
    
    def test_circuit_node_is_binary_gate(self):
        """Test is_binary_gate method on various gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        and_gate = builder.add_and(x, y)
        or_gate = builder.add_or(x, y)
        nand_gate = builder.add_nand(x, y)
        nor_gate = builder.add_nor(x, y)
        xor_gate = builder.add_xor(x, y)
        
        circuit = builder.build()
        
        assert circuit.nodes[and_gate].is_binary_gate() == True
        assert circuit.nodes[or_gate].is_binary_gate() == True
        assert circuit.nodes[nand_gate].is_binary_gate() == True
        assert circuit.nodes[nor_gate].is_binary_gate() == True
        assert circuit.nodes[xor_gate].is_binary_gate() == True
    
    def test_circuit_node_is_unary_gate(self):
        """Test is_unary_gate method."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        not_x = builder.add_not(x)
        
        circuit = builder.build()
        
        assert circuit.nodes[not_x].is_unary_gate() == True
        assert circuit.nodes[x].is_unary_gate() == False
    
    def test_circuit_node_get_arity_binary(self):
        """Test get_arity for binary gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        and_gate = builder.add_and(x, y)
        
        circuit = builder.build()
        assert circuit.nodes[and_gate].get_arity() == 2
    
    def test_circuit_node_get_arity_unary(self):
        """Test get_arity for unary gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        not_x = builder.add_not(x)
        
        circuit = builder.build()
        assert circuit.nodes[not_x].get_arity() == 1
    
    def test_circuit_node_get_arity_input(self):
        """Test get_arity for input nodes."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        
        circuit = builder.build()
        assert circuit.nodes[x].get_arity() == 0
    
    def test_circuit_node_constant(self):
        """Test circuit node with constant values."""
        circuit = BooleanCircuit()
        true_node = CircuitNode(
            node_id="const_true",
            gate_type="CONSTANT",
            inputs=[],
            is_input=False,
            is_output=False,
            constant_value=True
        )
        false_node = CircuitNode(
            node_id="const_false",
            gate_type="CONSTANT",
            inputs=[],
            is_input=False,
            is_output=False,
            constant_value=False
        )
        
        circuit.add_node(true_node)
        circuit.add_node(false_node)
        
        assert circuit.nodes["const_true"].constant_value == True
        assert circuit.nodes["const_false"].constant_value == False


class TestCircuitStructure:
    """Test circuit structure and analysis methods."""
    
    def test_circuit_get_topological_order(self):
        """Test getting topological ordering of gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        and1 = builder.add_and(x, y)
        or1 = builder.add_or(and1, x)
        builder.add_output(or1, "out")
        
        circuit = builder.build()
        topo_order = circuit.get_topological_order()
        
        assert len(topo_order) > 0
        assert or1 in topo_order
        assert and1 in topo_order
    
    def test_circuit_stats(self):
        """Test circuit statistics."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        and1 = builder.add_and(x, y)
        or1 = builder.add_or(and1, z)
        builder.add_output(or1, "out")
        
        circuit = builder.build()
        stats = circuit.get_stat_info()
        
        assert "num_inputs" in stats
        assert "total_nodes" in stats
        assert stats["num_inputs"] == 3
    
    def test_circuit_with_three_variable_and(self):
        """Test circuit with three-variable AND gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        
        and1 = builder.add_and(x, y)
        and2 = builder.add_and(and1, z)
        
        builder.add_output(and2, "out")
        circuit = builder.build()
        
        # All True
        result = circuit.evaluate({"x": True, "y": True, "z": True})
        assert result["out"] == True
        
        # One False
        result = circuit.evaluate({"x": True, "y": True, "z": False})
        assert result["out"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
