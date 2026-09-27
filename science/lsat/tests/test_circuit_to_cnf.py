"""
Tests for circuit_to_cnf module (circuit_to_cnf.py)
"""

import pytest
from src.circuit import BooleanCircuit, CircuitBuilder
from src.circuit_to_cnf import CircuitToCNFTransformer
from src.cnf import CNFFormula


class TestCircuitToCNFTransformer:
    """Tests for CircuitToCNFTransformer."""
    
    def test_transformer_creation(self):
        """Test creating transformer."""
        transformer = CircuitToCNFTransformer()
        assert transformer is not None
    
    def test_aux_var_generation(self):
        """Test auxiliary variable generation."""
        transformer = CircuitToCNFTransformer()
        var1 = transformer._get_aux_var()
        var2 = transformer._get_aux_var()
        
        assert var1 != var2
        assert "_g" in var1
    
    def test_transform_simple_circuit(self):
        """Test transforming simple circuit to CNF."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert isinstance(cnf, CNFFormula)
        assert cnf.get_num_clauses() > 0
    
    def test_transform_not_gate(self):
        """Test transforming NOT gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        not_x = builder.add_not(x)
        builder.add_output(not_x, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() >= 2  # NOT requires at least 2 clauses
    
    def test_transform_or_gate(self):
        """Test transforming OR gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_or(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() > 0


class TestCircuitToCNFAdvanced:
    """Advanced tests for circuit-to-CNF transformation."""
    
    def test_transformer_xor_gate(self):
        """Test transforming XOR gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_xor(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        # XOR requires multiple clauses
        assert cnf.get_num_clauses() > 2
    
    def test_transformer_complex_circuit(self):
        """Test transforming complex circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        and1 = builder.add_and(x, y)
        and2 = builder.add_and(and1, z)
        builder.add_output(and2, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() > 0
        # Check variables are in CNF
        assert len(cnf.get_variables()) >= 3
    
    def test_transformer_input_only(self):
        """Test transforming simple INPUT circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        # INPUT circuit should produce CNF
        assert cnf is not None
    
    def test_transformer_constant_input(self):
        """Test transforming circuit with constants."""
        circuit = BooleanCircuit()
        circuit.add_constant_node("const1", True)
        circuit.add_output("out", "const1")
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        # Constant circuit should produce CNF
        assert cnf is not None


class TestCircuitToCNFNand:
    """Tests for NAND gate transformation."""
    
    def test_transformer_nand_gate(self):
        """Test transforming NAND gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_nand(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() > 0
        assert cnf is not None
    
    def test_transformer_nand_complex(self):
        """Test transforming complex NAND circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        nand1 = builder.add_nand(x, y)
        nand2 = builder.add_nand(nand1, z)
        builder.add_output(nand2, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0


class TestCircuitToCNFNor:
    """Tests for NOR gate transformation."""
    
    def test_transformer_nor_gate(self):
        """Test transforming NOR gate."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_nor(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() > 0
        assert cnf is not None
    
    def test_transformer_nor_complex(self):
        """Test transforming complex NOR circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        nor1 = builder.add_nor(x, y)
        nor2 = builder.add_nor(nor1, z)
        builder.add_output(nor2, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0


class TestCircuitToCNFAuxVariables:
    """Tests for auxiliary variable handling."""
    
    def test_aux_var_unique_generation(self):
        """Test auxiliary variable uniqueness."""
        transformer = CircuitToCNFTransformer()
        
        vars = set()
        for i in range(10):
            var = transformer._get_aux_var()
            vars.add(var)
        
        # All should be unique
        assert len(vars) == 10
    
    def test_aux_var_custom_prefix(self):
        """Test custom prefix for auxiliary variables."""
        transformer = CircuitToCNFTransformer()
        
        var1 = transformer._get_aux_var("_x")
        var2 = transformer._get_aux_var("_y")
        
        assert "_x" in var1
        assert "_y" in var2
    
    def test_aux_var_counter_increment(self):
        """Test auxiliary variable counter increments."""
        transformer = CircuitToCNFTransformer()
        
        assert transformer.aux_var_counter == 0
        transformer._get_aux_var()
        assert transformer.aux_var_counter == 1
        transformer._get_aux_var()
        assert transformer.aux_var_counter == 2


class TestCircuitToCNFMixedGates:
    """Tests for circuits with mixed gate types."""
    
    def test_transformer_and_or_circuit(self):
        """Test transforming circuit with AND and OR gates."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        
        and_result = builder.add_and(x, y)
        or_result = builder.add_or(and_result, z)
        builder.add_output(or_result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0
    
    def test_transformer_all_gates_circuit(self):
        """Test transforming circuit with all gate types."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        not_x = builder.add_not(x)
        and_xy = builder.add_and(x, y)
        or_xy = builder.add_or(x, y)
        xor_xy = builder.add_xor(x, y)
        nand_xy = builder.add_nand(x, y)
        nor_xy = builder.add_nor(x, y)
        
        # Combine some results
        final = builder.add_or(and_xy, not_x)
        builder.add_output(final, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0
    
    def test_transformer_deep_nesting(self):
        """Test transforming deeply nested circuit."""
        builder = CircuitBuilder()
        
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        # Build deep nesting: ((x AND y) OR y) AND (x OR (x AND y))
        and1 = builder.add_and(x, y)
        or1 = builder.add_or(and1, y)
        
        or2 = builder.add_or(x, and1)
        final = builder.add_and(or1, or2)
        
        builder.add_output(final, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0


class TestCircuitToCNFEdgeCases:
    """Tests for edge cases in circuit-to-CNF transformation."""
    
    def test_transformer_single_input_circuit(self):
        """Test transforming single-input circuit."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
    
    def test_transformer_true_constant(self):
        """Test transforming circuit with constant TRUE."""
        circuit = BooleanCircuit()
        circuit.add_constant_node("const", True)
        circuit.add_output("out", "const")
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
    
    def test_transformer_false_constant(self):
        """Test transforming circuit with constant FALSE."""
        circuit = BooleanCircuit()
        circuit.add_constant_node("const", False)
        circuit.add_output("out", "const")
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
    
    def test_transformer_not_of_not(self):
        """Test transforming double negation."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        not_x = builder.add_not(x)
        not_not_x = builder.add_not(not_x)
        builder.add_output(not_not_x, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() >= 4  # At least 2 per NOT gate
    
    def test_transformer_multi_input_and(self):
        """Test transforming AND with more than 2 inputs."""
        builder = CircuitBuilder()
        
        x = builder.add_input("x")
        y = builder.add_input("y")
        z = builder.add_input("z")
        w = builder.add_input("w")
        
        and1 = builder.add_and(x, y)
        and2 = builder.add_and(and1, z)
        and3 = builder.add_and(and2, w)
        
        builder.add_output(and3, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0
    
    def test_transformer_variables_extracted(self):
        """Test that variables are properly extracted from CNF."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_or(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        variables = cnf.get_variables()
        assert len(variables) > 0
        # Should have at least the input variables
        assert "x" in variables or any("x" in str(v) for v in variables)


class TestCircuitToCNFConsistency:
    """Tests for consistency in circuit-to-CNF transformation."""
    
    def test_transformer_idempotent_ids(self):
        """Test that transformation with same circuit uses consistent IDs."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer1 = CircuitToCNFTransformer()
        transformer2 = CircuitToCNFTransformer()
        
        cnf1 = transformer1.transform(circuit)
        cnf2 = transformer2.transform(circuit)
        
        # Both should produce same number of clauses and variables
        assert cnf1.get_num_clauses() == cnf2.get_num_clauses()
    
    def test_transformer_new_instance_resets_counter(self):
        """Test that new transformer instance resets counter."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        builder.add_output(x, "out")
        circuit = builder.build()
        
        transformer1 = CircuitToCNFTransformer()
        transformer1.transform(circuit)
        first_counter = transformer1.aux_var_counter
        
        transformer2 = CircuitToCNFTransformer()
        assert transformer2.aux_var_counter == 0
        assert transformer2.aux_var_counter != first_counter


class TestCircuitToCNFComparison:
    """Tests for circuit equivalence and comparison methods."""
    
    def test_build_equivalence_cnf_same_circuits(self):
        """Test equivalence CNF for identical circuits."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        equiv_cnf = transformer.build_equivalence_cnf(circuit, circuit)
        
        assert isinstance(equiv_cnf, CNFFormula)
        assert equiv_cnf.get_num_clauses() > 0
    
    def test_build_equivalence_cnf_different_circuits(self):
        """Test equivalence CNF for different circuits."""
        builder1 = CircuitBuilder()
        x1 = builder1.add_input("x")
        y1 = builder1.add_input("y")
        result1 = builder1.add_and(x1, y1)
        builder1.add_output(result1, "out")
        circuit1 = builder1.build()
        
        builder2 = CircuitBuilder()
        x2 = builder2.add_input("x")
        y2 = builder2.add_input("y")
        result2 = builder2.add_or(x2, y2)
        builder2.add_output(result2, "out")
        circuit2 = builder2.build()
        
        transformer = CircuitToCNFTransformer()
        equiv_cnf = transformer.build_equivalence_cnf(circuit1, circuit2)
        
        assert isinstance(equiv_cnf, CNFFormula)


class TestCircuitToCNFLearning:
    """Tests for learning-related CNF building methods."""
    
    def test_build_h_satisfying_cnf(self):
        """Test building h-satisfying CNF."""
        from src.boolean_functions import DNFFormula, Term
        
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        target_builder = CircuitBuilder()
        tx = target_builder.add_input("x")
        ty = target_builder.add_input("y")
        target_result = target_builder.add_or(tx, ty)
        target_builder.add_output(target_result, "out")
        target = target_builder.build()
        
        hyp = DNFFormula()
        transformer = CircuitToCNFTransformer()
        h_sat_cnf = transformer.build_h_satisfying_cnf(hyp, circuit, target)
        
        assert isinstance(h_sat_cnf, CNFFormula)
        assert h_sat_cnf.get_num_clauses() > 0
    
    def test_build_removable_cnf(self):
        """Test building CNF for variable removability check."""
        from src.boolean_functions import Term
        from src.cnf import Literal
        
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        target_builder = CircuitBuilder()
        tx = target_builder.add_input("x")
        ty = target_builder.add_input("y")
        target_result = target_builder.add_or(tx, ty)
        target_builder.add_output(target_result, "out")
        target = target_builder.build()
        
        term = Term({Literal("x", False), Literal("y", False)})
        transformer = CircuitToCNFTransformer()
        removable_cnf = transformer.build_removable_cnf(term, "x", circuit, target)
        
        assert isinstance(removable_cnf, CNFFormula)
        assert removable_cnf.get_num_clauses() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
