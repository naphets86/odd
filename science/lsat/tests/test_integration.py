"""
Tests for integration scenarios, error handling, and boundaries
"""

import pytest
from src.cnf import CNFFormula, Literal, Clause, SATResult
from src.circuit import BooleanCircuit, CircuitBuilder
from src.boolean_functions import BooleanFunction, DNFFormula, Term
from src.circuit_to_cnf import CircuitToCNFTransformer
from src.tool import generate_all_assignments


class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_circuit_to_cnf_and_evaluation(self):
        """Test circuit-to-CNF workflow."""
        builder = CircuitBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        result = builder.add_and(x, y)
        builder.add_output(result, "out")
        circuit = builder.build()
        
        transformer = CircuitToCNFTransformer()
        cnf = transformer.transform(circuit)
        
        assert cnf.get_num_clauses() > 0
    
    def test_sat_result_str(self):
        """Test SAT result string representation."""
        result_sat = SATResult(satisfiable=True, assignment={"x": True, "y": False})
        str_sat = str(result_sat)
        assert "SAT" in str_sat
        
        result_unsat = SATResult(satisfiable=False)
        str_unsat = str(result_unsat)
        assert "UNSAT" in str_unsat
    
    def test_sat_result_repr(self):
        """Test SAT result representation."""
        result = SATResult(satisfiable=True, assignment={"x": True})
        repr_str = repr(result)
        assert "SATResult" in repr_str
    
    def test_full_circuit_workflow(self):
        """Test complete circuit creation and evaluation workflow."""
        builder = CircuitBuilder("XOR")
        x = builder.add_input("x")
        y = builder.add_input("y")
        not_x = builder.add_not(x)
        not_y = builder.add_not(y)
        and1 = builder.add_and(x, not_y)
        and2 = builder.add_and(not_x, y)
        xor = builder.add_or(and1, and2)
        builder.add_output(xor, "out")
        
        circuit = builder.build()
        
        # Test all 4 truth assignments
        test_cases = [
            ({"x": False, "y": False}, False),
            ({"x": False, "y": True}, True),
            ({"x": True, "y": False}, True),
            ({"x": True, "y": True}, False),
        ]
        
        for assignment, expected in test_cases:
            result = circuit.evaluate(assignment)["out"]
            assert result == expected
    
    def test_dnf_to_evaluation(self):
        """Test DNF formula evaluation workflow."""
        dnf = DNFFormula()
        
        # Add terms: (x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) [XOR]
        term1 = Term({Literal("x1", False), Literal("x2", True)})
        term2 = Term({Literal("x1", True), Literal("x2", False)})
        
        dnf.add_term(term1)
        dnf.add_term(term2)
        
        # Test evaluation
        assert dnf.evaluate({"x1": False, "x2": False}) == False
        assert dnf.evaluate({"x1": False, "x2": True}) == True
        assert dnf.evaluate({"x1": True, "x2": False}) == True
        assert dnf.evaluate({"x1": True, "x2": True}) == False


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_circuit_missing_input_node(self):
        """Test circuit with missing input node."""
        circuit = BooleanCircuit()
        
        with pytest.raises(ValueError):
            circuit.add_gate("g1", "AND", ["missing_node"])
    
    def test_unknown_gate_type(self):
        """Test circuit with unknown gate type."""
        circuit = BooleanCircuit()
        circuit.add_input("x")
        
        with pytest.raises(ValueError):
            circuit.add_gate("g1", "UNKNOWN_GATE", ["x"])
    
    def test_function_invalid_arity(self):
        """Test creating functions with invalid arity."""
        # This should not raise, as it's valid
        func = BooleanFunction("Test", 2, [False, False, False, False])
        assert func.arity == 2
    
    def test_circuit_output_single_output_requirement(self):
        """Test circuit output single output requirement."""
        circuit = BooleanCircuit()
        
        with pytest.raises(ValueError):
            circuit.get_output_value({})  # No outputs defined


class TestBoundaryConditions:
    """Tests for boundary conditions and special cases."""
    
    def test_empty_assignments(self):
        """Test with empty assignments."""
        assignments = generate_all_assignments([])
        assert len(assignments) == 1
        assert assignments[0] == {}
    
    def test_single_variable(self):
        """Test functions with single variable."""
        func = BooleanFunction.NAND(1)
        
        assignment1 = {"x1": False}
        assignment2 = {"x1": True}
        
        assert func.evaluate(assignment1) == True
        assert func.evaluate(assignment2) == False
    
    def test_large_circuit(self):
        """Test creating larger circuit."""
        builder = CircuitBuilder("Large")
        
        # Add 10 inputs
        inputs = [builder.add_input(f"x{i}") for i in range(10)]
        
        # Create AND chain
        current = inputs[0]
        for inp in inputs[1:]:
            current = builder.add_and(current, inp)
        
        builder.add_output(current, "out")
        circuit = builder.build()
        
        assert len(circuit) == 10 + 9 + 1  # inputs + AND gates + output
    
    def test_clause_with_many_literals(self):
        """Test clause with many literals."""
        literals = {Literal(f"x{i}", False) for i in range(20)}
        clause = Clause(literals)
        
        assert len(clause) == 20


class TestReprAndStr:
    """Tests for string representations."""
    
    def test_literal_repr(self):
        """Test literal repr."""
        lit = Literal("x", False)
        assert "Literal" in repr(lit)
    
    def test_clause_repr(self):
        """Test clause repr."""
        clause = Clause({Literal("x", False)})
        assert "Clause" in repr(clause)
    
    def test_cnf_repr(self):
        """Test CNF repr."""
        cnf = CNFFormula(num_variables=2)
        cnf.add_clause(Clause({Literal("x", False)}))
        assert "CNFFormula" in repr(cnf)
    
    def test_circuit_repr(self):
        """Test circuit repr."""
        circuit = BooleanCircuit("Test")
        assert "BooleanCircuit" in repr(circuit)
    
    def test_dnf_repr(self):
        """Test DNF repr."""
        dnf = DNFFormula()
        assert "DNFFormula" in repr(dnf)
    
    def test_term_repr(self):
        """Test term repr."""
        term = Term({Literal("x", False)})
        assert "Term" in repr(term)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
