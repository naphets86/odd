"""
Tests for boolean_functions module (boolean_functions.py)
"""

import pytest
from src.boolean_functions import BooleanFunction, DNFFormula, Term
from src.cnf import Literal


class TestBooleanFunction:
    """Test BooleanFunction class."""
    
    def test_and_function(self):
        """Test AND function."""
        func = BooleanFunction.AND(2)
        assert func.evaluate({"x1": True, "x2": True}) == True
        assert func.evaluate({"x1": True, "x2": False}) == False
    
    def test_or_function(self):
        """Test OR function."""
        func = BooleanFunction.OR(2)
        assert func.evaluate({"x1": True, "x2": False}) == True
        assert func.evaluate({"x1": False, "x2": False}) == False
    
    def test_xor_function(self):
        """Test XOR function."""
        func = BooleanFunction.XOR(2)
        assert func.evaluate({"x1": True, "x2": False}) == True
        assert func.evaluate({"x1": True, "x2": True}) == False
    
    def test_nand_function(self):
        """Test NAND function."""
        func = BooleanFunction.NAND(2)
        assert func.evaluate({"x1": True, "x2": True}) == False
        assert func.evaluate({"x1": False, "x2": True}) == True
    
    def test_nor_function(self):
        """Test NOR function."""
        func = BooleanFunction.NOR(2)
        assert func.evaluate({"x1": False, "x2": False}) == True
        assert func.evaluate({"x1": True, "x2": False}) == False
    
    def test_function_to_dnf(self):
        """Test function to DNF conversion."""
        func = BooleanFunction.AND(2)
        dnf = func.to_dnf()
        assert dnf is not None
        assert len(dnf.terms) > 0
    
    def test_function_to_cnf(self):
        """Test function to CNF conversion."""
        func = BooleanFunction.OR(2)
        cnf = func.to_cnf()
        assert cnf is not None
        assert cnf.get_num_clauses() > 0


class TestBooleanFunctionAdvanced:
    """Advanced tests for Boolean functions."""
    
    def test_function_evaluate_from_bits(self):
        """Test evaluate from bit representation."""
        func = BooleanFunction.AND(2)
        # AND truth table: 0,0,0,1 for bit patterns 00,01,10,11
        assert func.evaluate_from_bits(3) == True  # 11
        assert func.evaluate_from_bits(0) == False  # 00
    
    def test_function_truth_table_size(self):
        """Test truth table size."""
        func = BooleanFunction.AND(3)
        assert len(func.truth_table) == 2**3
    
    def test_function_invalid_truth_table(self):
        """Test invalid truth table."""
        with pytest.raises(ValueError):
            BooleanFunction("Bad", 2, [True, False])  # wrong size
    
    def test_prime_implicant_creation(self):
        """Test prime implicant creation."""
        from src.boolean_functions import PrimeImplicant
        term = Term({Literal("x", False)})
        implicant = PrimeImplicant(term)
        assert implicant is not None


class TestDNFFormula:
    """Test DNF formula class."""
    
    def test_dnf_creation(self):
        """Test creating DNF formulas."""
        dnf = DNFFormula()
        term = Term({Literal("x", False)})
        dnf.add_term(term)
        assert len(dnf) == 1
    
    def test_dnf_evaluation(self):
        """Test evaluating DNF formulas."""
        dnf = DNFFormula()
        term = Term({Literal("x", False)})
        dnf.add_term(term)
        
        assert dnf.evaluate({"x": True}) == True
        assert dnf.evaluate({"x": False}) == False


class TestTermAndDNFAdvanced:
    """Advanced tests for Term and DNF formulas."""
    
    def test_term_variables(self):
        """Test extracting variables from term."""
        term = Term({Literal("x", False), Literal("y", True)})
        variables = term.get_variables()
        assert "x" in variables
        assert "y" in variables
    
    def test_term_implies(self):
        """Test term implication."""
        term1 = Term({Literal("x", False), Literal("y", False)})
        term2 = Term({Literal("x", False)})
        assert term1.implies(term2)
    
    def test_term_complementary(self):
        """Test complementary pair detection."""
        term = Term({Literal("x", False), Literal("x", True)})
        assert term.contains_complementary_pair()
    
    def test_dnf_monotone(self):
        """Test monotone DNF."""
        dnf = DNFFormula(is_monotone=True)
        assert dnf.is_monotone == True
    
    def test_dnf_is_satisfiable(self):
        """Test DNF satisfiability."""
        dnf = DNFFormula()
        term = Term({Literal("x", False)})
        dnf.add_term(term)
        assert dnf.is_satisfiable() == True
    
    def test_dnf_copy(self):
        """Test copying DNF formula."""
        dnf1 = DNFFormula()
        dnf1.add_term(Term({Literal("x", False)}))
        dnf2 = dnf1.copy()
        dnf2.add_term(Term({Literal("y", False)}))
        assert len(dnf1.terms) == 1
        assert len(dnf2.terms) == 2


class TestCircuitToCNFEquivalence:
    """Tests for circuit-to-CNF equivalence checking."""
    
    def test_function_to_cnf_complete(self):
        """Test complete function-to-CNF conversion."""
        # AND function
        func = BooleanFunction.AND(2)
        cnf = func.to_cnf()
        
        assert cnf is not None
        assert cnf.get_num_clauses() > 0
    
    def test_function_to_dnf_complete(self):
        """Test complete function-to-DNF conversion."""
        # OR function
        func = BooleanFunction.OR(2)
        dnf = func.to_dnf()
        
        assert dnf is not None
        assert len(dnf.terms) > 0
    
    def test_xor_function_conversion(self):
        """Test XOR function CNF/DNF conversion."""
        func = BooleanFunction.XOR(2)
        
        cnf = func.to_cnf()
        dnf = func.to_dnf()
        
        assert cnf is not None
        assert dnf is not None
        assert len(dnf.terms) == 2  # XOR has 2 satisfying terms


class TestTermImplies:
    """Test Term.implies() method."""
    
    def test_term_implies_same_term(self):
        """Test that a term implies itself."""
        lit_x = Literal("x", False)
        lit_y = Literal("y", False)
        term = Term({lit_x, lit_y})
        
        assert term.implies(term) == True
    
    def test_term_implies_subset(self):
        """Test that term with more literals implies term with fewer."""
        lit_x = Literal("x", False)
        lit_y = Literal("y", False)
        term1 = Term({lit_x, lit_y})
        term2 = Term({lit_x})
        
        assert term1.implies(term2) == True
        assert term2.implies(term1) == False
    
    def test_term_implies_empty(self):
        """Test that any term implies empty term."""
        lit_x = Literal("x", False)
        term = Term({lit_x})
        empty_term = Term()
        
        assert term.implies(empty_term) == True
        assert empty_term.implies(term) == False
    
    def test_term_implies_disjoint(self):
        """Test implies with disjoint terms."""
        lit_x = Literal("x", False)
        lit_y = Literal("y", False)
        term1 = Term({lit_x})
        term2 = Term({lit_y})
        
        assert term1.implies(term2) == False
        assert term2.implies(term1) == False


class TestPrimeImplicant:
    """Test PrimeImplicant class."""
    
    def test_prime_implicant_creation(self):
        """Test creating a prime implicant."""
        from src.boolean_functions import PrimeImplicant
        
        lit_x = Literal("x", False)
        term = Term({lit_x})
        pi = PrimeImplicant(term, is_prime=True)
        
        assert pi.is_prime == True
        assert str(pi) == str(term)
    
    def test_prime_implicant_non_prime(self):
        """Test creating a non-prime implicant."""
        from src.boolean_functions import PrimeImplicant
        
        lit_x = Literal("x", False)
        term = Term({lit_x})
        pi = PrimeImplicant(term, is_prime=False)
        
        assert pi.is_prime == False
    
    def test_prime_implicant_of_empty_set(self):
        """Test is_prime_implicant_of with empty assignment set."""
        from src.boolean_functions import PrimeImplicant
        
        lit_x = Literal("x", False)
        term = Term({lit_x})
        pi = PrimeImplicant(term, is_prime=True)
        
        result = pi.is_prime_implicant_of([])
        assert result == True
    
    def test_prime_implicant_of_assignments(self):
        """Test is_prime_implicant_of with specific assignments."""
        from src.boolean_functions import PrimeImplicant
        
        lit_x = Literal("x", False)
        term = Term({lit_x})
        pi = PrimeImplicant(term, is_prime=True)
        
        assignments = [
            {"x": True, "y": True},
            {"x": True, "y": False}
        ]
        result = pi.is_prime_implicant_of(assignments)
        assert result == True


class TestDNFFormulaEdgeCases:
    """Test DNF formula edge cases."""
    
    def test_dnf_empty_evaluation(self):
        """Test evaluating empty DNF formula."""
        dnf = DNFFormula()
        
        assert dnf.is_empty() == True
        assert dnf.evaluate({"x": True}) == False
    
    def test_dnf_single_term(self):
        """Test DNF with single term."""
        lit_x = Literal("x", False)
        term = Term({lit_x})
        dnf = DNFFormula([term])
        
        assert len(dnf) == 1
        assert dnf.evaluate({"x": True}) == True
        assert dnf.evaluate({"x": False}) == False
    
    def test_dnf_add_term(self):
        """Test adding terms to DNF."""
        dnf = DNFFormula()
        lit_x = Literal("x", False)
        term1 = Term({lit_x})
        
        dnf.add_term(term1)
        assert len(dnf) == 1
        assert "x" in dnf.variables
    
    def test_dnf_monotone_flag(self):
        """Test monotone DNF flag."""
        lit_x = Literal("x", False)
        term = Term({lit_x})
        
        dnf_regular = DNFFormula([term], is_monotone=False)
        dnf_monotone = DNFFormula([term], is_monotone=True)
        
        assert dnf_regular.is_monotone == False
        assert dnf_monotone.is_monotone == True
    
    def test_dnf_copy(self):
        """Test copying DNF formula."""
        lit_x = Literal("x", False)
        lit_y = Literal("y", False)
        term1 = Term({lit_x})
        term2 = Term({lit_y})
        dnf = DNFFormula([term1, term2])
        
        dnf_copy = dnf.copy()
        assert len(dnf_copy) == len(dnf)
        assert dnf_copy.variables == dnf.variables
        assert dnf_copy is not dnf
    
    def test_dnf_two_satisfying_terms(self):
        """Test DNF with two satisfying terms."""
        lit_x = Literal("x", False)
        lit_y = Literal("y", False)
        term1 = Term({lit_x})
        term2 = Term({lit_y})
        dnf = DNFFormula([term1, term2])
        
        # Should be true when x=True OR y=True
        assert dnf.evaluate({"x": True, "y": False}) == True
        assert dnf.evaluate({"x": False, "y": True}) == True
        assert dnf.evaluate({"x": False, "y": False}) == False
        assert dnf.evaluate({"x": True, "y": True}) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
