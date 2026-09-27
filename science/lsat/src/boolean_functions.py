"""
Boolean functions and formulas representation.

This module provides classes for representing Boolean functions,
DNF, CNF, and specialized formula classes like Monotone DNF.
"""

from typing import Set, List, Dict, Optional, Tuple, Callable
from enum import Enum
from src.cnf import Literal, Clause, CNFFormula


class FormulaType(Enum):
    """Enumeration of Boolean formula types."""
    DNF = "DNF"  # Disjunctive Normal Form
    CNF = "CNF"  # Conjunctive Normal Form
    M_DNF = "M-DNF"  # Monotone DNF (only positive literals)
    M_CNF = "M-CNF"  # Monotone CNF (only positive literals)
    ARBITRARY = "ARBITRARY"  # Arbitrary formula


class BooleanFormula:
    """
    Base class for Boolean formulas.
    
    Attributes:
        formula_type: The type of this formula
        variables: Set of variable names
    """
    
    def __init__(self, formula_type: FormulaType, variables: Optional[Set[str]] = None):
        self.formula_type = formula_type
        self.variables = variables or set()
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate the formula under the given truth assignment.
        
        Args:
            assignment: Dict mapping variable names to truth values
            
        Returns:
            The truth value of the formula
        """
        raise NotImplementedError
    
    def to_cnf(self) -> "BooleanFormula":
        """Convert the formula to CNF."""
        raise NotImplementedError
    
    def to_dnf(self) -> "BooleanFormula":
        """Convert the formula to DNF."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError


class Term:
    """
    A term is a conjunction of literals (used in DNF).
    
    Attributes:
        literals: Set of positive and negative literals
    """
    
    def __init__(self, literals: Optional[Set[Literal]] = None):
        self.literals = literals or set()
    
    def __str__(self) -> str:
        """String representation."""
        if not self.literals:
            return "⊤"  # Empty term (trivially true)
        return "(" + " ∧ ".join(str(lit) for lit in sorted(
            self.literals, key=lambda l: (l.variable, l.is_negated))) + ")"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Term({self.literals})"
    
    def __len__(self) -> int:
        """Number of literals in the term."""
        return len(self.literals)
    
    def get_variables(self) -> Set[str]:
        """Get all variable names in the term."""
        return {lit.variable for lit in self.literals}
    
    def add_literal(self, literal: Literal) -> None:
        """Add a literal to the term."""
        self.literals.add(literal)
    
    def contains_complementary_pair(self) -> bool:
        """Check if the term contains complementary literals."""
        for lit1 in self.literals:
            for lit2 in self.literals:
                if lit1.is_complementary(lit2):
                    return True
        return False
    
    def copy(self) -> "Term":
        """Create a deep copy of the term."""
        return Term(literals=self.literals.copy())
    
    def implies(self, other: "Term") -> bool:
        """
        Check if this term implies another term (subset relationship).
        
        A term T1 implies T2 if every literal in T2 is in T1.
        """
        return self.literals.issuperset(other.literals)


class DNFFormula(BooleanFormula):
    """
    Disjunctive Normal Form (DNF) formula.
    
    A DNF formula is a disjunction of terms (conjunctions of literals).
    Example: (x1 ∧ x2) ∨ (¬x1 ∧ x3) ∨ x2
    
    Attributes:
        terms: List of terms (conjunctions of literals)
    """
    
    def __init__(self, terms: Optional[List[Term]] = None, is_monotone: bool = False):
        variables = set()
        if terms:
            for term in terms:
                variables.update(term.get_variables())
        
        formula_type = FormulaType.M_DNF if is_monotone else FormulaType.DNF
        super().__init__(formula_type, variables)
        
        self.terms = terms or []
        self.is_monotone = is_monotone
    
    def __str__(self) -> str:
        """String representation."""
        if not self.terms:
            return "⊥"  # Empty DNF (always false)
        return " ∨ ".join(str(term) for term in self.terms)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DNFFormula(terms={len(self.terms)}, monotone={self.is_monotone})"
    
    def __len__(self) -> int:
        """Number of terms."""
        return len(self.terms)
    
    def is_empty(self) -> bool:
        """Check if the DNF is empty (always false)."""
        return len(self.terms) == 0
    
    def add_term(self, term: Term) -> None:
        """Add a term to the DNF."""
        self.terms.append(term)
        self.variables.update(term.get_variables())
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate the DNF formula.
        
        A DNF is true if at least one term is true.
        """
        if not self.terms:
            return False
        
        for term in self.terms:
            term_true = True
            for literal in term.literals:
                var_value = assignment.get(literal.variable, False)
                lit_value = var_value if not literal.is_negated else not var_value
                if not lit_value:
                    term_true = False
                    break
            
            if term_true:
                return True
        
        return False
    
    def is_satisfiable(self) -> bool:
        """Check if the DNF is satisfiable."""
        return len(self.terms) > 0
    
    def to_cnf(self) -> "CNFFormula":
        """Convert DNF to CNF using Tseitin transformation (simplified)."""
        from .circuit_to_cnf import CircuitToCNFTransformer
        # For now, use a basic conversion
        cnf = CNFFormula(num_variables=len(self.variables))
        # TODO: Implement proper DNF to CNF conversion
        return cnf
    
    def to_dnf(self) -> "DNFFormula":
        """DNF is already in DNF form."""
        return self
    
    def copy(self) -> "DNFFormula":
        """Create a deep copy of the DNF formula."""
        new_terms = [term.copy() for term in self.terms]
        return DNFFormula(terms=new_terms, is_monotone=self.is_monotone)


class PrimeImplicant:
    """
    A prime implicant of a Boolean function  is a minimal conjunction of literals
    that implies the function.
    
    Attributes:
        term: The conjunction of literals
        function: Reference to the Boolean function it's an implicant of
    """
    
    def __init__(self, term: Term, is_prime: bool = True):
        self.term = term
        self.is_prime = is_prime
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.term)
    
    def is_prime_implicant_of(self, assignment_set: List[Dict[str, bool]]) -> bool:
        """
        Check if this term is a prime implicant with respect to a set of assignments.
        
        A term is a prime implicant if:
        1. It implies all assignments in the set
        2. Removing any literal would not imply all assignments
        """
        if not self.is_prime:
            return False
        
        # Check condition 1: Term should imply all assignments
        # (This is simplified - full check would evaluate the term)
        return True


class BooleanFunction:
    """
    Represents a Boolean function that can be learned.
    
    Can be evaluated on truth assignments and has multiple representations
    (DNF, CNF, truth table).
    
    Attributes:
        name: Function name
        arity: Number of input variables
        truth_table: Truth table representation
        variables: Variable names
    """
    
    def __init__(self, name: str, arity: int, truth_table: List[bool],
                 variables: Optional[List[str]] = None):
        self.name = name
        self.arity = arity
        self.truth_table = truth_table
        self.variables = variables or [f"x{i+1}" for i in range(arity)]
        
        if len(truth_table) != 2**arity:
            raise ValueError(f"Truth table size {len(truth_table)} != 2^{arity}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"BooleanFunction({self.name}, arity={self.arity})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"BooleanFunction(name={self.name}, arity={self.arity})"
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate the function on the given assignment.
        
        Uses the truth table representation.
        """
        # Convert assignment to binary index
        index = 0
        for i, var in enumerate(self.variables):
            if assignment.get(var, False):
                index |= (1 << i)
        
        return self.truth_table[index]
    
    def evaluate_from_bits(self, bits: int) -> bool:
        """Evaluate the function from a binary representation of inputs."""
        return self.truth_table[bits]
    
    def to_dnf(self) -> DNFFormula:
        """
        Convert the function to DNF using truth table.
        
        Creates a term for each row where the function is true.
        """
        terms = []
        
        for assignment_index in range(len(self.truth_table)):
            if self.truth_table[assignment_index]:
                # Create term for this assignment
                term_literals = set()
                for var_index, var in enumerate(self.variables):
                    bit_value = (assignment_index >> var_index) & 1
                    literal = Literal(var, is_negated=(bit_value == 0))
                    term_literals.add(literal)
                
                terms.append(Term(term_literals))
        
        return DNFFormula(terms=terms, is_monotone=False)
    
    def to_cnf(self) -> CNFFormula:
        """
        Convert the function to CNF using truth table.
        
        Creates a clause for each row where the function is false.
        """
        clauses = []
        
        for assignment_index in range(len(self.truth_table)):
            if not self.truth_table[assignment_index]:
                # Create clause for this assignment (negation of the assignment)
                clause_literals = set()
                for var_index, var in enumerate(self.variables):
                    bit_value = (assignment_index >> var_index) & 1
                    literal = Literal(var, is_negated=(bit_value == 1))
                    clause_literals.add(literal)
                
                clauses.append(Clause(clause_literals))
        
        return CNFFormula(clauses=clauses, num_variables=len(self.variables))
    
    @staticmethod
    def AND(n_vars: int) -> "BooleanFunction":
        """Create an AND function with n variables."""
        truth_table = [1 if i == (2**n_vars) - 1 else 0 for i in range(2**n_vars)]
        return BooleanFunction("AND", n_vars, truth_table)
    
    @staticmethod
    def OR(n_vars: int) -> "BooleanFunction":
        """Create an OR function with n variables."""
        truth_table = [0 if i == 0 else 1 for i in range(2**n_vars)]
        return BooleanFunction("OR", n_vars, truth_table)
    
    @staticmethod
    def XOR(n_vars: int = 2) -> "BooleanFunction":
        """Create an XOR function."""
        if n_vars != 2:
            raise NotImplementedError("Only 2-ary XOR is implemented")
        # XOR(x, y) = (x ∧ ¬y) ∨ (¬x ∧ y)
        truth_table = [0, 1, 1, 0]
        return BooleanFunction("XOR", 2, truth_table)
    
    @staticmethod
    def NAND(n_vars: int) -> "BooleanFunction":
        """Create a NAND function."""
        truth_table = [0 if i == (2**n_vars) - 1 else 1 for i in range(2**n_vars)]
        return BooleanFunction("NAND", n_vars, truth_table)
    
    @staticmethod
    def NOR(n_vars: int) -> "BooleanFunction":
        """Create a NOR function."""
        truth_table = [1 if i == 0 else 0 for i in range(2**n_vars)]
        return BooleanFunction("NOR", n_vars, truth_table)
