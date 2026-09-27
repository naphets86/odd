"""
CNF (Conjunctive Normal Form) representation and manipulation.

This module provides classes for representing CNF formulas with literals,
clauses, and formula operations.
"""

from typing import Set, List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Literal:
    """
    A literal is either a variable or its negation.
    
    Attributes:
        variable: The variable name (e.g., "x1")
        is_negated: True if the literal is negated
    """
    variable: str
    is_negated: bool = False
    
    def __str__(self) -> str:
        """String representation of the literal."""
        return f"¬{self.variable}" if self.is_negated else self.variable
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Literal({self.variable}, negated={self.is_negated})"
    
    def negate(self) -> "Literal":
        """Return the negation of this literal."""
        return Literal(self.variable, not self.is_negated)
    
    def is_complementary(self, other: "Literal") -> bool:
        """Check if two literals are complementary (x and ¬x)."""
        return (self.variable == other.variable and 
                self.is_negated != other.is_negated)


@dataclass
class Clause:
    """
    A clause is a disjunction of literals.
    
    Attributes:
        literals: Set of literals in the clause
    """
    literals: Set[Literal] = field(default_factory=set)
    
    def __str__(self) -> str:
        """String representation of the clause."""
        if not self.literals:
            return "⊥"  # Empty clause (unsatisfiable)
        return "(" + " ∨ ".join(str(lit) for lit in sorted(
            self.literals, key=lambda l: (l.variable, l.is_negated))) + ")"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Clause({self.literals})"
    
    def __len__(self) -> int:
        """Number of literals in the clause."""
        return len(self.literals)
    
    def is_empty(self) -> bool:
        """Check if the clause is empty (unsatisfiable)."""
        return len(self.literals) == 0
    
    def is_unit(self) -> bool:
        """Check if the clause is a unit clause (single literal)."""
        return len(self.literals) == 1
    
    def get_unit_literal(self) -> Optional[Literal]:
        """Return the literal if this is a unit clause, None otherwise."""
        if self.is_unit():
            return next(iter(self.literals))
        return None
    
    def contains_complementary_pair(self) -> bool:
        """Check if the clause contains complementary literals (always true)."""
        literals_list = list(self.literals)
        for i, lit1 in enumerate(literals_list):
            for lit2 in literals_list[i+1:]:
                if lit1.is_complementary(lit2):
                    return True
        return False
    
    def add_literal(self, literal: Literal) -> None:
        """Add a literal to the clause."""
        self.literals.add(literal)
    
    def remove_literal(self, literal: Literal) -> None:
        """Remove a literal from the clause if present."""
        self.literals.discard(literal)
    
    def copy(self) -> "Clause":
        """Create a deep copy of the clause."""
        return Clause(literals=self.literals.copy())


@dataclass
class CNFFormula:
    """
    A CNF formula is a conjunction of clauses.
    
    Attributes:
        clauses: List of clauses in the formula
        num_variables: Number of variables (optional, for reference)
    """
    clauses: List[Clause] = field(default_factory=list)
    num_variables: Optional[int] = None
    
    def __str__(self) -> str:
        """String representation of the CNF formula."""
        if not self.clauses:
            return "⊤"  # Empty formula (trivially true)
        return " ∧ ".join(str(clause) for clause in self.clauses)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CNFFormula(clauses={len(self.clauses)}, vars={self.num_variables})"
    
    def __len__(self) -> int:
        """Number of clauses in the formula."""
        return len(self.clauses)
    
    def __iter__(self):
        """Iterate over clauses."""
        return iter(self.clauses)
    
    def is_empty(self) -> bool:
        """Check if the formula is empty (trivially true)."""
        return len(self.clauses) == 0
    
    def has_empty_clause(self) -> bool:
        """Check if any clause is empty (formula is unsatisfiable)."""
        return any(clause.is_empty() for clause in self.clauses)
    
    def add_clause(self, clause: Clause) -> None:
        """Add a clause to the formula."""
        self.clauses.append(clause)
    
    def add_literal_clause(self, literal: Literal) -> None:
        """Add a unit clause with a single literal."""
        clause = Clause(literals={literal})
        self.add_clause(clause)
    
    def remove_clause(self, clause_idx: int) -> None:
        """Remove a clause at the given index."""
        if 0 <= clause_idx < len(self.clauses):
            self.clauses.pop(clause_idx)
    
    def copy(self) -> "CNFFormula":
        """Create a deep copy of the formula."""
        new_clauses = [clause.copy() for clause in self.clauses]
        return CNFFormula(clauses=new_clauses, num_variables=self.num_variables)
    
    def get_variables(self) -> Set[str]:
        """Get all variable names in the formula."""
        variables = set()
        for clause in self.clauses:
            for literal in clause.literals:
                variables.add(literal.variable)
        return variables
    
    def get_num_clauses(self) -> int:
        """Get the number of clauses."""
        return len(self.clauses)
    
    def get_num_literals(self) -> int:
        """Get the total number of literals across all clauses."""
        return sum(len(clause) for clause in self.clauses)
    
    def unit_propagation(self) -> bool:
        """
        Apply unit propagation to simplify the formula.
        Returns True if the formula is unsatisfiable, False otherwise.
        """
        changed = True
        while changed:
            changed = False
            unit_clauses = [c for c in self.clauses if c.is_unit()]
            
            if any(c.is_empty() for c in self.clauses):
                return True  # Unsatisfiable
            
            for unit_clause in unit_clauses:
                unit_literal = unit_clause.get_unit_literal()
                if unit_literal is None:
                    continue
                
                # Remove unit literal from all other clauses
                negated_literal = unit_literal.negate()
                for clause in self.clauses:
                    if clause is not unit_clause:
                        if negated_literal in clause.literals:
                            clause.remove_literal(negated_literal)
                            if clause.is_empty():
                                return True  # Unsatisfiable
                            changed = True
        
        return False  # Potentially satisfiable
    
    def pure_literal_elimination(self) -> None:
        """
        Eliminate pure literals (literals that appear only positive or only negative)
        from the formula.
        """
        variables = self.get_variables()
        
        for var in variables:
            has_positive = any(
                Literal(var, False) in clause.literals
                for clause in self.clauses
            )
            has_negative = any(
                Literal(var, True) in clause.literals
                for clause in self.clauses
            )
            
            if has_positive and not has_negative:
                # var is pure positive, remove clauses containing it
                self.clauses = [
                    c for c in self.clauses
                    if Literal(var, False) not in c.literals
                ]
            elif has_negative and not has_positive:
                # var is pure negative, remove clauses containing ¬var
                self.clauses = [
                    c for c in self.clauses
                    if Literal(var, True) not in c.literals
                ]


class SATResult:
    """
    Result of a SAT solver call.
    
    Attributes:
        satisfiable: True if the formula is satisfiable
        assignment: Variable assignment if satisfiable, None otherwise
    """
    
    def __init__(self, satisfiable: bool, assignment: Optional[Dict[str, bool]] = None):
        self.satisfiable = satisfiable
        self.assignment = assignment or {}
    
    def __str__(self) -> str:
        """String representation."""
        if self.satisfiable:
            return f"SAT with assignment: {self.assignment}"
        return "UNSAT"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"SATResult(satisfiable={self.satisfiable}, assignment={self.assignment})"
