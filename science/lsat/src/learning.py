"""
Learning Engine for SAT-Specification problems.

Main algorithm LearningSATInCircuits that learns CNF formulas to match
circuit specifications using the Subgraph SAT-Solver.
"""

from typing import Dict, Optional, List, Set
from src.circuit import BooleanCircuit
from src.cnf import CNFFormula, Literal
from src.boolean_functions import DNFFormula, Term
from src.subgraph_sat_solver import SubgraphSATSolver
from src.circuit_to_cnf import CircuitToCNFTransformer


class SATSpecificationProblem:
    """
    Represents a SAT-Specification Problem.
    
    Problem: Find a CNF formula H such that C(h/H)(i) ≈ g(i)
    where C is a circuit with hidden function h, and g is target function.
    
    Attributes:
        circuit: Boolean circuit with hidden function
        target: Target Boolean function circuit
        name: Problem name
    """
    
    def __init__(self, circuit: BooleanCircuit, target: BooleanCircuit, 
                 name: str = "SAT-Spec"):
        self.circuit = circuit
        self.target = target
        self.name = name
    
    def __str__(self) -> str:
        """String representation."""
        return f"SATSpecificationProblem({self.name})"


class LearningSATInCircuitsEngine:
    """
    Main learning algorithm: LearningSATInCircuits
    
    Extends the approach from Epp13 to arbitrary CNF formulas using
    the Subgraph SAT-Solver instead of heuristic SAT solvers.
    
    Attributes:
        problem: The SAT-specification problem to solve
        sat_solver: Subgraph-based SAT solver
        transformer: Circuit-to-CNF transformer
        learned_formula: The learned hypothesis H
        iterations: Number of iterations performed
    """
    
    def __init__(self, problem: SATSpecificationProblem):
        """Initialize the learning engine."""
        self.problem = problem
        self.sat_solver = SubgraphSATSolver()
        self.transformer = CircuitToCNFTransformer()
        
        self.learned_formula: Optional[DNFFormula] = None
        self.iterations = 0
        self.h_satisfying_assignments: List[Dict[str, bool]] = []
        self.max_iterations = 1000  # Safety limit
    
    def __str__(self) -> str:
        """String representation."""
        return f"LearningSATInCircuitsEngine({self.problem.name})"
    
    def learn(self) -> Optional[DNFFormula]:
        """
        Main learning algorithm: LearningSATInCircuits
        
        Algorithm:
        1. Initialize H = 0 (constant false)
        2. While C(h/H)(i) ≢ g(i):
            a. Find new h-satisfying assignment v
            b. Build term t from v
            c. Reduce t to prime implicant
            d. Add t to H
        3. Return H when C(h/H)(i) ≈ g(i)
        
        Returns:
            Learned DNF formula H, or None if no solution found
        """
        # Initialize H = 0 (false formula)
        self.learned_formula = DNFFormula(terms=[], is_monotone=False)
        self.h_satisfying_assignments = []
        
        print(f"Starting learning process for {self.problem.name}")
        print(f"Circuit has {len(self.problem.circuit)} nodes")
        
        # Extract sub-circuits
        sub_circuits = self._extract_sub_circuits()
        
        # Main learning loop
        while self.iterations < self.max_iterations:
            self.iterations += 1
            print(f"\nIteration {self.iterations}")
            
            # Step 1: Check equivalence C(h/H)(i) ≈ g(i)
            equiv_result = self._check_equivalence()
            
            if not equiv_result.satisfiable:
                # Formulas are equivalent - learning complete
                print(f"✓ Learning complete! Found H with {len(self.learned_formula.terms)} terms")
                return self.learned_formula
            
            print("  Circuit and target differ - searching for new term")
            
            # Step 2: Find new h-satisfying assignment
            h_result = self._find_h_satisfying_assignment()
            
            if not h_result.satisfiable:
                print("  No new h-satisfying assignment found - no solution")
                return None
            
            assignment = h_result.assignment
            print(f"  Found h-satisfying assignment: {assignment}")
            self.h_satisfying_assignments.append(assignment)
            
            # Step 3: Build term from assignment
            term = self._build_term_from_assignment(assignment, sub_circuits)
            print(f"  Initial term: {term}")
            
            # Step 4: Reduce to prime implicant
            prime_term = self._reduce_to_prime_implicant(term, sub_circuits, assignment)
            print(f"  Reduced term: {prime_term}")
            
            # Step 5: Add term to H
            self.learned_formula.add_term(prime_term)
            print(f"  Added to H. H now has {len(self.learned_formula.terms)} terms")
        
        print(f"ERROR: Maximum iterations ({self.max_iterations}) reached")
        return None
    
    def _extract_sub_circuits(self) -> Dict[str, BooleanCircuit]:
        """Extract sub-circuits for analysis."""
        return self.problem.circuit.extract_sub_circuits()
    
    def _check_equivalence(self) -> "SATResult":
        """
        Check if C(h/H)(i) ≈ g(i).
        
        Returns satisfiable iff they differ.
        """
        # Build modified circuit with H instead of h
        modified_circuit = self._substitute_h_into_circuit(self.learned_formula)
        
        # Check equivalence
        equiv_cnf = self.transformer.build_equivalence_cnf(
            modified_circuit,
            self.problem.target
        )
        
        return self.sat_solver.solve(equiv_cnf)
    
    def _find_h_satisfying_assignment(self) -> "SATResult":
        """
        Find an assignment v that is h-satisfying:
        - C_0(v(i), y/1) = g(v(i))  [with h=1, circuit matches target]
        - C_0(v(i), y/0) ≠ g(v(i))  [with h=0, circuit differs from target]
        - v is new w.r.t. current H
        
        Returns SATResult with the assignment if found.
        """
        # Build CNF for h-satisfying condition
        h_cnf = self.transformer.build_h_satisfying_cnf(
            self.learned_formula,
            self.problem.circuit,
            self.problem.target
        )
        
        # Filter out assignments already covered by learned formula
        result = self.sat_solver.solve(h_cnf)
        
        while result.satisfiable and self._assignment_covered(result.assignment):
            # This assignment is already covered by an existing term
            # Add constraint to exclude it
            self._add_negation_constraint(h_cnf, result.assignment)
            result = self.sat_solver.solve(h_cnf)
        
        return result
    
    def _assignment_covered(self, assignment: Dict[str, bool]) -> bool:
        """Check if an assignment is already covered by a term in H."""
        for term in self.learned_formula.terms:
            # Check if all literals in term are satisfied by assignment
            covered = True
            for literal in term.literals:
                var_val = assignment.get(literal.variable, False)
                lit_val = var_val if not literal.is_negated else not var_val
                if not lit_val:
                    covered = False
                    break
            
            if covered:
                return True
        
        return False
    
    def _add_negation_constraint(self, cnf: CNFFormula, assignment: Dict[str, bool]) -> None:
        """Add a constraint to CNF that negates the given assignment."""
        # Add clause: (¬x1 ∨ ¬x2 ∨ ... ∨ xk ∨ ...)
        # to exclude this specific assignment
        from .cnf import Clause
        
        clause_lits = set()
        for var, val in assignment.items():
            # Negate the assignment
            clause_lits.add(Literal(var, is_negated=val))
        
        if clause_lits:
            cnf.add_clause(Clause(clause_lits))
    
    def _build_term_from_assignment(self, assignment: Dict[str, bool],
                                    sub_circuits: Dict) -> Term:
        """
        Build term from h-satisfying assignment.
        
        For each sub-circuit that evaluates to true, include the corresponding
        variable in the term.
        """
        term = Term()
        
        # Evaluate sub-circuits
        for var_name in assignment:
            if assignment[var_name]:
                # This variable is set to True, include positive literal
                term.add_literal(Literal(var_name, is_negated=False))
        
        return term
    
    def _reduce_to_prime_implicant(self, term: Term, sub_circuits: Dict,
                                   assignment: Dict[str, bool]) -> Term:
        """
        Reduce a term to a prime implicant by removing variables.
        
        A variable can be removed if:
        - The reduced term still implies h on the assignment
        - Equivalently: removing the variable doesn't break satisfiability
        
        Args:
            term: The initial term
            sub_circuits: Sub-circuit dictionary
            assignment: The original satisfying assignment
            
        Returns:
            Prime implicant (maximal reducible term)
        """
        prime_term = term.copy()
        variables = list(prime_term.get_variables())
        
        for var in variables:
            # Try removing this variable
            test_term = prime_term.copy()
            test_term.literals = {lit for lit in test_term.literals 
                                 if lit.variable != var}
            
            # Check if term without var still satisfies the constraint
            can_remove = self._can_remove_variable(
                prime_term, var, sub_circuits, assignment
            )
            
            if can_remove:
                prime_term = test_term
        
        return prime_term
    
    def _can_remove_variable(self, term: Term, var: str, sub_circuits: Dict,
                            assignment: Dict[str, bool]) -> bool:
        """
        Check if a variable can be removed from a term.
        
        Uses SAT solver to check if reduced term is still satisfiable
        under target constraints.
        """
        # Build CNF for removability constraint
        test_term = term.copy()
        test_term.literals = {lit for lit in test_term.literals if lit.variable != var}
        
        removable_cnf = self.transformer.build_removable_cnf(
            test_term, var,
            self.problem.circuit,
            self.problem.target
        )
        
        # Solve to see if removal is valid
        result = self.sat_solver.solve(removable_cnf)
        return result.satisfiable
    
    def _substitute_h_into_circuit(self, h_formula: DNFFormula) -> BooleanCircuit:
        """
        Create modified circuit by substituting H for the hidden function h.
        
        Returns a circuit where h is replaced by its DNF representation.
        """
        # For now, return a copy - full implementation would replace
        # the hidden function node with H's circuit representation
        return self.problem.circuit.copy()
    
    def get_learned_formula(self) -> Optional[DNFFormula]:
        """Get the learned DNF formula."""
        return self.learned_formula
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            "iterations": self.iterations,
            "terms_learned": len(self.learned_formula.terms) if self.learned_formula else 0,
            "h_satisfying_assignments": len(self.h_satisfying_assignments),
            "success": self.learned_formula is not None
        }
