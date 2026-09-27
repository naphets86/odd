"""
Utility functions and helper classes for LSAT framework.
"""

from typing import Dict, List, Optional, Set, Tuple
from src.cnf import CNFFormula, Literal, Clause, SATResult
from src.boolean_functions import BooleanFunction, DNFFormula, Term


def cnf_to_dimacs(cnf: CNFFormula) -> str:
    """
    Convert CNF formula to DIMACS format.
    
    DIMACS is a standard format for CNF formulas used by SAT solvers.
    
    Args:
        cnf: CNF formula to convert
        
    Returns:
        String in DIMACS format
    """
    variables = cnf.get_variables()
    var_to_int = {var: i+1 for i, var in enumerate(sorted(variables))}
    
    lines = []
    
    # Header: c comment, p cnf num_vars num_clauses
    lines.append(f"c DIMACS format CNF")
    lines.append(f"p cnf {len(variables)} {cnf.get_num_clauses()}")
    
    # Clauses
    for clause in cnf.clauses:
        clause_str = ""
        for literal in clause.literals:
            var_int = var_to_int[literal.variable]
            if literal.is_negated:
                clause_str += f"-{var_int} "
            else:
                clause_str += f"{var_int} "
        
        clause_str += "0"  # 0 terminates clause
        lines.append(clause_str)
    
    return "\n".join(lines)


def dimacs_to_cnf(dimacs_str: str) -> CNFFormula:
    """
    Convert DIMACS format to CNF formula.
    
    Args:
        dimacs_str: String in DIMACS format
        
    Returns:
        CNF formula
    """
    cnf = CNFFormula()
    int_to_var = {}
    
    lines = dimacs_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('c'):
            # Comment or empty
            continue
        
        if line.startswith('p'):
            # Header
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            
            # Create variable mapping
            for i in range(1, num_vars + 1):
                int_to_var[i] = f"x{i}"
            
            cnf.num_variables = num_vars
            continue
        
        # Clause
        lit_ints = [int(x) for x in line.split() if x != '0']
        literals = set()
        
        for lit_int in lit_ints:
            if lit_int == 0:
                break
            
            var_id = abs(lit_int)
            var_name = int_to_var.get(var_id, f"x{var_id}")
            is_negated = lit_int < 0
            
            literals.add(Literal(var_name, is_negated))
        
        if literals:
            cnf.add_clause(Clause(literals))
    
    return cnf


def evaluate_dnf_on_assignment(dnf: DNFFormula, assignment: Dict[str, bool]) -> bool:
    """
    Evaluate a DNF formula on an assignment.
    
    Args:
        dnf: DNF formula
        assignment: Variable assignment
        
    Returns:
        Truth value of the formula
    """
    return dnf.evaluate(assignment)


def evaluate_cnf_on_assignment(cnf: CNFFormula, assignment: Dict[str, bool]) -> bool:
    """
    Evaluate a CNF formula on an assignment.
    
    Args:
        cnf: CNF formula
        assignment: Variable assignment
        
    Returns:
        Truth value of the formula
    """
    for clause in cnf.clauses:
        clause_true = False
        
        for literal in clause.literals:
            var_val = assignment.get(literal.variable, False)
            lit_val = var_val if not literal.is_negated else not var_val
            
            if lit_val:
                clause_true = True
                break
        
        if not clause_true:
            return False
    
    return True


def generate_all_assignments(variables: List[str]) -> List[Dict[str, bool]]:
    """
    Generate all possible truth assignments for variables.
    
    Args:
        variables: List of variable names
        
    Returns:
        List of all possible assignments
    """
    if not variables:
        return [{}]
    
    assignments = []
    n = len(variables)
    
    for i in range(2**n):
        assignment = {}
        for j, var in enumerate(variables):
            assignment[var] = bool((i >> j) & 1)
        assignments.append(assignment)
    
    return assignments


def count_satisfying_assignments(cnf: CNFFormula) -> int:
    """
    Count the number of satisfying assignments for a CNF formula.
    
    This is a brute-force approach, exponential in the number of variables.
    
    Args:
        cnf: CNF formula
        
    Returns:
        Number of satisfying assignments
    """
    variables = list(cnf.get_variables())
    assignments = generate_all_assignments(variables)
    
    count = 0
    for assignment in assignments:
        if evaluate_cnf_on_assignment(cnf, assignment):
            count += 1
    
    return count


def simplify_cnf_basic(cnf: CNFFormula) -> CNFFormula:
    """
    Basic simplification of CNF formula.
    
    Removes duplicate clauses and empty clauses.
    
    Args:
        cnf: CNF formula
        
    Returns:
        Simplified CNF formula
    """
    simplified = CNFFormula(num_variables=cnf.num_variables)
    seen_clauses = set()
    
    for clause in cnf.clauses:
        if clause.is_empty():
            # Skip empty clauses (or mark UNSAT)
            continue
        
        # Convert clause to hashable form
        clause_sig = frozenset((lit.variable, lit.is_negated) for lit in clause.literals)
        
        if clause_sig not in seen_clauses:
            seen_clauses.add(clause_sig)
            simplified.add_clause(clause.copy())
    
    return simplified


def print_formula_stats(cnf: CNFFormula) -> None:
    """
    Print statistics about a CNF formula.
    
    Args:
        cnf: CNF formula
    """
    variables = cnf.get_variables()
    
    print("\nCNF Formula Statistics:")
    print(f"  Variables:        {len(variables)}")
    print(f"  Clauses:          {cnf.get_num_clauses()}")
    print(f"  Total literals:   {cnf.get_num_literals()}")
    
    if cnf.get_num_clauses() > 0:
        avg_clause_size = cnf.get_num_literals() / cnf.get_num_clauses()
        print(f"  Avg clause size:  {avg_clause_size:.2f}")
    
    if cnf.has_empty_clause():
        print("  Status:           UNSATISFIABLE (contains empty clause)")
    else:
        print("  Status:           Potentially satisfiable")


def benchmark_sat_solver(solver, cnf_formulas: List[Tuple[str, CNFFormula]]) -> None:
    """
    Benchmark a SAT solver on multiple formulas.
    
    Args:
        solver: The SAT solver to benchmark
        cnf_formulas: List of (name, formula) tuples
    """
    import time
    
    print("\n" + "="*70)
    print("SAT Solver Benchmark Results")
    print("="*70)
    print(f"{'Formula':<30} {'Variables':<12} {'Clauses':<12} {'Time (ms)':<12} {'Result':<10}")
    print("-"*70)
    
    total_time = 0
    
    for name, cnf in cnf_formulas:
        num_vars = len(cnf.get_variables())
        num_clauses = cnf.get_num_clauses()
        
        start_time = time.time()
        result = solver.solve(cnf)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        total_time += elapsed
        
        result_str = "SAT" if result.satisfiable else "UNSAT"
        print(f"{name:<30} {num_vars:<12} {num_clauses:<12} {elapsed:<12.3f} {result_str:<10}")
    
    print("-"*70)
    print(f"{'Total time:':<54} {total_time:.3f} ms")
    print("="*70 + "\n")
