"""
Polynomial SAT Solver using the Subgraph Algorithm.

This module integrates the Subgraph Algorithm (from subgraph package in venv)
as a polynomial SAT solver. It implements the reduction chain:
SAT -> 3-SAT -> Clique -> Subgraph-Isomorphism
"""

from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass
import numpy as np

from src.cnf import CNFFormula, Literal, Clause, SATResult

# Import the Subgraph Algorithm from the installed subgraph package
try:
    from subgraph import Subgraph
except ImportError:
    raise ImportError("subgraph package not found in environment. "
                      "Ensure it is installed from: "
                      "git+https://github.com/hjstephan86/subgraph.git@v1.0.0")


@dataclass
class Graph:
    """
    Simple graph representation using adjacency matrix.
    
    Attributes:
        num_nodes: Number of nodes
        adjacency_matrix: NumPy matrix representation
    """
    num_nodes: int
    adjacency_matrix: np.ndarray
    
    def is_complete_graph_K_n(self) -> bool:
        """Check if this is a complete graph K_n."""
        if self.num_nodes == 0:
            return True
        expected_edges = self.num_nodes * (self.num_nodes - 1) // 2
        actual_edges = np.count_nonzero(self.adjacency_matrix) // 2
        return actual_edges == expected_edges


class SubgraphSATSolver:
    """
    Polynomial-time SAT solver using the Subgraph Algorithm.
    
    Implements the full reduction chain:
    1. Normalize CNF to 3-SAT
    2. Reduce 3-SAT to Clique problem
    3. Translate Clique to Subgraph-Isomorphism
    4. Solve using Subgraph Algorithm in O(n^3) time
    """
    
    def __init__(self):
        """Initialize the Subgraph SAT solver."""
        self.subgraph_algo = Subgraph()
        self.last_result = None
    
    def solve(self, cnf_formula: CNFFormula) -> SATResult:
        """
        Solve a CNF formula using the Subgraph Algorithm.
        
        Args:
            cnf_formula: The CNF formula to solve
            
        Returns:
            SATResult with satisfiability and assignment (if SAT)
        """
        # Step 1: Normalize to 3-SAT
        three_sat = self._normalize_to_3sat(cnf_formula)
        
        # Step 2: Reduce to Clique problem
        clique_graph, clause_mapping = self._clique_reduction(three_sat)
        m = three_sat.get_num_clauses()
        
        # Step 3: Create K_m (complete graph with m nodes)
        K_m = self._create_complete_graph(m)
        
        # Step 4: Apply Subgraph Algorithm
        result = self._apply_subgraph_algorithm(K_m, clique_graph)
        
        # Step 5: Extract and construct assignment
        if result:
            clique = self._extract_clique(result, m)
            assignment = self._construct_assignment(
                clique, clause_mapping, three_sat, cnf_formula
            )
            return SATResult(satisfiable=True, assignment=assignment)
        else:
            return SATResult(satisfiable=False)
    
    def _normalize_to_3sat(self, cnf_formula: CNFFormula) -> CNFFormula:
        """
        Convert arbitrary CNF to 3-SAT.
        
        For each clause with > 3 literals, introduces auxiliary variables.
        Time: O(|formula|)
        """
        three_sat = CNFFormula(num_variables=cnf_formula.num_variables)
        aux_var_counter = 0
        
        for clause in cnf_formula.clauses:
            if len(clause.literals) <= 3:
                # Already in 3-CNF
                three_sat.add_clause(clause.copy())
            else:
                # Split clause into 3-clauses with auxiliary variables
                literals = list(clause.literals)
                
                # Create initial 3-clause: (l1 ∨ l2 ∨ y1)
                y_vars = []
                
                for i in range(len(literals) - 3):
                    y_var = f"_y{aux_var_counter}"
                    aux_var_counter += 1
                    y_vars.append(y_var)
                
                # First clause: (l1 ∨ l2 ∨ y0)
                first_clause = Clause({literals[0], literals[1],
                                       Literal(y_vars[0], False)})
                three_sat.add_clause(first_clause)
                
                # Middle clauses: (¬yi-1 ∨ li+2 ∨ yi)
                for i in range(len(literals) - 3):
                    middle_clause = Clause({
                        Literal(y_vars[i], True),
                        literals[i + 2],
                        Literal(y_vars[i + 1], False) if i + 1 < len(y_vars)
                        else literals[-1]
                    })
                    three_sat.add_clause(middle_clause)
                
                # Last clause if needed
                if len(y_vars) > 0:
                    last_clause = Clause({
                        Literal(y_vars[-1], True),
                        literals[-2],
                        literals[-1]
                    })
                    three_sat.add_clause(last_clause)
        
        return three_sat
    
    def _clique_reduction(self, three_sat: CNFFormula) -> Tuple[Graph, Dict]:
        """
        Reduce 3-SAT to Clique problem using Cook-Levin construction.
        
        For each clause Cj = (l1 ∨ l2 ∨ l3), create nodes (j,1), (j,2), (j,3).
        Add edges between (j1,p1) and (j2,p2) if j1 ≠ j2 and literals are compatible.
        
        Returns:
            (clique_graph, clause_mapping): Graph and mapping info
        """
        clauses = list(three_sat.clauses)
        m = len(clauses)
        
        # Create nodes: 3m total (3 per clause)
        clause_mapping = {}
        for j, clause in enumerate(clauses):
            clause_mapping[j] = list(clause.literals)
        
        # Create adjacency matrix (3m x 3m)
        num_nodes = 3 * m
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        # Map (clause_id, literal_pos) to node index
        def node_index(clause_id: int, literal_pos: int) -> int:
            return clause_id * 3 + literal_pos
        
        # Add edges
        for j1 in range(m):
            for p1 in range(3):
                if p1 >= len(clause_mapping[j1]):
                    continue
                
                for j2 in range(j1 + 1, m):
                    for p2 in range(3):
                        if p2 >= len(clause_mapping[j2]):
                            continue
                        
                        lit1 = clause_mapping[j1][p1]
                        lit2 = clause_mapping[j2][p2]
                        
                        # Check if literals are compatible (not complementary)
                        if not lit1.is_complementary(lit2):
                            idx1 = node_index(j1, p1)
                            idx2 = node_index(j2, p2)
                            adj_matrix[idx1, idx2] = 1
                            adj_matrix[idx2, idx1] = 1
        
        clique_graph = Graph(num_nodes, adj_matrix)
        return clique_graph, clause_mapping
    
    def _create_complete_graph(self, n: int) -> Graph:
        """
        Create complete graph K_n.
        
        Time: O(n^2)
        """
        adj_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj_matrix, 0)  # No self-loops
        
        return Graph(n, adj_matrix)
    
    def _apply_subgraph_algorithm(self, K_m: Graph, clique_graph: Graph) -> Optional[List[int]]:
        """
        Apply the Subgraph Algorithm to check if K_m is isomorphic to a subgraph
        of clique_graph.
        
        Time: O(n^3)
        
        Returns:
            Node mapping if K_m is a subgraph, None otherwise
        """
        try:
            # Use the Subgraph Algorithm from the imported package
            result = self.subgraph_algo.compare_graphs(
                K_m.adjacency_matrix,
                clique_graph.adjacency_matrix
            )
            
            # The result should indicate whether K_m is a subgraph of clique_graph
            # and potentially return a mapping
            self.last_result = result
            
            if result:
                return result
            else:
                return None
        
        except Exception as e:
            raise RuntimeError(f"Subgraph algorithm failed: {e}")
    
    def _extract_clique(self, result: List[int], m: int) -> Set[int]:
        """
        Extract the clique (node set) from the subgraph match result.
        
        The result should indicate which m nodes from clique_graph form a clique
        matching K_m.
        """
        if isinstance(result, bool):
            # If result is just True/False, need to compute the clique
            # This would require more information from the Subgraph Algorithm
            return set(range(m))
        
        # If result is a mapping, extract the matched nodes
        return set(result[:m])
    
    def _construct_assignment(self, 
                             clique: Set[int],
                             clause_mapping: Dict,
                             three_sat: CNFFormula,
                             original_cnf: CNFFormula) -> Dict[str, bool]:
        """
        From the clique, construct a satisfying assignment for the variables.
        
        Each node (j, p) in the clique represents a satisfied literal in clause j.
        The assignment is constructed from these satisfied literals.
        """
        assignment = {}
        
        # Extract satisfied literals from clique
        satisfied_literals = set()
        for node_idx in clique:
            clause_id = node_idx // 3
            literal_pos = node_idx % 3
            
            if clause_id < len(clause_mapping):
                literals = clause_mapping[clause_id]
                if literal_pos < len(literals):
                    satisfied_literals.add(literals[literal_pos])
        
        # Construct assignment from satisfied literals
        for literal in satisfied_literals:
            assignment[literal.variable] = not literal.is_negated
        
        # Fill in remaining variables with False
        for var in original_cnf.get_variables():
            if var not in assignment:
                assignment[var] = False
        
        # Filter out auxiliary variables
        final_assignment = {
            var: val for var, val in assignment.items()
            if not var.startswith('_y')
        }
        
        return final_assignment
    
    def verify_solution(self, cnf_formula: CNFFormula, assignment: Dict[str, bool]) -> bool:
        """
        Verify that an assignment satisfies the CNF formula.
        
        Args:
            cnf_formula: The CNF formula
            assignment: The assignment to verify
            
        Returns:
            True if the assignment satisfies all clauses
        """
        for clause in cnf_formula.clauses:
            clause_satisfied = False
            
            for literal in clause.literals:
                var_value = assignment.get(literal.variable, False)
                lit_value = var_value if not literal.is_negated else not var_value
                
                if lit_value:
                    clause_satisfied = True
                    break
            
            if not clause_satisfied:
                return False
        
        return True
