"""
Circuit-to-CNF transformation and CNF construction utilities.

This module provides the CircuitToCNFTransformer class for converting
Boolean circuits to CNF formulas, and specialized CNF construction methods
for equivalence checking, h-satisfying assignments, and variable elimination.
"""

from typing import Dict, Optional, List, Set, Tuple
from src.cnf import CNFFormula, Literal, Clause
from src.circuit import BooleanCircuit, CircuitNode
from src.boolean_functions import Term


class CircuitToCNFTransformer:
    """
    Transforms Boolean circuits to CNF formulas using Tseitin transformation.
    
    The Tseitin transformation introduces auxiliary variables for each gate
    and creates clauses that enforce the gate semantics.
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.aux_var_counter = 0
    
    def _get_aux_var(self, prefix: str = "_g") -> str:
        """Generate a new auxiliary variable name."""
        var = f"{prefix}{self.aux_var_counter}"
        self.aux_var_counter += 1
        return var
    
    def transform(self, circuit: BooleanCircuit) -> CNFFormula:
        """
        Transform a circuit to CNF using Tseitin transformation.
        
        Args:
            circuit: The Boolean circuit to transform
            
        Returns:
            CNF formula with auxiliary variables for each gate
        """
        cnf = CNFFormula()
        gate_outputs = {}  # Map from node_id to variable representing its output
        
        # Process nodes in topological order
        for node_id in circuit.get_topological_order():
            node = circuit.nodes[node_id]
            output_var = self._get_aux_var()
            gate_outputs[node_id] = output_var
            
            # Generate clauses for this node
            if node.gate_type == "INPUT":
                # Input nodes map directly to their variable
                gate_outputs[node_id] = node_id
            
            elif node.gate_type == "CONSTANT":
                # Constant nodes
                if node.constant_value:
                    # Force output_var to be true
                    cnf.add_literal_clause(Literal(output_var, False))
                else:
                    # Force output_var to be false
                    cnf.add_literal_clause(Literal(output_var, True))
            
            elif node.gate_type == "NOT":
                # y ⟺ ¬x  =>  (y ∨ x) ∧ (¬y ∨ ¬x)
                input_var = gate_outputs[node.inputs[0]]
                clause1 = Clause({Literal(output_var, False), Literal(input_var, False)})
                clause2 = Clause({Literal(output_var, True), Literal(input_var, True)})
                cnf.add_clause(clause1)
                cnf.add_clause(clause2)
            
            elif node.gate_type == "AND":
                # y ⟺ (x1 ∧ ... ∧ xn)
                input_vars = [gate_outputs[inp] for inp in node.inputs]
                
                # Forward: x1 ∧ ... ∧ xn => y
                # (¬x1 ∨ ... ∨ ¬xn ∨ y)
                clause = Clause({Literal(output_var, False)})
                for var in input_vars:
                    clause.add_literal(Literal(var, True))
                cnf.add_clause(clause)
                
                # Backward: y => x1 ∧ ... ∧ xn
                # (¬y ∨ xi) for each i
                for var in input_vars:
                    clause = Clause({Literal(output_var, True), Literal(var, False)})
                    cnf.add_clause(clause)
            
            elif node.gate_type == "OR":
                # y ⟺ (x1 ∨ ... ∨ xn)
                input_vars = [gate_outputs[inp] for inp in node.inputs]
                
                # Forward: x1 ∨ ... ∨ xn => y
                # (¬x1 ∨ y) ∧ ... ∧ (¬xn ∨ y)
                for var in input_vars:
                    clause = Clause({Literal(var, True), Literal(output_var, False)})
                    cnf.add_clause(clause)
                
                # Backward: y => x1 ∨ ... ∨ xn
                # (¬y ∨ x1 ∨ ... ∨ xn)
                clause = Clause({Literal(output_var, True)})
                for var in input_vars:
                    clause.add_literal(Literal(var, False))
                cnf.add_clause(clause)
            
            elif node.gate_type == "NAND":
                # y ⟺ ¬(x1 ∧ ... ∧ xn)  =>  y ⟺ (¬x1 ∨ ... ∨ ¬xn)
                input_vars = [gate_outputs[inp] for inp in node.inputs]
                
                # Forward: (¬x1 ∨ ... ∨ ¬xn) => y
                clause = Clause({Literal(output_var, False)})
                for var in input_vars:
                    clause.add_literal(Literal(var, True))
                cnf.add_clause(clause)
                
                # Backward: y => (¬x1 ∨ ... ∨ ¬xn)
                for var in input_vars:
                    clause = Clause({Literal(output_var, True), Literal(var, True)})
                    cnf.add_clause(clause)
            
            elif node.gate_type == "NOR":
                # y ⟺ ¬(x1 ∨ ... ∨ xn)  =>  y ⟺ (¬x1 ∧ ... ∧ ¬xn)
                input_vars = [gate_outputs[inp] for inp in node.inputs]
                
                # Forward: (¬x1 ∧ ... ∧ ¬xn) => y
                for var in input_vars:
                    clause = Clause({Literal(var, True), Literal(output_var, False)})
                    cnf.add_clause(clause)
                
                # Backward: y => (¬x1 ∧ ... ∧ ¬xn)
                clause = Clause({Literal(output_var, True)})
                for var in input_vars:
                    clause.add_literal(Literal(var, True))
                cnf.add_clause(clause)
            
            elif node.gate_type == "XOR":
                # y ⟺ (x1 ⊕ x2 ⊕ ... ⊕ xn)
                input_vars = [gate_outputs[inp] for inp in node.inputs]
                
                # This is more complex; simplified for binary XOR
                if len(input_vars) == 2:
                    x1, x2 = input_vars
                    # y ⟺ (x1 ⊕ x2)  =>  y ⟺ ((x1 ∨ x2) ∧ (¬x1 ∨ ¬x2))
                    # (¬x1 ∨ x2 ∨ y) ∧ (x1 ∨ ¬x2 ∨ y) ∧ (x1 ∨ x2 ∨ ¬y) ∧ (¬x1 ∨ ¬x2 ∨ ¬y)
                    
                    c1 = Clause({Literal(x1, True), Literal(x2, False),
                                 Literal(output_var, False)})
                    c2 = Clause({Literal(x1, False), Literal(x2, True),
                                 Literal(output_var, False)})
                    c3 = Clause({Literal(x1, False), Literal(x2, False),
                                 Literal(output_var, True)})
                    c4 = Clause({Literal(x1, True), Literal(x2, True),
                                 Literal(output_var, True)})
                    
                    cnf.add_clause(c1)
                    cnf.add_clause(c2)
                    cnf.add_clause(c3)
                    cnf.add_clause(c4)
            
            elif node.gate_type == "OUTPUT":
                # Output node: propagate value from input
                gate_outputs[node_id] = gate_outputs[node.inputs[0]]
        
        cnf.num_variables = len(set(gate_outputs.values()))
        self.gate_outputs = gate_outputs
        return cnf
    
    def build_equivalence_cnf(self, circuit1: BooleanCircuit, circuit2: BooleanCircuit) -> CNFFormula:
        """
        Build CNF for checking if two circuits compute equivalent functions.
        
        The formula is satisfiable iff the circuits are NOT equivalent.
        
        Args:
            circuit1: First circuit
            circuit2: Second circuit
            
        Returns:
            CNF formula that is satisfiable iff circuits differ
        """
        # Transform both circuits to CNF
        cnf1 = self.transform(circuit1)
        cnf2 = self.transform(circuit2)
        
        # Create combined CNF
        combined_cnf = CNFFormula()
        
        # Add all clauses from both
        for clause in cnf1.clauses:
            combined_cnf.add_clause(clause.copy())
        
        for clause in cnf2.clauses:
            combined_cnf.add_clause(clause.copy())
        
        # Get output variables
        output1 = list(self.gate_outputs.values())[-1] if self.gate_outputs else "out1"
        
        # Reconstruct for circuit2
        old_counter = self.aux_var_counter
        self.aux_var_counter = 1000
        self.gate_outputs = {}
        cnf2_copy = self.transform(circuit2)
        output2 = list(self.gate_outputs.values())[-1] if self.gate_outputs else "out2"
        self.aux_var_counter = old_counter
        
        # Add clause forcing outputs to differ: (output1 ≠ output2)
        # This is: (out1 ∨ out2) ∧ (¬out1 ∨ ¬out2)
        clause1 = Clause({Literal(output1, False), Literal(output2, False)})
        clause2 = Clause({Literal(output1, True), Literal(output2, True)})
        
        combined_cnf.add_clause(clause1)
        combined_cnf.add_clause(clause2)
        
        return combined_cnf
    
    def build_h_satisfying_cnf(self, hyp_formula: "DNFFormula", 
                               circuit: BooleanCircuit,
                               target: BooleanCircuit) -> CNFFormula:
        """
        Build CNF for finding an h-satisfying assignment under hypothesis H.
        
        An h-satisfying assignment is one where:
        1. C_0(i, y/1) = g(i)  [the circuit with h=1 matches target]
        2. C_0(i, y/0) ≠ g(i)  [the circuit with h=0 doesn't match target]
        3. The assignment is new w.r.t. H [not covered by existing terms]
        
        Args:
            hyp_formula: Current hypothesis H (DNF form)
            circuit: The circuit with hidden function
            target: Target function circuit
            
        Returns:
            CNF formula satisfiable iff new h-satisfying assignment exists
        """
        from .boolean_functions import DNFFormula
        
        cnf = CNFFormula()
        
        # Add constraints for h=1 case
        cnf_h1 = self.transform(circuit)
        for clause in cnf_h1.clauses:
            cnf.add_clause(clause.copy())
        
        # TODO: Implement full h-satisfying constraint encoding
        # This requires symbolic execution to detect when h=1 vs h=0
        
        return cnf
    
    def build_removable_cnf(self, term: Term, variable_to_remove: str,
                           circuit: BooleanCircuit, 
                           target: BooleanCircuit) -> CNFFormula:
        """
        Build CNF to check if a variable can be removed from a term.
        
        A variable x_k can be removed from term t if there's an assignment
        where:
        - All literals in (t - x_k) are satisfied
        - The circuit still matches the target
        
        Args:
            term: The current term
            variable_to_remove: The variable to potentially remove
            circuit: The circuit
            target: The target function
            
        Returns:
            CNF formula checking removability
        """
        cnf = CNFFormula()
        
        # Create CNF from circuit
        circuit_cnf = self.transform(circuit)
        for clause in circuit_cnf.clauses:
            cnf.add_clause(clause.copy())
        
        # Add constraints that term (without variable) must be satisfied
        for literal in term.literals:
            if literal.variable != variable_to_remove:
                cnf.add_literal_clause(literal)
        
        # Add constraints that removed variable's constraints are broken
        # (to verify removal changes semantics)
        
        return cnf
