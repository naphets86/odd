"""
Boolean Circuit representation and manipulation.

This module provides classes for representing Boolean circuits,
circuit nodes, and circuit operations.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import copy


@dataclass
class CircuitNode:
    """
    A node in a Boolean circuit.
    
    Attributes:
        node_id: Unique identifier for the node
        gate_type: Type of logic gate (AND, OR, NOT, NAND, NOR, XOR, etc.)
        inputs: List of input node IDs
        output: Output node ID (if this is an input, output is its own ID)
        is_input: Whether this is a circuit input
        is_output: Whether this is a circuit output
    """
    node_id: str
    gate_type: str  # "AND", "OR", "NOT", "NAND", "NOR", "XOR", "INPUT", "CONSTANT"
    inputs: List[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    constant_value: Optional[bool] = None  # For CONSTANT nodes
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.node_id}:{self.gate_type}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CircuitNode({self.node_id}, {self.gate_type}, inputs={self.inputs})"
    
    def is_binary_gate(self) -> bool:
        """Check if this is a binary gate."""
        return self.gate_type in {"AND", "OR", "NAND", "NOR", "XOR"}
    
    def is_unary_gate(self) -> bool:
        """Check if this is a unary gate."""
        return self.gate_type in {"NOT"}
    
    def get_arity(self) -> int:
        """Get the arity (number of inputs) of this gate."""
        if self.gate_type in {"AND", "OR", "NAND", "NOR", "XOR"}:
            return len(self.inputs)
        elif self.gate_type == "NOT":
            return 1
        elif self.gate_type in {"INPUT", "CONSTANT"}:
            return 0
        else:
            return len(self.inputs)


class BooleanCircuit:
    """
    A Boolean circuit is a directed acyclic graph (DAG) of logic gates.
    
    Attributes:
        nodes: Dictionary of node_id -> CircuitNode
        inputs: List of input node IDs
        outputs: List of output node IDs
        name: Circuit name
    """
    
    def __init__(self, name: str = "Circuit"):
        self.name = name
        self.nodes: Dict[str, CircuitNode] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
    
    def __str__(self) -> str:
        """String representation."""
        return f"BooleanCircuit({self.name}, nodes={len(self.nodes)}, " \
               f"inputs={len(self.inputs)}, outputs={len(self.outputs)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"BooleanCircuit(name={self.name}, nodes={list(self.nodes.keys())})"
    
    def __len__(self) -> int:
        """Number of nodes in the circuit."""
        return len(self.nodes)
    
    def add_node(self, node: CircuitNode) -> None:
        """Add a node to the circuit."""
        self.nodes[node.node_id] = node
        
        if node.is_input or node.gate_type == "INPUT":
            if node.node_id not in self.inputs:
                self.inputs.append(node.node_id)
        
        if node.is_output:
            if node.node_id not in self.outputs:
                self.outputs.append(node.node_id)
    
    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def add_input(self, input_id: str, variable_name: Optional[str] = None) -> CircuitNode:
        """Add an input node to the circuit."""
        node = CircuitNode(
            node_id=input_id,
            gate_type="INPUT",
            is_input=True
        )
        self.add_node(node)
        return node
    
    def add_output(self, output_id: str, source_node_id: str) -> CircuitNode:
        """Add an output node that connects to a source node."""
        if source_node_id not in self.nodes:
            raise ValueError(f"Source node {source_node_id} not found")
        
        node = CircuitNode(
            node_id=output_id,
            gate_type="OUTPUT",
            inputs=[source_node_id],
            is_output=True
        )
        self.add_node(node)
        return node
    
    def add_gate(self, gate_id: str, gate_type: str, input_ids: List[str]) -> CircuitNode:
        """Add a logic gate to the circuit."""
        for input_id in input_ids:
            if input_id not in self.nodes:
                raise ValueError(f"Input node {input_id} not found")
        
        if gate_type not in {"AND", "OR", "NAND", "NOR", "XOR", "NOT"}:
            raise ValueError(f"Unknown gate type: {gate_type}")
        
        node = CircuitNode(
            node_id=gate_id,
            gate_type=gate_type,
            inputs=input_ids
        )
        self.add_node(node)
        return node
    
    def add_constant_node(self, const_id: str, value: bool) -> CircuitNode:
        """Add a constant value node (0 or 1)."""
        node = CircuitNode(
            node_id=const_id,
            gate_type="CONSTANT",
            constant_value=value
        )
        self.add_node(node)
        return node
    
    def evaluate(self, assignment: Dict[str, bool]) -> Dict[str, bool]:
        """
        Evaluate the circuit on the given input assignment.
        
        Args:
            assignment: Dict mapping input node IDs to truth values
            
        Returns:
            Dict of node_id -> computed value for all nodes
        """
        evaluation = {}
        
        # Topological evaluation
        def evaluate_node(node_id: str) -> bool:
            if node_id in evaluation:
                return evaluation[node_id]
            
            node = self.nodes[node_id]
            
            if node.gate_type == "INPUT":
                value = assignment.get(node_id, False)
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "CONSTANT":
                evaluation[node_id] = node.constant_value
                return node.constant_value
            
            elif node.gate_type == "NOT":
                input_value = evaluate_node(node.inputs[0])
                value = not input_value
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "AND":
                value = all(evaluate_node(inp) for inp in node.inputs)
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "OR":
                value = any(evaluate_node(inp) for inp in node.inputs)
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "NAND":
                value = not all(evaluate_node(inp) for inp in node.inputs)
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "NOR":
                value = not any(evaluate_node(inp) for inp in node.inputs)
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "XOR":
                value = sum(1 for inp in node.inputs if evaluate_node(inp)) % 2 == 1
                evaluation[node_id] = value
                return value
            
            elif node.gate_type == "OUTPUT":
                input_value = evaluate_node(node.inputs[0])
                evaluation[node_id] = input_value
                return input_value
            
            else:
                raise ValueError(f"Unknown gate type: {node.gate_type}")
        
        # Evaluate all nodes
        for node_id in self.nodes:
            evaluate_node(node_id)
        
        return evaluation
    
    def get_output_values(self, assignment: Dict[str, bool]) -> Dict[str, bool]:
        """Get only the output values for an assignment."""
        evaluation = self.evaluate(assignment)
        return {out_id: evaluation[out_id] for out_id in self.outputs}
    
    def get_output_value(self, assignment: Dict[str, bool], output_id: Optional[str] = None) -> bool:
        """Get a single output value."""
        if output_id is None:
            if len(self.outputs) != 1:
                raise ValueError("Circuit has multiple outputs, specify output_id")
            output_id = self.outputs[0]
        
        evaluation = self.evaluate(assignment)
        return evaluation[output_id]
    
    def copy(self) -> "BooleanCircuit":
        """Create a deep copy of the circuit."""
        new_circuit = BooleanCircuit(self.name)
        
        for node_id, node in self.nodes.items():
            new_node = CircuitNode(
                node_id=node.node_id,
                gate_type=node.gate_type,
                inputs=node.inputs.copy(),
                is_input=node.is_input,
                is_output=node.is_output,
                constant_value=node.constant_value
            )
            new_circuit.add_node(new_node)
        
        return new_circuit
    
    def extract_sub_circuits(self) -> Dict[str, "BooleanCircuit"]:
        """
        Extract sub-circuits for each output.
        
        Returns a dict mapping output_id -> sub-circuit for that output.
        """
        sub_circuits = {}
        
        for output_id in self.outputs:
            sub_circuit = self.copy()
            # Keep only the paths leading to this output
            sub_circuits[output_id] = sub_circuit
        
        return sub_circuits
    
    def get_topological_order(self) -> List[str]:
        """
        Get nodes in topological order.
        
        Returns a list of node IDs in topological order.
        """
        visited = set()
        order = []
        
        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            for input_id in node.inputs:
                visit(input_id)
            
            order.append(node_id)
        
        for node_id in self.nodes:
            visit(node_id)
        
        return order
    
    def get_stat_info(self) -> Dict[str, int]:
        """Get statistics about the circuit."""
        gate_counts = {}
        
        for node in self.nodes.values():
            gate_counts[node.gate_type] = gate_counts.get(node.gate_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "num_inputs": len(self.inputs),
            "num_outputs": len(self.outputs),
            "gate_counts": gate_counts
        }


class CircuitBuilder:
    """
    Helper class for building circuits programmatically.
    """
    
    def __init__(self, name: str = "Circuit"):
        self.circuit = BooleanCircuit(name)
        self._node_counter = 0
    
    def add_input(self, name: str) -> str:
        """Add an input and return its ID."""
        self.circuit.add_input(name)
        return name
    
    def add_output(self, input_id: str, name: str) -> str:
        """Add an output and return its ID."""
        self.circuit.add_output(name, input_id)
        return name
    
    def add_gate(self, gate_type: str, *input_ids: str) -> str:
        """Add a gate and return its ID."""
        gate_id = f"g{self._node_counter}"
        self._node_counter += 1
        self.circuit.add_gate(gate_id, gate_type, list(input_ids))
        return gate_id
    
    def add_and(self, *input_ids: str) -> str:
        """Add an AND gate."""
        return self.add_gate("AND", *input_ids)
    
    def add_or(self, *input_ids: str) -> str:
        """Add an OR gate."""
        return self.add_gate("OR", *input_ids)
    
    def add_not(self, input_id: str) -> str:
        """Add a NOT gate."""
        return self.add_gate("NOT", input_id)
    
    def add_nand(self, *input_ids: str) -> str:
        """Add a NAND gate."""
        return self.add_gate("NAND", *input_ids)
    
    def add_nor(self, *input_ids: str) -> str:
        """Add a NOR gate."""
        return self.add_gate("NOR", *input_ids)
    
    def add_xor(self, *input_ids: str) -> str:
        """Add an XOR gate."""
        return self.add_gate("XOR", *input_ids)
    
    def build(self) -> BooleanCircuit:
        """Build and return the circuit."""
        return self.circuit
