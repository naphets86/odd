"""
Learning SAT in Boolean Circuits (LSAT) Framework

A Python implementation of the polynomial SAT-solver approach using the
Subgraph Algorithm, as described in the academic work.

Key modules:
- cnf: CNF formula representation and manipulation
- boolean_functions: Boolean functions and formulas
- circuit: Boolean circuit representation
- subgraph_sat_solver: Polynomial SAT solver using Subgraph Algorithm
- circuit_to_cnf: Circuit-to-CNF transformation
- learning: Main learning algorithm
- lbv: Logarithmic Assignment Procedure
"""

__version__ = "1.0.0"
__author__ = "Stephan Epp"

from .cnf import CNFFormula, Literal, Clause
from .boolean_functions import BooleanFunction, BooleanFormula
from .circuit import BooleanCircuit
from .subgraph_sat_solver import SubgraphSATSolver
from .circuit_to_cnf import CircuitToCNFTransformer
from .learning import LearningSATInCircuitsEngine
from .lbv import LogarithmicAssignmentProcedure

__all__ = [
    "CNFFormula",
    "Literal",
    "Clause",
    "BooleanFunction",
    "BooleanFormula",
    "BooleanCircuit",
    "SubgraphSATSolver",
    "CircuitToCNFTransformer",
    "LearningSATInCircuitsEngine",
    "LogarithmicAssignmentProcedure",
]
