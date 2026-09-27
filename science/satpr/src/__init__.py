"""
SATPR - Wahrscheinlichkeitsverteilung und Laufzeitanalyse des Subgraph-SAT-Solvers

Version: 1.0.0
Autor: Stephan Epp
Datum: 10. Juli 2026

Diese Implementierung enthält alle Algorithmen und mathematischen Konzepte
aus der wissenschaftlichen Arbeit:

"Wahrscheinlichkeitsverteilung und Laufzeitanalyse des Subgraph-SAT-Solvers"
"""

__version__ = '1.0.0'
__author__ = 'Stephan Epp'
__date__ = '2026-07-10'

# Importiere Hauptmodule
from metadistribution import (
    ProbabilitySpace,
    DistributionSpace,
    MetaDistribution,
    ShiftTrajectory
)

from sat_solver import (
    Clause,
    SATFormula,
    CombinationPyramid,
    SubgraphSATSolver
)

from probability_shift import (
    ProbabilityDistribution,
    ProbabilityShiftAnalyzer,
    RuntimeAnalysis,
    ThreeWayPartition
)

from lbv_solver import (
    LogarithmicAssignmentProcedure,
    CombinedLBVSubgraphSolver,
    LBVAnalysis
)

from simulation import (
    SimulationResult,
    MonteCarloSimulator,
    LBVSimulator
)

from statistics import (
    PhaseTransitionAnalysis,
    ConvergenceAnalysis,
    BimodalDistributionAnalysis,
    EntropyAnalysis,
    StatisticalTests
)

__all__ = [
    # Metaverteilungs-Theorie
    'ProbabilitySpace',
    'DistributionSpace',
    'MetaDistribution',
    'ShiftTrajectory',
    
    # SAT-Solver
    'Clause',
    'SATFormula',
    'CombinationPyramid',
    'SubgraphSATSolver',
    
    # Wahrscheinlichkeitsverschiebung
    'ProbabilityDistribution',
    'ProbabilityShiftAnalyzer',
    'RuntimeAnalysis',
    'ThreeWayPartition',
    
    # LBV
    'LogarithmicAssignmentProcedure',
    'CombinedLBVSubgraphSolver',
    'LBVAnalysis',
    
    # Simulationen
    'SimulationResult',
    'MonteCarloSimulator',
    'LBVSimulator',
    
    # Statistiken
    'PhaseTransitionAnalysis',
    'ConvergenceAnalysis',
    'BimodalDistributionAnalysis',
    'EntropyAnalysis',
    'StatisticalTests'
]
