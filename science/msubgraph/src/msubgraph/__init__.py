"""
msubgraph – Hierarchical Subgraph Depth-First Search (HS-DFS).

Exposes the complete public API of the HS-DFS algorithm (Algorithm 2):
the algorithm itself, its result container, and all supporting data types.
"""

from msubgraph.hsdfs import (
    HSDFS,
    HSDFSResult,
    MetaGraph,
    DFSResult,
    EdgeClass,
)

__all__ = [
    "HSDFS",
    "HSDFSResult",
    "MetaGraph",
    "DFSResult",
    "EdgeClass",
]
