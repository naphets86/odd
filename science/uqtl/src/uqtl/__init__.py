"""
uqtl – Unified Query Translation Layer
========================================
Version 2.0 des UQTL-Standards für die sprachunabhängige Darstellung
und Äquivalenzprüfung relationaler Datenbankabfragen.

Paketstruktur
-------------
:mod:`uqtl.iqt`
    Alle Datenklassen des *Intermediate Query Tree* (IQT):
    Knotentypen, Ausdruckstypen, Enumerationen und Hilfsklassen.

:mod:`uqtl.normalizer`
    :class:`~uqtl.normalizer.IQTNormalizer` – setzt die fünf
    Normalisierungsregeln (3.1–3.5) um und bringt jeden IQT-Baum
    in kanonische Form.

:mod:`uqtl.sql_generator`
    :class:`~uqtl.sql_generator.SQLGenerator` – wandelt einen
    normalisierten IQT-Baum in ANSI SQL:2016 um.

:mod:`uqtl.equivalence`
    :class:`~uqtl.equivalence.IQTEquivalenceChecker` – zweistufige
    Äquivalenzprüfung via Subgraph Algorithmus und SQL-Vergleich.

Schnellstart
------------
::

    from uqtl import (
        ScanNode, FilterNode, ProjectNode,
        NodeType, CompareOp, CompareExpr, ColumnRef, Literal,
        ProjectionItem,
        IQTNormalizer, SQLGenerator, IQTEquivalenceChecker,
    )

    scan = ScanNode(node_type=NodeType.SCAN, table="orders", alias="o")
    filt = FilterNode(
        node_type=NodeType.FILTER,
        source=scan,
        predicate=CompareExpr(
            ColumnRef("status", "o"), CompareOp.EQ, Literal("delivered")
        ),
    )

    sql = SQLGenerator().generate(IQTNormalizer().normalize(filt))
    print(sql)
    # SELECT *
    # FROM orders AS t1
    # WHERE t1.status = 'delivered'

    checker = IQTEquivalenceChecker()
    print(checker.are_equivalent(filt, filt))  # True
"""

__version__ = "2.0.0"

from uqtl.iqt import (
    IQTNode, NodeType, JoinType, SortDirection, NullsOrder,
    AggFunc, SetOpType, FrameUnit, WindowFunc,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode,
    Expr, ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowDef, WindowFrame,
    JoinCondition, OnCondition,
)
from uqtl.normalizer import IQTNormalizer
from uqtl.sql_generator import SQLGenerator
from uqtl.equivalence import IQTEquivalenceChecker, EquivalenceResult

__all__ = [
    "IQTNode", "NodeType", "JoinType", "SortDirection", "NullsOrder",
    "AggFunc", "SetOpType", "FrameUnit", "WindowFunc",
    "ScanNode", "FilterNode", "ProjectNode", "JoinNode",
    "GroupAggNode", "SortNode", "LimitNode", "WindowNode",
    "SubqueryNode", "SetOpNode", "WithNode",
    "Expr", "ColumnRef", "Literal", "FuncCall",
    "CompareExpr", "LogicExpr", "InExpr", "BetweenExpr",
    "LikeExpr", "NullExpr", "ExistsExpr",
    "ProjectionItem", "SortItem", "AggItem", "WindowDef", "WindowFrame",
    "JoinCondition", "OnCondition",
    "IQTNormalizer",
    "SQLGenerator",
    "IQTEquivalenceChecker", "EquivalenceResult",
    "__version__",
]
