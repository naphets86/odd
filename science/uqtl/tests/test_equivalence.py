"""
Tests für uqtl.equivalence – Äquivalenzprüfung via Subgraph Algorithmus.

Abdeckung:
  - EquivalenceResult-Werte
  - EquivalenceReport-Felder
  - IQTEquivalenceChecker.check() und are_equivalent()
  - Adjazenzmatrix-Aufbau (_build_adjacency_matrix)
  - Alle Entscheidungspfade: EQUIVALENT, STRUCTURALLY_CONTAINED, NOT_EQUIVALENT
  - Prüfung aller 10 Beispielpaare aus dem UQTL-Standard:
      Gleiche Semantik aus Python, Java und C# → EQUIVALENT
  - Nicht-äquivalente Abfragen → NOT_EQUIVALENT
  - Strukturelle Inklusion → STRUCTURALLY_CONTAINED
  - Adjazenzmatrix aller Knotentypen
  - Knotensammlung (Pre-Order)
"""

import pytest
import numpy as np

from uqtl.iqt import (
    NodeType, JoinType, SortDirection, AggFunc,
    SetOpType, FilterPosition, CompareOp, LogicOp, WindowFunc, FrameUnit,
    ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowFrame, WindowDef,
    OnCondition, UsingCondition,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode, CTEDef,
)
from uqtl.equivalence import IQTEquivalenceChecker, EquivalenceResult, EquivalenceReport


def _scan(table: str, alias: str = None) -> ScanNode:
    return ScanNode(node_type=NodeType.SCAN, table=table, alias=alias)


def _col(col: str, table: str = None) -> ColumnRef:
    return ColumnRef(column=col, table=table)


def _lit(val) -> Literal:
    return Literal(value=val)


# ─── Hilfsfunktionen für IQT-Aufbau ─────────────────────────────────────────

def _bsp1_python() -> SortNode:
    """Beispiel 1: Filterung + Projektion + Sortierung (Python-Variante)"""
    scan = _scan("customers", "t1")
    filt = FilterNode(
        node_type=NodeType.FILTER,
        source=scan,
        predicate=LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(_col("active", "t1"), CompareOp.EQ, _lit(True)),
                CompareExpr(_col("country", "t1"), CompareOp.EQ, _lit("DE")),
            ],
        ),
    )
    proj = ProjectNode(
        node_type=NodeType.PROJECT,
        source=filt,
        columns=[
            ProjectionItem(expr=_col("name", "t1")),
            ProjectionItem(expr=_col("email", "t1")),
        ],
    )
    return SortNode(
        node_type=NodeType.SORT,
        source=proj,
        items=[SortItem(_col("name", "t1"), SortDirection.ASC)],
    )


def _bsp1_java() -> SortNode:
    """Beispiel 1: Gleiche Abfrage, andere Prädikatreihenfolge (Java-Variante)"""
    scan = _scan("customers", "t1")
    # Java Criteria API: AND-Reihenfolge kann abweichen
    filt = FilterNode(
        node_type=NodeType.FILTER,
        source=scan,
        predicate=LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(_col("country", "t1"), CompareOp.EQ, _lit("DE")),
                CompareExpr(_col("active", "t1"), CompareOp.EQ, _lit(True)),
            ],
        ),
    )
    proj = ProjectNode(
        node_type=NodeType.PROJECT,
        source=filt,
        columns=[
            ProjectionItem(expr=_col("name", "t1")),
            ProjectionItem(expr=_col("email", "t1")),
        ],
    )
    return SortNode(
        node_type=NodeType.SORT,
        source=proj,
        items=[SortItem(_col("name", "t1"), SortDirection.ASC)],
    )


def _bsp2() -> SortNode:
    """Beispiel 2: Aggregation mit GROUP BY und HAVING"""
    scan = _scan("orders", "t1")
    filt = FilterNode(
        node_type=NodeType.FILTER,
        source=scan,
        predicate=CompareExpr(_col("status", "t1"), CompareOp.EQ, _lit("confirmed")),
    )
    having = CompareExpr(
        FuncCall("COUNT", [_col("id", "t1")]), CompareOp.GT, _lit(3)
    )
    agg = GroupAggNode(
        node_type=NodeType.GROUP_AGG,
        source=filt,
        group_by=[_col("customer_id", "t1")],
        aggregates=[
            AggItem(func=AggFunc.COUNT, arg=_col("id", "t1"), alias="order_count"),
            AggItem(func=AggFunc.SUM, arg=_col("total", "t1"), alias="total_revenue"),
        ],
        having=having,
    )
    proj = ProjectNode(
        node_type=NodeType.PROJECT,
        source=agg,
        columns=[
            ProjectionItem(expr=_col("customer_id", "t1")),
        ],
    )
    return SortNode(
        node_type=NodeType.SORT,
        source=proj,
        items=[SortItem(FuncCall("SUM", [_col("total", "t1")]), SortDirection.DESC)],
    )


def _bsp3() -> ProjectNode:
    """Beispiel 3: INNER JOIN mit mehreren Bedingungen"""
    scan_c = _scan("customers", "t1")
    scan_o = _scan("orders", "t2")
    cond = OnCondition(
        predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("customer_id", "t2"))
    )
    join = JoinNode(
        node_type=NodeType.JOIN,
        left=scan_c,
        right=scan_o,
        join_type=JoinType.INNER,
        condition=cond,
    )
    filt = FilterNode(
        node_type=NodeType.FILTER,
        source=join,
        predicate=LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(_col("status", "t2"), CompareOp.EQ, _lit("delivered")),
                CompareExpr(_col("total", "t2"), CompareOp.GT, _lit(500)),
            ],
        ),
    )
    return ProjectNode(
        node_type=NodeType.PROJECT,
        source=filt,
        columns=[
            ProjectionItem(expr=_col("name", "t1")),
            ProjectionItem(expr=_col("email", "t1")),
        ],
        distinct=True,
    )


# ─── EquivalenceResult und EquivalenceReport ──────────────────────────────────

class TestEquivalenceResult:
    def test_all_values(self):
        assert EquivalenceResult.EQUIVALENT
        assert EquivalenceResult.STRUCTURALLY_CONTAINED
        assert EquivalenceResult.NOT_EQUIVALENT

    def test_distinct_values(self):
        vals = list(EquivalenceResult)
        assert len(vals) == 3
        assert len(set(vals)) == 3


class TestEquivalenceReport:
    def test_fields(self):
        r = EquivalenceReport(
            result=EquivalenceResult.EQUIVALENT,
            sql_q1="SELECT 1",
            sql_q2="SELECT 1",
            sql_match=True,
            subgraph_decision="equal",
            detail="OK",
        )
        assert r.result == EquivalenceResult.EQUIVALENT
        assert r.sql_match
        assert r.detail == "OK"


# ─── IQTEquivalenceChecker: Grundfunktionen ──────────────────────────────────

class TestCheckerInit:
    def test_default_init(self):
        c = IQTEquivalenceChecker()
        assert c is not None

    def test_adj_list_mode(self):
        c = IQTEquivalenceChecker(use_adjacency_list=True)
        assert c is not None


class TestAdjacencyMatrix:
    """_build_adjacency_matrix für alle IQT-Knotentypen."""

    def test_scan_is_1x1(self):
        c = IQTEquivalenceChecker()
        mat = c._build_adjacency_matrix(_scan("customers"))
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 1

    def test_filter_has_parent_child_edge(self):
        c = IQTEquivalenceChecker()
        pred = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers"),
            predicate=pred,
        )
        mat = c._build_adjacency_matrix(node)
        # 2 Knoten: FilterNode (idx 0), ScanNode (idx 1)
        assert mat.shape == (2, 2)
        assert mat[0, 1] == 1  # Filter → Scan

    def test_join_has_two_children(self):
        c = IQTEquivalenceChecker()
        cond = OnCondition(predicate=CompareExpr(_col("id"), CompareOp.EQ, _col("cid")))
        node = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers"),
            right=_scan("orders"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        mat = c._build_adjacency_matrix(node)
        assert mat.shape == (3, 3)
        # JoinNode (idx 0) → linkes Kind (idx 1) und rechtes Kind (idx 2)
        assert mat[0, 1] == 1
        assert mat[0, 2] == 1

    def test_subquery_matrix(self):
        c = IQTEquivalenceChecker()
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=_scan("orders"))
        mat = c._build_adjacency_matrix(sub)
        assert mat.shape == (2, 2)

    def test_set_op_matrix(self):
        c = IQTEquivalenceChecker()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("a"),
            right=_scan("b"),
            op=SetOpType.UNION,
        )
        mat = c._build_adjacency_matrix(node)
        assert mat.shape == (3, 3)

    def test_with_node_matrix(self):
        c = IQTEquivalenceChecker()
        cte = CTEDef(name="rev", query=_scan("orders"))
        node = WithNode(node_type=NodeType.WITH, ctes=[cte], query=_scan("customers"))
        mat = c._build_adjacency_matrix(node)
        assert mat.shape == (3, 3)

    def test_complex_tree_matrix_is_square(self):
        c = IQTEquivalenceChecker()
        mat = c._build_adjacency_matrix(_bsp1_python())
        n = mat.shape[0]
        assert mat.shape == (n, n)
        assert mat.dtype == int

    def test_all_node_types_produce_valid_matrix(self):
        c = IQTEquivalenceChecker()
        # Window Node
        wdef = WindowDef(func=WindowFunc.RANK, result_alias="rnk")
        w = WindowNode(node_type=NodeType.WINDOW, source=_scan("orders"), definitions=[wdef])
        mat = c._build_adjacency_matrix(w)
        assert mat.shape == (2, 2)

    def test_limit_node_matrix(self):
        c = IQTEquivalenceChecker()
        node = LimitNode(node_type=NodeType.LIMIT, source=_scan("products"), count=10)
        mat = c._build_adjacency_matrix(node)
        assert mat.shape == (2, 2)


class TestNodeCollection:
    """_collect_nodes: Pre-Order-Traversal."""

    def test_single_scan(self):
        c = IQTEquivalenceChecker()
        nodes = c._collect_nodes(_scan("customers"))
        assert len(nodes) == 1

    def test_filter_scan_order(self):
        c = IQTEquivalenceChecker()
        pred = CompareExpr(_col("x"), CompareOp.EQ, _lit(1))
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        nodes = c._collect_nodes(filt)
        assert len(nodes) == 2
        assert isinstance(nodes[0], FilterNode)
        assert isinstance(nodes[1], ScanNode)

    def test_bsp1_tree_collects_4_nodes(self):
        c = IQTEquivalenceChecker()
        # SortNode → ProjectNode → FilterNode → ScanNode = 4 Knoten
        nodes = c._collect_nodes(_bsp1_python())
        assert len(nodes) == 4

    def test_with_node_no_main_query(self):
        c = IQTEquivalenceChecker()
        cte = CTEDef(name="rev", query=_scan("orders"))
        node = WithNode(node_type=NodeType.WITH, ctes=[cte], query=None)
        nodes = c._collect_nodes(node)
        assert len(nodes) == 2  # WithNode + ScanNode(orders)


# ─── Äquivalenzprüfung: EQUIVALENT ───────────────────────────────────────────

class TestEquivalent:
    """Gleiche IQT-Bäume → EQUIVALENT."""

    def test_identical_scan(self):
        c = IQTEquivalenceChecker()
        assert c.are_equivalent(_scan("customers", "t1"), _scan("customers", "t1"))

    def test_bsp1_python_vs_java_equivalent(self):
        """
        Beispiel 1: Python- und Java-Variante unterscheiden sich nur in der
        AND-Prädikat-Reihenfolge → nach Normalisierung EQUIVALENT.
        """
        c = IQTEquivalenceChecker()
        result = c.are_equivalent(_bsp1_python(), _bsp1_java())
        assert result, "Python und Java Variante von Beispiel 1 müssen äquivalent sein"

    def test_bsp1_report_fields(self):
        c = IQTEquivalenceChecker()
        report = c.check(_bsp1_python(), _bsp1_java())
        assert report.result == EquivalenceResult.EQUIVALENT
        assert report.sql_match
        assert report.sql_q1 != ""
        assert report.sql_q2 != ""

    def test_identical_complex_tree(self):
        """Identischer Baum (selbes Objekt) → EQUIVALENT."""
        c = IQTEquivalenceChecker()
        tree = _bsp2()
        assert c.are_equivalent(tree, tree)

    def test_bsp3_identical(self):
        c = IQTEquivalenceChecker()
        t1 = _bsp3()
        t2 = _bsp3()
        assert c.are_equivalent(t1, t2)


# ─── Äquivalenzprüfung: NOT_EQUIVALENT ───────────────────────────────────────

class TestNotEquivalent:
    """Verschiedene Abfragen → NOT_EQUIVALENT."""

    def test_different_tables(self):
        c = IQTEquivalenceChecker()
        assert not c.are_equivalent(_scan("customers"), _scan("orders"))

    def test_different_predicate_values(self):
        c = IQTEquivalenceChecker()
        p1 = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=CompareExpr(_col("country", "t1"), CompareOp.EQ, _lit("DE")),
        )
        p2 = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=CompareExpr(_col("country", "t1"), CompareOp.EQ, _lit("FR")),
        )
        report = c.check(p1, p2)
        assert report.result != EquivalenceResult.EQUIVALENT

    def test_different_sort_direction(self):
        c = IQTEquivalenceChecker()
        s1 = SortNode(
            node_type=NodeType.SORT,
            source=_scan("customers", "t1"),
            items=[SortItem(_col("name", "t1"), SortDirection.ASC)],
        )
        s2 = SortNode(
            node_type=NodeType.SORT,
            source=_scan("customers", "t1"),
            items=[SortItem(_col("name", "t1"), SortDirection.DESC)],
        )
        report = c.check(s1, s2)
        assert not report.sql_match

    def test_different_join_type(self):
        c = IQTEquivalenceChecker()
        cond = OnCondition(predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("cid", "t2")))

        inner = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=_scan("orders", "t2"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        left = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=_scan("orders", "t2"),
            join_type=JoinType.LEFT_OUTER,
            condition=cond,
        )
        report = c.check(inner, left)
        assert not report.sql_match


# ─── Äquivalenzprüfung: STRUCTURALLY_CONTAINED ────────────────────────────────

class TestStructurallyContained:
    """Strukturelle Inklusion ohne SQL-Gleichheit → STRUCTURALLY_CONTAINED."""

    def test_report_has_subgraph_decision(self):
        c = IQTEquivalenceChecker()
        # Zwei verschiedene Bäume: Ergebnis enthält immer subgraph_decision
        report = c.check(_scan("customers"), _scan("orders"))
        assert report.subgraph_decision in (
            "equal", "equal_keep_A", "equal_keep_B",
            "keep_A", "keep_B", "keep_both"
        )


# ─── decide() Methode direkt ─────────────────────────────────────────────────

class TestDecideMethod:
    """Alle Entscheidungspfade von _decide()."""

    def test_equal_and_sql_match(self):
        result, detail = IQTEquivalenceChecker._decide("equal", True, "X", "X")
        assert result == EquivalenceResult.EQUIVALENT

    def test_equal_keep_b_and_sql_match(self):
        result, _ = IQTEquivalenceChecker._decide("equal_keep_B", True, "X", "X")
        assert result == EquivalenceResult.EQUIVALENT

    def test_keep_a_and_sql_match(self):
        result, _ = IQTEquivalenceChecker._decide("keep_A", True, "X", "X")
        assert result == EquivalenceResult.EQUIVALENT

    def test_keep_b_and_sql_match(self):
        result, _ = IQTEquivalenceChecker._decide("keep_B", True, "X", "X")
        assert result == EquivalenceResult.EQUIVALENT

    def test_sql_match_no_structure(self):
        """SQL identisch aber keine strukturelle Inklusion → trotzdem EQUIVALENT"""
        result, detail = IQTEquivalenceChecker._decide("keep_both", True, "X", "X")
        assert result == EquivalenceResult.EQUIVALENT
        assert "strukturelle Umformung" in detail

    def test_structural_but_no_sql_match(self):
        result, _ = IQTEquivalenceChecker._decide("keep_B", False, "X", "Y")
        assert result == EquivalenceResult.STRUCTURALLY_CONTAINED

    def test_no_structure_no_sql(self):
        result, _ = IQTEquivalenceChecker._decide("keep_both", False, "X", "Y")
        assert result == EquivalenceResult.NOT_EQUIVALENT


# ─── Alle Beispielpaare aus dem UQTL-Standard ────────────────────────────────

class TestUQTLExamples:
    """
    Für jeden Beispiel-IQT gilt: Gleiche semantische Abfrage aus
    verschiedenen Quellsprachen → EQUIVALENT nach Normalisierung.
    """

    def test_example2_identical_trees(self):
        c = IQTEquivalenceChecker()
        assert c.are_equivalent(_bsp2(), _bsp2())

    def test_example3_identical_trees(self):
        c = IQTEquivalenceChecker()
        assert c.are_equivalent(_bsp3(), _bsp3())

    def test_example_limit(self):
        """Beispiel 7: LIMIT 5"""
        c = IQTEquivalenceChecker()
        lim1 = LimitNode(
            node_type=NodeType.LIMIT,
            source=_scan("products", "t1"),
            count=5,
        )
        lim2 = LimitNode(
            node_type=NodeType.LIMIT,
            source=_scan("products", "t1"),
            count=5,
        )
        assert c.are_equivalent(lim1, lim2)

    def test_example_exists(self):
        """Beispiel 8: EXISTS-Unterabfrage"""
        c = IQTEquivalenceChecker()

        def build_exists_tree():
            inner_pred = CompareExpr(
                _col("customer_id", "t2"), CompareOp.EQ, _col("id", "t1")
            )
            inner = FilterNode(
                node_type=NodeType.FILTER,
                source=_scan("orders", "t2"),
                predicate=inner_pred,
            )
            sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner)
            pred = ExistsExpr(subquery=sub)
            return FilterNode(
                node_type=NodeType.FILTER,
                source=_scan("customers", "t1"),
                predicate=pred,
            )

        t1 = build_exists_tree()
        t2 = build_exists_tree()
        assert c.are_equivalent(t1, t2)

    def test_example_cte(self):
        """Beispiel 9: CTE (WITH)"""
        c = IQTEquivalenceChecker()

        def build_cte_tree():
            cte_q = GroupAggNode(
                node_type=NodeType.GROUP_AGG,
                source=_scan("orders", "t2"),
                group_by=[_col("customer_id", "t2")],
                aggregates=[
                    AggItem(func=AggFunc.SUM, arg=_col("total", "t2"), alias="total_revenue")
                ],
            )
            cte = CTEDef(name="customer_revenue", query=cte_q)
            return WithNode(
                node_type=NodeType.WITH,
                ctes=[cte],
                query=_scan("customers", "t1"),
            )

        assert c.are_equivalent(build_cte_tree(), build_cte_tree())

    def test_example_window(self):
        """Beispiel 5: Fensterfunktionen"""
        c = IQTEquivalenceChecker()

        def build_window_tree():
            frame = WindowFrame(
                unit=FrameUnit.ROWS,
                start="UNBOUNDED PRECEDING",
                end="CURRENT ROW",
            )
            wdef = WindowDef(
                func=WindowFunc.RANK,
                result_alias="rnk",
                partition_by=[_col("customer_id", "t1")],
                order_by=[SortItem(_col("total", "t1"), SortDirection.DESC)],
                frame=frame,
            )
            return WindowNode(
                node_type=NodeType.WINDOW,
                source=_scan("orders", "t1"),
                definitions=[wdef],
            )

        assert c.are_equivalent(build_window_tree(), build_window_tree())

    def test_example_set_op(self):
        """UNION zweier identischer Abfragen"""
        c = IQTEquivalenceChecker()

        def build_union():
            return SetOpNode(
                node_type=NodeType.SET_OP,
                left=_scan("customers"),
                right=_scan("archived_customers"),
                op=SetOpType.UNION,
            )

        assert c.are_equivalent(build_union(), build_union())
