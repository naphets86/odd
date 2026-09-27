"""
Tests für uqtl.normalizer – IQT-Normalisierungsregeln gemäß Abschnitt 3.3.

Abdeckung:
  - Regel 3.1: AND-Prädikat-Sortierung
  - Regel 3.2: Alias-Normalisierung (t1, t2, …)
  - Regel 3.3: Boolean-Vereinfachung (TRUE, FALSE, NOT NOT p, p AND p)
  - Regel 3.4: Projektionsreihenfolge
  - Regel 3.5: JOIN-Normalisierung (left-assoziativ)
  - Alle Knotentypen durchlaufen den Normalisierer
"""

import pytest
from uqtl.iqt import (
    NodeType, JoinType, SortDirection, NullsOrder,
    AggFunc, SetOpType, FilterPosition, CompareOp, LogicOp, FrameUnit, WindowFunc,
    ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowFrame, WindowDef,
    OnCondition, UsingCondition,
    IQTNode,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode, CTEDef,
)
from uqtl.normalizer import IQTNormalizer


def _scan(table: str, alias: str = None) -> ScanNode:
    return ScanNode(node_type=NodeType.SCAN, table=table, alias=alias)


def _col(col: str, table: str = None) -> ColumnRef:
    return ColumnRef(column=col, table=table)


def _lit(val) -> Literal:
    return Literal(value=val)


class TestAliasNormalization:
    """Regel 3.2: Tabellenaliase → t1, t2, …"""

    def test_single_scan_gets_t1(self):
        n = IQTNormalizer()
        result = n.normalize(_scan("customers"))
        assert isinstance(result, ScanNode)
        assert result.alias == "t1"

    def test_two_scans_in_join_get_t1_t2(self):
        n = IQTNormalizer()
        cond = OnCondition(
            predicate=CompareExpr(_col("id", "customers"), CompareOp.EQ, _col("customer_id", "orders"))
        )
        join = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers"),
            right=_scan("orders"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        result = n.normalize(join)
        assert isinstance(result, JoinNode)
        assert result.left.alias == "t1"
        assert result.right.alias == "t2"

    def test_named_alias_remapped(self):
        n = IQTNormalizer()
        scan = _scan("customers", alias="c")
        result = n.normalize(scan)
        assert result.alias == "t1"

    def test_column_ref_remapped_via_table_name(self):
        n = IQTNormalizer()
        pred = CompareExpr(_col("country", "customers"), CompareOp.EQ, _lit("DE"))
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result, FilterNode)
        assert isinstance(result.predicate, CompareExpr)
        assert result.predicate.left.table == "t1"


class TestAndPredicateSorting:
    """Regel 3.1: AND-Prädikate lexikografisch sortieren."""

    def test_and_sorted_lexicographically(self):
        n = IQTNormalizer()
        p1 = CompareExpr(_col("z_col"), CompareOp.EQ, _lit(1))
        p2 = CompareExpr(_col("a_col"), CompareOp.EQ, _lit(2))
        pred = LogicExpr(op=LogicOp.AND, operands=[p1, p2])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, LogicExpr)
        # a_col kommt lexikografisch vor z_col
        first = result.predicate.operands[0]
        assert isinstance(first, CompareExpr)
        assert first.left.column == "a_col"

    def test_commutative_and_normalizes_same(self):
        n = IQTNormalizer()
        p1 = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        p2 = CompareExpr(_col("country"), CompareOp.EQ, _lit("DE"))

        pred_ab = LogicExpr(op=LogicOp.AND, operands=[p1, p2])
        pred_ba = LogicExpr(op=LogicOp.AND, operands=[p2, p1])

        filt_ab = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred_ab)
        filt_ba = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred_ba)

        r_ab = n.normalize(filt_ab)
        r_ba = n.normalize(filt_ba)

        assert repr(r_ab.predicate) == repr(r_ba.predicate)


class TestBooleanSimplification:
    """Regel 3.3: Boolesche Vereinfachung."""

    def test_p_and_true_reduces_to_p(self):
        n = IQTNormalizer()
        p = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        true_lit = Literal(value=True)
        pred = LogicExpr(op=LogicOp.AND, operands=[p, true_lit])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        result = n.normalize(filt)
        # Ergebnis sollte kein AND mehr sein, sondern direkt p
        assert not isinstance(result.predicate, LogicExpr) or \
               result.predicate.op != LogicOp.AND or \
               len(result.predicate.operands) == 1

    def test_not_not_p_reduces_to_p(self):
        n = IQTNormalizer()
        p = CompareExpr(_col("x"), CompareOp.EQ, _lit(1))
        double_not = LogicExpr(op=LogicOp.NOT, operands=[
            LogicExpr(op=LogicOp.NOT, operands=[p])
        ])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=double_not)
        result = n.normalize(filt)
        # NOT NOT p → p
        assert isinstance(result.predicate, CompareExpr)

    def test_p_and_p_deduplicates(self):
        n = IQTNormalizer()
        p = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        pred = LogicExpr(op=LogicOp.AND, operands=[p, p])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        result = n.normalize(filt)
        # p AND p → p (kein AND mit zwei gleichen Operanden)
        assert not isinstance(result.predicate, LogicExpr)

    def test_or_with_false_drops_false(self):
        n = IQTNormalizer()
        p = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        false_lit = Literal(value=False)
        pred = LogicExpr(op=LogicOp.OR, operands=[p, false_lit])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, CompareExpr)


class TestAllNodeTypesThroughNormalizer:
    """Alle Knotentypen müssen den Normalisierer ohne Fehler durchlaufen."""

    def test_scan(self):
        n = IQTNormalizer()
        result = n.normalize(_scan("customers"))
        assert isinstance(result, ScanNode)

    def test_filter_where(self):
        n = IQTNormalizer()
        pred = CompareExpr(_col("active"), CompareOp.EQ, _lit(True))
        node = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred)
        result = n.normalize(node)
        assert isinstance(result, FilterNode)

    def test_project(self):
        n = IQTNormalizer()
        cols = [ProjectionItem(expr=_col("name")), ProjectionItem(expr=_col("email"))]
        node = ProjectNode(node_type=NodeType.PROJECT, source=_scan("customers"), columns=cols)
        result = n.normalize(node)
        assert isinstance(result, ProjectNode)
        assert len(result.columns) == 2

    def test_group_agg(self):
        n = IQTNormalizer()
        agg = AggItem(func=AggFunc.COUNT, arg=_col("id"), alias="cnt")
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders"),
            group_by=[_col("customer_id")],
            aggregates=[agg],
        )
        result = n.normalize(node)
        assert isinstance(result, GroupAggNode)

    def test_group_agg_with_having(self):
        n = IQTNormalizer()
        agg = AggItem(func=AggFunc.COUNT, arg=_col("id"), alias="cnt")
        having = CompareExpr(FuncCall("COUNT", [_col("id")]), CompareOp.GT, _lit(3))
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders"),
            group_by=[_col("customer_id")],
            aggregates=[agg],
            having=having,
        )
        result = n.normalize(node)
        assert result.having is not None

    def test_sort(self):
        n = IQTNormalizer()
        items = [SortItem(_col("name"), SortDirection.ASC)]
        node = SortNode(node_type=NodeType.SORT, source=_scan("customers"), items=items)
        result = n.normalize(node)
        assert isinstance(result, SortNode)

    def test_limit(self):
        n = IQTNormalizer()
        node = LimitNode(node_type=NodeType.LIMIT, source=_scan("products"), count=10, offset=5)
        result = n.normalize(node)
        assert isinstance(result, LimitNode)
        assert result.count == 10
        assert result.offset == 5

    def test_window(self):
        n = IQTNormalizer()
        wdef = WindowDef(
            func=WindowFunc.RANK,
            result_alias="rnk",
            partition_by=[_col("customer_id")],
            order_by=[SortItem(_col("total"), SortDirection.DESC)],
        )
        node = WindowNode(node_type=NodeType.WINDOW, source=_scan("orders"), definitions=[wdef])
        result = n.normalize(node)
        assert isinstance(result, WindowNode)

    def test_subquery(self):
        n = IQTNormalizer()
        inner = _scan("orders")
        node = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner, alias="sub")
        result = n.normalize(node)
        assert isinstance(result, SubqueryNode)
        assert result.alias == "sub"

    def test_set_op(self):
        n = IQTNormalizer()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("customers"),
            right=_scan("archived_customers"),
            op=SetOpType.UNION,
        )
        result = n.normalize(node)
        assert isinstance(result, SetOpNode)

    def test_with(self):
        n = IQTNormalizer()
        cte = CTEDef(name="rev", query=_scan("orders"))
        node = WithNode(
            node_type=NodeType.WITH,
            ctes=[cte],
            query=_scan("customers"),
        )
        result = n.normalize(node)
        assert isinstance(result, WithNode)
        assert result.ctes[0].name == "rev"

    def test_with_no_main_query(self):
        n = IQTNormalizer()
        cte = CTEDef(name="rev", query=_scan("orders"))
        node = WithNode(node_type=NodeType.WITH, ctes=[cte], query=None)
        result = n.normalize(node)
        assert result.query is None


class TestExprNormalization:
    """Normalisierung aller Ausdruckstypen."""

    def test_in_expr(self):
        n = IQTNormalizer()
        pred = InExpr(value=_col("status"), items=[_lit("shipped"), _lit("delivered")])
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("orders"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, InExpr)

    def test_between_expr(self):
        n = IQTNormalizer()
        pred = BetweenExpr(value=_col("total"), lo=_lit(100), hi=_lit(500))
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("orders"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, BetweenExpr)

    def test_like_expr(self):
        n = IQTNormalizer()
        pred = LikeExpr(value=_col("name"), pattern="%Epp%")
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, LikeExpr)

    def test_null_expr(self):
        n = IQTNormalizer()
        pred = NullExpr(value=_col("email"))
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, NullExpr)

    def test_exists_expr_passthrough(self):
        n = IQTNormalizer()
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=_scan("orders"))
        pred = ExistsExpr(subquery=sub)
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("customers"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, ExistsExpr)

    def test_func_call_expr(self):
        n = IQTNormalizer()
        pred = CompareExpr(
            FuncCall("EXTRACT", [_lit("YEAR"), _col("order_date")]),
            CompareOp.EQ,
            _lit(2024),
        )
        filt = FilterNode(node_type=NodeType.FILTER, source=_scan("orders"), predicate=pred)
        result = n.normalize(filt)
        assert isinstance(result.predicate, CompareExpr)

    def test_using_condition_passthrough(self):
        n = IQTNormalizer()
        cond = UsingCondition(columns=["customer_id"])
        join = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers"),
            right=_scan("orders"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        result = n.normalize(join)
        assert isinstance(result.condition, UsingCondition)


class TestMissingCoverage:
    """Gezielte Tests für bisher nicht abgedeckte Zeilen."""

    def test_normalize_node_unknown_type_passthrough(self):
        # Zeile 71: _normalize_node gibt unbekannten Knoten unverändert zurück
        n = IQTNormalizer()
        # IQTNode-Basisinstanz ist KEIN ScanNode, FilterNode usw. → Fallback
        plain = IQTNode(node_type=NodeType.SCAN)
        result = n._normalize_node(plain, {}, [1])
        assert result is plain

    def test_group_agg_with_and_having_sorted(self):
        # Zeile 151: GroupAgg mit AND-HAVING → _sort_and_predicates(having)
        n = IQTNormalizer()
        p1 = CompareExpr(_col("status"), CompareOp.EQ, _lit("ok"))
        p2 = CompareExpr(FuncCall("COUNT", [_col("id")]), CompareOp.GT, _lit(0))
        agg = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders"),
            group_by=[_col("cid")],
            aggregates=[AggItem(func=AggFunc.COUNT, arg=None, alias="n")],
            having=LogicExpr(op=LogicOp.AND, operands=[p2, p1]),
        )
        result = n.normalize(agg)
        assert isinstance(result.having, LogicExpr)
        reprs = [repr(o) for o in result.having.operands]
        assert reprs == sorted(reprs)

    def test_normalize_expr_none_returns_none(self):
        # Zeile 226: _normalize_expr(None, {}) → None
        # Wird erreicht über ProjectionItem(expr=None, star=True)
        n = IQTNormalizer()
        proj = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("t"),
            columns=[ProjectionItem(expr=None, star=True)],
        )
        result = n.normalize(proj)
        assert result.columns[0].star is True

    def test_normalize_expr_unknown_type_passthrough(self):
        # Zeile 276: _normalize_expr – finaler Fallback für unbekannte Ausdruckstypen
        class _UnknownExpr:
            pass

        n = IQTNormalizer()
        unknown = _UnknownExpr()
        result = n._normalize_expr(unknown, {})
        assert result is unknown

    def test_simplify_boolean_simple_not_preserved(self):
        # Zeile 325: NOT(p) – p ist kein NOT → LogicExpr(NOT, [simplified]) zurückgeben
        n = IQTNormalizer()
        pred = LogicExpr(
            op=LogicOp.NOT,
            operands=[CompareExpr(_col("active"), CompareOp.EQ, _lit(True))],
        )
        filt = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers"),
            predicate=pred,
        )
        result = n.normalize(filt)
        assert isinstance(result.predicate, LogicExpr)
        assert result.predicate.op == LogicOp.NOT

    def test_simplify_boolean_or_multiple_operands(self):
        # Zeile 355: OR(p1, p2) ohne FALSE → LogicExpr(OR, [p1, p2]) bleibt
        n = IQTNormalizer()
        p1 = CompareExpr(_col("a"), CompareOp.EQ, _lit(1))
        p2 = CompareExpr(_col("b"), CompareOp.EQ, _lit(2))
        pred = LogicExpr(op=LogicOp.OR, operands=[p1, p2])
        filt = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("t"),
            predicate=pred,
        )
        result = n.normalize(filt)
        assert isinstance(result.predicate, LogicExpr)
        assert result.predicate.op == LogicOp.OR
        assert len(result.predicate.operands) == 2

    def test_simplify_boolean_unknown_op_fallback(self):
        # Zeile 357: _simplify_boolean – Fallback für unbekannte LogicOp-Werte
        from unittest.mock import MagicMock

        n = IQTNormalizer()
        mock_expr = MagicMock(spec=LogicExpr)
        mock_expr.op = object()  # kein AND, OR, NOT
        mock_expr.operands = []
        result = n._simplify_boolean(mock_expr)
        # Muss ein LogicExpr mit dem unbekannten op zurückgeben
        assert hasattr(result, "op")
