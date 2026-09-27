"""
Tests für uqtl.iqt – IQT-Knotentypen und Datenklassen.

Abdeckung:
  - Alle Enum-Werte
  - Alle Knoten-Konstruktoren (ScanNode, FilterNode, …)
  - Alle Ausdrucks-Typen (ColumnRef, Literal, FuncCall, CompareExpr, …)
  - Hilfs-Datenklassen (AggItem, SortItem, ProjectionItem, WindowDef, …)
"""

import pytest
import numpy as np

from uqtl.iqt import (
    NodeType, JoinType, SortDirection, NullsOrder,
    AggFunc, SetOpType, FrameUnit, WindowFunc,
    CompareOp, LogicOp, FilterPosition,
    ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowFrame, WindowDef,
    OnCondition, UsingCondition,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode, CTEDef,
    IQTNode,
)


# ─── Enumerationen ────────────────────────────────────────────────────────────

class TestNodeType:
    def test_all_values(self):
        expected = {
            "SCAN", "FILTER", "PROJECT", "JOIN", "GROUP_AGG",
            "SORT", "LIMIT", "WINDOW", "SUBQUERY", "SET_OP", "WITH",
        }
        assert {n.name for n in NodeType} == expected


class TestJoinType:
    def test_values(self):
        assert JoinType.INNER.value == "INNER"
        assert JoinType.LEFT_OUTER.value == "LEFT OUTER"
        assert JoinType.RIGHT_OUTER.value == "RIGHT OUTER"
        assert JoinType.FULL_OUTER.value == "FULL OUTER"
        assert JoinType.CROSS.value == "CROSS"


class TestSortDirection:
    def test_asc_desc(self):
        assert SortDirection.ASC.value == "ASC"
        assert SortDirection.DESC.value == "DESC"


class TestNullsOrder:
    def test_values(self):
        assert NullsOrder.NULLS_FIRST.value == "NULLS FIRST"
        assert NullsOrder.NULLS_LAST.value == "NULLS LAST"
        assert NullsOrder.DEFAULT.value == ""


class TestAggFunc:
    def test_standard_funcs(self):
        assert AggFunc.COUNT.value == "COUNT"
        assert AggFunc.SUM.value == "SUM"
        assert AggFunc.AVG.value == "AVG"
        assert AggFunc.MIN.value == "MIN"
        assert AggFunc.MAX.value == "MAX"
        assert AggFunc.COUNT_DISTINCT.value == "COUNT_DISTINCT"


class TestSetOpType:
    def test_values(self):
        assert SetOpType.UNION.value == "UNION"
        assert SetOpType.UNION_ALL.value == "UNION ALL"
        assert SetOpType.INTERSECT.value == "INTERSECT"
        assert SetOpType.EXCEPT.value == "EXCEPT"


class TestWindowFunc:
    def test_ranking_funcs(self):
        assert WindowFunc.ROW_NUMBER.value == "ROW_NUMBER"
        assert WindowFunc.RANK.value == "RANK"
        assert WindowFunc.DENSE_RANK.value == "DENSE_RANK"
        assert WindowFunc.LAG.value == "LAG"
        assert WindowFunc.LEAD.value == "LEAD"


# ─── Ausdrücke ────────────────────────────────────────────────────────────────

class TestColumnRef:
    def test_with_table(self):
        c = ColumnRef(column="name", table="t1")
        assert repr(c) == "t1.name"

    def test_without_table(self):
        c = ColumnRef(column="name")
        assert repr(c) == "name"
        assert c.table is None


class TestLiteral:
    def test_string_literal(self):
        lit = Literal(value="DE", dtype="str")
        assert repr(lit) == "'DE'"

    def test_int_literal(self):
        lit = Literal(value=42)
        assert repr(lit) == "42"

    def test_bool_literal(self):
        lit = Literal(value=True)
        assert repr(lit) == "True"

    def test_none_literal(self):
        lit = Literal(value=None)
        assert repr(lit) == "NULL"


class TestFuncCall:
    def test_simple_func(self):
        f = FuncCall(name="COUNT", args=[ColumnRef(column="id")])
        assert f.name == "COUNT"
        assert not f.distinct

    def test_distinct_func(self):
        f = FuncCall(
            name="COUNT",
            args=[ColumnRef(column="customer_id")],
            distinct=True,
        )
        assert f.distinct


class TestCompareExpr:
    def test_eq(self):
        e = CompareExpr(
            left=ColumnRef(column="country", table="t1"),
            op=CompareOp.EQ,
            right=Literal(value="DE"),
        )
        assert e.op == CompareOp.EQ

    def test_all_ops(self):
        for op in CompareOp:
            e = CompareExpr(left=ColumnRef("x"), op=op, right=Literal(1))
            assert e.op == op


class TestLogicExpr:
    def test_and(self):
        e = LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(ColumnRef("a"), CompareOp.EQ, Literal(1)),
                CompareExpr(ColumnRef("b"), CompareOp.GT, Literal(0)),
            ],
        )
        assert len(e.operands) == 2

    def test_not(self):
        inner = CompareExpr(ColumnRef("x"), CompareOp.EQ, Literal("y"))
        e = LogicExpr(op=LogicOp.NOT, operands=[inner])
        assert e.op == LogicOp.NOT


class TestInExpr:
    def test_in_list(self):
        e = InExpr(
            value=ColumnRef("status"),
            items=[Literal("shipped"), Literal("delivered")],
        )
        assert not e.negated

    def test_not_in(self):
        e = InExpr(value=ColumnRef("status"), items=[Literal("pending")], negated=True)
        assert e.negated


class TestBetweenExpr:
    def test_between(self):
        e = BetweenExpr(value=ColumnRef("total"), lo=Literal(100), hi=Literal(500))
        assert not e.negated

    def test_not_between(self):
        e = BetweenExpr(
            value=ColumnRef("total"), lo=Literal(100), hi=Literal(500), negated=True
        )
        assert e.negated


class TestLikeExpr:
    def test_like(self):
        e = LikeExpr(value=ColumnRef("name"), pattern="%Epp%")
        assert e.pattern == "%Epp%"
        assert not e.negated

    def test_not_like_with_escape(self):
        e = LikeExpr(value=ColumnRef("name"), pattern="%x%", escape="\\", negated=True)
        assert e.escape == "\\"
        assert e.negated


class TestNullExpr:
    def test_is_null(self):
        e = NullExpr(value=ColumnRef("email"))
        assert not e.negated

    def test_is_not_null(self):
        e = NullExpr(value=ColumnRef("email"), negated=True)
        assert e.negated


class TestExistsExpr:
    def test_exists(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=scan)
        e = ExistsExpr(subquery=sub)
        assert not e.negated

    def test_not_exists(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=scan)
        e = ExistsExpr(subquery=sub, negated=True)
        assert e.negated


# ─── Hilfs-Datenklassen ───────────────────────────────────────────────────────

class TestProjectionItem:
    def test_star(self):
        p = ProjectionItem(expr=ColumnRef("*"), star=True)
        assert p.star

    def test_aliased(self):
        p = ProjectionItem(expr=ColumnRef("id"), alias="customer_id")
        assert p.alias == "customer_id"


class TestSortItem:
    def test_default_asc(self):
        s = SortItem(expr=ColumnRef("name"))
        assert s.direction == SortDirection.ASC
        assert s.nulls == NullsOrder.DEFAULT

    def test_desc_nulls_first(self):
        s = SortItem(
            expr=ColumnRef("total"),
            direction=SortDirection.DESC,
            nulls=NullsOrder.NULLS_FIRST,
        )
        assert s.direction == SortDirection.DESC


class TestAggItem:
    def test_count_star(self):
        a = AggItem(func=AggFunc.COUNT, arg=None)
        assert a.arg is None

    def test_sum_with_alias(self):
        a = AggItem(func=AggFunc.SUM, arg=ColumnRef("total"), alias="total_sum")
        assert a.alias == "total_sum"

    def test_count_distinct(self):
        a = AggItem(func=AggFunc.COUNT_DISTINCT, arg=ColumnRef("customer_id"))
        assert a.func == AggFunc.COUNT_DISTINCT


class TestWindowFrame:
    def test_rows_unbounded(self):
        f = WindowFrame(unit=FrameUnit.ROWS)
        assert f.start == "UNBOUNDED PRECEDING"
        assert f.end == "CURRENT ROW"

    def test_range_custom(self):
        f = WindowFrame(unit=FrameUnit.RANGE, start="1 PRECEDING", end="1 FOLLOWING")
        assert f.start == "1 PRECEDING"


class TestWindowDef:
    def test_rank_over_partition(self):
        w = WindowDef(
            func=WindowFunc.RANK,
            result_alias="rnk",
            partition_by=[ColumnRef("customer_id", "t1")],
            order_by=[SortItem(ColumnRef("total", "t1"), SortDirection.DESC)],
        )
        assert w.func == WindowFunc.RANK
        assert w.result_alias == "rnk"


class TestJoinCondition:
    def test_on_condition(self):
        pred = CompareExpr(
            ColumnRef("id", "t1"), CompareOp.EQ, ColumnRef("customer_id", "t2")
        )
        cond = OnCondition(predicate=pred)
        assert cond.predicate == pred

    def test_using_condition(self):
        cond = UsingCondition(columns=["customer_id"])
        assert "customer_id" in cond.columns


# ─── IQT-Knoten ───────────────────────────────────────────────────────────────

class TestScanNode:
    def test_basic(self):
        n = ScanNode(node_type=NodeType.SCAN, table="customers")
        assert n.table == "customers"
        assert n.node_type == NodeType.SCAN

    def test_with_alias(self):
        n = ScanNode(node_type=NodeType.SCAN, table="customers", alias="t1")
        assert n.alias == "t1"


class TestFilterNode:
    def test_where(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="customers")
        pred = CompareExpr(ColumnRef("active"), CompareOp.EQ, Literal(True))
        f = FilterNode(
            node_type=NodeType.FILTER, source=scan, predicate=pred
        )
        assert f.position == FilterPosition.WHERE
        assert f.source is scan

    def test_having(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        pred = CompareExpr(
            FuncCall("COUNT", [ColumnRef("id")]), CompareOp.GT, Literal(3)
        )
        f = FilterNode(
            node_type=NodeType.FILTER,
            source=scan,
            predicate=pred,
            position=FilterPosition.HAVING,
        )
        assert f.position == FilterPosition.HAVING


class TestProjectNode:
    def test_columns(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="customers")
        cols = [
            ProjectionItem(expr=ColumnRef("name")),
            ProjectionItem(expr=ColumnRef("email")),
        ]
        p = ProjectNode(node_type=NodeType.PROJECT, source=scan, columns=cols)
        assert len(p.columns) == 2
        assert not p.distinct

    def test_distinct(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="customers")
        p = ProjectNode(
            node_type=NodeType.PROJECT,
            source=scan,
            columns=[ProjectionItem(expr=ColumnRef("email"))],
            distinct=True,
        )
        assert p.distinct


class TestJoinNode:
    def test_inner_join(self):
        scan_c = ScanNode(node_type=NodeType.SCAN, table="customers", alias="t1")
        scan_o = ScanNode(node_type=NodeType.SCAN, table="orders", alias="t2")
        cond = OnCondition(
            predicate=CompareExpr(
                ColumnRef("id", "t1"), CompareOp.EQ, ColumnRef("customer_id", "t2")
            )
        )
        j = JoinNode(
            node_type=NodeType.JOIN,
            left=scan_c,
            right=scan_o,
            join_type=JoinType.INNER,
            condition=cond,
        )
        assert j.join_type == JoinType.INNER

    def test_left_outer_join(self):
        scan_c = ScanNode(node_type=NodeType.SCAN, table="customers", alias="t1")
        scan_o = ScanNode(node_type=NodeType.SCAN, table="orders", alias="t2")
        cond = OnCondition(
            predicate=CompareExpr(
                ColumnRef("id", "t1"), CompareOp.EQ, ColumnRef("customer_id", "t2")
            )
        )
        j = JoinNode(
            node_type=NodeType.JOIN,
            left=scan_c,
            right=scan_o,
            join_type=JoinType.LEFT_OUTER,
            condition=cond,
        )
        assert j.join_type == JoinType.LEFT_OUTER


class TestGroupAggNode:
    def test_group_by_with_having(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        agg = AggItem(func=AggFunc.COUNT, arg=ColumnRef("id"), alias="cnt")
        having = CompareExpr(
            FuncCall("COUNT", [ColumnRef("id")]), CompareOp.GT, Literal(3)
        )
        g = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=scan,
            group_by=[ColumnRef("customer_id")],
            aggregates=[agg],
            having=having,
        )
        assert len(g.group_by) == 1
        assert g.having is not None


class TestSortNode:
    def test_multi_sort(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="customers")
        items = [
            SortItem(ColumnRef("name"), SortDirection.ASC),
            SortItem(ColumnRef("email"), SortDirection.DESC),
        ]
        s = SortNode(node_type=NodeType.SORT, source=scan, items=items)
        assert len(s.items) == 2


class TestLimitNode:
    def test_limit_offset(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="products")
        lim = LimitNode(node_type=NodeType.LIMIT, source=scan, count=10, offset=20)
        assert lim.count == 10
        assert lim.offset == 20

    def test_limit_only(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="products")
        lim = LimitNode(node_type=NodeType.LIMIT, source=scan, count=5)
        assert lim.offset is None


class TestWindowNode:
    def test_window_definition(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        wdef = WindowDef(
            func=WindowFunc.RANK,
            result_alias="rnk",
            partition_by=[ColumnRef("customer_id")],
            order_by=[SortItem(ColumnRef("total"), SortDirection.DESC)],
        )
        w = WindowNode(node_type=NodeType.WINDOW, source=scan, definitions=[wdef])
        assert len(w.definitions) == 1


class TestSubqueryNode:
    def test_subquery(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=scan, alias="sub")
        assert sub.alias == "sub"
        assert sub.query is scan


class TestSetOpNode:
    def test_union(self):
        s1 = ScanNode(node_type=NodeType.SCAN, table="customers")
        s2 = ScanNode(node_type=NodeType.SCAN, table="archived_customers")
        sop = SetOpNode(node_type=NodeType.SET_OP, left=s1, right=s2, op=SetOpType.UNION)
        assert sop.op == SetOpType.UNION


class TestWithNode:
    def test_with_cte(self):
        scan = ScanNode(node_type=NodeType.SCAN, table="orders")
        cte_def = CTEDef(name="revenue_cte", query=scan)
        main = ScanNode(node_type=NodeType.SCAN, table="customers")
        with_node = WithNode(
            node_type=NodeType.WITH, ctes=[cte_def], query=main
        )
        assert len(with_node.ctes) == 1
        assert with_node.ctes[0].name == "revenue_cte"
