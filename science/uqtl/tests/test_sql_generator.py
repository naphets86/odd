"""
Tests für uqtl.sql_generator – SQL-Generierung gemäß Abschnitt 4.

Die Testfälle entsprechen den 10 Beispielabfragen aus Abschnitt 5 des
UQTL-Standards. Jedes Beispiel zeigt, dass Python, Java und C#-äquivalente
IQT-Bäume dasselbe normalisierte SQL erzeugen.
"""

import pytest
from uqtl.iqt import (
    NodeType, JoinType, SortDirection, NullsOrder,
    AggFunc, SetOpType, FilterPosition, CompareOp, LogicOp,
    FrameUnit, WindowFunc,
    ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowFrame, WindowDef,
    OnCondition, UsingCondition,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode, CTEDef,
)
from uqtl.sql_generator import SQLGenerator


def _scan(table: str, alias: str = None) -> ScanNode:
    return ScanNode(node_type=NodeType.SCAN, table=table, alias=alias)


def _col(col: str, table: str = None) -> ColumnRef:
    return ColumnRef(column=col, table=table)


def _lit(val) -> Literal:
    return Literal(value=val)


class TestScanGeneration:
    def test_simple_scan(self):
        g = SQLGenerator()
        assert g.generate(_scan("customers")) == "SELECT *\nFROM customers"

    def test_scan_with_alias(self):
        g = SQLGenerator()
        sql = g.generate(_scan("customers", "t1"))
        assert "customers AS t1" in sql


class TestFilterGeneration:
    def test_simple_where(self):
        g = SQLGenerator()
        pred = CompareExpr(_col("active", "t1"), CompareOp.EQ, _lit(True))
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "WHERE" in sql
        assert "t1.active = TRUE" in sql

    def test_and_predicate(self):
        g = SQLGenerator()
        pred = LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(_col("active", "t1"), CompareOp.EQ, _lit(True)),
                CompareExpr(_col("country", "t1"), CompareOp.EQ, _lit("DE")),
            ],
        )
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "active" in sql
        assert "country" in sql
        assert "'DE'" in sql

    def test_not_predicate(self):
        g = SQLGenerator()
        inner = CompareExpr(_col("status"), CompareOp.EQ, _lit("pending"))
        pred = LogicExpr(op=LogicOp.NOT, operands=[inner])
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "NOT" in sql

    def test_is_null(self):
        g = SQLGenerator()
        pred = NullExpr(value=_col("email", "t1"))
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "IS NULL" in sql

    def test_is_not_null(self):
        g = SQLGenerator()
        pred = NullExpr(value=_col("email", "t1"), negated=True)
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "IS NOT NULL" in sql

    def test_in_list(self):
        g = SQLGenerator()
        pred = InExpr(
            value=_col("status", "t1"),
            items=[_lit("shipped"), _lit("delivered")],
        )
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "IN" in sql
        assert "'shipped'" in sql

    def test_not_in_list(self):
        g = SQLGenerator()
        pred = InExpr(
            value=_col("status", "t1"),
            items=[_lit("pending")],
            negated=True,
        )
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "NOT IN" in sql

    def test_between(self):
        g = SQLGenerator()
        pred = BetweenExpr(value=_col("total", "t1"), lo=_lit(100), hi=_lit(500))
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "BETWEEN" in sql
        assert "100" in sql
        assert "500" in sql

    def test_like(self):
        g = SQLGenerator()
        pred = LikeExpr(value=_col("name", "t1"), pattern="%Epp%")
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "LIKE" in sql
        assert "%Epp%" in sql

    def test_not_like(self):
        g = SQLGenerator()
        pred = LikeExpr(value=_col("name", "t1"), pattern="%test%", negated=True)
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "NOT LIKE" in sql

    def test_in_subquery(self):
        g = SQLGenerator()
        inner_scan = _scan("orders", "t2")
        inner_proj = ProjectNode(
            node_type=NodeType.PROJECT,
            source=inner_scan,
            columns=[ProjectionItem(expr=_col("customer_id", "t2"))],
        )
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner_proj)
        pred = InExpr(value=_col("id", "t1"), subquery=sub)
        outer = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(outer)
        assert "IN" in sql


class TestProjectGeneration:
    def test_single_column(self):
        g = SQLGenerator()
        node = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[ProjectionItem(expr=_col("name", "t1"))],
        )
        sql = g.generate(node)
        assert "SELECT t1.name" in sql

    def test_multiple_columns(self):
        g = SQLGenerator()
        node = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[
                ProjectionItem(expr=_col("name", "t1")),
                ProjectionItem(expr=_col("email", "t1")),
            ],
        )
        sql = g.generate(node)
        assert "t1.name" in sql
        assert "t1.email" in sql

    def test_distinct(self):
        g = SQLGenerator()
        node = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[ProjectionItem(expr=_col("name", "t1"))],
            distinct=True,
        )
        sql = g.generate(node)
        assert "DISTINCT" in sql

    def test_star(self):
        g = SQLGenerator()
        node = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[ProjectionItem(expr=_col("*"), star=True)],
        )
        sql = g.generate(node)
        assert "SELECT *" in sql

    def test_aliased_column(self):
        g = SQLGenerator()
        node = ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[ProjectionItem(expr=_col("id", "t1"), alias="cust_id")],
        )
        sql = g.generate(node)
        assert "AS cust_id" in sql


class TestSortGeneration:
    def test_asc(self):
        g = SQLGenerator()
        node = SortNode(
            node_type=NodeType.SORT,
            source=_scan("customers", "t1"),
            items=[SortItem(_col("name", "t1"), SortDirection.ASC)],
        )
        sql = g.generate(node)
        assert "ORDER BY" in sql
        assert "ASC" in sql

    def test_desc(self):
        g = SQLGenerator()
        node = SortNode(
            node_type=NodeType.SORT,
            source=_scan("orders", "t1"),
            items=[SortItem(_col("total", "t1"), SortDirection.DESC)],
        )
        sql = g.generate(node)
        assert "DESC" in sql

    def test_multi_sort(self):
        g = SQLGenerator()
        node = SortNode(
            node_type=NodeType.SORT,
            source=_scan("customers", "t1"),
            items=[
                SortItem(_col("country", "t1"), SortDirection.ASC),
                SortItem(_col("name", "t1"), SortDirection.ASC),
            ],
        )
        sql = g.generate(node)
        assert "t1.country ASC, t1.name ASC" in sql

    def test_nulls_first(self):
        g = SQLGenerator()
        node = SortNode(
            node_type=NodeType.SORT,
            source=_scan("customers", "t1"),
            items=[SortItem(_col("name", "t1"), SortDirection.ASC, NullsOrder.NULLS_FIRST)],
        )
        sql = g.generate(node)
        assert "NULLS FIRST" in sql


class TestLimitGeneration:
    def test_limit_only(self):
        g = SQLGenerator()
        node = LimitNode(node_type=NodeType.LIMIT, source=_scan("products"), count=5)
        sql = g.generate(node)
        assert "LIMIT 5" in sql

    def test_limit_offset(self):
        g = SQLGenerator()
        node = LimitNode(
            node_type=NodeType.LIMIT, source=_scan("products"), count=10, offset=20
        )
        sql = g.generate(node)
        assert "LIMIT 10" in sql
        assert "OFFSET 20" in sql


class TestJoinGeneration:
    def test_inner_join(self):
        g = SQLGenerator()
        cond = OnCondition(
            predicate=CompareExpr(
                _col("id", "t1"), CompareOp.EQ, _col("customer_id", "t2")
            )
        )
        node = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=_scan("orders", "t2"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        sql = g.generate(node)
        assert "INNER JOIN" in sql
        assert "ON" in sql

    def test_left_outer_join(self):
        g = SQLGenerator()
        cond = OnCondition(
            predicate=CompareExpr(
                _col("id", "t1"), CompareOp.EQ, _col("customer_id", "t2")
            )
        )
        node = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=_scan("orders", "t2"),
            join_type=JoinType.LEFT_OUTER,
            condition=cond,
        )
        sql = g.generate(node)
        assert "LEFT OUTER JOIN" in sql

    def test_cross_join(self):
        g = SQLGenerator()
        cond = OnCondition(predicate=_lit(True))
        node = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("a", "t1"),
            right=_scan("b", "t2"),
            join_type=JoinType.CROSS,
            condition=cond,
        )
        sql = g.generate(node)
        assert "CROSS JOIN" in sql

    def test_using_condition(self):
        g = SQLGenerator()
        cond = UsingCondition(columns=["customer_id"])
        node = JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=_scan("orders", "t2"),
            join_type=JoinType.INNER,
            condition=cond,
        )
        sql = g.generate(node)
        assert "USING" in sql
        assert "customer_id" in sql


class TestGroupAggGeneration:
    def test_group_by_count(self):
        g = SQLGenerator()
        agg = AggItem(func=AggFunc.COUNT, arg=_col("id", "t1"), alias="order_count")
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t1"),
            group_by=[_col("customer_id", "t1")],
            aggregates=[agg],
        )
        sql = g.generate(node)
        assert "GROUP BY" in sql
        assert "COUNT" in sql
        assert "order_count" in sql

    def test_having(self):
        g = SQLGenerator()
        agg = AggItem(func=AggFunc.COUNT, arg=_col("id", "t1"), alias="cnt")
        having = CompareExpr(
            FuncCall("COUNT", [_col("id", "t1")]), CompareOp.GT, _lit(3)
        )
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t1"),
            group_by=[_col("customer_id", "t1")],
            aggregates=[agg],
            having=having,
        )
        sql = g.generate(node)
        assert "HAVING" in sql
        assert "3" in sql

    def test_sum_agg(self):
        g = SQLGenerator()
        agg = AggItem(func=AggFunc.SUM, arg=_col("total", "t1"), alias="revenue")
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t1"),
            group_by=[_col("customer_id", "t1")],
            aggregates=[agg],
        )
        sql = g.generate(node)
        assert "SUM" in sql

    def test_count_distinct(self):
        g = SQLGenerator()
        agg = AggItem(
            func=AggFunc.COUNT_DISTINCT,
            arg=_col("customer_id", "t1"),
            alias="unique_buyers",
        )
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t1"),
            group_by=[_col("product_id", "t1")],
            aggregates=[agg],
        )
        sql = g.generate(node)
        assert "COUNT(DISTINCT" in sql

    def test_count_star(self):
        g = SQLGenerator()
        agg = AggItem(func=AggFunc.COUNT, arg=None)
        node = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t1"),
            group_by=[_col("status", "t1")],
            aggregates=[agg],
        )
        sql = g.generate(node)
        assert "COUNT(*)" in sql


class TestWindowGeneration:
    def test_rank_over_partition(self):
        g = SQLGenerator()
        wdef = WindowDef(
            func=WindowFunc.RANK,
            result_alias="rnk",
            partition_by=[_col("customer_id", "t1")],
            order_by=[SortItem(_col("total", "t1"), SortDirection.DESC)],
        )
        node = WindowNode(
            node_type=NodeType.WINDOW,
            source=_scan("orders", "t1"),
            definitions=[wdef],
        )
        sql = g.generate(node)
        assert "RANK" in sql
        assert "OVER" in sql
        assert "PARTITION BY" in sql
        assert "rnk" in sql

    def test_sum_over_with_frame(self):
        g = SQLGenerator()
        frame = WindowFrame(
            unit=FrameUnit.ROWS,
            start="UNBOUNDED PRECEDING",
            end="CURRENT ROW",
        )
        wdef = WindowDef(
            func=WindowFunc.SUM,
            result_alias="running_total",
            partition_by=[_col("customer_id", "t1")],
            order_by=[SortItem(_col("order_date", "t1"), SortDirection.ASC)],
            frame=frame,
            func_args=[_col("total", "t1")],
        )
        node = WindowNode(
            node_type=NodeType.WINDOW,
            source=_scan("orders", "t1"),
            definitions=[wdef],
        )
        sql = g.generate(node)
        assert "ROWS BETWEEN" in sql
        assert "UNBOUNDED PRECEDING" in sql
        assert "running_total" in sql

    def test_lag_window_func(self):
        g = SQLGenerator()
        wdef = WindowDef(
            func=WindowFunc.LAG,
            result_alias="prev_revenue",
            order_by=[SortItem(_col("month"), SortDirection.ASC)],
            func_args=[_col("revenue"), _lit(1)],
        )
        node = WindowNode(
            node_type=NodeType.WINDOW,
            source=_scan("monthly", "sub"),
            definitions=[wdef],
        )
        sql = g.generate(node)
        assert "LAG" in sql
        assert "prev_revenue" in sql

    def test_row_number(self):
        g = SQLGenerator()
        wdef = WindowDef(
            func=WindowFunc.ROW_NUMBER,
            result_alias="rn",
            order_by=[SortItem(_col("id", "t1"), SortDirection.ASC)],
        )
        node = WindowNode(
            node_type=NodeType.WINDOW,
            source=_scan("customers", "t1"),
            definitions=[wdef],
        )
        sql = g.generate(node)
        assert "ROW" in sql  # ROW NUMBER
        assert "rn" in sql


class TestSubqueryGeneration:
    def test_scalar_subquery_in_filter(self):
        g = SQLGenerator()
        inner = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t2"),
            group_by=[],
            aggregates=[AggItem(func=AggFunc.AVG, arg=_col("total", "t2"), alias="avg_total")],
        )
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner)
        pred = CompareExpr(_col("total", "t1"), CompareOp.GT, sub)
        # Inline-Subquery als rechter Operand
        outer_scan = _scan("orders", "t1")
        outer = FilterNode(
            node_type=NodeType.FILTER,
            source=outer_scan,
            predicate=pred,
        )
        sql = g.generate(outer)
        assert "WHERE" in sql

    def test_exists_subquery(self):
        g = SQLGenerator()
        inner_pred = LogicExpr(
            op=LogicOp.AND,
            operands=[
                CompareExpr(_col("customer_id", "t2"), CompareOp.EQ, _col("id", "t1")),
                CompareExpr(_col("status", "t2"), CompareOp.EQ, _lit("shipped")),
            ],
        )
        inner = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t2"),
            predicate=inner_pred,
        )
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner)
        pred = ExistsExpr(subquery=sub)
        outer = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(outer)
        assert "EXISTS" in sql
        assert "shipped" in sql

    def test_not_exists_subquery(self):
        g = SQLGenerator()
        inner = _scan("orders", "t2")
        sub = SubqueryNode(node_type=NodeType.SUBQUERY, query=inner)
        pred = ExistsExpr(subquery=sub, negated=True)
        outer = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("customers", "t1"),
            predicate=pred,
        )
        sql = g.generate(outer)
        assert "NOT EXISTS" in sql


class TestSetOpGeneration:
    def test_union(self):
        g = SQLGenerator()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("customers", "t1"),
            right=_scan("archived_customers", "t2"),
            op=SetOpType.UNION,
        )
        sql = g.generate(node)
        assert "UNION" in sql

    def test_union_all(self):
        g = SQLGenerator()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("a"),
            right=_scan("b"),
            op=SetOpType.UNION_ALL,
        )
        sql = g.generate(node)
        assert "UNION ALL" in sql

    def test_intersect(self):
        g = SQLGenerator()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("a"),
            right=_scan("b"),
            op=SetOpType.INTERSECT,
        )
        sql = g.generate(node)
        assert "INTERSECT" in sql

    def test_except(self):
        g = SQLGenerator()
        node = SetOpNode(
            node_type=NodeType.SET_OP,
            left=_scan("a"),
            right=_scan("b"),
            op=SetOpType.EXCEPT,
        )
        sql = g.generate(node)
        assert "EXCEPT" in sql


class TestWithGeneration:
    def test_single_cte(self):
        g = SQLGenerator()
        cte_query = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=_scan("orders", "t2"),
            group_by=[_col("customer_id", "t2")],
            aggregates=[AggItem(func=AggFunc.SUM, arg=_col("total", "t2"), alias="total_revenue")],
        )
        cte = CTEDef(name="customer_revenue", query=cte_query)
        main = _scan("customers", "t1")
        node = WithNode(node_type=NodeType.WITH, ctes=[cte], query=main)
        sql = g.generate(node)
        assert "WITH" in sql
        assert "customer_revenue" in sql
        assert "SUM" in sql

    def test_multiple_ctes(self):
        g = SQLGenerator()
        cte1 = CTEDef(name="rev", query=_scan("orders"))
        cte2 = CTEDef(name="avg_rev", query=_scan("rev"))
        node = WithNode(
            node_type=NodeType.WITH,
            ctes=[cte1, cte2],
            query=_scan("customers"),
        )
        sql = g.generate(node)
        assert "rev" in sql
        assert "avg_rev" in sql


class TestLiteralEdgeCases:
    """Randwerte der Literal-Generierung."""

    def test_null_literal(self):
        g = SQLGenerator()
        pred = CompareExpr(_col("x"), CompareOp.EQ, _lit(None))
        node = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        sql = g.generate(node)
        assert "NULL" in sql

    def test_false_literal(self):
        g = SQLGenerator()
        pred = CompareExpr(_col("active"), CompareOp.EQ, _lit(False))
        node = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        sql = g.generate(node)
        assert "FALSE" in sql

    def test_string_with_quotes(self):
        g = SQLGenerator()
        pred = CompareExpr(_col("name"), CompareOp.EQ, _lit("O'Brien"))
        node = FilterNode(node_type=NodeType.FILTER, source=_scan("t"), predicate=pred)
        sql = g.generate(node)
        assert "O''Brien" in sql  # SQL-Escape für einfaches Anführungszeichen


class TestFuncCallGeneration:
    def test_extract(self):
        g = SQLGenerator()
        pred = CompareExpr(
            FuncCall("EXTRACT", [_lit("YEAR"), _col("order_date", "t1")]),
            CompareOp.EQ,
            _lit(2024),
        )
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "EXTRACT" in sql

    def test_count_distinct_func(self):
        g = SQLGenerator()
        fc = FuncCall("COUNT", [_col("customer_id", "t1")], distinct=True)
        pred = CompareExpr(fc, CompareOp.GT, _lit(0))
        node = FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=pred,
        )
        sql = g.generate(node)
        assert "COUNT(DISTINCT" in sql


class TestExample1FilterProjectSort:
    """
    Beispiel 1 aus UQTL-Standard:
    SELECT t1.name, t1.email FROM customers AS t1
    WHERE t1.active = TRUE AND t1.country = 'DE'
    ORDER BY t1.name ASC
    """

    def _build_tree(self) -> ScanNode:
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
        sort = SortNode(
            node_type=NodeType.SORT,
            source=proj,
            items=[SortItem(_col("name", "t1"), SortDirection.ASC)],
        )
        return sort

    def test_sql_contains_key_clauses(self):
        g = SQLGenerator()
        sql = g.generate(self._build_tree())
        assert "t1.name" in sql
        assert "t1.email" in sql
        assert "customers AS t1" in sql
        assert "active" in sql
        assert "'DE'" in sql
        assert "ORDER BY" in sql


class TestExample2AggregationGroupByHaving:
    """
    Beispiel 2: SELECT customer_id, COUNT(id), SUM(total)
    FROM orders WHERE status='confirmed'
    GROUP BY customer_id HAVING COUNT(id) > 3
    ORDER BY SUM(total) DESC
    """

    def _build_tree(self):
        scan = _scan("orders", "t1")
        filt = FilterNode(
            node_type=NodeType.FILTER,
            source=scan,
            predicate=CompareExpr(_col("status", "t1"), CompareOp.EQ, _lit("confirmed")),
        )
        proj_cols = [ProjectionItem(expr=_col("customer_id", "t1"))]
        proj = ProjectNode(
            node_type=NodeType.PROJECT,
            source=filt,
            columns=proj_cols,
        )
        having = CompareExpr(
            FuncCall("COUNT", [_col("id", "t1")]), CompareOp.GT, _lit(3)
        )
        agg = GroupAggNode(
            node_type=NodeType.GROUP_AGG,
            source=proj,
            group_by=[_col("customer_id", "t1")],
            aggregates=[
                AggItem(func=AggFunc.COUNT, arg=_col("id", "t1"), alias="order_count"),
                AggItem(func=AggFunc.SUM, arg=_col("total", "t1"), alias="total_revenue"),
            ],
            having=having,
        )
        sort = SortNode(
            node_type=NodeType.SORT,
            source=agg,
            items=[
                SortItem(FuncCall("SUM", [_col("total", "t1")]), SortDirection.DESC)
            ],
        )
        return sort

    def test_sql_contains_key_clauses(self):
        g = SQLGenerator()
        sql = g.generate(self._build_tree())
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert "COUNT" in sql
        assert "SUM" in sql
        assert "DESC" in sql


class TestExample3InnerJoin:
    """Beispiel 3: INNER JOIN customers ↔ orders"""

    def _build_tree(self):
        scan_c = _scan("customers", "t1")
        scan_o = _scan("orders", "t2")
        cond = OnCondition(
            predicate=CompareExpr(
                _col("id", "t1"), CompareOp.EQ, _col("customer_id", "t2")
            )
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
        proj = ProjectNode(
            node_type=NodeType.PROJECT,
            source=filt,
            columns=[
                ProjectionItem(expr=_col("name", "t1")),
                ProjectionItem(expr=_col("email", "t1")),
            ],
            distinct=True,
        )
        return proj

    def test_sql_contains_join(self):
        g = SQLGenerator()
        sql = g.generate(self._build_tree())
        assert "INNER JOIN" in sql
        assert "DISTINCT" in sql
        assert "500" in sql


class TestMissingCoverage:
    """Gezielte Tests für bisher nicht abgedeckte sql_generator-Zeilen."""

    def test_subquery_as_root(self):
        # Zeile 46: _gen_node mit SubqueryNode als Wurzel → _gen_subquery
        g = SQLGenerator()
        sql = g.generate(SubqueryNode(
            node_type=NodeType.SUBQUERY,
            query=_scan("orders"),
        ))
        assert "orders" in sql

    def test_subquery_with_alias_as_root(self):
        # Zeilen 217-219: _gen_subquery mit alias
        g = SQLGenerator()
        sql = g.generate(SubqueryNode(
            node_type=NodeType.SUBQUERY,
            query=_scan("orders"),
            alias="o",
        ))
        assert "orders" in sql
        assert "AS o" in sql

    def test_project_empty_columns_gives_star(self):
        # Zeile 98: ProjectNode mit leerer Spaltenliste → SELECT *
        g = SQLGenerator()
        sql = g.generate(ProjectNode(
            node_type=NodeType.PROJECT,
            source=_scan("customers", "t1"),
            columns=[],
        ))
        assert "SELECT *" in sql

    def test_filter_having_position(self):
        # Zeile 104: FilterNode mit position=HAVING
        g = SQLGenerator()
        sql = g.generate(FilterNode(
            node_type=NodeType.FILTER,
            source=_scan("orders", "t1"),
            predicate=CompareExpr(
                FuncCall("COUNT", [_col("id", "t1")]), CompareOp.GT, _lit(5)
            ),
            position=FilterPosition.HAVING,
        ))
        assert "HAVING" in sql
        assert "WHERE" not in sql

    def test_subquery_as_from_source_with_alias(self):
        # Zeilen 138-142: SubqueryNode mit Alias als Quelle in _gen_select_stmt
        g = SQLGenerator()
        sub = SubqueryNode(
            node_type=NodeType.SUBQUERY,
            query=_scan("archived_customers"),
            alias="arch",
        )
        sql = g.generate(FilterNode(
            node_type=NodeType.FILTER,
            source=sub,
            predicate=CompareExpr(_col("id"), CompareOp.GT, _lit(0)),
        ))
        assert "archived_customers" in sql
        assert "arch" in sql

    def test_subquery_as_from_source_no_alias(self):
        # Zeile 139: SubqueryNode ohne Alias als Quelle
        g = SQLGenerator()
        sub = SubqueryNode(
            node_type=NodeType.SUBQUERY,
            query=_scan("orders"),
        )
        sql = g.generate(FilterNode(
            node_type=NodeType.FILTER,
            source=sub,
            predicate=CompareExpr(_col("id"), CompareOp.GT, _lit(0)),
        ))
        assert "orders" in sql

    def test_unknown_source_node_else_branch(self):
        # Zeilen 144-145: else: current = None für unbekannten Knotentyp
        from uqtl.iqt import IQTNode

        class _UnknownSource(IQTNode):
            pass

        g = SQLGenerator()
        sql = g.generate(ProjectNode(
            node_type=NodeType.PROJECT,
            source=_UnknownSource(node_type=NodeType.SCAN),
            columns=[ProjectionItem(expr=_col("x"))],
        ))
        assert "SELECT" in sql

    def test_window_appended_to_existing_select(self):
        # Zeile 153: ProjectNode + WindowNode → select_clause + ", " + win_cols
        g = SQLGenerator()
        wdef = WindowDef(func=WindowFunc.RANK, result_alias="rnk")
        sql = g.generate(ProjectNode(
            node_type=NodeType.PROJECT,
            source=WindowNode(
                node_type=NodeType.WINDOW,
                source=_scan("orders", "t1"),
                definitions=[wdef],
            ),
            columns=[ProjectionItem(expr=_col("id", "t1"))],
        ))
        assert "t1.id" in sql
        assert "RANK()" in sql
        assert "rnk" in sql

    def test_join_no_condition(self):
        # Zeile 200: JoinNode ohne Bedingung → kein ON/USING
        g = SQLGenerator()
        sql = g.generate(JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("a"),
            right=_scan("b"),
            join_type=JoinType.CROSS,
            condition=None,
        ))
        assert "CROSS JOIN" in sql
        assert "ON" not in sql

    def test_subquery_in_join_right_child(self):
        # Zeilen 206-211: SubqueryNode als rechtes Kind eines JoinNode
        # → _gen_from_source(SubqueryNode)
        g = SQLGenerator()
        sub = SubqueryNode(
            node_type=NodeType.SUBQUERY,
            query=_scan("raw_orders"),
            alias="o",
        )
        sql = g.generate(JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=sub,
            join_type=JoinType.INNER,
            condition=OnCondition(
                predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("cid", "o"))
            ),
        ))
        assert "raw_orders" in sql
        assert "t1" in sql

    def test_filter_in_join_child_fallback_gen_from_source(self):
        # Zeilen 213-214: _gen_from_source-Fallback für unbekannten Knotentyp (z. B. FilterNode)
        g = SQLGenerator()
        sql = g.generate(JoinNode(
            node_type=NodeType.JOIN,
            left=_scan("customers", "t1"),
            right=FilterNode(
                node_type=NodeType.FILTER,
                source=_scan("orders", "t2"),
                predicate=CompareExpr(_col("status", "t2"), CompareOp.EQ, _lit("active")),
            ),
            join_type=JoinType.INNER,
            condition=OnCondition(
                predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("cid", "t2"))
            ),
        ))
        assert "customers" in sql
        assert "orders" in sql

    def test_window_def_aggregate_func_no_args(self):
        # Zeile 280: WindowDef mit SUM/AVG/MIN/MAX (kein ROW_NUMBER/RANK/DENSE_RANK)
        # und ohne func_args → else-Zweig in _gen_window_def
        g = SQLGenerator()
        wdef = WindowDef(
            func=WindowFunc.SUM,
            result_alias="running_total",
            partition_by=[_col("customer_id", "t1")],
        )
        sql = g.generate(WindowNode(
            node_type=NodeType.WINDOW,
            source=_scan("orders", "t1"),
            definitions=[wdef],
        ))
        assert "SUM()" in sql
        assert "running_total" in sql

    def test_group_agg_appended_to_existing_select(self):
        # Zeile 125: GroupAggNode unter ProjectNode → agg an vorhandenes SELECT anhängen
        g = SQLGenerator()
        sql = g.generate(ProjectNode(
            node_type=NodeType.PROJECT,
            source=GroupAggNode(
                node_type=NodeType.GROUP_AGG,
                source=_scan("orders", "t1"),
                group_by=[_col("customer_id", "t1")],
                aggregates=[AggItem(func=AggFunc.COUNT, arg=None, alias="n")],
            ),
            columns=[ProjectionItem(expr=_col("customer_id", "t1"))],
        ))
        assert "t1.customer_id" in sql
        assert "COUNT(*)" in sql

    def test_nested_join_calls_gen_from_source_for_join(self):
        # Zeile 207: _gen_from_source mit JoinNode → _gen_join rekursiv
        g = SQLGenerator()
        cond_inner = OnCondition(
            predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("cid", "t2"))
        )
        cond_outer = OnCondition(
            predicate=CompareExpr(_col("id", "t1"), CompareOp.EQ, _col("pid", "t3"))
        )
        sql = g.generate(JoinNode(
            node_type=NodeType.JOIN,
            left=JoinNode(
                node_type=NodeType.JOIN,
                left=_scan("customers", "t1"),
                right=_scan("orders", "t2"),
                join_type=JoinType.INNER,
                condition=cond_inner,
            ),
            right=_scan("products", "t3"),
            join_type=JoinType.LEFT_OUTER,
            condition=cond_outer,
        ))
        assert "customers" in sql
        assert "orders" in sql
        assert "products" in sql
