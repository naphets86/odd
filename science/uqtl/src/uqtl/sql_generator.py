"""
SQL-Generator für den IQT (Abschnitt 4 des UQTL-Standards)
===========================================================
Erzeugt ANSI SQL:2016-konformes SQL aus einem normalisierten IQT-Baum.
Alle Generierungsregeln entsprechen den formalen Definitionen in Abschnitt 4.

Zur Verwendung wird der Baum zunächst mit :class:`~uqtl.normalizer.IQTNormalizer`
normalisiert und anschließend mit :class:`SQLGenerator` in SQL übersetzt.

Eingang ist stets der Wurzelknoten; die Traversierung erfolgt iterativ
Schicht für Schicht (Pipelining-Modell):

.. code-block:: text

    LimitNode
      └─ SortNode
           └─ WindowNode
                └─ ProjectNode
                     └─ FilterNode (WHERE)
                          └─ GroupAggNode
                               └─ FilterNode ... (HAVING)
                                    └─ JoinNode | ScanNode

Beispiel
--------
::

    from uqtl.normalizer import IQTNormalizer
    from uqtl.sql_generator import SQLGenerator

    norm = IQTNormalizer()
    gen  = SQLGenerator()

    sql = gen.generate(norm.normalize(tree))
    print(sql)
"""


from __future__ import annotations

from typing import List, Optional

from uqtl.iqt import (
    IQTNode,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode,
    Expr, ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, LogicOp, CompareOp,
    InExpr, BetweenExpr, LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, AggFunc,
    WindowDef, WindowFrame, WindowFunc,
    JoinType, SortDirection, NullsOrder, FilterPosition,
    SetOpType, OnCondition, UsingCondition,
)


class SQLGenerator:
    """Traversiert einen IQT-Baum und erzeugt ANSI SQL:2016 als String.

    Die Klasse ist *zustandslos* – alle relevanten Informationen werden
    beim Aufruf von :meth:`generate` aus dem IQT-Baum extrahiert.

    Unterstützte Klauseln
    ---------------------
    * SELECT (mit DISTINCT und Projektionsliste)
    * FROM (Basistabelle, JOINs, Unterabfragen)
    * WHERE / HAVING (ein- und mehrstufige Prädikate)
    * GROUP BY
    * WINDOW-Fensterfunktionen
    * ORDER BY (mit NULLS FIRST / NULLS LAST)
    * LIMIT / OFFSET
    * WITH … AS (…) (CTEs)
    * UNION / UNION ALL / INTERSECT / EXCEPT

    Einstiegspunkt
    --------------
    :meth:`generate`
    """

    def generate(self, root: IQTNode) -> str:
        """Generiert SQL für einen vollständigen IQT-Baum.

        Args:
            root: Wurzelknoten des (normalisierten) IQT-Baums.

        Returns:
            ANSI-SQL :2016-String ohne führende oder abschließende Leerzeichen.
        """
        return self._gen_node(root).strip()

    # ── Knoten-Dispatch ───────────────────────────────────────────────────────

    def _gen_node(self, node: IQTNode) -> str:
        """Dispatcht auf den passenden Generator je nach Knotentyp.

        :class:`~uqtl.iqt.WithNode`, :class:`~uqtl.iqt.SetOpNode` und
        :class:`~uqtl.iqt.SubqueryNode` haben eigene Generator-Methoden;
        alle anderen Knoten werden über :meth:`_gen_select_stmt` behandelt.
        """
        if isinstance(node, WithNode):
            return self._gen_with(node)
        if isinstance(node, SetOpNode):
            return self._gen_set_op(node)
        if isinstance(node, SubqueryNode):
            return self._gen_subquery(node)
        # Aufbau: sammle alle Clauseln aus dem Baum
        return self._gen_select_stmt(node)

    # ── SELECT-Statement Aufbau ───────────────────────────────────────────────

    def _gen_select_stmt(self, node: IQTNode) -> str:
        """Baut aus einem geschichteten IQT-Teilbaum ein vollständiges SELECT-Statement.

        Die Methode traversiert den Baum iterativ von oben nach unten und
        sammelt die SQL-Klauseln Schicht für Schicht:

        .. code-block:: text

            LimitNode → SortNode → WindowNode → ProjectNode →
            FilterNode (WHERE) → GroupAggNode → FilterNode (HAVING) →
            JoinNode | ScanNode | SubqueryNode

        Args:
            node: Oberster Knoten des zu generierenden Teilbaums.

        Returns:
            Mehrzeiliger SQL-String mit allen relevanten Klauseln.
        """
        select_clause = "SELECT *"
        distinct = False
        from_clause = ""
        where_clauses: List[str] = []
        having_clause: Optional[str] = None
        group_by_clause: Optional[str] = None
        order_by_clause: Optional[str] = None
        limit_clause: Optional[str] = None
        offset_clause: Optional[str] = None
        window_defs: List[WindowDef] = []

        current = node

        # Schicht für Schicht abbauen
        while current is not None:
            if isinstance(current, LimitNode):
                if current.count is not None:
                    limit_clause = str(current.count)
                if current.offset is not None:
                    offset_clause = str(current.offset)
                current = current.source

            elif isinstance(current, SortNode):
                order_by_clause = ", ".join(
                    self._gen_sort_item(s) for s in current.items
                )
                current = current.source

            elif isinstance(current, WindowNode):
                window_defs = current.definitions
                current = current.source

            elif isinstance(current, ProjectNode):
                distinct = current.distinct
                if current.columns:
                    cols = ", ".join(
                        self._gen_projection_item(c) for c in current.columns
                    )
                    select_clause = f"SELECT {'DISTINCT ' if distinct else ''}{cols}"
                else:
                    select_clause = f"SELECT {'DISTINCT ' if distinct else ''}*"
                current = current.source

            elif isinstance(current, FilterNode):
                pred_sql = self._gen_expr(current.predicate)
                if current.position == FilterPosition.HAVING:
                    having_clause = pred_sql
                else:
                    where_clauses.append(pred_sql)
                current = current.source

            elif isinstance(current, GroupAggNode):
                if current.group_by:
                    group_by_clause = ", ".join(
                        self._gen_expr(e) for e in current.group_by
                    )
                # Aggregat-Spalten in SELECT einbauen
                agg_cols = ", ".join(
                    self._gen_agg_item(a) for a in current.aggregates
                )
                if agg_cols:
                    # Falls SELECT noch * ist, durch Aggregate ersetzen
                    if select_clause in ("SELECT *", "SELECT DISTINCT *"):
                        prefix = "SELECT DISTINCT " if distinct else "SELECT "
                        select_clause = prefix + agg_cols
                    else:
                        # Aggregate anhängen (GROUP BY mit vorhandener Projektion)
                        select_clause = select_clause + ", " + agg_cols
                if current.having:
                    having_clause = self._gen_expr(current.having)
                current = current.source

            elif isinstance(current, JoinNode):
                from_clause = self._gen_join(current)
                current = None

            elif isinstance(current, ScanNode):
                from_clause = self._gen_scan(current)
                current = None

            elif isinstance(current, SubqueryNode):
                from_clause = self._gen_subquery(current)
                if current.alias:
                    from_clause = f"({self._gen_node(current.query)}) AS {current.alias}"
                current = None

            else:
                current = None

        # Window-Definitionen in SELECT einfügen
        if window_defs:
            win_cols = ", ".join(self._gen_window_def(w) for w in window_defs)
            if select_clause in ("SELECT *", "SELECT DISTINCT *"):
                select_clause = "SELECT " + win_cols
            else:
                select_clause = select_clause + ", " + win_cols

        # Zusammenbauen
        parts = [select_clause]
        if from_clause:
            parts.append(f"FROM {from_clause}")
        if where_clauses:
            combined = "\n  AND ".join(where_clauses[::-1])
            parts.append(f"WHERE {combined}")
        if group_by_clause:
            parts.append(f"GROUP BY {group_by_clause}")
        if having_clause:
            parts.append(f"HAVING {having_clause}")
        if order_by_clause:
            parts.append(f"ORDER BY {order_by_clause}")
        if limit_clause:
            parts.append(f"LIMIT {limit_clause}")
        if offset_clause:
            parts.append(f"OFFSET {offset_clause}")

        return "\n".join(parts)

    # ── Einzelne Knoten ───────────────────────────────────────────────────────

    def _gen_scan(self, node: ScanNode) -> str:
        """Gibt den FROM-Ausdruck für einen :class:`~uqtl.iqt.ScanNode` zurück.

        Returns:
            ``"table AS alias"`` bzw. nur ``"table"`` ohne Alias.
        """
        if node.alias:
            return f"{node.table} AS {node.alias}"
        return node.table

    def _gen_join(self, node: JoinNode) -> str:
        """Erzeugt den FROM-Ausdruck für einen JOIN-Knoten.

        Unterstützt ``ON``-Bedingungen, ``USING``-Listen und Joins ohne Bedingung
        (CROSS JOIN).  Kinder werden rekursiv über :meth:`_gen_from_source` aufgelöst.
        """
        left = self._gen_from_source(node.left)
        right = self._gen_from_source(node.right)

        join_kw = {
            JoinType.INNER: "INNER JOIN",
            JoinType.LEFT_OUTER: "LEFT OUTER JOIN",
            JoinType.RIGHT_OUTER: "RIGHT OUTER JOIN",
            JoinType.FULL_OUTER: "FULL OUTER JOIN",
            JoinType.CROSS: "CROSS JOIN",
        }[node.join_type]

        if isinstance(node.condition, OnCondition):
            cond_sql = self._gen_expr(node.condition.predicate)
            return f"{left}\n{join_kw} {right}\n  ON {cond_sql}"
        if isinstance(node.condition, UsingCondition):
            cols = ", ".join(node.condition.columns)
            return f"{left}\n{join_kw} {right}\n  USING ({cols})"
        return f"{left}\n{join_kw} {right}"

    def _gen_from_source(self, node: IQTNode) -> str:
        """Gibt einen FROM-Ausdruck für einen beliebigen Teilbaum zurück.

        Wird innerhalb von :meth:`_gen_join` verwendet, um linke und rechte
        Kindknoten aufzulösen.  Für unbekannte Knotentypen wird ein
        Fallback-Unterabfragen-Ausdruck erzeugt.
        """
        if isinstance(node, ScanNode):
            return self._gen_scan(node)
        if isinstance(node, JoinNode):
            return self._gen_join(node)
        if isinstance(node, SubqueryNode):
            inner = self._gen_node(node.query)
            alias = f" AS {node.alias}" if node.alias else ""
            return f"({inner}){alias}"
        # Fallback: Unterabfrage
        inner = self._gen_select_stmt(node)
        return f"({inner})"

    def _gen_subquery(self, node: SubqueryNode) -> str:
        """Gibt eine geklammerte Unterabfrage ``(SELECT …) [AS alias]`` zurück."""
        inner = self._gen_node(node.query)
        alias = f" AS {node.alias}" if node.alias else ""
        return f"({inner}){alias}"

    def _gen_with(self, node: WithNode) -> str:
        """Generiert einen WITH-Ausdruck mit einer oder mehreren CTE-Definitionen.

        Jede CTE wird als ``name AS (\n    SELECT …\n)`` formatiert.
        Die optionale Hauptabfrage wird nach dem WITH-Block angehängt.
        """
        cte_parts = []
        for cte in node.ctes:
            inner = self._gen_node(cte.query)
            cte_parts.append(f"{cte.name} AS (\n    {inner}\n)")
        with_clause = "WITH " + ",\n".join(cte_parts)
        main = self._gen_node(node.query) if node.query else ""
        return f"{with_clause}\n{main}"

    def _gen_set_op(self, node: SetOpNode) -> str:
        """Generiert eine Mengenoperation ``left_sql\nOP\nright_sql``."""
        left = self._gen_node(node.left)
        right = self._gen_node(node.right)
        op = node.op.value
        return f"{left}\n{op}\n{right}"

    # ── Projektions-Items ─────────────────────────────────────────────────────

    def _gen_projection_item(self, item: ProjectionItem) -> str:
        """Erzeugt den SQL-String für einen einzelnen SELECT-Spaltenausdruck.

        Returns:
            ``"*"``, ``"expr"`` oder ``"expr AS alias"``.
        """
        if item.star:
            return "*"
        sql = self._gen_expr(item.expr)
        if item.alias:
            return f"{sql} AS {item.alias}"
        return sql

    # ── Aggregationsausdrücke ─────────────────────────────────────────────────

    def _gen_agg_item(self, item: AggItem) -> str:
        """Erzeugt den SQL-String für eine Aggregatfunktion.

        Sonderfälle:
          * :attr:`~uqtl.iqt.AggFunc.COUNT_DISTINCT` → ``COUNT(DISTINCT arg)``
          * :attr:`~uqtl.iqt.AggFunc.COUNT` ohne Argument → ``COUNT(*)``

        Returns:
            Aggregatausdruck ggf. mit ``AS alias``.
        """
        if item.func == AggFunc.COUNT_DISTINCT:
            arg = self._gen_expr(item.arg) if item.arg else "*"
            sql = f"COUNT(DISTINCT {arg})"
        elif item.func == AggFunc.COUNT and item.arg is None:
            sql = "COUNT(*)"
        else:
            arg = self._gen_expr(item.arg) if item.arg else "*"
            distinct_kw = "DISTINCT " if item.distinct else ""
            sql = f"{item.func.value}({distinct_kw}{arg})"
        if item.alias:
            return f"{sql} AS {item.alias}"
        return sql

    # ── Sortier-Items ─────────────────────────────────────────────────────────

    def _gen_sort_item(self, item: SortItem) -> str:
        """Erzeugt einen ORDER-BY-Teilausdruck ``expr ASC|DESC [NULLS FIRST|LAST]``."""
        expr = self._gen_expr(item.expr)
        direction = item.direction.value
        nulls = f" {item.nulls.value}" if item.nulls != NullsOrder.DEFAULT else ""
        return f"{expr} {direction}{nulls}"

    # ── Window-Definitionen ───────────────────────────────────────────────────

    def _gen_window_def(self, wdef: WindowDef) -> str:
        """Erzeugt den vollständigen Fensterfunktionsausdruck.

        Format: ``FUNC([args]) OVER ([PARTITION BY ...] [ORDER BY ...] [frame]) AS alias``

        Rangfunktionen (ROW_NUMBER, RANK, DENSE_RANK) werden immer mit
        leerer Argumentliste ``()`` generiert.
        """
        func_name = wdef.func.value.replace("_", " ")
        if wdef.func_args:
            args = ", ".join(self._gen_expr(a) for a in wdef.func_args)
            func_call = f"{func_name}({args})"
        elif wdef.func in (WindowFunc.ROW_NUMBER, WindowFunc.RANK, WindowFunc.DENSE_RANK):
            func_call = f"{func_name}()"
        else:
            func_call = f"{func_name}()"

        over_parts = []
        if wdef.partition_by:
            pb = ", ".join(self._gen_expr(e) for e in wdef.partition_by)
            over_parts.append(f"PARTITION BY {pb}")
        if wdef.order_by:
            ob = ", ".join(self._gen_sort_item(s) for s in wdef.order_by)
            over_parts.append(f"ORDER BY {ob}")
        if wdef.frame:
            over_parts.append(self._gen_window_frame(wdef.frame))

        over_clause = " ".join(over_parts)
        result = f"{func_call} OVER ({over_clause})"
        if wdef.result_alias:
            result = f"{result} AS {wdef.result_alias}"
        return result

    def _gen_window_frame(self, frame: WindowFrame) -> str:
        """Erzeugt die Fensterrahmen-Klausel ``ROWS|RANGE BETWEEN start AND end``."""
        return f"{frame.unit.value} BETWEEN {frame.start} AND {frame.end}"

    # ── Ausdrücke ─────────────────────────────────────────────────────────────

    def _gen_expr(self, expr: Expr) -> str:
        """Wandelt einen Ausdrucksknoten rekursiv in SQL um.

        Unterstützte Ausdruckstypen:

        * :class:`~uqtl.iqt.ColumnRef` → ``table.column`` oder ``column``
        * :class:`~uqtl.iqt.Literal` → ``NULL``, ``TRUE``/``FALSE``, ``'str'``, Zahl
        * :class:`~uqtl.iqt.FuncCall` → ``FUNC([DISTINCT] args)``
        * :class:`~uqtl.iqt.CompareExpr` → ``left op right``
        * :class:`~uqtl.iqt.LogicExpr` → ``a AND b``, ``a OR b``, ``NOT (a)``
        * :class:`~uqtl.iqt.InExpr` → ``expr [NOT] IN (items|subquery)``
        * :class:`~uqtl.iqt.BetweenExpr` → ``expr [NOT] BETWEEN lo AND hi``
        * :class:`~uqtl.iqt.LikeExpr` → ``expr [NOT] LIKE 'pattern' [ESCAPE 'c']``
        * :class:`~uqtl.iqt.NullExpr` → ``expr IS [NOT] NULL``
        * :class:`~uqtl.iqt.ExistsExpr` → ``[NOT] EXISTS (subquery)``

        Args:
            expr: Zu übersetzender Ausdrucksknoten.

        Returns:
            SQL-Teilstring des Ausdrucks.
        """
        if isinstance(expr, ColumnRef):
            if expr.table:
                return f"{expr.table}.{expr.column}"
            return expr.column

        if isinstance(expr, Literal):
            if expr.value is None:
                return "NULL"
            if isinstance(expr.value, bool):
                return "TRUE" if expr.value else "FALSE"
            if isinstance(expr.value, str):
                escaped = expr.value.replace("'", "''")
                return f"'{escaped}'"
            return str(expr.value)

        if isinstance(expr, FuncCall):
            distinct_kw = "DISTINCT " if expr.distinct else ""
            args = ", ".join(self._gen_expr(a) for a in expr.args)
            return f"{expr.name.upper()}({distinct_kw}{args})"

        if isinstance(expr, CompareExpr):
            left = self._gen_expr(expr.left)
            right = self._gen_expr(expr.right)
            return f"{left} {expr.op.value} {right}"

        if isinstance(expr, LogicExpr):
            if expr.op == LogicOp.NOT:
                inner = self._gen_expr(expr.operands[0])
                return f"NOT ({inner})"
            sep = f" {expr.op.value} "
            parts = [self._gen_expr(o) for o in expr.operands]
            return sep.join(parts)

        if isinstance(expr, InExpr):
            val = self._gen_expr(expr.value)
            not_kw = "NOT " if expr.negated else ""
            if expr.subquery:
                inner = self._gen_node(expr.subquery.query)
                return f"{val} {not_kw}IN ({inner})"
            items = ", ".join(self._gen_expr(i) for i in expr.items)
            return f"{val} {not_kw}IN ({items})"

        if isinstance(expr, BetweenExpr):
            val = self._gen_expr(expr.value)
            lo = self._gen_expr(expr.lo)
            hi = self._gen_expr(expr.hi)
            not_kw = "NOT " if expr.negated else ""
            return f"{val} {not_kw}BETWEEN {lo} AND {hi}"

        if isinstance(expr, LikeExpr):
            val = self._gen_expr(expr.value)
            not_kw = "NOT " if expr.negated else ""
            escape = f" ESCAPE '{expr.escape}'" if expr.escape else ""
            return f"{val} {not_kw}LIKE '{expr.pattern}'{escape}"

        if isinstance(expr, NullExpr):
            val = self._gen_expr(expr.value)
            return f"{val} IS NOT NULL" if expr.negated else f"{val} IS NULL"

        if isinstance(expr, ExistsExpr):
            inner = self._gen_node(expr.subquery.query)
            not_kw = "NOT " if expr.negated else ""
            return f"{not_kw}EXISTS (\n    {inner}\n)"

        return str(expr)
