"""
IQT-Normalisierer gemäß UQTL-Standard (Abschnitt 3.3)
=======================================================
Führt alle Normalisierungsregeln auf einem IQT-Baum aus und gibt
einen neuen, semantisch äquivalenten Baum in kanonischer Form zurück.
Der Eingabebaum bleibt stets unverändert.

Normalisierungsregeln
---------------------
**Regel 3.1 – Prädikat-Normalisierung**
  AND-Operanden werden lexikografisch nach ``repr()`` sortiert.
  Dadurch gilt ``p1 AND p2 ≡ p2 AND p1`` nach der Normalisierung.

**Regel 3.2 – Alias-Normalisierung**
  Alle Tabellenaliase werden in Pre-Order-Traversalreihenfolge
  durch ``t1``, ``t2``, ... ersetzt.  Spaltenbezüge werden über
  eine interne Abbildungstabelle aktualisiert.

**Regel 3.3 – Boolean-Vereinfachung**
  Redundante boolesche Teilausdrücke werden eliminiert:
  ``p AND TRUE → p``, ``p OR FALSE → p``,
  ``NOT NOT p → p``, ``p AND p → p``.

**Regel 3.4 – SELECT-Reihenfolge**
  Projektionsspalten bleiben in ihrer Deklarationsreihenfolge
  (Kanonisierung durch den Aufrufer, nicht durch den Normalisierer).

**Regel 3.5 – JOIN-Kanonisierung**
  LEFT-assoziative Normierung: Der linke Teilbaum wird stets zuerst
  normalisiert, damit seine Alias-Nummern stabil bleiben.

Beispiel
--------
::

    from uqtl.normalizer import IQTNormalizer
    from uqtl.iqt import *

    n = IQTNormalizer()
    tree = FilterNode(
        node_type=NodeType.FILTER,
        source=ScanNode(node_type=NodeType.SCAN, table="orders", alias="o"),
        predicate=LogicExpr(
            op=LogicOp.AND,
            operands=[pred2, pred1],   # beliebige Reihenfolge
        ),
    )
    normalized = n.normalize(tree)
    # normalized.source.alias == "t1"
    # normalized.predicate.operands == sorted([pred1, pred2], key=repr)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from uqtl.iqt import (
    IQTNode, NodeType,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode, CTEDef,
    Expr, ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, LogicOp, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
    ProjectionItem, SortItem, AggItem, WindowDef,
    JoinCondition, OnCondition, UsingCondition,
)


class IQTNormalizer:
    """Normalisiert einen IQT-Baum gemäß den UQTL-Normalisierungsregeln 3.1–3.5.

    Die Klasse ist *zustandslos* – sie kann für beliebig viele Aufrufe
    wiederverwendet werden.  Jeder Aufruf von :meth:`normalize` arbeitet mit
    einem internen, aufrufspezifischen Alias-Kontext.

    Methoden
    --------
    normalize(root)
        Einstiegspunkt: gibt den normalisierten Baum zurück.
    """

    def normalize(self, root: IQTNode) -> IQTNode:
        """Normalisiert den übergebenen IQT-Baum.

        Startet einen neuen Alias-Kontext und wendet alle Normalisierungs-
        regeln (3.1–3.5) rekursiv auf den gesamten Baum an.

        Args:
            root: Wurzelknoten des zu normalisierenden IQT-Baums.

        Returns:
            Ein neuer, semantisch äquivalenter Baum in kanonischer Form.
        """
        alias_map: Dict[str, str] = {}
        counter = [1]
        return self._normalize_node(root, alias_map, counter)

    # ── Knoten-Dispatch ───────────────────────────────────────────────────────────────────────

    def _normalize_node(
        self,
        node: IQTNode,
        alias_map: Dict[str, str],
        counter: List[int],
    ) -> IQTNode:
        """Delegiert an den typ-spezifischen Normalisierer.

        Unbekannte Knotentypen werden unverändert zurückgegeben.

        Args:
            node:      Zu normalisierender Knoten.
            alias_map: Gemeinsame Alias-Abbildungstabelle (wird gefüllt).
            counter:   Einelementige Liste mit dem nächsten Alias-Zähler.

        Returns:
            Der normalisierte Knoten.
        """
        if isinstance(node, ScanNode):
            return self._normalize_scan(node, alias_map, counter)
        if isinstance(node, FilterNode):
            return self._normalize_filter(node, alias_map, counter)
        if isinstance(node, ProjectNode):
            return self._normalize_project(node, alias_map, counter)
        if isinstance(node, JoinNode):
            return self._normalize_join(node, alias_map, counter)
        if isinstance(node, GroupAggNode):
            return self._normalize_group_agg(node, alias_map, counter)
        if isinstance(node, SortNode):
            return self._normalize_sort(node, alias_map, counter)
        if isinstance(node, LimitNode):
            return self._normalize_limit(node, alias_map, counter)
        if isinstance(node, WindowNode):
            return self._normalize_window(node, alias_map, counter)
        if isinstance(node, SubqueryNode):
            return self._normalize_subquery(node, alias_map, counter)
        if isinstance(node, SetOpNode):
            return self._normalize_set_op(node, alias_map, counter)
        if isinstance(node, WithNode):
            return self._normalize_with(node, alias_map, counter)
        return node

    # ── Einzelne Knoten ───────────────────────────────────────────────────────

    def _normalize_scan(
        self, node: ScanNode, alias_map: Dict[str, str], counter: List[int]
    ) -> ScanNode:
        """Regel 3.2: Weist dem ScanNode einen kanonischen Alias ``tN`` zu.

        Der bisherige Alias (falls vorhanden) und der Tabellenname werden
        beide auf den neuen kanonischen Alias abgebildet.
        """
        canonical_alias = f"t{counter[0]}"
        counter[0] += 1
        if node.alias:
            alias_map[node.alias] = canonical_alias
        alias_map[node.table] = canonical_alias
        return ScanNode(node_type=node.node_type, table=node.table, alias=canonical_alias)

    def _normalize_filter(
        self, node: FilterNode, alias_map: Dict[str, str], counter: List[int]
    ) -> FilterNode:
        """Normalisiert Quelle, Prädikat und wendet Regeln 3.1 + 3.3 an."""
        src = self._normalize_node(node.source, alias_map, counter)
        pred = self._normalize_expr(node.predicate, alias_map)
        pred = self._simplify_boolean(pred)
        if isinstance(pred, LogicExpr) and pred.op == LogicOp.AND:
            pred = self._sort_and_predicates(pred)
        return FilterNode(
            node_type=node.node_type,
            source=src,
            predicate=pred,
            position=node.position,
        )

    def _normalize_project(
        self, node: ProjectNode, alias_map: Dict[str, str], counter: List[int]
    ) -> ProjectNode:
        """Normalisiert Quelle und alle Projektionsspalten (Regel 3.4: Reihenfolge bleibt)."""
        src = self._normalize_node(node.source, alias_map, counter)
        cols = [
            ProjectionItem(
                expr=self._normalize_expr(c.expr, alias_map),
                alias=c.alias,
                star=c.star,
            )
            for c in node.columns
        ]
        return ProjectNode(
            node_type=node.node_type,
            source=src,
            columns=cols,
            distinct=node.distinct,
        )

    def _normalize_join(
        self, node: JoinNode, alias_map: Dict[str, str], counter: List[int]
    ) -> JoinNode:
        """Regel 3.5: Normalisiert linken Teilbaum vor dem rechten (left-assoziativ)."""
        # Regel 3.5: Left-assoziativ – linken Teilbaum zuerst normalisieren
        left = self._normalize_node(node.left, alias_map, counter)
        right = self._normalize_node(node.right, alias_map, counter)
        cond = self._normalize_join_condition(node.condition, alias_map)
        return JoinNode(
            node_type=node.node_type,
            left=left,
            right=right,
            join_type=node.join_type,
            condition=cond,
        )

    def _normalize_group_agg(
        self, node: GroupAggNode, alias_map: Dict[str, str], counter: List[int]
    ) -> GroupAggNode:
        """Normalisiert GROUP-BY-Ausdrücke, Aggregate und optionalen HAVING-Filter.

        Ein AND-verknüpftes HAVING-Prädikat wird zusätzlich nach Regel 3.1 sortiert.
        """
        src = self._normalize_node(node.source, alias_map, counter)
        group_by = [self._normalize_expr(e, alias_map) for e in node.group_by]
        aggs = [
            AggItem(
                func=a.func,
                arg=self._normalize_expr(a.arg, alias_map) if a.arg else None,
                distinct=a.distinct,
                alias=a.alias,
            )
            for a in node.aggregates
        ]
        having = self._normalize_expr(node.having, alias_map) if node.having else None
        if having and isinstance(having, LogicExpr) and having.op == LogicOp.AND:
            having = self._sort_and_predicates(having)
        return GroupAggNode(
            node_type=node.node_type,
            source=src,
            group_by=group_by,
            aggregates=aggs,
            having=having,
        )

    def _normalize_sort(
        self, node: SortNode, alias_map: Dict[str, str], counter: List[int]
    ) -> SortNode:
        """Normalisiert Quelle und alle ORDER-BY-Ausdrücke."""
        src = self._normalize_node(node.source, alias_map, counter)
        items = [
            SortItem(
                expr=self._normalize_expr(s.expr, alias_map),
                direction=s.direction,
                nulls=s.nulls,
            )
            for s in node.items
        ]
        return SortNode(node_type=node.node_type, source=src, items=items)

    def _normalize_limit(
        self, node: LimitNode, alias_map: Dict[str, str], counter: List[int]
    ) -> LimitNode:
        """Normalisiert die Quelle; LIMIT/OFFSET-Werte bleiben unverändert."""
        src = self._normalize_node(node.source, alias_map, counter)
        return LimitNode(
            node_type=node.node_type,
            source=src,
            count=node.count,
            offset=node.offset,
        )

    def _normalize_window(
        self, node: WindowNode, alias_map: Dict[str, str], counter: List[int]
    ) -> WindowNode:
        """Normalisiert Quelle und alle Fensterfunktionsdefinitionen."""
        src = self._normalize_node(node.source, alias_map, counter)
        defs = [self._normalize_window_def(d, alias_map) for d in node.definitions]
        return WindowNode(node_type=node.node_type, source=src, definitions=defs)

    def _normalize_subquery(
        self, node: SubqueryNode, alias_map: Dict[str, str], counter: List[int]
    ) -> SubqueryNode:
        """Normalisiert die eingebettete Unterabfrage in einem eigenen Alias-Kontext.

        Unterabfragen erhalten einen frischen Alias-Zählkontext, der nach
        ihrer Normalisierung an den äußeren Kontext zurückgegeben wird.
        """
        inner_alias_map: Dict[str, str] = {}
        inner_counter = [counter[0]]
        inner = self._normalize_node(node.query, inner_alias_map, inner_counter)
        counter[0] = inner_counter[0]
        return SubqueryNode(node_type=node.node_type, query=inner, alias=node.alias)

    def _normalize_set_op(
        self, node: SetOpNode, alias_map: Dict[str, str], counter: List[int]
    ) -> SetOpNode:
        """Normalisiert linken und rechten Eingabeteilbaum der Mengenoperation."""
        left = self._normalize_node(node.left, alias_map, counter)
        right = self._normalize_node(node.right, alias_map, counter)
        return SetOpNode(node_type=node.node_type, left=left, right=right, op=node.op)

    def _normalize_with(
        self, node: WithNode, alias_map: Dict[str, str], counter: List[int]
    ) -> WithNode:
        """Normalisiert alle CTE-Abfragen und die optionale Hauptabfrage."""
        ctes = [
            CTEDef(
                name=c.name,
                query=self._normalize_node(c.query, alias_map, counter),
            )
            for c in node.ctes
        ]
        query = self._normalize_node(node.query, alias_map, counter) if node.query else None
        return WithNode(node_type=node.node_type, ctes=ctes, query=query)

    # ── Ausdrucks-Normalisierung ──────────────────────────────────────────────

    def _normalize_expr(self, expr: Optional[Expr], alias_map: Dict[str, str]) -> Optional[Expr]:
        """Wendet die Alias-Abbildung rekursiv auf einen Ausdrucksknoten an.

        Args:
            expr:      Zu normalisierender Ausdrucksknoten oder ``None``.
            alias_map: Aktuelle Alias-Abbildungstabelle.

        Returns:
            Normalisierter Ausdruck oder ``None`` (falls Eingabe ``None`` war).
        """
        if expr is None:
            return None
        if isinstance(expr, ColumnRef):
            table = alias_map.get(expr.table) if expr.table else None
            return ColumnRef(column=expr.column, table=table)
        if isinstance(expr, Literal):
            return expr
        if isinstance(expr, FuncCall):
            return FuncCall(
                name=expr.name,
                args=[self._normalize_expr(a, alias_map) for a in expr.args],
                distinct=expr.distinct,
            )
        if isinstance(expr, CompareExpr):
            return CompareExpr(
                left=self._normalize_expr(expr.left, alias_map),
                op=expr.op,
                right=self._normalize_expr(expr.right, alias_map),
            )
        if isinstance(expr, LogicExpr):
            ops = [self._normalize_expr(o, alias_map) for o in expr.operands]
            return LogicExpr(op=expr.op, operands=ops)
        if isinstance(expr, InExpr):
            items = [self._normalize_expr(i, alias_map) for i in expr.items]
            return InExpr(
                value=self._normalize_expr(expr.value, alias_map),
                items=items,
                subquery=expr.subquery,
                negated=expr.negated,
            )
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(
                value=self._normalize_expr(expr.value, alias_map),
                lo=self._normalize_expr(expr.lo, alias_map),
                hi=self._normalize_expr(expr.hi, alias_map),
                negated=expr.negated,
            )
        if isinstance(expr, LikeExpr):
            return LikeExpr(
                value=self._normalize_expr(expr.value, alias_map),
                pattern=expr.pattern,
                escape=expr.escape,
                negated=expr.negated,
            )
        if isinstance(expr, NullExpr):
            return NullExpr(
                value=self._normalize_expr(expr.value, alias_map),
                negated=expr.negated,
            )
        if isinstance(expr, ExistsExpr):
            return expr  # Unterabfrage separat normalisiert
        return expr

    def _normalize_join_condition(
        self, cond: JoinCondition, alias_map: Dict[str, str]
    ) -> JoinCondition:
        """Normalisiert eine JOIN-Bedingung.

        :class:`OnCondition`: Das Prädikat wird über :meth:`_normalize_expr` normalisiert.
        :class:`UsingCondition`: Spaltennamen sind tabellenunabhängig und bleiben unverändert.
        """
        if isinstance(cond, OnCondition):
            return OnCondition(predicate=self._normalize_expr(cond.predicate, alias_map))
        return cond  # UsingCondition: Spaltennamen unverändert

    def _normalize_window_def(
        self, wdef: WindowDef, alias_map: Dict[str, str]
    ) -> WindowDef:
        """Normalisiert alle Ausdrücke innerhalb einer Fensterfunktionsdefinition.

        Betroffen sind PARTITION-BY-Ausdrücke, ORDER-BY-Ausdrücke und
        Funktionsargumente.  Rahmenspezifikation und Alias bleiben unverändert.
        """
        partition = [self._normalize_expr(e, alias_map) for e in wdef.partition_by]
        order = [
            SortItem(
                expr=self._normalize_expr(s.expr, alias_map),
                direction=s.direction,
                nulls=s.nulls,
            )
            for s in wdef.order_by
        ]
        args = [self._normalize_expr(a, alias_map) for a in wdef.func_args]
        return WindowDef(
            func=wdef.func,
            result_alias=wdef.result_alias,
            partition_by=partition,
            order_by=order,
            frame=wdef.frame,
            func_args=args,
        )

    # ── Boolean-Vereinfachung (Regel 3.3) ─────────────────────────────────────

    def _simplify_boolean(self, expr: Expr) -> Expr:
        """Eliminiert redundante boolesche Ausdrücke (Regel 3.3).

        Transformationen:
          - ``p AND TRUE``  → ``p``
          - ``p OR FALSE``  → ``p``
          - ``NOT NOT p``   → ``p``
          - ``p AND p``     → ``p``  (Deduplizierung)
          - ``p AND [TRUE, TRUE]`` → alle TRUE-Literale werden entfernt

        Args:
            expr: Zu vereinfachender Ausdruck.

        Returns:
            Vereinfachter Ausdruck (ggf. unverandert, falls keine Regel greift).
        """
        if not isinstance(expr, LogicExpr):
            return expr

        if expr.op == LogicOp.NOT:
            inner = expr.operands[0] if expr.operands else expr
            # NOT NOT p → p
            if isinstance(inner, LogicExpr) and inner.op == LogicOp.NOT:
                return self._simplify_boolean(inner.operands[0])
            return LogicExpr(op=LogicOp.NOT, operands=[self._simplify_boolean(inner)])

        simplified = [self._simplify_boolean(o) for o in expr.operands]

        if expr.op == LogicOp.AND:
            # Entferne TRUE-Literale
            filtered = [
                o for o in simplified
                if not (isinstance(o, Literal) and o.value is True)
            ]
            # Deduplizierung
            seen: List[str] = []
            deduped = []
            for o in filtered:
                key = repr(o)
                if key not in seen:
                    seen.append(key)
                    deduped.append(o)
            if len(deduped) == 1:
                return deduped[0]
            return LogicExpr(op=LogicOp.AND, operands=deduped)

        if expr.op == LogicOp.OR:
            # Entferne FALSE-Literale
            filtered = [
                o for o in simplified
                if not (isinstance(o, Literal) and o.value is False)
            ]
            if len(filtered) == 1:
                return filtered[0]
            return LogicExpr(op=LogicOp.OR, operands=filtered)

        return LogicExpr(op=expr.op, operands=simplified)

    # ── AND-Prädikat-Sortierung (Regel 3.1) ──────────────────────────────────

    @staticmethod
    def _sort_and_predicates(expr: LogicExpr) -> LogicExpr:
        """Sortiert AND-Operanden lexikografisch nach ``repr()`` (Regel 3.1).

        Damit gilt nach der Normalisierung:
        ``p1 AND p2 ≡ p2 AND p1``
        (beide erzeugen dieselbe normalisierte Darstellung).

        Args:
            expr: Ein ``LogicExpr`` mit ``op == LogicOp.AND``.

        Returns:
            Ein neues ``LogicExpr`` mit sortierten Operanden.
        """
        sorted_ops = sorted(expr.operands, key=repr)
        return LogicExpr(op=LogicOp.AND, operands=sorted_ops)
