"""
Äquivalenzprüfung von IQT-Bäumen mittels Subgraph Algorithmus
=============================================================
Implementiert Definition 2.3 des UQTL-Standards: Zwei IQT-Bäume Q1 und Q2 sind
semantisch äquivalent (Q1 ≡ Q2), wenn für alle Datenbankinstanzen σ gilt:
``[[Q1]]σ = [[Q2]]σ``.

Architektur der Prüfung (zweistufig)
-------------------------------------
**Stufe 1 – Strukturprüfung** (Subgraph Algorithmus, O(n³)):
  Jeder IQT-Baum wird als quadratische Adjazenzmatrix kodiert.
  Kanten entsprechen Eltern-Kind-Beziehungen.
  Der Subgraph Algorithmus berechnet eine strukturelle Inklusion:

  * ``equal`` / ``equal_keep_A`` / ``equal_keep_B`` – identische Struktur
  * ``keep_A`` – Q1 enthält Q2 strukturell
  * ``keep_B`` – Q2 enthält Q1 strukturell
  * ``keep_both`` – keine strukturelle Inklusion in eine Richtung

**Stufe 2 – Semantikprüfung** (normalisierter SQL-Vergleich):
  Beide Bäume werden zunächst mit :class:`~uqtl.normalizer.IQTNormalizer`
  normalisiert und anschließend mit :class:`~uqtl.sql_generator.SQLGenerator`
  in ANSI-SQL übersetzt.  Stimmen die SQL-Strings überein, gilt semantische
  Äquivalenz (Konformitätsstufen CL-1 bis CL-4, Abschnitt 6).

Ergebnis
--------
Das Ergebnis wird als :class:`EquivalenceReport` zurückgegeben:

* :attr:`EquivalenceResult.EQUIVALENT` – SQL identisch
* :attr:`EquivalenceResult.STRUCTURALLY_CONTAINED` – strukturelle Inklusion, aber SQL abweichend
* :attr:`EquivalenceResult.NOT_EQUIVALENT` – weder Inklusion noch SQL-Match

Beispiel
--------
::

    from uqtl.equivalence import IQTEquivalenceChecker
    from uqtl.iqt import *

    q1 = ScanNode(node_type=NodeType.SCAN, table="orders", alias="o")
    q2 = ScanNode(node_type=NodeType.SCAN, table="orders", alias="ord")

    checker = IQTEquivalenceChecker()
    report  = checker.check(q1, q2)
    print(report.result)   # EquivalenceResult.EQUIVALENT
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from subgraph import Subgraph

from uqtl.iqt import (
    IQTNode, NodeType,
    ScanNode, FilterNode, ProjectNode, JoinNode,
    GroupAggNode, SortNode, LimitNode, WindowNode,
    SubqueryNode, SetOpNode, WithNode,
)
from uqtl.normalizer import IQTNormalizer
from uqtl.sql_generator import SQLGenerator


class EquivalenceResult(Enum):
    """Gesamtergebnis der zweistufigen Äquivalenzprüfung."""
    EQUIVALENT = auto()
    """Beide IQT-Bäume erzeugen strukturell und semantisch dasselbe SQL."""

    STRUCTURALLY_CONTAINED = auto()
    """Q1 ist strukturell in Q2 enthalten (Q2 ist ausdrucksstärker),
    aber die normalisierten SQL-Ausgaben weichen voneinander ab."""

    NOT_EQUIVALENT = auto()
    """Keine strukturelle Inklusion und kein SQL-Match – Bäume sind nicht äquivalent."""


@dataclass
class EquivalenceReport:
    """Vollständiger Bericht einer Äquivalenzprüfung.

    Attributes:
        result:             Gesamtentscheidung (:class:`EquivalenceResult`).
        sql_q1:             Normalisiertes SQL des ersten IQT-Baums.
        sql_q2:             Normalisiertes SQL des zweiten IQT-Baums.
        sql_match:          ``True``, wenn ``sql_q1 == sql_q2`` (nach strip).
        subgraph_decision:  Rückgabe des Subgraph Algorithmus
                            (z. B. ``"equal"``, ``"keep_A"``, ``"keep_both"``).
        detail:             Menschenlesbare Begründung der Entscheidung.
    """
    result: EquivalenceResult
    sql_q1: str
    sql_q2: str
    sql_match: bool
    subgraph_decision: str
    detail: str = ""


class IQTEquivalenceChecker:
    """Prüft die semantische Äquivalenz zweier IQT-Bäume gemäß UQTL-Standard.

    Das Vorgehen ist zweistufig:

    1. **Strukturprüfung** via Subgraph Algorithmus auf den Adjazenzmatrizen
       der (normalisierten) IQT-Bäume (O(n³)).
    2. **Semantikprüfung** via normalisiertem SQL-Vergleich.

    Beide Stufen müssen EQUIVALENT ergeben, damit das Gesamtergebnis
    :attr:`EquivalenceResult.EQUIVALENT` ist.

    Args:
        use_adjacency_list: Wird direkt an den :class:`subgraph.Subgraph`-Algorithmus
                            weitergereicht.  ``False`` (Standard) verwendet Adjazenzmatrizen.

    Example::

        checker = IQTEquivalenceChecker()
        if checker.are_equivalent(q1, q2):
            print("Q1 ≡ Q2")
        else:
            report = checker.check(q1, q2)
            print(report.detail)
    """

    def __init__(self, use_adjacency_list: bool = False):
        self._subgraph = Subgraph(use_adjacency_list=use_adjacency_list)
        self._normalizer = IQTNormalizer()
        self._generator = SQLGenerator()

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def check(self, q1: IQTNode, q2: IQTNode) -> EquivalenceReport:
        """
        Prüft ob Q1 ≡ Q2 (semantische Äquivalenz).

        Args:
            q1: Erster IQT-Baum.
            q2: Zweiter IQT-Baum.

        Returns:
            EquivalenceReport mit Ergebnis, SQL-Ausgaben und Details.
        """
        # Schritt 1: Normalisierung beider Bäume
        n1 = self._normalizer.normalize(q1)
        n2 = self._normalizer.normalize(q2)

        # Schritt 2: SQL-Generierung
        sql1 = self._generator.generate(n1)
        sql2 = self._generator.generate(n2)
        sql_match = (sql1.strip() == sql2.strip())

        # Schritt 3: Strukturvergleich via Subgraph
        mat_a = self._build_adjacency_matrix(n1)
        mat_b = self._build_adjacency_matrix(n2)
        decision, _ = self._subgraph.compare_graphs(mat_a, mat_b)

        # Schritt 4: Gesamtentscheidung
        result, detail = self._decide(decision, sql_match, sql1, sql2)

        return EquivalenceReport(
            result=result,
            sql_q1=sql1,
            sql_q2=sql2,
            sql_match=sql_match,
            subgraph_decision=decision,
            detail=detail,
        )

    def are_equivalent(self, q1: IQTNode, q2: IQTNode) -> bool:
        """Kurzform der Äquivalenzprüfung.

        Args:
            q1: Erster IQT-Baum.
            q2: Zweiter IQT-Baum.

        Returns:
            ``True`` genau dann, wenn :meth:`check` :attr:`EquivalenceResult.EQUIVALENT` liefert.
        """
        return self.check(q1, q2).result == EquivalenceResult.EQUIVALENT

    # ── Adjazenzmatrix-Aufbau ─────────────────────────────────────────────────

    def _collect_nodes(self, root: IQTNode) -> List[IQTNode]:
        """Sammelt alle Knoten des IQT-Baums in Pre-Order-Reihenfolge.

        Die Reihenfolge entspricht der Alias-Vergabe durch den Normalisierer
        (Regel 3.2): Elternknoten werden vor Kindknoten verarbeitet.

        Args:
            root: Wurzelknoten des IQT-Baums.

        Returns:
            Geordnete Liste aller Knoten.
        """
        result: List[IQTNode] = []
        stack = [root]
        while stack:
            node = stack.pop(0)
            result.append(node)
            for child in self._children_of(node):
                stack.append(child)
        return result

    def _children_of(self, node: IQTNode) -> List[IQTNode]:
        """Gibt die direkten Kind-Knoten eines IQT-Knotens zurück.

        Die Reihenfolge entspricht der kanonischen Traversal-Reihenfolge:
        bei binären Knoten (JoinNode, SetOpNode) zuerst links, dann rechts.
        :class:`~uqtl.iqt.ScanNode` ist ein Blattknoten und liefert eine leere Liste.

        Args:
            node: Zu inspizierender Knoten.

        Returns:
            Liste der direkten Kinder (ggf. leer).
        """
        if isinstance(node, FilterNode):
            return [node.source]
        if isinstance(node, ProjectNode):
            return [node.source]
        if isinstance(node, SortNode):
            return [node.source]
        if isinstance(node, LimitNode):
            return [node.source]
        if isinstance(node, GroupAggNode):
            return [node.source]
        if isinstance(node, WindowNode):
            return [node.source]
        if isinstance(node, JoinNode):
            return [node.left, node.right]
        if isinstance(node, SubqueryNode):
            return [node.query]
        if isinstance(node, SetOpNode):
            return [node.left, node.right]
        if isinstance(node, WithNode):
            children: List[IQTNode] = [cte.query for cte in node.ctes]
            if node.query:
                children.append(node.query)
            return children
        # ScanNode: Blatt
        return []

    def _build_adjacency_matrix(self, root: IQTNode) -> np.ndarray:
        """
        Kodiert den IQT-Baum als quadratische Adjazenzmatrix.

        Knoten werden als Zeilen/Spalten kodiert (Pre-Order-Index).
        Eine 1 an Position [i, j] bedeutet: Knoten i ist Elternteil von j.

        Der Knotentyp wird als Diagonaleintrag gespeichert (1 = belegt),
        um die Knotenanzahl für den Subgraph Algorithmus zu erhalten.
        """
        nodes = self._collect_nodes(root)
        n = len(nodes)
        idx = {id(node): i for i, node in enumerate(nodes)}

        matrix = np.zeros((n, n), dtype=int)

        for node in nodes:
            i = idx[id(node)]
            # Diagonale markiert vorhandenen Knoten
            matrix[i, i] = 1
            for child in self._children_of(node):
                j = idx.get(id(child))
                if j is not None:
                    matrix[i, j] = 1

        return matrix

    # ── Entscheidungslogik ────────────────────────────────────────────────────

    @staticmethod
    def _decide(
        subgraph_decision: str,
        sql_match: bool,
        sql1: str,
        sql2: str,
    ) -> Tuple[EquivalenceResult, str]:
        """
        Kombiniert Subgraph-Ergebnis und SQL-Vergleich zur Endentscheidung.

        Äquivalenz liegt vor, wenn:
          - Subgraph meldet equal_keep_A, equal_keep_B, keep_A (Q1⊇Q2)
            oder keep_B (Q2⊇Q1), d.h. eine strukturelle Inklusion, UND
          - die normalisierten SQL-Ausgaben identisch sind.

        Nur keep_both (keine Inklusion in beide Richtungen) ohne SQL-Match
        gilt als nicht äquivalent.
        """
        structurally_related = subgraph_decision in (
            "equal", "equal_keep_A", "equal_keep_B", "keep_A", "keep_B"
        )

        if sql_match and structurally_related:
            return (
                EquivalenceResult.EQUIVALENT,
                f"Subgraph: {subgraph_decision}, SQL identisch.",
            )

        if sql_match and not structurally_related:
            # SQL gleich aber Subgraph sieht keine Inklusion →
            # semantisch äquivalent, strukturell verschieden (z.B. Umstrukturierung)
            return (
                EquivalenceResult.EQUIVALENT,
                f"SQL identisch (Subgraph: {subgraph_decision} – strukturelle "
                "Umformung ohne Informationsverlust).",
            )

        if structurally_related and not sql_match:
            return (
                EquivalenceResult.STRUCTURALLY_CONTAINED,
                f"Subgraph: {subgraph_decision}, aber SQL unterschiedlich. "
                "Strukturelle Inklusion ohne vollständige semantische Äquivalenz.",
            )

        return (
            EquivalenceResult.NOT_EQUIVALENT,
            f"Subgraph: {subgraph_decision}, SQL unterschiedlich. "
            f"Q1: {sql1!r} | Q2: {sql2!r}",
        )
