"""
IQT – Intermediate Query Tree
==============================
Sprachunabhängige Baumstruktur zur Darstellung relationaler Abfragen
gemäß UQTL-Standard, Abschnitt 3.

Hierarchie der Typen
--------------------
*Enumerationen* – klassifizieren Operatoren und Optionen:
  :class:`NodeType`, :class:`JoinType`, :class:`SortDirection`,
  :class:`NullsOrder`, :class:`AggFunc`, :class:`SetOpType`,
  :class:`FrameUnit`, :class:`WindowFunc`, :class:`CompareOp`,
  :class:`LogicOp`, :class:`FilterPosition`

*Ausdrucks-Typen* – Blätter und Teilausdrücke der WHERE/HAVING-Prädikate:
  :class:`ColumnRef`, :class:`Literal`, :class:`FuncCall`,
  :class:`CompareExpr`, :class:`LogicExpr`, :class:`InExpr`,
  :class:`BetweenExpr`, :class:`LikeExpr`, :class:`NullExpr`,
  :class:`ExistsExpr`

*Hilfs-Datenklassen* – strukturieren JOIN, Aggregation, Fenster, Projektion:
  :class:`ProjectionItem`, :class:`SortItem`, :class:`AggItem`,
  :class:`WindowFrame`, :class:`WindowDef`,
  :class:`OnCondition`, :class:`UsingCondition`

*IQT-Knoten* – die eigentlichen Baumknoten, je abgeleitet von :class:`IQTNode`:
  :class:`ScanNode`, :class:`FilterNode`, :class:`ProjectNode`,
  :class:`JoinNode`, :class:`GroupAggNode`, :class:`SortNode`,
  :class:`LimitNode`, :class:`WindowNode`, :class:`SubqueryNode`,
  :class:`SetOpNode`, :class:`WithNode`

Kurzbeispiel
------------
::

    from uqtl.iqt import *

    scan   = ScanNode(node_type=NodeType.SCAN, table="orders", alias="t1")
    pred   = CompareExpr(ColumnRef("status", "t1"), CompareOp.EQ, Literal("delivered"))
    filt   = FilterNode(node_type=NodeType.FILTER, source=scan, predicate=pred)
    proj   = ProjectNode(node_type=NodeType.PROJECT, source=filt,
                         columns=[ProjectionItem(expr=ColumnRef("id", "t1"))])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Union


# ─── Enumerationen ────────────────────────────────────────────────────────────

class NodeType(Enum):
    """Kennzeichnet den Typ eines IQT-Knotens.

    Jeder :class:`IQTNode` trägt diesen Typ als Attribut; er wird
    in :meth:`IQTNode.__post_init__` automatisch gesetzt.
    """
    SCAN = auto()       #: Basistabellenzugriff: SELECT … FROM table
    FILTER = auto()     #: Filteroperation: WHERE- oder HAVING-Klausel
    PROJECT = auto()    #: Projektion: SELECT-Spaltenliste
    JOIN = auto()       #: Verbund zweier Teilbäume
    GROUP_AGG = auto()  #: Gruppierung mit Aggregatfunktionen
    SORT = auto()       #: Sortierung: ORDER BY
    LIMIT = auto()      #: Mengenbegrenzung: LIMIT / OFFSET
    WINDOW = auto()     #: Fensterfunktionen: OVER (...)
    SUBQUERY = auto()   #: Eingebettete Unterabfrage
    SET_OP = auto()     #: Mengenoperation: UNION, INTERSECT, EXCEPT
    WITH = auto()       #: Gemeinsamer Tabellenausdruck: WITH … AS (…)


class JoinType(Enum):
    """SQL-JOIN-Varianten gemäß ANSI SQL:2016."""
    INNER = "INNER"             #: Schnittmenge beider Mengen
    LEFT_OUTER = "LEFT OUTER"   #: Alle Zeilen der linken Seite
    RIGHT_OUTER = "RIGHT OUTER" #: Alle Zeilen der rechten Seite
    FULL_OUTER = "FULL OUTER"   #: Vereinigung beider Seiten
    CROSS = "CROSS"             #: Kartesisches Produkt


class SortDirection(Enum):
    """Sortierrichtung eines ORDER-BY-Ausdrucks."""
    ASC = "ASC"   #: Aufsteigend (Standard)
    DESC = "DESC" #: Absteigend


class NullsOrder(Enum):
    """Positionierung von NULL-Werten innerhalb einer sortierten Ergebnismenge."""
    NULLS_FIRST = "NULLS FIRST" #: NULLs vor allen anderen Werten
    NULLS_LAST = "NULLS LAST"   #: NULLs nach allen anderen Werten
    DEFAULT = ""                #: Datenbankspezifisches Standardverhalten


class AggFunc(Enum):
    """Unterstützte SQL-Aggregatfunktionen für :class:`GroupAggNode`."""
    COUNT = "COUNT"               #: Anzahl der Zeilen
    SUM = "SUM"                   #: Summe
    AVG = "AVG"                   #: Arithmetischer Mittelwert
    MIN = "MIN"                   #: Minimum
    MAX = "MAX"                   #: Maximum
    COUNT_DISTINCT = "COUNT_DISTINCT" #: Anzahl verschiedener Werte
    STDDEV = "STDDEV"             #: Standardabweichung
    VAR = "VAR"                   #: Varianz
    STRING_AGG = "STRING_AGG"     #: Verkettung von Zeichenketten


class SetOpType(Enum):
    """Mengenoperatoren für :class:`SetOpNode`."""
    UNION = "UNION"           #: Vereinigung (ohne Duplikate)
    UNION_ALL = "UNION ALL"   #: Vereinigung (mit Duplikaten)
    INTERSECT = "INTERSECT"   #: Schnittmenge
    EXCEPT = "EXCEPT"         #: Differenz


class FrameUnit(Enum):
    """Einheit des Fensterrahmens in :class:`WindowFrame`."""
    ROWS = "ROWS"   #: Physische Zeilenanzahl
    RANGE = "RANGE" #: Logischer Wertebereich


class WindowFunc(Enum):
    """Fensterfunktionen für :class:`WindowDef`.

    Rangfunktionen geben innerhalb einer Partition eine fortlaufende
    Positions- oder Rangzahl zurück; Offset-Funktionen greifen auf
    benachbarte Zeilen zu; Aggregatfunktionen berechnen rollende Werte.
    """
    ROW_NUMBER = "ROW_NUMBER"     #: Eindeutige Zeilennummer pro Partition
    RANK = "RANK"                 #: Rang mit Lücken bei Gleichständen
    DENSE_RANK = "DENSE_RANK"     #: Rang ohne Lücken
    LAG = "LAG"                   #: Wert der vorherigen Zeile
    LEAD = "LEAD"                 #: Wert der nächsten Zeile
    FIRST_VALUE = "FIRST_VALUE"   #: Erster Wert im Fenster
    LAST_VALUE = "LAST_VALUE"     #: Letzter Wert im Fenster
    NTH_VALUE = "NTH_VALUE"       #: N-ter Wert im Fenster
    SUM = "SUM"                   #: Laufende Summe
    AVG = "AVG"                   #: Gleitender Durchschnitt
    COUNT = "COUNT"               #: Laufende Anzahl
    MIN = "MIN"                   #: Laufendes Minimum
    MAX = "MAX"                   #: Laufendes Maximum


class CompareOp(Enum):
    """Binäre Vergleichsoperatoren für :class:`CompareExpr`."""
    EQ = "="    #: Gleichheit
    NEQ = "!="  #: Ungleichheit
    LT = "<"    #: Kleiner als
    GT = ">"    #: Größer als
    LTE = "<="  #: Kleiner oder gleich
    GTE = ">="  #: Größer oder gleich


class LogicOp(Enum):
    """Logische Operatoren für :class:`LogicExpr`."""
    AND = "AND" #: Konjunktion (alle Operanden müssen wahr sein)
    OR = "OR"   #: Disjunktion (mindestens ein Operand muss wahr sein)
    NOT = "NOT" #: Negation (invertiert den einzelnen Operanden)


class FilterPosition(Enum):
    """Gibt an, ob ein :class:`FilterNode` als WHERE- oder HAVING-Klausel generiert wird."""
    WHERE = "WHERE"   #: Standard-Filterklausel nach FROM
    HAVING = "HAVING" #: Filterklausel nach GROUP BY


# ─── Ausdrucks-Typen (Expr) ───────────────────────────────────────────────────

@dataclass
class ColumnRef:
    """Referenz auf eine Tabellenspalte.

    Attributes:
        column: Name der Spalte.
        table:  Optionaler Tabellenalias oder Tabellenname (z. B. ``t1``).

    Example::

        ColumnRef(column="id", table="t1")   # → t1.id
        ColumnRef(column="name")             # → name
    """
    column: str
    table: Optional[str] = None

    def __repr__(self) -> str:
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column


@dataclass
class Literal:
    """Skalarer Literalwert im Ausdrucksbaum.

    Attributes:
        value: Der tatsächliche Python-Wert.
                Erlaubte Typen: ``str``, ``int``, ``float``, ``bool``, ``None``.
        dtype: Optionaler Typ-Hinweis (z. B. ``"integer"``, ``"text"``).

    Example::

        Literal(42)          # → 42
        Literal("Hallo")     # → 'Hallo'
        Literal(None)        # → NULL
        Literal(True)        # → TRUE
    """
    value: Any
    dtype: str = "unknown"

    def __repr__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        if self.value is None:
            return "NULL"
        return str(self.value)


@dataclass
class FuncCall:
    """Aufruf einer SQL-Funktion (skalare oder Aggregatfunktion).

    Attributes:
        name:     Funktionsname in Großbuchstaben (z. B. ``"COALESCE"``, ``"EXTRACT"``).
        args:     Geordnete Liste der Argumente als Ausdrucksknoten.
        distinct: Falls ``True``, wird ``DISTINCT`` vor die Argumentliste gesetzt
                  (relevant für ``COUNT(DISTINCT …)``).
    """
    name: str
    args: List["Expr"] = field(default_factory=list)
    distinct: bool = False


@dataclass
class CompareExpr:
    """Binärer Vergleichsausdruck ``left op right``.

    Attributes:
        left:  Linker Operand.
        op:    Vergleichsoperator (``=``, ``!=``, ``<``, ``>``, ``<=``, ``>=``).
        right: Rechter Operand.

    Example::

        CompareExpr(ColumnRef("age"), CompareOp.GTE, Literal(18))
        # → age >= 18
    """
    left: "Expr"
    op: CompareOp
    right: "Expr"


@dataclass
class LogicExpr:
    """Logischer Ausdruck mit einem oder mehreren Operanden.

    Attributes:
        op:       Logischer Operator: ``AND``, ``OR`` oder ``NOT``.
        operands: Liste der Teilausdrücke.
                  Bei ``NOT`` genau ein Element.

    Example::

        LogicExpr(LogicOp.AND, [pred1, pred2])  # → pred1 AND pred2
        LogicExpr(LogicOp.NOT, [pred1])         # → NOT (pred1)
    """
    op: LogicOp
    operands: List["Expr"] = field(default_factory=list)


@dataclass
class InExpr:
    """Prüft, ob ein Wert in einer Liste oder Unterabfrage enthalten ist.

    Attributes:
        value:    Zu prüfender Ausdruck.
        items:    Literal-Werte für ``IN (v1, v2, …)``.
        subquery: Unterabfrage für ``IN (SELECT …)``; schließt ``items`` aus.
        negated:  Falls ``True``, wird ``NOT IN`` generiert.
    """
    value: "Expr"
    items: List["Expr"] = field(default_factory=list)
    subquery: Optional["SubqueryNode"] = None
    negated: bool = False


@dataclass
class BetweenExpr:
    """Bereichsprüfung ``value [NOT] BETWEEN lo AND hi``.

    Attributes:
        value:   Zu prüfender Ausdruck.
        lo:      Untere Grenze (inklusiv).
        hi:      Obere Grenze (inklusiv).
        negated: Falls ``True``, wird ``NOT BETWEEN`` generiert.
    """
    value: "Expr"
    lo: "Expr"
    hi: "Expr"
    negated: bool = False


@dataclass
class LikeExpr:
    """Mustervergleich ``value [NOT] LIKE pattern [ESCAPE escape]``.

    Attributes:
        value:   Zu prüfender Ausdruck.
        pattern: SQL-LIKE-Muster (``%`` = beliebige Zeichenfolge, ``_`` = ein Zeichen).
        escape:  Optionales Escape-Zeichen.
        negated: Falls ``True``, wird ``NOT LIKE`` generiert.
    """
    value: "Expr"
    pattern: str
    escape: Optional[str] = None
    negated: bool = False


@dataclass
class NullExpr:
    """NULL-Prüfung ``value IS [NOT] NULL``.

    Attributes:
        value:   Zu prüfender Ausdruck.
        negated: ``False`` → ``IS NULL``, ``True`` → ``IS NOT NULL``.
    """
    value: "Expr"
    negated: bool = False  # False = IS NULL, True = IS NOT NULL


@dataclass
class ExistsExpr:
    """Existenzprüfung ``[NOT] EXISTS (subquery)``.

    Attributes:
        subquery: Die eingebettete Unterabfrage.
        negated:  ``False`` → ``EXISTS``, ``True`` → ``NOT EXISTS``.
    """
    subquery: "SubqueryNode"
    negated: bool = False


# Union-Typ für alle Ausdrücke
Expr = Union[
    ColumnRef, Literal, FuncCall,
    CompareExpr, LogicExpr, InExpr, BetweenExpr,
    LikeExpr, NullExpr, ExistsExpr,
]


# ─── Hilfs-Datenklassen ───────────────────────────────────────────────────────

@dataclass
class ProjectionItem:
    """Ein einzelner Eintrag in der SELECT-Spaltenliste.

    Attributes:
        expr:  Ausdruck (Spaltenreferenz, Funktion, …).
        alias: Optionaler Ausgabename (``AS alias``).
        star:  Falls ``True``, wird ``*`` generiert (``expr`` wird ignoriert).
    """
    expr: Expr
    alias: Optional[str] = None
    star: bool = False


@dataclass
class SortItem:
    """Ein Sortierkriterium in ORDER BY oder OVER (… ORDER BY …).

    Attributes:
        expr:      Sortierausdruck.
        direction: :attr:`SortDirection.ASC` oder :attr:`SortDirection.DESC`.
        nulls:     Positionierung von NULL-Werten.
    """
    expr: Expr
    direction: SortDirection = SortDirection.ASC
    nulls: NullsOrder = NullsOrder.DEFAULT


@dataclass
class AggItem:
    """Beschreibt eine Aggregatfunktion samt Ausgabealias.

    Attributes:
        func:     Aggregatfunktion (z. B. :attr:`AggFunc.SUM`).
        arg:      Argument der Funktion; ``None`` für ``COUNT(*)``.
        distinct: Falls ``True``, wird ``DISTINCT`` innerhalb der Funktion gesetzt.
        alias:    Optionaler Ausgabename der Aggregatspalte.
    """
    func: AggFunc
    arg: Optional[Expr] = None   # None = COUNT(*)
    distinct: bool = False
    alias: Optional[str] = None


@dataclass
class WindowFrame:
    """Definiert den Fensterrahmen einer Fensterfunktion.

    Attributes:
        unit:  :attr:`FrameUnit.ROWS` oder :attr:`FrameUnit.RANGE`.
        start: Startgrenze, z. B. ``"UNBOUNDED PRECEDING"`` oder ``"1 PRECEDING"``.
        end:   Endgrenze,   z. B. ``"CURRENT ROW"`` oder ``"UNBOUNDED FOLLOWING"``.
    """
    unit: FrameUnit
    start: str = "UNBOUNDED PRECEDING"
    end: str = "CURRENT ROW"


@dataclass
class WindowDef:
    """Vollständige Definition einer Fensterfunktion.

    Entspricht ``func([func_args]) OVER (PARTITION BY … ORDER BY … frame) AS result_alias``.

    Attributes:
        func:         Zu verwendende Fensterfunktion.
        result_alias: Spaltenname im Ergebnis.
        partition_by: Ausdrücke für die PARTITION-BY-Klausel.
        order_by:     Sortierkriterien innerhalb des Fensters.
        frame:        Optionale Rahmenspezifikation (:class:`WindowFrame`).
        func_args:    Argumente der Fensterfunktion (z. B. Offset bei LAG/LEAD).
    """
    func: WindowFunc
    result_alias: str
    partition_by: List[Expr] = field(default_factory=list)
    order_by: List[SortItem] = field(default_factory=list)
    frame: Optional[WindowFrame] = None
    func_args: List[Expr] = field(default_factory=list)


@dataclass
class OnCondition:
    """JOIN-Bedingung in der Form ``ON predicate``.

    Attributes:
        predicate: Boolescher Ausdruck, der die Verbundsbedingung beschreibt.
    """
    predicate: Expr


@dataclass
class UsingCondition:
    """JOIN-Bedingung in der Form ``USING (col1, col2, …)``.

    Attributes:
        columns: Spaltennamen, die in beiden Tabellen vorhanden sein müssen.
    """
    columns: List[str]


JoinCondition = Union[OnCondition, UsingCondition]
"""Union-Typ für JOIN-Bedingungen: entweder :class:`OnCondition` oder :class:`UsingCondition`."""



# ─── IQT-Knoten ───────────────────────────────────────────────────────────────

@dataclass
class IQTNode:
    """Abstrakte Basisklasse aller IQT-Operatorknoten.

    Alle konkreten Knotentypen erben von dieser Klasse und setzen
    :attr:`node_type` in :meth:`__post_init__` automatisch.

    Attributes:
        node_type: Kategorisierung des Knotens (wird durch Unterklassen gesetzt).
    """
    node_type: NodeType


@dataclass
class ScanNode(IQTNode):
    """Basistabellenzugriff – entspricht ``FROM table [AS alias]``.

    Attributes:
        table: Name der Tabelle oder View in der Datenbank.
        alias: Tabellenalias; nach Normalisierung stets ``t1``, ``t2``, …
    """
    table: str
    alias: Optional[str] = None

    def __post_init__(self):
        self.node_type = NodeType.SCAN


@dataclass
class FilterNode(IQTNode):
    """Filteroperation – entspricht einer WHERE- oder HAVING-Klausel.

    Attributes:
        source:    Eingabeknoten (zu filternde Relation).
        predicate: Boolescher Ausdrucksbaum der Filterbedingung.
        position:  :attr:`FilterPosition.WHERE` (Standard) oder
                   :attr:`FilterPosition.HAVING` für Aggregierungsfilter.
    """
    source: IQTNode
    predicate: Expr
    position: FilterPosition = FilterPosition.WHERE

    def __post_init__(self):
        self.node_type = NodeType.FILTER


@dataclass
class ProjectNode(IQTNode):
    """Projektionsoperation – entspricht der SELECT-Spaltenliste.

    Attributes:
        source:   Eingabeknoten.
        columns:  Geordnete Liste der Ausgabespalten als :class:`ProjectionItem`.
                  Leere Liste entspricht ``SELECT *``.
        distinct: Falls ``True``, wird ``SELECT DISTINCT`` generiert.
    """
    source: IQTNode
    columns: List[ProjectionItem] = field(default_factory=list)
    distinct: bool = False

    def __post_init__(self):
        self.node_type = NodeType.PROJECT


@dataclass
class JoinNode(IQTNode):
    """Verbundoperation zweier Eingaberelationen.

    Attributes:
        left:      Linker Eingabeknoten.
        right:     Rechter Eingabeknoten.
        join_type: Art des Verbunds (INNER, LEFT OUTER, …).
        condition: Verbundsbedingung als :class:`OnCondition` oder :class:`UsingCondition`.
    """
    left: IQTNode
    right: IQTNode
    join_type: JoinType
    condition: JoinCondition

    def __post_init__(self):
        self.node_type = NodeType.JOIN


@dataclass
class GroupAggNode(IQTNode):
    """Gruppierung mit Aggregatfunktionen – entspricht GROUP BY … [HAVING …].

    Attributes:
        source:     Eingabeknoten.
        group_by:   Ausdrücke der GROUP-BY-Klausel.
        aggregates: Liste der Aggregatfunktionen als :class:`AggItem`.
        having:     Optionaler HAVING-Ausdruck (alternativ als :class:`FilterNode`
                    mit :attr:`FilterPosition.HAVING` modellierbar).
    """
    source: IQTNode
    group_by: List[Expr] = field(default_factory=list)
    aggregates: List[AggItem] = field(default_factory=list)
    having: Optional[Expr] = None

    def __post_init__(self):
        self.node_type = NodeType.GROUP_AGG


@dataclass
class SortNode(IQTNode):
    """Sortieroperator – entspricht ORDER BY.

    Attributes:
        source: Eingabeknoten.
        items:  Sortierkriterien in absteigender Priorität als :class:`SortItem`.
    """
    source: IQTNode
    items: List[SortItem] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.SORT


@dataclass
class LimitNode(IQTNode):
    """Mengenbegrenzung – entspricht LIMIT [OFFSET].

    Attributes:
        source: Eingabeknoten.
        count:  Maximale Anzahl zurückzugebender Zeilen.
        offset: Anzahl übersprungener Zeilen (0-basiert).
    """
    source: IQTNode
    count: Optional[int] = None
    offset: Optional[int] = None

    def __post_init__(self):
        self.node_type = NodeType.LIMIT


@dataclass
class WindowNode(IQTNode):
    """Berechnung von Fensterfunktionen.

    Fügt dem Ergebnis der Eingaberelation eine oder mehrere Fensterspalten
    hinzu (``func() OVER (…) AS alias``).

    Attributes:
        source:      Eingabeknoten.
        definitions: Liste der Fensterfunktionsdefinitionen (:class:`WindowDef`).
    """
    source: IQTNode
    definitions: List[WindowDef] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.WINDOW


@dataclass
class SubqueryNode(IQTNode):
    """Eingebettete Unterabfrage, verwendbar als Ausdruck oder FROM-Quelle.

    Attributes:
        query: Wurzelknoten der Unterabfrage.
        alias: Optionaler Tabellenalias, wenn die Unterabfrage im FROM steht.
    """
    query: IQTNode
    alias: Optional[str] = None

    def __post_init__(self):
        self.node_type = NodeType.SUBQUERY


@dataclass
class SetOpNode(IQTNode):
    """Mengenoperation über zwei Abfragen (UNION, INTERSECT, EXCEPT).

    Attributes:
        left:  Linke Eingabeabfrage.
        right: Rechte Eingabeabfrage.
        op:    Art der Mengenoperation (:class:`SetOpType`).
    """
    left: IQTNode
    right: IQTNode
    op: SetOpType

    def __post_init__(self):
        self.node_type = NodeType.SET_OP


@dataclass
class CTEDef:
    """Definition eines einzelnen CTE innerhalb eines WITH-Ausdrucks.

    Attributes:
        name:  Bezeichner des CTE (Tabellenname im WITH-Körper).
        query: Wurzelknoten der CTE-Abfrage.
    """
    name: str
    query: IQTNode


@dataclass
class WithNode(IQTNode):
    """Gemeinsamer Tabellenausdruck (Common Table Expression, CTE).

    Entspricht ``WITH cte1 AS (…), cte2 AS (…) SELECT …``.

    Attributes:
        ctes:  Geordnete Liste der CTE-Definitionen (:class:`CTEDef`).
        query: Optionale Hauptabfrage, die die CTEs verwendet.
               ``None``, wenn nur CTEs ohne abschließende Abfrage definiert sind.
    """
    ctes: List[CTEDef] = field(default_factory=list)
    query: Optional[IQTNode] = None

    def __post_init__(self):
        self.node_type = NodeType.WITH
