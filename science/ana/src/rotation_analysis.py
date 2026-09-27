"""
rotation_analysis.py
====================
Implementierung der **Rotationsmethode zur Kurvendiskussion** (Epp 2025).

Kernidee
--------
Wendepunkte des Graphen von f werden durch geometrische Rotation zu
Extrema des rotierten Graphen – und können damit allein über die erste
Ableitung f' identifiziert werden.

Zentraler Satz (Rotationsprinzip)
----------------------------------
Sei f zweimal stetig differenzierbar und x₀ ein Wendepunkt mit f'(x₀) = m.
Rotiert man Γ_f um φ = −arctan(m), so gilt f̃'(x̃₀) = 0.

Öffentliche API
---------------
Geometrie:
    rotate_point(x, y, phi)          R_φ auf einen Punkt anwenden
    rotate_curve(points, phi)         R_φ auf eine ganze Kurve anwenden
    rotated_slope(m, phi)             Steigung f̃'(x̃) nach Rotation

Analyse:
    analytical_angle(m)               α(P) = −arctan(m) in Grad
    classify_point(f_prime, x)        Typ eines Kandidatenpunkts ermitteln

Winkel-Sortier-Tabelle (WST):
    PointType                         Enum: MAX, MIN, WENDEPUNKT, FLACH, UNBEKANNT
    CurvePoint                        Datenpunkt mit Koordinaten + α + Typ
    AngleSortTable                    Sortierte WST-Tabelle
    build_wst(f, f_prime, xs)         Fabrik-Funktion für AngleSortTable
    format_wst(wst)                   Formatierte ASCII-Ausgabe der Tabelle
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PointType",
    "CurvePoint",
    "AngleSortTable",
    "rotate_point",
    "rotate_curve",
    "rotated_slope",
    "analytical_angle",
    "classify_point",
    "build_wst",
    "format_wst",
]

# ---------------------------------------------------------------------------
# Interne Standardwerte
# ---------------------------------------------------------------------------

_SLOPE_TOL: float = 1e-9   # |m| < TOL gilt als Null
_SIGN_H:    float = 1e-6   # Schrittweite für Vorzeichenprüfung


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class PointType(Enum):
    """Klassifikation eines analytisch relevanten Punktes auf Γ_f."""
    MAXIMUM     = auto()   # Lokales Maximum       (f': + → −)
    MINIMUM     = auto()   # Lokales Minimum       (f': − → +)
    WENDEPUNKT  = auto()   # Wendepunkt             (f'(x) ≠ 0  ODER  f'' VZW)
    FLACH       = auto()   # Terrassenpunkt/Flach   (f'=0, kein VZW von f')
    UNBEKANNT   = auto()   # Noch nicht klassifiziert


# ---------------------------------------------------------------------------
# Datenklasse CurvePoint
# ---------------------------------------------------------------------------

@dataclass
class CurvePoint:
    """Ein analytisch relevanter Punkt auf dem Graphen von f.

    Attribute
    ---------
    x : float
        x-Koordinate.
    y : float
        Funktionswert f(x).
    slope : float
        Tangentensteigung f'(x) = m.
    kind : PointType
        Klassifikation des Punktes.
    angle_deg : float
        Analytischer Winkel α = −arctan(m) in Grad  [auto-berechnet].
    """

    x:     float
    y:     float
    slope: float
    kind:  PointType = PointType.UNBEKANNT

    # Automatisch aus slope berechnet; nicht im Konstruktor übergeben
    angle_deg: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.angle_deg = analytical_angle(self.slope)

    # -----------------------------------------------------------------------
    # Eigenschaften
    # -----------------------------------------------------------------------

    @property
    def angle_rad(self) -> float:
        """Analytischer Winkel in Radiant."""
        return math.radians(self.angle_deg)

    @property
    def coords(self) -> Tuple[float, float]:
        """(x, y) als Tupel."""
        return (self.x, self.y)

    def __str__(self) -> str:
        return (
            f"CurvePoint(x={self.x:+.4f}, y={self.y:+.4f}, "
            f"m={self.slope:+.4f}, α={self.angle_deg:+.2f}°, "
            f"kind={self.kind.name})"
        )


# ---------------------------------------------------------------------------
# Geometrie: Rotation R_φ
# ---------------------------------------------------------------------------

def rotate_point(x: float, y: float, phi: float) -> Tuple[float, float]:
    """Wende die Rotationsmatrix R_φ auf den Punkt (x, y) an.

    Formel:
        x' = x·cos φ − y·sin φ
        y' = x·sin φ + y·cos φ

    Parameter
    ---------
    x, y : float
        Koordinaten des Punktes.
    phi : float
        Rotationswinkel in **Radiant** (positiv = Gegenuhrzeigersinn).

    Rückgabe
    --------
    (x', y') : Tuple[float, float]
    """
    c = math.cos(phi)
    s = math.sin(phi)
    return (x * c - y * s,
            x * s + y * c)


def rotate_curve(
    points: Iterable[Tuple[float, float]],
    phi: float,
) -> List[Tuple[float, float]]:
    """Rotiere eine Kurve (Folge von (x, y)-Paaren) um den Winkel φ.

    Parameter
    ---------
    points : Iterable[(float, float)]
        Kurve Γ_f als Folge von (x, f(x))-Paaren.
    phi : float
        Rotationswinkel in Radiant.

    Rückgabe
    --------
    List[(float, float)]
        Rotiertes Bild Γ_f^(φ).
    """
    return [rotate_point(px, py, phi) for px, py in points]


def rotated_slope(m: float, phi: float) -> Optional[float]:
    """Berechne die Steigung f̃'(x̃) der rotierten Bildfunktion.

    Lemma 2.3 (Epp 2025):
        f̃'(x̃) = (sin φ + m·cos φ) / (cos φ − m·sin φ)

    Gibt **None** zurück, wenn der Nenner Null ist (rotierte Kurve
    ist an dieser Stelle senkrecht — kein lokaler Funktionsgraph).

    Parameter
    ---------
    m : float
        Tangentensteigung f'(x).
    phi : float
        Rotationswinkel in Radiant.
    """
    c = math.cos(phi)
    s = math.sin(phi)
    denom = c - m * s
    if math.isclose(denom, 0.0, abs_tol=1e-14):
        return None
    return (s + m * c) / denom


# ---------------------------------------------------------------------------
# Analytischer Winkel
# ---------------------------------------------------------------------------

def analytical_angle(slope: float) -> float:
    """Berechne den analytischen Winkel α(P) = −arctan(m) in **Grad**.

    Dies ist der Rotationswinkel, bei dem der Punkt P mit Tangentensteigung m
    als Extremum der rotierten Bildfunktion f̃ erscheint.

    Rückgabe immer im offenen Intervall (−90°, 90°).
    """
    return -math.degrees(math.atan(slope))


# ---------------------------------------------------------------------------
# Klassifikation
# ---------------------------------------------------------------------------

def classify_point(
    f_prime: Callable[[float], float],
    x: float,
    *,
    h:   float = _SIGN_H,
    tol: float = _SLOPE_TOL,
) -> PointType:
    """Klassifiziere einen Kandidatenpunkt durch Vorzeichenanalyse von f'.

    Regeln
    ------
    |f'(x)| > tol            → WENDEPUNKT (nichthorizontale Tangente)
    f'(x) ≈ 0, f': + → −    → MAXIMUM
    f'(x) ≈ 0, f': − → +    → MINIMUM
    f'(x) ≈ 0, kein VZW      → FLACH (Terrassenpunkt)

    Parameter
    ---------
    f_prime : Callable
        Erste Ableitung f'.
    x : float
        Kandidatenpunkt (Nullstelle von f' oder f'').
    h : float
        Schrittweite für die Vorzeichenprüfung links/rechts.
    tol : float
        Schwellenwert, unter dem |f'(x)| als Null gilt.
    """
    m     = f_prime(x)
    left  = f_prime(x - h)
    right = f_prime(x + h)

    if abs(m) > tol:
        return PointType.WENDEPUNKT

    # Steigung ≈ 0: Vorzeichenwechsel?
    if left > tol and right < -tol:
        return PointType.MAXIMUM
    if left < -tol and right > tol:
        return PointType.MINIMUM
    return PointType.FLACH


# ---------------------------------------------------------------------------
# Winkel-Sortier-Tabelle (WST)
# ---------------------------------------------------------------------------

@dataclass
class AngleSortTable:
    """Winkel-Sortier-Tabelle: alle analytisch relevanten Punkte,
    aufsteigend sortiert nach ihrem analytischen Winkel α_i.

    Attribute
    ---------
    points : List[CurvePoint]
        Sortierte Punkte.
    alpha_min : float
        Kleinster analytischer Winkel α (Grad).
    alpha_max : float
        Größter analytischer Winkel α (Grad).
    total_span : float
        Gesamtdrehung = α_max − α_min (Grad).
    """

    points: List[CurvePoint]

    alpha_min:  float = field(init=False)
    alpha_max:  float = field(init=False)
    total_span: float = field(init=False)

    def __post_init__(self) -> None:
        if self.points:
            angles         = [p.angle_deg for p in self.points]
            self.alpha_min = min(angles)
            self.alpha_max = max(angles)
        else:
            self.alpha_min = 0.0
            self.alpha_max = 0.0
        self.total_span = self.alpha_max - self.alpha_min

    # -----------------------------------------------------------------------
    # Abfragen
    # -----------------------------------------------------------------------

    def by_kind(self, kind: PointType) -> List[CurvePoint]:
        """Alle Punkte eines bestimmten Typs."""
        return [p for p in self.points if p.kind == kind]

    def support_angles(self) -> List[float]:
        """Deduplizierte, sortierte Stützwinkel (Grad).

        Das sind die Winkel, bei denen während der einmaligen Drehbewegung
        eine Analyse ausgeführt werden muss (WST-Kernidee: O(N²) → O(N)).
        """
        seen:   set           = set()
        result: List[float]   = []
        for p in self.points:
            key = round(p.angle_deg, 10)
            if key not in seen:
                seen.add(key)
                result.append(p.angle_deg)
        return result

    # -----------------------------------------------------------------------
    # Dunder
    # -----------------------------------------------------------------------

    def __len__(self)  -> int:           return len(self.points)
    def __iter__(self):                  return iter(self.points)
    def __getitem__(self, i: int):       return self.points[i]


# ---------------------------------------------------------------------------
# Fabrik-Funktion
# ---------------------------------------------------------------------------

def build_wst(
    f:        Callable[[float], float],
    f_prime:  Callable[[float], float],
    xs:       Sequence[float],
    *,
    classify: bool  = True,
    h:        float = _SIGN_H,
    tol:      float = _SLOPE_TOL,
) -> AngleSortTable:
    """Erzeuge eine AngleSortTable aus einer Liste von Kandidaten-x-Werten.

    Parameter
    ---------
    f, f_prime : Callable
        Funktion und ihre erste Ableitung.
    xs : Sequence[float]
        Kandidaten-x-Werte (Nullstellen von f' und/oder f'').
    classify : bool
        True → jeden Punkt klassifizieren; False → alle UNBEKANNT lassen.
    h, tol : float
        Weitergegeben an classify_point.

    Rückgabe
    --------
    AngleSortTable  (aufsteigend nach analytischem Winkel sortiert)
    """
    pts: List[CurvePoint] = []
    for x in xs:
        kind = (
            classify_point(f_prime, x, h=h, tol=tol)
            if classify
            else PointType.UNBEKANNT
        )
        pts.append(CurvePoint(x=x, y=f(x), slope=f_prime(x), kind=kind))

    pts.sort(key=lambda p: p.angle_deg)
    return AngleSortTable(points=pts)


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------

_HDR = ("x", "y", "Steigung m", "Winkel α°", "Typ", "Nachweis")
_WID = (9, 10, 12, 10, 12, 14)


def format_wst(wst: AngleSortTable) -> str:
    """Formatiere eine AngleSortTable als ausgerichtete ASCII-Tabelle.

    Enthält: x, y, Steigung, analytischen Winkel, Typ und Nachweis-Text.
    """
    def row(*cols: str) -> str:
        return " | ".join(c.ljust(_WID[i]) for i, c in enumerate(cols))

    def nachweis(p: CurvePoint) -> str:
        if   p.kind == PointType.MAXIMUM:    return "f': + → −"
        elif p.kind == PointType.MINIMUM:    return "f': − → +"
        elif p.kind == PointType.WENDEPUNKT: return f"VZW bei {p.angle_deg:+.1f}°"
        elif p.kind == PointType.FLACH:      return "kein VZW"
        else:                                return "—"

    sep   = "-+-".join("-" * w for w in _WID)
    lines = [row(*_HDR), sep]

    for p in wst.points:
        lines.append(row(
            f"{p.x:+.4f}",
            f"{p.y:+.4f}",
            f"{p.slope:+.4f}",
            f"{p.angle_deg:+.2f}",
            p.kind.name,
            nachweis(p),
        ))

    lines += [
        sep,
        (f"Punkte: {len(wst)}  |  "
         f"α_min = {wst.alpha_min:+.2f}°  |  "
         f"α_max = {wst.alpha_max:+.2f}°  |  "
         f"Gesamtdrehung = {wst.total_span:.2f}°"),
    ]
    return "\n".join(lines)
