"""
experiments.py
==============
Umfassende experimentelle Demonstrationen der Rotationsmethode.

Dieses Modul zeigt die Rotationsmethode zur Kurvendiskussion
(Epp, 2025) durch detaillierte matplotlib-Visualisierungen und
Experiment-Daten. Die Rotationsmethode transformiert Wendepunkte
zu Extrema durch geometrische Rotation des Funktionsgraphen.

Zentrale Idee
=============
Sei f eine Funktion mit Wendepunkt x₀ und Tangentensteigung m = f'(x₀).
Rotiert man den Graphen Γ_f um φ* = −arctan(m), so wird (x₀, f(x₀))
ein Punkt mit f̃'(x̃₀) = 0 in der rotierten Bildfunktion f̃.

Dargestellt wird hier:
  1. Originalgraphen mit analytischem Verhalten
  2. Rotierte Bilder bei verschiedenen Winkeln
  3. Zusammenhang zwischen f'- und f̃'-Verläufen
  4. Winkel-Sortier-Tabellen (WST) für alle relevanten Punkte
  5. Numerische Experiment-Daten mit Genauigkeitsmetriken

Verwendete Funktionen (Testbeispiele)
======================================
  • f₁(x) = x³ − 3x: Kubische Funktion mit Wendepunkt bei x = 0
  • f₂(x) = x⁵ − 5x³ + 4x: Quint. Fkt. mit 2 Wendepunkten
  • f₃(x) = sin(x): Trigonometr. Fkt. mit periodyischen Wendepunkten
  • f₄(x) = x³ − 2x² + x: Kubische Fkt. mit flacher Stelle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import math
from typing import Callable, Tuple, List, Optional
import sys
from pathlib import Path

# Füge src-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent))

from rotation_analysis import (
    PointType, CurvePoint, AngleSortTable,
    rotate_point, rotate_curve, rotated_slope, analytical_angle,
    classify_point, build_wst, format_wst
)

__all__ = [
    "test_functions",
    "plot_function_analysis",
    "plot_rotation_series",
    "plot_rotation_angle_effect",
    "plot_wst_visualization",
    "run_all_experiments",
]


# ============================================================================
# TESTFUNKTIONEN UND IHRE ABLEITUNGEN
# ============================================================================

class TestFunction:
    """Wrapper für eine Testfunktion mit Ableitungen und Nullstellen."""

    def __init__(
        self,
        name: str,
        f: Callable[[float], float],
        f_prime: Callable[[float], float],
        f_double_prime: Optional[Callable[[float], float]] = None,
        critical_points: Optional[List[float]] = None,
        domain: Tuple[float, float] = (-3, 3),
    ):
        """
        name: Beschreibung der Funktion
        f: Funktion
        f_prime: Erste Ableitung
        f_double_prime: Zweite Ableitung (optional)
        critical_points: Nullstellen von f', f'', oder Wendepunktkandidaten
        domain: Definitionsbereich für Plots
        """
        self.name = name
        self.f = f
        self.f_prime = f_prime
        self.f_double_prime = f_double_prime
        self.critical_points = critical_points or []
        self.domain = domain

    def eval_f(self, x_array: np.ndarray) -> np.ndarray:
        """Vektorisierte Auswertung von f."""
        return np.array([self.f(x) for x in x_array])

    def eval_f_prime(self, x_array: np.ndarray) -> np.ndarray:
        """Vektorisierte Auswertung von f'."""
        return np.array([self.f_prime(x) for x in x_array])

    def eval_f_prime_prime(self, x_array: np.ndarray) -> np.ndarray:
        """Vektorisierte Auswertung von f''."""
        if self.f_double_prime is None:
            raise ValueError("f'' nicht definiert")
        return np.array([self.f_double_prime(x) for x in x_array])


def test_functions() -> dict[str, TestFunction]:
    """Rückgabe aller Testfunktionen als Wörterbuch."""
    functions = {}

    # F1: x³ − 3x (kubische Standardfunktion)
    functions["f1_cubic"] = TestFunction(
        name=r"$f(x) = x^3 - 3x$",
        f=lambda x: x**3 - 3*x,
        f_prime=lambda x: 3*x**2 - 3,
        f_double_prime=lambda x: 6*x,
        critical_points=[0.0, -1.0, 1.0],
        domain=(-2.5, 2.5),
    )

    # F2: x⁵ − 5x³ + 4x (Quint. Fkt. mit 2 Wendepunkten)
    # Wendepunkte bei x = ±1
    functions["f2_quintic"] = TestFunction(
        name=r"$f(x) = x^5 - 5x^3 + 4x$",
        f=lambda x: x**5 - 5*x**3 + 4*x,
        f_prime=lambda x: 5*x**4 - 15*x**2 + 4,
        f_double_prime=lambda x: 20*x**3 - 30*x,
        critical_points=[-1.0, 0.0, 1.0],
        domain=(-2.2, 2.2),
    )

    # F3: sin(x) (Trigonometrische Funktion)
    functions["f3_sine"] = TestFunction(
        name=r"$f(x) = \sin(x)$",
        f=lambda x: np.sin(x),
        f_prime=lambda x: np.cos(x),
        f_double_prime=lambda x: -np.sin(x),
        critical_points=[-np.pi, 0.0, np.pi],
        domain=(-2*np.pi, 2*np.pi),
    )

    # F4: x³ − 2x² + x (Kubische Fkt. mit Wendepunkt + Terassenpunkt)
    # f'(x) = 3x² − 4x + 1 = (3x − 1)(x − 1)
    # f''(x) = 6x − 4; WP bei x = 2/3
    functions["f4_mixed"] = TestFunction(
        name=r"$f(x) = x^3 - 2x^2 + x$",
        f=lambda x: x**3 - 2*x**2 + x,
        f_prime=lambda x: 3*x**2 - 4*x + 1,
        f_double_prime=lambda x: 6*x - 4,
        critical_points=[1/3, 1.0, 2/3],
        domain=(-0.5, 1.8),
    )

    return functions


# ============================================================================
# VISUALISIERUNG 1: Komplette Funktionsanalyse
# ============================================================================

def plot_function_analysis(
    test_func: TestFunction,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Detaillierte Analyse einer Funktion mit 4 Sub-Plots:
    
    [Oben-Links]     Originalfunktion f(x) mit kritischen Punkten
    [Oben-Rechts]    Erste Ableitung f'(x) mit Nullstellen
    [Unten-Links]    Zweite Ableitung f''(x) mit Wendepunktkandidaten
    [Unten-Rechts]   Winkel-Sortier-Tabelle (Textformat)
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    x_min, x_max = test_func.domain
    x_fine = np.linspace(x_min, x_max, 500)

    # ---- Subplot 1: Originalfunktion f(x) ----
    ax1 = fig.add_subplot(gs[0, 0])
    y_f = test_func.eval_f(x_fine)
    ax1.plot(x_fine, y_f, "b-", linewidth=2.5, label=test_func.name)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.grid(True, alpha=0.3)

    # Kritische Punkte eintragen
    for cp in test_func.critical_points:
        if x_min <= cp <= x_max:
            y_cp = test_func.f(cp)
            ax1.plot(cp, y_cp, "ro", markersize=8, zorder=5)
            ax1.annotate(
                f"x = {cp:.2f}", xy=(cp, y_cp), xytext=(10, 10),
                textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
            )

    ax1.set_xlabel("x", fontsize=11, fontweight="bold")
    ax1.set_ylabel("f(x)", fontsize=11, fontweight="bold")
    ax1.set_title("Originalfunktion f(x)", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)

    # ---- Subplot 2: Erste Ableitung f'(x) ----
    ax2 = fig.add_subplot(gs[0, 1])
    y_f_prime = test_func.eval_f_prime(x_fine)
    ax2.plot(x_fine, y_f_prime, "g-", linewidth=2.5, label="f'(x)")
    ax2.axhline(0, color="red", linestyle="-", alpha=0.7, linewidth=1.5, label="f'(x) = 0")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.fill_between(x_fine, 0, y_f_prime, where=(y_f_prime >= 0), alpha=0.2, color="green", label="f' > 0 (steigend)")
    ax2.fill_between(x_fine, 0, y_f_prime, where=(y_f_prime < 0), alpha=0.2, color="red", label="f' < 0 (fallend)")
    ax2.grid(True, alpha=0.3)

    # Nullstellen von f' markieren
    for cp in test_func.critical_points:
        if x_min <= cp <= x_max and abs(test_func.f_prime(cp)) < 0.1:
            ax2.plot(cp, 0, "go", markersize=8, zorder=5)

    ax2.set_xlabel("x", fontsize=11, fontweight="bold")
    ax2.set_ylabel("f'(x)", fontsize=11, fontweight="bold")
    ax2.set_title("Erste Ableitung f'(x) – Monotonie", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)

    # ---- Subplot 3: Zweite Ableitung f''(x) ----
    ax3 = fig.add_subplot(gs[1, 0])
    try:
        y_f_prime_prime = test_func.eval_f_prime_prime(x_fine)
        ax3.plot(x_fine, y_f_prime_prime, "m-", linewidth=2.5, label="f''(x)")
        ax3.axhline(0, color="red", linestyle="-", alpha=0.7, linewidth=1.5, label="f''(x) = 0")
        ax3.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax3.fill_between(x_fine, 0, y_f_prime_prime, where=(y_f_prime_prime >= 0), alpha=0.2, color="purple", label="f'' > 0 (konvex)")
        ax3.fill_between(x_fine, 0, y_f_prime_prime, where=(y_f_prime_prime < 0), alpha=0.2, color="orange", label="f'' < 0 (konkav)")

        # Wendepunktkandidaten markieren
        for cp in test_func.critical_points:
            if x_min <= cp <= x_max and abs(test_func.f_double_prime(cp)) < 0.1:
                ax3.plot(cp, 0, "mo", markersize=8, zorder=5)

    except ValueError:
        ax3.text(0.5, 0.5, "f'' nicht verfügbar", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("x", fontsize=11, fontweight="bold")
    ax3.set_ylabel("f''(x)", fontsize=11, fontweight="bold")
    ax3.set_title("Zweite Ableitung f''(x) – Krümmung", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)

    # ---- Subplot 4: Winkel-Sortier-Tabelle (WST) als Text ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # WST erzeugen
    wst = build_wst(test_func.f, test_func.f_prime, test_func.critical_points, classify=True)
    wst_text = format_wst(wst)

    # Zusätzliche Info-Box
    info_text = f"Rotation-Methode: Winkel-Sortier-Tabelle (WST)\n\n"
    info_text += f"Experiment-Daten für {test_func.name}:\n"
    info_text += f"  Domain: [{x_min:.2f}, {x_max:.2f}]\n"
    info_text += f"  Kritische Punkte: {len(test_func.critical_points)}\n"
    info_text += f"  WST-Punkte: {len(wst)}\n"
    info_text += f"  Rotationswinkel-Spannweite: {wst.total_span:.2f}°\n\n"
    info_text += "Analytische Winkel (sortiert):\n"
    info_text += "─" * 50 + "\n"

    for i, p in enumerate(wst.points):
        marker = "●" if p.kind == PointType.WENDEPUNKT else "○"
        info_text += f"{marker} x={p.x:+6.3f}  α={p.angle_deg:+7.2f}°  {p.kind.name}\n"

    info_text += "─" * 50 + "\n"
    info_text += f"α_min = {wst.alpha_min:+.2f}° | α_max = {wst.alpha_max:+.2f}°\n"

    ax4.text(
        0.05, 0.95, info_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(
        f"Vollständige Funktionsanalyse: {test_func.name}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    return fig


# ============================================================================
# VISUALISIERUNG 2: Rotationsserie bei verschiedenen Winkeln
# ============================================================================

def plot_rotation_series(
    test_func: TestFunction,
    angles_deg: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (18, 12),
) -> plt.Figure:
    """
    Zeigt den Effekt der Rotation auf den Funktionsgraphen
    bei verschiedenen Rotationswinkeln.
    
    Für jeden Winkel werden gezeigt:
      [Links]  Originalgraph (blau) + Rotierter Graph (rot)
      [Rechts] Vergleich der Ableitungen: f'(x) vs. f̃'(x̃)
    
    Parameter
    ---------
    angles_deg: Liste von Winkeln in Grad. Falls None, werden
               die analytischen Winkel aus den kritischen Punkten verwendet.
    """

    if angles_deg is None:
        # Analytische Winkel aus kritischen Punkten
        angles_deg = [analytical_angle(test_func.f_prime(cp)) for cp in test_func.critical_points]

    # Begrenzen auf maximal 4 Winkel für Übersicht
    angles_deg = sorted(set(round(a, 2) for a in angles_deg))[:4]

    num_angles = len(angles_deg)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(num_angles, 2, figure=fig, hspace=0.4, wspace=0.3)

    x_min, x_max = test_func.domain
    x_fine = np.linspace(x_min, x_max, 300)
    original_curve = list(zip(x_fine, test_func.eval_f(x_fine)))

    for row, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)

        # ---- Links: Geometrische Darstellung ----
        ax_geom = fig.add_subplot(gs[row, 0])

        # Originalgraph
        ax_geom.plot(x_fine, test_func.eval_f(x_fine), "b-", linewidth=2.5, label="Γ_f (Original)", alpha=0.8)

        # Rotierter Graph
        rotated = rotate_curve(original_curve, angle_rad)
        rot_x, rot_y = zip(*rotated)
        ax_geom.plot(rot_x, rot_y, "r-", linewidth=2.5, label=f"Γ_f^({angle_deg:.1f}°) (Rotiert)", alpha=0.8)

        # Kritische Punkte markieren
        for cp in test_func.critical_points:
            if x_min <= cp <= x_max:
                y_orig = test_func.f(cp)
                x_rot, y_rot = rotate_point(cp, y_orig, angle_rad)
                ax_geom.plot(cp, y_orig, "bo", markersize=6, zorder=4)
                ax_geom.plot(x_rot, y_rot, "ro", markersize=6, zorder=4)

        ax_geom.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.7)
        ax_geom.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.7)
        ax_geom.grid(True, alpha=0.2)
        ax_geom.set_xlabel("x", fontsize=10, fontweight="bold")
        ax_geom.set_ylabel("y", fontsize=10, fontweight="bold")
        ax_geom.set_title(f"Rotation φ = {angle_deg:.1f}°", fontsize=11, fontweight="bold")
        ax_geom.legend(loc="best", fontsize=9)

        # ---- Rechts: Ableitungen f'(x) und f̃'(x̃) ----
        ax_deriv = fig.add_subplot(gs[row, 1])

        y_f_prime = test_func.eval_f_prime(x_fine)
        ax_deriv.plot(x_fine, y_f_prime, "b-", linewidth=2.5, label="f'(x)", alpha=0.8)

        # f̃'(x̃) berechnen (numerisch über Finiten Differenzen)
        y_rot_prime = []
        for x in x_fine:
            m = test_func.f_prime(x)
            m_rot = rotated_slope(m, angle_rad)
            if m_rot is not None:
                y_rot_prime.append(m_rot)
            else:
                y_rot_prime.append(np.nan)

        ax_deriv.plot(x_fine, y_rot_prime, "r-", linewidth=2.5, label=f"f̃'(x̃) nach Rotation φ={angle_deg:.1f}°", alpha=0.8)
        ax_deriv.axhline(0, color="green", linestyle="-", linewidth=1.5, alpha=0.6, label="Extremum-Linie")
        ax_deriv.fill_between(x_fine, 0, y_f_prime, where=(np.array(y_f_prime) >= 0), alpha=0.15, color="blue")
        ax_deriv.fill_between(x_fine, 0, y_f_prime, where=(np.array(y_f_prime) < 0), alpha=0.15, color="blue")

        ax_deriv.grid(True, alpha=0.2)
        ax_deriv.set_xlabel("x", fontsize=10, fontweight="bold")
        ax_deriv.set_ylabel("f'(x) oder f̃'(x̃)", fontsize=10, fontweight="bold")
        ax_deriv.set_title(f"Ableitungen bei φ = {angle_deg:.1f}°", fontsize=11, fontweight="bold")
        ax_deriv.legend(loc="best", fontsize=9)
        ax_deriv.set_ylim(-3, 3)

    fig.suptitle(
        f"Rotationsserie für {test_func.name}: Geometrische Transformation",
        fontsize=14, fontweight="bold", y=0.995,
    )

    return fig


# ============================================================================
# VISUALISIERUNG 3: Rotationswinkeleffekt auf ein Extremum
# ============================================================================

def plot_rotation_angle_effect(
    test_func: TestFunction,
    critical_point: float,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Detaillierte Analyse: Wie ändert sich f̃'(x̃) wenn φ variiert?
    
    Sub-Plots:
    [Oben-Links]     f̃'(x̃) für verschiedene Rotationswinkel φ
    [Oben-Rechts]    Null-Stellen von f̃' als Funktion von φ
    [Unten-Links]    Funktionswert f im Originalkoordinaten
    [Unten-Rechts]   Krümmungsverhalten (f'') der rotierten Kurve
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    x_min, x_max = test_func.domain
    x_fine = np.linspace(x_min, x_max, 300)

    # Winkelbereich: −90° bis +90°
    angles_deg = np.linspace(-80, 80, 20)
    angles_rad = np.radians(angles_deg)

    # ---- Subplot 1: f̃'(x̃) für verschiedene Winkel ----
    ax1 = fig.add_subplot(gs[0, 0])

    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(angles_deg)))
    for angle_deg, angle_rad, color in zip(angles_deg, angles_rad, colors):
        y_rot_prime = []
        for x in x_fine:
            m = test_func.f_prime(x)
            m_rot = rotated_slope(m, angle_rad)
            y_rot_prime.append(m_rot if m_rot is not None else np.nan)

        ax1.plot(x_fine, y_rot_prime, color=color, linewidth=1.5, alpha=0.7, label=f"φ={angle_deg:.0f}°")

    ax1.axhline(0, color="black", linestyle="-", linewidth=2, alpha=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("x", fontsize=11, fontweight="bold")
    ax1.set_ylabel("f̃'(x̃)", fontsize=11, fontweight="bold")
    ax1.set_title("Ableitung der rotierten Kurve für verschiedene φ", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.set_ylim(-3, 3)

    # ---- Subplot 2: Null-Stellen von f̃' als Funktion von φ ----
    ax2 = fig.add_subplot(gs[0, 1])

    zero_angles = []
    zero_x_positions = []

    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        # Suche Null-Stellen von f̃'
        for i in range(len(x_fine) - 1):
            m1 = test_func.f_prime(x_fine[i])
            m2 = test_func.f_prime(x_fine[i+1])
            m_rot1 = rotated_slope(m1, angle_rad)
            m_rot2 = rotated_slope(m2, angle_rad)

            if m_rot1 is not None and m_rot2 is not None:
                if m_rot1 * m_rot2 < 0:  # Vorzeichenwechsel
                    zero_x = x_fine[i] + (x_fine[i+1] - x_fine[i]) * abs(m_rot1) / (abs(m_rot1) + abs(m_rot2))
                    zero_angles.append(angle_deg)
                    zero_x_positions.append(zero_x)

    if zero_angles:
        scatter = ax2.scatter(zero_angles, zero_x_positions, c=zero_angles, cmap="viridis", s=100, alpha=0.7, edgecolors="black", linewidth=1.5)
        plt.colorbar(scatter, ax=ax2, label="Rotationswinkel (°)")

    # Theoretischer analytischer Winkel
    m_crit = test_func.f_prime(critical_point)
    analytical_angle_deg = analytical_angle(m_crit)
    ax2.axvline(analytical_angle_deg, color="red", linestyle="--", linewidth=2, label=f"Analytischer Winkel α={analytical_angle_deg:.2f}°")
    ax2.axhline(critical_point, color="blue", linestyle="--", linewidth=2, alpha=0.6, label=f"Kritischer Punkt x={critical_point:.3f}")

    ax2.set_xlabel("Rotationswinkel φ (°)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("x-Position der Null-Stelle von f̃'", fontsize=11, fontweight="bold")
    ax2.set_title("Null-Stellen-Locus: φ vs. x", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    # ---- Subplot 3: Funktionsverlauf im Original ----
    ax3 = fig.add_subplot(gs[1, 0])

    y_f = test_func.eval_f(x_fine)
    ax3.plot(x_fine, y_f, "b-", linewidth=2.5, label=test_func.name)
    ax3.plot(critical_point, test_func.f(critical_point), "ro", markersize=10, label=f"Kritischer Punkt x={critical_point:.3f}", zorder=5)

    # Tangente zeichnen
    slope = test_func.f_prime(critical_point)
    y_crit = test_func.f(critical_point)
    x_tangent = np.array([critical_point - 1, critical_point + 1])
    y_tangent = y_crit + slope * (x_tangent - critical_point)
    ax3.plot(x_tangent, y_tangent, "r--", linewidth=2, alpha=0.7, label=f"Tangente (m={slope:.3f})")

    ax3.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.7)
    ax3.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("x", fontsize=11, fontweight="bold")
    ax3.set_ylabel("f(x)", fontsize=11, fontweight="bold")
    ax3.set_title("Originalfunktion mit kritischem Punkt", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)

    # ---- Subplot 4: Krümmung f''(x) ----
    ax4 = fig.add_subplot(gs[1, 1])

    try:
        y_f_prime_prime = test_func.eval_f_prime_prime(x_fine)
        ax4.plot(x_fine, y_f_prime_prime, "m-", linewidth=2.5, label="f''(x)")
        ax4.axhline(0, color="red", linestyle="-", linewidth=1.5, alpha=0.7)
        ax4.fill_between(x_fine, 0, y_f_prime_prime, where=(y_f_prime_prime >= 0), alpha=0.2, color="purple", label="Konvex")
        ax4.fill_between(x_fine, 0, y_f_prime_prime, where=(y_f_prime_prime < 0), alpha=0.2, color="orange", label="Konkav")
        ax4.plot(critical_point, test_func.f_double_prime(critical_point), "ro", markersize=10, zorder=5)
    except ValueError:
        ax4.text(0.5, 0.5, "f'' nicht verfügbar", ha="center", va="center", transform=ax4.transAxes)

    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("x", fontsize=11, fontweight="bold")
    ax4.set_ylabel("f''(x)", fontsize=11, fontweight="bold")
    ax4.set_title("Krümmungsverhalten f''(x)", fontsize=12, fontweight="bold")
    ax4.legend(loc="best", fontsize=10)

    fig.suptitle(
        f"Winkel-Effekte: Wie φ die rotierte Ableitung f̃' beeinflusst\nFunktion: {test_func.name}, Punkt: x={critical_point:.3f}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    return fig


# ============================================================================
# VISUALISIERUNG 4: Winkel-Sortier-Tabelle mit Daten
# ============================================================================

def plot_wst_visualization(
    test_func: TestFunction,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Visualisierung der Winkel-Sortier-Tabelle (WST).
    
    Sub-Plots:
    [Oben-Links]     Analytische Winkel α nach Größe sortiert (Säulendiagramm)
    [Oben-Rechts]    Steigungen m der kritischen Punkte (Balkendiagramm)
    [Unten-Links]    Zusammenhang: m ↦ α (Scatter mit Kurve)
    [Unten-Rechts]   Tabellarische Zusammenfassung (WST)
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # WST aufbauen
    wst = build_wst(test_func.f, test_func.f_prime, test_func.critical_points, classify=True)

    # ---- Subplot 1: Analytische Winkel (sortiert) ----
    ax1 = fig.add_subplot(gs[0, 0])

    colors_type = {
        PointType.MAXIMUM: "green",
        PointType.MINIMUM: "blue",
        PointType.WENDEPUNKT: "red",
        PointType.FLACH: "orange",
        PointType.UNBEKANNT: "gray",
    }

    point_labels = [f"x={p.x:.2f}\n({p.kind.name})" for p in wst.points]
    colors = [colors_type[p.kind] for p in wst.points]
    angles = [p.angle_deg for p in wst.points]

    bars = ax1.bar(range(len(wst.points)), angles, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    ax1.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax1.set_xticks(range(len(wst.points)))
    ax1.set_xticklabels(point_labels, fontsize=9)
    ax1.set_ylabel("Analytischer Winkel α (°)", fontsize=11, fontweight="bold")
    ax1.set_title("Analytische Winkel aller kritischen Punkte", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Legende
    legend_elements = [mpatches.Patch(facecolor=colors_type[pt], alpha=0.7, label=pt.name) for pt in PointType]
    ax1.legend(handles=legend_elements, loc="best", fontsize=9)

    # ---- Subplot 2: Steigungen m ----
    ax2 = fig.add_subplot(gs[0, 1])

    slopes = [p.slope for p in wst.points]
    colors = [colors_type[p.kind] for p in wst.points]

    bars = ax2.barh(range(len(wst.points)), slopes, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax2.set_yticks(range(len(wst.points)))
    ax2.set_yticklabels([f"x={p.x:.2f}" for p in wst.points], fontsize=9)
    ax2.set_xlabel("Tangentensteigung m = f'(x)", fontsize=11, fontweight="bold")
    ax2.set_title("Steigungen an kritischen Punkten", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # ---- Subplot 3: Zusammenhang m ↦ α ----
    ax3 = fig.add_subplot(gs[1, 0])

    # Theoretische Kurve: α(m) = −arctan(m)
    m_range = np.linspace(-5, 5, 200)
    alpha_range = np.array([analytical_angle(m) for m in m_range])

    ax3.plot(m_range, alpha_range, "k-", linewidth=2.5, label=r"α(m) = −arctan(m)", alpha=0.6)

    # Datenpunkte
    slopes = [p.slope for p in wst.points]
    angles = [p.angle_deg for p in wst.points]
    colors = [colors_type[p.kind] for p in wst.points]

    ax3.scatter(slopes, angles, c=colors, s=200, alpha=0.8, edgecolors="black", linewidth=2, zorder=5)

    for p in wst.points:
        ax3.annotate(f"x={p.x:.2f}", xy=(p.slope, p.angle_deg), xytext=(5, 5),
                     textcoords="offset points", fontsize=8, alpha=0.7)

    ax3.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
    ax3.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Steigung m = f'(x)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Analytischer Winkel α (°)", fontsize=11, fontweight="bold")
    ax3.set_title("Transformation: Steigung → Analytischer Winkel", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)

    # ---- Subplot 4: Tabellarische WST ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Tabelle als Text
    table_text = "WINKEL-SORTIER-TABELLE (WST)\n"
    table_text += "=" * 70 + "\n\n"
    table_text += f"{'i':<3} {'x':<10} {'f(x)':<12} {'m=f\'(x)':<10} {'α (°)':<10} {'Typ':<15}\n"
    table_text += "─" * 70 + "\n"

    for i, p in enumerate(wst.points):
        table_text += f"{i:<3} {p.x:+.4f}   {p.y:+.6f}   {p.slope:+.4f}   {p.angle_deg:+.2f}   {p.kind.name:<15}\n"

    table_text += "─" * 70 + "\n"
    table_text += f"Punkte: {len(wst)}\n"
    table_text += f"α_min = {wst.alpha_min:+.2f}° (Wendepunkt bei {wst.points[0].x:.3f})\n"
    table_text += f"α_max = {wst.alpha_max:+.2f}° (Wendepunkt bei {wst.points[-1].x:.3f})\n"
    table_text += f"Rotations-Spannweite = {wst.total_span:.2f}°\n"

    ax4.text(
        0.05, 0.95, table_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    fig.suptitle(
        f"Winkel-Sortier-Tabelle für {test_func.name}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    return fig


# ============================================================================
# EXPERIMENT: Alle Tests ausführen
# ============================================================================

def run_all_experiments(output_dir: str = "src/results") -> None:
    """
    Führt alle Experimente für alle Testfunktionen aus
    und speichert Visualisierungen als SVG-Dateien sowie
    Experiment-Daten in einer results.txt Datei.
    
    Parameter
    ---------
    output_dir : str
        Verzeichnis, in das die Ergebnisse gespeichert werden.
    """
    import os

    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║  ROTATIONSMETHODE ZUR KURVENDISKUSSION – EXPERIMENTELLE DEMONSTRATIONEN       ║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝\n")

    funcs = test_functions()
    
    # Ergebnisdatei vorbereiten
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write("=" * 80 + "\n")
        rf.write("EXPERIMENTELLE DEMONSTRATIONEN DER ROTATIONSMETHODE\n")
        rf.write("Epp, 2025\n")
        rf.write("=" * 80 + "\n\n")

    for func_key, test_func in funcs.items():
        print(f"\n{'─' * 80}")
        print(f"Experiment: {test_func.name}")
        print(f"{'─' * 80}")

        # Experiment-Daten sammeln
        with open(results_file, "a", encoding="utf-8") as rf:
            rf.write(f"\n{'─' * 80}\n")
            rf.write(f"Funktion: {test_func.name}\n")
            rf.write(f"{'─' * 80}\n\n")
            rf.write(f"Definitionsbereich: [{test_func.domain[0]:.3f}, {test_func.domain[1]:.3f}]\n")
            rf.write(f"Kritische Punkte: {len(test_func.critical_points)}\n")
            rf.write(f"Kritische Punkt-Positionen: {', '.join(f'{cp:.4f}' for cp in test_func.critical_points)}\n\n")

        # Experiment 1: Vollständige Analyse
        print(f"  ✓ Erzeuge: Vollständige Funktionsanalyse...")
        fig1 = plot_function_analysis(test_func)
        output_file_1 = os.path.join(output_dir, f"01_analysis_{func_key}.svg")
        fig1.savefig(output_file_1, format="svg", bbox_inches="tight")
        plt.close(fig1)
        print(f"    → Gespeichert: {output_file_1}")

        # Experiment 2: Rotationsserie
        print(f"  ✓ Erzeuge: Rotationsserie...")
        fig2 = plot_rotation_series(test_func)
        output_file_2 = os.path.join(output_dir, f"02_rotation_series_{func_key}.svg")
        fig2.savefig(output_file_2, format="svg", bbox_inches="tight")
        plt.close(fig2)
        print(f"    → Gespeichert: {output_file_2}")

        # Experiment 3: Rotationswinkeleffekt
        if test_func.critical_points:
            for critical_pt in test_func.critical_points[:2]:  # Maximal 2 pro Funktion
                print(f"  ✓ Erzeuge: Winkeleffekt bei x = {critical_pt:.3f}...")
                fig3 = plot_rotation_angle_effect(test_func, critical_pt)
                output_file_3 = os.path.join(
                    output_dir,
                    f"03_angle_effect_{func_key}_x{critical_pt:.3f}.svg"
                )
                fig3.savefig(output_file_3, format="svg", bbox_inches="tight")
                plt.close(fig3)
                print(f"    → Gespeichert: {output_file_3}")

        # Experiment 4: WST-Visualisierung
        print(f"  ✓ Erzeuge: Winkel-Sortier-Tabelle Visualisierung...")
        fig4 = plot_wst_visualization(test_func)
        output_file_4 = os.path.join(output_dir, f"04_wst_visualization_{func_key}.svg")
        fig4.savefig(output_file_4, format="svg", bbox_inches="tight")
        plt.close(fig4)
        print(f"    → Gespeichert: {output_file_4}")

        # WST-Daten in results.txt schreiben
        wst = build_wst(test_func.f, test_func.f_prime, test_func.critical_points, classify=True)
        with open(results_file, "a", encoding="utf-8") as rf:
            rf.write("WINKEL-SORTIER-TABELLE (WST):\n")
            rf.write("─" * 80 + "\n")
            rf.write(format_wst(wst) + "\n\n")

    print(f"\n{'─' * 80}")
    print(f"✓ Alle Experimente erfolgreich abgeschlossen!")
    print(f"✓ Ergebnisse gespeichert in: {results_file}")
    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    # Wenn direkt ausgeführt, alle Experimente starten
    run_all_experiments()
