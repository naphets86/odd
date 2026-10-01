"""Der trigonometrisch-harmonische Resonanzpunkt: Rechenmodul zum Kapitel
"Der trigonometrisch-harmonische Resonanzpunkt" (lowp.tex).

Abgebildet werden

* der Goldene Schnitt phi und seine Identitäten (Definition 1, Satz "Algebraische Eigenschaften"),
* der trigonometrische Gleichheitspunkt pi/4 (Definition 2),
* die harmonische Differenz Delta_h und die harmonische Triade (Theorem 1),
* das Resonanzfenster [1/phi, 1/phi + 2*Delta_h] (Definition 3) und Theorem 2,
* Zonen, Innovations-/Varianz-/Biodiversitätsmodelle (Theorem 3 und 4),
* die Fallstudien Schweiz, Argentinien, Sowjetunion und das Korngesetz.

Das Fenster ist eine Modellkonstruktion: Die exakten Werte der Konstanten sind bewiesen, die
Aussage, gerade dieses Intervall kennzeichne besondere Stabilität, ist eine Modellannahme
(so auch im Schlussteil der Arbeit abgegrenzt).

Korrigierte Zahlenwerte (aus den Formeln des Kapitels nachgerechnet)
    F(1/phi) = -ln(1 - 1/phi) = 0,9624       (Kapitel: 0,9547 bzw. 0,955)
    F(0,7071)                  = 1,2279       (Kapitel: 1,204)
    F(0,7962)                  = 1,5906       (Kapitel: 1,5406 im Satz, 1,5906 im Beweis)
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

_SQRT5 = math.sqrt(5.0)


def _check_count(n: int, name: str = "n", minimum: int = 1) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
        raise ValueError(f"{name} muss eine ganze Zahl >= {minimum} sein")
    return n


def _check_omega(omega: float) -> float:
    if not 0.0 <= omega < 1.0:
        raise ValueError("omega muss in [0, 1) liegen")
    return float(omega)


# --------------------------------------------------------------------------
# Definition 1 und Satz: Goldener Schnitt
# --------------------------------------------------------------------------

def phi() -> float:
    """Goldener Schnitt (1 + sqrt 5)/2 = 1,6180339887..., positive Lösung von x^2 = x + 1."""
    return (1.0 + _SQRT5) / 2.0


def phi_inv() -> float:
    """Kehrwert 1/phi = phi - 1 = (sqrt 5 - 1)/2 = 0,6180339887..."""
    return (_SQRT5 - 1.0) / 2.0


def golden_identities(tol: float = 1e-12) -> dict[str, bool]:
    """Prüft die Identitäten 1/phi = phi - 1, phi^2 = phi + 1 und phi^2 - phi = 1."""
    p = phi()
    return {
        "self_similarity": abs(1.0 / p - (p - 1.0)) <= tol,
        "quadratic_reduction": abs(p * p - (p + 1.0)) <= tol,
        "square_minus_phi": abs(p * p - p - 1.0) <= tol,
        "inverse_closed_form": abs(phi_inv() - (p - 1.0)) <= tol,
    }


def golden_quadratic_roots() -> tuple[float, float]:
    """Beide Lösungen von x^2 - x - 1 = 0: phi und -1/phi."""
    return phi(), -phi_inv()


def continued_fraction_phi(depth: int) -> float:
    """Kettenbruch 1 + 1/(1 + 1/(1 + ...)) mit ``depth`` Ebenen."""
    _check_count(depth, "depth", 0)
    value = 1.0
    for _ in range(depth):
        value = 1.0 + 1.0 / value
    return value


def fibonacci(n: int) -> int:
    """n-te Fibonacci-Zahl (F_0 = 0, F_1 = 1)."""
    _check_count(n, "n", 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fibonacci_ratios(n_max: int) -> list[float]:
    """Quotienten F_(n+1)/F_n für n = 1..n_max, die gegen phi konvergieren."""
    _check_count(n_max, "n_max")
    return [fibonacci(n + 1) / fibonacci(n) for n in range(1, n_max + 1)]


def fibonacci_ratio_error(n: int) -> float:
    """Betrag |F_(n+1)/F_n - phi| (alterniert im Vorzeichen, fällt wie phi^(-2n))."""
    _check_count(n, "n")
    return abs(fibonacci(n + 1) / fibonacci(n) - phi())


def golden_powers(n_max: int) -> list[float]:
    """Skalierung phi^(-n), n = 0..n_max, der wiederholten Rechteckzerlegung."""
    _check_count(n_max, "n_max", 0)
    return [phi() ** (-n) for n in range(n_max + 1)]


# --------------------------------------------------------------------------
# Definition 2: Trigonometrischer Gleichheitspunkt
# --------------------------------------------------------------------------

def theta_sym() -> float:
    """Eindeutiger Winkel in [0, pi/2] mit sin = cos: pi/4."""
    return math.pi / 4.0


def theta_sym_numeric(tol: float = 1e-14) -> float:
    """Bisektion der Gleichung sin(t) - cos(t) = 0 auf [0, pi/2] (unabhängige Probe von theta_sym)."""
    lo, hi = 0.0, math.pi / 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if math.sin(mid) - math.cos(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def s_c() -> float:
    """Gemeinsamer Wert sin(pi/4) = cos(pi/4) = sqrt(2)/2 = 0,7071067812..."""
    return math.sqrt(2.0) / 2.0


def isosceles_legs_equal(tol: float = 1e-12) -> bool:
    """Geometrische Deutung: Im rechtwinkligen Dreieck mit Winkel pi/4 sind die Katheten gleich lang."""
    t = theta_sym()
    return abs(math.sin(t) - math.cos(t)) <= tol


# --------------------------------------------------------------------------
# Theorem 1: Harmonische Triade
# --------------------------------------------------------------------------

def harmonic_difference() -> float:
    """Delta_h = sqrt(2)/2 - 1/phi = (sqrt 2 - sqrt 5 + 1)/2 = 0,089073..."""
    return s_c() - phi_inv()


def harmonic_difference_closed_form() -> float:
    """Geschlossene Form (sqrt 2 - sqrt 5 + 1)/2 von Delta_h."""
    return (math.sqrt(2.0) - _SQRT5 + 1.0) / 2.0


def window_bounds() -> tuple[float, float]:
    """Resonanzfenster [1/phi, 1/phi + 2*Delta_h] = [0,618034, 0,796181]."""
    lo = phi_inv()
    return lo, lo + 2.0 * harmonic_difference()


def window_upper_closed_form() -> float:
    """Obere Grenze (2 sqrt 2 - sqrt 5 + 1)/2."""
    return (2.0 * math.sqrt(2.0) - _SQRT5 + 1.0) / 2.0


def window_center() -> float:
    """Mittelpunkt des Fensters = sqrt(2)/2."""
    lo, hi = window_bounds()
    return 0.5 * (lo + hi)


def window_half_width() -> float:
    """Halbbreite des Fensters = Delta_h."""
    lo, hi = window_bounds()
    return 0.5 * (hi - lo)


def window_is_symmetric(tol: float = 1e-12) -> bool:
    """Das Fenster ist symmetrisch um s_c, und der Mittelpunkt ist lo + Delta_h."""
    lo, hi = window_bounds()
    return abs((lo + hi) / 2.0 - s_c()) <= tol and abs(lo + harmonic_difference() - s_c()) <= tol


def delta_h_is_irrational_witness() -> bool:
    """Plausibilitätsprobe: sqrt 2 - sqrt 5 + 1 ist keine rationale Zahl mit kleinem Nenner.

    Numerischer Hinweis, kein Beweis: kein Bruch p/q mit q <= 1000 trifft den Wert auf 1e-9 genau.
    """
    x = 2.0 * harmonic_difference()
    return all(abs(x * q - round(x * q)) > 1e-9 * q for q in range(1, 1001))


# --------------------------------------------------------------------------
# Zonen des Fensters
# --------------------------------------------------------------------------

def in_window(omega: float, tol: float = 0.0) -> bool:
    """Liegt omega im Resonanzfenster (Ränder eingeschlossen, optional mit Toleranz)?"""
    lo, hi = window_bounds()
    return lo - tol <= omega <= hi + tol


def window_position(omega: float) -> str:
    """'below' (< 1/phi), 'inside' oder 'above' (> Obergrenze)."""
    lo, hi = window_bounds()
    if omega < lo:
        return "below"
    if omega > hi:
        return "above"
    return "inside"


def economic_zone(omega: float) -> str:
    """'stagnation' unterhalb, 'flourishing' im Fenster, 'crisis' oberhalb (Theorem 3)."""
    return {"below": "stagnation", "inside": "flourishing", "above": "crisis"}[window_position(omega)]


def distance_to_window(omega: float) -> float:
    """Abstand zum Fenster (0 im Fenster)."""
    lo, hi = window_bounds()
    return max(lo - omega, 0.0, omega - hi)


def tolerance_to_bounds(omega: float) -> tuple[float, float]:
    """Spielraum (nach unten, nach oben) bis zu den Fenstergrenzen für ein System im Fenster."""
    if not in_window(omega):
        raise ValueError("omega liegt nicht im Fenster")
    lo, hi = window_bounds()
    return omega - lo, hi - omega


def absorbs_disturbance(omega: float, d_omega: float) -> bool:
    """Bleibt omega + d_omega im Fenster?"""
    return in_window(omega + d_omega)


# --------------------------------------------------------------------------
# Theorem 2: Stabilität und Produktivität im Fenster
# --------------------------------------------------------------------------

def freedom(omega: float) -> float:
    """F(Omega) = -ln(1 - Omega)."""
    return -math.log(1.0 - _check_omega(omega))


def freedom_range_in_window() -> tuple[float, float]:
    """F an den Fenstergrenzen: (0,9624, 1,5906)."""
    lo, hi = window_bounds()
    return freedom(lo), freedom(hi)


def productivity(omega: float, lam: float = 2.0) -> float:
    """P(Omega) = -ln(1 - Omega) - lam*(Omega - Omega_center)^2."""
    if not lam > 0.0:
        raise ValueError("lam muss positiv sein")
    return freedom(omega) - lam * (omega - window_center()) ** 2


def productivity_derivative(omega: float, lam: float = 2.0) -> float:
    """dP/dOmega = 1/(1 - Omega) - 2 lam (Omega - Omega_center)."""
    return 1.0 / (1.0 - _check_omega(omega)) - 2.0 * lam * (omega - window_center())


def productivity_second_derivative(omega: float, lam: float = 2.0) -> float:
    """d^2P/dOmega^2 = 1/(1 - Omega)^2 - 2 lam."""
    return 1.0 / (1.0 - _check_omega(omega)) ** 2 - 2.0 * lam


def productivity_convex_in_window(lam: float = 2.0, n_grid: int = 201) -> bool:
    """Ist d^2P/dOmega^2 > 0 auf dem ganzen Fenster (Gitterprüfung)?"""
    lo, hi = window_bounds()
    return all(productivity_second_derivative(float(o), lam) > 0.0 for o in np.linspace(lo, hi, n_grid))


def predictability(omega: float) -> float:
    """V(Omega) = 1 - |Omega - Omega_center|/0,5 (maximal 1 in der Mitte)."""
    return 1.0 - abs(omega - window_center()) / 0.5


def min_predictability_in_window() -> float:
    """Minimum von V im Fenster = 1 - Delta_h/0,5 = 0,8219 (an den Rändern, 'mindestens 0,82')."""
    lo, hi = window_bounds()
    return min(predictability(lo), predictability(hi))


def freedom_times_predictability(omega: float) -> float:
    """Produkt F*V, das im Innovationsmodell auftritt."""
    return freedom(omega) * predictability(omega)


def lyapunov_proxy(omega: float) -> float:
    """Proportionalitätsfaktor -1/(1 - Omega) < 0 aus der Skizze des Beweises (5)."""
    return -1.0 / (1.0 - _check_omega(omega))


# --------------------------------------------------------------------------
# Theorem 3 und 4: Wirtschaftliche Anwendungen, Universalität
# --------------------------------------------------------------------------

def innovation_rate(omega: float, alpha: float = 1.0, beta: float = 0.0, cost: float = 0.0) -> float:
    """dI/dt = alpha*F(Omega)*V(Omega) - beta*C(Omega); ``cost`` ist der Wert C(Omega)."""
    return alpha * freedom_times_predictability(omega) - beta * cost


def price_variance(omega: float, gamma: float = 1.0) -> float:
    """sigma_p^2 = gamma*Omega*(1 - Omega), maximal bei Omega = 0,5."""
    _check_omega(omega)
    return gamma * omega * (1.0 - omega)


def biodiversity(d: float, alpha: float = 1.0) -> float:
    """Biodiversität(d) = alpha*d*(1 - d) (Intermediate Disturbance Hypothesis), maximal bei d = 0,5."""
    if not 0.0 <= d <= 1.0:
        raise ValueError("d muss in [0, 1] liegen")
    return alpha * d * (1.0 - d)


def weighted_cov_index(series: Sequence[Sequence[float]], weights: Sequence[float],
                       ranges: Sequence[float]) -> float:
    """Omega_econ = sum_i w_i * CoV_i / Range_i, begrenzt auf [0, 1] (Messkonstruktion der Fallstudie Schweiz).

    CoV_i ist der Variationskoeffizient std/|mean| der i-ten Reihe, Range_i der historische Wertebereich
    der CoV, auf den normiert wird.
    """
    if not (len(series) == len(weights) == len(ranges)) or not series:
        raise ValueError("series, weights und ranges müssen gleich lang und nicht leer sein")
    total = 0.0
    for s, w, rng in zip(series, weights, ranges):
        arr = np.asarray(s, dtype=float)
        if arr.size < 2 or arr.mean() == 0.0 or not rng > 0.0:
            raise ValueError("ungültige Reihe oder Range")
        total += w * (arr.std(ddof=0) / abs(arr.mean())) / rng
    return float(min(max(total, 0.0), 1.0))


# --------------------------------------------------------------------------
# Korollar Korngesetz und Fallstudien
# --------------------------------------------------------------------------

CORN_LAW_OMEGAS = {"before": 0.72, "with_law": 0.55, "after_repeal": 0.70}

SWITZERLAND = [
    {"period": "1950-1970", "omega": 0.62, "growth": 3.8, "innovation": "Moderat", "hdi": 0.82},
    {"period": "1971-1990", "omega": 0.68, "growth": 2.1, "innovation": "Hoch", "hdi": 0.89},
    {"period": "1991-2007", "omega": 0.71, "growth": 1.6, "innovation": "Sehr Hoch", "hdi": 0.94},
    {"period": "2008-2015", "omega": 0.73, "growth": 1.2, "innovation": "Hoch", "hdi": 0.95},
    {"period": "2016-2023", "omega": 0.69, "growth": 1.1, "innovation": "Hoch", "hdi": 0.96},
]

ARGENTINA = [
    {"period": "1960-1974", "omega": 0.68, "crisis": False, "growth": 4.2},
    {"period": "1975-1983", "omega": 0.82, "crisis": True, "growth": -0.5},
    {"period": "1984-1988", "omega": 0.71, "crisis": False, "growth": 0.8},
    {"period": "1989-1990", "omega": 0.85, "crisis": True, "growth": -3.0},
    {"period": "1991-2000", "omega": 0.59, "crisis": False, "growth": 3.8},
    {"period": "2001-2002", "omega": 0.88, "crisis": True, "growth": -10.9},
    {"period": "2003-2007", "omega": 0.74, "crisis": False, "growth": 8.4},
    {"period": "2008-2023", "omega": 0.77, "crisis": False, "growth": 0.3},
]

USSR_OMEGA_RANGE = (0.35, 0.45)

ESTIMATES = {
    "Sowjetunion": (0.3, 0.4), "DDR": (0.25, 0.35), "Nordkorea": (0.1, 0.1),
    "Weimarer Hyperinflation 1923": (0.95, 0.95), "Argentinien 2001": (0.88, 0.88),
    "Zimbabwe 2008": (0.98, 0.98),
}


def corn_law_left_window() -> bool:
    """Mit Korngesetz verlässt Omega_grain das Fenster (nach unten); davor und danach liegt es im Fenster."""
    o = CORN_LAW_OMEGAS
    return in_window(o["before"]) and not in_window(o["with_law"]) and in_window(o["after_repeal"])


def fraction_in_window(omegas: Iterable[float]) -> float:
    """Anteil der Werte, die im Fenster liegen."""
    vals = list(omegas)
    if not vals:
        raise ValueError("omegas darf nicht leer sein")
    return sum(1 for o in vals if in_window(o)) / len(vals)


def excursions(omegas: Sequence[float]) -> list[tuple[int, str]]:
    """Indizes und Richtung ('below'/'above') aller Werte außerhalb des Fensters."""
    return [(i, window_position(o)) for i, o in enumerate(omegas) if window_position(o) != "inside"]


def crises_above_window(table: Sequence[dict]) -> bool:
    """Argentinien: Jede Periode mit Omega > Obergrenze ist eine Krisenperiode (Hypothese der Fallstudie)."""
    return all(row["crisis"] for row in table if window_position(row["omega"]) == "above")


def switzerland_in_window_fraction(tol: float = 0.0) -> float:
    """Anteil der Schweizer Perioden im Fenster (alle Zeilen, optional mit Toleranz)."""
    rows = [r["omega"] for r in SWITZERLAND]
    return sum(1 for o in rows if in_window(o, tol)) / len(rows)


def ussr_below_window() -> bool:
    """Die geschätzte sowjetische Spanne liegt vollständig unterhalb des Fensters."""
    return window_position(USSR_OMEGA_RANGE[1]) == "below"


def policy_target_window() -> tuple[float, float]:
    """Zielbereich der Wirtschaftspolitik (Korollar 2): das Resonanzfenster."""
    return window_bounds()
