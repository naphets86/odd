"""Acht Zentraldreiecke mit Spitze im Mittelpunkt: Rechenmodul zum Kapitel
"Acht Zentraldreiecke mit Spitze im Mittelpunkt" (lowp.tex, Label sec:ok).

Nur Standardbibliothek. Die Funktionen sind nach den Sätzen des Kapitels benannt;
der Standardwert ``n = 8`` ist die Achtelung (drei Teilungen), alle Funktionen
gelten aber für jedes reguläre n-Eck mit n >= 3, soweit der Satz das hergibt.

Notation (wie im Kapitel)
    K            Einheitskreis, Mittelpunkt M = 0
    omega        = exp(i*pi/4)          (allgemein exp(2*pi*i/n))
    P_k          = omega^k              (Ecken, Indizes modulo n)
    T_k          = conv{0, P_k, P_(k+1)}  (Zentraldreieck)
    A_n, B_n     = Flächen des ein- bzw. umbeschriebenen regulären n-Ecks
    t_n          = tan(pi/n)            (Halbbasis / Höhe)
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

EPS = 1e-12


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def _check_n(n: int) -> int:
    """Prüft, dass n eine ganze Zahl >= 3 ist (Voraussetzung n >= 3 der Sätze)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("n muss eine ganze Zahl >= 3 sein")
    return n


def _check_nonneg(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} muss eine ganze Zahl >= 0 sein")
    return value


def is_power_of_two(n: int) -> bool:
    """True genau dann, wenn n = 2^m mit m >= 0."""
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1 and (n & (n - 1)) == 0


# --------------------------------------------------------------------------
# Satz 1: Die dreifache Teilung
# --------------------------------------------------------------------------

def sector_angle(m: int) -> float:
    """Zentriwinkel alpha_m = 2*pi / 2^m nach m Teilungsrunden."""
    return 2.0 * math.pi / 2 ** _check_nonneg(m, "m")


def sector_count(m: int) -> int:
    """Anzahl 2^m der Sektoren nach m Runden (Satz 1)."""
    return 2 ** _check_nonneg(m, "m")


def rounds_for_parts(n: int) -> int:
    """Rundenzahl m = log2(n) für n = 2^m Teile (Satz 1)."""
    if not is_power_of_two(n):
        raise ValueError("n muss eine Zweierpotenz sein")
    return n.bit_length() - 1


def split_sector(phi0: float, alpha: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Teilung eines Sektors S(phi0, alpha) in S(phi0, alpha/2) und S(phi0+alpha/2, alpha/2)."""
    if not 0.0 < alpha <= 2.0 * math.pi + EPS:
        raise ValueError("alpha muss in (0, 2*pi] liegen")
    half = alpha / 2.0
    return (phi0, half), (phi0 + half, half)


def sectors_after(m: int) -> list[tuple[float, float]]:
    """Alle Sektoren (Anfangswinkel, Zentriwinkel) nach m Runden, durch Teilung aus S(0, 2*pi)."""
    sectors = [(0.0, 2.0 * math.pi)]
    for _ in range(_check_nonneg(m, "m")):
        nxt: list[tuple[float, float]] = []
        for phi0, alpha in sectors:
            nxt.extend(split_sector(phi0, alpha))
        sectors = nxt
    return sectors


# --------------------------------------------------------------------------
# Satz 2: Die Zentraldreiecke
# --------------------------------------------------------------------------

def omega(n: int = 8) -> complex:
    """Primitive n-te Einheitswurzel exp(2*pi*i/n); für n = 8 ist das exp(i*pi/4)."""
    return cmath.exp(2j * math.pi / _check_n(n))


def vertex(k: int, n: int = 8) -> complex:
    """Ecke P_k = omega^k (Indizes modulo n)."""
    return cmath.exp(2j * math.pi * (k % _check_n(n)) / n)


def vertices(n: int = 8) -> list[complex]:
    """Alle Ecken P_0, ..., P_(n-1)."""
    return [vertex(k, n) for k in range(_check_n(n))]


def apex_angle(n: int = 8) -> float:
    """Spitzenwinkel 2*pi/n (n = 8: pi/4)."""
    return 2.0 * math.pi / _check_n(n)


def base_angle(n: int = 8) -> float:
    """Basiswinkel (pi - 2*pi/n)/2 (n = 8: 3*pi/8)."""
    return (math.pi - apex_angle(n)) / 2.0


def base_length(n: int = 8) -> float:
    """Basis s = |P_(k+1) - P_k| = 2*sin(pi/n) (n = 8: sqrt(2 - sqrt 2))."""
    return 2.0 * math.sin(math.pi / _check_n(n))


def height(n: int = 8) -> float:
    """Höhe h = cos(pi/n) (n = 8: sqrt(2 + sqrt 2)/2)."""
    return math.cos(math.pi / _check_n(n))


def triangle_area(n: int = 8) -> float:
    """Fläche (1/2)*sin(2*pi/n) eines Zentraldreiecks (n = 8: sqrt(2)/4)."""
    return 0.5 * math.sin(2.0 * math.pi / _check_n(n))


def half_base_over_height(n: int = 8) -> float:
    """t_n = tan(pi/n) = Halbbasis / Höhe (n = 8: sqrt(2) - 1)."""
    return math.tan(math.pi / _check_n(n))


def foot_point(k: int, n: int = 8) -> complex:
    """Fußpunkt F_k = (P_k + P_(k+1))/2 = cos(pi/n)*exp(i*(2k+1)*pi/n)."""
    _check_n(n)
    return (vertex(k, n) + vertex(k + 1, n)) / 2.0


def foot_point_polar(k: int, n: int = 8) -> complex:
    """Geschlossene Darstellung cos(pi/n)*exp(i*(2k+1)*pi/n) des Fußpunkts."""
    _check_n(n)
    return math.cos(math.pi / n) * cmath.exp(1j * (2 * k + 1) * math.pi / n)


def triangle_corners(k: int, n: int = 8) -> tuple[complex, complex, complex]:
    """Ecken (0, P_k, P_(k+1)) des Zentraldreiecks T_k."""
    return 0j, vertex(k, n), vertex(k + 1, n)


def rotate_triangle(corners: Sequence[complex], n: int = 8) -> tuple[complex, ...]:
    """Bild eines Dreiecks unter der Drehung z -> omega*z (T_(k+1) = omega*T_k)."""
    w = omega(n)
    return tuple(w * c for c in corners)


def exact_values() -> dict[str, float]:
    """Exakte Werte sin, cos, tan von pi/8 in Wurzelform (Bemerkung 'Exakte Werte')."""
    r2 = math.sqrt(2.0)
    return {
        "sin_pi_8": 0.5 * math.sqrt(2.0 - r2),
        "cos_pi_8": 0.5 * math.sqrt(2.0 + r2),
        "tan_pi_8": r2 - 1.0,
    }


# --------------------------------------------------------------------------
# Lemma, Sätze 3-5: Überdeckung, Parkettierung, Eindeutigkeit
# --------------------------------------------------------------------------

def triangle_coordinates(z: complex, k: int, n: int = 8) -> tuple[float, float]:
    """Eindeutige Koordinaten (a, b) mit z = a*P_k + b*P_(k+1) (Lemma Sektorkegel)."""
    _check_n(n)
    pk, pk1 = vertex(k, n), vertex(k + 1, n)
    sin_alpha = math.sin(2.0 * math.pi / n)
    b = (z * pk.conjugate()).imag / sin_alpha
    a = (pk1.conjugate() * z).imag * -1.0 / sin_alpha
    return a, b


def in_cone(z: complex, k: int, n: int = 8, tol: float = EPS) -> bool:
    """z liegt im unbeschränkten Sektor Sigma_k genau dann, wenn a, b >= 0."""
    a, b = triangle_coordinates(z, k, n)
    return a >= -tol and b >= -tol


def in_triangle(z: complex, k: int, n: int = 8, tol: float = EPS) -> bool:
    """z liegt in T_k genau dann, wenn a, b >= 0 und a + b <= 1 (Lemma Sektorkegel)."""
    a, b = triangle_coordinates(z, k, n)
    return a >= -tol and b >= -tol and a + b <= 1.0 + tol


def locate(z: complex, n: int = 8, tol: float = EPS) -> list[int]:
    """Indizes aller Zentraldreiecke, die z enthalten (mehrere nur auf gemeinsamen Rändern)."""
    return [k for k in range(_check_n(n)) if in_triangle(z, k, n, tol)]


def shared_edge_vertices(k: int, j: int, n: int = 8) -> list[int]:
    """Gemeinsame Ecken zweier Dreiecke T_k, T_j (Indizes modulo n), ohne M.

    Benachbarte Dreiecke teilen die Strecke M-P_(k+1), nicht benachbarte nur M
    (Satz Parkettierung); hier die Indizes der gemeinsamen Ecken P_i.
    """
    _check_n(n)
    return sorted({k % n, (k + 1) % n} & {j % n, (j + 1) % n})


def octagon_area(n: int = 8) -> float:
    """Fläche des Parketts O = Vereinigung der T_k: n * triangle_area(n) (n = 8: 2*sqrt 2)."""
    return _check_n(n) * triangle_area(n)


def face_normal(k: int, n: int = 8) -> complex:
    """Einheitsnormale n_k = exp(i*(2k+1)*pi/n) der Seite zwischen P_k und P_(k+1)."""
    return cmath.exp(1j * (2 * k + 1) * math.pi / _check_n(n))


def in_half_plane(z: complex, k: int, n: int = 8, tol: float = EPS) -> bool:
    """Halbebene H_k = {Re(z * conj(n_k)) <= cos(pi/n)}."""
    return (z * face_normal(k, n).conjugate()).real <= math.cos(math.pi / n) + tol


def in_polygon(z: complex, n: int = 8, tol: float = EPS) -> bool:
    """z in O = Schnitt der Halbebenen H_k (Satz: O ist das reguläre n-Eck)."""
    return all(in_half_plane(z, k, n, tol) for k in range(_check_n(n)))


def in_circumscribed_polygon(z: complex, n: int = 8, tol: float = EPS) -> bool:
    """z im umbeschriebenen n-Eck O' = {Re(z * conj(omega^k)) <= 1 für alle k}."""
    w = omega(_check_n(n))
    return all((z * (w ** k).conjugate()).real <= 1.0 + tol for k in range(n))


def interior_angle(n: int = 8) -> float:
    """Innenwinkel des regulären n-Ecks = 2 * Basiswinkel (n = 8: 3*pi/4)."""
    return 2.0 * base_angle(n)


def central_angles(args: Sequence[float]) -> list[float]:
    """Winkelabstände alpha_j = phi_(j+1) - phi_j der streng wachsenden Argumente (zyklisch).

    Voraussetzung: mindestens 3 Punkte, Argumente in [0, 2*pi), streng wachsend.
    """
    phis = list(args)
    if len(phis) < 3:
        raise ValueError("mindestens drei Punkte erforderlich")
    if any(not 0.0 <= p < 2.0 * math.pi for p in phis):
        raise ValueError("Argumente müssen in [0, 2*pi) liegen")
    if any(b <= a for a, b in zip(phis, phis[1:])):
        raise ValueError("Argumente müssen streng wachsen")
    gaps = [b - a for a, b in zip(phis, phis[1:])]
    gaps.append(phis[0] + 2.0 * math.pi - phis[-1])
    return gaps


def triangles_congruent(args: Sequence[float], tol: float = 1e-9) -> bool:
    """Sind alle Dreiecke M-A_j-A_(j+1) kongruent? (gleiche Basis 2*sin(alpha_j/2))."""
    bases = [2.0 * math.sin(a / 2.0) for a in central_angles(args)]
    return all(abs(b - bases[0]) <= tol for b in bases)


def uniqueness_gap(args: Sequence[float], tol: float = 1e-9) -> float | None:
    """Satz Eindeutigkeit: Sind die Dreiecke kongruent, ist der Abstand 2*pi/n; sonst None."""
    if not triangles_congruent(args, tol):
        return None
    return 2.0 * math.pi / len(list(args))


def angle_sums(n: int = 8) -> dict[str, float]:
    """Winkelsummen (Korollar): Spitzenwinkel, Basiswinkel, Gesamtsumme aller Dreieckswinkel."""
    _check_n(n)
    return {
        "apex_total": n * apex_angle(n),          # 2*pi
        "base_total": 2 * n * base_angle(n),      # (n-2)*pi
        "all": n * math.pi,                       # n*pi
    }


# --------------------------------------------------------------------------
# Satz 6: Umfang, Apothem, Fläche
# --------------------------------------------------------------------------

def perimeter(n: int = 8) -> float:
    """Umfang U_n = n*s = 2*n*sin(pi/n) (n = 8: 8*sqrt(2 - sqrt 2))."""
    return _check_n(n) * base_length(n)


def apothem(n: int = 8) -> float:
    """Apothem a_n = h = cos(pi/n)."""
    return height(n)


def polygon_area(n: int = 8) -> float:
    """Fläche A_n = U_n * a_n / 2 (n = 8: 2*sqrt 2)."""
    return 0.5 * perimeter(n) * apothem(n)


def fill_ratio(n: int = 8) -> float:
    """Anteil A_n / pi der Kreisscheibe, den die Zentraldreiecke füllen (n = 8: 0,9003)."""
    return polygon_area(n) / math.pi


def sector_area(n: int = 8) -> float:
    """Fläche pi/n des Kreissektors über einer Dreiecksbasis."""
    return math.pi / _check_n(n)


def segment_area(n: int = 8) -> float:
    """Kreissegment über der Basis: pi/n - (1/2)*sin(2*pi/n) (n = 8: pi/8 - sqrt(2)/4)."""
    return sector_area(n) - triangle_area(n)


# --------------------------------------------------------------------------
# Sätze 7-8, Korollar: Einschließung von pi und Verfeinerung
# --------------------------------------------------------------------------

def inscribed_area(n: int) -> float:
    """A_n = (n/2)*sin(2*pi/n): Fläche des einbeschriebenen regulären n-Ecks."""
    return 0.5 * _check_n(n) * math.sin(2.0 * math.pi / n)


def circumscribed_area(n: int) -> float:
    """B_n = n*tan(pi/n): Fläche des umbeschriebenen regulären n-Ecks."""
    return _check_n(n) * math.tan(math.pi / n)


def pi_bounds(n: int = 8) -> tuple[float, float]:
    """Einschließung A_n < pi < B_n (n = 8: 2*sqrt 2 < pi < 8*(sqrt 2 - 1))."""
    return inscribed_area(n), circumscribed_area(n)


def doubled_areas(a_n: float, b_n: float) -> tuple[float, float]:
    """Verdopplung n -> 2n aus (A_n, B_n): A_2n = sqrt(A_n*B_n), B_2n = 2*A_2n*B_n/(A_2n+B_n)."""
    if a_n <= 0.0 or b_n <= 0.0:
        raise ValueError("Flächen müssen positiv sein")
    a2 = math.sqrt(a_n * b_n)
    b2 = 2.0 * a2 * b_n / (a2 + b_n)
    return a2, b2


def refine_bounds(m_start: int = 2, steps: int = 5) -> list[tuple[int, float, float]]:
    """Einschließungen für n = 2^(m_start), 2^(m_start+1), ... allein über die Mittelwertformeln.

    Start mit dem Quadrat (m_start = 2): A_4 = 2, B_4 = 4. Liefert (n, A_n, B_n).
    """
    if _check_nonneg(m_start, "m_start") < 2:
        raise ValueError("m_start muss >= 2 sein (n >= 4)")
    _check_nonneg(steps, "steps")
    n = 2 ** m_start
    a, b = inscribed_area(n), circumscribed_area(n)
    rows = [(n, a, b)]
    for _ in range(steps):
        a, b = doubled_areas(a, b)
        n *= 2
        rows.append((n, a, b))
    return rows


def asymptotic_inscribed(n: int, order: int = 2) -> float:
    """Entwicklung A_n = pi - 2*pi^3/(3 n^2) + 2*pi^5/(15 n^4) + O(n^-6); order in {0, 2, 4}."""
    if order not in (0, 2, 4):
        raise ValueError("order muss 0, 2 oder 4 sein")
    _check_n(n)
    value = math.pi
    if order >= 2:
        value -= 2.0 * math.pi ** 3 / (3.0 * n ** 2)
    if order >= 4:
        value += 2.0 * math.pi ** 5 / (15.0 * n ** 4)
    return value


def asymptotic_circumscribed(n: int, order: int = 2) -> float:
    """Entwicklung B_n = pi + pi^3/(3 n^2) + 2*pi^5/(15 n^4) + O(n^-6); order in {0, 2, 4}."""
    if order not in (0, 2, 4):
        raise ValueError("order muss 0, 2 oder 4 sein")
    _check_n(n)
    value = math.pi
    if order >= 2:
        value += math.pi ** 3 / (3.0 * n ** 2)
    if order >= 4:
        value += 2.0 * math.pi ** 5 / (15.0 * n ** 4)
    return value


def richardson_combination(n: int = 8) -> float:
    """R_n = (A_n + 2*B_n)/3 = pi + 2*pi^5/(15 n^4) + O(n^-6) (n = 8: 6*sqrt 2 - 16/3)."""
    return (inscribed_area(n) + 2.0 * circumscribed_area(n)) / 3.0


def richardson_closed_form_8() -> float:
    """Geschlossene Form 6*sqrt(2) - 16/3 von R_8."""
    return 6.0 * math.sqrt(2.0) - 16.0 / 3.0


def einschluss_table(ms: Iterable[int] = range(2, 9)) -> list[dict[str, float]]:
    """Tabelle der ein-/umbeschriebenen 2^m-Ecke samt Fehlern und Fehlerverhältnis."""
    rows = []
    for m in ms:
        n = 2 ** _check_nonneg(m, "m")
        a, b = inscribed_area(n), circumscribed_area(n)
        rows.append({
            "n": n, "A": a, "B": b,
            "err_A": math.pi - a, "err_B": b - math.pi,
            "ratio": (math.pi - a) / (b - math.pi),
        })
    return rows


# --------------------------------------------------------------------------
# Lemma Arkustangens, Satz Zentraldreiecksreihe, Korollar Quadrant/Oktant
# --------------------------------------------------------------------------

def arctan_partial(t: float, n_terms: int) -> float:
    """Partialsumme sum_{j<N} (-1)^j t^(2j+1)/(2j+1) der Arkustangensreihe, t in [0, 1]."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("t muss in [0, 1] liegen")
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError("n_terms muss eine ganze Zahl >= 1 sein")
    return math.fsum((-1) ** j * t ** (2 * j + 1) / (2 * j + 1) for j in range(n_terms))


def arctan_remainder_bound(t: float, n_terms: int) -> float:
    """Restgliedschranke t^(2N+1)/(2N+1) (Lemma Arkustangens)."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("t muss in [0, 1] liegen")
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError("n_terms muss eine ganze Zahl >= 1 sein")
    return t ** (2 * n_terms + 1) / (2 * n_terms + 1)


def central_triangle_series(n: int, n_terms: int) -> float:
    """Partialsumme S_N^(n) = n * sum_{j<N} (-1)^j t_n^(2j+1)/(2j+1) (Satz Zentraldreiecksreihe)."""
    _check_n(n)
    if n < 4:
        raise ValueError("Der Satz verlangt n >= 4 (t_n <= 1)")
    return n * arctan_partial(math.tan(math.pi / n), n_terms)


def central_triangle_error_bound(n: int, n_terms: int) -> float:
    """Schranke n * t_n^(2N+1)/(2N+1) für |pi - S_N^(n)|."""
    _check_n(n)
    if n < 4:
        raise ValueError("Der Satz verlangt n >= 4 (t_n <= 1)")
    return n * arctan_remainder_bound(math.tan(math.pi / n), n_terms)


def central_triangle_error_sign(n_terms: int) -> int:
    """Vorzeichen (-1)^N von pi - S_N^(n)."""
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError("n_terms muss eine ganze Zahl >= 1 sein")
    return -1 if n_terms % 2 else 1


def leibniz_partial(n_terms: int) -> float:
    """4 * sum_{j<N} (-1)^j/(2j+1): Leibniz-Reihe = Quadrant (n = 4, t_4 = 1)."""
    return central_triangle_series(4, n_terms)


def octant_partial(n_terms: int) -> float:
    """S_N^(8): Oktant-Reihe mit t_8 = sqrt(2) - 1 (Korollar (b))."""
    return central_triangle_series(8, n_terms)


def leibniz_octant_identity(n_terms: int) -> tuple[float, float]:
    """Beide Seiten der Leibniz-Achtel-Identität (Korollar (c)) nach N Gliedern.

    Links sum (-1)^j/(2j+1), rechts 2*sum (-1)^j (sqrt2-1)^(2j+1)/(2j+1); beide -> pi/4.
    """
    left = arctan_partial(1.0, n_terms)
    right = 2.0 * arctan_partial(math.sqrt(2.0) - 1.0, n_terms)
    return left, right


def double_angle_identity_holds(tol: float = 1e-12) -> bool:
    """Algebraische Probe tan(2x) = 2*tan(x)/(1 - tan(x)^2) = 1 für x = arctan(sqrt 2 - 1)."""
    t = math.sqrt(2.0) - 1.0
    return abs(2.0 * t / (1.0 - t * t) - 1.0) <= tol


# --------------------------------------------------------------------------
# Satz Halbierung des Zentraldreiecks, Beobachtung Gliederzahl
# --------------------------------------------------------------------------

def tau_exact(k: int) -> float:
    """tau_k = tan(pi / 2^(k+2)) direkt (tau_0 = 1, tau_1 = sqrt 2 - 1)."""
    if _check_nonneg(k, "k") == 0:
        return 1.0   # tan(pi/4) = 1 exakt (Gleitkomma liefert 0,9999999999999999)
    return math.tan(math.pi / 2 ** (k + 2))


def tau_next(tau: float) -> float:
    """Halbierungsrekursion tau_(k+1) = tau_k / (1 + sqrt(1 + tau_k^2))."""
    if tau < 0.0:
        raise ValueError("tau muss >= 0 sein")
    return tau / (1.0 + math.sqrt(1.0 + tau * tau))


def tau_sequence(k_max: int) -> list[float]:
    """tau_0, ..., tau_k_max über die Rekursion, beginnend bei tau_0 = 1."""
    seq = [1.0]
    for _ in range(_check_nonneg(k_max, "k_max")):
        seq.append(tau_next(seq[-1]))
    return seq


def tau_degree_bound(k: int) -> int:
    """Grad-Schranke 2^k: tau_k ist algebraisch vom Grad <= 2^k über Q."""
    return 2 ** _check_nonneg(k, "k")


def pi_from_tau(k: int, n_terms: int) -> float:
    """pi = 2^(k+2) * sum (-1)^j tau_k^(2j+1)/(2j+1), Partialsumme mit N Gliedern."""
    return 2 ** (_check_nonneg(k, "k") + 2) * arctan_partial(tau_exact(k), n_terms)


def terms_needed(k: int, eps: float) -> int:
    """Kleinstes N mit 2^(k+2) * tau_k^(2N+1)/(2N+1) < eps (garantierte Gliederzahl)."""
    if not eps > 0.0:
        raise ValueError("eps muss positiv sein")
    n_corners = 2 ** (_check_nonneg(k, "k") + 2)
    tau = tau_exact(k)

    def bound(n_terms: int) -> float:
        return n_corners * tau ** (2 * n_terms + 1) / (2 * n_terms + 1)

    hi = 1
    while bound(hi) >= eps:
        hi *= 2
    lo = hi // 2 if hi > 1 else 0   # bound(lo) >= eps (oder lo = 0), bound(hi) < eps
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bound(mid) < eps:
            hi = mid
        else:
            lo = mid
    return hi


def terms_needed_table(ks: Iterable[int] = range(0, 6),
                       eps_values: Sequence[float] = (1e-6, 1e-10)) -> list[dict[str, float]]:
    """Tabelle der Beobachtung 'Wie viele Glieder braucht die Teilung?'."""
    rows = []
    for k in ks:
        row: dict[str, float] = {
            "k": k,
            "corners": 2 ** (k + 2),
            "tau": tau_exact(k),
        }
        for eps in eps_values:
            row[f"N_{eps:g}"] = terms_needed(k, eps)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Satz Mercator-Reihe, Satz Achtel-Reihen
# --------------------------------------------------------------------------

def mercator_partial(theta: float, n_terms: int) -> complex:
    """Partialsumme sum_{n=1}^N (-1)^(n+1) exp(i*n*theta)/n."""
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError("n_terms muss eine ganze Zahl >= 1 sein")
    return sum((-1) ** (n + 1) * cmath.exp(1j * n * theta) / n for n in range(1, n_terms + 1))


def mercator_limit(theta: float) -> complex:
    """Grenzwert ln(2*cos(theta/2)) + i*theta/2 für theta in (-pi, pi)."""
    if not -math.pi < theta < math.pi:
        raise ValueError("theta muss in (-pi, pi) liegen")
    return complex(math.log(2.0 * math.cos(theta / 2.0)), theta / 2.0)


def eighth_sine_limit() -> float:
    """sum (-1)^(n+1) sin(n*pi/4)/n = pi/8."""
    return math.pi / 8.0


def eighth_cosine_limit() -> float:
    """sum (-1)^(n+1) cos(n*pi/4)/n = (1/2)*ln(2 + sqrt 2)."""
    return 0.5 * math.log(2.0 + math.sqrt(2.0))


def newton_signs(kind: int) -> tuple[int, int, int, int]:
    """Vorzeichenmuster der Achtel-Reihen, Periode 4 in j.

    kind 1: (+, +, -, -)  -> Vorzeichen von Im(omega^(2j+1)), Reihe mit Wert pi/(2*sqrt 2)
    kind 2: (+, -, -, +)  -> Vorzeichen von Re(omega^(2j+1)), Reihe mit Wert ln(1+sqrt 2)/sqrt 2
    """
    if kind == 1:
        return (1, 1, -1, -1)
    if kind == 2:
        return (1, -1, -1, 1)
    raise ValueError("kind muss 1 oder 2 sein")


def newton_series_partial(kind: int, n_terms: int) -> float:
    """Partialsumme sum_{j<N} eps_j/(2j+1) mit dem Vorzeichenmuster ``kind``."""
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError("n_terms muss eine ganze Zahl >= 1 sein")
    signs = newton_signs(kind)
    return math.fsum(signs[j % 4] / (2 * j + 1) for j in range(n_terms))


def newton_series_limit(kind: int) -> float:
    """Grenzwerte pi/(2*sqrt 2) (kind 1) bzw. ln(1 + sqrt 2)/sqrt 2 (kind 2)."""
    if kind == 1:
        return math.pi / (2.0 * math.sqrt(2.0))
    if kind == 2:
        return math.log(1.0 + math.sqrt(2.0)) / math.sqrt(2.0)
    raise ValueError("kind muss 1 oder 2 sein")


def newton_series_extrapolated(kind: int, n_terms: int) -> float:
    """Richardson-Extrapolation 2*S_(2N) - S_N der Partialsummen (N = n_terms Glieder).

    Die Partialsummen S_N der Achtel-Reihen haben einen Fehler c/N + O(1/N^2); die
    Kombination hebt den 1/N-Term auf und konvergiert deutlich schneller.
    """
    return 2.0 * newton_series_partial(kind, 2 * n_terms) - newton_series_partial(kind, n_terms)


# --------------------------------------------------------------------------
# Satz Sehnen und Summen der acht Ecken
# --------------------------------------------------------------------------

def chord(k: int, n: int = 8) -> float:
    """Sehne d_k = |P_0 - P_k| = 2*sin(k*pi/n) für k = 1, ..., n-1."""
    _check_n(n)
    if not 1 <= k <= n - 1:
        raise ValueError("k muss in 1..n-1 liegen")
    return abs(vertex(0, n) - vertex(k, n))


def chord_lengths(n: int = 8) -> list[float]:
    """d_1, ..., d_(n-1)."""
    return [chord(k, n) for k in range(1, _check_n(n))]


def chord_values_8() -> dict[int, float]:
    """Exakte Wurzelwerte der Sehnen des Achtecks (d_k = d_(8-k))."""
    r2 = math.sqrt(2.0)
    small, big = math.sqrt(2.0 - r2), math.sqrt(2.0 + r2)
    return {1: small, 2: r2, 3: big, 4: 2.0, 5: big, 6: r2, 7: small}


def vertex_sum(n: int = 8) -> complex:
    """sum P_k = 0."""
    return sum(vertices(n))


def chord_square_sum(n: int = 8) -> float:
    """sum_{k=1}^{n-1} d_k^2 = 2*n (n = 8: 16)."""
    return math.fsum(d * d for d in chord_lengths(n))


def chord_product(n: int = 8) -> float:
    """prod_{k=1}^{n-1} d_k = n (n = 8: 8)."""
    return math.prod(chord_lengths(n))


def sine_product(n: int = 8) -> float:
    """prod_{k=1}^{n-1} sin(k*pi/n) = n / 2^(n-1) (n = 8: 1/16)."""
    return math.prod(math.sin(k * math.pi / _check_n(n)) for k in range(1, n))


# --------------------------------------------------------------------------
# Satz Symmetriegruppe D_n
# --------------------------------------------------------------------------

def dihedral_group(n: int = 8) -> list[tuple[int, bool]]:
    """Elemente (m, spiegelung) der Diedergruppe D_n: z -> w^m z bzw. z -> w^m conj(z)."""
    _check_n(n)
    return [(m, refl) for refl in (False, True) for m in range(n)]


def apply_symmetry(g: tuple[int, bool], z: complex, n: int = 8) -> complex:
    """Wendet ein Gruppenelement (m, spiegelung) auf z an."""
    m, refl = g
    w = omega(n) ** m
    return w * (z.conjugate() if refl else z)


def group_order(n: int = 8) -> int:
    """|D_n| = 2n (n = 8: 16)."""
    return 2 * _check_n(n)


def maps_vertices_to_vertices(g: tuple[int, bool], n: int = 8, tol: float = 1e-9) -> bool:
    """Permutiert g die Eckenmenge {P_k}? (Bedingung für g(O) = O)."""
    verts = vertices(n)
    return all(any(abs(apply_symmetry(g, p, n) - v) <= tol for v in verts) for p in verts)


def triangle_image(g: tuple[int, bool], k: int, n: int = 8) -> int:
    """Index j mit g(T_k) = T_j für ein Element g der Diedergruppe D_n.

    Elemente von D_n bilden benachbarte Ecken auf benachbarte Ecken ab; das Bild von T_k ist
    daher wieder ein Zentraldreieck. Drehungen erhalten, Spiegelungen vertauschen die Reihenfolge
    der beiden Basisecken.
    """
    _check_n(n)
    i1 = _vertex_index(apply_symmetry(g, vertex(k, n), n), n)
    i2 = _vertex_index(apply_symmetry(g, vertex(k + 1, n), n), n)
    return i1 if i2 == (i1 + 1) % n else i2


def _vertex_index(z: complex, n: int) -> int:
    """Index k mit P_k = z (z muss eine Ecke sein)."""
    k = round(cmath.phase(z) / (2.0 * math.pi / n)) % n
    return int(k)


def orbit_of_triangle(k: int = 0, n: int = 8) -> set[int]:
    """Bahn von T_k unter D_n; transitiv, also alle n Dreiecke."""
    return {triangle_image(g, k, n) for g in dihedral_group(n)}


def stabilizer(k: int = 0, n: int = 8) -> list[tuple[int, bool]]:
    """Stabilisator von T_k in D_n (Ordnung 2: Identität und Spiegelung an der Höhe)."""
    return [g for g in dihedral_group(n) if triangle_image(g, k, n) == k % n]


def is_isometry_of_polygon(g: Callable[[complex], complex], n: int = 8, tol: float = 1e-9) -> bool:
    """Prüft an Stichproben, ob eine Abbildung O auf sich abbildet und Abstände erhält."""
    verts = vertices(n)
    for a in verts:
        for b in verts:
            if abs(abs(g(a) - g(b)) - abs(a - b)) > tol:
                return False
    return all(any(abs(g(a) - v) <= tol for v in verts) for a in verts)


# --------------------------------------------------------------------------
# Satz Viète
# --------------------------------------------------------------------------

def area_ratio(n: int) -> float:
    """A_n / A_2n = cos(pi/n) (Apothem als Flächenverhältnis)."""
    return inscribed_area(n) / inscribed_area(2 * _check_n(n))


def viete_factors(m: int) -> list[float]:
    """Faktoren cos(pi/2^k), k = 2..m, über die Radikalrekursion a_(j+1) = sqrt(2 + a_j)."""
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError("m muss eine ganze Zahl >= 2 sein")
    factors = []
    a = math.sqrt(2.0)          # 2*cos(pi/4)
    for _ in range(2, m + 1):
        factors.append(a / 2.0)
        a = math.sqrt(2.0 + a)
    return factors


def viete_product(m: int) -> float:
    """prod_{k=2}^m cos(pi/2^k)."""
    return math.prod(viete_factors(m))


def viete_area(m: int) -> float:
    """A_(2^(m+1)) = A_4 / prod_{k=2}^m cos(pi/2^k) mit A_4 = 2."""
    return 2.0 / viete_product(m)


def viete_pi(m: int) -> float:
    """Näherung 2 / prod_{k=2}^m cos(pi/2^k) -> pi."""
    return viete_area(m)


# --------------------------------------------------------------------------
# Zusammenfassung und Beweisstatus
# --------------------------------------------------------------------------

PROOF_STATUS: list[tuple[str, str]] = [
    ("Dreimalige Teilung: 8 Sektoren, Zentriwinkel pi/4", "bewiesen"),
    ("Acht kongruente gleichschenklige Dreiecke", "bewiesen"),
    ("Parkettierung des Achtecks, Fläche 2*sqrt 2; Eindeutigkeit", "bewiesen"),
    ("2*sqrt 2 < pi < 8*(sqrt 2 - 1); Halbierungsformeln", "bewiesen"),
    ("pi = n*sum(-1)^j t_n^(2j+1)/(2j+1) mit Fehlerschranke", "bewiesen"),
    ("Leibniz-Achtel-Identität pi/4 = 2*arctan(sqrt 2 - 1)", "bewiesen"),
    ("Achtel-Reihen", "bewiesen"),
    ("Sehnenprodukt, D_8-Symmetrie, Viète-Produkt", "bewiesen"),
    ("Größter Gewinn der Reihenbeschleunigung in der dritten Teilung", "numerisch belegt"),
    ("Leibniz habe aus diesem Grund bei drei Teilungen halt gemacht", "historische These, nicht beweisbar"),
]


def proof_status() -> list[tuple[str, str]]:
    """Beweisstatus der Aussagen des Kapitels (Tabelle 'Zusammenfassung und Beweisstatus')."""
    return list(PROOF_STATUS)
