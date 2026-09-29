"""Der ungerade Anteil der e-Reihe: Rechenmodul zum Kapitel
"Der ungerade Anteil der e-Reihe: Leibniz und der Verzicht auf die geraden Glieder".

Nur Standardbibliothek. Die Funktionen sind nach den Sätzen des Kapitels benannt.

Notation (wie im Kapitel)
    E(z)   = sum z^n / n!
    E_g(z) = sum z^(2k) / (2k)!        (gerader Anteil, = cosh)
    E_u(z) = sum z^(2k+1) / (2k+1)!    (ungerader Anteil, = sinh)
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Callable, Iterable, Sequence

DEFAULT_TERMS = 80  # genügt für |x| <= 20 bis auf Rundungsfehler


# --------------------------------------------------------------------------
# Definition 1: Reihen und Partialsummen
# --------------------------------------------------------------------------

def _check_terms(n_terms: int) -> None:
    if n_terms < 1:
        raise ValueError("n_terms muss >= 1 sein")


def e_series(z: complex | float, n_terms: int = DEFAULT_TERMS) -> complex | float:
    """Partialsumme der e-Reihe mit n_terms Gliedern."""
    _check_terms(n_terms)
    total = term = 1.0 * (z * 0 + 1)
    for n in range(1, n_terms):
        term = term * z / n
        total += term
    return total


def e_gerade(z: complex | float, n_terms: int = DEFAULT_TERMS) -> complex | float:
    """Gerader Anteil E_g(z) mit n_terms Gliedern (Potenzen z^0, z^2, ...)."""
    _check_terms(n_terms)
    z2 = z * z
    total = term = 1.0 * (z * 0 + 1)
    for k in range(1, n_terms):
        term = term * z2 / ((2 * k - 1) * (2 * k))
        total += term
    return total


def e_ungerade(z: complex | float, n_terms: int = DEFAULT_TERMS) -> complex | float:
    """Ungerader Anteil E_u(z) mit n_terms Gliedern (Potenzen z^1, z^3, ...)."""
    _check_terms(n_terms)
    z2 = z * z
    total = term = 1.0 * z
    for k in range(1, n_terms):
        term = term * z2 / ((2 * k) * (2 * k + 1))
        total += term
    return total


def odd_partial_sum(n: int) -> Fraction:
    """S_N = sum_{k=0}^{N} 1/(2k+1)!  (exakt, rational)."""
    if n < 0:
        raise ValueError("N muss >= 0 sein")
    return sum((Fraction(1, math.factorial(2 * k + 1)) for k in range(n + 1)), Fraction(0))


def even_partial_sum(n: int) -> Fraction:
    """C_N = sum_{k=0}^{N} 1/(2k)!  (exakt, rational)."""
    if n < 0:
        raise ValueError("N muss >= 0 sein")
    return sum((Fraction(1, math.factorial(2 * k)) for k in range(n + 1)), Fraction(0))


# --------------------------------------------------------------------------
# Lemma 2 / Bemerkung 1: Ableitung und Paritätswechsel
# --------------------------------------------------------------------------

def derivative_coefficients(coeffs: Sequence[Fraction]) -> list[Fraction]:
    """Gliedweise Ableitung einer Potenzreihe (Koeffizienten c_n von x^n)."""
    return [n * c for n, c in enumerate(coeffs)][1:] if len(coeffs) > 1 else [Fraction(0)]


def series_coefficients(parity: str, degree: int) -> list[Fraction]:
    """Taylorkoeffizienten von E ('all'), E_g ('even') oder E_u ('odd') bis x^degree."""
    if degree < 0:
        raise ValueError("degree muss >= 0 sein")
    keep = {
        "all": lambda n: True,
        "even": lambda n: n % 2 == 0,
        "odd": lambda n: n % 2 == 1,
    }
    if parity not in keep:
        raise ValueError("parity muss 'all', 'even' oder 'odd' sein")
    return [Fraction(1, math.factorial(n)) if keep[parity](n) else Fraction(0)
            for n in range(degree + 1)]


# --------------------------------------------------------------------------
# Satz 1 / Korollar 1: Rekonstruktion aus dem ungeraden Anteil
# --------------------------------------------------------------------------

def even_from_odd(u: float) -> float:
    """E_g = sqrt(1 + E_u^2)  (Gleichung 3)."""
    return math.sqrt(1.0 + u * u)


def exp_from_odd(x: float, n_terms: int = DEFAULT_TERMS) -> float:
    """e^x = E_u(x) + sqrt(1 + E_u(x)^2)  (Gleichung 4)."""
    u = e_ungerade(x, n_terms)
    return u + even_from_odd(u)


def inv_exp_from_odd(x: float, n_terms: int = DEFAULT_TERMS) -> float:
    """e^{-x} = sqrt(1 + E_u(x)^2) - E_u(x)."""
    u = e_ungerade(x, n_terms)
    return even_from_odd(u) - u


def e_from_odd(n: int) -> float:
    """e = g(S_N) mit g(s) = s + sqrt(1+s^2), aus N+1 ungeraden Gliedern."""
    return _g(float(odd_partial_sum(n)))


def _g(s: float) -> float:
    return s + math.sqrt(1.0 + s * s)


def even_coefficients_from_odd(k_max: int) -> list[Fraction]:
    """Koeffizienten von x^0, x^2, ..., x^(2 k_max) der Reihe sqrt(1 + E_u(x)^2),
    exakt berechnet ausschließlich aus den ungeraden Koeffizienten (Korollar 1).
    Ergebnis muss 1/(2k)! sein."""
    if k_max < 0:
        raise ValueError("k_max muss >= 0 sein")
    deg = 2 * k_max
    odd = series_coefficients("odd", deg)
    u = _poly_mul(odd, odd, deg)            # E_u^2
    result = [Fraction(0)] * (deg + 1)
    power = [Fraction(1)] + [Fraction(0)] * deg   # u^0
    binom = Fraction(1)                     # binom(1/2, 0)
    for m in range(0, k_max + 1):
        if m > 0:
            binom = binom * (Fraction(1, 2) - (m - 1)) / m
            power = _poly_mul(power, u, deg)
        result = [r + binom * p for r, p in zip(result, power)]
    return [result[2 * k] for k in range(k_max + 1)]


def _poly_mul(a: Sequence[Fraction], b: Sequence[Fraction], deg: int) -> list[Fraction]:
    out = [Fraction(0)] * (deg + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > deg:
                break
            out[i + j] += ai * bj
    return out


# --------------------------------------------------------------------------
# Korollar 2 / Lemma 3 / Satz 2 / Satz 3: Asymmetrie
# --------------------------------------------------------------------------

def exp_candidates_from_even(c: float) -> tuple[float, float]:
    """Aus c = E_g(1) folgen nur die zwei Kandidaten {e, 1/e}."""
    if c < 1.0:
        raise ValueError("c = E_g(x) ist stets >= 1")
    r = math.sqrt(c * c - 1.0)
    return c + r, c - r


def exp_from_even(c: float, sign: int) -> float:
    """Rekonstruktion aus dem geraden Anteil erfordert das Vorzeichen (+1 oder -1)."""
    if sign not in (1, -1):
        raise ValueError("sign muss +1 oder -1 sein")
    plus, minus = exp_candidates_from_even(c)
    return plus if sign == 1 else minus


def split_parity(f: Callable[[float], float]) -> tuple[Callable[[float], float],
                                                        Callable[[float], float]]:
    """Zerlegung f = f_g + f_u (Lemma 1c, Spiegelungslemma 3)."""
    def f_g(x: float) -> float:
        return 0.5 * (f(x) + f(-x))

    def f_u(x: float) -> float:
        return 0.5 * (f(x) - f(-x))

    return f_g, f_u


def arsinh_via_odd(y: float) -> float:
    """Umkehrfunktion von E_u:  ln(y + sqrt(1+y^2))  (Satz 2c)."""
    return math.log(y + math.sqrt(1.0 + y * y))


def inverse_odd(y: float, tol: float = 1e-13) -> float:
    """Löst E_u(x) = y durch Bisektion (unabhängig von arsinh_via_odd)."""
    lo, hi = -1.0, 1.0
    while e_ungerade(hi) < y:
        hi *= 2.0
    while e_ungerade(lo) > y:
        lo *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if e_ungerade(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _entropy(counts: Iterable[int]) -> float:
    counts = list(counts)
    n = sum(counts)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def conditional_sign_entropy(samples: Sequence[float],
                             feature: Callable[[float], float],
                             ndigits: int = 9) -> float:
    """Empirische bedingte Entropie H(sgn X | feature(X)) in Bit (Satz 3).
    Für symmetrische Stichproben: 1 Bit für E_g, 0 Bit für E_u."""
    if any(x == 0 for x in samples):
        raise ValueError("Stichprobe darf 0 nicht enthalten")
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for x in samples:
        groups[round(feature(x), ndigits)][x > 0] += 1
    n = len(samples)
    return sum((sum(c) / n) * _entropy(c) for c in groups.values())


# --------------------------------------------------------------------------
# Satz 4 / Korollar 3: Zeitrichtung und Stabilität
# --------------------------------------------------------------------------

def rc_decay_parts(t: float, tau: float) -> tuple[float, float, float]:
    """(E_g, E_u, e^{-t/tau}) mit e^{-t/tau} = E_g - E_u."""
    if tau <= 0:
        raise ValueError("tau muss > 0 sein")
    a = t / tau
    g, u = e_gerade(a), e_ungerade(a)
    return g, u, g - u


def rc_step_response(t: float, tau: float) -> float:
    """Sprungantwort 1 - e^{-t/tau} = 1 - E_g + E_u."""
    g, u, _ = rc_decay_parts(t, tau)
    return 1.0 - g + u


# --------------------------------------------------------------------------
# Satz 5: Fundamentallösung von y'' = y
# --------------------------------------------------------------------------

def general_solution(y0: float, v0: float, x: float) -> float:
    """y = y0 * E_g + v0 * E_u  (Satz 5)."""
    return y0 * e_gerade(x) + v0 * e_ungerade(x)


def rk4_second_order(y0: float, v0: float, x_end: float, steps: int = 2000) -> float:
    """Numerische Integration von y'' = y (unabhängiger Vergleichsweg)."""
    if steps < 1:
        raise ValueError("steps muss >= 1 sein")
    h = x_end / steps
    y, v = y0, v0
    for _ in range(steps):
        k1y, k1v = v, y
        k2y, k2v = v + 0.5 * h * k1v, y + 0.5 * h * k1y
        k3y, k3v = v + 0.5 * h * k2v, y + 0.5 * h * k2y
        k4y, k4v = v + h * k3v, y + h * k3y
        y += h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    return y


# --------------------------------------------------------------------------
# Satz 6: Gewichtsvergleich (Betrag und Energie)
# --------------------------------------------------------------------------

def magnitude_gap(x: float) -> float:
    """E_g(x) - |E_u(x)|; nach Satz 6(a) gleich e^{-|x|}."""
    return e_gerade(x) - abs(e_ungerade(x))


def odd_even_ratio(x: float) -> float:
    """|E_u|/E_g = tanh|x|."""
    return abs(e_ungerade(x)) / e_gerade(x)


def energy_norms(a: float) -> tuple[float, float, float, float]:
    """(||E_g||^2, ||E_u||^2, <E_g,E_u>, ||E||^2) auf [-a, a] nach Satz 6(b)."""
    if a <= 0:
        raise ValueError("a muss > 0 sein")
    s2a = e_ungerade(2 * a)
    return a + 0.5 * s2a, -a + 0.5 * s2a, 0.0, s2a


def odd_energy_share(a: float) -> float:
    """Energieanteil des ungeraden Anteils: 1/2 - a/E_u(2a)."""
    _, odd, _, total = energy_norms(a)
    return odd / total


# --------------------------------------------------------------------------
# Satz 7 / Beobachtung 2: Fehlerabschätzung
# --------------------------------------------------------------------------

def tail_bound(n: int) -> float:
    """Obere Schranke 16/15 / (2N+3)! für r_N = E_u(1) - S_N."""
    if n < 0:
        raise ValueError("N muss >= 0 sein")
    return (16.0 / 15.0) / math.factorial(2 * n + 3)


def error_bound(n: int) -> float:
    """Schranke (1+tanh 1) * 16/15 / (2N+3)! für e - g(S_N)."""
    return (1.0 + math.tanh(1.0)) * tail_bound(n)


def error_odd(n: int) -> float:
    """Tatsächlicher Fehler e - g(S_N)."""
    return math.e - e_from_odd(n)


def error_even(n: int) -> float:
    """Fehler der geraden Variante C_N + sqrt(C_N^2 - 1) (Vorzeichen +1 bekannt)."""
    return abs(math.e - exp_from_even(float(even_partial_sum(n)), +1))


def error_table(n_max: int) -> list[dict[str, float]]:
    """Tabelle wie in Beobachtung 2 für N = 0..n_max."""
    if n_max < 0:
        raise ValueError("n_max muss >= 0 sein")
    return [
        {
            "N": n,
            "S_N": float(odd_partial_sum(n)),
            "g(S_N)": e_from_odd(n),
            "error_odd": error_odd(n),
            "bound": error_bound(n),
            "error_even": error_even(n),
        }
        for n in range(n_max + 1)
    ]


# --------------------------------------------------------------------------
# Satz 8-11: Leibniz-Reihe, Logarithmusreihe bei z = i, Gleichgewicht
# --------------------------------------------------------------------------

def leibniz_partial(n: int) -> float:
    """L_N = sum_{k=0}^{N} (-1)^k / (2k+1)."""
    if n < 0:
        raise ValueError("N muss >= 0 sein")
    return sum((-1) ** k / (2 * k + 1) for k in range(n + 1))


def log_series_at_i(n_terms: int) -> complex:
    """Partialsumme von l(i) = sum_{n>=1} (-1)^(n+1) i^n / n (n_terms Glieder)."""
    _check_terms(n_terms)
    return sum(((-1) ** (n + 1)) * (1j ** n) / n for n in range(1, n_terms + 1))


def log_series_even_odd(n_terms: int) -> tuple[float, float]:
    """(Realteil aus geraden Indizes, Imaginärteil aus ungeraden Indizes) der
    Partialsumme von l(i); der Imaginärteil ist die Leibniz-Partialsumme."""
    _check_terms(n_terms)
    even = sum((-1) ** (n // 2 + 1) / n for n in range(2, n_terms + 1, 2))
    odd = sum((-1) ** ((n - 1) // 2) / n for n in range(1, n_terms + 1, 2))
    return even, odd


def averaged(partial: Callable[[int], float], n: int) -> float:
    """Mittel zweier aufeinanderfolgender Partialsummen (glättet Alternieren)."""
    return 0.5 * (partial(n) + partial(n + 1))


def imaginary_axis_parts(theta: float) -> tuple[complex, complex]:
    """(E_g(i theta), E_u(i theta)) = (cos theta, i sin theta)."""
    return e_gerade(1j * theta), e_ungerade(1j * theta)


def balance_angle(tol: float = 1e-14) -> float:
    """Kleinster positiver Winkel mit |E_g(i t)| = |E_u(i t)|; Bisektion auf (0, pi/2).
    Ergebnis: pi/4 = Wert der Leibniz-Reihe (Satz 11)."""
    def diff(t: float) -> float:
        g, u = imaginary_axis_parts(t)
        return abs(g) - abs(u)

    lo, hi = 0.0, math.pi / 2
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def polar_from_log_parts() -> tuple[float, complex]:
    """Korollar 5: (E(ln sqrt2), E(i pi/4)) = (sqrt 2, (1+i)/sqrt 2)."""
    modulus = e_series(0.5 * math.log(2.0))
    phase = e_series(1j * math.pi / 4)
    return modulus, phase
