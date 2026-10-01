"""Schaltkreise als Modelle für Wirtschaftsdynamiken: Rechenmodul zum Kapitel
"Schaltkreise als Modelle für Wirtschaftsdynamiken" (Unsicherheitsindex, Theoreme 1-6,
Designrichtlinien, Nachhaltigkeitssatz, Fallstudien) in lowp.tex.

Die Funktionen sind nach den Theoremen benannt. Wo eine Formel des Kapitels von dem abweicht,
was sich aus ihrer eigenen Herleitung ergibt, gibt es beide Varianten (``*_tex`` = Formel wie im
Kapitel abgedruckt, ``*_derived`` bzw. ``*_corners`` = aus der Herleitung bzw. exakt berechnet):

* delta_max:  Das Kapitel nennt (R0^2 C0 - 4 L0)/(R0^2 C0 + 4 L0); die Herleitung führt auf
  (1 - x)/(1 + x) mit x = 2*sqrt(L0/C0)/R0 (die abgedruckte Formel ist (1 - x^2)/(1 + x^2)).
* Die Kapitel-Bedingung R/L > 2*sqrt(1/(L C)) ist die Bedingung für den aperiodischen
  (überdämpften) Fall; Hurwitz-Stabilität selbst gilt für alle R, L, C > 0.
* Admittanz: Der Frequenzgang im Kapitel hat im Zähler ein zusätzliches 1/C; mit
  H(omega_0) = 1/R gemeint ist die Admittanz Y = i*omega/(-omega^2 L + i*omega R + 1/C).
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np

#: Ω_crit, das Theorem 1 nennt (maximaler Freiheitszuwachs); F'' hat im Inneren kein Maximum.
OMEGA_CRIT_CLAIMED = 0.5
#: Stabiler Bereich der Systemunsicherheit (Nachhaltigkeitssatz, Bedingung 5).
OMEGA_VIABLE = (0.25, 0.75)
#: Praktisch sicherer Bereich der Unsicherheitstoleranz delta_max.
DELTA_MAX_PRACTICAL = (0.15, 0.25)
#: Schwelle für Wartung bei Unsicherheits-Monitoring.
MAINTENANCE_OMEGA = 0.6


def _positive(value: float, name: str) -> float:
    if not value > 0.0:
        raise ValueError(f"{name} muss positiv sein")
    return float(value)


def _check_delta(delta: float) -> float:
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta muss in [0, 1) liegen")
    return float(delta)


def _check_omega(omega: float) -> float:
    if not 0.0 <= omega < 1.0:
        raise ValueError("omega muss in [0, 1) liegen")
    return float(omega)


def _check_count(n: int, name: str = "n", minimum: int = 1) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
        raise ValueError(f"{name} muss eine ganze Zahl >= {minimum} sein")
    return n


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# --------------------------------------------------------------------------
# Definition: Unsicherheitsindex Omega_circuit
# --------------------------------------------------------------------------

def omega_circuit(dv_dt: Iterable[float]) -> np.ndarray:
    """Omega(t) = 1 - |V'(t)| / max_s |V'(s)| für eine Folge von Änderungsraten V'."""
    d = np.abs(np.asarray(list(dv_dt), dtype=float))
    if d.size == 0:
        raise ValueError("dv_dt darf nicht leer sein")
    peak = d.max()
    if peak == 0.0:
        raise ValueError("maximale Änderungsrate ist null")
    return 1.0 - d / peak


def omega_circuit_from_signal(v: Sequence[float], dt: float) -> np.ndarray:
    """Omega(t) aus einer abgetasteten Ausgangsspannung (Differenz über ``numpy.gradient``)."""
    _positive(dt, "dt")
    arr = np.asarray(v, dtype=float)
    if arr.size < 2:
        raise ValueError("mindestens zwei Abtastwerte erforderlich")
    return omega_circuit(np.gradient(arr, dt))


def normalized_entropy(samples: Sequence[float], bins: int = 16) -> float:
    """Shannon-Entropie der Histogrammverteilung, normiert auf [0, 1] (Alternativdefinition)."""
    _check_count(bins, "bins", 2)
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("samples darf nicht leer sein")
    counts, _ = np.histogram(arr, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(bins))


# --------------------------------------------------------------------------
# Theorem 1: Freiheitsgrad
# --------------------------------------------------------------------------

def freedom(omega: float) -> float:
    """F(Omega) = -ln(1 - Omega), Omega in [0, 1)."""
    return -math.log(1.0 - _check_omega(omega))


def freedom_derivative(omega: float) -> float:
    """dF/dOmega = 1/(1 - Omega) > 0."""
    return 1.0 / (1.0 - _check_omega(omega))


def freedom_second_derivative(omega: float) -> float:
    """d^2F/dOmega^2 = 1/(1 - Omega)^2 > 0 (strikte Konvexität)."""
    return 1.0 / (1.0 - _check_omega(omega)) ** 2


def freedom_growth_peak(grid: Sequence[float]) -> float:
    """Argmax von F'' auf einem Gitter: liegt stets am oberen Rand, nicht bei 0,5 (siehe Theorem 1, Punkt 3)."""
    vals = [freedom_second_derivative(o) for o in grid]
    return float(grid[int(np.argmax(vals))])


def deterministic_has_freedom() -> bool:
    """Ein deterministischer Schaltkreis (Omega = 0) hat F(0) = 0, also keinen Freiheitsgrad."""
    return freedom(0.0) > 0.0


# --------------------------------------------------------------------------
# Theorem 2: Stabilität von RLC-Schaltkreisen unter Unsicherheit
# --------------------------------------------------------------------------

def perturbed_value(nominal: float, delta: float) -> float:
    """X(t) = X_0 (1 + Delta)."""
    return nominal * (1.0 + delta)


def nominal_condition(r0: float, l0: float, c0: float) -> bool:
    """Nominale Bedingung des Kapitels: R0/L0 > 2*sqrt(1/(L0 C0)) (aperiodischer Fall)."""
    _positive(r0, "R0"), _positive(l0, "L0"), _positive(c0, "C0")
    return r0 / l0 > 2.0 * math.sqrt(1.0 / (l0 * c0))


def critical_resistance(l: float, c: float) -> float:
    """R_crit = 2*sqrt(L/C)."""
    return 2.0 * math.sqrt(_positive(l, "L") / _positive(c, "C"))


def damping_ratio(r: float, l: float, c: float) -> float:
    """Dämpfungsgrad zeta = R/(2 sqrt(L/C)); zeta > 1 entspricht der Bedingung R > R_crit."""
    return _positive(r, "R") / critical_resistance(l, c)


def hurwitz_stable(r: float, l: float, c: float) -> bool:
    """Alle Pole von s^2 + (R/L) s + 1/(L C) = 0 haben negativen Realteil (gilt für R, L, C > 0)."""
    _positive(r, "R"), _positive(l, "L"), _positive(c, "C")
    poles = np.roots([1.0, r / l, 1.0 / (l * c)])
    return bool(np.all(poles.real < 0.0))


def delta_max_tex(r0: float, l0: float, c0: float) -> float:
    """delta_max = (R0^2 C0 - 4 L0)/(R0^2 C0 + 4 L0) wie im Kapitel abgedruckt."""
    a = _positive(r0, "R0") ** 2 * _positive(c0, "C0")
    b = 4.0 * _positive(l0, "L0")
    return (a - b) / (a + b)


def delta_max_derived(r0: float, l0: float, c0: float) -> float:
    """Aus der Herleitung des Kapitels: (1 - x)/(1 + x) mit x = 2*sqrt(L0/C0)/R0."""
    x = critical_resistance(l0, c0) / _positive(r0, "R0")
    return (1.0 - x) / (1.0 + x)


def tex_worst_case_condition(r0: float, l0: float, c0: float, delta: float) -> bool:
    """Worst-Case-Ungleichung der Herleitung: R0 (1-delta)/(L0 (1+delta)) > 2/sqrt(L0 C0)."""
    _check_delta(delta)
    return r0 * (1.0 - delta) / (l0 * (1.0 + delta)) > 2.0 / math.sqrt(l0 * c0)


def overdamped_perturbed(r0: float, l0: float, c0: float, d_r: float, d_l: float, d_c: float) -> bool:
    """Gilt R > 2*sqrt(L/C) für die gestörten Werte R0(1+d_r), L0(1+d_l), C0(1+d_c)?"""
    r, l, c = perturbed_value(r0, d_r), perturbed_value(l0, d_l), perturbed_value(c0, d_c)
    return r > critical_resistance(l, c)


def corner_overdamped(r0: float, l0: float, c0: float, delta: float) -> bool:
    """Gilt die Bedingung in allen 8 Ecken des Toleranzwürfels [-delta, delta]^3?"""
    _check_delta(delta)
    return all(overdamped_perturbed(r0, l0, c0, dr, dl, dc)
               for dr, dl, dc in itertools.product((-delta, delta), repeat=3))


def delta_max_corners(r0: float, l0: float, c0: float, tol: float = 1e-12) -> float:
    """Größtes delta in [0, 1), für das die Bedingung in allen Ecken gilt (Bisektion); 0 falls nominal verletzt."""
    if not corner_overdamped(r0, l0, c0, 0.0):
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-15
    if corner_overdamped(r0, l0, c0, hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if corner_overdamped(r0, l0, c0, mid):
            lo = mid
        else:
            hi = mid
    return lo


def resonance_ratio(d_l: float, d_c: float) -> float:
    """omega_perturbed/omega_nominal = 1/sqrt((1+d_L)(1+d_C))."""
    return 1.0 / math.sqrt((1.0 + d_l) * (1.0 + d_c))


def resonance_shift_extremes(delta: float) -> tuple[float, float]:
    """Kleinste und größte relative Verschiebung (1/(1+delta) - 1, 1/(1-delta) - 1)."""
    _check_delta(delta)
    return 1.0 / (1.0 + delta) - 1.0, 1.0 / (1.0 - delta) - 1.0


def resonance_shift_ok_tex(delta: float, limit: float = 0.5) -> bool:
    """Bedingung des Kapitels: Verschiebung delta < 50 %."""
    return _check_delta(delta) < limit


def resonance_shift_ok(delta: float, limit: float = 0.5) -> bool:
    """Exakte Variante: der größte Betrag der Verschiebung liegt unter ``limit``."""
    lo, hi = resonance_shift_extremes(delta)
    return max(abs(lo), abs(hi)) < limit


def robust_stability_tex(r0: float, l0: float, c0: float, delta: float) -> bool:
    """Alle drei Bedingungen des Theorems 2 (Kapitel-Formeln): nominal, delta < delta_max, Verschiebung < 50 %."""
    return (nominal_condition(r0, l0, c0)
            and delta < delta_max_tex(r0, l0, c0)
            and resonance_shift_ok_tex(delta))


# --------------------------------------------------------------------------
# Theorem 3: Optimale Unsicherheit (Pareto)
# --------------------------------------------------------------------------

def predictability(omega: float, target: float = 0.5) -> float:
    """P(Omega) = -(Omega - Omega*)^2."""
    return -((omega - target) ** 2)


def pareto_objective(omega: float, alpha: float = 0.5, target: float = 0.5) -> float:
    """alpha*F(Omega) + (1 - alpha)*P(Omega)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha muss in (0, 1) liegen")
    return alpha * freedom(omega) + (1.0 - alpha) * predictability(omega, target)


def pareto_derivative(omega: float, alpha: float = 0.5, target: float = 0.5) -> float:
    """d/dOmega der Zielfunktion: alpha/(1 - Omega) - 2 (1 - alpha)(Omega - Omega*)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha muss in (0, 1) liegen")
    return alpha / (1.0 - _check_omega(omega)) - 2.0 * (1.0 - alpha) * (omega - target)


def stationarity_discriminant(alpha: float = 0.5, target: float = 0.5) -> float:
    """Diskriminante der Stationaritätsgleichung -2(1-a)x^2 + 2(1-a)(1+Omega*)x - 2(1-a)Omega* - a = 0.

    Für alpha = 0,5 und Omega* = 0,5 ist sie -1,75 < 0: es gibt keinen stationären Punkt
    (das "Problem" im Beweis von Theorem 3).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha muss in (0, 1) liegen")
    a = -2.0 * (1.0 - alpha)
    b = 2.0 * (1.0 - alpha) * (1.0 + target)
    c = -2.0 * (1.0 - alpha) * target - alpha
    return b * b - 4.0 * a * c


def stationary_points(alpha: float = 0.5, target: float = 0.5) -> list[float]:
    """Reelle Nullstellen von dJ/dOmega im Intervall [0, 1)."""
    disc = stationarity_discriminant(alpha, target)
    if disc < 0.0:
        return []
    a = -2.0 * (1.0 - alpha)
    b = 2.0 * (1.0 - alpha) * (1.0 + target)
    roots = sorted({(-b + s * math.sqrt(disc)) / (2.0 * a) for s in (-1.0, 1.0)})
    return [r for r in roots if 0.0 <= r < 1.0]


def golden_omega() -> float:
    """Omega* = 1/phi = (sqrt 5 - 1)/2 = 0,618034 (Normierung des Kapitels)."""
    return (math.sqrt(5.0) - 1.0) / 2.0


def golden_freedom() -> float:
    """F(1/phi) = -ln(1 - 1/phi) = 2 ln(phi) = 0,9624."""
    return freedom(golden_omega())


def omega_zone(omega: float) -> str:
    """'stagnant' (Omega < 0,25), 'chaotic' (Omega > 0,75), sonst 'viable'."""
    _check_omega(omega)
    if omega < OMEGA_VIABLE[0]:
        return "stagnant"
    if omega > OMEGA_VIABLE[1]:
        return "chaotic"
    return "viable"


# --------------------------------------------------------------------------
# Theorem 4: Deterministische vs. stochastische Stabilität
# --------------------------------------------------------------------------

def deterministic_margin(r: float, l: float, c: float) -> float:
    """R - R_crit: scharfe Entscheidung stabil (> 0) / instabil (<= 0) im deterministischen Fall."""
    return r - critical_resistance(l, c)


def stable_fraction(r0: float, l0: float, c0: float, delta: float, n_samples: int = 20000,
                    rng: np.random.Generator | int | None = 0) -> float:
    """Anteil der gleichverteilt gestörten Parameterkombinationen mit R > 2 sqrt(L/C) (Monte-Carlo)."""
    _check_delta(delta)
    _check_count(n_samples, "n_samples")
    gen = _rng(rng)
    d = gen.uniform(-delta, delta, size=(n_samples, 3))
    r = r0 * (1.0 + d[:, 0])
    l = l0 * (1.0 + d[:, 1])
    c = c0 * (1.0 + d[:, 2])
    return float(np.mean(r > 2.0 * np.sqrt(l / c)))


def stability_region_width(r0: float, l0: float, c0: float, delta: float) -> float:
    """Breite des Toleranzintervalls für R, das im Parameterbereich stabil bleibt: Länge von [R0(1-d), R0(1+d)] ∩ (R_crit, ∞)."""
    _check_delta(delta)
    lo, hi = r0 * (1.0 - delta), r0 * (1.0 + delta)
    rc = critical_resistance(l0, c0)
    return max(0.0, hi - max(lo, rc))


# --------------------------------------------------------------------------
# Theorem 5: Leibniz-Struktur unter Unsicherheit
# --------------------------------------------------------------------------

def perturbed_cascade(l1: float, c1: float, r1: float, n_stages: int, delta: float,
                      rng: np.random.Generator | int | None = 0) -> dict[str, np.ndarray]:
    """L_n = L_1/(2n+1)^2 (1+Δ_L), C_n = C_1 (1+Δ_C), R_n = R_1 (1+Δ_R) mit unabhängigen |Δ| <= delta."""
    _check_delta(delta)
    _check_count(n_stages, "n_stages")
    gen = _rng(rng)
    n = np.arange(n_stages)
    d = gen.uniform(-delta, delta, size=(3, n_stages))
    return {
        "L": l1 / (2 * n + 1) ** 2 * (1.0 + d[0]),
        "C": c1 * (1.0 + d[1]),
        "R": r1 * (1.0 + d[2]),
    }


def poles(l: float, c: float, r: float) -> tuple[complex, complex]:
    """Pole s = -R/(2L) +- sqrt((R/(2L))^2 - 1/(L C)) eines RLC-Gliedes."""
    a = _positive(r, "R") / (2.0 * _positive(l, "L"))
    root = np.lib.scimath.sqrt(a * a - 1.0 / (l * _positive(c, "C")))
    return complex(-a + root), complex(-a - root)


def resonance_deviation_bound(delta: float) -> float:
    """Obere Schranke delta/(1 - delta) für die relative Abweichung von omega_n von (2n+1) omega_0."""
    return _check_delta(delta) / (1.0 - delta)


def resonance_deviation(l1: float, c1: float, n_stages: int, delta: float,
                        rng: np.random.Generator | int | None = 0) -> np.ndarray:
    """Relative Abweichungen omega_n/((2n+1) omega_0) - 1 einer zufällig gestörten Kaskade."""
    cas = perturbed_cascade(l1, c1, 1.0, n_stages, delta, rng)
    w0 = 1.0 / math.sqrt(l1 * c1)
    wn = 1.0 / np.sqrt(cas["L"] * cas["C"])
    return wn / ((2 * np.arange(n_stages) + 1) * w0) - 1.0


def leibniz_regime(delta: float) -> str:
    """'robust' (delta < 0,15), 'degraded' (0,15 <= delta < 0,35), 'lost' (delta >= 0,35)."""
    _check_delta(delta)
    if delta < 0.15:
        return "robust"
    if delta < 0.35:
        return "degraded"
    return "lost"


def stage_output_amplitude(n: int, l1: float, c1: float, r1: float, v0: float,
                           d_l: float = 0.0, d_c: float = 0.0, d_r: float = 0.0) -> float:
    """Ausgangsamplitude der Stufe n beim Rechteck-Harmonischen (2n+1) omega_0: |Y_n| * 4 V0/(pi (2n+1))."""
    _check_count(n, "n", 0)
    k = 2 * n + 1
    w0 = 1.0 / math.sqrt(l1 * c1)
    y = rlc_admittance(k * w0, l1 / k ** 2 * (1.0 + d_l), r1 * (1.0 + d_r), c1 * (1.0 + d_c))
    return abs(y) * 4.0 * v0 / (math.pi * k)


def leibniz_weight_error(n_stages: int, l1: float, c1: float, r1: float, delta: float,
                         rng: np.random.Generator | int | None = 0) -> float:
    """Größte relative Abweichung der Stufenamplituden von der Leibniz-Gewichtung 1/(2n+1)."""
    gen = _rng(rng)
    d = gen.uniform(-delta, delta, size=(3, n_stages)) if delta > 0 else np.zeros((3, n_stages))
    errs = []
    for n in range(n_stages):
        nominal = stage_output_amplitude(n, l1, c1, r1, 1.0)
        actual = stage_output_amplitude(n, l1, c1, r1, 1.0, d[0, n], d[1, n], d[2, n])
        errs.append(abs(actual / nominal - 1.0))
    return float(max(errs))


# --------------------------------------------------------------------------
# Theorem 6: Frequenzgang unter Unsicherheit
# --------------------------------------------------------------------------

def rlc_admittance(omega: float, l: float, r: float, c: float) -> complex:
    """Y(i omega) = i omega/(-omega^2 L + i omega R + 1/C); Y(omega_0) = 1/R."""
    _positive(l, "L"), _positive(r, "R"), _positive(c, "C")
    return 1j * omega / (-omega ** 2 * l + 1j * omega * r + 1.0 / c)


def frequency_response_tex(omega: float, l: float, r: float, c: float) -> complex:
    """Formel des Kapitels mit Zähler i omega/C (trägt gegenüber der Admittanz den Faktor 1/C)."""
    return rlc_admittance(omega, l, r, c) / c


def resonance_omega(l: float, c: float) -> float:
    """omega_0 = 1/sqrt(L C)."""
    return 1.0 / math.sqrt(_positive(l, "L") * _positive(c, "C"))


def gain_at_resonance(r: float) -> float:
    """|Y(omega_0)| = 1/R."""
    return 1.0 / _positive(r, "R")


def gain_deviation(l0: float, r0: float, c0: float, d_l: float, d_r: float, d_c: float) -> float:
    """|Y_gestört(omega_0)| - |Y_nominal(omega_0)| bei der nominalen Resonanzfrequenz omega_0."""
    w0 = resonance_omega(l0, c0)
    pert = abs(rlc_admittance(w0, l0 * (1.0 + d_l), r0 * (1.0 + d_r), c0 * (1.0 + d_c)))
    return pert - gain_at_resonance(r0)


def phase_deviation(l0: float, r0: float, c0: float, d_l: float, d_r: float, d_c: float) -> float:
    """Phase von Y_gestört bei omega_0 (die nominale Phase ist 0)."""
    w0 = resonance_omega(l0, c0)
    y = rlc_admittance(w0, l0 * (1.0 + d_l), r0 * (1.0 + d_r), c0 * (1.0 + d_c))
    return float(np.angle(y))


def corner_gain_deviations(l0: float, r0: float, c0: float, delta: float) -> list[float]:
    """Gain-Abweichungen in den 8 Ecken des Toleranzwürfels."""
    _check_delta(delta)
    return [gain_deviation(l0, r0, c0, dl, dr, dc)
            for dl, dr, dc in itertools.product((-delta, delta), repeat=3)]


def worst_case_gain_deviation(l0: float, r0: float, c0: float, delta: float) -> float:
    """Größter Betrag der Gain-Abweichung über die Ecken (relativ zum Nominalwert 1/R0 in ``relative_*``)."""
    return max(abs(x) for x in corner_gain_deviations(l0, r0, c0, delta))


def worst_case_phase_deviation(l0: float, r0: float, c0: float, delta: float) -> float:
    """Größter Betrag der Phasenabweichung über die Ecken (Bogenmaß)."""
    _check_delta(delta)
    return max(abs(phase_deviation(l0, r0, c0, dl, dr, dc))
               for dl, dr, dc in itertools.product((-delta, delta), repeat=3))


def mean_gain_deviation(l0: float, r0: float, c0: float, delta: float, n_samples: int = 5000,
                        rng: np.random.Generator | int | None = 0) -> float:
    """Mittlerer Betrag der Gain-Abweichung bei gleichverteilter Störung (Monte-Carlo)."""
    _check_delta(delta)
    _check_count(n_samples, "n_samples")
    gen = _rng(rng)
    d = gen.uniform(-delta, delta, size=(n_samples, 3))
    return float(np.mean([abs(gain_deviation(l0, r0, c0, a, b, c)) for a, b, c in d]))


def gain_deviation_model(delta: float, kappa: float = 1.0) -> float:
    """Modell des Kapitels: kappa*delta (delta < 0,3), kappa*(delta^2 + delta) (0,3 <= delta < 0,7)."""
    if delta < 0.0 or delta >= 0.7:
        raise ValueError("delta muss in [0, 0,7) liegen")
    return kappa * delta if delta < 0.3 else kappa * (delta * delta + delta)


def is_gain_robust(l0: float, r0: float, c0: float, delta: float, spec: float = 0.10) -> bool:
    """Robust, wenn die größte relative Gain-Abweichung unter ``spec`` (10 %) bleibt."""
    return worst_case_gain_deviation(l0, r0, c0, delta) * r0 < spec


# --------------------------------------------------------------------------
# Anwendungen, Designrichtlinien, Nachhaltigkeit
# --------------------------------------------------------------------------

CORN_LAW_PARAMETERS = {"L0": 3.0, "R0": 0.2, "C0": 2.0}
CORN_LAW_OMEGAS = {"L": 0.4, "R": 0.3, "C": 0.5}


def system_uncertainty(omegas: Iterable[float]) -> float:
    """Gesamtunsicherheit als Mittelwert der Komponentenunsicherheiten."""
    vals = list(omegas)
    if not vals:
        raise ValueError("omegas darf nicht leer sein")
    return math.fsum(vals) / len(vals)


def corn_law_delta_max() -> float:
    """Kapitel-Formel mit L0 = 3, R0 = 0,2, C0 = 2: -0,98675 (im Kapitel steht gerundet -0,994)."""
    cp = CORN_LAW_PARAMETERS
    return delta_max_tex(cp["R0"], cp["L0"], cp["C0"])


def nonlinear_delta_max(delta_linear: float) -> float:
    """Durch die Preisgrenze halbiert sich die tolerierbare Unsicherheit (Kapitel)."""
    return delta_linear / 2.0


def circuit_spec() -> dict[str, float]:
    """Nominale Werte des Schaltkreis-Simulators (L = 1 mH, C = 1 uF, R = 1 kOhm).

    Daraus folgen omega_0 = 31623 rad/s und f_0 = 5033 Hz (im Kapitel stehen 1000/(2 pi) = 159 Hz,
    das gilt für L = 1 H bei C = 1 uF).
    """
    l1, c, r = 1e-3, 1e-6, 1e3
    w0 = resonance_omega(l1, c)
    return {"L1": l1, "C": c, "R": r, "omega0": w0, "f0": w0 / (2.0 * math.pi), "v_amp": 5.0}


UNCERTAINTY_SCENARIOS: list[dict[str, float | str]] = [
    {"delta": 0.0, "omega_circuit": 0.0, "label": "ideal"},
    {"delta": 0.1, "omega_circuit": 0.35, "label": "realistisch"},
    {"delta": 0.2, "omega_circuit": 0.50, "label": "herausfordernd"},
    {"delta": 0.3, "omega_circuit": 0.62, "label": "kritisch"},
    {"delta": 0.4, "omega_circuit": 0.70, "label": "unstabil"},
]


def scenario_label(delta: float) -> str:
    """Bezeichnung des größten Szenarios, dessen delta nicht überschritten wird."""
    _check_delta(delta)
    label = str(UNCERTAINTY_SCENARIOS[0]["label"])
    for sc in UNCERTAINTY_SCENARIOS:
        if delta >= float(sc["delta"]):
            label = str(sc["label"])
    return label


def component_tolerance(delta_max: float) -> float:
    """Komponententoleranz < delta_max/3 (Fehler kombinieren sich)."""
    return _check_delta(delta_max) / 3.0


def needs_maintenance(omega: float) -> bool:
    """Unsicherheits-Monitoring: Wartung, wenn Omega_circuit > 0,6."""
    return _check_omega(omega) > MAINTENANCE_OMEGA


def sustainability_conditions(alpha: float, period: float, mean_supply: float, mean_demand: float,
                              omega_demand_max: float, cascade_ok: bool, omega_system: float,
                              domega_dt: float, critical_rate: float) -> dict[str, bool]:
    """Die fünf Bedingungen des Nachhaltigkeitssatzes (True = erfüllt) und ``all``."""
    cond = {
        "stability": alpha > 2.0 * math.pi / _positive(period, "T"),
        "resources": mean_supply - mean_demand > 0.0,
        "frequency_separation": omega_demand_max < alpha,
        "leibniz_structure": bool(cascade_ok),
        "uncertainty": (OMEGA_VIABLE[0] <= omega_system <= OMEGA_VIABLE[1]) and domega_dt < critical_rate,
    }
    cond["all"] = all(cond.values())
    return cond


CORN_LAW_CASE = {
    "before": {"omega": 0.45, "dynamics": "RLC mit leichter Dämpfung", "stability": "marginal"},
    "below_limit": {"omega": 0.70},
    "above_limit": {"omega": 0.15},
}

WEIMAR_PHASES: list[dict[str, float | str]] = [
    {"phase": "vor 1923", "omega_lo": 0.40, "omega_hi": 0.50, "domega_dt": 0.02},
    {"phase": "1923 Q1-Q2", "omega_lo": 0.60, "omega_hi": 0.70, "domega_dt": 0.15},
    {"phase": "1923 Q3-Q4", "omega_lo": 0.95, "omega_hi": 0.95, "domega_dt": math.inf},
    {"phase": "Rentenmark", "omega_lo": 0.35, "omega_hi": 0.40, "domega_dt": 0.0},
]


def omega_rate(omegas: Sequence[float], dt: float) -> np.ndarray:
    """Änderungsrate dOmega/dt einer Zeitreihe (Differenzenquotient)."""
    _positive(dt, "dt")
    arr = np.asarray(omegas, dtype=float)
    if arr.size < 2:
        raise ValueError("mindestens zwei Werte erforderlich")
    return np.diff(arr) / dt


def bipolar_jump(omega_a: float, omega_b: float, threshold: float = 0.4) -> bool:
    """Sprung zwischen zwei Zuständen (Korngesetz): |Omega_a - Omega_b| >= threshold."""
    return abs(omega_a - omega_b) >= threshold
