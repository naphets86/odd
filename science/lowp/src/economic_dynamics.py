"""Die Leibniz-Struktur in Wirtschaftsdynamiken: Rechenmodul zum Kapitel
"Die Leibniz-Struktur in Wirtschaftsdynamiken" (lowp.tex).

Abgebildet werden die drei Theoreme, das Kollaps-Korollar, die Anwendungsbeispiele und die
Kollaps-Prognose des Kapitels. Mathematisch steht dahinter immer die lineare Gleichung erster
Ordnung  dx/dt = -alpha*x + eta*sin(2*pi*t/T)  (Tiefpass mit tau = 1/alpha).

Hinweise zu den Formeln des Kapitels
    * Das Kapitel schreibt dP/dt = alpha*(P - P0) + ...; mit dem Vorzeichen "+" wüchse jede
      Abweichung exponentiell. Gemeint ist (gemäß der Zuordnung alpha <-> 1/tau) die
      abklingende Form mit -alpha; so ist sie hier implementiert.
    * Für einen Tiefpass erster Ordnung ist |H| = 1/sqrt(1 + (omega tau)^2) <= 1; er hat keine
      Resonanzüberhöhung. Die Funktionen ``resonance_*`` setzen deshalb nur die im Kapitel
      formulierten Kriterien um (Vergleich alpha mit 2*pi/T usw.).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad

#: Gütefaktor, den das Kapitel für den einfachen RC-Filter angibt.
RC_QUALITY_FACTOR = 0.5


def _positive(value: float, name: str) -> float:
    if not value > 0.0:
        raise ValueError(f"{name} muss positiv sein")
    return float(value)


def _check_count(n: int, name: str = "n_terms", minimum: int = 1) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
        raise ValueError(f"{name} muss eine ganze Zahl >= {minimum} sein")
    return n


# --------------------------------------------------------------------------
# Theorem 1: Tiefpassfilter-Struktur als Wirtschaftsstabilisator
# --------------------------------------------------------------------------

def adjustment_time(r: float, c: float) -> float:
    """Wirtschaftliche Anpassungszeit tau = R*C."""
    return _positive(r, "R") * _positive(c, "C")


def sensitivity(tau: float) -> float:
    """Sensitivität des Systems 1/(R C) = 1/tau."""
    return 1.0 / _positive(tau, "tau")


def lowpass_response(t: float, tau: float, v0: float, v_in: Callable[[float], float]) -> float:
    """V_out(t) = e^{-t/tau} V_out(0) + (1/tau) int_0^t e^{-(t-s)/tau} V_in(s) ds (Variation der Konstanten)."""
    _positive(tau, "tau")
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    if t == 0.0:
        return float(v0)
    integral, _ = quad(lambda s: math.exp(-(t - s) / tau) * v_in(s), 0.0, t)
    return math.exp(-t / tau) * v0 + integral / tau


def shock_influence(t: float, s: float, tau: float) -> float:
    """Einfluss eines Schocks zur Zeit s auf V_out(t): (1/tau) e^{-(t-s)/tau}, t >= s."""
    _positive(tau, "tau")
    if t < s:
        raise ValueError("t muss >= s sein")
    return math.exp(-(t - s) / tau) / tau


def max_shock_influence(tau: float) -> float:
    """Obere Schranke 1/tau des Schockeinflusses (angenommen bei t = s)."""
    return 1.0 / _positive(tau, "tau")


def magnitude(omega: float, tau: float) -> float:
    """|H(i omega)| = 1/sqrt(1 + omega^2 tau^2)."""
    return 1.0 / math.sqrt(1.0 + (omega * _positive(tau, "tau")) ** 2)


def phase(omega: float, tau: float) -> float:
    """Phase -arctan(omega tau) des Tiefpasses (Bogenmaß)."""
    return -math.atan(omega * _positive(tau, "tau"))


def cutoff_omega(tau: float) -> float:
    """Grenzkreisfrequenz omega_c = 1/tau (|H| = 1/sqrt 2)."""
    return 1.0 / _positive(tau, "tau")


def cutoff_frequency(tau: float) -> float:
    """Kritische Frequenz der Marktvolatilität f_c = 1/(2 pi tau)."""
    return 1.0 / (2.0 * math.pi * _positive(tau, "tau"))


def period_damping(period: float, tau: float) -> float:
    """|H(2 pi i/T)|: Dämpfungsfaktor eines periodischen Schocks mit Periode T."""
    return magnitude(2.0 * math.pi / _positive(period, "T"), tau)


def shock_classification(period: float, tau: float) -> str:
    """'damped' für T < 2 pi tau (f > f_c, |H| < 0,707); 'follows' für T > 2 pi tau; sonst 'critical'."""
    _positive(period, "T")
    _positive(tau, "tau")
    crit = 2.0 * math.pi * tau
    if math.isclose(period, crit, rel_tol=1e-12):
        return "critical"
    return "follows" if period > crit else "damped"


def periodic_shock_absorbed(period: float, tau: float) -> bool:
    """Kriterium des Theorems: Der Schock mit Periode T > 2 pi tau wird stabil abgefedert (f < f_c)."""
    return shock_classification(period, tau) == "follows"


def phase_at_cutoff_degrees() -> float:
    """Phase des Tiefpasses bei der Grenzfrequenz: -45 Grad."""
    return math.degrees(phase(1.0, 1.0))


def rc_quality_factor() -> float:
    """Gütefaktor Q = 1/2, den das Kapitel für den einfachen RC-Filter ansetzt."""
    return RC_QUALITY_FACTOR


# --------------------------------------------------------------------------
# Theorem 2: Ungerade Harmonische in wirtschaftlichen Zyklen
# --------------------------------------------------------------------------

def heaviside_square_series(t: float, omega: float, n_terms: int = 50) -> float:
    """Fourier-Reihe 1/2 + (2/pi) sum sin((2n+1) omega t)/(2n+1) des 0/1-Rechtecks."""
    _check_count(n_terms)
    s = math.fsum(math.sin((2 * n + 1) * omega * t) / (2 * n + 1) for n in range(n_terms))
    return 0.5 + 2.0 / math.pi * s


def saturate(demand: float, s_max: float) -> float:
    """Angebot S = min(D, S_max) (Kapazitätsbegrenzung, Sättigung)."""
    return demand if demand <= s_max else s_max


def clipped_sine_harmonics(s_max: float, n_harmonics: int = 9, n_samples: int = 4096) -> list[float]:
    """Amplituden der Harmonischen k = 1..n_harmonics eines bei +-s_max begrenzten Sinus der Amplitude 1.

    Wegen der Halbwellensymmetrie erzeugt die Sättigung nur ungerade Harmonische; die
    geradzahligen Amplituden verschwinden (bis auf Rundungsfehler).
    """
    _positive(s_max, "s_max")
    _check_count(n_harmonics, "n_harmonics")
    _check_count(n_samples, "n_samples", 8)
    phi = 2.0 * np.pi * np.arange(n_samples) / n_samples
    clipped = np.clip(np.sin(phi), -s_max, s_max)
    spec = np.fft.rfft(clipped) / n_samples
    return [float(2.0 * abs(spec[k])) for k in range(1, n_harmonics + 1)]


def supply_function(t: float, s0: float, coeffs: Sequence[float], omega: float,
                    phases: Sequence[float] | None = None) -> float:
    """S(t) = S_0 + sum_n S_n/(2n+1) sin((2n+1) omega t + phi_n) (Leibniz-Skalierung)."""
    if phases is None:
        phases = [0.0] * len(coeffs)
    if len(phases) != len(coeffs):
        raise ValueError("phases und coeffs müssen gleich lang sein")
    return s0 + math.fsum(c / (2 * n + 1) * math.sin((2 * n + 1) * omega * t + ph)
                          for n, (c, ph) in enumerate(zip(coeffs, phases)))


def l_odd(t: float, lam: float, n_terms: int = 200) -> float:
    """L_odd(t) = sum (-1)^n/(2n+1) e^{-lam (2n+1) t} (Marktdämpfung), t >= 0."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    _positive(lam, "lam")
    _check_count(n_terms)
    x = math.exp(-lam * t)
    return math.fsum((-1) ** n * x ** (2 * n + 1) / (2 * n + 1) for n in range(n_terms))


def l_odd_closed_form(t: float, lam: float) -> float:
    """Grenzwert arctan(e^{-lam t}) von L_odd(t)."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    return math.atan(math.exp(-_positive(lam, "lam") * t))


def market_damping_rate(tau_market: float) -> float:
    """lambda = 1/tau_market_response (Wochen bis Monate)."""
    return 1.0 / _positive(tau_market, "tau_market")


# --------------------------------------------------------------------------
# Theorem 3: Anthropogene Kraft und natürliche Periodizität
# --------------------------------------------------------------------------

def forced_amplitude(alpha: float, period: float, eta: float) -> float:
    """Amplitude A = eta/sqrt(alpha^2 + (2 pi/T)^2) des stabilen Grenzzyklus."""
    w = 2.0 * math.pi / _positive(period, "T")
    return eta / math.sqrt(_positive(alpha, "alpha") ** 2 + w * w)


def forced_phase(alpha: float, period: float) -> float:
    """Phase phi = -arctan((2 pi/T)/alpha) des Grenzzyklus."""
    return -math.atan(2.0 * math.pi / _positive(period, "T") / _positive(alpha, "alpha"))


def limit_cycle(t: float, p0: float, eta: float, alpha: float, period: float) -> float:
    """P*(t) = P_0 + A sin(2 pi t/T + phi)."""
    return p0 + forced_amplitude(alpha, period, eta) * math.sin(
        2.0 * math.pi * t / period + forced_phase(alpha, period))


def simulate_price(p_start: float, p0: float, eta: float, alpha: float, period: float,
                   t_end: float, steps: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Löst dP/dt = -alpha (P - P0) + eta sin(2 pi t/T) mit klassischem Runge-Kutta (RK4)."""
    _positive(alpha, "alpha")
    _positive(period, "T")
    _positive(t_end, "t_end")
    _check_count(steps, "steps", 2)
    h = t_end / steps
    w = 2.0 * math.pi / period

    def f(t: float, p: float) -> float:
        return -alpha * (p - p0) + eta * math.sin(w * t)

    ts = np.linspace(0.0, t_end, steps + 1)
    ps = np.empty(steps + 1)
    ps[0] = p_start
    for i in range(steps):
        t, p = ts[i], ps[i]
        k1 = f(t, p)
        k2 = f(t + h / 2, p + h * k1 / 2)
        k3 = f(t + h / 2, p + h * k2 / 2)
        k4 = f(t + h, p + h * k3)
        ps[i + 1] = p + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return ts, ps


def stability_condition(alpha: float, period: float) -> bool:
    """Stabilitätskriterium des Kapitels: 2 pi/T < alpha, äquivalent T > 2 pi/alpha."""
    return 2.0 * math.pi / _positive(period, "T") < _positive(alpha, "alpha")


def critical_period(alpha: float) -> float:
    """Kritische Periode T* = 2 pi/alpha (alpha = 2 pi/T)."""
    return 2.0 * math.pi / _positive(alpha, "alpha")


def is_critical_resonance(alpha: float, period: float, rel_tol: float = 1e-9) -> bool:
    """Kritische Resonanz: alpha = 2 pi/T."""
    return math.isclose(alpha, 2.0 * math.pi / _positive(period, "T"), rel_tol=rel_tol)


def minimal_alpha(period: float) -> float:
    """Kleinstes alpha für Stabilität bei Periode T: 2 pi/T (T = 365 Tage: 0,0172 pro Tag)."""
    return 2.0 * math.pi / _positive(period, "T")


# --------------------------------------------------------------------------
# Korollar: Wirtschaftliche Kollaps-Bedingungen und Kollaps-Prognose
# --------------------------------------------------------------------------

def resource_level(r0: float, demand: Callable[[float], float], supply: Callable[[float], float],
                   t: float) -> float:
    """R(t) = R_0 - int_0^t [D(s) - S(s)] ds."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    if t == 0.0:
        return float(r0)
    integral, _ = quad(lambda s: demand(s) - supply(s), 0.0, t)
    return r0 - integral


def resources_exhausted(r0: float, demand: Callable[[float], float],
                        supply: Callable[[float], float], t: float) -> bool:
    """Ressourcenerschöpfung: R(t) < 0 (harte Nebenbedingung)."""
    return resource_level(r0, demand, supply, t) < 0.0


def overdemand(demand: float, resources: float) -> bool:
    """Unbegründeter Überbedarf D > R."""
    return demand > resources


def high_frequency_shock(f_shock: float, alpha: float) -> bool:
    """Hochfrequenter Schock: f_shock > alpha/(2 pi) = f_c."""
    return f_shock > _positive(alpha, "alpha") / (2.0 * math.pi)


def too_sluggish(alpha: float, period: float) -> bool:
    """Trägheit größer als Anregung: alpha < 2 pi/T."""
    return alpha < 2.0 * math.pi / _positive(period, "T")


def collapse_flags(alpha: float, period: float, f_shock: float, demand: float,
                   resources: float, reserve: float) -> dict[str, bool]:
    """Bedingungen 1-4 des Kollaps-Korollars (True = Bedingung für Kollaps erfüllt)."""
    return {
        "high_frequency_shock": high_frequency_shock(f_shock, alpha),
        "overdemand": overdemand(demand, resources),
        "too_sluggish": too_sluggish(alpha, period),
        "resource_exhaustion": reserve < 0.0,
    }


def collapse_timescale_resonance(alpha: float, period: float) -> float:
    """Zeitskala 1/|alpha - 2 pi/T| der Resonanz-Bedingung (unendlich im kritischen Fall)."""
    diff = abs(alpha - 2.0 * math.pi / _positive(period, "T"))
    return math.inf if diff == 0.0 else 1.0 / diff


def collapse_time_resources(r0: float, mean_demand: float) -> float:
    """Kollaps-Zeitpunkt t_c = R_0/D_mean der Ressourcen-Bedingung."""
    return _positive(r0, "R0") / _positive(mean_demand, "D_mean")


def shock_growth_rate(f_shock: float, alpha: float) -> float:
    """Wachstumsrate sqrt(f^2 - alpha^2/(4 pi^2)) bei f_shock > alpha/(2 pi), sonst 0."""
    _positive(alpha, "alpha")
    fc = alpha / (2.0 * math.pi)
    if f_shock <= fc:
        return 0.0
    return math.sqrt(f_shock ** 2 - fc ** 2)


def will_collapse(alpha: float, period: float, f_shock: float, r0: float, mean_demand: float,
                  horizon: float) -> bool:
    """Prognose-Satz: Kollaps, wenn eine der drei Bedingungen erfüllt ist.

    Resonanz-Bedingung alpha < 2 pi/T, Hochfrequenz-Bedingung f_shock > alpha/(2 pi) oder
    Ressourcen-Bedingung t_c = R_0/D_mean <= horizon.
    """
    return (too_sluggish(alpha, period)
            or high_frequency_shock(f_shock, alpha)
            or collapse_time_resources(r0, mean_demand) <= horizon)


# --------------------------------------------------------------------------
# Anwendungsbeispiele
# --------------------------------------------------------------------------

def medieval_example(alpha: float = 0.03, eta: float = 0.5, period: float = 15.0) -> dict[str, float]:
    """Mittelalterlicher Agrar-Zyklus (alpha = 0,03/Jahr, T = 10..20 Jahre, eta = 50 %).

    Liefert T* = 2 pi/alpha (ca. 209 Jahre), die exakte Amplitude und die Kapitel-Näherung eta/alpha.
    """
    return {
        "critical_period": critical_period(alpha),
        "stable": float(stability_condition(alpha, period)),
        "amplitude_exact": forced_amplitude(alpha, period, eta),
        "amplitude_over_alpha": eta / alpha,
    }


def modern_example(alpha_per_day: float = 10.0, eta: float = 1.0, period_days: float = 365.0) -> dict[str, float]:
    """Moderne Märkte (alpha = 10/Tag, T = 365 Tage): A ~ eta/10."""
    return {
        "minimal_alpha": minimal_alpha(period_days),
        "stable": float(stability_condition(alpha_per_day, period_days)),
        "amplitude": forced_amplitude(alpha_per_day, period_days, eta),
    }


def alpha_ratio(alpha_modern_per_day: float = 10.0, alpha_medieval_per_year: float = 0.03,
                days_per_year: float = 365.0) -> float:
    """Verhältnis alpha_modern/alpha_medieval in gleichen Einheiten (pro Tag).

    Mit den Zahlen des Kapitels ergibt sich ca. 1,2*10^5, nicht der im Text genannte Faktor 300 bis 1000.
    """
    return alpha_modern_per_day / (_positive(alpha_medieval_per_year, "alpha") / days_per_year)
