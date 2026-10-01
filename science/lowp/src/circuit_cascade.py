"""Elektronische Schaltkreise und die Leibniz-Struktur: Rechenmodul zum Kapitel
"Elektronische Schaltkreise und die Leibniz-Struktur" (lowp.tex).

Nur Standardbibliothek. Abgebildet werden

* die Hierarchie der Schaltungstypen (Satz "Schaltungstypen und Leibniz-Struktur"),
* die ungeraden Harmonischen des Rechtecksignals und die Leibniz-Reihe,
* die parallele RLC-Kaskade mit Resonanzen (2n+1)*omega_0 und L_n = L_1/(2n+1)^2,
* die Impulsantwort h(t) = sum e^{-(2n+1) alpha t}/(2n+1),
* die technischen Beispiele (Stromrichter, Synthesizer),
* der Vergleich Standard-RC-Kaskade und Leibniz-RC-Kaskade,
* die Designrichtlinien.

Indexkonvention: Stufe n = 0, 1, 2, ... hat die Resonanzfrequenz (2n+1)*omega_0.
Im Kapitel heißen die Stufen deshalb auch "1., 3., 5., 7. Stufe".

Hinweise zu Aussagen des Kapitels, die sich nicht ohne weiteres reproduzieren lassen,
stehen in den jeweiligen Docstrings (``h(t)`` ohne alternierende Vorzeichen,
Spalte "Leibniz-RC" der Vergleichstabelle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _positive(value: float, name: str) -> float:
    if not value > 0.0:
        raise ValueError(f"{name} muss positiv sein")
    return float(value)


def _check_count(n: int, name: str = "n_terms", minimum: int = 1) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
        raise ValueError(f"{name} muss eine ganze Zahl >= {minimum} sein")
    return n


# --------------------------------------------------------------------------
# Satz: Schaltungstypen und Leibniz-Struktur
# --------------------------------------------------------------------------

CIRCUIT_TYPES: dict[str, dict[str, str]] = {
    "rc_lowpass": {
        "name": "RC-Tiefpass",
        "equation": "RC dV/dt + V = V_in",
        "time_constant": "tau = RC",
        "suitability": "gering",
    },
    "rl_lowpass": {
        "name": "RL-Tiefpass",
        "equation": "L dI/dt + R I = V_in",
        "time_constant": "tau = L/R",
        "suitability": "gering",
    },
    "rlc_resonator": {
        "name": "RLC-Resonator",
        "equation": "L I'' + R I' + I/C = V_in",
        "time_constant": "eine Resonanzfrequenz",
        "suitability": "nicht ausreichend",
    },
    "parallel_rlc_cascade": {
        "name": "Parallele RLC-Resonator-Kaskade mit ungeraden Resonanzfrequenzen",
        "equation": "omega_n = (2n+1) omega_0, L_n ~ 1/(2n+1)^2",
        "time_constant": "mehrere Resonanzen",
        "suitability": "ideal",
    },
}

SUITABILITY_RANK = {"gering": 1, "nicht ausreichend": 0, "ideal": 3}


def circuit_type(kind: str) -> dict[str, str]:
    """Beschreibung eines Schaltungstyps (Schlüssel siehe ``CIRCUIT_TYPES``)."""
    try:
        return dict(CIRCUIT_TYPES[kind])
    except KeyError:
        raise ValueError(f"unbekannter Schaltungstyp: {kind!r}") from None


def leibniz_suitability(kind: str) -> str:
    """Eignung eines Schaltungstyps zur Realisierung der Leibniz-Struktur."""
    return circuit_type(kind)["suitability"]


def best_circuit_type() -> str:
    """Schaltungstyp mit der besten Eignung (die parallele RLC-Kaskade)."""
    return max(CIRCUIT_TYPES, key=lambda k: SUITABILITY_RANK[CIRCUIT_TYPES[k]["suitability"]])


# --------------------------------------------------------------------------
# Rechtecksignal, ungerade Harmonische, Leibniz-Reihe
# --------------------------------------------------------------------------

def odd_harmonic_numbers(n_terms: int) -> list[int]:
    """Ordnungszahlen 1, 3, 5, ... der ersten n_terms ungeraden Harmonischen."""
    return [2 * n + 1 for n in range(_check_count(n_terms))]


def square_wave_series(t: float, omega: float, v0: float = 1.0, n_terms: int = 50) -> float:
    """Partialsumme (4 V0/pi) * sum sin((2n+1) omega t)/(2n+1) der Rechteck-Fourier-Reihe."""
    _check_count(n_terms)
    s = math.fsum(math.sin((2 * n + 1) * omega * t) / (2 * n + 1) for n in range(n_terms))
    return 4.0 * v0 / math.pi * s


def square_wave(t: float, omega: float, v0: float = 1.0) -> float:
    """Ideales Rechtecksignal der Amplitude V0 (Vorzeichen von sin(omega t); 0 an den Sprüngen)."""
    s = math.sin(omega * t)
    if abs(s) < 1e-15:
        return 0.0
    return v0 if s > 0 else -v0


def harmonic_amplitudes(v0: float, n_terms: int) -> list[float]:
    """Amplituden 4 V0/(pi (2n+1)) der ungeraden Harmonischen des Rechtecksignals."""
    return [4.0 * v0 / (math.pi * k) for k in odd_harmonic_numbers(n_terms)]


def leibniz_partial_sum(n_terms: int) -> float:
    """sum_{n<N} (-1)^n/(2n+1) -> pi/4."""
    _check_count(n_terms)
    return math.fsum((-1) ** n / (2 * n + 1) for n in range(n_terms))


def leibniz_from_square_wave(n_terms: int) -> float:
    """Leibniz-Partialsumme aus der Rechteck-Reihe: (pi/4) * (Rechteckreihe bei omega t = pi/2).

    Bei omega t = pi/2 ist sin((2n+1) pi/2) = (-1)^n und das Rechteck gleich V0 = 1, also
    (4/pi) sum (-1)^n/(2n+1) -> 1. Multiplikation mit pi/4 liefert die Leibniz-Partialsumme
    (Verbindung der beiden Reihen im Kapitel); sie strebt gegen pi/4.
    """
    return math.pi / 4.0 * square_wave_series(math.pi / 2.0, 1.0, 1.0, n_terms)


# --------------------------------------------------------------------------
# Parallele RLC-Kaskade
# --------------------------------------------------------------------------

def resonance_frequency(inductance: float, capacitance: float) -> float:
    """Kreisfrequenz omega = 1/sqrt(L C)."""
    return 1.0 / math.sqrt(_positive(inductance, "L") * _positive(capacitance, "C"))


def stage_inductance(l1: float, n: int) -> float:
    """L_n = L_1/(2n+1)^2 (Skalierungsgesetz)."""
    _check_count(n, "n", 0)
    return _positive(l1, "L1") / (2 * n + 1) ** 2


def inductance_ratios(n_stages: int) -> list[float]:
    """Verhältnisse L_n/L_1 = 1 : 1/9 : 1/25 : 1/49 : ..."""
    return [1.0 / (2 * n + 1) ** 2 for n in range(_check_count(n_stages))]


@dataclass(frozen=True)
class LeibnizCascade:
    """Parallele RLC-Resonatoren, abgestimmt auf (2n+1)*omega_0, mit gleichem C und R."""

    L1: float
    C: float
    R: float
    n_stages: int = 4

    def __post_init__(self) -> None:
        _positive(self.L1, "L1")
        _positive(self.C, "C")
        _positive(self.R, "R")
        _check_count(self.n_stages, "n_stages")

    # -- Komponenten -------------------------------------------------------
    def inductance(self, n: int) -> float:
        """Induktivität L_n der Stufe n."""
        return stage_inductance(self.L1, n)

    def inductances(self) -> list[float]:
        return [self.inductance(n) for n in range(self.n_stages)]

    def capacitances(self) -> list[float]:
        """Alle Kapazitäten sind identisch gleich C."""
        return [self.C] * self.n_stages

    def resistances(self) -> list[float]:
        """Alle Dämpfungswiderstände sind identisch gleich R."""
        return [self.R] * self.n_stages

    # -- Resonanzen ----------------------------------------------------------
    @property
    def omega0(self) -> float:
        """Grundresonanz 1/sqrt(L_1 C)."""
        return resonance_frequency(self.L1, self.C)

    def resonance(self, n: int) -> float:
        """omega_n = 1/sqrt(L_n C)."""
        return resonance_frequency(self.inductance(n), self.C)

    def resonances(self) -> list[float]:
        return [self.resonance(n) for n in range(self.n_stages)]

    def resonance_ratios(self) -> list[float]:
        """omega_n/omega_0; muss 1, 3, 5, 7, ... sein."""
        return [w / self.omega0 for w in self.resonances()]

    def verify_resonances(self, rel_tol: float = 1e-12) -> bool:
        """Prüft omega_n = (2n+1) omega_0 für alle Stufen (Verifikationsschritt)."""
        return all(math.isclose(self.resonance(n), (2 * n + 1) * self.omega0, rel_tol=rel_tol)
                   for n in range(self.n_stages))

    # -- Dämpfung und Güte ---------------------------------------------------
    @property
    def alpha(self) -> float:
        """Dämpfungskonstante alpha = R/(2 L_1)."""
        return self.R / (2.0 * self.L1)

    def stage_damping(self, n: int) -> float:
        """Physikalische Dämpfung R/(2 L_n) der Stufe n (wächst bei festem R mit (2n+1)^2)."""
        return self.R / (2.0 * self.inductance(n))

    def quality_factor(self, n: int) -> float:
        """Güte Q_n = omega_n L_n / R = Q_0/(2n+1)."""
        return self.resonance(n) * self.inductance(n) / self.R

    # -- Impulsantwort -------------------------------------------------------
    def impulse_response(self, t: float, signed: bool = False) -> float:
        """h(t) = sum_{n<N} (+-1)^n e^{-(2n+1) alpha t}/(2n+1) der Kapitelformel.

        ``signed=False``: Gewichte 1/(2n+1) wie im Kapitel (Grenzwert N -> oo ist artanh(e^{-alpha t})).
        ``signed=True``:  alternierend (-1)^n/(2n+1) (Grenzwert arctan(e^{-alpha t}), die eigentliche
        parametrisierte Leibniz-Reihe L_odd). Das Kapitel setzt beide gleich; sie unterscheiden sich
        durch die Vorzeichen.
        """
        return impulse_response(t, self.alpha, self.n_stages, signed)


def impulse_response(t: float, alpha: float, n_terms: int, signed: bool = False) -> float:
    """Partialsumme der Impulsantwort sum_{n<N} (+-1)^n e^{-(2n+1) alpha t}/(2n+1), t >= 0."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    _positive(alpha, "alpha")
    _check_count(n_terms)
    x = math.exp(-alpha * t)
    return math.fsum(((-1) ** n if signed else 1) * x ** (2 * n + 1) / (2 * n + 1)
                     for n in range(n_terms))


def impulse_response_closed_form(t: float, alpha: float, signed: bool = False) -> float:
    """Grenzwert N -> oo: artanh(e^{-alpha t}) bzw. (alternierend) arctan(e^{-alpha t}); t > 0 bzw. >= 0."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    _positive(alpha, "alpha")
    x = math.exp(-alpha * t)
    if signed:
        return math.atan(x)
    if x >= 1.0:
        raise ValueError("artanh(1) divergiert: t muss > 0 sein")
    return math.atanh(x)


# --------------------------------------------------------------------------
# Technische Anwendungsbeispiele
# --------------------------------------------------------------------------

def converter_spectrum(fundamental_amp: float = 10.0, f0: float = 50.0,
                       orders: Sequence[int] = (1, 3, 5, 7)) -> list[dict[str, float]]:
    """Spektrum eines Stromrichters (Beispiel 1): Amplitude I_1/k bei der Frequenz k*f0.

    Standardwerte: 10 A bei 50 Hz -> 3,33 A / 2,0 A / 1,43 A bei 150 / 250 / 350 Hz.
    """
    _positive(fundamental_amp, "fundamental_amp")
    _positive(f0, "f0")
    rows = []
    for k in orders:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k % 2 == 0:
            raise ValueError("Ordnungszahlen müssen ungerade ganze Zahlen >= 1 sein")
        rows.append({"order": k, "frequency": k * f0, "amplitude": fundamental_amp / k})
    return rows


def harmonic_content(amplitudes: Iterable[float]) -> float:
    """Effektiver Oberschwingungsanteil sqrt(sum a_k^2) der übergebenen Amplituden."""
    return math.sqrt(math.fsum(a * a for a in amplitudes))


def distortion_factor(fundamental: float, harmonics: Iterable[float]) -> float:
    """Klirrfaktor-Verhältnis sqrt(sum a_k^2)/a_1."""
    return harmonic_content(harmonics) / _positive(fundamental, "fundamental")


def apply_filter_bank(rows: Sequence[dict[str, float]], attenuation: float = 0.85,
                      filtered_orders: Iterable[int] = (3, 5, 7)) -> list[dict[str, float]]:
    """Dämpft die Oberschwingungen ``filtered_orders`` um den Anteil ``attenuation`` (0..1).

    Die Grundwelle bleibt unverändert (nahezu ungedämpft durchgelassen).
    """
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("attenuation muss in [0, 1] liegen")
    orders = set(filtered_orders)
    out = []
    for row in rows:
        new = dict(row)
        if int(row["order"]) in orders:
            new["amplitude"] = row["amplitude"] * (1.0 - attenuation)
        out.append(new)
    return out


def reduction_percent(before: Iterable[float], after: Iterable[float]) -> float:
    """Prozentuale Reduktion des effektiven Oberschwingungsanteils."""
    b = harmonic_content(before)
    if b == 0.0:
        raise ValueError("Ausgangsanteil darf nicht null sein")
    return 100.0 * (1.0 - harmonic_content(after) / b)


def synthesizer_harmonics(f1: float = 440.0, n_terms: int = 4) -> list[float]:
    """Frequenzen f1, 3 f1, 5 f1, ... des Rechteck-Oszillators (Beispiel 2: 440, 1320, 2200, 3080 Hz)."""
    _positive(f1, "f1")
    return [k * f1 for k in odd_harmonic_numbers(n_terms)]


def synthesizer_weights(n_terms: int, gains: Sequence[float] | None = None) -> list[float]:
    """Relative Amplituden 1/(2n+1) der Harmonischen, optional mit Gewichten ``gains`` skaliert."""
    base = [1.0 / k for k in odd_harmonic_numbers(n_terms)]
    if gains is None:
        return base
    if len(gains) != n_terms:
        raise ValueError("gains muss n_terms Einträge haben")
    return [b * g for b, g in zip(base, gains)]


# --------------------------------------------------------------------------
# Standard-Kaskade versus Leibniz-Kaskade
# --------------------------------------------------------------------------

def rc_time_constant(r: float, c: float) -> float:
    """tau = R*C."""
    return _positive(r, "R") * _positive(c, "C")


def standard_rc_taus(r: float, c: float, n_stages: int = 3) -> list[float]:
    """Standard-Design: alle Stufen identisch, tau = R C."""
    return [rc_time_constant(r, c)] * _check_count(n_stages, "n_stages")


def leibniz_rc_taus(r: float, c1: float, n_stages: int = 3) -> list[float]:
    """Leibniz-Design: C_n = C_1/(2n+1)^2, damit tau_n = R C_1/(2n+1)^2 (1,0 / 0,111 / 0,04 ms)."""
    _positive(r, "R")
    _positive(c1, "C1")
    return [rc_time_constant(r, c1 / (2 * n + 1) ** 2) for n in range(_check_count(n_stages, "n_stages"))]


def rc_decay(t: float, tau: float) -> float:
    """Abklingfaktor e^{-t/tau}."""
    if t < 0.0:
        raise ValueError("t muss >= 0 sein")
    return math.exp(-t / _positive(tau, "tau"))


def weighted_decay_sum(t: float, taus: Sequence[float], signed: bool = False,
                       normalize: bool = False) -> float:
    """Summe sum_n w_n e^{-t/tau_n} mit w_n = (+-1)^n/(2n+1).

    ``normalize=True`` teilt durch die Summe der Gewichte. Mit diesen natürlichen Definitionen
    ergibt sich bei t = tau_1 der Wert e^{-1} (+ winzige Beiträge), nicht der im Kapitel
    angegebene Wert 34,9 % (siehe ``CASCADE_TABLE_CLAIMED``).
    """
    if not taus:
        raise ValueError("taus darf nicht leer sein")
    total = math.fsum(((-1) ** n if signed else 1) / (2 * n + 1) * rc_decay(t, tau)
                      for n, tau in enumerate(taus))
    if normalize:
        total /= math.fsum(((-1) ** n if signed else 1) / (2 * n + 1) for n in range(len(taus)))
    return total


# Tabelle des Kapitels: Zeit (in tau_1), Standard-RC, Leibniz-RC, Verhältnis (Angaben in Prozent).
CASCADE_TABLE_CLAIMED: list[dict[str, float]] = [
    {"t_over_tau": 1.0, "standard_percent": 36.8, "leibniz_percent": 34.9, "ratio": 0.95},
    {"t_over_tau": 2.0, "standard_percent": 13.5, "leibniz_percent": 0.13, "ratio": 0.01},
    {"t_over_tau": 5.0, "standard_percent": 0.67, "leibniz_percent": 1e-6, "ratio": 1e-4},
]


def standard_table_values() -> list[float]:
    """Spalte 'Standard-RC' der Tabelle in Prozent: 100*e^{-k} für k = 1, 2, 5 (36,8 / 13,5 / 0,67)."""
    return [100.0 * math.exp(-row["t_over_tau"]) for row in CASCADE_TABLE_CLAIMED]


def decay_speed_ratio(t: float, taus_fast: Sequence[float], tau_ref: float) -> float:
    """Verhältnis der gewichteten Leibniz-Summe zum Einzelabfall e^{-t/tau_ref} bei Zeit t."""
    return weighted_decay_sum(t, taus_fast) / rc_decay(t, tau_ref)


# --------------------------------------------------------------------------
# Designrichtlinien
# --------------------------------------------------------------------------

def design_cascade(f0: float, c: float, r: float, n_stages: int) -> LeibnizCascade:
    """Implementierungsschritte: aus der Grundfrequenz f0 und C folgt L_1 = 1/((2 pi f0)^2 C)."""
    _positive(f0, "f0")
    _positive(c, "C")
    w0 = 2.0 * math.pi * f0
    return LeibnizCascade(L1=1.0 / (w0 * w0 * c), C=c, R=r, n_stages=n_stages)


def resonance_frequencies_hz(f0: float, n_stages: int) -> list[float]:
    """Resonanzfrequenzen f_n = (2n+1) f_0 der Stufen."""
    _positive(f0, "f0")
    return [(2 * n + 1) * f0 for n in range(_check_count(n_stages, "n_stages"))]


def contains_only_odd_harmonics(orders: Iterable[int]) -> bool:
    """Spektralanalyse (Schritt 1): Sind ausschließlich ungerade Harmonische vorhanden?"""
    return all(isinstance(k, int) and k % 2 == 1 for k in orders)
