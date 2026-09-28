#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments.py – Numerische Experimente zur Leibniz-Arbeit
==========================================================

Jedes Experiment
    1. berechnet seine Daten,
    2. speichert sie als JSON   (<outdir>/expNN_<name>.json),
    3. zeichnet einen matplotlib-Plot und speichert ihn als PDF
       (<outdir>/expNN_<name>.pdf).

Der Plot wird aus den *aus der JSON-Datei zurückgelesenen* Daten erzeugt,
d. h. PDF und JSON sind garantiert konsistent.

Experimente
-----------
 1  leibniz_basis          Leibniz-Reihe, O(1/N)-Konvergenz, asymmetrische Zerlegung
 2  beschleunigung         Euler / Mitteln / Shanks / Formeln der Arbeit im Vergleich
 3  dirichlet_beta         β(s) auf ℝ, Funktionalgleichung, Sonderwerte (β ist ganz!)
 4  catalan                Darstellungen der Catalan-Konstante G
 5  parametrisierte_leibniz L(t;λ) = arctan(exp(-λt)), Integrodifferentialgleichung
 6  e_funktion_taylor      Ableitung der Taylor-Reihe (Beobachtung 49): "ein Glied geht verloren"
 7  rc_tiefpass            Bode-Diagramm, Sprung- und Sinusantwort des RC-Filters
 8  rlc_daempfung          RLC-Sprungantworten, Überschwingen vs. Dämpfung ζ
 9  formel_audit           Numerische Prüfung der Formeln in Arbeit und Modulen

Aufruf
------
    python experiments.py                    # alle Experimente
    python experiments.py --only 2 4 9       # Auswahl (Nummern oder Namen)
    python experiments.py --list             # Übersicht
    python experiments.py --outdir ergebnisse

Benötigt: numpy, scipy, matplotlib. Die Projektmodule (dirichlet_beta.py,
convergence_acceleration.py, lowpass_filter.py, special_functions.py,
leibniz_analysis.py) werden, wo sinnvoll, wiederverwendet.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
import warnings
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: nur Dateien schreiben
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate, special

warnings.filterwarnings("ignore")

# --- Projektmodule --------------------------------------------------------
from convergence_acceleration import EulerTransformation, RichardsonExtrapolation
from dirichlet_beta import DirichletBeta
from lowpass_filter import RCLowpassFilter, RLCFilter
from special_functions import BetaFunction, IntegralRepresentations

try:  # enthält L_odd(t, λ) – optional, nur für Gegenprobe
    from leibniz_analysis import ParametrizedLeibniz
except Exception:  # pragma: no cover
    ParametrizedLeibniz = None

# --- Konstanten -----------------------------------------------------------
PI = math.pi
PI4 = PI / 4
CATALAN = 0.915965594177219015054603514932384110774
FLOOR = 1e-17  # Untergrenze für Log-Plots (Maschinengenauigkeit)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
OK_COLOR, BAD_COLOR = "#2a9d8f", "#d1495b"


# ==========================================================================
# Hilfsfunktionen: JSON, Registry, Referenzfunktionen
# ==========================================================================
def _clean(o):
    """Macht beliebige numpy/complex-Strukturen JSON-tauglich (NaN/Inf -> null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (complex, np.complexfloating)):
        return {"re": _clean(o.real), "im": _clean(o.imag)}
    return o


def A(x) -> np.ndarray:
    """Liste (evtl. mit None) -> float-Array (None -> NaN)."""
    return np.array(x, dtype=float)


def clip(x):
    """Für Log-Plots: Fehler unter Maschinengenauigkeit anheben."""
    return np.maximum(np.abs(A(x)), FLOOR)


def beta_alt(s: float, M: int = 60) -> float:
    """β(s) für s>0 via wiederholtes Mitteln der Partialsummen
    (Euler/van-Wijngaarden-Beschleunigung, numerisch stabil)."""
    n = np.arange(M)
    P = np.cumsum((-1.0) ** n * (2 * n + 1.0) ** (-s))
    for _ in range(M - 1):
        P = (P[:-1] + P[1:]) / 2
    return float(P[0])


def beta_ref(s: float) -> float:
    """Referenz für die Dirichlet-Betafunktion auf ganz ℝ.

    s>1 : β(s) = 4^{-s} [ζ(s,1/4) − ζ(s,3/4)]   (Hurwitz-Zeta)
    0<s≤1: beschleunigte Reihe
    s≤0 : Funktionalgleichung β(s) = (2/π)^{1-s} cos(πs/2) Γ(1-s) β(1-s)
    """
    if s > 1:
        return float(4.0 ** (-s) * (special.zeta(s, 0.25) - special.zeta(s, 0.75)))
    if s > 0:
        return beta_alt(s)
    return float((2 / PI) ** (1 - s) * math.cos(PI * s / 2) * special.gamma(1 - s) * beta_ref(1 - s))


def leibniz_partial_sums(n_max: int) -> np.ndarray:
    """S[k-1] = Σ_{n=0}^{k-1} (-1)^n/(2n+1)  (k Terme)."""
    n = np.arange(n_max)
    sign = np.where(n % 2 == 0, 1.0, -1.0)
    return np.cumsum(sign / (2 * n + 1.0))


def aitken(S: np.ndarray) -> np.ndarray:
    """Shanks e₁ / Aitken Δ²: (S₂S₀−S₁²)/(S₂−2S₁+S₀), stabile Form S₂ − (S₂−S₁)²/Δ²."""
    d2 = S[2:] - 2 * S[1:-1] + S[:-2]
    with np.errstate(all="ignore"):
        out = S[2:] - (S[2:] - S[1:-1]) ** 2 / d2
    out[np.abs(d2) < 1e-300] = np.nan
    return out


def repeated_average(S: np.ndarray, levels: int) -> np.ndarray:
    for _ in range(levels):
        S = (S[:-1] + S[1:]) / 2
    return S


def euler_exact(n_max: int, paper_variant: bool = False) -> np.ndarray:
    """Euler-Transformation der Leibniz-Reihe in exakter Bruchrechnung.

    korrekt (paper_variant=False):
        E_N = Σ_{n<N} 2^{-(n+1)} Σ_k (-1)^k C(n,k) a_k ,   a_k = 1/(2k+1)
    Formel der Arbeit (paper_variant=True): a_{n-k} statt a_k, also
        inner_paper(n) = (-1)^n · inner_korrekt(n)  -> die Vorzeichen alternieren nicht mehr.
    """
    a = [Fraction(1, 2 * k + 1) for k in range(n_max)]
    acc, out = Fraction(0), []
    for n in range(n_max):
        if paper_variant:
            inner = sum(comb(n, k) * (-1) ** k * a[n - k] for k in range(n + 1))
        else:
            inner = sum(comb(n, k) * (-1) ** k * a[k] for k in range(n + 1))
        acc += inner / 2 ** (n + 1)
        out.append(float(acc))
    return np.array(out)


def L_series(t, lam, N=400):
    """L(t;λ) = Σ (-1)^n/(2n+1) e^{-λ(2n+1)t}  (vektorisiert, t Array)."""
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(N)[:, None]
    return np.sum((-1.0) ** n / (2 * n + 1) * np.exp(-lam * (2 * n + 1) * t[None, :]), axis=0)


def dL_series(t, lam, N=400):
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(N)[:, None]
    return -lam * np.sum((-1.0) ** n * np.exp(-lam * (2 * n + 1) * t[None, :]), axis=0)


# --- Registry -------------------------------------------------------------
EXPERIMENTS: Dict[int, dict] = {}


def experiment(num: int, name: str, title: str):
    """Dekorator: registriert (run, plot)-Paar. plot wird via @run.plot gesetzt."""

    def deco(run_fn: Callable[[], dict]):
        EXPERIMENTS[num] = {"num": num, "name": name, "title": title, "run": run_fn, "plot": None}

        def set_plot(plot_fn):
            EXPERIMENTS[num]["plot"] = plot_fn
            return plot_fn

        run_fn.plot = set_plot
        return run_fn

    return deco


# ==========================================================================
# Experiment 1 – Leibniz-Reihe, Konvergenz und asymmetrische Zerlegung
# ==========================================================================
@experiment(1, "leibniz_basis", "Leibniz-Reihe: O(1/N)-Konvergenz und asymmetrische Zerlegung")
def exp_leibniz_basis(n_max: int = 10**6, n_points: int = 60) -> dict:
    S = leibniz_partial_sums(n_max)
    N = np.unique(np.logspace(0, math.log10(n_max), n_points).astype(int))
    err = S[N - 1] - PI4

    n = np.arange(n_max)
    S_odd = np.cumsum(1.0 / (4 * n + 1))  # Σ 1/(4n+1)
    S_even = np.cumsum(1.0 / (4 * n + 3))  # Σ 1/(4n+3)
    mask = N >= 100
    slope = np.polyfit(np.log(N[mask]), np.log(np.abs(err[mask])), 1)[0]

    return {
        "parameters": {"n_max": n_max},
        "N": N,
        "S_N": S[N - 1],
        "pi_over_4": PI4,
        "error_signed": err,
        "abs_error": np.abs(err),
        "leibniz_bound_1_over_2Np1": 1.0 / (2 * N + 1),
        "N_times_abs_error": N * np.abs(err),
        "asymptotic_constant_expected": 0.25,
        "loglog_slope_of_error": slope,
        "bound_respected": bool(np.all(np.abs(err) <= 1.0 / (2 * N + 1) + 1e-15)),
        "S_odd": S_odd[N - 1],
        "S_even": S_even[N - 1],
        "S_odd_minus_S_even": S_odd[N - 1] - S_even[N - 1],
        "quarter_ln_N": 0.25 * np.log(N),
        "S_odd_minus_quarter_lnN": S_odd[N - 1] - 0.25 * np.log(N),
    }


@exp_leibniz_basis.plot
def plot_leibniz_basis(d, fig):
    N, ax = A(d["N"]), fig.subplots(2, 2)
    a = ax[0, 0]
    a.semilogx(N, A(d["S_N"]), ".-", color=C[0], label="$S_N$")
    a.axhline(PI4, color="k", ls="--", lw=1, label=r"$\pi/4$")
    a.set(xlabel="Anzahl Terme N", ylabel="Partialsumme", title="Partialsummen der Leibniz-Reihe")
    a.legend()
    a = ax[0, 1]
    a.loglog(N, A(d["abs_error"]), ".-", color=C[0], label=r"$|S_N-\pi/4|$")
    a.loglog(N, A(d["leibniz_bound_1_over_2Np1"]), "--", color=C[3], label="Leibniz-Schranke $1/(2N+1)$")
    a.loglog(N, 1 / (4 * N), ":", color="k", label="$1/(4N)$")
    a.set(xlabel="N", ylabel="Fehler", title=f"Fehler: Steigung {d['loglog_slope_of_error']:.3f} (⇒ O(1/N))")
    a.legend()
    a = ax[1, 0]
    a.semilogx(N, A(d["N_times_abs_error"]), ".-", color=C[2])
    a.axhline(0.25, color="k", ls="--", lw=1, label="Grenzwert 1/4")
    a.set(xlabel="N", ylabel=r"$N\,|S_N-\pi/4|$", title="Konstante des führenden Fehlerterms")
    a.legend()
    a = ax[1, 1]
    a.semilogx(N, A(d["S_odd"]), color=C[1], label=r"$S_{odd}=\sum 1/(4n+1)$")
    a.semilogx(N, A(d["S_even"]), color=C[4], label=r"$S_{even}=\sum 1/(4n+3)$")
    a.semilogx(N, A(d["S_odd_minus_S_even"]), color="k", lw=2, label=r"$S_{odd}-S_{even}\to\pi/4$")
    a.semilogx(N, A(d["quarter_ln_N"]), ":", color="gray", label=r"$\frac{1}{4}\ln N$")
    a.set(xlabel="N", title="Asymmetrische Zerlegung: beide divergieren (∼¼ ln N), Differenz → π/4")
    a.legend(loc="center left")


# ==========================================================================
# Experiment 2 – Beschleunigungsverfahren
# ==========================================================================
@experiment(2, "beschleunigung", "Beschleunigung der Leibniz-Reihe: Verfahren im Vergleich")
def exp_beschleunigung(T_max: int = 45) -> dict:
    S = leibniz_partial_sums(T_max + 12)  # S[k-1] = k Terme
    methods: Dict[str, dict] = {}

    def add(key, label, terms, values, note=""):
        v = np.asarray(values, float)
        methods[key] = {
            "label": label,
            "note": note,
            "terms_used": np.asarray(terms),
            "value": v,
            "abs_error": np.abs(v - PI4),
        }

    T = np.arange(1, T_max + 1)
    add("naive", "naive Partialsumme", T, S[T - 1])

    # Euler: korrekt und Formel der Arbeit
    add("euler", "Euler-Transformation (korrekt)", T, euler_exact(T_max),
        "Σ 2^-(n+1) Σ_k (-1)^k C(n,k) a_k, exakt gerechnet")
    add("euler_paper", "Euler-Formel wie in Arbeit/Modul", T, euler_exact(T_max, True),
        "a_{n-k} statt a_k: fehlt Faktor (-1)^n -> falscher Grenzwert")

    # Richardson-Formel der Arbeit: ((2N+3)S_N − (2N+1)S_{N+1})/2, benötigt N+1 Terme
    Nn = np.arange(1, T_max)
    rich = ((2 * Nn + 3) * S[Nn - 1] - (2 * Nn + 1) * S[Nn]) / 2
    add("richardson_paper", "Richardson-Formel wie in Arbeit/Modul", Nn + 1, rich,
        "Vorzeichen der Fehlerterme alterniert -> Fehler ≈ 1/2")

    # Wiederholtes Mitteln (Euler-van-Wijngaarden)
    for lev in (1, 3, 6):
        R = repeated_average(S[: T_max + lev], lev)[:T_max - lev + 1 if lev else None]
        k = np.arange(len(R))
        add(f"avg{lev}", f"{lev}-faches Mitteln", k + 1 + lev, R)

    # Shanks (e1) und iteriert
    e1 = aitken(S[: T_max + 2])
    add("shanks1", "Shanks $e_1$ (Aitken)", np.arange(len(e1)) + 3, e1)
    e2 = aitken(e1)
    add("shanks2", "Shanks iteriert ($e_1\\circ e_1$)", np.arange(len(e2)) + 5, e2)

    # Empirische Konvergenzrate des Euler-Verfahrens (Faktor pro Term)
    eu = methods["euler"]
    sel = (eu["terms_used"] >= 8) & (eu["terms_used"] <= 30)
    slope = np.polyfit(eu["terms_used"][sel], np.log(eu["abs_error"][sel]), 1)[0]
    # Grenzwert der "Euler-Formel der Arbeit"
    lim_paper = float(methods["euler_paper"]["value"][-1])

    ref_T = 20
    at_T = {}
    for k, m in methods.items():
        idx = np.where(m["terms_used"] == ref_T)[0]
        at_T[k] = float(m["abs_error"][idx[0]]) if len(idx) else None

    return {
        "parameters": {"T_max": T_max, "reference_terms": ref_T},
        "pi_over_4": PI4,
        "methods": methods,
        "error_at_reference_terms": at_T,
        "euler_error_reduction_factor_per_term": math.exp(-slope),
        "euler_paper_formula_limit": lim_paper,
        "euler_paper_formula_limit_error": abs(lim_paper - PI4),
    }


@exp_beschleunigung.plot
def plot_beschleunigung(d, fig):
    ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    style = {
        "naive": (C[7], "-"), "euler": (C[0], "-"), "euler_paper": (BAD_COLOR, ":"),
        "richardson_paper": ("#e08e45", ":"), "avg1": (C[2], "--"), "avg3": (C[2], "-."),
        "avg6": (C[2], "-"), "shanks1": (C[4], "--"), "shanks2": (C[4], "-"),
    }
    a = ax[0]
    for k, m in d["methods"].items():
        col, ls = style[k]
        a.semilogy(A(m["terms_used"]), clip(m["abs_error"]), ls=ls, color=col, lw=1.8, label=m["label"])
    a.set(xlabel="benötigte Terme der Reihe", ylabel=r"$|\,\mathrm{Näherung}-\pi/4\,|$",
          ylim=(FLOOR, 2), title="Konvergenz der Verfahren (Maschinengenauigkeit ≈ 1e-16)")
    a.legend(ncol=1, loc="lower left")
    a = ax[1]
    ref = d["parameters"]["reference_terms"]
    keys = [k for k in d["methods"] if d["error_at_reference_terms"][k] is not None]
    vals = [max(d["error_at_reference_terms"][k], FLOOR) for k in keys]
    order = np.argsort(vals)[::-1]
    a.barh([d["methods"][keys[i]]["label"] for i in order], [vals[i] for i in order],
           color=[style[keys[i]][0] for i in order])
    a.set_xscale("log")
    a.set(xlabel="Fehler", title=f"Fehler bei {ref} Termen")
    a.tick_params(axis="y", labelsize=7.5)
    a.invert_yaxis()


# ==========================================================================
# Experiment 3 – Dirichlet-Betafunktion
# ==========================================================================
@experiment(3, "dirichlet_beta", "Dirichlet-Betafunktion β(s): Fortsetzung, Nullstellen, Sonderwerte")
def exp_dirichlet_beta() -> dict:
    s_grid = np.linspace(-6, 4, 1001)
    beta_curve = np.array([beta_ref(s) for s in s_grid])

    exact = {  # bekannte Werte (β(-2k) = E_{2k}/2 mit Euler-Zahlen)
        "0": 0.5, "1": PI4, "2": CATALAN, "3": PI**3 / 32, "5": 5 * PI**5 / 1536,
        "-1": 0.0, "-2": -0.5, "-3": 0.0, "-4": 2.5, "-5": 0.0, "-6": -30.5,
    }
    special_values = [
        {"s": int(k), "exact": v, "computed": beta_ref(int(k)), "abs_error": abs(beta_ref(int(k)) - v)}
        for k, v in exact.items()
    ]

    # Funktionalgleichung: korrekte Form vs. Form in Arbeit/Modul (Test im Streifen 0<s<1)
    s_fe = np.linspace(0.05, 0.95, 10)
    direct = np.array([beta_ref(s) for s in s_fe])
    fe_ok = np.array([(2 / PI) ** (1 - s) * math.cos(PI * s / 2) * special.gamma(1 - s) * beta_ref(1 - s) for s in s_fe])
    fe_paper = np.array([2.0**s * PI ** (s - 1) * math.sin(PI * s / 2) * special.gamma(1 - s) * beta_ref(1 - s) for s in s_fe])

    # Reihen-Konvergenz für s = 1,2,3
    n_max = 10**5
    n = np.arange(n_max)
    Ngrid = np.unique(np.logspace(0, 5, 40).astype(int))
    conv = {}
    for s in (1, 2, 3):
        P = np.cumsum((-1.0) ** n * (2 * n + 1.0) ** (-s))
        conv[str(s)] = np.abs(P[Ngrid - 1] - beta_ref(s))

    # β hat KEINE Pole: numerisch Residuentest lim ε·β(−1+ε)
    eps = 1e-6
    residue_test = {str(-(2 * k + 1)): eps * beta_ref(-(2 * k + 1) + eps) for k in range(3)}
    paper_res = DirichletBeta().poles_and_residues()

    return {
        "s_grid": s_grid, "beta": beta_curve, "special_values": special_values,
        "functional_equation": {
            "s": s_fe, "beta_direct": direct, "beta_via_correct_FE": fe_ok, "beta_via_paper_FE": fe_paper,
            "abs_residual_correct": np.abs(fe_ok - direct), "abs_residual_paper": np.abs(fe_paper - direct),
            "correct_form": "β(s) = (2/π)^{1-s} cos(πs/2) Γ(1-s) β(1-s)",
            "paper_form": "β(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) β(1-s)  (Riemann-Zeta-Form, gilt nicht für β)",
        },
        "series_convergence": {"N": Ngrid, **{f"abs_error_s{k}": v for k, v in conv.items()}},
        "pole_test": {
            "epsilon": eps,
            "numerical_eps_times_beta": residue_test,
            "paper_claimed_residues": {str(k): v for k, v in list(paper_res.items())[:3]},
            "conclusion": "ε·β(s0+ε) → 0: β ist eine ganze Funktion, s=-1,-3,-5,… sind Nullstellen, keine Pole.",
        },
    }


@exp_dirichlet_beta.plot
def plot_dirichlet_beta(d, fig):
    ax = fig.subplots(2, 2)
    a = ax[0, 0]
    a.plot(A(d["s_grid"]), A(d["beta"]), color=C[0], lw=2)
    zeros = [-1, -3, -5]
    a.plot(zeros, [0, 0, 0], "o", color=BAD_COLOR, label="β(−1)=β(−3)=β(−5)=0")
    a.plot([0, 1, 2, 3], [0.5, PI4, CATALAN, PI**3 / 32], "s", color=C[2], label="β(0), β(1)=π/4, β(2)=G, β(3)")
    a.axhline(0, color="k", lw=0.6)
    a.set(xlabel="s", ylabel="β(s)", ylim=(-40, 40), title="β(s) auf ℝ – keine Pole (ganze Funktion)")
    a.legend(loc="upper left")
    a = ax[0, 1]
    fe = d["functional_equation"]
    s = A(fe["s"])
    a.semilogy(s, clip(fe["abs_residual_correct"]), "o-", color=OK_COLOR, label="korrekte Form (cos, (2/π)^{1-s})")
    a.semilogy(s, clip(fe["abs_residual_paper"]), "s-", color=BAD_COLOR, label="Form der Arbeit (ζ-Funktionalgleichung)")
    a.set(xlabel="s", ylabel="|β(s) − β_FE(s)|", ylim=(FLOOR, 10), title="Test der Funktionalgleichung")
    a.legend(loc="center right")
    a = ax[1, 0]
    sc = d["series_convergence"]
    N = A(sc["N"])
    for i, s in enumerate((1, 2, 3)):
        a.loglog(N, clip(sc[f"abs_error_s{s}"]), ".-", color=C[i], label=f"s = {s}")
    a.set(xlabel="N Terme", ylabel="Abbruchfehler", title=r"Reihenkonvergenz: Fehler $\sim N^{-s}$")
    a.legend()
    a = ax[1, 1]
    sv = d["special_values"]
    labels = [f"β({v['s']})" for v in sv]
    a.bar(labels, clip([v["abs_error"] for v in sv]), color=OK_COLOR)
    a.set_yscale("log")
    a.set(ylim=(FLOOR, 1e-8), ylabel="|berechnet − exakt|", title="Sonderwerte (Euler-Zahlen, Catalan, π³/32 …)")
    a.tick_params(axis="x", rotation=60, labelsize=8)


# ==========================================================================
# Experiment 4 – Catalan-Konstante
# ==========================================================================
@experiment(4, "catalan", "Catalan-Konstante G: Integral- und Reihendarstellungen")
def exp_catalan() -> dict:
    def q(f, a, b):
        return integrate.quad(f, a, b, epsabs=1e-14, epsrel=1e-14, limit=400)[0]

    reps = [
        ("Reihe Σ(-1)^n/(2n+1)² (10⁶ Terme)", "korrekt",
         float(np.sum(np.where(np.arange(10**6) % 2 == 0, 1.0, -1.0) / (2 * np.arange(10**6) + 1.0) ** 2))),
        ("∫₀¹ arctan(x)/x dx", "korrekt", q(lambda x: math.atan(x) / x, 0, 1)),
        ("−∫₀¹ ln(t)/(1+t²) dt", "korrekt", q(lambda t: -math.log(t) / (1 + t * t), 0, 1)),
        ("½ ∫₀^{π/2} x/sin(x) dx", "korrekt", 0.5 * q(lambda x: x / math.sin(x), 0, PI / 2)),
        ("½ ∫₀¹ K(k) dk", "korrekt", 0.5 * q(lambda k: special.ellipk(k * k), 0, 1)),
        ("π/8·ln(2+√3) + 3/8·Σ 1/((2n+1)²C(2n,n))", "korrekt",
         PI / 8 * math.log(2 + math.sqrt(3)) + 3 / 8 * sum(1 / ((2 * n + 1) ** 2 * comb(2 * n, n)) for n in range(40))),
        # Formeln aus Arbeit/Modul
        ("Arbeit: ∫₀^{π/2} x/sin(x) dx  (ohne ½)", "Arbeit", DirichletBeta().catalan_sine_integral()),
        ("Arbeit: −∫₀¹ ln(1+x²)/(2x) dx", "Arbeit", DirichletBeta().catalan_logarithmic_integral()),
        ("Arbeit: ½B(½,½) − π/8", "Arbeit", BetaFunction().catalan_via_beta()),
    ]
    rows = [{"formula": f, "source": src, "value": v, "abs_error": abs(v - CATALAN), "matches_G": abs(v - CATALAN) < 1e-6}
            for f, src, v in reps]

    # Konvergenz: einfache Reihe vs. schnelle Reihe
    N = np.unique(np.logspace(0, 5, 40).astype(int))
    n = np.arange(10**5)
    P = np.cumsum(np.where(n % 2 == 0, 1.0, -1.0) / (2 * n + 1.0) ** 2)
    Nf = np.arange(1, 21)
    fast = np.array([PI / 8 * math.log(2 + math.sqrt(3)) + 3 / 8 * sum(1 / ((2 * k + 1) ** 2 * comb(2 * k, k)) for k in range(m)) for m in Nf])
    return {
        "catalan_reference": CATALAN, "representations": rows,
        "series_convergence": {"N": N, "abs_error": np.abs(P[N - 1] - CATALAN)},
        "fast_series_convergence": {"N": Nf, "abs_error": np.abs(fast - CATALAN)},
        "notes": {
            "sin_integral": "∫₀^{π/2} x/sin x dx = 2G (nicht G)",
            "log_integral": "∫₀¹ ln(1+x²)/(2x) dx = π²/48; damit ist −∫… = −π²/48 ≠ G",
            "beta_form": "½B(½,½) − π/8 = 3π/8 ≠ G",
        },
    }


@exp_catalan.plot
def plot_catalan(d, fig):
    ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    rows = d["representations"][::-1]
    a.barh([r["formula"] for r in rows], clip([r["abs_error"] for r in rows]),
           color=[OK_COLOR if r["matches_G"] else BAD_COLOR for r in rows])
    a.set_xscale("log")
    a.set(xlabel="|Wert − G|", xlim=(FLOOR, 10), title="Darstellungen von G (grün: korrekt, rot: Formel der Arbeit)")
    a.tick_params(axis="y", labelsize=7.5)
    a = ax[1]
    s1, s2 = d["series_convergence"], d["fast_series_convergence"]
    a.loglog(A(s1["N"]), clip(s1["abs_error"]), ".-", color=C[0], label=r"Reihe $\sum(-1)^n/(2n+1)^2$")
    a.loglog(A(s2["N"]), clip(s2["abs_error"]), "s-", color=C[2], label=r"schnelle Reihe (∼ $4^{-n}$)")
    a.loglog(A(s1["N"]), 1 / (16 * A(s1["N"]) ** 2), ":", color="k", label=r"$1/(16N^2)$")
    a.set(xlabel="Terme", ylabel="Fehler", ylim=(FLOOR, 1), title="Konvergenz")
    a.legend()


# ==========================================================================
# Experiment 5 – Parametrisierte Leibniz-Funktion L(t;λ)
# ==========================================================================
@experiment(5, "parametrisierte_leibniz", "Parametrisierte Leibniz-Funktion L(t;λ) = arctan(exp(-λt)) und ihre Differentialgleichung")
def exp_parametrisierte_leibniz(N: int = 400) -> dict:
    lams = [0.5, 1.0, 2.0, 5.0]
    curves = {}
    for lam in lams:
        t = np.linspace(0, 8 / lam, 400)
        Ls = L_series(t, lam, N)
        Lc = np.arctan(np.exp(-lam * t))
        m = lam * t >= 0.05  # bei t→0 ist die abgebrochene alternierende Reihe langsam
        curves[str(lam)] = {"t": t, "L_series": Ls, "L_closed_form": Lc,
                            "max_abs_diff_series_vs_arctan": float(np.max(np.abs(Ls - Lc)[m]))}

    # Residuen der Gleichungen (λ = 1)
    lam = 1.0
    tau = np.linspace(0.05, 8, 400)
    L, dL = L_series(tau, lam, N), dL_series(tau, lam, N)
    res_paper1 = dL + lam * L - lam * (PI4 - L)  # Arbeit, 1. Zeile
    res_paper2 = dL + lam * L - lam * (PI / 2 - L)  # Arbeit, 2. Zeile (λ[π/4 − ∫L']) ergibt π/2 − L
    res_exact = dL + lam / 2 * np.sin(2 * L)  # exakte ODE: L' = −(λ/2) sin 2L

    cross = None
    if ParametrizedLeibniz is not None:
        pts = [(0.3, 1.0), (1.0, 2.0), (2.0, 0.5), (0.5, 5.0)]
        cross = max(abs(ParametrizedLeibniz.L_odd(t, lm, 300) - float(L_series(t, lm, 300)[0])) for t, lm in pts)

    Lgrid = np.linspace(0.001, PI4, 200)
    return {
        "parameters": {"N_terms": N, "lambdas": lams},
        "curves": curves,
        "closed_form": "L(t;λ) = arctan(e^{-λt}); L' = -λ/(2cosh λt) = -(λ/2)·sin(2L); L(0)=π/4, L(∞)=0",
        "residuals_lambda1": {
            "t": tau,
            "paper_eq_line1": res_paper1, "paper_eq_line2": res_paper2, "exact_ode": res_exact,
            "max_abs_paper_eq_line1": float(np.max(np.abs(res_paper1))),
            "max_abs_paper_eq_line2": float(np.max(np.abs(res_paper2))),
            "max_abs_exact_ode": float(np.max(np.abs(res_exact))),
            "paper_equation": "L' + λL = λ(π/4 − L)  bzw.  = λ[π/4 − ∫₀ᵗ L' dτ] = λ(π/2 − L)",
        },
        "phase_plane": {"L": Lgrid, "dL_over_lambda": -0.5 * np.sin(2 * Lgrid)},
        "crosscheck_module_L_odd_max_diff": cross,
    }


@exp_parametrisierte_leibniz.plot
def plot_parametrisierte_leibniz(d, fig):
    ax = fig.subplots(2, 2)
    lams = d["parameters"]["lambdas"]
    a = ax[0, 0]
    for i, lam in enumerate(lams):
        c = d["curves"][str(lam)]
        a.plot(A(c["t"]), A(c["L_series"]), color=C[i], label=f"λ = {lam}")
    a.axhline(PI4, color="k", ls="--", lw=0.8)
    a.set(xlabel="t", ylabel="L(t;λ)", xlim=(0, 8), title="L(t;λ): L(0)=π/4, L(∞)=0")
    a.legend()
    a = ax[0, 1]
    for i, lam in enumerate(lams):
        c = d["curves"][str(lam)]
        a.plot(lam * A(c["t"]), A(c["L_series"]), color=C[i], lw=3 - 0.5 * i, alpha=0.7)
    x = np.linspace(0, 8, 300)
    a.plot(x, np.arctan(np.exp(-x)), "k--", label=r"$\arctan(e^{-\lambda t})$")
    a.set(xlabel="λt", ylabel="L", title="Datenkollaps: L hängt nur von λt ab")
    a.legend()
    a = ax[1, 0]
    r = d["residuals_lambda1"]
    t = A(r["t"])
    a.semilogy(t, clip(r["paper_eq_line1"]), color=C[3], label="Arbeit, Zeile 1:  L′+λL = λ(π/4−L)")
    a.semilogy(t, clip(r["paper_eq_line2"]), color=C[1], label="Arbeit, Zeile 2:  … = λ(π/2−L)")
    a.semilogy(t, clip(r["exact_ode"]), color=OK_COLOR, label="exakt:  L′ = −(λ/2)·sin 2L")
    a.set(xlabel="t (λ=1)", ylabel="|Residuum|", ylim=(FLOOR, 10), title="Residuum der Gleichungen (0 = erfüllt)")
    a.legend(loc="center right")
    a = ax[1, 1]
    pp = d["phase_plane"]
    a.plot(A(pp["L"]), A(pp["dL_over_lambda"]), "k-", label=r"$-\frac{1}{2}\sin 2L$")
    for i, lam in enumerate(lams):
        c = d["curves"][str(lam)]
        t = A(c["t"])[5:]
        a.plot(A(c["L_closed_form"])[5:], -1 / (2 * np.cosh(lam * t)), ".", ms=3, color=C[i], label=f"λ={lam}")
    a.set(xlabel="L", ylabel="L′/λ", title="Phasenraum: autonome ODE gilt für alle λ")
    a.legend(ncol=2)


# ==========================================================================
# Experiment 6 – e-Funktion: Ableitung der abgebrochenen Taylor-Reihe
# ==========================================================================
@experiment(6, "e_funktion_taylor", "Beobachtung 49: Ableitung der Taylor-Reihe verschiebt um ein Glied")
def exp_e_funktion_taylor(N_max: int = 60) -> dict:
    def taylor_cum(z: complex, N: int) -> np.ndarray:
        """T[N] = Σ_{n<N} z^n/n!, N = 0..N_max"""
        T, term, acc = np.zeros(N + 1, complex), 1.0 + 0j, 0j
        for n in range(N):
            acc += term
            T[n + 1] = acc
            term *= z / (n + 1)
        return T

    cases = [(1.0, 1.0), (1.0, 3.0), (1.0, 10.0), (-2.0, 3.0), (3j, 2.0)]
    out = []
    for s, t in cases:
        z = s * t
        T = taylor_cum(z, N_max)
        exact = np.exp(z)
        err_T = np.abs(exact - T)
        # Ableitung der N-gliedrigen Reihe = s·T_{N-1}  (explizit gegengeprüft)
        dT_explicit = []
        for Nn in (2, 5, 10, 20):
            coeffs = np.array([s**n / math.factorial(n) for n in range(Nn)], complex)
            dT_explicit.append(abs(sum(n * coeffs[n] * t ** (n - 1) for n in range(1, Nn)) - s * T[Nn - 1]))
        err_dT = np.abs(s * exact - s * np.concatenate([[0], T[:-1]]))  # index N: Fehler von d/dt T_N
        out.append({"s": s, "t": t, "N": np.arange(1, N_max + 1), "err_T": err_T[1:], "err_dT": err_dT[1:],
                    "identity_check_max_diff": float(max(dT_explicit)),
                    "ratio_err_dT_over_err_T": err_dT[1:] / np.maximum(err_T[1:], 1e-300)})

    # gleichmäßige Konvergenz auf [0,5]
    tt = np.linspace(0, 5, 501)
    sup_T, sup_dT = [], []
    for Nn in range(1, N_max + 1):
        # T_N(t) und (T_N)'(t) = T_{N-1}(t) für s=1
        n = np.arange(Nn)[:, None]
        TN = np.sum(tt[None, :] ** n / special.factorial(n), axis=0)
        TNm1 = np.sum(tt[None, :] ** n[:-1] / special.factorial(n[:-1]), axis=0) if Nn > 1 else np.zeros_like(tt)
        sup_T.append(float(np.max(np.abs(np.exp(tt) - TN))))
        sup_dT.append(float(np.max(np.abs(np.exp(tt) - TNm1))))
    return {"parameters": {"N_max": N_max}, "cases": out,
            "uniform_convergence_on_0_5": {"N": np.arange(1, N_max + 1), "sup_err_T": sup_T, "sup_err_dT": sup_dT},
            "statement": "d/dt T_N = s·T_{N-1}: die Ableitung kostet genau ein Glied; für N→∞ verschwindet der Fehler beider gleichmäßig auf kompakten Intervallen."}


@exp_e_funktion_taylor.plot
def plot_e_funktion_taylor(d, fig):
    ax = fig.subplots(1, 3)
    a = ax[0]
    for i, c in enumerate(d["cases"][:3]):
        N = A(c["N"])
        a.semilogy(N, clip(c["err_T"]), color=C[i], label=f"$T_N$, s=1, t={c['t']:g}")
        a.semilogy(N, clip(c["err_dT"]), "--", color=C[i])
    a.set(xlabel="N Glieder", ylabel="Fehler", ylim=(1e-16, 1e6),
          title=r"Fehler von $T_N$ (—) und $T_N'$ (- -)")
    a.legend(loc="lower left")
    a = ax[1]
    for i, c in enumerate(d["cases"][:3]):
        N = A(c["N"])[:40]
        ratio = A(c["ratio_err_dT_over_err_T"])
        ratio[A(c["err_T"]) < 1e-9] = np.nan  # unterhalb der Rundungsgrenze sinnlos
        a.semilogy(N, ratio[:40], color=C[i], label=f"t={c['t']:g}")
        a.semilogy(N, N / c["t"], ":", color=C[i])
    a.set(xlabel="N", ylabel=r"$|\mathrm{Fehler}(T_N')|/|\mathrm{Fehler}(T_N)|$", title="Ableitung kostet ein Glied  (Verhältnis ≈ N/t, ···)")
    a.legend()
    a = ax[2]
    u = d["uniform_convergence_on_0_5"]
    a.semilogy(A(u["N"]), clip(u["sup_err_T"]), color=C[0], label=r"$\sup|e^t-T_N|$")
    a.semilogy(A(u["N"]), clip(u["sup_err_dT"]), "--", color=C[3], label=r"$\sup|e^t-T_N'|$")
    a.set(xlabel="N", ylim=(1e-16, 1e3), title="Gleichmäßige Konvergenz auf [0, 5]")
    a.legend()


# ==========================================================================
# Experiment 7 – RC-Tiefpass
# ==========================================================================
@experiment(7, "rc_tiefpass", "RC-Tiefpass: Frequenzgang, Sprung- und Sinusantwort")
def exp_rc_tiefpass(tau: float = 1.0) -> dict:
    rc = RCLowpassFilter(tau=tau)
    w = np.logspace(-2, 2, 400) / tau
    t = np.linspace(0, 8 * tau, 800)
    step_num = rc.solve_ode(lambda tt: 1.0, t, 0.0)
    step_ana = rc.analytical_solution_step(1.0, t)

    wc = rc.omega_cutoff
    t2 = np.linspace(0, 12 * tau, 1200)
    y_num = rc.solve_ode(lambda tt: math.sin(wc * tt), t2, 0.0)
    wt = wc * tau
    y_ana = (np.sin(wc * t2) - wt * np.cos(wc * t2) + wt * np.exp(-t2 / tau)) / (1 + wt**2)
    y_ss = np.abs(rc.frequency_response(wc)) * np.sin(wc * t2 + rc.phase_response(wc))

    return {
        "parameters": {"tau": tau},
        "metrics": {
            "cutoff_frequency_Hz": rc.cutoff_frequency, "omega_c": wc,
            "|H(jω_c)|": float(rc.magnitude_response(wc)), "attenuation_dB_at_omega_c": float(rc.attenuation_db(wc)),
            "phase_deg_at_omega_c": float(rc.phase_response(wc) * 180 / PI),
            "step_at_tau": float(rc.step_response(np.array([tau]))[0]),
            "step_at_3tau": float(rc.step_response(np.array([3 * tau]))[0]),
            "step_at_4.6tau": float(rc.step_response(np.array([rc.settling_time_99()]))[0]),
            "ode_vs_analytic_max_abs_step": float(np.max(np.abs(step_num - step_ana))),
            "ode_vs_analytic_max_abs_sine": float(np.max(np.abs(y_num - y_ana))),
        },
        "bode": {"omega": w, "magnitude_dB": rc.attenuation_db(w), "phase_deg": rc.phase_response(w) * 180 / PI,
                 "asymptote_dB": -10 * np.log10(1 + (w * tau) ** 2)},
        "step": {"t": t, "numeric": step_num, "analytic": step_ana},
        "sine_at_omega_c": {"t": t2, "input": np.sin(wc * t2), "numeric": y_num, "analytic": y_ana, "steady_state": y_ss},
    }


@exp_rc_tiefpass.plot
def plot_rc_tiefpass(d, fig):
    ax = fig.subplots(2, 2)
    tau, wc = d["parameters"]["tau"], d["metrics"]["omega_c"]
    b = d["bode"]
    w = A(b["omega"])
    a = ax[0, 0]
    a.semilogx(w, A(b["magnitude_dB"]), color=C[0], lw=2, label="|H(jω)|")
    a.semilogx(w, -20 * np.log10(np.maximum(w * tau, 1e-9)), ":", color="gray", label="−20 dB/Dekade")
    a.plot([wc], [-3.0103], "o", color=C[3], label="−3 dB bei ω_c = 1/τ")
    a.set(xlabel="ω [rad/s]", ylabel="Betrag [dB]", ylim=(-42, 3), title="Bode: Amplitudengang")
    a.legend()
    a = ax[0, 1]
    a.semilogx(w, A(b["phase_deg"]), color=C[1], lw=2)
    a.plot([wc], [-45], "o", color=C[3])
    a.set(xlabel="ω [rad/s]", ylabel="Phase [°]", title="Bode: Phasengang (−45° bei ω_c)")
    a = ax[1, 0]
    s = d["step"]
    t = A(s["t"]) / tau
    a.plot(t, A(s["analytic"]), color=C[0], lw=2, label=r"$1-e^{-t/\tau}$")
    a.plot(t[::25], A(s["numeric"])[::25], "o", ms=4, color=C[3], label="ODE-Solver")
    for k, lab in ((1, "63,2 %"), (3, "95,0 %"), (4.605, "99 %")):
        a.axvline(k, color="gray", ls=":", lw=0.8)
        a.text(k, 0.05, lab, rotation=90, va="bottom", ha="right", fontsize=8)
    a.set(xlabel="t/τ", ylabel="V_out/V_in", title="Sprungantwort")
    a.legend(loc="center right")
    a = ax[1, 1]
    q = d["sine_at_omega_c"]
    t = A(q["t"]) / tau
    a.plot(t, A(q["input"]), color="gray", lw=1, label="Eingang sin(ω_c t)")
    a.plot(t, A(q["analytic"]), color=C[0], lw=2, label="Ausgang (analytisch)")
    a.plot(t[::30], A(q["numeric"])[::30], "o", ms=3.5, color=C[3], label="ODE-Solver")
    a.plot(t, A(q["steady_state"]), ":", color="k", label="eingeschwungen: 0,707·sin(ω_c t − 45°)")
    a.set(xlabel="t/τ", title="Sinusantwort bei ω = ω_c (Transiente + stationär)")
    a.legend(loc="lower right", fontsize=7.5)


# ==========================================================================
# Experiment 8 – RLC
# ==========================================================================
@experiment(8, "rlc_daempfung", "RLC-Kreis: Sprungantworten und Überschwingen vs. Dämpfung ζ")
def exp_rlc_daempfung() -> dict:
    from scipy.integrate import odeint

    def step_module(f: RLCFilter, t):
        if f.is_underdamped:
            return f.step_response_underdamped(t)
        if f.is_critically_damped:
            return f.step_response_critically_damped(t)
        return f.step_response_overdamped(t)

    t = np.linspace(0, 20, 2000)
    cases = []
    for zeta in (0.1, 0.3, 0.7, 1.0, 2.0):
        f = RLCFilter(L=1.0, R=2.0 * zeta, C=1.0)  # ω0 = 1, ζ = R/2
        ana = step_module(f, t)
        w0 = f.omega_0
        num = odeint(lambda y, _t: [y[1], w0**2 * (1 - y[0]) - 2 * zeta * w0 * y[1]], [0, 0], t, rtol=1e-11, atol=1e-13)[:, 0]
        cases.append({"zeta": zeta, "t": t, "analytic": ana, "numeric": num,
                      "max_abs_diff": float(np.max(np.abs(ana - num)))})

    zs = np.linspace(0.05, 0.95, 46)
    tt = np.linspace(0, 200, 40001)
    meas = [float(np.max(RLCFilter(1.0, 2 * z, 1.0).step_response_underdamped(tt)) - 1) for z in zs]
    theo = np.exp(-PI * zs / np.sqrt(1 - zs**2))
    return {"parameters": {"L": 1.0, "C": 1.0, "R=2ζ": True}, "cases": cases,
            "overshoot": {"zeta": zs, "measured": meas, "theory_exp(-pi*zeta/sqrt(1-zeta^2))": theo,
                          "max_abs_diff": float(np.max(np.abs(np.array(meas) - theo)))}}


@exp_rlc_daempfung.plot
def plot_rlc_daempfung(d, fig):
    ax = fig.subplots(1, 2)
    a = ax[0]
    for i, c in enumerate(d["cases"]):
        a.plot(A(c["t"]), A(c["analytic"]), color=C[i], lw=2, label=f"ζ = {c['zeta']}")
        a.plot(A(c["t"])[::80], A(c["numeric"])[::80], "o", ms=3, color="k")
    a.axhline(1, color="gray", ls=":")
    a.plot([], [], "ko", ms=3, label="ODE-Solver")
    a.set(xlabel=r"$\omega_0 t$", ylabel="Sprungantwort", title="unter-, kritisch und überdämpft")
    a.legend()
    a = ax[1]
    o = d["overshoot"]
    a.plot(A(o["zeta"]), A(o["theory_exp(-pi*zeta/sqrt(1-zeta^2))"]), "-", color=C[0], lw=2, label=r"$e^{-\pi\zeta/\sqrt{1-\zeta^2}}$")
    a.plot(A(o["zeta"])[::2], A(o["measured"])[::2], "o", color=C[3], ms=4, label="gemessen (Modul)")
    a.set(xlabel="ζ", ylabel="relatives Überschwingen", title="Überschwingen gegen Dämpfung")
    a.legend()


# ==========================================================================
# Experiment 9 – Formel-Audit
# ==========================================================================
@experiment(9, "formel_audit", "Formel-Audit: numerische Prüfung der Aussagen in Arbeit und Modulen")
def exp_formel_audit() -> dict:
    items: List[dict] = []

    def add(idn, source, text, value, ref, comment="", tol=1e-6):
        dev = abs(value - ref)
        rel = dev / abs(ref) if abs(ref) > 1e-12 else dev
        items.append({"id": idn, "source": source, "claim": text, "claimed_value": value, "reference": ref,
                      "abs_deviation": dev, "rel_deviation": rel, "ok": bool(rel < tol), "comment": comment})

    N = 40
    S = leibniz_partial_sums(N + 2)
    add("E1", "Arbeit V / Modul", "Euler-Formel (a_{n-k}), N=40", EulerTransformation().euler_transform_sum(N), PI4,
        "Faktor (-1)^n fehlt; zusätzlich steht in der Arbeit ein weiterer Faktor 1/2 vor der Summe")
    add("E2", "korrigiert", "Euler-Transformation mit a_k, N=40", float(euler_exact(N)[-1]), PI4, "O(2^-N) bestätigt", tol=1e-11)
    add("R1", "Arbeit V / Modul", "Richardson ((2N+3)S_N−(2N+1)S_{N+1})/2, N=40",
        RichardsonExtrapolation().richardson_extrapolation_2terms(S[N - 1], S[N], N), PI4,
        "Fehlerterme haben wechselndes Vorzeichen -> Fehler ≈ 1/2; sinnvoll wäre (S_N+S_{N+1})/2 (O(1/N²))")
    add("R2", "korrigiert", "Mittel (S_N+S_{N+1})/2, N=40", (S[N - 1] + S[N]) / 2, PI4, "Fehler O(1/N²)", tol=1e-4)
    dbeta = DirichletBeta()
    add("C1", "Arbeit I / Modul", "G = −∫₀¹ ln(1+x²)/(2x) dx", dbeta.catalan_logarithmic_integral(), CATALAN, "Integral = π²/48")
    add("C2", "Arbeit I / Modul", "G = ∫₀^{π/2} x/sin x dx", dbeta.catalan_sine_integral(), CATALAN, "Integral = 2G")
    add("C3", "Arbeit I / Modul", "G = ½B(½,½) − π/8", BetaFunction().catalan_via_beta(), CATALAN, "ergibt 3π/8")
    ir = IntegralRepresentations()
    add("P1", "Arbeit IV / Modul", "π/4 = ∫₀¹ 1/√(1−x⁴) dx", ir.pi_over_4_sine_integral(), PI4, "Lemniskaten-Integral ≈ 1,311")
    add("P2", "Arbeit IV / Modul", "π/4 = ∫₀¹ arcsin(x)/x dx", ir.pi_over_4_arcsin_integral(), PI4, "Integral = (π/2)·ln 2")
    add("P3", "Arbeit IV / Modul", "π/4 = ∫₀¹ ln(1+x²)/(2x) dx", ir.pi_over_4_log_integral(), PI4, "Integral = π²/48")
    add("P4", "Modul", "π/4 = ∫₀¹ √(1−x²) dx", ir.pi_over_4_geometric(), PI4, "korrekt")
    add("P5", "Modul", "G = ∫₀¹ arctan(x)/x dx", ir.catalan_arctan_power_integral(), CATALAN, "korrekt")
    s = 2.0
    zeta, eta = float(special.zeta(s)), float((1 - 2 ** (1 - s)) * special.zeta(s))
    add("B1", "Arbeit I / Modul", "β(s) = (ζ(s)+η(s))/2^{s+1}, s=2", (zeta + eta) / 2 ** (s + 1), CATALAN, "β ist nicht durch ζ allein darstellbar")
    s = 0.25
    fe_paper = 2**s * PI ** (s - 1) * math.sin(PI * s / 2) * float(special.gamma(1 - s)) * beta_ref(1 - s)
    add("B2", "Arbeit I / Modul", "Funktionalgleichung (ζ-Form), s=1/4", fe_paper, beta_ref(s), "korrekt: (2/π)^{1-s} cos(πs/2) Γ(1-s) β(1-s)")
    fe_ok = (2 / PI) ** (1 - s) * math.cos(PI * s / 2) * float(special.gamma(1 - s)) * beta_ref(1 - s)
    add("B3", "korrigiert", "Funktionalgleichung (β-Form), s=1/4", fe_ok, beta_ref(s), "", tol=1e-12)
    eps = 1e-8
    add("B4", "Arbeit I / Modul", "Residuum von β bei s=−1 (Arbeit: 1/4)", DirichletBeta().poles_and_residues()[-1],
        eps * beta_ref(-1 + eps), "β ist ganz: s=−1 ist eine Nullstelle, kein Pol", tol=1e-3)
    lam, t = 1.0, np.array([0.5])
    L, dL = float(L_series(t, lam)[0]), float(dL_series(t, lam)[0])
    add("D1", "lowp.tex", "L′+λL = λ(π/4−L), t=0,5 (Residuum)", dL + lam * L, lam * (PI4 - L), "Gleichung wird nicht erfüllt")
    add("D2", "korrigiert", "L′ = −(λ/2) sin 2L, t=0,5", dL, -lam / 2 * math.sin(2 * L), "exakte autonome ODE", tol=1e-12)
    eu = euler_exact(30)
    err = np.abs(eu - PI4)
    factor = float(err[14] / err[15])
    items.append({"id": "E3", "source": "Arbeit V", "claim": "Euler-Fehler ~ 2^{-N}: Fehlerverhältnis pro Term ≈ 2",
                  "claimed_value": 2.0, "reference": factor, "abs_deviation": abs(factor - 2), "rel_deviation": abs(factor - 2) / 2,
                  "ok": bool(1.7 < factor < 2.3), "comment": "Rate stimmt (mit korrekter Formel)"})
    return {"tolerance_ok": "relative Abweichung < 1e-6 (falls nicht anders angegeben)", "items": items,
            "summary": {"n_items": len(items), "n_ok": sum(i["ok"] for i in items), "n_fail": sum(not i["ok"] for i in items)}}


@exp_formel_audit.plot
def plot_formel_audit(d, fig):
    ax = fig.subplots(1, 1)
    items = d["items"][::-1]
    dev = np.maximum(A([i["rel_deviation"] for i in items]), FLOOR)
    labels = [f"{i['id']}  {i['claim']}" for i in items]
    ax.barh(labels, dev, color=[OK_COLOR if i["ok"] else BAD_COLOR for i in items])
    ax.set_xscale("log")
    ax.axvline(1e-6, color="k", ls="--", lw=0.8)
    ax.set(xlabel="relative Abweichung vom Referenzwert (log)", xlim=(FLOOR / 10, 100),
           title=f"{d['summary']['n_ok']} von {d['summary']['n_items']} Aussagen numerisch bestätigt (grün), "
                 f"{d['summary']['n_fail']} nicht (rot)")
    ax.tick_params(axis="y", labelsize=7.5)


# ==========================================================================
# Runner
# ==========================================================================
def run_experiment(num: int, outdir: Path) -> dict:
    exp = EXPERIMENTS[num]
    stem = f"exp{num:02d}_{exp['name']}"
    t0 = time.perf_counter()
    data = exp["run"]()
    runtime = time.perf_counter() - t0

    payload = _clean({
        "experiment": {"number": num, "name": exp["name"], "title": exp["title"]},
        "meta": {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(runtime, 3),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "data": data,
    })
    json_path, pdf_path = outdir / f"{stem}.json", outdir / f"{stem}.pdf"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    # Plot aus den zurückgelesenen JSON-Daten
    with open(json_path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    fig = plt.figure(figsize=(13, 8.5) if num != 9 else (13, 7.5), layout="constrained")
    exp["plot"](loaded["data"], fig)
    fig.suptitle(f"Experiment {num}: {exp['title']}", fontsize=13, fontweight="bold")
    fig.savefig(pdf_path, format="pdf", metadata={"Title": f"Experiment {num}: {exp['title']}"})
    plt.close(fig)
    return {"number": num, "name": exp["name"], "title": exp["title"], "json": json_path.name,
            "pdf": pdf_path.name, "runtime_seconds": round(runtime, 3)}


def _resolve(selectors: Optional[List[str]]) -> List[int]:
    if not selectors:
        return sorted(EXPERIMENTS)
    nums = []
    for s in selectors:
        if s.isdigit() and int(s) in EXPERIMENTS:
            nums.append(int(s))
        else:
            hit = [n for n, e in EXPERIMENTS.items() if e["name"] == s]
            if not hit:
                raise SystemExit(f"Unbekanntes Experiment: {s!r}  (siehe --list)")
            nums.append(hit[0])
    return sorted(set(nums))


def main(argv=None):
    p = argparse.ArgumentParser(description="Leibniz-Experimente: JSON-Daten + PDF-Plots")
    p.add_argument("--outdir", default="experiment_results", help="Ausgabeverzeichnis (Standard: experiment_results)")
    p.add_argument("--only", nargs="*", help="Nummern oder Namen der auszuführenden Experimente")
    p.add_argument("--list", action="store_true", help="Experimente auflisten und beenden")
    args = p.parse_args(argv)

    if args.list:
        for n in sorted(EXPERIMENTS):
            print(f"{n:2d}  {EXPERIMENTS[n]['name']:26s} {EXPERIMENTS[n]['title']}")
        return 0

    outdir = Path(args.outdir)
    outdir = outdir if args.outdir != "experiment_results" else Path(__file__).parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for n in _resolve(args.only):
        info = run_experiment(n, outdir)
        summary.append(info)
        print(f"[{n}] {info['title']}\n     -> {info['json']}, {info['pdf']}  ({info['runtime_seconds']} s)")
    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(_clean({"created": _dt.datetime.now().isoformat(timespec="seconds"), "experiments": summary}),
                  fh, ensure_ascii=False, indent=1)
    print(f"\nFertig. Ergebnisse in: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
