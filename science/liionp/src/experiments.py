#!/usr/bin/env python3
"""Umfangreiches Experiment-Modul für das liionp-Paket.

Führt 15 thematische Experimente durch und erzeugt je einen SVG-Plot mit
zwei Subplots sowie eine oder mehrere CSV-Ergebnistabellen.
Alle Ausgabedateien werden in ``src/results/`` gespeichert.

Experimente
-----------
01  Arrhenius-Ratenverhältnis & SEI-Dickenwachstum
02  Spannungsbedingte Degradationsraten (Über-/Unterspannung)
03  Thermisches Zellmodell – Temperaturverlauf & Wärmegeneration
04  Thermischer Verbesserungsfaktor Φ_T
05  Spannungs- & kombinierter Verbesserungsfaktor
06  Reichweite vs. Zyklen & kältebedingte Kapazitätsreduktion
07  Mechanikmodell: Quellung, Tortuosität & mechanische Degradation
08  Erweiterter Verbesserungsfaktor Φ_ges (mit Plattenabstandskanal)
09  Zeitseriensimulation LP1 (Smartphone)
10  Zeitseriensimulation LP2 (E-Bike)
11  Zeitseriensimulation LP3 (Stationärspeicher)
12  Lastprofilvergleich LP1/LP2/LP3
13  SEI-Kapazitätsverlust & Kapazitätsdrift D(t)
14  Lebensdauer-Sensitivität & Vorheizeffizienz
15  E-Fahrzeug-Lebenszyklus & Plattenabstands-MPC-Referenztrajektorie

Aufruf (vom Projektstammverzeichnis):
    python src/experiments.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kein interaktives Fenster erforderlich
import matplotlib.pyplot as plt
import numpy as np

# Paket aus src/ einbinden (Fallback, falls nicht im PYTHONPATH)
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from liionp import (
    # Degradation
    arrhenius_ratio,
    capacity_drift,
    capacity_loss_sei,
    cold_capacity,
    cumulative_degradation,
    ev_cumulative_km,
    ev_lifetime_years,
    improvement_factor_combined,
    improvement_factor_thermal,
    improvement_factor_total,
    improvement_factor_voltage,
    lifetime_from_rate,
    mechanical_degradation_rate,
    mechanical_reduction_factor,
    overvoltage_degradation_rate,
    phi_factor_from_rates,
    # Zellmodell
    CellModel,
    CellParameters,
    CellState,
    # Mechanik
    PlateDistanceMPCController,
    preheating_efficiency,
    range_vs_cycles,
    sei_thickness,
    swelling_strain,
    tortuosity,
    undervoltage_degradation_rate,
    # Simulation
    LoadProfile,
    ThermalModel,
    compare_systems,
    # Konstanten
    D_EOL,
    E_A_SEI_DEFAULT,
    K_SEI_DEFAULT,
    Q0_DEFAULT,
    V_MAX_DEFAULT,
    V_MIN_DEFAULT,
)
from liionp.constants import (
    ALPHA_SEI_DEFAULT,
    D_SEP_0,
    D_SEP_MAX,
    D_SEP_MIN,
    GAMMA_T_DEFAULT,
    GAMMA_V_DEFAULT,
    KAPPA_T,
    M_MECH_DEFAULT,
    T_REF_COLD,
)

# ── Ergebnisverzeichnis ───────────────────────────────────────────────────────
RESULTS: Path = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── Matplotlib-Globaleinstellungen ────────────────────────────────────────────
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "figure.constrained_layout.use": True,
        "figure.dpi": 120,
    }
)

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _save_csv(filename: str, header: list[str], rows: list[list]) -> None:
    """Schreibt *rows* als CSV-Datei nach RESULTS/filename."""
    path = RESULTS / filename
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"    CSV → {path.name}")


def _save_svg(fig: plt.Figure, name: str) -> None:
    """Speichert *fig* als SVG und schließt die Figur."""
    path = RESULTS / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"    SVG → {path.name}")


def _fig2(title: str, figsize: tuple[float, float] = (13, 5)) -> tuple:
    """Erzeugt eine neue Figure mit zwei nebeneinander liegenden Subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig, ax1, ax2


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 01 – Arrhenius-Ratenverhältnis & SEI-Dickenwachstum
# ═══════════════════════════════════════════════════════════════════════════════

def exp01_arrhenius_sei() -> None:
    """Arrhenius-Ratenverhältnis k(T)/k(T₀) und SEI-Schichtdickenwachstum δ(t)."""
    T_range = np.linspace(263.0, 348.0, 400)          # −10 … +75 °C
    T0 = 298.0
    Ea_vals = [30_000.0, 50_000.0, 70_000.0, 90_000.0]
    t_arr = np.linspace(0.0, 3.15e7, 600)              # 0 … 1 Jahr [s]
    T_sei = [278.0, 288.0, 298.0, 308.0, 318.0]

    fig, ax1, ax2 = _fig2("Exp 01 – Arrhenius-Beschleunigung & SEI-Wachstum")

    # Subplot 1: Ratenverhältnis vs. Temperatur
    csv_rows: list[list] = []
    for Ea, col in zip(Ea_vals, _COLORS):
        ratios = np.array([arrhenius_ratio(Ea, T0, T) for T in T_range])
        ax1.plot(T_range - 273.15, ratios, color=col, label=f"Eₐ = {Ea/1000:.0f} kJ/mol")
        for T, ratio in zip(T_range[::20], ratios[::20]):
            csv_rows.append([round(Ea, 0), round(T - 273.15, 2), round(float(ratio), 8)])
    ax1.axvline(T0 - 273.15, ls="--", color="gray", lw=0.9, label=f"T₀ = {T0-273:.0f} °C")
    ax1.set_yscale("log")
    ax1.set_xlabel("Temperatur [°C]")
    ax1.set_ylabel("k(T) / k(T₀)  [log]")
    ax1.set_title("Arrhenius-Ratenverhältnis k(T)/k(T₀)")
    ax1.legend()

    # Subplot 2: SEI-Dicke vs. Zeit bei verschiedenen Temperaturen
    csv_sei: list[list] = []
    for T_s, col in zip(T_sei, _COLORS):
        delta = np.array([sei_thickness(K_SEI_DEFAULT, t, E_A_SEI_DEFAULT, T_s) for t in t_arr])
        ax2.plot(t_arr / 86400.0, delta * 1e9, color=col, label=f"T = {T_s-273:.0f} °C")
        for t, d in zip(t_arr[::30], delta[::30]):
            csv_sei.append([round(T_s - 273.0, 1), round(t / 86400.0, 3), round(float(d) * 1e9, 6)])
    ax2.set_xlabel("Zeit [Tage]")
    ax2.set_ylabel("SEI-Dicke δ [nm]")
    ax2.set_title("SEI-Schichtdickenwachstum δ(t)")
    ax2.legend()

    _save_svg(fig, "exp01_arrhenius_sei.svg")
    _save_csv("exp01_arrhenius_ratio.csv", ["Ea_J_mol", "T_degC", "k_ratio"], csv_rows)
    _save_csv("exp01_sei_thickness_nm.csv", ["T_degC", "t_days", "delta_nm"], csv_sei)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 02 – Spannungsbedingte Degradationsraten
# ═══════════════════════════════════════════════════════════════════════════════

def exp02_voltage_degradation() -> None:
    """Über- und Unterspannungs-Degradationsraten für verschiedene Schärfefaktoren β."""
    V_over = np.linspace(V_MAX_DEFAULT - 0.6, V_MAX_DEFAULT + 0.6, 400)
    V_under = np.linspace(V_MIN_DEFAULT - 0.6, V_MIN_DEFAULT + 0.6, 400)
    beta_vals = [5.0, 10.0, 20.0, 40.0]

    fig, ax1, ax2 = _fig2("Exp 02 – Spannungsbedingte Degradationsraten")

    csv_ov: list[list] = []
    for beta, col in zip(beta_vals, _COLORS):
        r = [overvoltage_degradation_rate(V, V_MAX_DEFAULT, k_OV=0.01, beta_OV=beta) for V in V_over]
        ax1.plot(V_over, r, color=col, label=f"β_OV = {beta:.0f} V⁻¹")
        csv_ov.extend([[round(beta, 0), round(float(V), 4), round(float(ri), 10)]
                       for V, ri in zip(V_over[::20], r[::20])])
    ax1.axvline(V_MAX_DEFAULT, ls="--", color="red", lw=0.9, label=f"V_max = {V_MAX_DEFAULT} V")
    ax1.set_xlabel("Zellspannung V [V]")
    ax1.set_ylabel("Degradationsrate r_OV")
    ax1.set_title(f"Überspannungsdegradation  (V_max = {V_MAX_DEFAULT} V)")
    ax1.set_yscale("symlog", linthresh=1e-5)
    ax1.legend()

    csv_uv: list[list] = []
    for beta, col in zip(beta_vals, _COLORS):
        r = [undervoltage_degradation_rate(V, V_MIN_DEFAULT, k_UV=0.01, beta_UV=beta) for V in V_under]
        ax2.plot(V_under, r, color=col, label=f"β_UV = {beta:.0f} V⁻¹")
        csv_uv.extend([[round(beta, 0), round(float(V), 4), round(float(ri), 10)]
                       for V, ri in zip(V_under[::20], r[::20])])
    ax2.axvline(V_MIN_DEFAULT, ls="--", color="red", lw=0.9, label=f"V_min = {V_MIN_DEFAULT} V")
    ax2.set_xlabel("Zellspannung V [V]")
    ax2.set_ylabel("Degradationsrate r_UV")
    ax2.set_title(f"Unterspannungsdegradation  (V_min = {V_MIN_DEFAULT} V)")
    ax2.set_yscale("symlog", linthresh=1e-5)
    ax2.legend()

    _save_svg(fig, "exp02_voltage_degradation.svg")
    _save_csv("exp02_overvoltage_rate.csv", ["beta_OV_1_V", "V_V", "rate"], csv_ov)
    _save_csv("exp02_undervoltage_rate.csv", ["beta_UV_1_V", "V_V", "rate"], csv_uv)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 03 – Thermisches Zellmodell
# ═══════════════════════════════════════════════════════════════════════════════

def exp03_thermal() -> None:
    """Temperaturverlauf (Euler-Integration) und Wärmegeneration Q̇(I) beim thermischen Modell."""
    model = ThermalModel()
    dt = 1.0
    t_end = 3600.0
    t_steps = np.arange(0.0, t_end + dt, dt)
    T_amb = 298.0
    R_int = 0.05
    I_vals = [0.5, 1.0, 2.0, 3.0, 5.0]

    fig, ax1, ax2 = _fig2("Exp 03 – Thermisches Zellmodell")

    # Subplot 1: Temperaturverlauf bei verschiedenen Strömen
    csv_temp: list[list] = []
    for I_cur, col in zip(I_vals, _COLORS):
        T_cur = T_amb
        temps = [T_cur]
        for _ in t_steps[1:]:
            T_cur = model.step(T_cur, T_amb, I_cur, R_int, dt)
            temps.append(T_cur)
        temps_arr = np.array(temps)
        ax1.plot(t_steps / 60.0, temps_arr - 273.15, color=col, label=f"I = {I_cur:.1f} A")
        for t, T in zip(t_steps[::60], temps_arr[::60]):
            csv_temp.append([round(I_cur, 2), round(t, 0), round(float(T) - 273.15, 4)])
    ax1.axhline(T_amb - 273.15, ls="--", color="gray", lw=0.9, label=f"T_amb = {T_amb-273:.0f} °C")
    ax1.set_xlabel("Zeit [min]")
    ax1.set_ylabel("Zelltemperatur [°C]")
    ax1.set_title("Temperaturverlauf bei konstantem Strom (1 h)")
    ax1.legend()

    # Subplot 2: Wärmegeneration Q̇ vs. Strom bei verschiedenen Temperaturen
    I_range = np.linspace(0.0, 8.0, 300)
    T_th_vals = [268.0, 283.0, 298.0, 313.0, 328.0]
    csv_heat: list[list] = []
    for T_th, col in zip(T_th_vals, _COLORS):
        Q_gen = np.array([model.heat_generation(I, R_int, T_th) for I in I_range])
        ax2.plot(I_range, Q_gen, color=col, label=f"T = {T_th-273:.0f} °C")
        for I, Q in zip(I_range[::15], Q_gen[::15]):
            csv_heat.append([round(T_th - 273.0, 0), round(float(I), 3), round(float(Q), 6)])
    ax2.set_xlabel("Strom I [A]")
    ax2.set_ylabel("Wärmegeneration Q̇_gen [W]")
    ax2.set_title("Wärmegeneration Q̇(I) bei verschiedenen Temperaturen")
    ax2.legend()

    _save_svg(fig, "exp03_thermal.svg")
    _save_csv("exp03_temperature_step_response.csv", ["I_A", "t_s", "T_degC"], csv_temp)
    _save_csv("exp03_heat_generation.csv", ["T_degC", "I_A", "Q_W"], csv_heat)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 04 – Thermischer Verbesserungsfaktor Φ_T
# ═══════════════════════════════════════════════════════════════════════════════

def exp04_phi_thermal() -> None:
    """Φ_T als Funktion der Temperaturdifferenz ΔT und der KI-PMM-Betriebstemperatur."""
    delta_T = np.linspace(0.0, 35.0, 300)
    T_static_ref = 308.0
    Ea_vals = [30_000.0, 50_000.0, 70_000.0, 90_000.0]

    T_AI_range = np.linspace(263.0, 318.0, 300)
    T_static_vals = [298.0, 308.0, 318.0]

    fig, ax1, ax2 = _fig2("Exp 04 – Thermischer Verbesserungsfaktor Φ_T")

    # Subplot 1: Φ_T vs ΔT für verschiedene Eₐ
    csv1: list[list] = []
    for Ea, col in zip(Ea_vals, _COLORS):
        phi_T = np.array([improvement_factor_thermal(Ea, T_static_ref - dT, T_static_ref)
                          for dT in delta_T])
        ax1.plot(delta_T, phi_T, color=col, label=f"Eₐ = {Ea/1000:.0f} kJ/mol")
        for dT, p in zip(delta_T[::15], phi_T[::15]):
            csv1.append([round(Ea, 0), round(float(dT), 2), round(float(p), 6)])
    ax1.axhline(1.0, ls="--", color="gray", lw=0.9)
    ax1.set_xlabel("ΔT = T_static − T_AI [K]")
    ax1.set_ylabel("Φ_T")
    ax1.set_title(f"Φ_T vs. ΔT  (T_static = {T_static_ref-273:.0f} °C)")
    ax1.legend()

    # Subplot 2: Φ_T vs T_AI für verschiedene T_static (Eₐ = 50 kJ/mol)
    csv2: list[list] = []
    for T_s, col in zip(T_static_vals, _COLORS):
        phi_T = np.array([improvement_factor_thermal(50_000.0, T_ai, T_s)
                          for T_ai in T_AI_range])
        ax2.plot(T_AI_range - 273.15, phi_T, color=col,
                 label=f"T_static = {T_s-273:.0f} °C")
        for T_ai, p in zip(T_AI_range[::15], phi_T[::15]):
            csv2.append([round(T_s, 1), round(float(T_ai) - 273.15, 2), round(float(p), 6)])
    ax2.axhline(1.0, ls="--", color="gray", lw=0.9)
    ax2.set_xlabel("T_AI [°C]")
    ax2.set_ylabel("Φ_T")
    ax2.set_title("Φ_T vs. T_AI  (Eₐ = 50 kJ/mol)")
    ax2.legend()

    _save_svg(fig, "exp04_phi_thermal.svg")
    _save_csv("exp04_phi_T_vs_deltaT.csv", ["Ea_J_mol", "delta_T_K", "phi_T"], csv1)
    _save_csv("exp04_phi_T_vs_T_AI.csv", ["T_static_K", "T_AI_degC", "phi_T"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 05 – Spannungs- & kombinierter Verbesserungsfaktor
# ═══════════════════════════════════════════════════════════════════════════════

def exp05_phi_combined() -> None:
    """Φ_V als Funktion der Spannungsreduktion und kombinierter Φ vs. thermisches Gewicht γ_T."""
    eta_V_range = np.linspace(0.0, 0.97, 300)
    gamma_V_vals = [0.20, 0.35, 0.50, 0.70]

    gamma_T_range = np.linspace(0.01, 0.99, 300)
    phi_T_vals = [1.3, 1.5, 1.8, 2.2]
    phi_V_fixed = 2.0

    fig, ax1, ax2 = _fig2("Exp 05 – Spannungs- & kombinierter Verbesserungsfaktor")

    # Subplot 1: Φ_V vs η_V
    csv1: list[list] = []
    for gV, col in zip(gamma_V_vals, _COLORS):
        phi_V = np.array([improvement_factor_voltage(eta, gV) for eta in eta_V_range])
        ax1.plot(eta_V_range, phi_V, color=col, label=f"γ_V* = {gV}")
        for eta, p in zip(eta_V_range[::15], phi_V[::15]):
            csv1.append([round(gV, 2), round(float(eta), 4), round(float(p), 6)])
    ax1.set_xlabel("η_V – Reduktionsgrad der Spannungsdegradation")
    ax1.set_ylabel("Φ_V")
    ax1.set_title("Spannungsbedingter Verbesserungsfaktor Φ_V")
    ax1.set_ylim(0.0, 20.0)
    ax1.legend()

    # Subplot 2: Kombinierter Φ vs γ_T
    csv2: list[list] = []
    for phi_T, col in zip(phi_T_vals, _COLORS):
        phi_c = np.array([improvement_factor_combined(phi_T, phi_V_fixed, gT)
                          for gT in gamma_T_range])
        ax2.plot(gamma_T_range, phi_c, color=col, label=f"Φ_T = {phi_T}")
        for gT, p in zip(gamma_T_range[::15], phi_c[::15]):
            csv2.append([round(phi_T, 2), round(float(gT), 4), round(float(p), 6)])
    ax2.axvline(GAMMA_T_DEFAULT, ls="--", color="gray", lw=0.9,
                label=f"γ_T = {GAMMA_T_DEFAULT} (Standard)")
    ax2.set_xlabel("γ_T – thermisches Degradationsgewicht")
    ax2.set_ylabel("Φ")
    ax2.set_title(f"Kombinierter Verbesserungsfaktor Φ  (Φ_V = {phi_V_fixed})")
    ax2.legend()

    _save_svg(fig, "exp05_phi_combined.svg")
    _save_csv("exp05_phi_V.csv", ["gamma_V", "eta_V", "phi_V"], csv1)
    _save_csv("exp05_phi_combined.csv", ["phi_T", "gamma_T", "phi_combined"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 06 – Reichweite vs. Zyklen & kältebedingte Kapazitätsreduktion
# ═══════════════════════════════════════════════════════════════════════════════

def exp06_range_cold() -> None:
    """Reichweite R(n) für verschiedene Φ und kältebedingte Kapazitätsreduktion Q(T_amb)."""
    n_EoL_base = 1000.0
    R0 = 400.0          # km
    n_cycles = np.linspace(0.0, 1500.0, 500)
    phi_vals = [1.0, 1.5, 2.0, 3.0]

    T_amb_range = np.linspace(233.0, 313.0, 300)   # −40 … +40 °C
    Q0_vals = [2.0, 3.0, 5.0, 10.0]

    fig, ax1, ax2 = _fig2("Exp 06 – Reichweite vs. Zyklen & Kälteleistung")

    # Subplot 1: Reichweite vs. Zyklen
    csv1: list[list] = []
    for phi, col in zip(phi_vals, _COLORS):
        n_EoL = n_EoL_base * phi
        R_n = np.array([range_vs_cycles(R0, n, n_EoL) for n in n_cycles])
        label = "Statisch (Φ = 1)" if phi == 1.0 else f"KI-PMM (Φ = {phi})"
        ax1.plot(n_cycles, R_n, color=col, label=label)
        for n, R in zip(n_cycles[::25], R_n[::25]):
            csv1.append([round(phi, 2), round(float(n), 0), round(float(R), 3)])
    ax1.set_xlabel("Zyklenzahl n")
    ax1.set_ylabel("Reichweite R(n) [km]")
    ax1.set_title(f"Reichweitenentwicklung R(n)  (R₀ = {R0:.0f} km, n_EoL_base = {n_EoL_base:.0f})")
    ax1.legend()

    # Subplot 2: Kältebedingte Kapazitätsreduktion
    csv2: list[list] = []
    for Q0_item, col in zip(Q0_vals, _COLORS):
        Q_cold = np.array([cold_capacity(Q0_item, T, T_REF_COLD, KAPPA_T) for T in T_amb_range])
        ax2.plot(T_amb_range - 273.15, Q_cold, color=col, label=f"Q₀ = {Q0_item:.0f} Ah")
        for T, Q in zip(T_amb_range[::15], Q_cold[::15]):
            csv2.append([round(Q0_item, 1), round(float(T) - 273.15, 1), round(float(Q), 4)])
    ax2.axvline(T_REF_COLD - 273.15, ls="--", color="gray", lw=0.9,
                label=f"T_ref = {T_REF_COLD-273:.0f} °C")
    ax2.set_xlabel("Umgebungstemperatur T_amb [°C]")
    ax2.set_ylabel("Verfügbare Kapazität Q(T) [Ah]")
    ax2.set_title("Kältebedingte Kapazitätsreduktion Q(T_amb)")
    ax2.legend()

    _save_svg(fig, "exp06_range_cold.svg")
    _save_csv("exp06_range_vs_cycles.csv", ["phi", "n_cycles", "range_km"], csv1)
    _save_csv("exp06_cold_capacity.csv", ["Q0_Ah", "T_degC", "Q_derated_Ah"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 07 – Mechanikmodell
# ═══════════════════════════════════════════════════════════════════════════════

def exp07_mechanics() -> None:
    """Quellungsdehnung & Tortuosität vs. SoC und mechanische Degradationsrate vs. σ_mech."""
    SoC_range = np.linspace(0.0, 1.0, 400)
    m_mech_vals = [1.5, 2.0, 2.5, 3.5]
    sigma_ratio = np.linspace(0.0, 4.5, 400)

    fig, ax1, ax2 = _fig2("Exp 07 – Mechanikmodell: Quellung, Tortuosität & Degradation")

    # Subplot 1: Quellung (linke Achse) & Tortuosität (rechte Achse) vs. SoC
    strain = np.array([swelling_strain(s) for s in SoC_range])
    d_ref = np.clip(D_SEP_0 * (1.0 + strain), D_SEP_MIN, D_SEP_MAX)
    tau_vals = np.array([tortuosity(d) for d in d_ref])

    ax1_r = ax1.twinx()
    ax1.spines["right"].set_visible(True)
    l1, = ax1.plot(SoC_range, strain * 100.0, color=_COLORS[0], lw=2.0,
                   label="Dehnung ε_vol [%]")
    l2, = ax1_r.plot(SoC_range, tau_vals, color=_COLORS[1], ls="--", lw=2.0,
                     label="Tortuosität τ")
    ax1.set_xlabel("Ladezustand SoC")
    ax1.set_ylabel("Volumetrische Dehnung ε_vol [%]", color=_COLORS[0])
    ax1_r.set_ylabel("Tortuosität τ", color=_COLORS[1])
    ax1.set_title("Quellung & Tortuosität des Separators vs. SoC")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left")

    csv1 = [[round(float(s), 4), round(float(e) * 100.0, 6), round(float(d) * 1e6, 4),
             round(float(tau), 6)]
            for s, e, d, tau in zip(SoC_range[::20], strain[::20], d_ref[::20], tau_vals[::20])]

    # Subplot 2: Mechanische Degradationsrate vs. σ_mech / σ_yield
    csv2: list[list] = []
    for m, col in zip(m_mech_vals, _COLORS):
        rates = np.array([mechanical_degradation_rate(sr * 1.0, sigma_yield=1.0,
                                                       k_mech=1.0, m_mech=m)
                          for sr in sigma_ratio])
        ax2.plot(sigma_ratio, rates, color=col, label=f"m_mech = {m}")
        for sr, r in zip(sigma_ratio[::20], rates[::20]):
            csv2.append([round(m, 2), round(float(sr), 4), round(float(r), 8)])
    ax2.axvline(1.0, ls="--", color="gray", lw=0.9, label="σ_yield Grenze")
    ax2.set_xlabel("σ_mech / σ_yield")
    ax2.set_ylabel("Mechanische Degradationsrate r_mech")
    ax2.set_title("Mechanische Degradation r_mech(σ_mech / σ_yield)")
    ax2.legend()

    _save_svg(fig, "exp07_mechanics.svg")
    _save_csv("exp07_swelling_tortuosity.csv",
              ["SoC", "strain_pct", "d_ref_um", "tortuosity"], csv1)
    _save_csv("exp07_mechanical_degradation.csv",
              ["m_mech", "sigma_ratio", "rate"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 08 – Erweiterter Verbesserungsfaktor Φ_ges
# ═══════════════════════════════════════════════════════════════════════════════

def exp08_phi_total() -> None:
    """Φ_ges mit Plattenabstandskanal: Einfluss von ρ_M und η_M."""
    rho_M_range = np.linspace(0.0, 1.0, 300)
    phi_base_fixed = 2.0
    eta_M_vals = [0.10, 0.20, 0.30, 0.50]

    phi_range = np.linspace(1.0, 5.0, 300)
    rho_vals = [0.05, 0.10, 0.15, 0.25]
    eta_M_nmc = mechanical_reduction_factor(1.8, M_MECH_DEFAULT)

    fig, ax1, ax2 = _fig2("Exp 08 – Erweiterter Verbesserungsfaktor Φ_ges")

    # Subplot 1: Φ_ges vs. ρ_M für verschiedene η_M
    csv1: list[list] = []
    for eta, col in zip(eta_M_vals, _COLORS):
        phi_ges = np.array([improvement_factor_total(phi_base_fixed, rho, eta)
                            for rho in rho_M_range])
        ax1.plot(rho_M_range, phi_ges, color=col, label=f"η_M = {eta}")
        for rho, p in zip(rho_M_range[::15], phi_ges[::15]):
            csv1.append([round(eta, 2), round(float(rho), 4), round(float(p), 6)])
    ax1.axhline(phi_base_fixed, ls="--", color="gray", lw=0.9,
                label=f"Φ (ohne Mechanik) = {phi_base_fixed}")
    ax1.set_xlabel("ρ_M – mechanischer Degradationsanteil")
    ax1.set_ylabel("Φ_ges")
    ax1.set_title(f"Φ_ges vs. ρ_M  (Φ = {phi_base_fixed})")
    ax1.legend()

    # Subplot 2: Φ_ges vs. Φ für verschiedene ρ_M (η_M aus NMC/Graphit)
    csv2: list[list] = []
    for rho, col in zip(rho_vals, _COLORS):
        phi_ges = np.array([improvement_factor_total(p, rho, eta_M_nmc) for p in phi_range])
        ax2.plot(phi_range, phi_ges, color=col, label=f"ρ_M = {rho}")
        for p, pg in zip(phi_range[::15], phi_ges[::15]):
            csv2.append([round(rho, 2), round(float(p), 4), round(float(pg), 6)])
    ax2.plot(phi_range, phi_range, "k--", lw=0.9, label="Φ_ges = Φ (Referenz)")
    ax2.set_xlabel("Basis-Verbesserungsfaktor Φ")
    ax2.set_ylabel("Φ_ges")
    ax2.set_title(f"Φ_ges vs. Φ  (η_M = {eta_M_nmc:.3f}, NMC/Graphit)")
    ax2.legend()

    _save_svg(fig, "exp08_phi_total.svg")
    _save_csv("exp08_phi_total_vs_rhoM.csv", ["eta_M", "rho_M", "phi_ges"], csv1)
    _save_csv("exp08_phi_total_vs_phi.csv", ["rho_M", "phi_base", "phi_ges"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 09 – Zeitseriensimulation LP1 (Smartphone)
# ═══════════════════════════════════════════════════════════════════════════════

def exp09_simulation_lp1() -> None:
    """LP1 Smartphone: Zelltemperatur und kumulative Degradation – Statisch vs. KI-PMM."""
    profile = LoadProfile.lp1_smartphone(n_cycles=60, dt=60.0)
    static, ai = compare_systems(profile, n_cycles=60, dt=60.0)

    phi_sim = (phi_factor_from_rates(static.mean_degradation_rate, ai.mean_degradation_rate)
               if ai.mean_degradation_rate > 0 else float("nan"))

    fig, ax1, ax2 = _fig2("Exp 09 – LP1 Smartphone: Statisches PMS vs. KI-PMM")

    # Subplot 1: Temperaturverläufe
    ax1.plot(static.times / 3600.0, static.temperatures - 273.15,
             color=_COLORS[0], alpha=0.75, label="Statisches PMS")
    ax1.plot(ai.times / 3600.0, ai.temperatures - 273.15,
             color=_COLORS[1], alpha=0.75, label="KI-PMM")
    ax1.set_xlabel("Zeit [h]")
    ax1.set_ylabel("Zelltemperatur [°C]")
    ax1.set_title(
        f"Temperaturverlauf LP1\n"
        f"T̄_stat = {static.mean_temperature-273.15:.1f} °C  |  "
        f"T̄_AI = {ai.mean_temperature-273.15:.1f} °C"
    )
    ax1.legend()

    # Subplot 2: Kumulative Degradation
    ax2.plot(static.times / 3600.0, static.degradations,
             color=_COLORS[0], label="Statisches PMS")
    ax2.plot(ai.times / 3600.0, ai.degradations,
             color=_COLORS[1], label="KI-PMM")
    ax2.axhline(D_EOL, ls="--", color="red", lw=0.9, label=f"EoL ({D_EOL*100:.0f} %)")
    ax2.set_xlabel("Zeit [h]")
    ax2.set_ylabel("Kumulative Degradation D(t)")
    ax2.set_title(f"Degradationsverlauf LP1  |  Φ_sim = {phi_sim:.3f}")
    ax2.legend()

    _save_svg(fig, "exp09_simulation_lp1.svg")
    _write_sim_csv("exp09_lp1_static.csv", static)
    _write_sim_csv("exp09_lp1_ai.csv", ai)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 10 – Zeitseriensimulation LP2 (E-Bike)
# ═══════════════════════════════════════════════════════════════════════════════

def exp10_simulation_lp2() -> None:
    """LP2 E-Bike: SoC-Verlauf und momentane Degradationsrate – Statisch vs. KI-PMM."""
    profile = LoadProfile.lp2_ebike(n_cycles=60, dt=60.0)
    static, ai = compare_systems(profile, n_cycles=60, dt=60.0)

    fig, ax1, ax2 = _fig2("Exp 10 – LP2 E-Bike: Statisches PMS vs. KI-PMM")

    ax1.plot(static.times / 3600.0, static.SoCs,
             color=_COLORS[0], alpha=0.80, label="Statisches PMS")
    ax1.plot(ai.times / 3600.0, ai.SoCs,
             color=_COLORS[1], alpha=0.80, label="KI-PMM")
    ax1.set_xlabel("Zeit [h]")
    ax1.set_ylabel("Ladezustand SoC")
    ax1.set_title("SoC-Verlauf LP2 (E-Bike)")
    ax1.legend()

    ax2.plot(static.times / 3600.0, static.degradation_rates,
             color=_COLORS[0], alpha=0.70, label="Statisches PMS")
    ax2.plot(ai.times / 3600.0, ai.degradation_rates,
             color=_COLORS[1], alpha=0.70, label="KI-PMM")
    ax2.set_yscale("symlog", linthresh=1e-9)
    ax2.set_xlabel("Zeit [h]")
    ax2.set_ylabel("Momentane Degradationsrate D̊(t) [1/s]  [symlog]")
    ax2.set_title(
        f"Degradationsrate LP2  |  "
        f"D̊_stat = {static.mean_degradation_rate:.2e}  |  "
        f"D̊_AI = {ai.mean_degradation_rate:.2e}"
    )
    ax2.legend()

    _save_svg(fig, "exp10_simulation_lp2.svg")
    _write_sim_csv("exp10_lp2_static.csv", static)
    _write_sim_csv("exp10_lp2_ai.csv", ai)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 11 – Zeitseriensimulation LP3 (Stationärspeicher)
# ═══════════════════════════════════════════════════════════════════════════════

def exp11_simulation_lp3() -> None:
    """LP3 Stationärspeicher: Klemmenspannung und Zelltemperatur – Statisch vs. KI-PMM."""
    profile = LoadProfile.lp3_stationary(n_cycles=40, dt=300.0)
    static, ai = compare_systems(profile, n_cycles=40, dt=300.0)

    fig, ax1, ax2 = _fig2("Exp 11 – LP3 Stationär: Statisches PMS vs. KI-PMM")

    ax1.plot(static.times / 3600.0, static.voltages,
             color=_COLORS[0], alpha=0.80, label="Statisches PMS")
    ax1.plot(ai.times / 3600.0, ai.voltages,
             color=_COLORS[1], alpha=0.80, label="KI-PMM")
    ax1.set_xlabel("Zeit [h]")
    ax1.set_ylabel("Klemmenspannung V [V]")
    ax1.set_title("Spannungsverlauf LP3 (Stationär)")
    ax1.legend()

    ax2.plot(static.times / 3600.0, static.temperatures - 273.15,
             color=_COLORS[0], alpha=0.80, label="Statisches PMS")
    ax2.plot(ai.times / 3600.0, ai.temperatures - 273.15,
             color=_COLORS[1], alpha=0.80, label="KI-PMM")
    ax2.set_xlabel("Zeit [h]")
    ax2.set_ylabel("Zelltemperatur [°C]")
    ax2.set_title(
        f"Temperaturverlauf LP3  |  "
        f"T̄_stat = {static.mean_temperature-273.15:.1f} °C  |  "
        f"T̄_AI = {ai.mean_temperature-273.15:.1f} °C"
    )
    ax2.legend()

    _save_svg(fig, "exp11_simulation_lp3.svg")
    _write_sim_csv("exp11_lp3_static.csv", static)
    _write_sim_csv("exp11_lp3_ai.csv", ai)


# ── Hilfsfunktion für Simulations-CSV ────────────────────────────────────────

def _write_sim_csv(filename: str, result) -> None:
    """Schreibt die relevanten Zeitreihen eines SimulationResult als CSV."""
    stride = max(1, len(result.times) // 2000)   # max. ~2000 Zeilen
    rows = [
        [
            round(float(t), 1),
            round(float(T) - 273.15, 3),
            round(float(V), 5),
            round(float(s), 5),
            round(float(D), 8),
            round(float(Dr), 10),
        ]
        for t, T, V, s, D, Dr in zip(
            result.times[::stride],
            result.temperatures[::stride],
            result.voltages[::stride],
            result.SoCs[::stride],
            result.degradations[::stride],
            result.degradation_rates[::stride],
        )
    ]
    _save_csv(filename, ["t_s", "T_degC", "V_V", "SoC", "degradation", "deg_rate_1_s"], rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 12 – Lastprofilvergleich LP1/LP2/LP3
# ═══════════════════════════════════════════════════════════════════════════════

def exp12_profile_comparison() -> None:
    """Gruppenvergleich: Mittlere Zelltemperatur und Degradationsrate für LP1/LP2/LP3."""
    profiles = [
        ("LP1 Smartphone", LoadProfile.lp1_smartphone(n_cycles=30, dt=60.0)),
        ("LP2 E-Bike",     LoadProfile.lp2_ebike(n_cycles=30, dt=60.0)),
        ("LP3 Stationär",  LoadProfile.lp3_stationary(n_cycles=20, dt=300.0)),
    ]
    labels, T_stat_vals, T_ai_vals, Dr_stat_vals, Dr_ai_vals, phi_vals = (
        [], [], [], [], [], []
    )
    for lbl, prof in profiles:
        s, a = compare_systems(prof, n_cycles=30)
        phi_s = (phi_factor_from_rates(s.mean_degradation_rate, a.mean_degradation_rate)
                 if a.mean_degradation_rate > 0 else float("nan"))
        labels.append(lbl)
        T_stat_vals.append(s.mean_temperature - 273.15)
        T_ai_vals.append(a.mean_temperature - 273.15)
        Dr_stat_vals.append(s.mean_degradation_rate)
        Dr_ai_vals.append(a.mean_degradation_rate)
        phi_vals.append(phi_s)

    fig, ax1, ax2 = _fig2("Exp 12 – Lastprofilvergleich: Mittlere Kenngrößen")

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, T_stat_vals, w, color=_COLORS[0], label="Statisches PMS")
    ax1.bar(x + w / 2, T_ai_vals, w, color=_COLORS[1], label="KI-PMM")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=8)
    ax1.set_ylabel("Mittlere Zelltemperatur [°C]")
    ax1.set_title("Mittlere Zelltemperatur je Lastprofil")
    ax1.legend()

    ax2.bar(x - w / 2, Dr_stat_vals, w, color=_COLORS[0], label="Statisches PMS")
    ax2.bar(x + w / 2, Dr_ai_vals, w, color=_COLORS[1], label="KI-PMM")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=8)
    ax2.set_ylabel("Mittlere Degradationsrate D̊ [1/s]")
    ax2.set_title("Mittlere Degradationsrate & Φ_sim je Lastprofil")
    ax2.set_yscale("log")
    for xi, phi in zip(x, phi_vals):
        ax2.text(xi, max(Dr_stat_vals[xi], Dr_ai_vals[xi]) * 1.15,
                 f"Φ={phi:.2f}", ha="center", fontsize=8)
    ax2.legend()

    _save_svg(fig, "exp12_profile_comparison.svg")
    _save_csv(
        "exp12_profile_comparison.csv",
        ["Lastprofil", "T_static_degC", "T_ai_degC",
         "D_static_1_s", "D_ai_1_s", "phi_sim"],
        [
            [lbl, round(Ts, 3), round(Ta, 3),
             f"{Ds:.4e}", f"{Da:.4e}", round(float(phi), 4)]
            for lbl, Ts, Ta, Ds, Da, phi in
            zip(labels, T_stat_vals, T_ai_vals, Dr_stat_vals, Dr_ai_vals, phi_vals)
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 13 – SEI-Kapazitätsverlust & Kapazitätsdrift D(t)
# ═══════════════════════════════════════════════════════════════════════════════

def exp13_sei_capacity() -> None:
    """SEI-bedingter Kapazitätsverlust ΔQ_SEI(t) und Kapazitätsdrift D(t) bei verschiedenen T."""
    t_arr = np.linspace(0.0, 3.15e7, 600)        # 0 … 1 Jahr [s]
    T_vals = [278.0, 288.0, 298.0, 308.0, 318.0]
    Q0 = Q0_DEFAULT

    fig, ax1, ax2 = _fig2("Exp 13 – SEI-Kapazitätsverlust & Kapazitätsdrift D(t)")

    csv1: list[list] = []
    csv2: list[list] = []

    for T_s, col in zip(T_vals, _COLORS):
        delta_arr = np.array([sei_thickness(K_SEI_DEFAULT, t, E_A_SEI_DEFAULT, T_s)
                              for t in t_arr])
        dQ_arr = np.array([capacity_loss_sei(ALPHA_SEI_DEFAULT, d) for d in delta_arr])
        D_arr = np.array([capacity_drift(Q0 - dq, Q0) for dq in dQ_arr])
        lbl = f"T = {T_s-273:.0f} °C"

        ax1.plot(t_arr / 86400.0, dQ_arr * 1000.0, color=col, label=lbl)
        ax2.plot(t_arr / 86400.0, D_arr * 100.0, color=col, label=lbl)

        for t, dq, D in zip(t_arr[::30], dQ_arr[::30], D_arr[::30]):
            csv1.append([round(T_s - 273.0, 1), round(float(t) / 86400.0, 3),
                         round(float(dq) * 1000.0, 6)])
            csv2.append([round(T_s - 273.0, 1), round(float(t) / 86400.0, 3),
                         round(float(D) * 100.0, 6)])

    ax1.set_xlabel("Zeit [Tage]")
    ax1.set_ylabel("SEI-Kapazitätsverlust ΔQ_SEI [mAh]")
    ax1.set_title("SEI-bedingter Kapazitätsverlust ΔQ_SEI(t)")
    ax1.legend()

    ax2.axhline(D_EOL * 100.0, ls="--", color="red", lw=0.9,
                label=f"EoL = {D_EOL*100:.0f} %")
    ax2.set_xlabel("Zeit [Tage]")
    ax2.set_ylabel("Kapazitätsdrift D(t) [%]")
    ax2.set_title("Kapazitätsdrift D(t) – reine SEI-Degradation")
    ax2.legend()

    _save_svg(fig, "exp13_sei_capacity.svg")
    _save_csv("exp13_sei_capacity_loss_mAh.csv",
              ["T_degC", "t_days", "dQ_SEI_mAh"], csv1)
    _save_csv("exp13_capacity_drift_pct.csv",
              ["T_degC", "t_days", "D_pct"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 14 – Lebensdauer-Sensitivität & Vorheizeffizienz
# ═══════════════════════════════════════════════════════════════════════════════

def exp14_sensitivity() -> None:
    """SEI-Lebensdauer vs. Aktivierungsenergie Eₐ und Vorheizeffizienz η_heat(T_amb)."""
    Ea_range = np.linspace(20_000.0, 100_000.0, 300)
    T_lt_vals = [278.0, 288.0, 298.0, 308.0, 318.0]
    t_ref = 1.0e7      # 1×10⁷ s ≈ 116 Tage

    T_amb_range = np.linspace(233.0, 283.0, 300)    # −40 … +10 °C
    E_heat_vals = [0.05, 0.10, 0.20, 0.50]           # kWh Heizenergie
    R_base = 400.0                                    # km Ausgangsreichweite

    fig, ax1, ax2 = _fig2("Exp 14 – Lebensdauer-Sensitivität & Vorheizeffizienz")

    # Subplot 1: SEI-Lebensdauer [Jahre] vs. Eₐ
    csv1: list[list] = []
    for T_lt, col in zip(T_lt_vals, _COLORS):
        lifetimes = []
        for Ea in Ea_range:
            delta = sei_thickness(K_SEI_DEFAULT, t_ref, Ea, T_lt)
            dQ = capacity_loss_sei(ALPHA_SEI_DEFAULT, delta)
            D_val = dQ / Q0_DEFAULT
            mean_rate = D_val / t_ref
            lt_years = lifetime_from_rate(D_EOL, mean_rate) / (86400.0 * 365.25) \
                if mean_rate > 0 else float("nan")
            lifetimes.append(lt_years)
        lt_arr = np.array(lifetimes, dtype=float)
        ax1.plot(Ea_range / 1000.0, lt_arr, color=col, label=f"T = {T_lt-273:.0f} °C")
        for Ea, lt in zip(Ea_range[::15], lt_arr[::15]):
            if np.isfinite(lt):
                csv1.append([round(T_lt - 273.0, 0), round(float(Ea) / 1000.0, 1),
                             round(float(lt), 4)])
    ax1.set_yscale("log")
    ax1.set_xlabel("Aktivierungsenergie Eₐ [kJ/mol]")
    ax1.set_ylabel("SEI-bedingte Lebensdauer [Jahre]  [log]")
    ax1.set_title("Lebensdauer-Sensitivität gegenüber Eₐ und Temperatur")
    ax1.legend()

    # Subplot 2: Vorheizeffizienz [km/kWh] vs. Umgebungstemperatur
    Q_warm = cold_capacity(Q0_DEFAULT, T_REF_COLD, T_REF_COLD, KAPPA_T)
    csv2: list[list] = []
    for E_heat, col in zip(E_heat_vals, _COLORS):
        efficiencies = []
        for T_a in T_amb_range:
            Q_cold = cold_capacity(Q0_DEFAULT, T_a, T_REF_COLD, KAPPA_T)
            range_gain = R_base * (Q_warm - Q_cold) / Q0_DEFAULT
            efficiencies.append(preheating_efficiency(range_gain, E_heat))
        eff_arr = np.array(efficiencies)
        ax2.plot(T_amb_range - 273.15, eff_arr, color=col,
                 label=f"E_heat = {E_heat} kWh")
        for T_a, eta in zip(T_amb_range[::15], eff_arr[::15]):
            csv2.append([round(E_heat, 2), round(float(T_a) - 273.15, 1),
                         round(float(eta), 4)])
    ax2.set_xlabel("Umgebungstemperatur T_amb [°C]")
    ax2.set_ylabel("Vorheizeffizienz η_heat [km/kWh]")
    ax2.set_title(f"Vorheizeffizienz η_heat(T_amb)  (R₀ = {R_base:.0f} km)")
    ax2.legend()

    _save_svg(fig, "exp14_sensitivity.svg")
    _save_csv("exp14_lifetime_sensitivity.csv",
              ["T_degC", "Ea_kJ_mol", "lifetime_years"], csv1)
    _save_csv("exp14_preheating_efficiency.csv",
              ["E_heat_kWh", "T_amb_degC", "eta_km_per_kWh"], csv2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 15 – E-Fahrzeug-Lebenszyklus & Plattenabstands-MPC
# ═══════════════════════════════════════════════════════════════════════════════

def exp15_ev_lifecycle() -> None:
    """Gesamte Fahrleistung bis Akkutausch vs. Φ und MPC-Referenztrajektorie d_ref(SoC)."""
    phi_range = np.linspace(1.0, 4.5, 300)
    km_per_year_vals = [10_000, 15_000, 20_000, 30_000]
    n_EoL_base = 1000.0
    cycles_per_year = 365.0

    SoC_range = np.linspace(0.0, 1.0, 400)
    mpc = PlateDistanceMPCController()

    fig, ax1, ax2 = _fig2("Exp 15 – E-Fahrzeug-Lebenszyklus & Plattenabstands-MPC")

    # Subplot 1: Kumulative Fahrleistung [1000 km] vs. Φ
    csv1: list[list] = []
    for km_yr, col in zip(km_per_year_vals, _COLORS):
        total_kms = []
        for phi in phi_range:
            n_EoL = n_EoL_base * phi
            years = ev_lifetime_years(n_EoL, cycles_per_year)
            total_km = ev_cumulative_km(years, float(km_yr))
            total_kms.append(total_km / 1000.0)
        km_arr = np.array(total_kms)
        ax1.plot(phi_range, km_arr, color=col,
                 label=f"{km_yr // 1000:.0f} 000 km/Jahr")
        for phi, tkm in zip(phi_range[::15], km_arr[::15]):
            csv1.append([round(km_yr, 0), round(float(phi), 3), round(float(tkm), 2)])
    ax1.set_xlabel("Verbesserungsfaktor Φ")
    ax1.set_ylabel("Kumulative Fahrleistung [1 000 km]")
    ax1.set_title("Gesamtfahrleistung bis Akkutausch")
    ax1.legend()

    # Subplot 2: MPC-Referenzabstand d_ref(SoC) + Tortuosität
    d_ref_arr = np.array([mpc.reference(s) for s in SoC_range])
    tau_ref_arr = np.array([tortuosity(d) for d in d_ref_arr])

    ax2_r = ax2.twinx()
    ax2.spines["right"].set_visible(True)
    l1, = ax2.plot(SoC_range, d_ref_arr * 1e6, color=_COLORS[0], lw=2.0,
                   label="d_ref [µm]")
    ax2.axhline(D_SEP_MIN * 1e6, ls=":", color="gray", lw=0.9)
    ax2.axhline(D_SEP_MAX * 1e6, ls=":", color="gray", lw=0.9)
    ax2.axhline(D_SEP_0 * 1e6, ls="--", color="gray", lw=0.9)
    ax2.text(0.02, D_SEP_MIN * 1e6 + 0.3, f"d_min = {D_SEP_MIN*1e6:.0f} µm",
             fontsize=8, color="gray")
    ax2.text(0.02, D_SEP_MAX * 1e6 + 0.3, f"d_max = {D_SEP_MAX*1e6:.0f} µm",
             fontsize=8, color="gray")
    ax2.text(0.02, D_SEP_0 * 1e6 + 0.3, f"d₀ = {D_SEP_0*1e6:.0f} µm",
             fontsize=8, color="gray")
    l2, = ax2_r.plot(SoC_range, tau_ref_arr, color=_COLORS[1], ls="--", lw=2.0,
                     label="τ(d_ref)")
    ax2.set_xlabel("Ladezustand SoC")
    ax2.set_ylabel("Referenzabstand d_ref [µm]", color=_COLORS[0])
    ax2_r.set_ylabel("Tortuosität τ", color=_COLORS[1])
    ax2.set_title("MPC-Referenztrajektorie d_ref(SoC)")
    ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left")

    _save_svg(fig, "exp15_ev_lifecycle.svg")
    _save_csv("exp15_ev_cumulative_km.csv",
              ["km_per_year", "phi", "total_km_k"], csv1)
    _save_csv("exp15_mpc_reference.csv",
              ["SoC", "d_ref_um", "tortuosity"],
              [[round(float(s), 4), round(float(d) * 1e6, 4), round(float(tau), 6)]
               for s, d, tau in zip(SoC_range[::20], d_ref_arr[::20], tau_ref_arr[::20])])


# ═══════════════════════════════════════════════════════════════════════════════
# Hauptprogramm
# ═══════════════════════════════════════════════════════════════════════════════

_EXPERIMENTS = [
    exp01_arrhenius_sei,
    exp02_voltage_degradation,
    exp03_thermal,
    exp04_phi_thermal,
    exp05_phi_combined,
    exp06_range_cold,
    exp07_mechanics,
    exp08_phi_total,
    exp09_simulation_lp1,
    exp10_simulation_lp2,
    exp11_simulation_lp3,
    exp12_profile_comparison,
    exp13_sei_capacity,
    exp14_sensitivity,
    exp15_ev_lifecycle,
]


def main() -> None:
    """Führt alle 15 Experimente sequenziell aus."""
    print(f"liionp Experiment-Suite  –  Ergebnisse in: {RESULTS.resolve()}")
    print("=" * 70)
    errors = 0
    for fn in _EXPERIMENTS:
        title = fn.__doc__.splitlines()[0].strip() if fn.__doc__ else fn.__name__
        print(f"\n▶  {fn.__name__}  –  {title}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"    FEHLER: {exc}")
            import traceback
            traceback.print_exc()
            errors += 1

    print("\n" + "=" * 70)
    svgs = sorted(RESULTS.glob("*.svg"))
    csvs = sorted(RESULTS.glob("*.csv"))
    print(f"Abgeschlossen: {len(svgs)} SVG-Plots, {len(csvs)} CSV-Dateien"
          f"{f', {errors} Fehler' if errors else ' – alle erfolgreich'}.")
    print(f"Verzeichnis: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
