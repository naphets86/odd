"""
experiments.py
==============
Umfangreiche Demonstrations-Experimente zur Elastizität hydraulischer Öle.

Alle Berechnungen basieren auf *Elastizität hydraulischer Öle und Flüssigkeiten
im industriellen Maschinenbau* (Epp, 2026).

Ergebnisse werden gespeichert in:
  src/results/*.csv   – tabellarische Daten
  src/results/*.svg   – je zwei Subplots pro Experiment

Ausführung vom Projektstamm:
  PYTHONPATH=src python src/experiments.py
"""

from __future__ import annotations

import csv
import math
import os
import sys

# ---------------------------------------------------------------------------
# sys.path: src/ dem Python-Pfad hinzufügen, damit hydr importierbar ist
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib
matplotlib.use("Agg")          # kein GUI-Backend nötig
import matplotlib.pyplot as plt
import numpy as np

from hydr.bulk_modulus import (
    isentropic_modulus,
    sound_speed,
    bulk_modulus_tait_linear,
    bulk_modulus_dowson_higginson,
    bulk_modulus_temperature,
    effective_bulk_modulus,
)
from hydr.hydraulics import (
    compressibility_error,
    hydraulic_natural_frequency,
    sea_stiffness,
    max_contact_force,
    compression_energy,
    accumulator_constant,
    max_operating_pressure,
    bearing_cutoff_frequency,
)
from hydr.lifetime import (
    oil_lifetime,
    activation_energy_pressure,
    bulk_modulus_aging,
    failure_rate_oxidation,
    failure_rate_friction,
    failure_rate_cavitation,
    total_failure_rate,
)
from hydr.mixing import reuss_modulus, maxwell_emulsion_modulus, voigt_modulus
from hydr.monitoring import (
    ultrasonic_bulk_modulus,
    sound_speed_from_transit,
    recuperation_efficiency,
    OilCondition,
)
from hydr.surface import (
    laplace_pressure,
    cavitation_pressure,
    henry_gas_concentration,
    min_safe_pressure,
)
from hydr.viscoelastic import (
    storage_modulus,
    loss_modulus,
    loss_factor,
    complex_modulus_magnitude,
)
from hydr.viscosity import (
    relaxation_time,
    relaxation_frequency,
    barus_viscosity,
    vii_viscosity,
    shear_degradation_viscosity,
)

# ---------------------------------------------------------------------------
# Ergebnisverzeichnis
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(SRC_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_csv(filename: str, header: list[str], rows: list[list]) -> str:
    """Schreibt eine CSV-Datei nach RESULTS_DIR und gibt den Pfad zurück."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _save_svg(fig: plt.Figure, filename: str) -> str:
    """Speichert eine matplotlib-Figure als SVG."""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# Typische Materialparameter
# ============================================================================

# Mineralöl VG 46 bei 40 °C
OIL_K0         = 1.50e9    # Pa    Isothermer K bei Atmosphärendruck
OIL_RHO        = 870.0     # kg/m³ Dichte
OIL_ETA        = 0.046     # Pa·s  Dynamische Viskosität (≈ 52 cSt @ 40 °C)
OIL_CP         = 1900.0    # J/(kg·K) spez. Wärme
OIL_ALPHA_V    = 7.0e-4    # 1/K   therm. Volumenausdehnungskoeff.
OIL_T_REF      = 313.15    # K     Referenztemperatur (40 °C)
OIL_BETA_K     = 3.5e-3    # 1/K   Temperaturkoeff. des Moduls
OIL_N_TAIT    = 10.5       # –     Tait-Druckkoeffizient
OIL_ALPHA_P    = 2.0e-8    # 1/Pa  Barus-Druckkoeffizient

# Wasserglykol HFC
HFC_K0         = 2.80e9    # Pa
HFC_RHO        = 1060.0    # kg/m³
HFC_ETA        = 0.035     # Pa·s
HFC_CP         = 3400.0    # J/(kg·K)
HFC_ALPHA_V    = 4.5e-4    # 1/K

# Phosphatester HEP
HEP_K0         = 1.80e9    # Pa
HEP_RHO        = 1120.0    # kg/m³
HEP_ETA        = 0.065     # Pa·s
HEP_CP         = 1750.0    # J/(kg·K)
HEP_ALPHA_V    = 6.0e-4    # 1/K

# Biologisch abbaubares Esteröl HEES
HEES_K0        = 1.65e9    # Pa
HEES_RHO       = 920.0     # kg/m³
HEES_ETA       = 0.055     # Pa·s

# Polyalphaolefin (PAO/SHC)
PAO_K0         = 1.60e9    # Pa
PAO_RHO        = 850.0     # kg/m³
PAO_ETA        = 0.060     # Pa·s


# ============================================================================
# EXPERIMENT 1 – Druckabhängigkeit des Kompressionsmoduls
# ============================================================================

def experiment_01_bulk_vs_pressure() -> None:
    """
    Vergleich der Kompressionsmodul-Druckmodelle:
      - Lineare Tait-Gleichung   K_T(P) = K0 + n·P
      - Dowson-Higginson-Modell  K_T(P) = K0·exp(n·P/K0)
      - Effektiver Modul K_eff  bei verschiedenen Luftgehalten α
    """
    print("\n[ EXP 01 ] Druckabhängigkeit des Kompressionsmoduls …")

    pressures_bar = np.linspace(0, 500, 200)
    pressures_pa  = pressures_bar * 1e5

    K_tait = [bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, P) for P in pressures_pa]
    K_dh   = [bulk_modulus_dowson_higginson(OIL_K0, OIL_N_TAIT, P) for P in pressures_pa]

    alpha_values = [0.0, 0.005, 0.01, 0.02, 0.05]
    K_eff_curves: dict[float, list[float]] = {}
    P_ref = 100e5  # 100 bar Nennbetrieb
    for alpha in alpha_values:
        K_eff_curves[alpha] = [
            effective_bulk_modulus(bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, P), P if P > 0 else 1.0, alpha)
            for P in pressures_pa
        ]

    # CSV
    header = ["p_bar", "K_tait_MPa", "K_DH_MPa"] + [f"K_eff_alpha{a:.3f}_MPa" for a in alpha_values]
    rows = []
    for i, P in enumerate(pressures_bar):
        row = [
            f"{P:.1f}",
            f"{K_tait[i]/1e6:.2f}",
            f"{K_dh[i]/1e6:.2f}",
        ] + [f"{K_eff_curves[a][i]/1e6:.2f}" for a in alpha_values]
        rows.append(row)
    _save_csv("exp01_bulk_vs_pressure.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 1 – Druckabhängiger Kompressionsmodul (Mineralöl VG 46)", fontsize=13)

    ax1.plot(pressures_bar, [K / 1e6 for K in K_tait], "b-",  lw=2, label="Tait linear (Gl. 4)")
    ax1.plot(pressures_bar, [K / 1e6 for K in K_dh],   "r--", lw=2, label="Dowson-Higginson (Gl. 5)")
    ax1.set_xlabel("Druck [bar]")
    ax1.set_ylabel("Kompressionsmodul K [MPa]")
    ax1.set_title("Tait vs. Dowson-Higginson")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(alpha_values)))
    for alpha, color in zip(alpha_values, colors):
        ax2.plot(
            pressures_bar,
            [K / 1e6 for K in K_eff_curves[alpha]],
            color=color, lw=2,
            label=f"α = {alpha*100:.1f} %",
        )
    ax2.set_xlabel("Druck [bar]")
    ax2.set_ylabel("Effektiver Modul K_eff [MPa]")
    ax2.set_title("Effektiver Modul mit entlüftetem Anteil α")
    ax2.legend(title="Luftgehalt")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)

    _save_svg(fig, "exp01_bulk_vs_pressure.svg")
    print("   → exp01_bulk_vs_pressure.csv / .svg")


# ============================================================================
# EXPERIMENT 2 – Temperaturabhängigkeit: Modul und Schallgeschwindigkeit
# ============================================================================

def experiment_02_temperature_effects() -> None:
    """
    Einfluss der Temperatur auf Kompressionsmodul und Schallgeschwindigkeit
    für fünf verschiedene Hydraulikflüssigkeiten.
    """
    print("\n[ EXP 02 ] Temperaturabhängigkeit von K und c_s …")

    temps_C = np.linspace(0, 120, 120)
    temps_K = temps_C + 273.15

    fluids = {
        "Mineralöl VG 46 (HLP)": (OIL_K0, OIL_BETA_K, OIL_T_REF, OIL_RHO, OIL_ALPHA_V, OIL_CP),
        "Wasserglykol HFC":       (HFC_K0, 4.0e-3,     313.15, HFC_RHO, HFC_ALPHA_V, HFC_CP),
        "Phosphatester HEP":      (HEP_K0, 3.8e-3,     313.15, HEP_RHO, HEP_ALPHA_V, HEP_CP),
        "Esteröl HEES":           (HEES_K0, 3.6e-3,    313.15, HEES_RHO, 6.5e-4, 1950.0),
        "PAO / SHC":              (PAO_K0,  3.2e-3,    313.15, PAO_RHO,  6.2e-4, 2000.0),
    }

    K_curves: dict[str, list[float]] = {}
    c_curves: dict[str, list[float]] = {}

    for name, (K_ref, beta_K, T_ref, rho, alpha_V, cp) in fluids.items():
        K_list = []
        c_list = []
        for T in temps_K:
            K_T_val = bulk_modulus_temperature(K_ref, beta_K, T, T_ref)
            K_s_val = isentropic_modulus(K_T_val, alpha_V, T, rho, cp)
            K_list.append(K_s_val)
            c_list.append(sound_speed(K_s_val, rho))
        K_curves[name] = K_list
        c_curves[name] = c_list

    # CSV
    header = ["T_C"] + [f"K_{n}_MPa" for n in fluids] + [f"c_{n}_ms" for n in fluids]
    rows = []
    for i, T in enumerate(temps_C):
        row = [f"{T:.1f}"]
        for name in fluids:
            row.append(f"{K_curves[name][i]/1e6:.2f}")
        for name in fluids:
            row.append(f"{c_curves[name][i]:.1f}")
        rows.append(row)
    _save_csv("exp02_temperature_effects.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 2 – Temperaturabhängigkeit von K und Schallgeschwindigkeit", fontsize=13)

    for name in fluids:
        ax1.plot(temps_C, [K / 1e6 for K in K_curves[name]], lw=2, label=name)
    ax1.set_xlabel("Temperatur [°C]")
    ax1.set_ylabel("Isentroper Modul K_s [MPa]")
    ax1.set_title("Kompressionsmodul K_s(T) – Flüssigkeitsvergleich")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for name in fluids:
        ax2.plot(temps_C, c_curves[name], lw=2, label=name)
    ax2.set_xlabel("Temperatur [°C]")
    ax2.set_ylabel("Schallgeschwindigkeit c_s [m/s]")
    ax2.set_title("Schallgeschwindigkeit c_s(T) – Flüssigkeitsvergleich")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    _save_svg(fig, "exp02_temperature_effects.svg")
    print("   → exp02_temperature_effects.csv / .svg")


# ============================================================================
# EXPERIMENT 3 – Katastrophaler Einfluss von Gaseinschlüssen
# ============================================================================

def experiment_03_gas_content_impact() -> None:
    """
    Effektiver Modul K_eff als Funktion des Luftgehalts α bei verschiedenen
    Drücken. Verdeutlicht den dramatischen Modulrückgang durch Gaseinschlüsse.
    """
    print("\n[ EXP 03 ] Gasgehalt-Einfluss auf K_eff …")

    alpha_pct = np.linspace(0.0, 5.0, 500)   # 0 – 5 % Luftgehalt
    alphas    = alpha_pct / 100.0

    pressures_bar = [10, 50, 100, 200, 350]
    K_curves: dict[int, list[float]] = {}

    for p_bar in pressures_bar:
        P_pa = p_bar * 1e5
        K_T_val = bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, P_pa)
        K_curves[p_bar] = [
            effective_bulk_modulus(K_T_val, P_pa, a) for a in alphas
        ]

    K_base = bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, 100e5)

    # CSV
    header = ["alpha_pct"] + [f"Keff_p{p}bar_MPa" for p in pressures_bar]
    rows = []
    for i, a in enumerate(alpha_pct):
        row = [f"{a:.3f}"] + [f"{K_curves[p][i]/1e6:.3f}" for p in pressures_bar]
        rows.append(row)
    _save_csv("exp03_gas_content_impact.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 3 – Gasgehalt und effektiver Kompressionsmodul", fontsize=13)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pressures_bar)))
    for p_bar, color in zip(pressures_bar, colors):
        ax1.plot(alpha_pct, [K / 1e6 for K in K_curves[p_bar]], color=color, lw=2, label=f"{p_bar} bar")
    ax1.set_xlabel("Luftgehalt α [%]")
    ax1.set_ylabel("K_eff [MPa]")
    ax1.set_title("K_eff(α) bei verschiedenen Systemdrücken")
    ax1.legend(title="Druck")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # Relativer Rückgang bei 100 bar
    K_eff_100 = K_curves[100]
    relative_drop = [(K_base - K) / K_base * 100.0 for K in K_eff_100]
    ax2.fill_between(alpha_pct, 0, relative_drop, alpha=0.4, color="crimson")
    ax2.plot(alpha_pct, relative_drop, "crimson", lw=2)
    ax2.set_xlabel("Luftgehalt α [%]")
    ax2.set_ylabel("Modulrückgang [%]")
    ax2.set_title("Relativer K_eff-Rückgang bei 100 bar gegenüber α=0")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)
    ax2.axhline(10, ls="--", color="orange", lw=1, label="10 %-Grenze")
    ax2.axhline(50, ls="--", color="red",    lw=1, label="50 %-Grenze")
    ax2.legend()

    _save_svg(fig, "exp03_gas_content_impact.svg")
    print("   → exp03_gas_content_impact.csv / .svg")


# ============================================================================
# EXPERIMENT 4 – Hydraulische Eigenfrequenz und Positionsfehler
# ============================================================================

def experiment_04_eigenfrequency_positioning() -> None:
    """
    Hydraulische Eigenfrequenz f_h und Positionsfehler δx als Funktion des
    effektiven Kompressionsmoduls K_eff – illustriert den direkten Gewinn
    durch hochelastische Flüssigkeiten.
    """
    print("\n[ EXP 04 ] Eigenfrequenz und Positionsfehler …")

    # Zylinderanlage
    A   = 20e-4   # m²   Kolbenfläche  (≈ Ø50 mm)
    m   = 500.0   # kg   bewegte Masse
    V   = 2.0e-3  # m³   Nutzvolumen
    P   = 200e5   # Pa   Betriebsdruck (200 bar)

    K_eff_range = np.linspace(0.4e9, 2.5e9, 300)  # 400 – 2500 MPa

    f_h_vals  = [hydraulic_natural_frequency(K, A, m, V) for K in K_eff_range]
    delta_x   = [compressibility_error(P, V, K, A) * 1e3 for K in K_eff_range]  # mm

    # Vergleich: Flüssigkeitstypen @ 200 bar
    fluid_K = {
        "Mineralöl VG 46 (trocken)":   bulk_modulus_tait_linear(OIL_K0,  OIL_N_TAIT, P),
        "Mineralöl VG 46 (1 % Luft)":  effective_bulk_modulus(bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, P), P, 0.01),
        "Phosphatester HEP":            bulk_modulus_tait_linear(HEP_K0, OIL_N_TAIT, P),
        "Wasserglykol HFC":             bulk_modulus_tait_linear(HFC_K0, OIL_N_TAIT, P),
        "PAO / SHC":                    bulk_modulus_tait_linear(PAO_K0, OIL_N_TAIT, P),
    }

    # CSV
    header = ["K_eff_MPa", "f_h_Hz", "delta_x_mm"]
    rows = [[f"{K/1e6:.1f}", f"{f:.3f}", f"{dx:.4f}"]
            for K, f, dx in zip(K_eff_range, f_h_vals, delta_x)]
    _save_csv("exp04_eigenfrequency_positioning.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 4 – Eigenfrequenz & Positionsfehler (A=20 cm², m=500 kg, V=2 l, p=200 bar)", fontsize=12)

    ax1.plot([K / 1e6 for K in K_eff_range], f_h_vals, "steelblue", lw=2)
    for fname, Kval in fluid_K.items():
        fval = hydraulic_natural_frequency(Kval, A, m, V)
        ax1.axvline(Kval / 1e6, ls=":", color="gray", lw=0.8)
        ax1.scatter([Kval / 1e6], [fval], zorder=5)
        ax1.annotate(fname, (Kval / 1e6, fval), textcoords="offset points",
                     xytext=(5, 4), fontsize=7, rotation=15)
    ax1.set_xlabel("K_eff [MPa]")
    ax1.set_ylabel("Eigenfrequenz f_h [Hz]")
    ax1.set_title("f_h(K_eff)")
    ax1.grid(True, alpha=0.3)

    ax2.plot([K / 1e6 for K in K_eff_range], delta_x, "crimson", lw=2)
    for fname, Kval in fluid_K.items():
        dx = compressibility_error(P, V, Kval, A) * 1e3
        ax2.axvline(Kval / 1e6, ls=":", color="gray", lw=0.8)
        ax2.scatter([Kval / 1e6], [dx], zorder=5)
    ax2.set_xlabel("K_eff [MPa]")
    ax2.set_ylabel("Positionsfehler δx [mm]")
    ax2.set_title("Kompressibilitätsfehler δx(K_eff)")
    ax2.grid(True, alpha=0.3)

    _save_svg(fig, "exp04_eigenfrequency_positioning.svg")
    print("   → exp04_eigenfrequency_positioning.csv / .svg")


# ============================================================================
# EXPERIMENT 5 – Series Elastic Actuation und Cobot-Sicherheit
# ============================================================================

def experiment_05_sea_cobot_safety() -> None:
    """
    Hydraulische Series-Elastic-Actuation (SEA):
      - K_SEA als Funktion des Akkumulatorvolumens für verschiedene K_eff
      - Maximale Kontaktkraft F_max (Cobot-Sicherheit) vs. Einfahrgeschwindigkeit
    """
    print("\n[ EXP 05 ] SEA-Steifigkeit und Cobot-Sicherheit …")

    A = 20e-4     # m² Kolbenfläche
    m = 50.0      # kg Roboterarmlast

    V_acc_L   = np.logspace(-4, 0, 200)  # 0.1 ml – 1000 ml Akkuvolumen
    V_acc_m3  = V_acc_L * 1e-3

    K_eff_fluids = {
        "α = 0 % (ideal)":   bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, 100e5),
        "α = 0.5 %":         effective_bulk_modulus(bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, 100e5), 100e5, 0.005),
        "α = 2 %":           effective_bulk_modulus(bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, 100e5), 100e5, 0.02),
        "Wasserglykol HFC":  bulk_modulus_tait_linear(HFC_K0, OIL_N_TAIT, 100e5),
    }

    v0_range = np.linspace(0.01, 2.0, 200)  # 1 cm/s – 2 m/s Einfahrgeschwindigkeit

    # CSV
    header_sea = ["V_acc_mL"] + [f"K_SEA_{n}_Npm" for n in K_eff_fluids]
    rows_sea = []
    for i, V in enumerate(V_acc_m3):
        row = [f"{V*1e3:.4f}"]
        for K in K_eff_fluids.values():
            row.append(f"{sea_stiffness(K, A, V):.1f}")
        rows_sea.append(row)
    _save_csv("exp05_sea_cobot.csv", header_sea, rows_sea)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 5 – SEA-Steifigkeit K_SEA und Cobot Kontaktkraft F_max", fontsize=13)

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(K_eff_fluids)))
    for (name, K_val), color in zip(K_eff_fluids.items(), colors):
        K_sea_vals = [sea_stiffness(K_val, A, V) / 1e3 for V in V_acc_m3]  # kN/m
        ax1.loglog(V_acc_L * 1e3, K_sea_vals, color=color, lw=2, label=name)
    ax1.set_xlabel("Akkumulatorvolumen [mL]")
    ax1.set_ylabel("SEA-Steifigkeit K_SEA [kN/m]")
    ax1.set_title("Abhängigkeit K_SEA vom Akkuvolumen")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    V_acc_design = 0.5e-3  # 0.5 L Referenzvolumen
    F_limits_N   = {"ISO 10218-1 (250 N)": 250, "Cobot limit (150 N)": 150}
    for (name, K_val), color in zip(K_eff_fluids.items(), colors):
        K_sea = sea_stiffness(K_val, A, V_acc_design)
        F_vals = [max_contact_force(v0, K_sea, m) for v0 in v0_range]
        ax2.plot(v0_range, F_vals, color=color, lw=2, label=name)
    for label, limit in F_limits_N.items():
        ax2.axhline(limit, ls="--", color="black", lw=1.2, label=label)
    ax2.set_xlabel("Einfahrgeschwindigkeit v₀ [m/s]")
    ax2.set_ylabel("Max. Kontaktkraft F_max [N]")
    ax2.set_title(f"F_max(v₀) – Akkumulator V_acc = 0.5 L, m = {m} kg")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)

    _save_svg(fig, "exp05_sea_cobot_safety.svg")
    print("   → exp05_sea_cobot.csv / .svg")


# ============================================================================
# EXPERIMENT 6 – Viskoelastische Frequenzantwort (Zener-Modell)
# ============================================================================

def experiment_06_viscoelastic_frequency() -> None:
    """
    Speichermodul K', Verlustmodul K'' und Verlustfaktor tan δ als Funktion
    der Frequenz für verschiedene Öltypen. Zeigt das viskoelastische Fenster.
    """
    print("\n[ EXP 06 ] Viskoelastisches Frequenzverhalten …")

    E0   = OIL_K0
    E_inf = OIL_K0 * 1.08   # 8 % höher als statischer Wert (typisch)

    # Relaxationszeiten für fünf Viskositätsniveaus
    c_s_oil = sound_speed(isentropic_modulus(OIL_K0, OIL_ALPHA_V, OIL_T_REF, OIL_RHO, OIL_CP), OIL_RHO)

    eta_values = {
        "VG 10  (η = 10 mPa·s)":  0.010,
        "VG 22  (η = 22 mPa·s)":  0.022,
        "VG 46  (η = 46 mPa·s)":  0.046,
        "VG 100 (η = 100 mPa·s)": 0.100,
        "VG 320 (η = 320 mPa·s)": 0.320,
    }

    freqs = np.logspace(3, 12, 600)  # 1 kHz – 1 THz
    omegas = 2.0 * math.pi * freqs

    # CSV
    header = ["f_Hz"]
    for name in eta_values:
        header += [f"Kprime_{name}_GPa", f"Kdoubleprime_{name}_GPa", f"tand_{name}"]
    rows = []
    for i, (f, omega) in enumerate(zip(freqs, omegas)):
        row = [f"{f:.3e}"]
        c_ref = c_s_oil
        for name, eta in eta_values.items():
            tau_R = relaxation_time(eta, OIL_RHO, c_ref)
            Kp  = storage_modulus(E0, E_inf, omega, tau_R)
            Kpp = loss_modulus(E0, E_inf, omega, tau_R)
            td  = loss_factor(E0, E_inf, omega, tau_R)
            row += [f"{Kp/1e9:.6f}", f"{Kpp/1e9:.6f}", f"{td:.6f}"]
        rows.append(row)
    _save_csv("exp06_viscoelastic_frequency.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 6 – Viskoelastisches Frequenzverhalten (Zener-Modell)", fontsize=13)

    colors = plt.cm.cool(np.linspace(0, 1, len(eta_values)))
    for (name, eta), color in zip(eta_values.items(), colors):
        tau_R = relaxation_time(eta, OIL_RHO, c_s_oil)
        Kp_vals  = [storage_modulus(E0, E_inf, w, tau_R) / 1e6 for w in omegas]
        Kpp_vals = [loss_modulus(E0, E_inf, w, tau_R)    / 1e6 for w in omegas]
        ax1.semilogx(freqs, Kp_vals,  color=color, lw=2,      label=f"K' {name}")
        ax1.semilogx(freqs, Kpp_vals, color=color, lw=2, ls="--")
    ax1.set_xlabel("Frequenz [Hz]")
    ax1.set_ylabel("Modul [MPa]")
    ax1.set_title("Speichermodul K' (—) und Verlustmodul K'' (- -)")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)

    for (name, eta), color in zip(eta_values.items(), colors):
        tau_R = relaxation_time(eta, OIL_RHO, c_s_oil)
        f_R   = relaxation_frequency(OIL_RHO, c_s_oil, eta)
        td_vals = [loss_factor(E0, E_inf, w, tau_R) for w in omegas]
        ax2.semilogx(freqs, td_vals, color=color, lw=2, label=f"{name} (f_R={f_R:.1e} Hz)")
        ax2.axvline(f_R, color=color, lw=0.6, ls=":")
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylabel("Verlustfaktor tan δ = K''/K'")
    ax2.set_title("Verlustfaktor tan δ (Relaxationspeak)")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", alpha=0.3)

    _save_svg(fig, "exp06_viscoelastic_frequency.svg")
    print("   → exp06_viscoelastic_frequency.csv / .svg")


# ============================================================================
# EXPERIMENT 7 – Lebensdauer: Temperatur-, Druck- und Viskositätseinfluss
# ============================================================================

def experiment_07_oil_lifetime() -> None:
    """
    Öllebensdauer nach Coffin-Manson-Analogie:
      - t_L(T) – Arrheniuskurve (dramatischer Abfall mit Temperatur)
      - t_L(delta_eta_rate, T) – Konturdiagramm
    """
    print("\n[ EXP 07 ] Öllebensdauer – Temperatur- und Alterungseinfluss …")

    C_hyd  = 1000.0    # h   flüssigkeitsspez. Konstante
    beta   = 2.3       # –   Ermüdungsexponent
    E_a0   = 0.65      # eV  Aktivierungsenergie (Mineralöl-Oxidation)
    eta_0  = OIL_ETA   # Pa·s

    # 1) t_L(T) für verschiedene Viskositätsabbauraten
    temps_C = np.linspace(30, 110, 200)
    temps_K = temps_C + 273.15

    delta_eta_rates = {
        "niedrig (0.1 %/h)":   0.001 * eta_0,
        "mittel  (0.3 %/h)":   0.003 * eta_0,
        "typisch (1 %/h)":     0.010 * eta_0,
        "hoch    (3 %/h)":     0.030 * eta_0,
    }

    # 2) Kontour: T × delta_eta_rate
    T_grid_C, rate_grid = np.meshgrid(
        np.linspace(40, 100, 60),
        np.logspace(-4, -2, 60) * eta_0,  # 0.01 % – 1 % pro h der Nennvisk.
    )
    T_grid_K = T_grid_C + 273.15
    Z_lifetime = np.zeros_like(T_grid_K)
    for i in range(T_grid_K.shape[0]):
        for j in range(T_grid_K.shape[1]):
            try:
                Z_lifetime[i, j] = oil_lifetime(C_hyd, rate_grid[i, j], eta_0, beta, E_a0, T_grid_K[i, j])
            except Exception:
                Z_lifetime[i, j] = np.nan

    # CSV
    header = ["T_C"] + [f"t_L_{n}_h" for n in delta_eta_rates]
    rows = []
    for i, T in enumerate(temps_C):
        row = [f"{T:.1f}"]
        for drate in delta_eta_rates.values():
            try:
                tL = oil_lifetime(C_hyd, drate, eta_0, beta, E_a0, temps_K[i])
            except Exception:
                tL = float("nan")
            row.append(f"{tL:.1f}")
        rows.append(row)
    _save_csv("exp07_oil_lifetime.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 7 – Öllebensdauer: Arrhenius & Konturdiagramm", fontsize=13)

    colors = plt.cm.autumn_r(np.linspace(0.1, 0.9, len(delta_eta_rates)))
    for (name, drate), color in zip(delta_eta_rates.items(), colors):
        tL_vals = []
        for T in temps_K:
            try:
                tL_vals.append(oil_lifetime(C_hyd, drate, eta_0, beta, E_a0, T))
            except Exception:
                tL_vals.append(float("nan"))
        ax1.semilogy(temps_C, tL_vals, color=color, lw=2, label=name)
    ax1.axhline(2000, ls="--", color="green", lw=1.2, label="2000 h (Wechselintervall)")
    ax1.axhline(500,  ls="--", color="red",   lw=1.2, label="500 h (kritisch)")
    ax1.set_xlabel("Temperatur [°C]")
    ax1.set_ylabel("Lebensdauer t_L [h]")
    ax1.set_title("Arrhenius-Abfall der Öllebensdauer")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(30, 110)

    cf = ax2.contourf(T_grid_C, rate_grid / eta_0 * 100.0, Z_lifetime,
                      levels=np.logspace(1, 5, 30), locator=matplotlib.ticker.LogLocator(), cmap="YlGn")
    fig.colorbar(cf, ax=ax2, label="t_L [h]")
    ax2.set_xlabel("Betriebstemperatur [°C]")
    ax2.set_ylabel("Abbaurate Δη [% der Nennvisk. / h]")
    ax2.set_title("Lebensdauer-Kontur (Coffin-Manson)")
    ax2.set_yscale("log")

    _save_svg(fig, "exp07_oil_lifetime.svg")
    print("   → exp07_oil_lifetime.csv / .svg")


# ============================================================================
# EXPERIMENT 8 – Modulalterung und Versagensraten
# ============================================================================

def experiment_08_aging_failure_rates() -> None:
    """
    Zeitlicher Verlauf des Kompressionsmoduls durch Ölalterung und
    Vergleich der drei Versagensmechanismen (Oxidation, Reibung, Kavitation).
    """
    print("\n[ EXP 08 ] Alterung des Kompressionsmoduls und Versagensraten …")

    # Modell: lineare Viskositätsdrift mit Zeit
    hours = np.linspace(0, 5000, 500)   # Betriebsstunden
    eta_0 = OIL_ETA
    drift_rate = 0.0003   # Pa·s / h   (≈ 0.65 %/h kumulative Zunahme)

    delta_eta_vals = drift_rate * hours
    K_aging = [
        bulk_modulus_aging(OIL_K0, de, eta_0, delta_K_coeff=0.15)
        for de in delta_eta_vals
    ]

    # Betriebstemperatur driftet durch Alterung leicht nach oben
    T_opt  = OIL_T_REF          # optimale Betriebstemperatur (40 °C)
    T_min  = 293.15             # 20 °C Mindesttemperatur für Schmierung
    T0_K   = 15.0               # charakteristische Temperaturdifferenz [K]
    P_min  = 10e5               # Pa  Mindestsystemdruck
    P_vap  = 500.0              # Pa  Dampfdruck Mineralöl @ 40 °C
    P0_cav = 5e5                # Pa  charakteristischer Kavitationsdruck

    # Temperatur steigt proportional zur kumulativen Viskositätsänderung
    T_vals = [T_opt + 20.0 * (de / (0.2 * eta_0)) for de in delta_eta_vals]

    lambda_ox  = [failure_rate_oxidation(T, T_opt, T0_K) for T in T_vals]
    lambda_fri = [failure_rate_friction(T, T_min, T0_K)  for T in T_vals]
    lambda_cav = [failure_rate_cavitation(P_min, P_vap, P0_cav) for _ in hours]
    lambda_tot = [
        total_failure_rate(T, T_opt, T_min, T0_K, P_min, P_vap, P0_cav)
        for T in T_vals
    ]

    # Schwellwerte (OilCondition)
    t_yellow = next((h for K, h in zip(K_aging, hours) if K / OIL_K0 < 0.90), None)
    t_red    = next((h for K, h in zip(K_aging, hours) if K / OIL_K0 < 0.80), None)

    # CSV
    header = ["t_h", "delta_eta_Pas", "K_alt_MPa", "K_ratio", "lambda_ox", "lambda_fri", "lambda_cav", "lambda_tot"]
    rows = [
        [f"{h:.1f}", f"{de:.5f}", f"{K/1e6:.4f}", f"{K/OIL_K0:.4f}",
         f"{lo:.4f}", f"{lf:.4f}", f"{lc:.4f}", f"{lt:.4f}"]
        for h, de, K, lo, lf, lc, lt in zip(hours, delta_eta_vals, K_aging, lambda_ox, lambda_fri, lambda_cav, lambda_tot)
    ]
    _save_csv("exp08_aging_failure_rates.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 8 – Modulalterung und Versagensraten im Betrieb", fontsize=13)

    ax1.plot(hours, [K / 1e6 for K in K_aging], "steelblue", lw=2.5, label="K_alt(t)")
    ax1.axhline(OIL_K0 * 0.90 / 1e6, ls="--", color="orange", lw=1.5, label="YELLOW-Grenze (90 %)")
    ax1.axhline(OIL_K0 * 0.80 / 1e6, ls="--", color="red",    lw=1.5, label="RED-Grenze (80 %)")
    if t_yellow:
        ax1.axvline(t_yellow, ls=":", color="orange", lw=1)
        ax1.annotate(f"YELLOW\n{t_yellow:.0f} h", (t_yellow, OIL_K0 * 0.85 / 1e6),
                     fontsize=8, color="orange")
    if t_red:
        ax1.axvline(t_red, ls=":", color="red", lw=1)
        ax1.annotate(f"RED\n{t_red:.0f} h", (t_red, OIL_K0 * 0.75 / 1e6),
                     fontsize=8, color="red")
    ax1.set_xlabel("Betriebszeit [h]")
    ax1.set_ylabel("K_alt [MPa]")
    ax1.set_title("Alterungsbedingter Rückgang des Kompressionsmoduls")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(hours, lambda_ox,  "tomato",    lw=2, label="λ_Oxidation")
    ax2.semilogy(hours, lambda_fri, "goldenrod", lw=2, label="λ_Reibung")
    ax2.semilogy(hours, [lc for lc in lambda_cav], "steelblue", lw=2, ls="--", label="λ_Kavitation (konst.)")
    ax2.semilogy(hours, lambda_tot, "black",     lw=2.5, label="λ_gesamt")
    ax2.set_xlabel("Betriebszeit [h]")
    ax2.set_ylabel("Versagensrate λ [1/h]")
    ax2.set_title("Versagensraten-Akkumulation (drei Mechanismen)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    _save_svg(fig, "exp08_aging_failure_rates.svg")
    print("   → exp08_aging_failure_rates.csv / .svg")


# ============================================================================
# EXPERIMENT 9 – Kavitation und Oberflächenspannung
# ============================================================================

def experiment_09_cavitation_surface() -> None:
    """
    Laplace-Druck, Kavitationsschwelle und Gasgehalt-Sicherheitsanalyse:
      - ΔP_L(r)  – Laplace-Kurve über Blasenradius
      - Kavitationsdruck P_Kav(r) für verschiedene Dampfdrücke / Öltypen
    """
    print("\n[ EXP 09 ] Kavitation und Oberflächenspannung …")

    r_nm = np.logspace(0, 4, 400)   # 1 nm – 10 µm → in m
    r_m  = r_nm * 1e-9

    # Oberflächenspannungen typischer Hydraulikflüssigkeiten [N/m]
    fluids_gamma = {
        "Mineralöl VG 46":   0.030,
        "Phosphatester HEP": 0.040,
        "Wasserglykol HFC":  0.065,
        "Esteröl HEES":      0.032,
        "Silikonöl":         0.021,
    }

    P_vapor_oil   = 5.0e2   # Pa  (~0.005 mbar – typisch Mineralöl @ 40°C)
    P_vapor_water = 7.4e3   # Pa  (~74 mbar – Wasser @ 40°C)

    # CSV
    header = ["r_nm"] + [f"DeltaP_L_{n}_Pa" for n in fluids_gamma] + \
             [f"P_kav_{n}_bar" for n in fluids_gamma]
    rows = []
    for i, r in enumerate(r_m):
        row = [f"{r_nm[i]:.2f}"]
        for gamma in fluids_gamma.values():
            row.append(f"{laplace_pressure(gamma, r):.2f}")
        for gamma in fluids_gamma.values():
            row.append(f"{cavitation_pressure(P_vapor_oil, gamma, r)/1e5:.4f}")
        rows.append(row)
    _save_csv("exp09_cavitation_surface.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 9 – Laplace-Druck und Kavitationsschwelle", fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(fluids_gamma)))
    for (name, gamma), color in zip(fluids_gamma.items(), colors):
        lp = [laplace_pressure(gamma, r) / 1e6 for r in r_m]  # MPa
        ax1.loglog(r_nm, lp, color=color, lw=2, label=f"{name} (γ={gamma*1000:.0f} mN/m)")
    ax1.axhline(0.1, ls="--", color="gray", lw=1, label="0.1 MPa (1 bar)")
    ax1.axhline(10,  ls=":",  color="gray", lw=1, label="10 MPa (100 bar)")
    ax1.set_xlabel("Blasenradius r [nm]")
    ax1.set_ylabel("Laplace-Druck ΔP_L [MPa]")
    ax1.set_title("Laplace-Innendruck sphärischer Blasen")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)

    for (name, gamma), color in zip(fluids_gamma.items(), colors):
        Pkav = [cavitation_pressure(P_vapor_oil, gamma, r) / 1e5 for r in r_m]  # bar
        ax2.semilogx(r_nm, Pkav, color=color, lw=2, label=name)
    # Systemdrücke als Referenz
    for label, P_ref in [("Systemdruck 50 bar", 50), ("Systemdruck 200 bar", 200)]:
        ax2.axhline(P_ref, ls="--", lw=1.2, color="black", label=label)
    ax2.set_xlabel("Keimradius r_Keim [nm]")
    ax2.set_ylabel("Kavitationsdruck P_Kav [bar]")
    ax2.set_title("Kavitationsschwelle P_Kav(r, γ)")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(0, 300)

    _save_svg(fig, "exp09_cavitation_surface.svg")
    print("   → exp09_cavitation_surface.csv / .svg")


# ============================================================================
# EXPERIMENT 10 – Mischungsregeln und Flüssigkeitstypen-Vergleich
# ============================================================================

def experiment_10_mixing_fluid_comparison() -> None:
    """
    Reuss/Voigt/Maxwell-Mischungsmodelle für Öl-Wasser-Emulsionen und
    vollständiger Flüssigkeitsvergleich K – c_s – Lebensdauer – Eigenfrequenz.
    """
    print("\n[ EXP 10 ] Mischungsregeln und Flüssigkeitstypvergleich …")

    phi = np.linspace(0, 1, 300)   # Wasseranteil 0 – 100 %

    K_water = 2.18e9   # Pa  Wasser @ 20 °C
    K_oil   = OIL_K0

    K_reuss  = [reuss_modulus(K_water, K_oil, p)          for p in phi]   # Wasser = Matrix
    K_voigt  = [voigt_modulus(K_water, K_oil, p)          for p in phi]
    K_max    = [maxwell_emulsion_modulus(K_water, K_oil, p) for p in phi] # Öl in Wasser

    # Flüssigkeitsvergleich (Balkendiagramm)
    fluid_data = {
        "Mineralöl\nVG 46 (HLP)": {
            "K_MPa": bulk_modulus_tait_linear(OIL_K0,  OIL_N_TAIT, 100e5) / 1e6,
            "c_s_ms": sound_speed(isentropic_modulus(OIL_K0, OIL_ALPHA_V, OIL_T_REF, OIL_RHO, OIL_CP), OIL_RHO),
            "f_h_Hz": hydraulic_natural_frequency(bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, 100e5), 20e-4, 200.0, 2e-3),
        },
        "Phosphat-\nester HEP": {
            "K_MPa": bulk_modulus_tait_linear(HEP_K0, OIL_N_TAIT, 100e5) / 1e6,
            "c_s_ms": sound_speed(isentropic_modulus(HEP_K0, HEP_ALPHA_V, OIL_T_REF, HEP_RHO, HEP_CP), HEP_RHO),
            "f_h_Hz": hydraulic_natural_frequency(bulk_modulus_tait_linear(HEP_K0, OIL_N_TAIT, 100e5), 20e-4, 200.0, 2e-3),
        },
        "Wasserglykol\nHFC": {
            "K_MPa": bulk_modulus_tait_linear(HFC_K0, OIL_N_TAIT, 100e5) / 1e6,
            "c_s_ms": sound_speed(isentropic_modulus(HFC_K0, HFC_ALPHA_V, OIL_T_REF, HFC_RHO, HFC_CP), HFC_RHO),
            "f_h_Hz": hydraulic_natural_frequency(bulk_modulus_tait_linear(HFC_K0, OIL_N_TAIT, 100e5), 20e-4, 200.0, 2e-3),
        },
        "Esteröl\nHEES": {
            "K_MPa": bulk_modulus_tait_linear(HEES_K0, OIL_N_TAIT, 100e5) / 1e6,
            "c_s_ms": sound_speed(isentropic_modulus(HEES_K0, 6.5e-4, OIL_T_REF, HEES_RHO, 1950.0), HEES_RHO),
            "f_h_Hz": hydraulic_natural_frequency(bulk_modulus_tait_linear(HEES_K0, OIL_N_TAIT, 100e5), 20e-4, 200.0, 2e-3),
        },
        "PAO / SHC": {
            "K_MPa": bulk_modulus_tait_linear(PAO_K0, OIL_N_TAIT, 100e5) / 1e6,
            "c_s_ms": sound_speed(isentropic_modulus(PAO_K0, 6.2e-4, OIL_T_REF, PAO_RHO, 2000.0), PAO_RHO),
            "f_h_Hz": hydraulic_natural_frequency(bulk_modulus_tait_linear(PAO_K0, OIL_N_TAIT, 100e5), 20e-4, 200.0, 2e-3),
        },
    }

    # CSV
    header = ["phi_pct", "K_Reuss_MPa", "K_Voigt_MPa", "K_Maxwell_MPa"]
    rows = [[f"{p*100:.1f}", f"{r/1e6:.3f}", f"{v/1e6:.3f}", f"{mx/1e6:.3f}"]
            for p, r, v, mx in zip(phi, K_reuss, K_voigt, K_max)]
    _save_csv("exp10_mixing_fluid_comparison.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 10 – Mischungsregeln & Flüssigkeitsvergleich", fontsize=13)

    ax1.plot(phi * 100, [K / 1e6 for K in K_reuss],  "b-",  lw=2, label="Reuss (untere Schranke)")
    ax1.plot(phi * 100, [K / 1e6 for K in K_voigt],  "r-",  lw=2, label="Voigt (obere Schranke)")
    ax1.plot(phi * 100, [K / 1e6 for K in K_max],    "g--", lw=2, label="Maxwell (Emulsion Öl→Wasser)")
    ax1.set_xlabel("Ölanteil φ_Öl [%]")
    ax1.set_ylabel("Kompressionsmodul K_eff [MPa]")
    ax1.set_title("Mischungsregeln: Öl-Wasser-System\n(K_water=2180 MPa, K_oil=1500 MPa)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fluid_names   = list(fluid_data.keys())
    K_vals        = [d["K_MPa"]  for d in fluid_data.values()]
    c_vals        = [d["c_s_ms"] for d in fluid_data.values()]
    f_h_vals_2    = [d["f_h_Hz"] for d in fluid_data.values()]
    x = np.arange(len(fluid_names))
    w = 0.25
    ax2.bar(x - w, K_vals,     w, label="K_eff @ 100 bar [MPa]",  color="steelblue",  alpha=0.85)
    ax2.bar(x,     c_vals,     w, label="c_s [m/s]",               color="darkorange", alpha=0.85)
    ax2.bar(x + w, f_h_vals_2, w, label="f_h [Hz]",                color="green",      alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fluid_names, fontsize=8)
    ax2.set_ylabel("Wert (Einheit je nach Kenngröße)")
    ax2.set_title("Flüssigkeitsvergleich: K, c_s, f_h\n(A=20 cm², m=200 kg, V=2 l)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    _save_svg(fig, "exp10_mixing_fluid_comparison.svg")
    print("   → exp10_mixing_fluid_comparison.csv / .svg")


# ============================================================================
# EXPERIMENT 11 – Barus-Viskosität und Schallgeschwindigkeit unter Hochdruck
# ============================================================================

def experiment_11_barus_high_pressure() -> None:
    """
    Druckabhängigkeit der Viskosität (Barus) und ihr Einfluss auf die
    Relaxationsfrequenz – relevant für HD-Ölfilter und Zahnradpumpen.
    """
    print("\n[ EXP 11 ] Barus-Viskosität und Hochdruckeinfluss …")

    P_range_bar = np.linspace(0, 600, 300)
    P_range_pa  = P_range_bar * 1e5

    alpha_P_values = {
        "Mineralöl VG 10   (α_P = 1.5e-8 /Pa)": 1.5e-8,
        "Mineralöl VG 46   (α_P = 2.0e-8 /Pa)": 2.0e-8,
        "Mineralöl VG 100  (α_P = 2.5e-8 /Pa)": 2.5e-8,
        "Phosphatester HEP (α_P = 1.0e-8 /Pa)": 1.0e-8,
        "PAO / SHC         (α_P = 1.8e-8 /Pa)": 1.8e-8,
    }

    c_s_base = sound_speed(
        isentropic_modulus(OIL_K0, OIL_ALPHA_V, OIL_T_REF, OIL_RHO, OIL_CP), OIL_RHO
    )

    rows = []
    header = ["P_bar"] + [f"eta_{n}_Pas" for n in alpha_P_values] + \
                         [f"f_R_{n}_Hz"  for n in alpha_P_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 11 – Barus-Viskositätsanstieg und Relaxationsfrequenz", fontsize=13)

    colors = plt.cm.copper(np.linspace(0.1, 0.9, len(alpha_P_values)))
    for i, P_pa in enumerate(P_range_pa):
        row = [f"{P_range_bar[i]:.1f}"]
        for alpha_P in alpha_P_values.values():
            eta_val = barus_viscosity(OIL_ETA, alpha_P, P_pa)
            f_R_val = relaxation_frequency(OIL_RHO, c_s_base, eta_val)
            row += [f"{eta_val:.6f}", f"{f_R_val:.3e}"]
        rows.append(row)
    _save_csv("exp11_barus_high_pressure.csv", header, rows)

    for (name, alpha_P), color in zip(alpha_P_values.items(), colors):
        eta_vals = [barus_viscosity(OIL_ETA, alpha_P, P) for P in P_range_pa]
        f_R_vals = [relaxation_frequency(OIL_RHO, c_s_base, eta) for eta in eta_vals]
        ax1.semilogy(P_range_bar, [e * 1e3 for e in eta_vals], color=color, lw=2, label=name.split("(")[0].strip())
        ax2.semilogy(P_range_bar, f_R_vals, color=color, lw=2)
    ax1.set_xlabel("Druck [bar]")
    ax1.set_ylabel("Dynamische Viskosität η [mPa·s]")
    ax1.set_title("Barus: η(P) – exponentieller Anstieg")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Druck [bar]")
    ax2.set_ylabel("Relaxationsfrequenz f_R [Hz]")
    ax2.set_title("Abnahme der Relaxationsfrequenz mit Druck")
    ax2.legend(list(alpha_P_values.keys()), fontsize=7)
    ax2.grid(True, which="both", alpha=0.3)

    _save_svg(fig, "exp11_barus_high_pressure.svg")
    print("   → exp11_barus_high_pressure.csv / .svg")


# ============================================================================
# EXPERIMENT 12 – Ultraschall-Monitoring und Online-Ölzustandsüberwachung
# ============================================================================

def experiment_12_ultrasonic_monitoring() -> None:
    """
    Simulation einer Ultraschall-Laufzeitmessung über die Betriebszeit:
      - Ermittlung von K_eff aus Laufzeitverschiebung Δt(t)
      - OilCondition-Ampelbewertung über die Betriebszeit
    """
    print("\n[ EXP 12 ] Ultraschall-Monitoring und Öl-Zustandsüberwachung …")

    L = 0.05     # m  Messstreckenabstand (50 mm Wandabstand)
    hours = np.linspace(0, 4000, 400)

    # Simuliertes Alterungsszenario
    # – K_eff sinkt langsam (Gasbildung + Alterung)
    # – η steigt zunächst dann fällt
    # – TAN steigt
    K_fresh  = OIL_K0
    eta_0    = OIL_ETA

    def K_at(t: float) -> float:
        return K_fresh * (1.0 - 0.10 * math.sqrt(t / 4000.0) - 0.03 * (t / 4000.0) ** 2)

    def eta_at(t: float) -> float:
        return eta_0 * (1.0 + 0.25 * (t / 4000.0) - 0.05 * (t / 4000.0) ** 3)

    def TAN_at(t: float) -> float:
        return 0.3 + 1.9 * (t / 4000.0) ** 1.5  # steigt auf ~2.2

    K_vals   = [K_at(t)   for t in hours]
    eta_vals = [eta_at(t) for t in hours]
    TAN_vals = [TAN_at(t) for t in hours]

    # Schallgeschwindigkeit und Laufzeit aus K_eff
    rho = OIL_RHO
    c_s_vals  = [sound_speed(K, rho) for K in K_vals]
    dt_vals   = [L / c for c in c_s_vals]  # Laufzeit in µs-Bereich

    # Rückgerechneter Modul aus Laufzeit (sollte mit K_vals übereinstimmen)
    K_eff_meas = [ultrasonic_bulk_modulus(rho, L, dt) for dt in dt_vals]

    # Konditionsbewertung
    conditions = [
        OilCondition.evaluate(K, K_fresh, eta, eta_0, TAN)
        for K, eta, TAN in zip(K_vals, eta_vals, TAN_vals)
    ]
    cond_numeric = {OilCondition.GREEN: 1, OilCondition.YELLOW: 2, OilCondition.RED: 3}
    cond_vals = [cond_numeric[c] for c in conditions]

    # t bei erstem YELLOW / RED
    t_yellow = next((t for t, c in zip(hours, conditions) if c == OilCondition.YELLOW), None)
    t_red    = next((t for t, c in zip(hours, conditions) if c == OilCondition.RED),    None)

    # CSV
    header = ["t_h", "K_eff_MPa", "K_meas_MPa", "c_s_ms", "dt_ns", "eta_Pas", "TAN", "condition"]
    rows = [
        [f"{t:.1f}", f"{K/1e6:.4f}", f"{Km/1e6:.4f}", f"{c:.2f}", f"{dt*1e9:.4f}", f"{eta:.5f}", f"{TAN:.4f}", cond]
        for t, K, Km, c, dt, eta, TAN, cond in zip(hours, K_vals, K_eff_meas, c_s_vals, dt_vals, eta_vals, TAN_vals, conditions)
    ]
    _save_csv("exp12_ultrasonic_monitoring.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 12 – Ultraschall-Monitoring: K_eff(t) und Öl-Konditionsbewertung", fontsize=13)

    ax1.plot(hours, [K / 1e6 for K in K_vals], "steelblue", lw=2.5, label="K_eff (Simulation)")
    ax1.plot(hours, [K / 1e6 for K in K_eff_meas], "r--", lw=1.5, alpha=0.7, label="K_eff (aus Δt rekonstruiert)")
    ax1.axhline(K_fresh * 0.90 / 1e6, ls="--", color="orange", lw=1.2, label="YELLOW 90 %")
    ax1.axhline(K_fresh * 0.80 / 1e6, ls="--", color="red",    lw=1.2, label="RED 80 %")
    ax1.set_xlabel("Betriebszeit [h]")
    ax1.set_ylabel("K_eff [MPa]")
    ax1.set_title("Kompressionsmodul über Betriebszeit")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    cmap_c = {1: "limegreen", 2: "orange", 3: "crimson"}
    prev = cond_vals[0]
    seg_start = hours[0]
    for i, (t, cv) in enumerate(zip(hours[1:], cond_vals[1:]), 1):
        if cv != prev or i == len(hours) - 1:
            ax2.axvspan(seg_start, t, color=cmap_c[prev], alpha=0.35)
            seg_start = t
            prev = cv
    ax2.plot(hours, [v * 33.3 for v in cond_vals], "k-", lw=1.5, label="Konditionsstufe (1=G, 2=Y, 3=R)")
    ax2.plot(hours, TAN_vals, "purple", lw=2, label="TAN [mgKOH/g]")
    ax2.plot(hours, [(e - eta_0) / eta_0 * 100 for e in eta_vals], "saddlebrown", lw=2, label="Δη [% von η₀]")
    if t_yellow:
        ax2.axvline(t_yellow, color="orange", ls=":", lw=1.5, label=f"YELLOW @ {t_yellow:.0f} h")
    if t_red:
        ax2.axvline(t_red,    color="red",    ls=":", lw=1.5, label=f"RED @ {t_red:.0f} h")
    ax2.set_xlabel("Betriebszeit [h]")
    ax2.set_ylabel("Kenngröße (normiert bzw. [mgKOH/g] oder [%])")
    ax2.set_title("Konditionsindikator: TAN, Δη und Ampelbewertung")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    _save_svg(fig, "exp12_ultrasonic_monitoring.svg")
    print("   → exp12_ultrasonic_monitoring.csv / .svg")


# ============================================================================
# EXPERIMENT 13 – Energierückgewinnung (Rekuperation) mit Druckakkumulator
# ============================================================================

def experiment_13_recuperation_energy() -> None:
    """
    Rekuperationsgrad η_Rek als Funktion von K_eff, Akkumulatorvolumen und
    Verlustenergien – zeigt die energetische Überlegenheit hochelastischer Öle.
    """
    print("\n[ EXP 13 ] Rekuperation und Energiespeicherung …")

    K_eff_range = np.linspace(0.5e9, 2.5e9, 300)
    V_acc_vals  = [0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3]  # 0.5 – 5 L Akkumulator
    E_loss      = 500.0   # J pro Hub (Dichtungsreibung + Drosselverl.)

    # 2D: V_acc × K_eff
    V_grid, K_grid = np.meshgrid(
        np.linspace(0.1e-3, 5.0e-3, 60),
        np.linspace(0.5e9,  2.5e9,  60),
    )
    eta_grid = np.vectorize(recuperation_efficiency)(K_grid, V_grid, E_loss)

    # CSV
    header = ["K_eff_MPa"] + [f"eta_Rek_V{int(V*1e3*10)/10}L" for V in V_acc_vals]
    rows = []
    for K in K_eff_range:
        row = [f"{K/1e6:.2f}"]
        for V in V_acc_vals:
            row.append(f"{recuperation_efficiency(K, V, E_loss):.4f}")
        rows.append(row)
    _save_csv("exp13_recuperation_energy.csv", header, rows)

    # Kompressionsenergie für verschiedene Flüssigkeiten
    V_comp  = 2.0e-3   # m³
    P_range = np.linspace(0, 350e5, 300)  # 0–350 bar
    fluid_K_vals = {
        "Mineralöl VG 46": OIL_K0,
        "Wasserglykol HFC": HFC_K0,
        "Esteröl HEES": HEES_K0,
        "PAO / SHC": PAO_K0,
    }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 13 – Rekuperationsgrad und gespeicherte Kompressionsenergie", fontsize=13)

    cf = ax1.contourf(V_grid * 1e3, K_grid / 1e6, eta_grid * 100,
                      levels=np.linspace(0, 100, 21), cmap="YlGnBu")
    fig.colorbar(cf, ax=ax1, label="η_Rek [%]")
    ax1.set_xlabel("Akkumulatorvolumen V_acc [L]")
    ax1.set_ylabel("K_eff [MPa]")
    ax1.set_title(f"Rekuperationsgrad η_Rek (E_Verlust = {E_loss:.0f} J)")

    colors = plt.cm.Set2(np.linspace(0, 0.9, len(fluid_K_vals)))
    for (name, K_f), color in zip(fluid_K_vals.items(), colors):
        E_vals = [compression_energy(P, V_comp, K_f) / 1e3 for P in P_range]  # kJ
        ax2.plot(P_range / 1e5, E_vals, color=color, lw=2, label=name)
    ax2.set_xlabel("Druck [bar]")
    ax2.set_ylabel("Kompressionsenergie E [kJ]")
    ax2.set_title(f"Gespeicherte Kompressionsenergie (V = {V_comp*1e3:.0f} L)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _save_svg(fig, "exp13_recuperation_energy.svg")
    print("   → exp13_recuperation_energy.csv / .svg")


# ============================================================================
# EXPERIMENT 14 – Scherabbau von VII-Additiven und Modulrückgang
# ============================================================================

def experiment_14_shear_degradation() -> None:
    """
    Scherabbau von Viskositätsindex-Verbesserern (VII) und der korrespondierende
    Rückgang des effektiven Kompressionsmoduls mit der Betriebszeit.
    """
    print("\n[ EXP 14 ] Scherabbau von VII-Additiven …")

    hours   = np.linspace(0, 3000, 300)
    eta_0   = OIL_ETA
    eta_basis = 0.025  # Pa·s  Basisöl-Viskosität ohne VI-Verbesserer
    tau_sch_values = {
        "schneller Abbau  (τ=200 h)":   200,
        "mittlerer Abbau  (τ=500 h)":   500,
        "langsamer Abbau  (τ=1000 h)": 1000,
        "sehr langsam     (τ=2000 h)": 2000,
    }

    # Schallgeschwindigkeit aus K0 (druckunabhängige Näherung)
    c_s_base = sound_speed(isentropic_modulus(OIL_K0, OIL_ALPHA_V, OIL_T_REF, OIL_RHO, OIL_CP), OIL_RHO)

    # CSV
    header = ["t_h"] + [f"eta_{n}_Pas" for n in tau_sch_values] + \
                       [f"K_alt_{n}_MPa" for n in tau_sch_values]
    rows = []
    for t in hours:
        row = [f"{t:.1f}"]
        etas = []
        for tau_s in tau_sch_values.values():
            e = shear_degradation_viscosity(eta_0, eta_basis, t, tau_s)
            etas.append(e)
            row.append(f"{e:.5f}")
        for e in etas:
            delta_eta = eta_0 - e
            K_a = bulk_modulus_aging(OIL_K0, delta_eta, eta_0, 0.15)
            row.append(f"{K_a/1e6:.4f}")
        rows.append(row)
    _save_csv("exp14_shear_degradation.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 14 – VII-Scherabbau und Kompressionsmodul-Rückgang", fontsize=13)

    colors = plt.cm.winter(np.linspace(0.1, 0.9, len(tau_sch_values)))
    for (name, tau_s), color in zip(tau_sch_values.items(), colors):
        eta_vals   = [shear_degradation_viscosity(eta_0, eta_basis, t, tau_s) for t in hours]
        K_alt_vals = [bulk_modulus_aging(OIL_K0, eta_0 - e, eta_0, 0.15) for e in eta_vals]
        ax1.plot(hours, [e * 1000 for e in eta_vals], color=color, lw=2, label=name)
        ax2.plot(hours, [K / 1e6 for K in K_alt_vals], color=color, lw=2, label=name)
    ax1.axhline(eta_basis * 1000, ls="--", color="black", lw=1, label="Basisöl-η")
    ax1.set_xlabel("Betriebszeit [h]")
    ax1.set_ylabel("Dynamische Viskosität η [mPa·s]")
    ax1.set_title("VII-Scherabbau: η(t) für verschiedene τ_Sch")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.axhline(OIL_K0 * 0.90 / 1e6, ls="--", color="orange", lw=1.2, label="YELLOW-Grenze 90 %")
    ax2.axhline(OIL_K0 * 0.80 / 1e6, ls="--", color="red",    lw=1.2, label="RED-Grenze 80 %")
    ax2.set_xlabel("Betriebszeit [h]")
    ax2.set_ylabel("Kompressionsmodul K_alt [MPa]")
    ax2.set_title("Korrespondierender Modulrückgang durch Scherabbau")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    _save_svg(fig, "exp14_shear_degradation.svg")
    print("   → exp14_shear_degradation.csv / .svg")


# ============================================================================
# EXPERIMENT 15 – Parameterraum-Sensitivitätsstudie: K_eff-Einflussgrößen
# ============================================================================

def experiment_15_sensitivity_study() -> None:
    """
    Tornado-Diagramm und 2D-Sensitivitätsstudie: Welche Parameter haben den
    größten Einfluss auf den effektiven Kompressionsmodul K_eff und die
    hydraulische Eigenfrequenz f_h?
    """
    print("\n[ EXP 15 ] Sensitivitätsstudie K_eff und f_h …")

    # Nominale Parameter
    P_nom    = 100e5   # Pa
    T_nom    = OIL_T_REF
    alpha_nom = 0.005  # 0.5 % Luftgehalt
    A_nom    = 20e-4
    m_nom    = 200.0
    V_nom    = 2.0e-3

    def K_nominal() -> float:
        K_T = bulk_modulus_tait_linear(OIL_K0, OIL_N_TAIT, P_nom)
        return effective_bulk_modulus(K_T, P_nom, alpha_nom)

    def f_nominal() -> float:
        return hydraulic_natural_frequency(K_nominal(), A_nom, m_nom, V_nom)

    K_nom  = K_nominal()
    f_nom  = f_nominal()

    # Einflussgrößen: ±20 % Variation
    params = ["P [Pa]", "α [–]", "T [K]", "A [m²]", "m [kg]", "V [m³]"]
    delta  = 0.20   # ±20 %

    K_sens_lo, K_sens_hi = [], []
    f_sens_lo, f_sens_hi = [], []

    def K_varied(**kw) -> float:
        P    = kw.get("P",    P_nom)
        alph = kw.get("alpha", alpha_nom)
        T    = kw.get("T",    T_nom)
        K_T  = bulk_modulus_temperature(OIL_K0, OIL_BETA_K, T, OIL_T_REF)
        K_Tp = bulk_modulus_tait_linear(K_T, OIL_N_TAIT, P)
        return effective_bulk_modulus(K_Tp, P if P > 1 else 1.0, alph)

    def f_varied(**kw) -> float:
        A = kw.get("A", A_nom)
        m = kw.get("m", m_nom)
        V = kw.get("V", V_nom)
        K = K_varied(**{k: v for k, v in kw.items() if k not in ("A", "m", "V")})
        return hydraulic_natural_frequency(K, A, m, V)

    variations = [
        {"P": P_nom * (1 + delta)},    # Druck hoch
        {"alpha": alpha_nom * (1 + delta)},
        {"T": T_nom * (1 + delta)},
        {"A": A_nom * (1 + delta)},
        {"m": m_nom * (1 + delta)},
        {"V": V_nom * (1 + delta)},
    ]
    variations_lo = [
        {"P": P_nom * (1 - delta)},
        {"alpha": max(alpha_nom * (1 - delta), 1e-6)},
        {"T": T_nom * (1 - delta)},
        {"A": A_nom * (1 - delta)},
        {"m": m_nom * (1 - delta)},
        {"V": V_nom * (1 - delta)},
    ]

    for v_hi, v_lo in zip(variations, variations_lo):
        K_sens_hi.append((K_varied(**v_hi) - K_nom) / K_nom * 100)
        K_sens_lo.append((K_varied(**v_lo) - K_nom) / K_nom * 100)
        f_sens_hi.append((f_varied(**v_hi) - f_nom) / f_nom * 100)
        f_sens_lo.append((f_varied(**v_lo) - f_nom) / f_nom * 100)

    # 2D-Studie K_eff(T, α)
    T_range_C   = np.linspace(0, 100, 60)
    alpha_range = np.linspace(0, 0.03, 60)
    T_grid, A_grid = np.meshgrid(T_range_C + 273.15, alpha_range)
    K_2d = np.vectorize(
        lambda T, a: effective_bulk_modulus(
            bulk_modulus_tait_linear(bulk_modulus_temperature(OIL_K0, OIL_BETA_K, T, OIL_T_REF), OIL_N_TAIT, P_nom),
            P_nom, a
        )
    )(T_grid, A_grid)

    # CSV - Tornado
    header = ["parameter", "K_eff_lo_pct", "K_eff_hi_pct", "f_h_lo_pct", "f_h_hi_pct"]
    rows = [
        [p, f"{lo:.3f}", f"{hi:.3f}", f"{fl:.3f}", f"{fh:.3f}"]
        for p, lo, hi, fl, fh in zip(params, K_sens_lo, K_sens_hi, f_sens_lo, f_sens_hi)
    ]
    _save_csv("exp15_sensitivity_study.csv", header, rows)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 15 – Sensitivitätsstudie K_eff und f_h (±20 % Parametervariation)", fontsize=13)

    y_pos = np.arange(len(params))
    K_spans = [hi - lo for lo, hi in zip(K_sens_lo, K_sens_hi)]
    order   = np.argsort(np.abs(K_spans))[::-1]
    params_sorted  = [params[i] for i in order]
    lo_sorted = [K_sens_lo[i] for i in order]
    hi_sorted = [K_sens_hi[i] for i in order]

    bars_lo = ax1.barh(y_pos, [abs(l) for l in lo_sorted], left=[min(l, 0) for l in lo_sorted],
                       color="tomato", alpha=0.8, label="−20 % Variation")
    bars_hi = ax1.barh(y_pos, [abs(h) for h in hi_sorted], left=[max(h, 0) for h in hi_sorted],
                       color="steelblue", alpha=0.8, label="+20 % Variation")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params_sorted)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("ΔK_eff [% vom Nominalwert]")
    ax1.set_title("Tornado: Einfluss auf K_eff")
    ax1.legend()
    ax1.grid(True, axis="x", alpha=0.3)

    cf = ax2.contourf(T_range_C, alpha_range * 100, K_2d / 1e6, levels=20, cmap="plasma")
    fig.colorbar(cf, ax=ax2, label="K_eff [MPa]")
    ax2.set_xlabel("Temperatur [°C]")
    ax2.set_ylabel("Luftgehalt α [%]")
    ax2.set_title("K_eff(T, α) – Einflussraum (p=100 bar)")

    _save_svg(fig, "exp15_sensitivity_study.svg")
    print("   → exp15_sensitivity_study.csv / .svg")


# ============================================================================
# Hauptprogramm
# ============================================================================

def main() -> None:
    """Führt alle 15 Experimente aus und speichert Ergebnisse nach src/results/."""
    print("=" * 65)
    print("  EXPERIMENTE: Elastizität hydraulischer Öle  (Epp, 2026)")
    print(f"  Ausgabeverzeichnis: {RESULTS_DIR}")
    print("=" * 65)

    experiment_01_bulk_vs_pressure()
    experiment_02_temperature_effects()
    experiment_03_gas_content_impact()
    experiment_04_eigenfrequency_positioning()
    experiment_05_sea_cobot_safety()
    experiment_06_viscoelastic_frequency()
    experiment_07_oil_lifetime()
    experiment_08_aging_failure_rates()
    experiment_09_cavitation_surface()
    experiment_10_mixing_fluid_comparison()
    experiment_11_barus_high_pressure()
    experiment_12_ultrasonic_monitoring()
    experiment_13_recuperation_energy()
    experiment_14_shear_degradation()
    experiment_15_sensitivity_study()

    print("\n" + "=" * 65)
    print("  Alle Experimente abgeschlossen.")
    print(f"  {len(os.listdir(RESULTS_DIR))} Dateien in {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
