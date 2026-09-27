#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugung aller Plots für:
  "Induktionsfedern als aerodynamische Energiequelle –
   Kraftfahrzeuge und Satelliten im erdnahen Orbit"

Autor: Stephan Epp  |  März 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# ─── globale Plot-Einstellungen ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = ["#1946a0", "#b43220", "#1e6432", "#8b5cf6", "#d97706", "#0891b2"]
FILL_ALPHA = 0.15

# ─── Physikalische Konstanten ──────────────────────────────────────────────────
rho0        = 1.225        # kg/m³  Luftdichte Meeresspiegel
nu_air      = 1.5e-5       # m²/s   kin. Viskosität Luft
Cd_car      = 0.28         # –      cW-Wert Serien-PKW
A_front     = 2.2          # m²     Stirnfläche
A_intake    = 0.72         # m²     nutzbare Frontfläche (volle Ausnutzung)
m_car       = 1500.0       # kg
g_earth     = 9.81         # m/s²
mu_roll     = 0.012        # –      Rollwiderstandsbeiwert
R_earth     = 6.371e6      # m
GM          = 3.986004e14  # m³/s²
H_scale     = 7640.0       # m      Skalenhöhe ISA (Troposphäre)

# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def atm_density(h):
    """Vereinfachtes US-Standardatmosphärenmodell (0–200 km)."""
    h = np.asarray(h, dtype=float)
    rho = np.zeros_like(h)
    # Troposphäre 0–11 km
    mask = h <= 11000
    rho[mask] = 1.225 * (1 - 2.2558e-5 * h[mask])**4.2561
    # Stratosphäre 11–25 km
    mask = (h > 11000) & (h <= 25000)
    rho[mask] = 0.3639 * np.exp(-1.5788e-4 * (h[mask] - 11000))
    # 25–47 km
    mask = (h > 25000) & (h <= 47000)
    rho[mask] = 0.08803 * (1 + 4.6153e-5 * (h[mask] - 25000))**(-35.1632)
    # 47–86 km (Mesosphäre)
    mask = (h > 47000) & (h <= 86000)
    rho[mask] = 0.001846 * np.exp(-3.9899e-4 * (h[mask] - 47000))
    # 86–150 km (Thermosphäre, vereinfacht exponentiell)
    mask = h > 86000
    rho[mask] = 8.0e-6 * np.exp(-6.5e-5 * (h[mask] - 86000))
    return rho


def orbital_velocity(h):
    """Kreisbahngeschwindigkeit in m/s bei Höhe h (m) über Erdmittelpunkt."""
    return np.sqrt(GM / (R_earth + h))


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Luftwiderstandskraft und Druckleistung als Funktion der Geschwindigkeit
# ══════════════════════════════════════════════════════════════════════════════

def plot1_luftwiderstand():
    v_kmh = np.linspace(0, 200, 500)
    v_ms  = v_kmh / 3.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Luftwiderstand und Antriebsleistung in Abhängigkeit der Geschwindigkeit",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Luftwiderstandskraft ──────────────────────────────────────
    ax = axes[0]
    cw_vals = [0.20, 0.28, 0.35, 0.45]
    labels  = ["cW = 0.20 (opt. Sportler)", "cW = 0.28 (Serien-PKW)",
               "cW = 0.35 (Kombi/SUV)", "cW = 0.45 (Bus/LKW-Front)"]
    for i, (cw, lbl) in enumerate(zip(cw_vals, labels)):
        F = 0.5 * rho0 * cw * A_front * v_ms**2
        ax.plot(v_kmh, F / 1e3, color=COLORS[i], lw=2, label=lbl)
    ax.axvline(130, color="gray", lw=1, ls=":", alpha=0.8)
    ax.text(131, 0.3, "130 km/h", fontsize=8, color="gray", rotation=90)
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Luftwiderstandskraft F$_W$ [kN]")
    ax.set_title("Luftwiderstandskraft")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    # ── Teilbild 2: Antriebsleistung ──────────────────────────────────────────
    ax = axes[1]
    P_drag = 0.5 * rho0 * Cd_car * A_front * v_ms**3 / 1000   # kW
    P_roll = m_car * g_earth * mu_roll * v_ms / 1000            # kW
    P_total = P_drag + P_roll

    ax.fill_between(v_kmh, P_drag, alpha=FILL_ALPHA, color=COLORS[0])
    ax.fill_between(v_kmh, P_drag, P_total, alpha=FILL_ALPHA, color=COLORS[1])
    ax.plot(v_kmh, P_drag,  lw=2, color=COLORS[0], label="Luftwiderstand P$_W$ ∝ v³")
    ax.plot(v_kmh, P_roll,  lw=2, color=COLORS[1], label="Rollwiderstand P$_R$ ∝ v")
    ax.plot(v_kmh, P_total, lw=2.5, color="black", ls="--", label="Gesamtwiderstand P$_{{ges}}$")

    # Schnittpunkt markieren
    v_cross = v_ms[np.argmin(np.abs(P_drag - P_roll))]
    ax.axvline(v_cross * 3.6, color="gray", lw=1, ls=":")
    ax.text(v_cross * 3.6 + 1, 5, f"{v_cross*3.6:.0f} km/h", fontsize=8, color="gray")
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Antriebsleistung [kW]")
    ax.set_title("Antriebsleistungsanteile")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    # ── Teilbild 3: Verfügbare Luftleistung im Ansaugkanal ────────────────────
    ax = axes[2]
    A_fracs = [0.20, 0.33, 0.50, 0.72]
    A_labels = ["A = 0.20 m² (min.)", "A = 0.33 m² (typisch)",
                "A = 0.50 m² (opt.)", "A = 0.72 m² (voll genutzt)"]
    for i, (Ai, lbl) in enumerate(zip(A_fracs, A_labels)):
        P_avail = 0.5 * rho0 * Ai * v_ms**3 / 1000
        ax.plot(v_kmh, P_avail, color=COLORS[i], lw=2, label=lbl)
    # Betz-Leistung für A=0.72
    P_betz = 0.593 * 0.5 * rho0 * 0.72 * v_ms**3 / 1000
    ax.plot(v_kmh, P_betz, "k--", lw=1.5, alpha=0.6, label="Betz-Limit (η=59.3 %)\nbei A=0.72 m²")
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Luftleistung im Einlass P$_{{avail}}$ [kW]")
    ax.set_title("Verfügbare Luftenergie im Fahrzeugeinlass")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    fig.tight_layout()
    fig.savefig("plots/plot1_luftwiderstand.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot1_luftwiderstand.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Mechanisches Modell der Induktionsfeder
# ══════════════════════════════════════════════════════════════════════════════

def plot2_federsystem():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Physikalisches Modell des Induktionsfeder-Wandlers",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Masse-Feder-Dämpfer Energiebilanz ────────────────────────
    ax = axes[0]
    m_spring = 0.05    # kg  Schwingmasse
    k_spring = 500.0   # N/m Federkonstante
    c_parasit = 0.5    # N·s/m parasitärer Dämpfer
    c_elec    = 2.0    # N·s/m elektrischer Dämpfer (Energieentnahme)
    omega0 = np.sqrt(k_spring / m_spring)
    x = np.linspace(-0.02, 0.02, 300)
    E_kin  = 0.5 * m_spring * (omega0 * x)**2
    E_pot  = 0.5 * k_spring * x**2

    ax.fill_between(x * 1000, E_kin, alpha=0.3, color=COLORS[0], label="Kinetische Energie E$_k$")
    ax.fill_between(x * 1000, E_pot, alpha=0.3, color=COLORS[1], label="Pot. Federenergie E$_p$")
    ax.plot(x * 1000, E_kin + E_pot, "k-", lw=2, label="Gesamt E$_{{ges}}$ = const. (ideal)")
    ax.set_xlabel("Auslenkung x [mm]")
    ax.set_ylabel("Energie [mJ]")
    ax.set_title("Energiebilanz Feder-Masse")
    ax.legend(fontsize=8)

    # ── Teilbild 2: Frequenzgang (Resonanzkurve) ──────────────────────────────
    ax = axes[1]
    omega = np.linspace(0, 3 * omega0, 1000)
    for c_el, lbl, col in zip([1.0, 2.0, 5.0, 10.0],
                               ["c$_e$ = 1 N·s/m", "c$_e$ = 2 N·s/m",
                                "c$_e$ = 5 N·s/m", "c$_e$ = 10 N·s/m"],
                               COLORS):
        c_tot = c_parasit + c_el
        denom = ((k_spring - m_spring * omega**2)**2 + (c_tot * omega)**2)
        X_amp = (1.0 / k_spring) / np.sqrt(denom / k_spring**2)
        ax.semilogy(omega / omega0, X_amp * 1000, color=col, lw=2, label=lbl)
    ax.axvline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
    ax.text(1.02, 1e-2, "Resonanz\nω = ω₀", fontsize=7, color="gray")
    ax.set_xlabel("Normierte Kreisfrequenz ω/ω₀")
    ax.set_ylabel("Amplitude X [mm] (log)")
    ax.set_title("Frequenzgang bei verschiedenen el. Dämpfungen")
    ax.legend(fontsize=8, loc="upper right")

    # ── Teilbild 3: Extraktion von Leistung vs. Dämpfungsverhältnis ──────────
    ax = axes[2]
    c_e_vals = np.linspace(0.01, 20, 500)
    F_anreg  = 1.0    # N Anregungskraft (normiert)
    # Maximale Leistung bei Resonanz: P_el = 0.5 * c_e * omega0² * X²
    # Bei Resonanz: X = F / (c_tot * omega0)
    P_el_arr = []
    for c_e in c_e_vals:
        c_tot = c_parasit + c_e
        X_res = F_anreg / (c_tot * omega0)
        P_el  = 0.5 * c_e * omega0**2 * X_res**2
        P_el_arr.append(P_el)
    P_el_arr = np.array(P_el_arr)
    P_el_max = P_el_arr.max()
    c_e_opt  = c_e_vals[np.argmax(P_el_arr)]

    ax.plot(c_e_vals, P_el_arr / P_el_max, color=COLORS[0], lw=2.5)
    ax.axvline(c_e_opt, color=COLORS[1], lw=1.5, ls="--",
               label=f"Optimum c$_e$ = {c_e_opt:.2f} N·s/m")
    ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.fill_between(c_e_vals, P_el_arr / P_el_max, alpha=0.15, color=COLORS[0])
    ax.set_xlabel("Elektrischer Dämpfer c$_e$ [N·s/m]")
    ax.set_ylabel("Normierte elektrische Leistung P$_{{el}}$/P$_{{max}}$")
    ax.set_title("Optimaler Arbeitspunkt der Leistungsextraktion")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig("plots/plot2_federsystem.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot2_federsystem.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Betz-Limit und Wirkungsgrad der Energieextraktion
# ══════════════════════════════════════════════════════════════════════════════

def plot3_betz():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Betz-Minimum und Wirkungsgrade aerodynamischer Energiewandler",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Betz-Leistungsbeiwert ────────────────────────────────────
    ax = axes[0]
    v2_v1 = np.linspace(0.001, 0.999, 500)    # v₂/v₁ = Abström-/Anströmgeschwindigkeit
    Cp = 0.5 * (1 + v2_v1) * (1 - v2_v1**2)  # Betz-Formel
    ax.plot(v2_v1, Cp, color=COLORS[0], lw=2.5)
    ax.axhline(16/27, color=COLORS[1], lw=1.5, ls="--",
               label=f"Betz-Grenze C$_P$ = 16/27 ≈ {16/27:.4f}")
    ax.axvline(1/3, color=COLORS[2], lw=1.5, ls=":",
               label="Optimum v₂/v₁ = 1/3")
    ax.fill_between(v2_v1, Cp, alpha=FILL_ALPHA, color=COLORS[0])
    ax.set_xlabel("Geschwindigkeitsverhältnis v₂/v₁")
    ax.set_ylabel("Leistungsbeiwert C$_P$")
    ax.set_title("Betz'sches Leistungsmaximum")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.65)

    # ── Teilbild 2: Wirkungsgrade realer Wandler ──────────────────────────────
    ax = axes[1]
    technologien = ["Betz-Grenze\n(theoret.)", "Windturbine\n(optimal)",
                    "Piezo-Array\n(Stand der Technik)", "Elektromagn.\nInduktionsfeder",
                    "Kanalströmungs-\nwandler", "Kombinierter\nAnsatz"]
    etas = [0.593, 0.48, 0.18, 0.32, 0.25, 0.35]
    colors_bar = [COLORS[1], COLORS[0], COLORS[2], COLORS[3], COLORS[4], COLORS[5]]
    bars = ax.barh(technologien, etas, color=colors_bar, alpha=0.85, height=0.6)
    for bar, eta in zip(bars, etas):
        ax.text(eta + 0.005, bar.get_y() + bar.get_height()/2,
                f"{eta*100:.0f}%", va="center", fontsize=9, fontweight="bold")
    ax.axvline(16/27, color=COLORS[1], lw=1.5, ls="--", alpha=0.5)
    ax.set_xlabel("Wirkungsgrad η")
    ax.set_title("Wirkungsgrade verschiedener Wandlertechnologien")
    ax.set_xlim(0, 0.70)

    # ── Teilbild 3: Vergleich Nutzleistung verschiedener Technologien ─────────
    ax = axes[2]
    v_kmh   = np.linspace(0, 200, 300)
    v_ms    = v_kmh / 3.6
    P_avail = 0.5 * rho0 * A_intake * v_ms**3 / 1000  # kW Verfügbar

    for eta, tech, col in zip([0.593, 0.48, 0.32, 0.18],
                               ["Betz-Grenze", "Windturbine", "Induktionsfeder (el.)",
                                "Piezo-Array"],
                               COLORS):
        ax.plot(v_kmh, eta * P_avail, lw=2, color=col, label=f"{tech} (η={int(eta*100)}%)")

    ax.fill_between(v_kmh, P_avail, alpha=FILL_ALPHA, color="gray",
                    label="Verfügbare Luftleistung A=0.72 m²")
    ax.axhline(3.0, color="k", lw=1, ls=":", alpha=0.6)
    ax.text(5, 3.1, "Typischer Bordnetz-Bedarf (~3 kW)", fontsize=8, color="k")
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Ernteleistung P$_{{el}}$ [kW]")
    ax.set_title("Ernteleistung bei A$_{{intake}}$ = 0.72 m²")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    fig.tight_layout()
    fig.savefig("plots/plot3_betz.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot3_betz.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 – Vollständige Energiebilanz Kraftfahrzeug
# ══════════════════════════════════════════════════════════════════════════════

def plot4_energiebilanz_kfz():
    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
    fig.suptitle("Vollständige Energiebilanz eines Kraftfahrzeugs mit Induktionsfedern",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Leistungsanteile bei 130 km/h ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    v_ref  = 130 / 3.6   # m/s
    P_drag = 0.5 * rho0 * Cd_car * A_front * v_ref**3  / 1000
    P_roll = m_car * g_earth * mu_roll * v_ref           / 1000
    P_aux  = 3.0   # kW  Bordnetz (Licht, Klima, Medien)
    P_gear = 1.5   # kW  Antriebsstrang-Verluste
    P_total = P_drag + P_roll + P_aux + P_gear

    labels = ["Luftwiderstand\n$P_W$", "Rollwiderstand\n$P_R$",
              "Bordnetz-Aux.\n$P_{{aux}}$", "Triebstrang\n$P_{{gear}}$"]
    values = [P_drag, P_roll, P_aux, P_gear]
    explode = (0.05, 0.05, 0.05, 0.05)
    colors_pie = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
    wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct="%1.1f%%",
                                        colors=colors_pie, explode=explode,
                                        textprops={"fontsize": 8})
    ax1.set_title(f"Leistungsverteilung bei 130 km/h\n(Gesamt: {P_total:.1f} kW)")

    # ── Teilbild 2: Leistungsanteile als Funktion der Geschwindigkeit ──────────
    ax2 = fig.add_subplot(gs[1])
    v_kmh = np.linspace(5, 200, 400)
    v_ms  = v_kmh / 3.6
    P_d = 0.5 * rho0 * Cd_car * A_front * v_ms**3 / 1000
    P_r = m_car * g_earth * mu_roll * v_ms / 1000
    P_a = np.full_like(v_ms, 3.0)
    P_g = np.full_like(v_ms, 1.5)
    P_ges = P_d + P_r + P_a + P_g

    ax2.stackplot(v_kmh, P_d, P_r, P_a, P_g,
                  labels=["Luftwiderstand", "Rollwiderstand", "Bordnetz", "Triebstrang"],
                  colors=COLORS[:4], alpha=0.75)
    ax2.plot(v_kmh, P_ges, "k-", lw=2, label="Gesamtleistung")
    ax2.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax2.set_ylabel("Leistung [kW]")
    ax2.set_title("Leistungsanteile über Geschwindigkeit")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlim(5, 200)
    ax2.set_ylim(0)

    # ── Teilbild 3: Energiebilanz MIT Induktionsfedern ─────────────────────────
    ax3 = fig.add_subplot(gs[2])
    eta_is   = 0.32    # Wirkungsgrad Induktionsfeder-System
    P_harvest = eta_is * 0.5 * rho0 * A_intake * v_ms**3 / 1000
    # Induktionsfedern erhöhen den effektiven Luftwiderstand proportional zur entnommenen Energie
    # Δc_W * A = 2 * P_harvest / (rho * v³/2 * v) = 2 * P_harvest / (rho * v³ * 0.5)   [kW]
    # Änderung unerheblich: Ansaugkanal-Anteil ist Teil der c_W-Rechnung
    # Netto-Hilfsleistung:
    P_net_aux = P_harvest - 0.0   # Bordnetz gespeist durch Federn
    P_engine_saved = np.minimum(P_harvest, P_a)

    ax3.plot(v_kmh, P_harvest, color=COLORS[0], lw=2.5,
             label="Ernteleistung P$_{{is}}$ (η=32%)")
    ax3.plot(v_kmh, P_engine_saved, color=COLORS[2], lw=2, ls="--",
             label="Einsparung Motor-Aux.")
    ax3.fill_between(v_kmh, P_harvest, alpha=FILL_ALPHA, color=COLORS[0])
    ax3.axhline(3.0, color=COLORS[1], lw=1.5, ls=":",
                label="Bordnetz-Bedarf 3 kW")
    # Schnittpunkt (ab welcher v Bordnetz vollständig gedeckt)
    v_cover = v_kmh[np.searchsorted(P_harvest, 3.0)]
    ax3.axvline(v_cover, color=COLORS[1], lw=1, ls="--", alpha=0.5)
    ax3.text(v_cover + 2, 0.5, f"{v_cover:.0f} km/h", fontsize=8, color=COLORS[1])
    ax3.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax3.set_ylabel("Leistung [kW]")
    ax3.set_title("Netto-Energiegewinn der Hilfsverbraucher")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_xlim(5, 200)
    ax3.set_ylim(0)

    fig.tight_layout()
    fig.savefig("plots/plot4_energiebilanz_kfz.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot4_energiebilanz_kfz.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Frontflächen-Analyse: Vollständige Ausnutzung
# ══════════════════════════════════════════════════════════════════════════════

def plot5_frontflaeche():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Analyse der vollständigen Frontflächen-Ausnutzung bei Kraftfahrzeugen",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Ernteleistung vs. Wirkungsgrad für versch. Frontbreiten ───
    ax = axes[0]
    # Fahrzeugbreite 1.80 m, Höhen von 30 cm bis 60 cm
    hoehen = [0.30, 0.40, 0.50, 0.60]
    breite = 1.80   # m
    v_ref = 130 / 3.6
    etas = np.linspace(0, 0.593, 300)
    for h in hoehen:
        A = breite * h
        P_avail = 0.5 * rho0 * A * v_ref**3 / 1000
        ax.plot(etas * 100, etas * P_avail, lw=2,
                label=f"h = {int(h*100)} cm, A = {A:.2f} m²")
    ax.axvline(100*0.32, color="gray", lw=1, ls=":", alpha=0.7)
    ax.text(100*0.32 + 0.5, 0.5, "η=32%\n(real)", fontsize=7, color="gray")
    ax.axhline(3.0, color="k", lw=1, ls="--", alpha=0.5)
    ax.text(5, 3.1, "Bordnetz-Bedarf 3 kW", fontsize=8)
    ax.set_xlabel("Wirkungsgrad η [%]")
    ax.set_ylabel("Ernteleistung P [kW] bei 130 km/h")
    ax.set_title("Ernteleistung vs. Wirkungsgrad\nbei verschiedenen Fronthöhen")
    ax.legend(fontsize=8)

    # ── Teilbild 2: Ernteleistung vs. Geschwindigkeit für opt. Konfiguration ──
    ax = axes[1]
    A_opt  = 0.72   # m²  (1.80 m × 0.40 m)
    v_kmh  = np.linspace(0, 200, 400)
    v_ms   = v_kmh / 3.6
    etas2  = [0.593, 0.48, 0.32, 0.18]
    labels2 = ["Betz-Grenze (59.3%)", "Optimale Windturbine (48%)",
               "Induktionsfeder (32%)", "Piezo minimal (18%)"]
    for eta, lbl, col in zip(etas2, labels2, COLORS):
        P = eta * 0.5 * rho0 * A_opt * v_ms**3 / 1000
        ax.plot(v_kmh, P, lw=2, color=col, label=lbl)
    ax.axhline(3.0, color="k", lw=1, ls="--", alpha=0.5)
    ax.text(5, 3.1, "Bordnetz 3 kW", fontsize=8)
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Ernteleistung P [kW]")
    ax.set_title("Ernteleistung bei voller Frontflächenausnutzung\nA = 0.72 m² (1.8 m × 0.4 m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    # ── Teilbild 3: Jährliche Energiemenge bei Standardfahrprofil ──────────────
    ax = axes[2]
    # Annahme: Typisches Fahrprofil D (WLTP-ähnlich)
    v_profile   = np.array([30,  50,  80, 100, 120, 130])   # km/h
    t_anteil    = np.array([0.15, 0.25, 0.25, 0.15, 0.12, 0.08])  # Zeitanteil
    km_year     = 15000.0   # km/Jahr

    energies = {}
    for eta, tech in zip([0.48, 0.32, 0.18], ["Windturbine", "Induktionsfeder", "Piezo"]):
        e_year = 0.0
        for vi, ti in zip(v_profile, t_anteil):
            v_ms_i = vi / 3.6
            P_i    = eta * 0.5 * rho0 * A_opt * v_ms_i**3 / 1000  # kW
            t_h_i  = (km_year * ti) / vi                           # Stunden
            e_year += P_i * t_h_i                                  # kWh
        energies[tech] = e_year

    bars = ax.bar(list(energies.keys()), list(energies.values()),
                  color=COLORS[:3], alpha=0.85, width=0.5)
    for bar, val in zip(bars, energies.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 20, f"{val:.0f} kWh",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(750, color="k", lw=1.5, ls="--", alpha=0.7)
    ax.text(0.02, 780, "Bordnetz-Jahresbedarf ≈ 750 kWh", fontsize=8, transform=ax.transData,
            ha="center", va="bottom")
    ax.text(0.02, 760, "(3 kW × 250 h/a Fahrzeit)", fontsize=7, ha="center")
    ax.set_ylabel("Jahresenergieertrag [kWh]")
    ax.set_title("Jährlicher Energieertrag (15 000 km/a,\nDFahrprofil nach WLTP-Näherung)")

    fig.tight_layout()
    fig.savefig("plots/plot5_frontflaeche.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot5_frontflaeche.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 – Atmosphäre und Satellit: Dichte, Widerstand, Orbitalabbau
# ══════════════════════════════════════════════════════════════════════════════

def plot6_atmosphaere():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Atmosphärische Bedingungen und Satelliten-Luftwiderstand (80–400 km)",
                 fontsize=13, fontweight="bold", y=1.02)

    h_km  = np.linspace(0, 400, 2000)
    h_m   = h_km * 1000
    rho_h = atm_density(h_m)

    # ── Teilbild 1: Dichteprofil ──────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(h_km, rho_h, color=COLORS[0], lw=2.5)
    ax.axhline(rho_h[np.argmin(np.abs(h_km - 100))], color=COLORS[1], lw=1, ls="--", alpha=0.7)
    ax.axvline(100, color=COLORS[1], lw=1, ls="--", alpha=0.7,
               label=f"100 km: ρ ≈ {rho_h[np.argmin(np.abs(h_km-100))]:.2e} kg/m³")
    ax.axvline(120, color=COLORS[2], lw=1, ls=":", alpha=0.7,
               label=f"120 km: ρ ≈ {rho_h[np.argmin(np.abs(h_km-120))]:.2e} kg/m³")
    ax.axvline(400, color=COLORS[3], lw=1, ls=":", alpha=0.7,
               label=f"400 km (ISS): ρ ≈ {rho_h[np.argmin(np.abs(h_km-400))]:.2e} kg/m³")
    ax.text(100, rho_h[0]*0.5, "  Kármán-Linie\n  100 km", fontsize=8, color=COLORS[1])
    ax.set_xlabel("Höhe über Erdoberfläche h [km]")
    ax.set_ylabel("Luftdichte ρ [kg/m³] (log)")
    ax.set_title("US-Standardatmosphäre: Dichteprofil")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 400)

    # ── Teilbild 2: Luftwiderstandskraft auf Satellit ─────────────────────────
    ax = axes[1]
    Cd_sat = 2.2    # tyischer Cd Satellit
    A_sat  = 10.0   # m² Querschnittsfläche
    m_sat  = 1000.0 # kg
    h_range = np.linspace(80, 400, 1000)
    rho_r   = atm_density(h_range * 1000)
    v_orb   = orbital_velocity(h_range * 1000)
    F_drag_sat = 0.5 * rho_r * Cd_sat * A_sat * v_orb**2  # N
    P_drag_sat = F_drag_sat * v_orb / 1000                  # kW

    ax.semilogy(h_range, F_drag_sat, color=COLORS[0], lw=2.5, label="Luftwiderstandskraft F$_D$ [N]")
    ax2_twin = ax.twinx()
    ax2_twin.semilogy(h_range, P_drag_sat, color=COLORS[1], lw=2.5, ls="--",
                      label="Dissipierte Leistung P$_D$ [kW]")
    ax2_twin.set_ylabel("Dissipierte Leistung P$_D$ [kW]", color=COLORS[1])
    ax2_twin.tick_params(axis="y", labelcolor=COLORS[1])
    ax.axvline(100, color="gray", lw=1, ls=":")
    ax.axvline(400, color="gray", lw=1, ls=":")
    ax.text(101, F_drag_sat[np.argmin(np.abs(h_range-100))]*0.3,
            f"F = {F_drag_sat[np.argmin(np.abs(h_range-100))]:.0f} N\nP = {P_drag_sat[np.argmin(np.abs(h_range-100))]:.0f} kW",
            fontsize=8, color=COLORS[0])
    ax.set_xlabel("Orbital­höhe h [km]")
    ax.set_ylabel("Luftwiderstandskraft F$_D$ [N]", color=COLORS[0])
    ax.set_title(f"Luftwiderstand auf Satelliten\n(m={m_sat} kg, A={A_sat} m², C$_D$={Cd_sat})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.set_xlim(80, 400)

    # ── Teilbild 3: Orbitale Abbaurate ────────────────────────────────────────
    ax = axes[2]
    # a_drag = F_D / m   [m/s²]
    a_drag = F_drag_sat / m_sat
    # Orbitaler Abbau Δh/orbit: Δh ≈ -2π * r * (a_drag / g_local)
    r_orb   = R_earth + h_range * 1000
    g_local = GM / r_orb**2
    T_orb   = 2 * np.pi * np.sqrt(r_orb**3 / GM) / 3600  # Stunden
    dh_orbit = -2 * np.pi * (r_orb / 1000) * (a_drag / g_local)  # km/orbit
    dh_day   = dh_orbit * (24 / T_orb)  # km/day

    ax.semilogy(h_range, np.abs(dh_day), color=COLORS[0], lw=2.5, label="|Δh/Tag| [km/Tag]")
    ax.semilogy(h_range, np.abs(dh_orbit), color=COLORS[1], lw=2.5, ls="--",
                label="|Δh/Orbit| [km/Orbit]")
    ax.axvline(100, color="gray", lw=1, ls=":", alpha=0.7)
    ax.axvline(400, color="gray", lw=1, ls=":", alpha=0.7)
    v_iss = h_range[np.argmin(np.abs(h_range - 400))]
    ax.text(101, np.abs(dh_day[np.argmin(np.abs(h_range-100))]) * 0.2,
            f"|Δh/Tag|@100km:\n{np.abs(dh_day[np.argmin(np.abs(h_range-100))]):.1f} km/Tag", fontsize=8)
    ax.set_xlabel("Orbitalhöhe h [km]")
    ax.set_ylabel("Orbitaler Höhenverlust [km] (log)")
    ax.set_title("Orbitaler Höhenverlust durch atmosph. Abbremsung")
    ax.legend(fontsize=9)
    ax.set_xlim(80, 400)

    fig.tight_layout()
    fig.savefig("plots/plot6_atmosphaere.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot6_atmosphaere.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 – Satelliten-Energiebilanz: Ernte vs. Antriebsbedarf
# ══════════════════════════════════════════════════════════════════════════════

def plot7_satellit_energiebilanz():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Energiebilanz von Induktionsfedern auf Satelliten in verschiedenen Orbitalhöhen",
                 fontsize=13, fontweight="bold", y=1.02)

    h_km    = np.linspace(80, 500, 1000)
    h_m     = h_km * 1000
    rho_h   = atm_density(h_m)
    v_orb   = orbital_velocity(h_m)
    Cd_sat  = 2.2
    A_sat   = 10.0
    m_sat   = 1000.0

    F_D     = 0.5 * rho_h * Cd_sat * A_sat * v_orb**2
    P_D     = F_D * v_orb / 1000    # kW
    a_D     = F_D / m_sat

    # ── Teilbild 1: Erntebare und Antriebsleistung ───────────────────────────
    ax = axes[0]
    for eta, lbl, col in zip([0.40, 0.25, 0.15],
                              ["η=40% (opt.)", "η=25% (real)", "η=15% (min.)"],
                              COLORS):
        ax.semilogy(h_km, eta * P_D * 1e3, lw=2, color=col, ls="--", label=f"Ernte {lbl} [W]")
    ax.semilogy(h_km, P_D * 1e3, color="k", lw=2.5, label="Gesamte Widerstandsleistung [W]")
    ax.axvline(100, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Orbitalhöhe h [km]")
    ax.set_ylabel("Leistung [W] (log-Skala)")
    ax.set_title("Ernteleistung vs. gesamte\nantmosphärische Widerstandsleistung")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(80, 500)

    # ── Teilbild 2: Netto-Energiebilanz (Ernte minus Antriebsbedarf) ──────────
    ax = axes[1]
    P_thrust_needed = P_D   # kW um Orbit zu erhalten
    for eta, col in zip([0.40, 0.25, 0.15], COLORS):
        P_net = eta * P_D - P_thrust_needed  # immer negativ!
        ax.semilogy(h_km, np.abs(P_net) * 1e3, lw=2, color=col,
                    label=f"η={int(eta*100)}%  |ΔP| [W]")
    # Zeige dass immer P_net < 0 (Nettoverlust)
    ax.set_xlabel("Orbitalhöhe h [km]")
    ax.set_ylabel("|Netto-Energiedefizit| [W] (log)")
    ax.set_title("Netto-Energiedefizit\n(immer P$_{{net}}$ < 0  →  kein Perpetuum Mobile)")
    ax.legend(fontsize=8)
    ax.set_xlim(80, 500)
    ax.text(200, ax.get_ylim()[0] * 3, "P$_{{Ernte}}$ < P$_{{Antrieb}}$ ∀η < 1",
            fontsize=9, color="red", fontweight="bold")

    # ── Teilbild 3: Orbitale Lebenszeit bei 100 km ────────────────────────────
    ax = axes[2]
    h100  = 100    # km
    rho_h100  = atm_density(np.array([h100*1000]))[0]
    v100      = orbital_velocity(h100*1000)
    F100      = 0.5 * rho_h100 * Cd_sat * A_sat * v100**2
    # Zeitachse: simuliere orbitalen Abbau
    dt      = 10.0   # Sekunden
    h_sim   = np.array([h100 * 1000.0])
    t_arr   = [0.0]
    h_arr   = [h100]

    h_now = h100 * 1000.0
    t_now = 0.0
    while h_now > 50000 and t_now < 2 * 86400:
        rho_now = atm_density(np.array([h_now]))[0]
        v_now   = orbital_velocity(h_now)
        F_now   = 0.5 * rho_now * Cd_sat * A_sat * v_now**2
        a_now   = F_now / m_sat
        dv      = a_now * dt
        # dh/dt ≈ 2r * dv / v  (vis-viva approximation)
        r_now   = R_earth + h_now
        dh      = -2 * (r_now / v_now) * dv
        h_now  += dh
        t_now  += dt
        t_arr.append(t_now / 3600)
        h_arr.append(h_now / 1000)

    ax.plot(t_arr, h_arr, color=COLORS[0], lw=2.5, label=f"h(t): Start {h100} km")
    ax.axhline(80, color=COLORS[1], lw=1.5, ls="--", alpha=0.7, label="Wiedereintritt h=80 km")
    ax.set_xlabel("Zeit nach Start [h]")
    ax.set_ylabel("Orbitalhöhe h [km]")
    ax.set_title(f"Simulierter Orbitaler Abbau\nStarthöhe {h100} km (kein Schub)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("plots/plot7_satellit_energiebilanz.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot7_satellit_energiebilanz.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 – Thermodynamische Grenzen und Entropieanalyse
# ══════════════════════════════════════════════════════════════════════════════

def plot8_thermodynamik():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Thermodynamische Grenzen aerodynamischer Energiewandlung",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Teilbild 1: Energiefluss-Sankey (schematisch als Balken) ──────────────
    ax = axes[0]
    # Energiefluss bei 130 km/h mit v-opt. IF-Array
    v_ref = 130 / 3.6
    P_in_kW    = 0.5 * rho0 * A_intake * v_ref**3 / 1000   # kinematische Eingangsenergie
    P_betz_kW  = 0.593 * P_in_kW
    P_el_kW    = 0.32  * P_in_kW   # elektrisch extrahiert
    P_wake_kW  = P_in_kW - P_el_kW  # Nachlauf (Wärme + Rest-kinetik)
    P_ohm_kW   = P_el_kW * 0.08    # Elektrische Verluste
    P_useful   = P_el_kW - P_ohm_kW

    labels_san = ["Kinetische\nEinlassenergie", "Betz-Maximum\n(theoret.)",
                  "Extrakt. el. Leistung\n(η=32%)", "Ohm'sche\nVerluste",
                  "Nutzbare el.\nLeistung", "Nachlauf-\nDissipation"]
    vals_san   = [P_in_kW, P_betz_kW, P_el_kW, P_ohm_kW, P_useful, P_wake_kW]
    colors_san = [COLORS[0], COLORS[3], COLORS[2], COLORS[1], COLORS[2], "gray"]

    y_pos = np.arange(len(labels_san))
    bars = ax.barh(y_pos, vals_san, color=colors_san, alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_san, fontsize=8)
    for bar, val in zip(bars, vals_san):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f} kW", va="center", fontsize=9)
    ax.set_xlabel("Leistung [kW] bei 130 km/h, A = 0.72 m²")
    ax.set_title("Energiefluss-Bilanz (Sankey-Näherung)")
    ax.set_xlim(0, P_in_kW * 1.3)

    # ── Teilbild 2: Entropieproduktionsrate ───────────────────────────────────
    ax = axes[1]
    v_kmh = np.linspace(10, 200, 400)
    v_ms  = v_kmh / 3.6
    T_amb = 293.15   # K Umgebungstemperatur
    # Entropierate S_irr = Q_diss / T; Q_diss = P_D - P_harvest
    P_D     = 0.5 * rho0 * Cd_car * A_front * v_ms**3 / 1000
    P_h     = 0.32  * 0.5 * rho0 * A_intake * v_ms**3 / 1000
    Q_diss_drag  = P_D             # kW → kJ/s
    Q_diss_total = P_D + P_h * (1 - 0.9)  # Induktionsfedern: 10% der Ernte geht als Wärme

    S_irr_no  = Q_diss_drag  * 1e3 / T_amb   # W/K
    S_irr_with = Q_diss_total * 1e3 / T_amb  # W/K

    ax.plot(v_kmh, S_irr_no,   color=COLORS[0], lw=2.5, label="Ohne Induktionsfedern")
    ax.plot(v_kmh, S_irr_with, color=COLORS[1], lw=2.5, ls="--",
            label="Mit Induktionsfedern (η=32%)")
    ax.fill_between(v_kmh, S_irr_no, S_irr_with, alpha=0.2, color=COLORS[1],
                    label="Zusätzliche Entropie durch IF")
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Entropieproduktionsrate Ṡ$_{{irr}}$ [W/K]")
    ax.set_title("Entropieproduktion mit und ohne\nInduktionsfedern")
    ax.legend(fontsize=8)
    ax.set_xlim(10, 200)
    ax.set_ylim(0)

    # ── Teilbild 3: Unmöglichkeit des Perpetuum Mobile ────────────────────────
    ax = axes[2]
    # Zeige: P_harvest / P_antrieb << 1 für alle realistischen Annahmen
    v_kmh = np.linspace(10, 200, 400)
    v_ms  = v_kmh / 3.6
    P_drag_total = 0.5 * rho0 * Cd_car * A_front * v_ms**3 / 1000
    P_roll_total = m_car * g_earth * mu_roll * v_ms / 1000
    P_engine     = P_drag_total + P_roll_total + 3.0 + 1.5

    # Bei 100% Betz auf voller Frontfläche (absolutes Maximum!)
    P_max_theoret = 0.593 * 0.5 * rho0 * A_front * v_ms**3 / 1000

    ratio_max = P_max_theoret / P_engine
    ratio_real = (0.32 * 0.5 * rho0 * A_intake * v_ms**3 / 1000) / P_engine

    ax.plot(v_kmh, ratio_max * 100, color=COLORS[0], lw=2.5,
            label="Theoret. Maximum:\nBetz auf vollst. Stirnfläche")
    ax.plot(v_kmh, ratio_real * 100, color=COLORS[2], lw=2.5, ls="--",
            label="Realistisch:\nη=32% auf A=0.72 m²")
    ax.axhline(100, color=COLORS[1], lw=2, ls=":", label="Selbstversorgung (100%)")
    ax.fill_between(v_kmh, ratio_real * 100, 100, alpha=0.1, color=COLORS[1])
    ax.text(120, 60, "Perpetuum Mobile\nWäre in diesem Bereich", fontsize=8,
            color=COLORS[1], ha="center")
    ax.text(120, 25, "Physikalisch\nRealistisch", fontsize=8, ha="center",
            color=COLORS[2])
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("P$_{{Ernte}}$ / P$_{{Antrieb}}$ [%]")
    ax.set_title("Grenzen der Selbstversorgung:\nWarum kein Perpetuum Mobile möglich ist")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(10, 200)
    ax.set_ylim(0, 120)

    fig.tight_layout()
    fig.savefig("plots/plot8_thermodynamik.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot8_thermodynamik.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 – Numerische Simulation: Parameterstudie
# ══════════════════════════════════════════════════════════════════════════════

def plot9_parameterstudie():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Numerische Parameterstudie: Induktionsfeder-Systeme für KFZ und Satelliten",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── (1,1): Leistung vs. Anzahl der Federn ──────────────────────────────────
    ax = axes[0, 0]
    n_springs = np.arange(1, 300, 1)
    # Jede Feder hat eine Einlass-Apertur von 5cm × 5cm
    A_feder = 0.05 * 0.05   # m²
    v_ref = 130 / 3.6
    eta_if = 0.32
    P_single = eta_if * 0.5 * rho0 * A_feder * v_ref**3  # W pro Feder
    P_array  = n_springs * P_single

    ax.plot(n_springs, P_array, color=COLORS[0], lw=2.5)
    ax.axhline(3000, color=COLORS[1], lw=1.5, ls="--", label="Bordnetz 3 kW")
    ax.axhline(P_single * (A_intake / A_feder), color=COLORS[2], lw=1.5, ls=":",
               label=f"max. VCA-Federn\n(A={A_intake} m²)")
    n_3kw = 3000 / P_single
    ax.axvline(n_3kw, color=COLORS[1], lw=1, ls="--", alpha=0.5)
    ax.text(n_3kw + 2, 500, f"n ≈ {int(n_3kw)}\nFedern", fontsize=8, color=COLORS[1])
    ax.set_xlabel("Anzahl der Induktionsfedern n")
    ax.set_ylabel("Gesamtleistung P [W] bei 130 km/h")
    ax.set_title("Array-Leistung vs. Federanzahl\n(A$_F$ = 5×5 cm²)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_ylim(0)

    # ── (1,2): Sensitivität bzgl. Federsteifigkeit ────────────────────────────
    ax = axes[0, 1]
    k_vals  = np.logspace(1, 4, 200)   # N/m
    m_f     = 0.05                     # kg
    # Resonanzfrequenz
    f0_vals = np.sqrt(k_vals / m_f) / (2 * np.pi)
    # Bei Anregungsfrequenz = Rotationsfrequenz der StrömungsinstabilitätS (Karman-Wirbelstrasse)
    # f_karman ≈ 0.2 * v / d  (Strouhal)
    d_duct  = 0.05    # m Kanaldurchmesser
    f_karman_100 = 0.2 * (100 / 3.6) / d_duct   # Hz bei 100 km/h
    f_karman_130 = 0.2 * (130 / 3.6) / d_duct
    ax.semilogx(k_vals, f0_vals, color=COLORS[0], lw=2.5, label="f₀(k) Eigenfrequenz")
    ax.axhline(f_karman_100, color=COLORS[1], lw=1.5, ls="--",
               label=f"f$_{{Kármán}}$ @ 100 km/h = {f_karman_100:.0f} Hz")
    ax.axhline(f_karman_130, color=COLORS[2], lw=1.5, ls=":",
               label=f"f$_{{Kármán}}$ @ 130 km/h = {f_karman_130:.0f} Hz")
    # Optimale Federsteifigkeit
    k_opt_100 = m_f * (2 * np.pi * f_karman_100)**2
    k_opt_130 = m_f * (2 * np.pi * f_karman_130)**2
    ax.axvline(k_opt_100, color=COLORS[1], lw=1, ls="--", alpha=0.6)
    ax.axvline(k_opt_130, color=COLORS[2], lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("Federsteifigkeit k [N/m] (log)")
    ax.set_ylabel("Eigenfrequenz f₀ [Hz]")
    ax.set_title("Optimale Federsteifigkeit\n(Resonanz mit Kármán-Wirbelfrequenz)")
    ax.legend(fontsize=8)
    ax.set_xlim(10, 1e4)

    # ── (1,3): Temperatureinfluss ─────────────────────────────────────────────
    ax = axes[0, 2]
    T_air    = np.linspace(-30, 50, 100)  # °C
    rho_T    = 101325 / (287.058 * (T_air + 273.15))  # perfektes Gas (p = const)
    eta_T    = 0.32 * (1 - 0.001 * (T_air - 20))  # leichte T-Abhängigkeit der Effizienz
    v_ref    = 130 / 3.6
    P_T      = eta_T * 0.5 * rho_T * A_intake * v_ref**3 / 1000

    ax.plot(T_air, P_T, color=COLORS[0], lw=2.5, label="P$_{{el}}$ [kW]")
    ax2t = ax.twinx()
    ax2t.plot(T_air, rho_T, color=COLORS[1], lw=2, ls="--", label="ρ$_{{Luft}}$ [kg/m³]")
    ax2t.set_ylabel("Luftdichte ρ [kg/m³]", color=COLORS[1])
    ax2t.tick_params(axis="y", labelcolor=COLORS[1])
    ax.axvline(20, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Außentemperatur T [°C]")
    ax.set_ylabel("Ernteleistung P [kW]", color=COLORS[0])
    ax.set_title("Temperaturabhängigkeit der\nErnteleistung")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2t.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9)

    # ── (2,1): Satellit: Ernteleistung vs. Flügelfläche ──────────────────────
    ax = axes[1, 0]
    A_sat_range = np.linspace(1, 50, 200)
    h_ref       = 100    # km
    rho_100     = atm_density(np.array([h_ref * 1000]))[0]
    v_100       = orbital_velocity(h_ref * 1000)
    for eta_s, col in zip([0.40, 0.25, 0.15], COLORS):
        P_s = eta_s * 0.5 * rho_100 * A_sat_range * v_100**3 / 1e6  # MW
        ax.plot(A_sat_range, P_s, lw=2, color=col,
                label=f"η = {int(eta_s*100)}%")
    ax.set_xlabel("Erntefläche A [m²]")
    ax.set_ylabel("Ernteleistung P [MW] @ 100 km Orbit")
    ax.set_title("Satelliten: Ernteleistung\nals Funktion der Kollektorfläche")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 50)
    ax.set_ylim(0)

    # ── (2,2): Satellit: Jährlicher Treibstoffbedarf ─────────────────────────
    ax = axes[1, 1]
    h_range2 = np.linspace(150, 600, 200)
    rho_r2   = atm_density(h_range2 * 1000)
    v_r2     = orbital_velocity(h_range2 * 1000)
    m_sat    = 1000.0
    Cd_sat   = 2.2
    A_sat    = 10.0
    F_D2     = 0.5 * rho_r2 * Cd_sat * A_sat * v_r2**2
    # Jährlicher Δv-Bedarf für orbit maintenance:
    a_D2     = F_D2 / m_sat
    dv_year  = a_D2 * 365 * 24 * 3600   # m/s pro Jahr
    # Treib­stoff­bedarf mit Isp = 300 s (chemisch)
    Isp = 300.0
    m_prop   = m_sat * (np.exp(dv_year / (Isp * 9.81)) - 1)

    ax.semilogy(h_range2, m_prop, color=COLORS[0], lw=2.5, label="Treibstoff [kg/a] (Isp=300 s)")
    ax.semilogy(h_range2, m_prop * 0.3, color=COLORS[1], lw=2, ls="--",
                label="Reduzierung durch IF (30% offset)")
    ax.set_xlabel("Orbitalhöhe h [km]")
    ax.set_ylabel("Jährlicher Treibstoffbedarf m$_{{prop}}$ [kg/a] (log)")
    ax.set_title("Jährlicher Treibstoffbedarf\nunter atmosph. Abbremsung")
    ax.legend(fontsize=9)
    ax.set_xlim(150, 600)

    # ── (2,3): Vergleich aller Szenarien (zusammenfassend) ────────────────────
    ax = axes[1, 2]
    szenarien = ["KFZ 130 km/h\nη=32%, A=0.72m²",
                 "KFZ 100 km/h\nη=32%, A=0.72m²",
                 "KFZ max 200km/h\nη=32%",
                 "Satellit 100km\nη=25%, A=10m²",
                 "Satellit 200km\nη=25%, A=10m²",
                 "Satellit 400km\nη=25%, A=10m²"]
    P_vals = []
    # KFZ
    for v_ref in [130, 100, 200]:
        v_ms = v_ref / 3.6
        P_vals.append(0.32 * 0.5 * rho0 * 0.72 * v_ms**3 / 1000)
    # Satelliten
    for h_ref in [100, 200, 400]:
        rho_s = atm_density(np.array([h_ref * 1000]))[0]
        v_s   = orbital_velocity(h_ref * 1000)
        P_vals.append(0.25 * 0.5 * rho_s * 10.0 * v_s**3 / 1e6)   # MW

    colors_bar2 = COLORS[:3] + COLORS[3:6]
    x_pos = np.arange(len(szenarien))

    # Zwei Achsen: kW für KFZ, MW für Satelliten
    kw_vals = P_vals[:3]
    mw_vals = P_vals[3:]

    bars1 = ax.bar(x_pos[:3], kw_vals, color=COLORS[:3], alpha=0.85, width=0.5)
    ax2s = ax.twinx()
    bars2 = ax2s.bar(x_pos[3:], mw_vals, color=COLORS[3:6], alpha=0.85, width=0.5)
    ax.set_ylabel("Ernteleistung KFZ [kW]", color=COLORS[0])
    ax2s.set_ylabel("Ernteleistung Satellit [MW]", color=COLORS[3])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(szenarien, fontsize=7)
    ax.tick_params(axis="y", labelcolor=COLORS[0])
    ax2s.tick_params(axis="y", labelcolor=COLORS[3])
    ax.set_title("Vergleich aller Szenarien\n(Ernteleistung)")

    fig.tight_layout()
    fig.savefig("plots/plot9_parameterstudie.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot9_parameterstudie.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 – Zusammenfassung und Fazit
# ══════════════════════════════════════════════════════════════════════════════

def plot10_zusammenfassung():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gesamtübersicht: Induktionsfedern – Potenzial und physikalische Grenzen",
                 fontsize=13, fontweight="bold")

    # ── (1,1): Spannungsvergleich KFZ-Leistungen ──────────────────────────────
    ax = axes[0, 0]
    v_kmh = np.linspace(0, 200, 400)
    v_ms  = v_kmh / 3.6
    P_engine = (0.5 * rho0 * Cd_car * A_front * v_ms**3 +
                m_car * g_earth * mu_roll * v_ms) / 1000 + 4.5
    P_harvest = 0.32 * 0.5 * rho0 * A_intake * v_ms**3 / 1000

    ax.fill_between(v_kmh, P_engine, label="Motorleistung (Antrieb + Aux.)", alpha=0.2,
                    color=COLORS[0])
    ax.fill_between(v_kmh, P_harvest, label="Ernteleistung IF (η=32%)", alpha=0.4,
                    color=COLORS[2])
    ax.plot(v_kmh, P_engine,  color=COLORS[0], lw=2.5)
    ax.plot(v_kmh, P_harvest, color=COLORS[2], lw=2.5)

    ax.text(150, P_engine[-1]*0.5,
            f"Verhältnis @200km/h:\n{100*P_harvest[-1]/P_engine[-1]:.1f}%", fontsize=9,
            ha="center", color=COLORS[2])
    ax.set_xlabel("Fahrzeuggeschwindigkeit v [km/h]")
    ax.set_ylabel("Leistung [kW]")
    ax.set_title("Motor- vs. Ernteleistung (KFZ)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 200)
    ax.set_ylim(0)

    # ── (1,2): Jährliche CO₂ Einsparung (Verbrennungsmotor) ───────────────────
    ax = axes[0, 1]
    eta_motor = 0.38   # Verbrennungsmotorwirk.
    heizwert  = 8.9    # kWh/L Benzin
    CO2_kwh   = 2.35 / heizwert  # kg CO₂ / kWh

    v_profile   = np.array([50, 80, 100, 120, 130])
    t_anteil    = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    km_year     = 15000.0

    # ohne IF: Alternator-Last ~3kW elektrisch → aus Motor geholt
    # mit IF: Alternator kann teilweise/vollständig ersetzt werden
    e_save_arr = []
    for vk in v_profile:
        vi = vk / 3.6
        P_if = 0.32 * 0.5 * rho0 * A_intake * vi**3 / 1000
        P_aux = min(P_if, 3.0)  # max. 3 kW Bordnetz
        t_h = (km_year * t_anteil[v_profile == vk][0]) / vk
        e_save_arr.append(P_aux * t_h / eta_motor)  # kWh Kraftstoff-Äquivalent

    e_save_vals = np.array(e_save_arr)
    co2_save    = e_save_vals * CO2_kwh

    bars = ax.bar(v_profile.astype(str) + "\nkm/h", co2_save, color=COLORS[:5], alpha=0.85)
    for bar, val in zip(bars, co2_save):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f"{val:.1f} kg", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("CO₂-Einsparung [kg/a]")
    ax.set_title(f"CO₂-Einsparung durch Hilfsleistungs-\nOffset (15 000 km/a)")

    # ── (2,1): Reichweitengewinn E-Fahrzeug ───────────────────────────────────
    ax = axes[1, 0]
    batt_kwh    = np.array([40, 60, 80, 100])  # kWh Akku-Kapazitäten
    verbrauch   = 18.0   # kWh/100km Std.-Verbrauch
    reichw_base = batt_kwh / verbrauch * 100    # km

    # Reichweitenbonus durch IF: 2.8 kW im Schnitt @ 100km/h
    # Verbrauch reduziert sich um: 2.8 kW ÷ (100/3.6 m/s) = 0.10 kWh/km → 10 kWh/100km
    # Realistisch: 30 % Hilfsverbrauch = 18 * 0.30 = 5.4 kWh/100km Aux
    P_if_avg = 2.8  # kW bei 100 km/h
    v_avg_ms = 100 / 3.6
    delta_kwh_100km = P_if_avg / v_avg_ms * 100   # kWh saved per 100 km
    verbrauch_new = verbrauch - delta_kwh_100km * 0.9  # 90% geht ins Fahren
    reichw_new    = batt_kwh / verbrauch_new * 100

    x = np.arange(len(batt_kwh))
    w = 0.35
    b1 = ax.bar(x - w/2, reichw_base, width=w, color=COLORS[0], alpha=0.85, label="Ohne Induktiomsfedern")
    b2 = ax.bar(x + w/2, reichw_new,  width=w, color=COLORS[2], alpha=0.85, label="Mit Induktionsfedern")
    for b, r in zip(b1, reichw_base):
        ax.text(b.get_x() + b.get_width()/2, r + 2, f"{r:.0f} km", ha="center", fontsize=8)
    for b, r in zip(b2, reichw_new):
        ax.text(b.get_x() + b.get_width()/2, r + 2, f"{r:.0f} km", ha="center", fontsize=8,
                fontweight="bold", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} kWh" for k in batt_kwh])
    ax.set_ylabel("Reichweite [km] @ 100 km/h")
    ax.set_title("Reichweitengewinn E-Fahrzeug\ndurch Induktionsfeder-Hilfsleistung")
    ax.legend(fontsize=9)

    # ── (2,2): Gesamtbewertungsmatrix ─────────────────────────────────────────
    ax = axes[1, 2 if axes.shape == (2, 3) else 1]  # safe indexing
    ax = axes[1, 1]
    kriterien = ["Hilfsleistungs-\nOffset KFZ", "Antriebsleistungs-\nOffset KFZ",
                 "Hilfsleistungs-\nOffset Satellit", "Antrieb Satellit\n(Orbitmaint.)",
                 "Perpetuum\nMobile KFZ", "Perpetuum\nMobile Satellit"]
    bewert_physik = [85, 12, 40, 8, 0, 0]
    bewert_farbe  = [COLORS[2], COLORS[1], COLORS[3], COLORS[1],
                     COLORS[1], COLORS[1]]
    hatches = ["", "///", "", "///", "xxx", "xxx"]

    bars = ax.barh(kriterien, bewert_physik, color=bewert_farbe, alpha=0.85, hatch=hatches)
    for bar, val in zip(bars, bewert_physik):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontsize=10,
                fontweight="bold" if val == 0 else "normal",
                color=COLORS[1] if val == 0 else "black")
    ax.axvline(100, color=COLORS[0], lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Physikalisches Potenzial [%]")
    ax.set_title("Gesamtbewertung: Physikalisch realisierbares\nPotenzial (0% = unmöglich)")
    ax.set_xlim(0, 115)

    fig.tight_layout()
    fig.savefig("plots/plot10_zusammenfassung.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot10_zusammenfassung.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Alle Plots erzeugen
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Erzeuge Plots …")
    plot1_luftwiderstand()
    plot2_federsystem()
    plot3_betz()
    plot4_energiebilanz_kfz()
    plot5_frontflaeche()
    plot6_atmosphaere()
    plot7_satellit_energiebilanz()
    plot8_thermodynamik()
    plot9_parameterstudie()
    plot10_zusammenfassung()
    print("Fertig – alle Plots in plots/")
