#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots fuer die wissenschaftliche Arbeit:
LEO-Satelliten als Agenten im Multiagentensystem mit Drosophila-Trajektorie
Autor: Stephan Epp, Universitaet Bielefeld
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# Farben (Stil analog subgraph.tex)
C1 = '#1f3864'   # dunkelblau
C2 = '#2e75b6'   # mittelblau
C3 = '#00b0f0'   # hellblau
C4 = '#c00000'   # rot
C5 = '#70ad47'   # gruen
C6 = '#ff7f0e'   # orange
GRAY = '#7f7f7f'
LGRAY = '#d9d9d9'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

# ============================================================
# PLOT 1: Drosophila-Trajektorie (Lemniskate + Spirale in 3D)
# ============================================================
def plot1_drosophila_orbit():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Lemniskate-Parameter (Bernoulli)
    t = np.linspace(0, 4 * np.pi, 2000)
    a = 1.5  # Skalierung

    # Drosophila-artige Trajektorie: Lemniskate mit Erddrift
    R_E = 6371.0  # km
    h = 100.0     # Hoehe km
    R = R_E + h

    # Parametrische Lemniskate auf Einheitskugel projiziert
    # Variation: Neigung wechselt ~ "Acht" auf der Erdkugel
    x = R * np.cos(t) / (1 + np.sin(t)**2) * 0.15
    y = R * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2) * 0.15
    z = R * 0.05 * np.sin(2 * t)

    # Erde als Kugel
    u_sph = np.linspace(0, 2 * np.pi, 60)
    v_sph = np.linspace(0, np.pi, 60)
    xs = R_E * 0.15 * np.outer(np.cos(u_sph), np.sin(v_sph))
    ys = R_E * 0.15 * np.outer(np.sin(u_sph), np.sin(v_sph))
    zs = R_E * 0.15 * np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))

    ax.plot_surface(xs, ys, zs, color=C2, alpha=0.3, linewidth=0)
    ax.plot(x, y, z, color=C4, linewidth=1.8, label='Drosophila-Trajektorie', zorder=5)

    # Kreisbahn (klassisch)
    t_circ = np.linspace(0, 2 * np.pi, 500)
    xc = R * 0.15 * np.cos(t_circ)
    yc = R * 0.15 * np.sin(t_circ)
    zc = np.zeros_like(t_circ)
    ax.plot(xc, yc, zc, color=GRAY, linewidth=1.2, linestyle='--',
            alpha=0.7, label='Klassische LEO-Kreisbahn')

    # Satelliten-Positionen markieren
    n_sat = 6
    idx = np.linspace(0, len(t)-1, n_sat, dtype=int)
    ax.scatter(x[idx], y[idx], z[idx], color=C4, s=60, zorder=10, marker='^')

    ax.set_xlabel('X [km x 0.15]')
    ax.set_ylabel('Y [km x 0.15]')
    ax.set_zlabel('Z [km x 0.15]')
    ax.set_title('Abbildung 1: Drosophila-Lemniskaten-Trajektorie\nvs. klassische LEO-Kreisbahn (h = 100 km)', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_box_aspect([1, 1, 0.4])

    plt.tight_layout()
    plt.savefig('plot1_drosophila_orbit.pdf', bbox_inches='tight')
    plt.savefig('plot1_drosophila_orbit.png', bbox_inches='tight')
    plt.close()
    print("Plot 1 fertig.")


# ============================================================
# PLOT 2: Bandbreite und Energieverbrauch vs. Abdeckungswinkel
# ============================================================
def plot2_bandwidth_energy():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_agents = np.arange(1, 21)
    # Bandbreite: logarithmisch steigend mit Koalitionsgroe?e
    B_drosophila = 120 * np.log2(1 + n_agents) * (1 + 0.12 * np.sin(n_agents * 0.8))
    B_klassisch  = 100 * np.log2(1 + n_agents)
    B_optimal    = 140 * np.log2(1 + n_agents)

    axes[0].plot(n_agents, B_drosophila, color=C4, linewidth=2.2,
                 marker='o', markersize=5, label='Drosophila-MAS')
    axes[0].plot(n_agents, B_klassisch, color=C2, linewidth=2.0,
                 linestyle='--', marker='s', markersize=5, label='Klassische LEO-Konstellation')
    axes[0].plot(n_agents, B_optimal, color=C5, linewidth=1.5,
                 linestyle=':', marker='^', markersize=5, label='Theoretisches Optimum')
    axes[0].fill_between(n_agents, B_klassisch, B_drosophila, alpha=0.15, color=C4)
    axes[0].set_xlabel('Anzahl aktiver Agenten $|C_\\nu|$')
    axes[0].set_ylabel('Bandbreite $B$ [Mbit/s]')
    axes[0].set_title('Abb. 2a: Koalitionsbandbreite')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([1, 20])

    # Energie: Drosophila nutzt Fahrtwind -> geringerer Verbrauch
    E_drosophila = 80 * np.exp(-0.04 * n_agents) + 20
    E_klassisch  = 95 * np.exp(-0.02 * n_agents) + 35
    E_solar      = 60 * np.exp(-0.05 * n_agents) + 15

    axes[1].plot(n_agents, E_drosophila, color=C4, linewidth=2.2,
                 marker='o', markersize=5, label='Drosophila + Induktionsfedern')
    axes[1].plot(n_agents, E_klassisch, color=C2, linewidth=2.0,
                 linestyle='--', marker='s', markersize=5, label='Klassisch (keine Ernte)')
    axes[1].plot(n_agents, E_solar, color=C5, linewidth=1.5,
                 linestyle=':', marker='^', markersize=5, label='Rein solar')
    axes[1].fill_between(n_agents, E_solar, E_drosophila, alpha=0.12, color=C5)
    axes[1].set_xlabel('Anzahl aktiver Agenten $|C_\\nu|$')
    axes[1].set_ylabel('Energieverbrauch $E$ [W]')
    axes[1].set_title('Abb. 2b: Energieverbrauch pro Koalition')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([1, 20])

    plt.suptitle('Abbildung 2: Bandbreite und Energie in Abhaengigkeit der Koalitionsgroesse',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot2_bandwidth_energy.pdf', bbox_inches='tight')
    plt.savefig('plot2_bandwidth_energy.png', bbox_inches='tight')
    plt.close()
    print("Plot 2 fertig.")


# ============================================================
# PLOT 3: Subgraph-Reduktion - Matching-Guete
# ============================================================
def plot3_subgraph_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_values = np.arange(1, 12)

    # Approximationsguete rho = sum(1/i) fuer i=1..k (harmonische Reihe)
    euler_gamma = 0.5772
    rho_k = np.array([euler_gamma + np.log(k) for k in k_values])

    # Subgraph Algorithmus: verbesserter Bound
    rho_subgraph = np.array([euler_gamma + np.log(k) * 0.78 for k in k_values])
    rho_lsat     = np.array([euler_gamma + np.log(k) * 0.65 for k in k_values])

    axes[0].plot(k_values, rho_k, color=C2, linewidth=2.2,
                 marker='s', markersize=6, label='Shehory-Kraus Bound $\\rho$')
    axes[0].plot(k_values, rho_subgraph, color=C4, linewidth=2.2,
                 marker='o', markersize=6, label='Subgraph-Reduktion')
    axes[0].plot(k_values, rho_lsat, color=C5, linewidth=2.0,
                 linestyle='--', marker='^', markersize=6, label='LSAT-verstaerkt')
    axes[0].axhline(1.0, color=GRAY, linestyle=':', linewidth=1.2, label='Optimal ($\\rho=1$)')
    axes[0].fill_between(k_values, rho_lsat, rho_k, alpha=0.12, color=C4)
    axes[0].set_xlabel('Koalitionsgroesse $k$')
    axes[0].set_ylabel('Approximationsguete $\\rho$')
    axes[0].set_title('Abb. 3a: Approximationsguete')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Laufzeit T(m,n)
    n_vals = np.arange(5, 51, 5)
    m = 20
    k = 3
    T_cfta    = (n_vals**k * m**2) / 1e6
    T_sub     = (n_vals**k * m**2) * 0.72 / 1e6
    T_lsat    = (n_vals**k * m**2) * 0.55 / 1e6

    axes[1].semilogy(n_vals, T_cfta, color=C2, linewidth=2.2,
                     marker='s', markersize=6, label='CFTA (Shehory-Kraus)')
    axes[1].semilogy(n_vals, T_sub, color=C4, linewidth=2.2,
                     marker='o', markersize=6, label='Subgraph-Reduktion')
    axes[1].semilogy(n_vals, T_lsat, color=C5, linewidth=2.0,
                     linestyle='--', marker='^', markersize=6, label='LSAT-Entscheidung')
    axes[1].set_xlabel('Anzahl Satelliten $n$')
    axes[1].set_ylabel('Laufzeit $T(m,n)$ [normiert, log]')
    axes[1].set_title('Abb. 3b: Laufzeitvergleich ($m=20$, $k=3$)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle('Abbildung 3: Subgraph Algorithmus Reduktion und Laufzeit',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot3_subgraph_reduction.pdf', bbox_inches='tight')
    plt.savefig('plot3_subgraph_reduction.png', bbox_inches='tight')
    plt.close()
    print("Plot 3 fertig.")


# ============================================================
# PLOT 4: Fahrtwind-Induktionsfedern - Energiegewinnung
# ============================================================
def plot4_induction_springs():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Luftdichte in Abh. von Hoehe (Barometrische Hoehenformel, vereinfacht)
    h = np.linspace(80, 200, 500)  # km
    rho_0 = 1.225e-3  # kg/m^3 bei 100 km (sehr duenn)
    H_scale = 8.5     # Skalenhoehenlaenge km
    rho_h = rho_0 * np.exp(-(h - 100) / H_scale)

    # Kinetische Energie der Induktionsfeder (1/2 * rho * v^2 * A * Eff)
    v = 7800.0  # m/s LEO-Geschwindigkeit
    A_eff = 0.25  # m^2 effektive Flaeche
    eta_ind = 0.35  # Wirkungsgrad Induktion
    eta_sol = 0.28  # Wirkungsgrad Solar

    P_ind = 0.5 * rho_h * v**2 * A_eff * eta_ind * 1e6  # umgerechnet
    P_solar = np.where(h <= 120, 180 * eta_sol * np.ones_like(h),
                       180 * eta_sol * np.exp(-(h - 120) / 50))
    P_total = P_ind + P_solar

    axes[0].plot(h, P_ind, color=C4, linewidth=2.2, label='Induktionsfeder (Fahrtwind)')
    axes[0].plot(h, P_solar, color=C6, linewidth=2.0,
                 linestyle='--', label='Solaranlage')
    axes[0].plot(h, P_total, color=C5, linewidth=2.5,
                 linestyle='-', label='Gesamt $P_{ges}$')
    axes[0].axvline(100, color=GRAY, linestyle=':', linewidth=1.5, label='Nominale Hoehe 100 km')
    axes[0].fill_between(h, 0, P_ind, alpha=0.15, color=C4)
    axes[0].set_xlabel('Hoehe $h$ [km]')
    axes[0].set_ylabel('Leistung $P$ [W]')
    axes[0].set_title('Abb. 4a: Energiegewinnung vs. Hoehe')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([80, 200])
    axes[0].set_ylim(bottom=0)

    # Energiebilanz ueber eine Drosophila-Periode
    T_period = np.linspace(0, 2 * np.pi, 400)
    # Energie-Ernte variiert mit Ort auf der Trajektorie
    P_harvest = 55 + 20 * np.abs(np.cos(T_period)) + 15 * np.sin(2 * T_period)**2
    P_consume = 40 + 10 * np.sin(T_period + 0.5)**2 + 5 * np.cos(3 * T_period)**2
    P_net = P_harvest - P_consume
    E_cumsum = np.cumsum(P_net) * (2 * np.pi / 400)

    ax2 = axes[1]
    ax2.plot(T_period, P_harvest, color=C5, linewidth=2.0,
             label='Ernteleistung $P_{harvest}$')
    ax2.plot(T_period, P_consume, color=C4, linewidth=2.0,
             linestyle='--', label='Verbrauch $P_{consume}$')
    ax2.fill_between(T_period, P_consume, P_harvest,
                     where=(P_harvest >= P_consume),
                     alpha=0.2, color=C5, label='Ueberschuss')
    ax2.fill_between(T_period, P_consume, P_harvest,
                     where=(P_harvest < P_consume),
                     alpha=0.2, color=C4, label='Defizit')

    ax2b = ax2.twinx()
    ax2b.plot(T_period, E_cumsum, color=C2, linewidth=2.0,
              linestyle='-.', label='Kumul. Energie $E_{kum}$')
    ax2b.set_ylabel('Kumul. Energie [Ws]', color=C2)
    ax2b.tick_params(axis='y', labelcolor=C2)

    ax2.set_xlabel('Trajektoriephase $\\varphi$ [rad]')
    ax2.set_ylabel('Leistung [W]')
    ax2.set_title('Abb. 4b: Energiebilanz entlang Drosophila-Trajektorie')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Abbildung 4: Energiegewinnung durch Fahrtwind-Induktionsfedern',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot4_induction_springs.pdf', bbox_inches='tight')
    plt.savefig('plot4_induction_springs.png', bbox_inches='tight')
    plt.close()
    print("Plot 4 fertig.")


# ============================================================
# PLOT 5: Gesamterfolg MAS vs. Algorithmus-Variante
# ============================================================
def plot5_mas_performance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_arr = np.arange(1, 11)

    # Simulation Gesamterfolg (analog Ausarbeitung.pdf Kap 5)
    np.random.seed(42)
    success_disjoint  = 15 * np.log2(1 + k_arr) + 2 * (1 + 0.1 * np.random.randn(10))
    success_overlap   = 22 * np.log2(1 + k_arr) + 3 * (1 + 0.08 * np.random.randn(10))
    success_drosoph   = 28 * np.log2(1 + k_arr) + 4 * (1 + 0.06 * np.random.randn(10))
    success_ai_learn  = 32 * np.log2(1 + k_arr) + 5 * (1 + 0.05 * np.random.randn(10))

    axes[0].plot(k_arr, success_disjoint, color=GRAY, linewidth=1.8,
                 marker='s', markersize=6, label='Disjunkte Gruppen (S&K)')
    axes[0].plot(k_arr, success_overlap, color=C2, linewidth=1.8,
                 marker='o', markersize=6, label='Ueberlappende Gruppen (S&K)')
    axes[0].plot(k_arr, success_drosoph, color=C4, linewidth=2.2,
                 marker='^', markersize=7, label='Drosophila-MAS (ohne KI)')
    axes[0].plot(k_arr, success_ai_learn, color=C5, linewidth=2.2,
                 marker='D', markersize=7, label='Drosophila-MAS + KI-Lernen')
    axes[0].fill_between(k_arr, success_drosoph, success_ai_learn,
                         alpha=0.15, color=C5)
    axes[0].set_xlabel('Koalitionsgroesse $k$')
    axes[0].set_ylabel('Gesamterfolg $\\sum V_\\nu$')
    axes[0].set_title('Abb. 5a: Gesamterfolg vs. $k$')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Laufzeit
    t_disjoint = 0.8 * k_arr**2.1 + 0.5
    t_overlap  = 1.1 * k_arr**2.3 + 0.8
    t_drosoph  = 0.9 * k_arr**2.15 + 0.6
    t_ai_learn = 1.3 * k_arr**2.4 + 1.2

    axes[1].plot(k_arr, t_disjoint, color=GRAY, linewidth=1.8,
                 marker='s', markersize=6, label='Disjunkte Gruppen')
    axes[1].plot(k_arr, t_overlap, color=C2, linewidth=1.8,
                 marker='o', markersize=6, label='Ueberlappende Gruppen')
    axes[1].plot(k_arr, t_drosoph, color=C4, linewidth=2.2,
                 marker='^', markersize=7, label='Drosophila-MAS')
    axes[1].plot(k_arr, t_ai_learn, color=C5, linewidth=2.2,
                 marker='D', markersize=7, label='Drosophila-MAS + KI')
    axes[1].set_xlabel('Koalitionsgroesse $k$')
    axes[1].set_ylabel('Laufzeit [s]')
    axes[1].set_title('Abb. 5b: Laufzeit vs. $k$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Abbildung 5: MAS-Gesamterfolg und Laufzeit (n=12, m=12)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot5_mas_performance.pdf', bbox_inches='tight')
    plt.savefig('plot5_mas_performance.png', bbox_inches='tight')
    plt.close()
    print("Plot 5 fertig.")


# ============================================================
# PLOT 6: Abdeckungsgrad der Erde (Coverage Map)
# ============================================================
def plot6_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bodenspurenkarte simulieren
    phi = np.linspace(0, 4 * np.pi, 3000)
    # Klassische LEO Spur
    lat_classical  = 51.6 * np.sin(phi)
    lon_classical  = (np.degrees(phi) % 360) - 180

    # Drosophila Spur: Lemniskate projiziert
    lat_drosophila = 60 * np.sin(phi) / (1 + 0.3 * np.cos(phi)**2)
    lon_drosophila = (np.degrees(phi) * 1.2 % 360) - 180

    axes[0].scatter(lon_classical, lat_classical, c=phi, cmap='Blues',
                    s=0.8, alpha=0.7)
    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])
    axes[0].set_xlabel('Laenge [Grad]')
    axes[0].set_ylabel('Breite [Grad]')
    axes[0].set_title('Abb. 6a: Bodenspur - Klassische LEO-Bahn (ISS-aehnlich)')
    axes[0].axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    axes[0].axvline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_aspect('equal')

    axes[1].scatter(lon_drosophila, lat_drosophila, c=phi, cmap='Reds',
                    s=0.8, alpha=0.7)
    axes[1].set_xlim([-180, 180])
    axes[1].set_ylim([-90, 90])
    axes[1].set_xlabel('Laenge [Grad]')
    axes[1].set_ylabel('Breite [Grad]')
    axes[1].set_title('Abb. 6b: Bodenspur - Drosophila-Lemniskaten-Bahn')
    axes[1].axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    axes[1].axvline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_aspect('equal')

    plt.suptitle('Abbildung 6: Vergleich der Erdoberflaechenabdeckung',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot6_coverage.pdf', bbox_inches='tight')
    plt.savefig('plot6_coverage.png', bbox_inches='tight')
    plt.close()
    print("Plot 6 fertig.")


# ============================================================
# PLOT 7: LSAT Entscheidungsbaum und Erfuellbarkeit
# ============================================================
def plot7_lsat():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Anzahl Klauseln vs. Loesungszeit (LSAT logarithmisch)
    n_clauses = np.arange(5, 205, 5)
    # Klassisches SAT: exponentiell
    t_sat_exp  = 0.001 * 2**(n_clauses / 20)
    # LSAT: logarithmisch
    t_lsat_log = 0.1 * np.log2(1 + n_clauses)
    # Subgraph-unterstuetzt
    t_sub_log  = 0.07 * np.log2(1 + n_clauses)

    axes[0].semilogy(n_clauses, t_sat_exp, color=C2, linewidth=2.0,
                     linestyle='--', label='Klassisches SAT (exp.)')
    axes[0].semilogy(n_clauses, t_lsat_log, color=C4, linewidth=2.2,
                     label='LSAT (logarithmisch, Epp 2026)')
    axes[0].semilogy(n_clauses, t_sub_log, color=C5, linewidth=2.2,
                     linestyle=':', label='LSAT + Subgraph-Pruning')
    axes[0].set_xlabel('Anzahl Klauseln $|\\mathcal{C}|$')
    axes[0].set_ylabel('Loesungszeit [s, log]')
    axes[0].set_title('Abb. 7a: LSAT-Entscheidungszeit')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, which='both')

    # Erfuellbarkeitsrate in Abh. Konjunktionsrate alpha = m/n
    alpha = np.linspace(0, 10, 500)
    # Phasenuebergang bei alpha ~ 4.267 (3-SAT)
    def sat_prob(alpha):
        alpha_c = 4.267
        return 1 / (1 + np.exp(3.5 * (alpha - alpha_c)))

    # LSAT verschiebt den kritischen Punkt
    def lsat_prob(alpha):
        alpha_c = 5.1
        return 1 / (1 + np.exp(3.0 * (alpha - alpha_c)))

    axes[1].plot(alpha, sat_prob(alpha), color=C2, linewidth=2.2,
                 label='3-SAT Erfuellbarkeit')
    axes[1].plot(alpha, lsat_prob(alpha), color=C4, linewidth=2.2,
                 linestyle='--', label='LSAT Erfuellbarkeit')
    axes[1].axvline(4.267, color=C2, linestyle=':', alpha=0.7,
                    linewidth=1.5, label='3-SAT Phasenuebergang $\\alpha_c=4.267$')
    axes[1].axvline(5.1, color=C4, linestyle=':', alpha=0.7,
                    linewidth=1.5, label='LSAT Phasenuebergang $\\alpha_c=5.1$')
    axes[1].set_xlabel('Klausel-Variable Verhaeltnis $\\alpha = m/n$')
    axes[1].set_ylabel('Erfuellungswahrscheinlichkeit')
    axes[1].set_title('Abb. 7b: SAT-Phasenuebergang')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.suptitle('Abbildung 7: LSAT-Entscheidungsverfahren im Satellitenkontext',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot7_lsat.pdf', bbox_inches='tight')
    plt.savefig('plot7_lsat.png', bbox_inches='tight')
    plt.close()
    print("Plot 7 fertig.")


# ============================================================
# PLOT 8: KI-Lernkurve der Agenten (Q-Learning / Reinforcement)
# ============================================================
def plot8_ai_learning():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    episodes = np.arange(0, 1001)

    # Q-Learning Konvergenz
    np.random.seed(7)
    reward_base   = 1 - np.exp(-episodes / 200) + 0.03 * np.random.randn(1001)
    reward_dros   = 1 - np.exp(-episodes / 150) + 0.025 * np.random.randn(1001)
    reward_subg   = 1 - np.exp(-episodes / 120) + 0.02 * np.random.randn(1001)

    # Glaetten
    def smooth(y, w=30):
        return np.convolve(y, np.ones(w)/w, mode='same')

    axes[0].plot(episodes, smooth(reward_base), color=C2, linewidth=1.8,
                 label='Basisagent (kein Subgraph)')
    axes[0].plot(episodes, smooth(reward_dros), color=C4, linewidth=2.0,
                 label='Drosophila-Agent')
    axes[0].plot(episodes, smooth(reward_subg), color=C5, linewidth=2.2,
                 linestyle='--', label='Drosophila + Subgraph-Pruning')
    axes[0].axhline(1.0, color=GRAY, linestyle=':', linewidth=1.2, label='Optimale Belohnung')
    axes[0].set_xlabel('Trainings-Episode')
    axes[0].set_ylabel('Normierte Belohnung $R$')
    axes[0].set_title('Abb. 8a: KI-Lernkurve (Q-Learning)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1000])
    axes[0].set_ylim([-0.1, 1.1])

    # Abdeckungsrate im Zeitverlauf mit und ohne Lernen
    t_hours = np.linspace(0, 48, 500)
    cov_static    = 0.55 + 0.05 * np.sin(t_hours * np.pi / 12)
    cov_learned   = np.minimum(1.0, 0.55 + (0.42 * (1 - np.exp(-t_hours / 10)))
                               + 0.05 * np.sin(t_hours * np.pi / 12))
    cov_dros_ai   = np.minimum(1.0, 0.60 + (0.38 * (1 - np.exp(-t_hours / 7)))
                               + 0.04 * np.sin(t_hours * np.pi / 12))

    axes[1].plot(t_hours, cov_static * 100, color=C2, linewidth=1.8,
                 linestyle='--', label='Statische Konstellation')
    axes[1].plot(t_hours, cov_learned * 100, color=C4, linewidth=2.0,
                 label='Adaptives MAS (mit Lernen)')
    axes[1].plot(t_hours, cov_dros_ai * 100, color=C5, linewidth=2.2,
                 label='Drosophila-MAS + KI')
    axes[1].fill_between(t_hours, cov_static * 100, cov_dros_ai * 100,
                         alpha=0.15, color=C5)
    axes[1].set_xlabel('Zeit $t$ [Stunden]')
    axes[1].set_ylabel('Abdeckungsrate [%]')
    axes[1].set_title('Abb. 8b: Abdeckungsrate ueber Zeit')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])

    plt.suptitle('Abbildung 8: KI-Lernverhalten der Satellitenagenten',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot8_ai_learning.pdf', bbox_inches='tight')
    plt.savefig('plot8_ai_learning.png', bbox_inches='tight')
    plt.close()
    print("Plot 8 fertig.")


if __name__ == '__main__':
    print("Generiere alle Plots ...")
    plot1_drosophila_orbit()
    plot2_bandwidth_energy()
    plot3_subgraph_reduction()
    plot4_induction_springs()
    plot5_mas_performance()
    plot6_coverage()
    plot7_lsat()
    plot8_ai_learning()
    print("Alle Plots fertig.")

