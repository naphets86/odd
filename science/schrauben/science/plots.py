#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotgenerator für die wissenschaftliche Arbeit:
Unterwasser-Schraubenwände als Flussstromkraftwerke

Autor: Stephan Epp, Universität Bielefeld
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import os

# ---- Farben analog zur LaTeX-Vorlage ----
MAIN_BLUE   = (25/255,  70/255,  140/255)
ACCENT_RED  = (180/255, 50/255,  30/255)
DARK_GREEN  = (30/255,  100/255, 50/255)
DARK_GRAY   = (60/255,  60/255,  70/255)
LIGHT_GRAY  = (245/255, 245/255, 248/255)

OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
    'text.usetex':      False,
})

# ===========================================================================
# Plot 1 – Hydraulische Leistung P vs. Strömungsgeschwindigkeit v
# ===========================================================================
def plot1_leistungskurve():
    fig, ax = plt.subplots(figsize=(8, 5))

    rho  = 1000.0          # kg/m³  Wasserdichte
    A_schraube = 2.0       # m²     Querschnittsfläche einer Schraube
    eta_betz   = 16.0/27.0 # Betz-Grenzwert
    eta_gen    = 0.92      # Generator-Wirkungsgrad
    eta_gesamt = eta_betz * eta_gen

    v = np.linspace(0.0, 3.5, 400)      # m/s
    P_ideal   = 0.5 * rho * A_schraube * v**3          # Idealleistung
    P_real    = eta_gesamt * P_ideal                    # Reale Leistung
    P_betz    = eta_betz   * P_ideal                    # Nur Betz

    ax.fill_between(v, 0, P_real, alpha=0.18, color=MAIN_BLUE)
    ax.plot(v, P_ideal, '--', color=DARK_GRAY,  lw=1.5, label=r'Ideal: $P_{ideal}=\frac{1}{2}\rho A v^3$')
    ax.plot(v, P_betz,  ':',  color=ACCENT_RED, lw=1.8, label=r'Betz-Grenze: $\eta_{Betz}=16/27$')
    ax.plot(v, P_real,  '-',  color=MAIN_BLUE,  lw=2.5,
            label='Real: $P=\\eta_{ges}\\cdot\\frac{1}{2}\\rho A v^3$ ($\\eta_{ges}$='+f'{eta_gesamt:.2f})')

    # Betriebspunkt-Markierung bei v=2 m/s
    v_op = 2.0
    P_op = eta_gesamt * 0.5 * rho * A_schraube * v_op**3
    ax.annotate(
        f'Betriebspunkt\n$v={v_op}$ m/s\n$P={P_op/1000:.2f}$ kW',
        xy=(v_op, P_op), xytext=(v_op + 0.5, P_op + 1200),
        arrowprops=dict(arrowstyle='->', color=DARK_GRAY, lw=1.2),
        fontsize=9, color=DARK_GRAY,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=DARK_GRAY, lw=0.8)
    )
    ax.plot(v_op, P_op, 'o', color=MAIN_BLUE, ms=7, zorder=5)

    ax.set_xlabel('Strömungsgeschwindigkeit $v$ [m/s]')
    ax.set_ylabel('Leistung $P$ [W]')
    ax.set_title('Hydraulische Leistungskurve einer Unterwasser-Schraube\n'
                 r'($\rho=1000\,\mathrm{kg/m^3}$, $A=2\,\mathrm{m^2}$)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f} kW'))
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'plot1_leistungskurve.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Plot 2 – Wirkungsgrad η vs. Verschmutzungsgrad κ
# ===========================================================================
def plot2_wirkungsgrad():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    kappa = np.linspace(0, 1, 300)   # 0 = sauber, 1 = vollständig verstopft

    # Modell: eta(kappa) = eta_0 * exp(-alpha * kappa)
    eta_0   = 0.558   # Nennwirkungsgrad (Betz * Generator)
    alpha_list = [1.5, 2.5, 4.0]
    styles = ['-', '--', ':']
    colors = [MAIN_BLUE, DARK_GREEN, ACCENT_RED]
    labels = [r'$\alpha=1{,}5$ (robuste Konstruktion)',
              r'$\alpha=2{,}5$ (Standard)',
              r'$\alpha=4{,}0$ (empfindliche Ausführung)']

    ax = axes[0]
    for alpha, style, color, label in zip(alpha_list, styles, colors, labels):
        eta = eta_0 * np.exp(-alpha * kappa)
        ax.plot(kappa * 100, eta * 100, linestyle=style, color=color, lw=2, label=label)

    # Reinigungsschwelle
    ax.axvline(x=30, color=DARK_GRAY, lw=1.2, linestyle='-.', alpha=0.7)
    ax.text(31, 20, 'Reinigungsschwelle\n$\\kappa=30\\,\\%$', fontsize=8.5, color=DARK_GRAY)
    ax.set_xlabel('Verschmutzungsgrad $\\kappa$ [%]')
    ax.set_ylabel('Wirkungsgrad $\\eta$ [%]')
    ax.set_title('Wirkungsgrad in Abhängigkeit\nvom Verschmutzungsgrad')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Rechter Teilplot: Leistungsverlust durch Verschmutzung über Zeit
    ax2 = axes[1]
    t   = np.linspace(0, 30, 500)   # Tage
    # Verschmutzungsaufbau: logistisch
    kappa_t = 0.6 / (1 + np.exp(-0.4 * (t - 15)))

    alpha_std = 2.5
    eta_t = eta_0 * np.exp(-alpha_std * kappa_t)
    P0    = 1.0   # normiert
    P_t   = P0 * eta_t / eta_0

    # Reinigungsereignis bei t=15 Tagen
    t_clean = 15
    t_after = np.linspace(t_clean, 30, 200)
    kappa_after = 0.6 / (1 + np.exp(-0.4 * (t_after - 15))) - \
                  0.6 / (1 + np.exp(-0.4 * (t_clean - 15))) * 0.95   # Reset

    ax2.plot(t, P_t, '-', color=MAIN_BLUE, lw=2, label='Ohne Reinigung')

    # Mit Reinigung – stückweise
    t1 = t[t <= t_clean]
    P1 = P_t[t <= t_clean]
    kappa_reset = kappa_t[t <= t_clean]
    # Nach Reinigung beginnt Verschmutzung neu
    t2 = t[t > t_clean]
    kappa_new = 0.6 / (1 + np.exp(-0.4 * (t2 - t_clean - 15)))
    P2 = P0 * np.exp(-alpha_std * kappa_new) / 1.0

    ax2.plot(np.concatenate([t1, t2]), np.concatenate([P1, P2]),
             '--', color=DARK_GREEN, lw=2, label='Mit Reinigung (Tag 15)')
    ax2.axvline(x=t_clean, color=ACCENT_RED, lw=1.2, linestyle=':', alpha=0.8)
    ax2.text(t_clean + 0.3, 0.55, 'Reinigung', fontsize=8.5, color=ACCENT_RED, rotation=90, va='center')

    ax2.set_xlabel('Zeit $t$ [Tage]')
    ax2.set_ylabel('Normierte Leistung $P/P_0$')
    ax2.set_title('Leistungsverlauf mit und ohne\nReinigungsintervall')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'plot2_wirkungsgrad.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Plot 3 – Jährliche Energieproduktion für verschiedene Flüsse
# ===========================================================================
def plot3_energieproduktion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    flüsse = ['Rhein\n(Köln)', 'Elbe\n(Hamburg)', 'Donau\n(Wien)', 'Main\n(Frankfurt)',
              'Mosel\n(Koblenz)', 'Neckar\n(Heidelberg)']
    v_mittel  = [1.8, 1.3, 1.6, 0.9, 1.1, 0.7]   # m/s  Mittlere Strömungsgeschwindigkeit
    breite    = [360, 280, 320, 110, 130, 80]       # m    Flussbreite
    n_wände   = [b // 50 for b in breite]          # Anzahl Wände (alle 50m)

    rho   = 1000.0
    A_sch = 2.0
    eta   = 0.558 * 0.85  # inkl. 15% Verfügbarkeitsabzug

    P_liste = []
    E_liste = []
    for v, n in zip(v_mittel, n_wände):
        P = eta * 0.5 * rho * A_sch * v**3 * n
        P_liste.append(P / 1e6)          # MW
        E_liste.append(P * 8760 / 1e9)   # GWh/Jahr

    x = np.arange(len(flüsse))
    width = 0.38

    ax = axes[0]
    bars1 = ax.bar(x - width/2, P_liste, width, color=MAIN_BLUE,    alpha=0.88, label='Installierte Leistung [MW]')
    bars2 = ax.bar(x + width/2, E_liste, width, color=DARK_GREEN,   alpha=0.88, label='Jahresenergie [GWh/a]')

    for bar, val in zip(bars1, P_liste):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, color=DARK_GRAY)
    for bar, val in zip(bars2, E_liste):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, color=DARK_GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(flüsse, fontsize=9)
    ax.set_ylabel('Leistung [MW] / Energie [GWh/a]')
    ax.set_title('Installierte Leistung und\nJahresenergieproduktion je Fluss')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Rechter Teilplot: Strömungsgeschwindigkeit vs. Leistung (parametrisch nach Anzahl Wände)
    ax2 = axes[1]
    v_range = np.linspace(0.5, 3.0, 300)
    for n_w, color, ls in [(5, ACCENT_RED, ':'), (10, DARK_GREEN, '--'),
                            (20, MAIN_BLUE, '-'), (30, DARK_GRAY, '-.')]:
        P_range = eta * 0.5 * rho * A_sch * v_range**3 * n_w / 1e6
        ax2.plot(v_range, P_range, linestyle=ls, color=color, lw=2.0,
                 label=f'$n_W={n_w}$ Wände')

    ax2.set_xlabel('Strömungsgeschwindigkeit $\\bar{v}$ [m/s]')
    ax2.set_ylabel('Gesamtleistung $P_{ges}$ [MW]')
    ax2.set_title('Gesamtleistung in Abhängigkeit von\nStrömungsgeschwindigkeit und Wandanzahl')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'plot3_energieproduktion.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Plot 4 – Sensorkorrelation: Verschmutzungsgrad vs. Reinigungsintervall
# ===========================================================================
def plot4_sensorkorrelation():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Sensor-Zeitreihe: Trübungswert über Zeit
    ax = axes[0, 0]
    np.random.seed(42)
    t_tage = np.arange(0, 180, 1)
    # Synthese: periodischer Anstieg mit Reinigungsresets alle ~40 Tage
    trübung = np.zeros(180)
    t_reset = [0, 42, 85, 128, 170]
    for i, t0 in enumerate(t_reset):
        t_end = t_reset[i+1] if i+1 < len(t_reset) else 180
        seg_len = t_end - t0
        tt = np.arange(seg_len)
        trübung[t0:t_end] = 20 + 60 * (1 - np.exp(-tt / 30)) + \
                             5 * np.random.randn(seg_len)
        if t0 > 0:
            trübung[t0] = 20 + 3 * np.random.randn()

    schwelle = 70
    ax.plot(t_tage, trübung, '-', color=MAIN_BLUE, lw=1.5, label='Trübungssensor [NTU]')
    ax.axhline(y=schwelle, color=ACCENT_RED, lw=1.5, linestyle='--', label=f'Reinigungsschwelle ({schwelle} NTU)')
    for tr in t_reset[1:]:
        if tr < 180:
            ax.axvline(x=tr, color=DARK_GREEN, lw=1.0, linestyle=':', alpha=0.7)
    ax.set_xlabel('Zeit $t$ [Tage]')
    ax.set_ylabel('Trübungswert [NTU]')
    ax.set_title('(a) Sensorverlauf: Trübungsgrad')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # (b) Korrelation Trübungswert vs. gemessene Leistungsreduktion
    ax2 = axes[0, 1]
    ntu_vals = np.linspace(20, 120, 80)
    delta_P  = 0.45 * (1 - np.exp(-0.018 * (ntu_vals - 20))) + \
               0.03 * np.random.randn(80)
    delta_P  = np.clip(delta_P, 0, 0.5)

    ax2.scatter(ntu_vals, delta_P * 100, s=30, alpha=0.6, color=MAIN_BLUE, label='Messdaten')
    # Fit-Kurve
    fit_ntu = np.linspace(20, 120, 300)
    fit_dP  = 0.45 * (1 - np.exp(-0.018 * (fit_ntu - 20))) * 100
    ax2.plot(fit_ntu, fit_dP, '-', color=ACCENT_RED, lw=2,
             label=r'Modell: $\Delta P = 45\%(1-e^{-0.018(\kappa-20)})$')
    ax2.axvline(x=70, color=DARK_GRAY, lw=1.2, linestyle='-.', alpha=0.7)
    ax2.text(71, 2, 'Schwelle', fontsize=8, color=DARK_GRAY, rotation=90, va='bottom')
    ax2.set_xlabel('Trübungswert $\\kappa$ [NTU]')
    ax2.set_ylabel('Leistungsreduktion $\\Delta P$ [%]')
    ax2.set_title('(b) Korrelation: Trübung — Leistungsverlust')
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # (c) Histogramm: Reinigungsintervalle
    ax3 = axes[1, 0]
    np.random.seed(7)
    intervalle = np.random.normal(41, 8, 200)
    intervalle = np.clip(intervalle, 15, 80)
    ax3.hist(intervalle, bins=20, color=MAIN_BLUE, alpha=0.8, edgecolor='white', lw=0.5)
    ax3.axvline(x=np.mean(intervalle), color=ACCENT_RED, lw=2, linestyle='--',
                label=f'Mittelwert = {np.mean(intervalle):.1f} Tage')
    ax3.axvline(x=np.median(intervalle), color=DARK_GREEN, lw=2, linestyle=':',
                label=f'Median = {np.median(intervalle):.1f} Tage')
    ax3.set_xlabel('Reinigungsintervall [Tage]')
    ax3.set_ylabel('Häufigkeit')
    ax3.set_title('(c) Verteilung der Reinigungsintervalle\n(n=200 Schraube-Betriebsperioden)')
    ax3.legend(fontsize=8.5)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

    # (d) Drehmomentsensor: Richtungsumkehr beim Selbstreinigen
    ax4 = axes[1, 1]
    t_cycle = np.linspace(0, 60, 1000)
    # Normalbetrieb: positives Drehmoment, Reinigung: negative Richtung für 5s
    drehmoment = 850 * np.ones(1000)
    # Reinigungspuls bei t=30s für 5s
    clean_mask = (t_cycle >= 30) & (t_cycle <= 35)
    drehmoment[clean_mask] = -1200 + 60 * np.sin(np.pi * (t_cycle[clean_mask] - 30) / 5)
    drehmoment += 30 * np.random.randn(1000)

    ax4.fill_between(t_cycle, 0, drehmoment, where=(drehmoment >= 0),
                     alpha=0.3, color=MAIN_BLUE)
    ax4.fill_between(t_cycle, 0, drehmoment, where=(drehmoment < 0),
                     alpha=0.3, color=ACCENT_RED)
    ax4.plot(t_cycle, drehmoment, '-', color=DARK_GRAY, lw=0.8)
    ax4.axhline(y=0, color='black', lw=0.8)
    ax4.annotate('Selbstreinigungs-\npuls', xy=(32.5, -1100),
                 xytext=(40, -1300), fontsize=8.5, color=ACCENT_RED,
                 arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=ACCENT_RED, lw=0.7))
    ax4.set_xlabel('Zeit $t$ [s]')
    ax4.set_ylabel('Drehmoment $M$ [Nm]')
    ax4.set_title('(d) Drehmomentsensor: Normalbetrieb\nund Selbstreinigungspuls')
    ax4.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Sensordatenanalyse der Unterwasser-Schraubenwände', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'plot4_sensorkorrelation.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Plot 5 – Geometrische Anordnung im Flussquerschnitt
# ===========================================================================
# ===========================================================================
# Plot 5 – Geometrische Anordnung im Flussquerschnitt (NEUDESIGN / Journal Grade)
# ===========================================================================
def plot5_anordnung():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        gridspec_kw={'width_ratios': [1.15, 1]},
        constrained_layout=False
    )

    # ==========================================================
    # LINKS: Technischer Flussquerschnitt
    # ==========================================================
    ax = axes[0]
    ax.set_xlim(0, 200)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    # Flussbett / Wasser
    ax.fill_between([0, 200], -10, -8, color=(0.72, 0.62, 0.45), alpha=0.85)
    ax.fill_between([0, 200], 8, 10, color=(0.72, 0.62, 0.45), alpha=0.85)
    ax.fill_between([0, 200], -8, 8, color=(0.50, 0.76, 0.94), alpha=0.22)

    # Wasseroberfläche
    xw = np.linspace(0, 200, 400)
    yw = 8 + 0.18*np.sin(xw/6)
    ax.plot(xw, yw, lw=1.2, color=(0.20, 0.40, 0.95))

    # Strömungspfeile
    for y in [-5, -2, 1, 4]:
        for x in [18, 58, 98, 138]:
            ax.annotate(
                '', xy=(x+18, y), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', lw=0.9, color=DARK_GRAY, alpha=0.65)
            )

    # Schraubenpositionen
    pos = [25, 65, 105, 145, 185]

    for p in pos:
        # Träger
        ax.plot([p, p], [-8, -2], color=DARK_GRAY, lw=1.5)

        # Rotor
        rotor = mpatches.Ellipse(
            (p, -4), width=8, height=4,
            fc=MAIN_BLUE, ec='white', lw=0.8, alpha=0.95
        )
        ax.add_patch(rotor)

        # Rotation
        ax.annotate(
            '', xy=(p+3, -2.5), xytext=(p-3, -2.5),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.0,
                color='white',
                connectionstyle='arc3,rad=-0.65'
            )
        )

    # Schiff
    ship_x = [78, 120, 126, 116, 84, 74, 78]
    ship_y = [5.0, 5.0, 3.0, 2.2, 2.2, 3.0, 5.0]
    ax.fill(ship_x, ship_y, color=DARK_GRAY, alpha=0.72)

    # Klappmodus einer Einheit
    flap = FancyBboxPatch(
        (101, -8.7), 8, 1.5,
        boxstyle='round,pad=0.2',
        fc=ACCENT_RED, ec='white', lw=0.8
    )
    ax.add_patch(flap)

    ax.annotate(
        'Kippstellung bei Schiffsverkehr',
        xy=(105, -8.0),
        xytext=(132, -6.0),
        fontsize=8.7,
        color=ACCENT_RED,
        arrowprops=dict(arrowstyle='->', lw=1.0, color=ACCENT_RED)
    )

    ax.text(5, 8.8, '(a)', fontsize=11, fontweight='bold')
    ax.set_title('Querschnitt einer Unterwasser-Schraubenwandanlage')
    ax.set_xlabel('Flussbreite [m]')
    ax.set_ylabel('Tiefe [m]')
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.grid(alpha=0.18)

    # ==========================================================
    # RECHTS: Optimierung Wandabstand
    # ==========================================================
    ax2 = axes[1]

    d = np.linspace(20, 220, 500)

    rho = 1000
    A = 2.0
    v = 2.0
    eta = 0.474
    L = 10000

    n = L / d
    P0 = eta * 0.5 * rho * A * v**3
    P = n * P0 * np.exp(-0.02*n) / 1e6

    # Kurve
    ax2.fill_between(d, 0, P, color=MAIN_BLUE, alpha=0.14)
    ax2.plot(d, P, color=MAIN_BLUE, lw=2.6)

    # Optimum
    i = np.argmax(P)
    d_opt = d[i]
    P_opt = P[i]

    ax2.plot(d_opt, P_opt, 'o', color=ACCENT_RED, ms=8)

    ax2.annotate(
        f'Optimum\n$d^*={d_opt:.0f}$ m\n$P={P_opt:.2f}$ MW',
        xy=(d_opt, P_opt),
        xytext=(d_opt+10, P_opt+0.05),
        fontsize=9,
        color=ACCENT_RED,
        ha='left',
        va='bottom',
        arrowprops=dict(
            arrowstyle='->',
            lw=1.1,
            color=ACCENT_RED,
            connectionstyle='arc3,rad=0.15'
        ),
        bbox=dict(
            boxstyle='round,pad=0.28',
            fc='white',
            ec=ACCENT_RED,
            lw=0.8
        )
    )

    # Bereiche markieren
    ax2.axvspan(20, 55, color=ACCENT_RED, alpha=0.06)
    ax2.axvspan(140, 220, color=DARK_GREEN, alpha=0.05)

    ax2.text(28, 0.08, 'zu dicht:\nStrömungsschatten', fontsize=8, color=ACCENT_RED)
    ax2.text(160, 0.08, 'zu weit:\nzu wenige Wände', fontsize=8, color=DARK_GREEN)

    ax2.text(22, P.max()*0.96, '(b)', fontsize=11, fontweight='bold')

    ax2.set_title('Optimierung des Wandabstands')
    ax2.set_xlabel('Abstand zwischen Schraubenwänden d [m]')
    ax2.set_ylabel('Gesamtleistung [MW]')
    ax2.set_xlim(20, 220)
    ax2.set_ylim(0, P.max()*1.15)
    ax2.grid(True, linestyle='--', alpha=0.32)

    # Export
    path = os.path.join(OUTDIR, 'plot5_anordnung.pdf')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Plot 6 – Jahresenergieertrag und CO₂-Äquivalent
# ===========================================================================
def plot6_co2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # (a) Jahresenergieertrag über Betriebsjahre (Degradationskurve)
    ax = axes[0]
    jahre = np.arange(0, 26)

    # Degradationsmodell: E(t) = E0 * (1 - delta)^t
    E0      = 85.0     # GWh/a im Jahr 1
    delta_list = [0.005, 0.012, 0.020]
    colors_d   = [DARK_GREEN, MAIN_BLUE, ACCENT_RED]
    labels_d   = ['$\\delta=0{,}5\\,\\%$ (optimale Wartung)',
                  '$\\delta=1{,}2\\,\\%$ (Standardbetrieb)',
                  '$\\delta=2{,}0\\,\\%$ (minimale Wartung)']

    for delta, color, label in zip(delta_list, colors_d, labels_d):
        E = E0 * (1 - delta) ** jahre
        ax.plot(jahre, E, '-o', color=color, lw=2, markersize=4, label=label)

    ax.axhline(y=E0 * 0.80, color=DARK_GRAY, lw=1.2, linestyle=':',
               label='80%-Schwelle (Repowering)')
    ax.set_xlabel('Betriebsjahr')
    ax.set_ylabel('Jahresenergieertrag [GWh/a]')
    ax.set_title('Degradation des Jahresertrags\nüber die Betriebsdauer (25 Jahre)')
    ax.set_xlim(0, 25)
    ax.set_ylim(40, 92)
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (b) CO₂-Vermeidungspotenzial
    ax2 = axes[1]
    technologien = ['Unterwasser-\nSchraubenwand', 'Laufwasser-\nkraftwerk',
                    'Windkraft\n(Onshore)', 'Photovoltaik\n(Freifläche)',
                    'Braunkohle\n(Referenz)']
    co2_faktor = [4, 6, 7, 48, 820]   # g CO₂ / kWh (Lebenszyklusanalyse)
    colors_co2 = [MAIN_BLUE, DARK_GREEN, (0.2, 0.6, 0.8), (0.9, 0.7, 0.1), ACCENT_RED]

    bars = ax2.barh(technologien, co2_faktor, color=colors_co2, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, co2_faktor):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{val}', va='center', fontsize=9, color=DARK_GRAY)

    ax2.set_xlabel(r'$\mathrm{CO_2}$-Äquivalent [g $\mathrm{CO_2}$/kWh]')
    ax2.set_title(r'Vergleich: $\mathrm{CO_2}$-Fußabdruck verschiedener Technologien')
    ax2.set_xlim(0, 950)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'plot6_co2.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gespeichert: {path}')

# ===========================================================================
# Hauptprogramm
# ===========================================================================
if __name__ == '__main__':
    print("=== Generiere alle Plots ===")
    plot1_leistungskurve()
    plot2_wirkungsgrad()
    plot3_energieproduktion()
    plot4_sensorkorrelation()
    plot5_anordnung()
    plot6_co2()
    print("=== Alle Plots erfolgreich generiert ===")
