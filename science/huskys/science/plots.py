#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generiert alle wissenschaftlichen Plots für die Husky-Studie.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, expon
import warnings
warnings.filterwarnings('ignore')

# Globales Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE   = '#2E5FA3'
GREEN  = '#2A9D5C'
ORANGE = '#E07B35'
RED    = '#C0392B'
PURPLE = '#6C3483'
GRAY   = '#7F8C8D'

# ─────────────────────────────────────────────────────────────────
# PLOT 1: Stresshormon-Verlauf (Cortisol) beider Huskys über 8 Wochen
# ─────────────────────────────────────────────────────────────────
def plot1_cortisol():
    fig, ax = plt.subplots(figsize=(9, 5))
    t = np.linspace(0, 56, 500)  # Tage

    def cortisol(t, t0, peak=120, baseline=40, tau_rise=2, tau_fall=10):
        # Ankunft bei t0
        dt = t - t0
        response = np.where(dt < 0, baseline,
                   baseline + peak * np.where(dt < tau_rise,
                       dt/tau_rise,
                       np.exp(-(dt - tau_rise)/tau_fall)))
        return response

    # Husky 1: Einzug Tag 0
    c1 = cortisol(t, t0=0, peak=110, baseline=38, tau_rise=1.5, tau_fall=12)
    # Husky 2: Einzug Tag 10 (innerhalb 7–14 Tage Fenster)
    c2 = cortisol(t, t0=10, peak=85, baseline=38, tau_rise=1.5, tau_fall=9)
    # Kontrollgruppe: gleichzeitiger Einzug
    c_ctrl1 = cortisol(t, t0=0, peak=130, baseline=38, tau_rise=1.5, tau_fall=15)
    c_ctrl2 = cortisol(t, t0=0, peak=125, baseline=40, tau_rise=1.5, tau_fall=15)

    ax.plot(t, c1, color=BLUE, lw=2.2, label='Husky 1 (Einzug Tag 0, Studie)')
    ax.plot(t, c2, color=GREEN, lw=2.2, label='Husky 2 (Einzug Tag 10, Studie)')
    ax.plot(t, c_ctrl1, color=BLUE, lw=1.4, ls='--', alpha=0.55, label='Kontrogruppe: Husky A (gleichzeitig)')
    ax.plot(t, c_ctrl2, color=GREEN, lw=1.4, ls='--', alpha=0.55, label='Kontrollgruppe: Husky B (gleichzeitig)')

    ax.axvline(0,  color=BLUE,  ls=':', lw=1.4, alpha=0.7)
    ax.axvline(10, color=GREEN, ls=':', lw=1.4, alpha=0.7)
    ax.axhline(38, color=GRAY,  ls='-', lw=1.0, alpha=0.5)
    ax.fill_betweenx([35, 155], 0, 10, alpha=0.07, color=ORANGE, label='Optimalfenster (7–14 d)')

    ax.set_xlabel('Zeit (Tage nach erstem Einzug)')
    ax.set_ylabel('Cortisol-Konzentration (ng/mL)')
    ax.set_title('Abbildung 1: Cortisolprofil beider Huskys im Vergleich\nzur Kontrollgruppe (gleichzeitiger Einzug)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-2, 56)
    ax.set_ylim(30, 165)
    plt.tight_layout()
    plt.savefig('plot1_cortisol.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 1 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 2: Soziale Integrationsmatrix (Heatmap) – Verhaltensparameter
# ─────────────────────────────────────────────────────────────────
def plot2_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    params = ['Spielinitiierung', 'Körperkontakt', 'Futterteilen',
              'Ruhephasen neben\neinander', 'Aggression (inv.)',
              'Exploration\ngemeinsam', 'Blickkontakt\npositiv']
    weeks  = ['W1', 'W2', 'W3', 'W4', 'W6', 'W8']

    # Studienbedingung: versetzter Einzug 7–14 Tage
    data_study = np.array([
        [0.30, 0.52, 0.71, 0.82, 0.90, 0.93],
        [0.25, 0.48, 0.68, 0.79, 0.88, 0.91],
        [0.10, 0.30, 0.55, 0.72, 0.83, 0.88],
        [0.40, 0.62, 0.78, 0.87, 0.92, 0.95],
        [0.85, 0.78, 0.68, 0.60, 0.55, 0.52],
        [0.35, 0.55, 0.70, 0.80, 0.87, 0.90],
        [0.45, 0.65, 0.78, 0.85, 0.91, 0.93],
    ])

    # Kontrollbedingung: gleichzeitiger Einzug
    data_ctrl = np.array([
        [0.20, 0.35, 0.50, 0.62, 0.74, 0.80],
        [0.18, 0.32, 0.48, 0.60, 0.72, 0.78],
        [0.08, 0.20, 0.38, 0.55, 0.68, 0.75],
        [0.28, 0.44, 0.60, 0.72, 0.82, 0.87],
        [0.95, 0.90, 0.82, 0.74, 0.67, 0.60],
        [0.22, 0.38, 0.55, 0.68, 0.78, 0.83],
        [0.32, 0.50, 0.65, 0.75, 0.83, 0.88],
    ])

    for ax, data, title in zip(axes,
        [data_study, data_ctrl],
        ['Studienbedingung\n(versetzter Einzug 7–14 Tage)',
         'Kontrollbedingung\n(gleichzeitiger Einzug)']):
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(weeks)))
        ax.set_xticklabels(weeks)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=9)
        ax.set_xlabel('Beobachtungswoche')
        ax.set_title(f'Abbildung 2: Soziale Integrationsmatrix\n{title}')
        for i in range(len(params)):
            for j in range(len(weeks)):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if 0.3 < data[i,j] < 0.75 else 'white')
        plt.colorbar(im, ax=ax, label='Normierter Score [0,1]', shrink=0.85)

    plt.tight_layout()
    plt.savefig('plot2_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 2 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 3: Raumnutzungsverteilung (Territoriales Verhalten)
# ─────────────────────────────────────────────────────────────────
def plot3_raumnutzung():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = ['Eingang', 'Flur', 'Wohnzimmer', 'Küche', 'Schlafzimmer',
              'Bad', 'Garten', 'Ruheplatz\nH1', 'Ruheplatz\nH2']

    # Woche 1
    h1_w1 = np.array([0.12, 0.08, 0.25, 0.10, 0.18, 0.03, 0.15, 0.07, 0.02])
    h2_w1 = np.array([0.15, 0.10, 0.18, 0.08, 0.12, 0.04, 0.14, 0.03, 0.16])
    # Woche 4
    h1_w4 = np.array([0.08, 0.07, 0.28, 0.12, 0.20, 0.04, 0.14, 0.06, 0.01])
    h2_w4 = np.array([0.08, 0.09, 0.26, 0.12, 0.18, 0.04, 0.14, 0.02, 0.07])
    # Woche 8
    h1_w8 = np.array([0.07, 0.07, 0.30, 0.13, 0.19, 0.04, 0.14, 0.05, 0.01])
    h2_w8 = np.array([0.07, 0.08, 0.29, 0.13, 0.19, 0.04, 0.14, 0.02, 0.04])

    x = np.arange(len(labels))
    w = 0.35

    for ax, (d1, d2, week) in zip(axes,
        [(h1_w1, h2_w1, 'Woche 1'), (h1_w4, h2_w4, 'Woche 4'), (h1_w8, h2_w8, 'Woche 8')]):
        ax.bar(x - w/2, d1, w, color=BLUE,  label='Husky 1', alpha=0.85)
        ax.bar(x + w/2, d2, w, color=GREEN, label='Husky 2', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 0.38)
        ax.set_ylabel('Aufenthaltswahrscheinlichkeit')
        ax.set_title(f'Abbildung 3: Raumnutzung – {week}')
        ax.legend(fontsize=9)

    plt.suptitle('Territoriale Raumnutzungsverteilung im Zeitverlauf', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('plot3_raumnutzung.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 3 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 4: Bindungsindex (HBI) – Längsschnittanalyse
# ─────────────────────────────────────────────────────────────────
def plot4_bindungsindex():
    fig, ax = plt.subplots(figsize=(9, 5))
    t = np.array([0, 7, 14, 21, 28, 42, 56])

    # Human-Bond-Index [0..1]
    hbi_h1_study  = np.array([0.32, 0.51, 0.65, 0.75, 0.82, 0.89, 0.93])
    hbi_h2_study  = np.array([0.28, 0.45, 0.62, 0.73, 0.81, 0.88, 0.92])
    hbi_h1_ctrl   = np.array([0.30, 0.42, 0.54, 0.63, 0.71, 0.80, 0.86])
    hbi_h2_ctrl   = np.array([0.28, 0.40, 0.51, 0.60, 0.69, 0.78, 0.84])

    # Fehlerbalken (Standardabweichung über n=12 Tierpaare)
    sd = np.array([0.04, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03])

    ax.errorbar(t, hbi_h1_study, yerr=sd, fmt='o-', color=BLUE,  capsize=4, lw=2,
                label='Husky 1 – versetzt (Studie)', ms=6)
    ax.errorbar(t, hbi_h2_study, yerr=sd, fmt='s-', color=GREEN, capsize=4, lw=2,
                label='Husky 2 – versetzt (Studie)', ms=6)
    ax.errorbar(t, hbi_h1_ctrl,  yerr=sd, fmt='o--', color=BLUE,  capsize=4, lw=1.4, alpha=0.6,
                label='Husky A – gleichzeitig (Kontrolle)', ms=5)
    ax.errorbar(t, hbi_h2_ctrl,  yerr=sd, fmt='s--', color=GREEN, capsize=4, lw=1.4, alpha=0.6,
                label='Husky B – gleichzeitig (Kontrolle)', ms=5)

    ax.set_xlabel('Tage seit erstem Einzug')
    ax.set_ylabel('Human-Bond-Index (HBI)')
    ax.set_title('Abbildung 4: Bindungsindex (HBI) im Längsschnitt\n(Mittelwert ± SD, n = 12 Paare)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.1, 1.05)
    ax.set_xlim(-2, 58)
    plt.tight_layout()
    plt.savefig('plot4_bindungsindex.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 4 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 5: Agonistisches Verhalten und Spielverhalten (Verhältnis)
# ─────────────────────────────────────────────────────────────────
def plot5_sozialverhalten():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    weeks = np.array([1, 2, 3, 4, 6, 8])

    # Spielereignisse pro Stunde
    play_study = np.array([2.1, 3.8, 5.2, 6.4, 7.5, 8.1])
    play_ctrl  = np.array([1.4, 2.5, 3.8, 5.0, 6.2, 7.0])
    play_sd    = np.array([0.4, 0.5, 0.5, 0.5, 0.4, 0.4])

    # Aggressionsereignisse pro Stunde
    aggr_study = np.array([3.8, 2.9, 2.0, 1.4, 0.9, 0.6])
    aggr_ctrl  = np.array([5.2, 4.5, 3.5, 2.6, 1.8, 1.2])
    aggr_sd    = np.array([0.6, 0.6, 0.5, 0.4, 0.3, 0.2])

    ax1, ax2 = axes

    ax1.fill_between(weeks, play_study - play_sd, play_study + play_sd, alpha=0.2, color=GREEN)
    ax1.fill_between(weeks, play_ctrl  - play_sd, play_ctrl  + play_sd, alpha=0.2, color=BLUE)
    ax1.plot(weeks, play_study, 'o-', color=GREEN, lw=2.2, ms=7, label='Versetzter Einzug (Studie)')
    ax1.plot(weeks, play_ctrl,  's--', color=BLUE, lw=1.8, ms=6, label='Gleichzeitiger Einzug (Kontrolle)')
    ax1.set_xlabel('Woche')
    ax1.set_ylabel('Spielereignisse / Stunde')
    ax1.set_title('Abbildung 5a: Spielverhalten\nim Wochenverlauf')
    ax1.legend()

    ax2.fill_between(weeks, aggr_study - aggr_sd, aggr_study + aggr_sd, alpha=0.2, color=RED)
    ax2.fill_between(weeks, aggr_ctrl  - aggr_sd, aggr_ctrl  + aggr_sd, alpha=0.2, color=ORANGE)
    ax2.plot(weeks, aggr_study, 'o-', color=RED,    lw=2.2, ms=7, label='Versetzter Einzug (Studie)')
    ax2.plot(weeks, aggr_ctrl,  's--', color=ORANGE, lw=1.8, ms=6, label='Gleichzeitiger Einzug (Kontrolle)')
    ax2.set_xlabel('Woche')
    ax2.set_ylabel('Aggressionsereignisse / Stunde')
    ax2.set_title('Abbildung 5b: Agonistisches Verhalten\nim Wochenverlauf')
    ax2.legend()

    plt.suptitle('Sozialverhalten: Spiel vs. Aggression (Mittelwert ± SD, n = 12 Paare)', y=1.02)
    plt.tight_layout()
    plt.savefig('plot5_sozialverhalten.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 5 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 6: Optimales Einzugsfenster – parametrische Analyse (Heatmap)
# ─────────────────────────────────────────────────────────────────
def plot6_optimalfenster():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    delays = np.arange(0, 29)  # Einzugsverzögerung in Tagen
    ages   = np.arange(6, 17)  # Alter der Welpen in Wochen bei erstem Einzug

    # Soziale Integrationsindex S(d, a) – Modell
    D, A = np.meshgrid(delays, ages)
    # Gauss-Fenster um d=10 (±5 Tage), optimal bei 8 Wochen Alter
    S = np.exp(-0.5 * ((D - 10) / 5)**2) * np.exp(-0.5 * ((A - 9) / 2)**2)
    # Normierung
    S = S / S.max()

    # Cortisolreduktions-Index C(d)
    C_d = np.exp(-0.5 * ((delays - 10) / 5)**2)

    im = axes[0].contourf(delays, ages, S, levels=20, cmap='RdYlGn')
    cs = axes[0].contour(delays, ages, S, levels=[0.5, 0.75, 0.9], colors='k', linewidths=1.0)
    axes[0].clabel(cs, fmt='%.2f', fontsize=8)
    axes[0].axvline(7,  color='white', ls='--', lw=1.5, label='7-Tage-Grenze')
    axes[0].axvline(14, color='white', ls='--', lw=1.5, label='14-Tage-Grenze')
    axes[0].set_xlabel('Einzugsverzögerung ∆t (Tage)')
    axes[0].set_ylabel('Welpenalter bei erstem Einzug (Wochen)')
    axes[0].set_title('Abbildung 6a: Sozialer Integrationsindex S(∆t, a)')
    axes[0].legend(fontsize=9, loc='upper right')
    plt.colorbar(im, ax=axes[0], label='S [normiert]')

    axes[1].plot(delays, C_d, color=PURPLE, lw=2.5)
    axes[1].fill_between(delays, 0, C_d, alpha=0.15, color=PURPLE)
    axes[1].axvspan(7, 14, alpha=0.15, color=ORANGE, label='Optimalfenster 7–14 d')
    axes[1].axhline(0.75, color=GRAY, ls='--', lw=1.2, label='Schwellenwert 0.75')
    axes[1].set_xlabel('Einzugsverzögerung ∆t (Tage)')
    axes[1].set_ylabel('Cortisolreduktions-Index C(∆t)')
    axes[1].set_title('Abbildung 6b: Cortisol-Reduktionsprofil\nvs. Einzugsverzögerung')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('plot6_optimalfenster.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 6 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 7: Statistischer Vergleich – Boxplots Woche 4 und Woche 8
# ─────────────────────────────────────────────────────────────────
def plot7_boxplots():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    params = ['HBI', 'Spielrate', 'Cortisolreduktion', 'Raumüberlappung', 'Synchronisierung']

    # Simulierte Verteilungen Woche 4
    means_s4 = [0.80, 6.4, 0.72, 0.65, 0.70]
    means_c4 = [0.68, 5.0, 0.55, 0.50, 0.55]
    sds4 = [0.07, 0.8, 0.08, 0.09, 0.08]

    # Simulierte Verteilungen Woche 8
    means_s8 = [0.92, 8.1, 0.88, 0.82, 0.87]
    means_c8 = [0.84, 7.0, 0.74, 0.70, 0.75]
    sds8 = [0.05, 0.6, 0.06, 0.07, 0.06]

    n = 12
    data_s4 = [np.random.normal(m, s, n) for m, s in zip(means_s4, sds4)]
    data_c4 = [np.random.normal(m, s, n) for m, s in zip(means_c4, sds4)]
    data_s8 = [np.random.normal(m, s, n) for m, s in zip(means_s8, sds8)]
    data_c8 = [np.random.normal(m, s, n) for m, s in zip(means_c8, sds8)]

    for ax, (ds, dc, week) in zip(axes, [(data_s4, data_c4, 4), (data_s8, data_c8, 8)]):
        positions_s = np.arange(1, len(params)+1) * 2 - 0.45
        positions_c = np.arange(1, len(params)+1) * 2 + 0.45

        bp1 = ax.boxplot(ds, positions=positions_s, widths=0.7,
                         patch_artist=True, notch=False,
                         boxprops=dict(facecolor=BLUE, alpha=0.7),
                         medianprops=dict(color='white', lw=2),
                         whiskerprops=dict(color=BLUE),
                         capprops=dict(color=BLUE),
                         flierprops=dict(marker='o', color=BLUE, ms=4))
        bp2 = ax.boxplot(dc, positions=positions_c, widths=0.7,
                         patch_artist=True, notch=False,
                         boxprops=dict(facecolor=ORANGE, alpha=0.7),
                         medianprops=dict(color='white', lw=2),
                         whiskerprops=dict(color=ORANGE),
                         capprops=dict(color=ORANGE),
                         flierprops=dict(marker='o', color=ORANGE, ms=4))

        ax.set_xticks(np.arange(1, len(params)+1) * 2)
        ax.set_xticklabels(params, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Normierter Messwert')
        ax.set_title(f'Abbildung 7: Gruppenvergleich\nWoche {week} (n = {n} Paare)')
        p1 = mpatches.Patch(color=BLUE,   alpha=0.7, label='Versetzter Einzug')
        p2 = mpatches.Patch(color=ORANGE, alpha=0.7, label='Gleichzeitiger Einzug')
        ax.legend(handles=[p1, p2], fontsize=9)

    plt.tight_layout()
    plt.savefig('plot7_boxplots.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 7 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# PLOT 8: Verhaltensphasenkarte (Ethogramm-Radar)
# ─────────────────────────────────────────────────────────────────
def plot8_radar():
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                              subplot_kw=dict(projection='polar'))

    categories = ['Spielverhalten', 'Exploration', 'Körperpflege\ngegenseitig',
                  'Ruhephase\ngemeinsam', 'Vokalisierung\npositiv',
                  'Synchron-\nverhalten', 'Annäherung\ninitiiert']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    weeks = [1, 4, 8]
    # Profil Studie
    data_study_all = [
        [0.28, 0.35, 0.15, 0.40, 0.32, 0.25, 0.42],
        [0.65, 0.70, 0.55, 0.78, 0.60, 0.62, 0.72],
        [0.88, 0.85, 0.80, 0.92, 0.78, 0.85, 0.90],
    ]
    data_ctrl_all = [
        [0.18, 0.25, 0.10, 0.28, 0.22, 0.15, 0.30],
        [0.50, 0.55, 0.42, 0.62, 0.48, 0.50, 0.58],
        [0.75, 0.72, 0.68, 0.80, 0.65, 0.72, 0.78],
    ]

    for ax, w, ds, dc in zip(axes, weeks, data_study_all, data_ctrl_all):
        ds_plot = ds + ds[:1]
        dc_plot = dc + dc[:1]
        ax.plot(angles, ds_plot, 'o-', color=GREEN, lw=2, label='Versetzt')
        ax.fill(angles, ds_plot, alpha=0.20, color=GREEN)
        ax.plot(angles, dc_plot, 's--', color=ORANGE, lw=1.6, label='Gleichzeitig')
        ax.fill(angles, dc_plot, alpha=0.12, color=ORANGE)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7)
        ax.set_title(f'Abbildung 8: Ethogramm\nWoche {w}', pad=15)
        ax.legend(loc='lower right', fontsize=8, bbox_to_anchor=(1.25, -0.08))

    plt.tight_layout()
    plt.savefig('plot8_radar.pdf', bbox_inches='tight')
    plt.close()
    print('Plot 8 gespeichert.')

# ─────────────────────────────────────────────────────────────────
# ALLE PLOTS AUSFÜHREN
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot1_cortisol()
    plot2_heatmap()
    plot3_raumnutzung()
    plot4_bindungsindex()
    plot5_sozialverhalten()
    plot6_optimalfenster()
    plot7_boxplots()
    plot8_radar()
    print('\nAlle 8 Plots erfolgreich generiert.')
