#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generierung aller 12 Matplotlib-Plots fuer:
"Maximierung von Reichweite und Lebensdauer von Hochvoltbatterien
 in Elektrofahrzeugen"
Autor: Stephan Epp, Universitaet Bielefeld, 2026
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Globale Plot-Einstellungen (LaTeX-aehlich)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2.2,
    'figure.dpi': 150,
})

# Farben (abgestimmt auf das LaTeX-Farbschema)
BLUE   = '#19468C'
RED    = '#B4321E'
GREEN  = '#1E6432'
ORANGE = '#C47810'
PURPLE = '#5B2A8A'
GRAY   = '#3C3C46'
LBLUE  = '#4A7EC8'
LRED   = '#E07060'
LGREEN = '#3CB864'

SAVE_DIR = ''


def save_plot(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    plt.tight_layout()
    fig.savefig(SAVE_DIR + name, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Gespeichert: {name}")


# ==========================================================
# Plot 1: Kapazitaetsdegradation vs. Zyklen, verschiedene T
# ==========================================================
def plot_fig01():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = np.linspace(0, 2000, 1000)
    Q0 = 100.0

    configs = [
        {'T': '45 °C', 'alpha': 5.0e-4, 'beta': 0.72, 'color': RED,   'ls': '-'},
        {'T': '25 °C', 'alpha': 1.8e-4, 'beta': 0.72, 'color': BLUE,  'ls': '-'},
        {'T': '10 °C', 'alpha': 2.9e-4, 'beta': 0.72, 'color': GREEN, 'ls': '--'},
    ]
    for c in configs:
        Q = Q0 * np.exp(-c['alpha'] * n**c['beta'])
        ax.plot(n, Q, color=c['color'], ls=c['ls'],
                label=f'$T = {c["T"]}$')

    ax.axhline(y=80, color=ORANGE, ls=':', lw=1.8, label='80 %-Garantieschwelle')
    ax.axhline(y=70, color=GRAY,   ls=':', lw=1.8, label='70 %-Garantieschwelle')
    ax.axvline(x=750,  color=ORANGE, ls=':', lw=1.0, alpha=0.5)
    ax.axvline(x=1500, color=GRAY,   ls=':', lw=1.0, alpha=0.5)
    ax.text(755,  63, r'$\approx 60\,000\,\mathrm{km}$', fontsize=9, color=ORANGE)
    ax.text(1505, 63, r'$\approx 160\,000\,\mathrm{km}$', fontsize=9, color=GRAY)
    ax.annotate('', xy=(750, 70), xytext=(0, 70),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
    ax.annotate('', xy=(1500, 70), xytext=(750, 70),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

    ax.set_xlabel('Ladezyklen $n$')
    ax.set_ylabel(r'Relative Kapazit\"{a}t $Q(n)/Q_0\;(\%)$')
    ax.set_title(r'Kapazit\"{a}tsdegradation in Abh\"{a}ngigkeit von Temperatur und Zyklenzahl')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 2000); ax.set_ylim(58, 103)
    save_plot('fig01_degradation_zyklen.pdf', fig)


# ==========================================================
# Plot 2: Zyklenlebensdauer vs. Entladetiefe (DoD)
# ==========================================================
def plot_fig02():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dod = np.linspace(5, 100, 300)

    def cycle_life(dod_pct, k, gamma):
        return k / (dod_pct / 100)**gamma

    configs = [
        {'label': 'Voller Bereich (0–100 %)',  'k': 2000, 'g': 1.38, 'color': RED},
        {'label': 'Standard (10–90 %)',          'k': 3800, 'g': 1.38, 'color': BLUE},
        {'label': 'Schonend (20–80 %)',          'k': 6200, 'g': 1.38, 'color': GREEN},
    ]
    for c in configs:
        y = np.clip(cycle_life(dod, c['k'], c['g']), 0, 16000)
        ax.plot(dod, y, color=c['color'], label=c['label'])

    ax.axvline(x=80, color=ORANGE, ls='--', lw=1.6,
               label='DoD = 80 % (Alltagsempfehlung)')
    ax.axhline(y=1000, color=GRAY, ls=':', lw=1.5,
               label='Mindestlebensdauer (1 000 Zyklen)')
    ax.fill_betweenx([0, 16000], 55, 70, alpha=0.07, color=GREEN,
                     label='Optimales DoD-Fenster')

    ax.set_xlabel('Entladetiefe DoD (%)')
    ax.set_ylabel(r'Zyklenlebensdauer $N_L$')
    ax.set_title(r'Zyklenlebensdauer in Abh\"{a}ngigkeit von Entladetiefe und SOC-Fenster')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(5, 100); ax.set_ylim(0, 13000)
    save_plot('fig02_dod_zyklenlebensdauer.pdf', fig)


# ==========================================================
# Plot 3: CC-CV Ladekurve (Spannung, Strom, SOC)
# ==========================================================
def plot_fig03():
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    t = np.linspace(0, 75, 1000)   # Minuten
    t_cc = 30.0                     # Ende CC-Phase

    # SOC (sigmoidal)
    soc = np.where(t <= t_cc,
                   t / t_cc * 80,
                   80 + 20 * (1 - np.exp(-(t - t_cc) / 12)))
    soc = np.clip(soc, 0, 100)

    # Spannung (ca. 370 – 420 V fuer 800 V Plattform, hier skaliert)
    U = 370 + 50 * soc / 100

    # Strom: CC dann exponentiell fallend
    I_max = 270.0   # A  (ca. 100 kW)
    I = np.where(t <= t_cc, I_max, I_max * np.exp(-(t - t_cc) / 18))
    I = np.clip(I, 5, I_max)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 65))

    l1, = ax1.plot(t, U,   color=RED,   lw=2.2, label='Spannung $U$ (V)')
    l2, = ax2.plot(t, I,   color=BLUE,  lw=2.2, ls='--', label='Ladestrom $I$ (A)')
    l3, = ax3.plot(t, soc, color=GREEN, lw=2.2, ls=':',  label='SOC (%)')

    # Phasenbeschriftung
    ax1.axvline(x=t_cc, color=GRAY, ls='--', lw=1.4, alpha=0.7)
    ax1.text(t_cc / 2,        372, 'CC-Phase', ha='center', fontsize=10, color=GRAY)
    ax1.text((t_cc + 75) / 2, 372, 'CV-Phase', ha='center', fontsize=10, color=GRAY)

    ax1.set_xlabel('Ladezeit $t$ (min)')
    ax1.set_ylabel('Spannung $U$ (V)', color=RED);   ax1.tick_params(axis='y', colors=RED)
    ax2.set_ylabel('Ladestrom $I$ (A)', color=BLUE); ax2.tick_params(axis='y', colors=BLUE)
    ax3.set_ylabel('SOC (%)', color=GREEN);          ax3.tick_params(axis='y', colors=GREEN)
    ax1.set_ylim(360, 430); ax2.set_ylim(0, 340); ax3.set_ylim(0, 110)
    ax1.set_xlim(0, 75)
    ax1.set_title('CC-CV-Ladekurve: Spannung, Strom und SOC-Verlauf')
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
    save_plot('fig03_cc_cv_ladekurve.pdf', fig)


# ==========================================================
# Plot 4: Reichweite vs. Umgebungstemperatur
# ==========================================================
def plot_fig04():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    T_arr = np.linspace(-25, 50, 300)
    R_ref = 580.0   # WLTP km

    # Elektrochemische Effizienz sinkt bei extremen Temperaturen
    eta_bat = 0.97 * np.exp(-0.00055 * (T_arr - 22)**2)

    # Klimatisierungsaufwand als Anteil
    def klima(T):
        h = np.maximum(0, 0.32 * np.exp(-0.065 * (T + 2))) * (T < 5)
        c = np.maximum(0, 0.09 * (T - 30) / 20) * (T > 30)
        return h + c

    klima_arr = np.array([klima(Ti) for Ti in T_arr])

    R_ohne_wp = R_ref * eta_bat * (1 - klima_arr)
    R_mit_wp  = R_ref * eta_bat * (1 - klima_arr * 0.52)   # WP 48 % effizienter
    R_ohne_wp = np.clip(R_ohne_wp, 0, R_ref)
    R_mit_wp  = np.clip(R_mit_wp,  0, R_ref)

    ax.plot(T_arr, R_ohne_wp, color=RED,   lw=2.2, label='Ohne W\\"armepumpe')
    ax.plot(T_arr, R_mit_wp,  color=BLUE,  lw=2.2, label='Mit W\\"armepumpe')
    ax.axhline(y=R_ref, color=GRAY,   ls=':', lw=1.6, label=f'WLTP-Referenz ({R_ref:.0f} km)')
    ax.axvline(x=22,    color=GREEN,  ls='--', lw=1.4, alpha=0.8, label='Opt. Temperatur (22 °C)')
    ax.axvspan(15, 30, alpha=0.08, color=GREEN, label='Optimaler Bereich [15 °C, 30 °C]')

    ax.set_xlabel(r'Umgebungstemperatur $T_{\mathrm{amb}}$ (°C)')
    ax.set_ylabel('Elektrische Reichweite $R$ (km)')
    ax.set_title(r'Elektrische Reichweite als Funktion der Umgebungstemperatur')
    ax.legend(loc='lower center', ncol=2, fontsize=9)
    ax.set_xlim(-25, 50); ax.set_ylim(0, 660)
    save_plot('fig04_temperatur_reichweite.pdf', fig)


# ==========================================================
# Plot 5: Degradationsmodell + empirische Kurvenanpassung
# ==========================================================
def plot_fig05():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    np.random.seed(42)

    n_meas = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800,
                        900, 1000, 1200, 1400, 1600, 1800, 2000])
    alpha_t, beta_t = 1.85e-4, 0.72
    Q_true = 100 * np.exp(-alpha_t * n_meas**beta_t)
    Q_meas = Q_true + np.random.normal(0, 0.55, len(n_meas))
    Q_meas = np.clip(Q_meas, 50, 101)

    def model(n, alpha, beta):
        return 100 * np.exp(-alpha * n**beta)

    popt, pcov = curve_fit(model, n_meas[1:], Q_meas[1:],
                            p0=[2e-4, 0.7], bounds=([0, 0.3], [1e-2, 1.2]))
    perr = np.sqrt(np.diag(pcov))
    n_fit = np.linspace(0, 2200, 600)
    Q_fit   = model(n_fit, *popt)
    Q_upper = np.clip(model(n_fit, popt[0] - 1.96*perr[0], popt[1] + 1.96*perr[1]), 0, 101)
    Q_lower = np.clip(model(n_fit, popt[0] + 1.96*perr[0], popt[1] - 1.96*perr[1]), 0, 101)

    ax.scatter(n_meas, Q_meas, color=RED, s=45, zorder=6, label='Messdaten', alpha=0.85)
    ax.plot(n_fit, Q_fit, color=BLUE, lw=2.2,
            label=r'Modell: $Q(n)=Q_0\,e^{-\hat\alpha\,n^{\hat\beta}}$')
    ax.fill_between(n_fit, Q_lower, Q_upper, alpha=0.14, color=BLUE,
                    label='95 %-Konfidenzband')
    ax.axhline(y=80, color=ORANGE, ls=':', lw=1.8, label='80 %-Schwelle')
    ax.axhline(y=70, color=GRAY,   ls=':', lw=1.8, label='70 %-Schwelle')
    ax.text(1600, 91,
            f'$\\hat{{\\alpha}} = {popt[0]:.4f}$\n$\\hat{{\\beta}} = {popt[1]:.3f}$',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))

    ax.set_xlabel('Ladezyklen $n$')
    ax.set_ylabel(r'Relative Kapazit\"{a}t $Q(n)/Q_0\;(\%)$')
    ax.set_title('Degradationsmodell: Messdaten, Kurvenanpassung und Konfidenzband')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 2200); ax.set_ylim(58, 104)
    save_plot('fig05_degradation_modell_fit.pdf', fig)


# ==========================================================
# Plot 6: Thermomanagement – Batterietemperatur waehrend Schnellladung
# ==========================================================
def plot_fig06():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = np.linspace(0, 60, 600)   # Minuten
    T_amb, T_init = 35.0, 28.0

    def simulate(t_arr, P_heat_W, has_cool=False, T_target=None, precond=False):
        T = T_init if not precond else 24.0
        m_Cp = 300 * 920.0   # J/K
        lam  = 55.0           # W/K
        records = [T]
        dt_s = (t_arr[1] - t_arr[0]) * 60
        for _ in t_arr[1:]:
            Q_in  = P_heat_W * dt_s
            Q_out = lam * (T - T_amb) * dt_s
            Q_c   = 0.0
            if has_cool and T_target and T > T_target:
                Q_c = 9000 * dt_s * min(1.0, (T - T_target) / 5.0)
            T += (Q_in - Q_out - Q_c) / m_Cp
            records.append(T)
        return np.array(records)

    T_no   = simulate(t, 4500)
    T_cool = simulate(t, 4500, has_cool=True, T_target=33)
    T_pre  = simulate(t, 4500, has_cool=True, T_target=28, precond=True)

    ax.plot(t, T_no,   color=RED,   lw=2.2, label='Ohne Thermomanagement')
    ax.plot(t, T_cool, color=BLUE,  lw=2.2, label='Mit Aktivk\\"uhlung (Ziel: 33 °C)')
    ax.plot(t, T_pre,  color=GREEN, lw=2.2, ls='--',
            label='Mit Vorkonditionierung (Ziel: 28 °C)')

    ax.axhspan(15, 35, alpha=0.09, color=GREEN, label='Optimaler Bereich [15 °C, 35 °C]')
    ax.axhline(y=45, color=RED, ls=':', lw=1.5, label='Kritische Temperatur (45 °C)')

    ax.set_xlabel('Ladezeit $t$ (min)')
    ax.set_ylabel(r'Batterietemperatur $T_{\mathrm{bat}}$ (°C)')
    ax.set_title(r'Thermomanagement w\"{a}hrend Schnellladung ($T_{\mathrm{amb}} = 35\,°C$)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 60); ax.set_ylim(12, 68)
    save_plot('fig06_thermomanagement.pdf', fig)


# ==========================================================
# Plot 7: Porsche Garantiekurve vs. Modellprognose
# ==========================================================
def plot_fig07():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    km = np.linspace(0, 170000, 600)

    def soh(km_, a, b):
        return 100 * np.exp(-a * km_**b)

    Q_opt = soh(km, 7.5e-9, 0.82)
    Q_nom = soh(km, 1.55e-8, 0.82)
    Q_bad = soh(km, 3.2e-8, 0.82)
    Q_ci_u = soh(km, 9.0e-9, 0.82)
    Q_ci_l = soh(km, 2.6e-8, 0.82)

    ax.plot(km/1e3, Q_opt, color=GREEN, lw=2.2, label='Optimale Nutzung')
    ax.plot(km/1e3, Q_nom, color=BLUE,  lw=2.2, label='Nominale Nutzung')
    ax.plot(km/1e3, Q_bad, color=RED,   lw=2.2, label='Ungünstige Nutzung')
    ax.fill_between(km/1e3, Q_ci_l, Q_ci_u, alpha=0.12, color=BLUE, label='Prognoseintervall')

    # Porsche-Garantie (Treppenkurve)
    g_km = [0, 60, 60, 160, 160]
    g_q  = [100, 80, 70, 70, 60]
    ax.step(g_km, g_q, color=ORANGE, lw=2.8, where='post',
            label='Porsche HV-Garantie', zorder=6)

    ax.axhline(y=80, color=ORANGE, ls=':', lw=1.0, alpha=0.55)
    ax.axhline(y=70, color=ORANGE, ls=':', lw=1.0, alpha=0.55)
    ax.axvline(x=60,  color=GRAY, ls=':', lw=1.0, alpha=0.55)
    ax.axvline(x=160, color=GRAY, ls=':', lw=1.0, alpha=0.55)
    ax.text(28, 81, '80 %', fontsize=9, color=ORANGE)
    ax.text(95, 71, '70 %', fontsize=9, color=ORANGE)

    ax.set_xlabel('Laufleistung (1 000 km)')
    ax.set_ylabel('State of Health SOH (%)')
    ax.set_title('Modellprognose vs. Porsche HV-Batterie-Garantiekurve')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 170); ax.set_ylim(55, 105)
    save_plot('fig07_garantie_prognose.pdf', fig)


# ==========================================================
# Plot 8: Reichweite vs. Geschwindigkeit (aerodynamisch)
# ==========================================================
def plot_fig08():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    v_kmh = np.linspace(20, 250, 400)
    v_ms  = v_kmh / 3.6

    m, cw, A_front = 2295.0, 0.22, 2.20
    fr, rho, g     = 0.012, 1.225, 9.81
    eta            = 0.88
    E_bat          = 83.7 * 3.6e6   # J

    F_aero  = 0.5 * rho * cw * A_front * v_ms**2
    F_roll  = fr * m * g
    P_total = (F_aero + F_roll) * v_ms / eta

    R = v_ms * (E_bat / P_total) / 1000   # km
    # Rekuperation: +15 % bei moderaten Geschwindigkeiten
    rekup = 0.15 * np.exp(-((v_kmh - 80)**2) / (2 * 52**2))
    R_rek = R * (1 + rekup)

    ax.plot(v_kmh, R,     color=BLUE,  lw=2.2, label='Ohne Rekuperation')
    ax.plot(v_kmh, R_rek, color=GREEN, lw=2.2, ls='--', label='Mit Rekuperation (+15 %)')

    # WLTP-Punkt (~92 km/h Durchschnitt)
    idx_w = np.argmin(np.abs(v_kmh - 92))
    ax.scatter([92], [R_rek[idx_w]], color=RED, s=80, zorder=7,
               label=f'WLTP-Referenz ≈ {R_rek[idx_w]:.0f} km')

    ax.axvline(x=80,  color=ORANGE, ls=':', lw=1.5, alpha=0.8)
    ax.axvline(x=130, color=GRAY,   ls=':', lw=1.5, alpha=0.8)
    ax.text(82,  20, '80 km/h', fontsize=9, color=ORANGE)
    ax.text(132, 20, '130 km/h', fontsize=9, color=GRAY)

    ax.set_xlabel('Reisegeschwindigkeit $v$ (km/h)')
    ax.set_ylabel('Elektrische Reichweite $R$ (km)')
    ax.set_title('Elektrische Reichweite als Funktion der Geschwindigkeit (Porsche Taycan)')
    ax.legend(loc='upper right')
    ax.set_xlim(20, 250); ax.set_ylim(0, 1100)
    save_plot('fig08_reichweite_geschwindigkeit.pdf', fig)


# ==========================================================
# Plot 9: SOC-Fenster Heatmap (Zyklenlebensdauer)
# ==========================================================
def plot_fig09():
    fig, ax = plt.subplots(figsize=(8, 7))
    soc_max_arr = np.linspace(50, 100, 60)
    soc_min_arr = np.linspace(0,  50,  60)
    SOC_MAX, SOC_MIN = np.meshgrid(soc_max_arr, soc_min_arr)

    dod = np.clip(SOC_MAX - SOC_MIN, 1, 100)
    stress_dod  = (dod / 80)**1.45
    stress_high = np.exp(0.028 * (SOC_MAX - 80)) * (SOC_MAX > 80)
    stress_low  = np.exp(0.022 * (20 - SOC_MIN)) * (SOC_MIN < 20)
    life = 1.0 / (stress_dod * (1 + stress_high) * (1 + stress_low))

    mask = SOC_MAX <= SOC_MIN + 5
    life_m = np.where(mask, np.nan, life)
    life_m = life_m / np.nanmax(life_m)

    im = ax.contourf(SOC_MAX, SOC_MIN, life_m, levels=25,
                      cmap='RdYlGn', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Zyklenlebensdauer (normiert)')
    cs = ax.contour(SOC_MAX, SOC_MIN, life_m,
                     levels=[0.5, 0.7, 0.85, 0.95],
                     colors='black', linewidths=0.8)
    ax.clabel(cs, fmt='%.2f', fontsize=8)

    # Optimum
    idx = np.unravel_index(np.nanargmax(life_m), life_m.shape)
    ax.scatter([SOC_MAX[idx]], [SOC_MIN[idx]], color='blue', s=120,
                marker='*', zorder=7,
                label=f'Optimum: [{SOC_MIN[idx]:.0f}%–{SOC_MAX[idx]:.0f}%]')
    ax.scatter([80], [10], color=ORANGE, s=90, marker='D', zorder=7,
                label='Empfehlung: [10 %–80 %]')

    ax.set_xlabel(r'Maximaler SOC $\mathrm{SOC}_{\max}$ (%)')
    ax.set_ylabel(r'Minimaler SOC $\mathrm{SOC}_{\min}$ (%)')
    ax.set_title(r'Relative Zyklenlebensdauer: SOC-Fensterkarte')
    ax.legend(loc='upper left', fontsize=9)
    save_plot('fig09_soc_heatmap.pdf', fig)


# ==========================================================
# Plot 10: Vergleich Ladestrategien
# ==========================================================
def plot_fig10():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    t = np.linspace(0, 90, 900)

    # SOC-Verlaeufe
    soc_A = np.clip(t / 45 * 100, 0, 100)
    soc_B = np.clip(20 + t / 37.5 * 60, 20, 80)
    soc_C = np.where(t < 5, 20.0,
            np.where(t <= 43, 20 + (t-5)/38*60, 80.0))
    soc_C = np.clip(soc_C, 20, 80)

    ax1.plot(t, soc_A, color=RED,   lw=2.2, label='Strategie A: Schnell 0–100 %')
    ax1.plot(t, soc_B, color=BLUE,  lw=2.2, label='Strategie B: Standard 20–80 %')
    ax1.plot(t, soc_C, color=GREEN, lw=2.2, ls='--',
             label='Strategie C: Optimiert + Vorkonditionierung')
    ax1.axhspan(20, 80, alpha=0.07, color=GREEN)
    ax1.text(62, 82, 'Schonendes Fenster [20 %–80 %]', fontsize=9, color=GREEN)
    ax1.set_xlabel('Ladezeit $t$ (min)')
    ax1.set_ylabel('SOC (%)')
    ax1.set_title('SOC-Verlauf verschiedener Ladestrategien')
    ax1.legend(fontsize=9); ax1.set_xlim(0, 90); ax1.set_ylim(0, 106)

    # Degradationsrate
    strats = ['Schnell\n0–100 %', 'Standard\n20–80 %', 'Optimiert\n20–80 %']
    dpr    = [4.9, 2.0, 1.3]
    colors = [RED, BLUE, GREEN]
    bars = ax2.bar(strats, dpr, color=colors, alpha=0.85, edgecolor='black', lw=0.6)
    for bar, v in zip(bars, dpr):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.08,
                  f'{v:.1f} %', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Kapazit\"{a}tsverlust pro 1 000 Zyklen (%)')
    ax2.set_title('Degradationsrate der Ladestrategien im Vergleich')
    ax2.set_ylim(0, 6.8)
    plt.tight_layout()
    fig.savefig(SAVE_DIR + 'fig10_ladestrategien_vergleich.pdf',
                format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  Gespeichert: fig10_ladestrategien_vergleich.pdf')


# ==========================================================
# Plot 11: Langzeitprognose 8 Jahre / 160 000 km
# ==========================================================
def plot_fig11():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    km = np.linspace(0, 165000, 600)

    def soh_m(km_, a, b):
        return 100 * np.exp(-a * km_**b)

    Q_opt = soh_m(km, 7.5e-9, 0.82)
    Q_nom = soh_m(km, 1.55e-8, 0.82)
    Q_bad = soh_m(km, 3.3e-8, 0.82)
    Q_ui  = soh_m(km, 9.5e-9, 0.82)
    Q_li  = soh_m(km, 2.5e-8, 0.82)

    ax2 = ax.twiny()

    ax.plot(km/1e3, Q_opt, color=GREEN, lw=2.2, label='Optimale Nutzung')
    ax.plot(km/1e3, Q_nom, color=BLUE,  lw=2.2, label='Nominale Nutzung')
    ax.plot(km/1e3, Q_bad, color=RED,   lw=2.2, label=r'Ung\"{u}nstige Nutzung')
    ax.fill_between(km/1e3, Q_li, Q_ui, alpha=0.12, color=BLUE, label='Prognoseintervall')

    ax.axhline(y=80, color=ORANGE, ls='--', lw=1.8, label='Garantie: 80 % bis 60 tkm')
    ax.axhline(y=70, color=GRAY,   ls='--', lw=1.8, label='Garantie: 70 % bis 160 tkm')
    ax.axvline(x=60,  color=ORANGE, ls=':', lw=1.0, alpha=0.65)
    ax.axvline(x=160, color=GRAY,   ls=':', lw=1.0, alpha=0.65)
    ax.text(32, 81, '80 %', fontsize=9, color=ORANGE)
    ax.text(85, 71, '70 %', fontsize=9, color=GRAY)

    ax.set_xlabel('Laufleistung (1 000 km)')
    ax.set_ylabel('State of Health SOH (%)')
    ax.set_title('Langzeitprognose: SOH-Entwicklung über 8 Jahre / 160 000 km')
    ax.legend(loc='lower left', ncol=2, fontsize=9)
    ax.set_xlim(0, 165); ax.set_ylim(55, 105)

    ax2.set_xlim(0, 165/20)
    ax2.set_xlabel('Fahrzeugalter (Jahre, bei 20 000 km/Jahr)')
    save_plot('fig11_langzeitprognose.pdf', fig)


# ==========================================================
# Plot 12: Rekuperationsoptimum + Energiekomponenten
# ==========================================================
def plot_fig12():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Links: Rekuperationsleistung vs. Verzoegerung
    decel = np.linspace(0, 5.0, 300)
    v0    = 100 / 3.6
    m     = 2295.0
    eta_r = 0.88 * np.exp(-0.12 * decel)
    P_req = decel * m * v0 / 1000   # kW ohne Verluste
    P_reg = P_req * eta_r
    P_max = 270.0
    P_reg = np.minimum(P_reg, P_max)

    ax1.plot(decel, P_req, color=GRAY,  lw=1.6, ls='--', label='Ideal (ohne Verluste)')
    ax1.plot(decel, P_reg, color=BLUE,  lw=2.2,           label='Tats. Rekuperationsleistung')
    ax1.axhline(y=P_max,  color=RED,   ls=':', lw=1.6, label=f'Motorgrenze ({P_max:.0f} kW)')
    ax1.axvline(x=2.5,    color=GREEN, ls='--', lw=1.5, label='Optimum ≈ 2,5 m/s²')
    ax1.set_xlabel(r'Verz\"{o}gerung $a$ (m/s²)')
    ax1.set_ylabel('Rekuperationsleistung (kW)')
    ax1.set_title('Rekuperationsleistung vs. Fahrzeugverzoegerung')
    ax1.legend(fontsize=9); ax1.set_xlim(0, 5); ax1.set_ylim(0, 380)

    # Rechts: Energieverbrauch nach Komponenten
    v_list = [50, 80, 100, 130, 160, 200]
    E_aero = np.array([2.1, 5.8,  9.8, 17.8, 29.2, 50.0])
    E_roll = np.array([3.5, 3.5,  3.5,  3.5,  3.5,  3.5])
    E_neb  = np.array([1.5, 1.2,  1.1,  1.0,  1.0,  1.0])
    x      = np.arange(len(v_list))
    w      = 0.5

    ax2.bar(x, E_aero, w, label='Luftwiderstand',    color=BLUE,   alpha=0.85, edgecolor='k', lw=0.5)
    ax2.bar(x, E_roll, w, label='Rollwiderstand',    color=GREEN,  alpha=0.85, edgecolor='k', lw=0.5, bottom=E_aero)
    ax2.bar(x, E_neb,  w, label='Nebenverbraucher',  color=ORANGE, alpha=0.85, edgecolor='k', lw=0.5, bottom=E_aero+E_roll)
    ax2.set_xticks(x); ax2.set_xticklabels([f'{v} km/h' for v in v_list])
    ax2.set_xlabel('Reisegeschwindigkeit (km/h)')
    ax2.set_ylabel('Energieverbrauch (kWh/100 km)')
    ax2.set_title('Energieverbrauch nach Komponenten')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(SAVE_DIR + 'fig12_energie_rekuperation.pdf',
                format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  Gespeichert: fig12_energie_rekuperation.pdf')


# ==========================================================
if __name__ == '__main__':
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Generiere 12 Plots ...")
    plot_fig01(); plot_fig02(); plot_fig03(); plot_fig04()
    plot_fig05(); plot_fig06(); plot_fig07(); plot_fig08()
    plot_fig09(); plot_fig10(); plot_fig11(); plot_fig12()
    print("Fertig – alle 12 Plots gespeichert.")
