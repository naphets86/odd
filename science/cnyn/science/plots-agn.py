#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generierung aller matplotlib-Plots für das neue Kapitel:
"Präzessierender AGN-Jet in VV 340a: Physikalische Analyse und kosmologische Einordnung"
Basierend auf: Kader et al. (2026), Science 391, 911, DOI: 10.1126/science.adp8989
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# Farb-Palette (konsistent mit dem LaTeX-Dokument)
BLUE   = '#1946AA'
RED    = '#B4321E'
GREEN  = '#1E6432'
ORANGE = '#E08020'
PURPLE = '#6A1E8C'
GRAY   = '#505060'
LGREEN = '#A8D8A0'
LBLUE  = '#A0C0F0'
LRED   = '#F0A0A0'

DPI = 150
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQ   = (12, 10)
FIGSIZE_TALL = (12, 12)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Eddington-Verhältnis und Akkretionsrate
# ─────────────────────────────────────────────────────────────────────────────
def plot_eddington_ratio():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Schwarzes Loch in VV 340a: Akkretionsphysik und Eddington-Analyse',
                 fontsize=14, fontweight='bold', y=1.02)

    # --- Teilplot 1: Eddington-Leuchtdichte vs. Schwarzlochmasse ---
    ax = axes[0]
    M_BH = np.logspace(6, 10, 300)   # in M_sun
    L_Edd = 1.26e38 * M_BH           # erg/s
    L_AGN = 1.28e44 * np.ones_like(M_BH)  # gemessener Wert

    ax.loglog(M_BH, L_Edd, color=BLUE, lw=2.5, label=r'$L_\mathrm{Edd}(M_\mathrm{BH})$')
    ax.loglog(M_BH, L_AGN, color=RED, lw=2, ls='--', label=r'$L_\mathrm{AGN,bol} = 1{,}28\times10^{44}$ erg/s')
    M_meas = 3.0e7
    ax.axvline(M_meas, color=GREEN, lw=1.5, ls=':', label=r'$M_\mathrm{BH} = 3\times10^7\,M_\odot$')
    ax.scatter([M_meas], [1.26e38 * M_meas], color=GREEN, s=80, zorder=5)
    # Eddington ratio
    lam = 1.28e44 / (1.26e38 * M_meas)
    ax.annotate(fr'$\lambda_\mathrm{{Edd}} = {lam:.3f}$',
                xy=(M_meas, 1.26e38 * M_meas), xytext=(1e8, 5e44),
                arrowprops=dict(arrowstyle='->', color=GREEN), fontsize=10, color=GREEN)
    ax.set_xlabel(r'$M_\mathrm{BH}\;[M_\odot]$')
    ax.set_ylabel(r'Leuchtkraft $L$ [erg/s]')
    ax.set_title('Eddington-Leuchtkraft vs. Masse')
    ax.legend(fontsize=9)

    # --- Teilplot 2: Akkretionsrate-Vergleich ---
    ax = axes[1]
    categories = ['$\dot{M}_\mathrm{acc}$\n(gemessen)', '$\dot{M}_\mathrm{Edd}$\n(Eddington)', 'Faktor 27']
    values = [2.3e-2, 0.035 * 6.5e-1, 2.3e-2]  # vereinfacht zur Darstellung
    # Balkendiagramm mit Fehlerbalken
    M_acc = 2.3e-2    # M_sun/yr
    M_Edd_rate = 6.5e-1   # M_sun/yr (Eddington rate aus Papier)
    bars = ax.bar([0, 1], [M_acc, M_Edd_rate], color=[RED, BLUE], alpha=0.8,
                  edgecolor='black', width=0.5)
    ax.bar([0], [M_acc], color=RED, alpha=0.8, edgecolor='black', width=0.5,
           yerr=0.1*M_acc, capsize=6)
    ax.bar([1], [M_Edd_rate], color=BLUE, alpha=0.8, edgecolor='black', width=0.5,
           yerr=0.15*M_Edd_rate, capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$\dot{M}_\mathrm{acc}$' + '\n(tatsächlich)', r'$\dot{M}_\mathrm{Edd}$' + '\n(Eddington)'])
    ax.set_ylabel(r'Akkretionsrate $[\,M_\odot\,\mathrm{yr}^{-1}]$')
    ax.set_title('Akkretionsraten-Vergleich')
    ratio = M_Edd_rate / M_acc
    ax.annotate(fr'Verhältnis: {ratio:.1f}×', xy=(0.5, max(M_acc, M_Edd_rate) * 0.85),
                ha='center', fontsize=11, color=PURPLE,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # --- Teilplot 3: Ne V Fluss-Profil (simuliert) ---
    ax = axes[2]
    v = np.linspace(-800, 800, 400)  # km/s
    # Zwei Gauss-Komponenten (NE: redshift, SW: blueshift)
    gauss_NE = 0.8 * np.exp(-(v - 350)**2 / (2*120**2))
    gauss_SW = 0.9 * np.exp(-(v + 320)**2 / (2*130**2))
    gauss_sys = 1.0 * np.exp(-v**2 / (2*80**2))
    total = gauss_NE + gauss_SW + gauss_sys
    ax.fill_between(v, gauss_NE, alpha=0.3, color=RED, label='NE-Filament (rotverschoben)')
    ax.fill_between(v, gauss_SW, alpha=0.3, color=BLUE, label='SW-Filament (blauverschoben)')
    ax.fill_between(v, gauss_sys, alpha=0.3, color=GREEN, label='Systembewegung')
    ax.plot(v, total, color='black', lw=2, label='Gesamt [Ne V] 14 µm')
    ax.axvline(0, color=GRAY, ls='--', lw=1)
    ax.axvline(500, color=RED, ls=':', lw=1, alpha=0.7)
    ax.axvline(-500, color=BLUE, ls=':', lw=1, alpha=0.7)
    ax.set_xlabel('LOS-Geschwindigkeit [km/s]')
    ax.set_ylabel('Relative Intensität')
    ax.set_title('[Ne V] 14 µm Emissionslinienprofil')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('agn_plot1_eddington.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 1 gespeichert: agn_plot1_eddington.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Präzessierender Jet: 3D-Modell und Projektionsgeometrie
# ─────────────────────────────────────────────────────────────────────────────
def plot_precessing_jet():
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('Präzessierender AGN-Jet in VV 340a: Modell und Geometrie',
                 fontsize=14, fontweight='bold')

    # --- 3D-Jet-Spirale ---
    ax1 = fig.add_subplot(131, projection='3d')
    # Parameter aus dem Paper
    beta = 0.033
    phi_prec = np.radians(14)   # Präzessionswinkel
    psi_PA   = np.radians(33)   # Präzessionsachse PA
    i_incl   = np.radians(71)   # Neigung zur Sichtlinie
    P_prec   = 8.2e5            # Jahre

    t = np.linspace(0, 2 * np.pi * 3, 500)
    # Helikale Trajektorie
    r = t / (2 * np.pi) * 10   # kpc
    x = r * np.cos(t) * np.sin(phi_prec)
    y = r * np.sin(t) * np.sin(phi_prec)
    z = r * np.cos(phi_prec)
    # Rückseite (Gegenseite des Jets)
    xb = -x
    yb = -y
    zb = -z

    ax1.plot(x, y, z, color=RED, lw=2, label='Jet (Ost)')
    ax1.plot(xb, yb, zb, color=BLUE, lw=2, label='Gegenjet (West)')
    ax1.scatter([0], [0], [0], color='gold', s=150, zorder=5, label='AGN-Kern')
    ax1.set_xlabel('X [kpc]')
    ax1.set_ylabel('Y [kpc]')
    ax1.set_zlabel('Z [kpc]')
    ax1.set_title('3D-Jet-Trajektorie\n(helikale Präzession)')
    ax1.legend(fontsize=8)

    # --- Projektion auf den Himmel ---
    ax2 = fig.add_subplot(132)
    # Projizierter S-förmiger Jet (analog Abb. 4 des Papers)
    t2 = np.linspace(0, 4 * np.pi, 600)
    r2 = t2 / (4 * np.pi) * 15
    # Projektion mit Neigung i und PA psi
    xp =  r2 * np.cos(t2) * np.cos(psi_PA) - r2 * np.sin(t2) * np.sin(psi_PA) * np.cos(i_incl)
    yp =  r2 * np.cos(t2) * np.sin(psi_PA) + r2 * np.sin(t2) * np.cos(psi_PA) * np.cos(i_incl)
    mask = t2 < 2 * np.pi
    ax2.plot(xp[mask], yp[mask], color=RED, lw=2.5, label=f'Jet (Ost), PA={33}°')
    ax2.plot(-xp[~mask], -yp[~mask], color=BLUE, lw=2.5, label=f'Jet (West), PA={33+180}°')
    ax2.scatter([0], [0], color='gold', s=200, zorder=5, label='AGN-Kern')
    # Nordrichtung
    ax2.annotate('N', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                 ha='center', fontweight='bold')
    ax2.annotate('', xy=(0.05, 0.93), xytext=(0.05, 0.8),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    ax2.set_xlabel('RA-Offset [arcsec]')
    ax2.set_ylabel('Dec-Offset [arcsec]')
    ax2.set_title(f'Himmelsprojektion\n$\\psi={33}°$, $\\phi={14}°$, $i={71}°$')
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')

    # --- Präzessionsperiode und Energieabschätzung ---
    ax3 = fig.add_subplot(133)
    P_values = np.logspace(4, 7, 200)  # Jahre
    v_jet = beta * 3e5  # km/s
    # Kinetische Leistung (skaliert nach 1.5 GHz Flussdichte)
    L_radio_E = 12.4e-3 * 1e-23 * 4 * np.pi * (157e6 * 3.086e18)**2  # erg/s/Hz
    # Jet-Mächtigkeiten aus dem Paper
    P_jet_E = np.array([0.87e43, 2.20e43])  # erg/s
    P_jet_W = np.array([0.68e43, 2.00e43])  # erg/s
    P_labels = ['Skalenrel. (84)', 'Skalenrel. (85)']

    y_pos = [1, 2, 3.5, 4.5]
    values = [P_jet_E[0], P_jet_E[1], P_jet_W[0], P_jet_W[1]]
    colors_bar = [RED, RED, BLUE, BLUE]
    labels_bar = ['Ostjet (Rel. 84)', 'Ostjet (Rel. 85)', 'Westjet (Rel. 84)', 'Westjet (Rel. 85)']
    bars = ax3.barh(y_pos, [v/1e43 for v in values], color=colors_bar, alpha=0.75,
                    edgecolor='black', height=0.6)
    for bar, val, unc_frac in zip(bars, values,
                                  [15.4/0.87, 30.31/2.20, 11.96/0.68, 27.53/2.00]):
        err = val * min(unc_frac, 10) / 10 / 1e43
        ax3.errorbar(val/1e43, bar.get_y() + bar.get_height()/2,
                     xerr=err, fmt='none', color='black', capsize=4)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_bar, fontsize=9)
    ax3.set_xlabel(r'Jet-Leistung [$10^{43}$ erg/s]')
    ax3.set_title('Kinetische Jet-Leistungen\n(nach Skalenrelationen)')
    ax3.axvline(1.0, color=GRAY, ls='--', lw=1, label=r'$10^{43}$ erg/s')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('agn_plot2_jet_geometrie.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 2 gespeichert: agn_plot2_jet_geometrie.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – Bikonaler Ausfluss: P-V-Diagramm und Geschwindigkeitsgesetz
# ─────────────────────────────────────────────────────────────────────────────
def plot_bicone_outflow():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Bikonaler Gasausfluss in VV 340a: P-V-Diagramme und Modellkurven',
                 fontsize=14, fontweight='bold')

    # --- Teilplot 1: Bikonales Geschwindigkeitsgesetz ---
    ax = axes[0, 0]
    r = np.linspace(0, 25, 400)   # kpc
    k1 = 1000.0    # s^-1 (Beschleunigung)
    k2 = 100.0     # s^-1 (Verzögerung)
    rt = 1.25      # kpc (Umkehrradius)
    v_max = k1 * rt   # km/s (Achtung: Einheiten vereinfacht)
    v_max_kms = 1200  # km/s aus Paper

    v_model = np.where(r <= rt,
                       v_max_kms * (r / rt),
                       v_max_kms - k2 * (r - rt) * (v_max_kms / (k1 * rt)))
    v_model = np.clip(v_model, 0, v_max_kms)

    # Galaktisches Rotationsmodell (Miyamoto-Nagai vereinfacht)
    v_disk = 200 * np.sqrt(r / (r + 3))  # km/s, Sättigungs-Rotationskurve

    ax.plot(r, v_model, color=RED, lw=2.5, label=r'Bikonaler Ausfluss $v_r(r)$')
    ax.plot(r, v_disk, color=BLUE, lw=2, ls='--', label='Galaktische Rotation')
    ax.fill_between(r, v_model - 50, v_model + 50, alpha=0.2, color=RED)
    ax.axvline(rt, color=GRAY, ls=':', lw=1.5, label=fr'$r_t = {rt}$ kpc')
    ax.axhline(v_max_kms, color=GREEN, ls=':', lw=1.5, label=fr'$v_\mathrm{{max}} = {v_max_kms}$ km/s')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Deprojizierte Radialgeschwindigkeit [km/s]')
    ax.set_title('Geschwindigkeitsgesetz des bikonalen Ausflusses')
    ax.legend()
    ax.set_xlim(0, 20)

    # --- Teilplot 2: P-V Diagramm [Ne V] (simuliert nach Fig. 3E des Papers) ---
    ax = axes[0, 1]
    # Simuliertes P-V Diagramm
    pos = np.linspace(-20, 20, 200)   # arcsec
    vel = np.linspace(-600, 600, 200) # km/s
    P, V = np.meshgrid(pos, vel)
    # Zwei Ausfluss-Komponenten + Rotation
    outflow_NE = np.exp(-((P - 5)**2 / 8 + (V - 350)**2 / 7000))
    outflow_SW = np.exp(-((P + 5)**2 / 8 + (V + 320)**2 / 7000))
    rotation   = np.exp(-((P * 15 - V)**2 / 8000 + P**2 / 50))
    total_pv   = outflow_NE + outflow_SW + 0.5 * rotation
    im = ax.contourf(pos, vel, total_pv, levels=20, cmap='inferno')
    ax.contour(pos, vel, total_pv, levels=5, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label='Relative Intensität')
    ax.set_xlabel('Position entlang Spalt (PA=45°) [arcsec]')
    ax.set_ylabel('LOS-Geschwindigkeit [km/s]')
    ax.set_title('[Ne V] 14 µm P-V-Diagramm\n(Modell, PA = 45°)')
    ax.axhline(0, color='white', ls='--', lw=1, alpha=0.7)
    ax.axvline(0, color='white', ls=':', lw=1, alpha=0.7)
    ax.text(8, 400, 'NE\n(rot)', color='white', fontsize=10, ha='center')
    ax.text(-8, -400, 'SW\n(blau)', color='white', fontsize=10, ha='center')

    # --- Teilplot 3: Massenausflusrate als Funktion der Elektrondichte ---
    ax = axes[1, 0]
    ne_range = np.logspace(1, 4, 200)  # cm^-3
    m_p = 1.67e-24   # g
    v_max_cgs = 1.2e8  # cm/s
    A = 2.9e5 * 3.086e18**2  # cm^2 (aus Parsec)
    f_fill = 0.001
    # Umrechnung in M_sun/yr
    M_sun_g = 1.989e33
    yr_s = 3.156e7
    Mdot = (2 * m_p * ne_range * v_max_cgs * A * f_fill) / (M_sun_g / yr_s)

    ax.loglog(ne_range, Mdot, color=RED, lw=2.5, label=r'$\dot{M}_\mathrm{out}(n_e)$')
    # Gemessene Dichte
    ne_meas_NeV = 1.07e3
    ne_meas_OII = 7.6e2
    Mdot_NeV = (2 * m_p * ne_meas_NeV * v_max_cgs * A * f_fill) / (M_sun_g / yr_s)
    Mdot_OII = (2 * m_p * ne_meas_OII * v_max_cgs * A * f_fill) / (M_sun_g / yr_s)
    ax.scatter([ne_meas_NeV], [Mdot_NeV], color=PURPLE, s=120, zorder=5,
               label=fr'[Ne V]: $n_e={ne_meas_NeV:.0f}$ cm$^{{-3}}$, $\dot{{M}}={Mdot_NeV:.1f}\,M_\odot$/yr')
    ax.scatter([ne_meas_OII], [Mdot_OII], color=GREEN, s=120, zorder=5, marker='s',
               label=fr'[O II]: $n_e={ne_meas_OII:.0f}$ cm$^{{-3}}$, $\dot{{M}}={Mdot_OII:.1f}\,M_\odot$/yr')
    ax.fill_between(ne_range,
                    (2 * m_p * ne_range * v_max_cgs * A * 0.0005) / (M_sun_g / yr_s),
                    (2 * m_p * ne_range * v_max_cgs * A * 0.002)  / (M_sun_g / yr_s),
                    alpha=0.2, color=RED, label='Füllungsfaktor $f = 0{,}0005$–$0{,}002$')
    ax.set_xlabel(r'Elektronendichte $n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'Massenausflusrate $\dot{M}_\mathrm{out}$ [$M_\odot$/yr]')
    ax.set_title('Massenausflusrate vs. Elektronendichte')
    ax.legend(fontsize=8)

    # --- Teilplot 4: Kinetische Leistung des Ausflusses ---
    ax = axes[1, 1]
    # Berechnung der kinetischen Leistung für verschiedene v_max und sigma_turb
    v_range = np.linspace(500, 2000, 200)
    sigma_turb = 326  # km/s (gemessen)
    ne_val = ne_meas_NeV

    Mdot_v = (2 * m_p * ne_val * (v_range * 1e5) * A * f_fill) / (M_sun_g / yr_s)
    Edot_v = 0.5 * Mdot_v * (M_sun_g / yr_s) * ((v_range * 1e5)**2 + (sigma_turb * 1e5)**2)

    ax.plot(v_range, Edot_v / 1e43, color=RED, lw=2.5, label=r'$\dot{E}_\mathrm{out}(v_\mathrm{max})$')
    ax.axvline(1200, color=BLUE, ls='--', lw=2, label=r'$v_\mathrm{max} = 1200$ km/s (Modell)')
    ax.scatter([1200], [(0.5 * Mdot_NeV * (M_sun_g/yr_s) * ((1200e5)**2 + (326e5)**2)) / 1e43],
               color=GREEN, s=120, zorder=5, label=r'$\dot{E}_\mathrm{out} = 1{,}0 \times 10^{43}$ erg/s')
    # Vergleich mit Jet-Leistung
    ax.axhspan(0.87, 2.20, alpha=0.15, color=BLUE, label='Jet-Leistung (Ostjet)')
    ax.set_xlabel(r'$v_\mathrm{max}$ [km/s]')
    ax.set_ylabel(r'Kinetische Ausflussleistung $\dot{E}_\mathrm{out}$ [$10^{43}$ erg/s]')
    ax.set_title('Kinetische Ausflussleistung\n(inkl. turbulenter Beitrag)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('agn_plot3_ausfluss.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 3 gespeichert: agn_plot3_ausfluss.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – Ionisationsmechanismen: BPT-Diagramm
# ─────────────────────────────────────────────────────────────────────────────
def plot_ionization_bpt():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Ionisationsmechanismen in VV 340a: BPT-Diagramm und räumliche Verteilung',
                 fontsize=14, fontweight='bold')

    # --- BPT-Diagramm ---
    ax = axes[0]
    # Trennlinie nach Kewley et al. (2001) - Theorie-Maximum
    log_OII_Hb = np.linspace(-1.5, 0.4, 300)
    log_NII_Ha_kewley = 0.61 / (log_OII_Hb - 0.47) + 1.19

    # Kauffmann et al. (2003) - empirische SF/AGN-Grenze
    log_NII_Ha_kauf = np.linspace(-1.5, -0.1, 200)
    log_OII_Hb_kauf = 0.61 / (log_NII_Ha_kauf - 0.05) + 1.3

    ax.plot(log_OII_Hb, log_NII_Ha_kewley, color=RED, lw=2,
            label='Kewley et al. (2001) – Theorie-Grenze')
    ax.plot(log_NII_Ha_kauf, log_OII_Hb_kauf, color=BLUE, lw=2, ls='--',
            label='Kauffmann et al. (2003) – SF-Grenze')

    # HII-Regionen (Wolke)
    np.random.seed(42)
    n_hii = 200
    x_hii = np.random.normal(-0.5, 0.2, n_hii)
    y_hii = np.random.normal(-0.3, 0.25, n_hii)
    ax.scatter(x_hii, y_hii, c=LGREEN, s=15, alpha=0.5, label='HII-Regionen (typisch)', zorder=2)

    # Messungen aus dem Paper
    disk_apertures = [
        {'pos': (0.25, -0.1), 'label': 'Kern', 'color': RED},
        {'pos': (-0.09, -0.2), 'label': 'N-Disk (+10")', 'color': ORANGE},
        {'pos': (-0.12, -0.3), 'label': 'S-Disk (-11")', 'color': ORANGE},
        {'pos': (0.42, 0.1),  'label': 'SE-Nebel', 'color': PURPLE},
        {'pos': (0.50, 0.15), 'label': 'NW-Nebel', 'color': PURPLE},
        {'pos': (1.00, 0.5),  'label': 'NE-Filament', 'color': GREEN},
        {'pos': (0.84, 0.4),  'label': 'SW-Filament (9")', 'color': GREEN},
        {'pos': (1.03, 0.6),  'label': 'SW-Filament (20")', 'color': GREEN},
    ]
    for dp in disk_apertures:
        ax.scatter(*dp['pos'], color=dp['color'], s=100, zorder=5,
                   edgecolors='black', linewidths=0.7)
        ax.annotate(dp['label'], xy=dp['pos'], xytext=(dp['pos'][0]+0.05, dp['pos'][1]+0.08),
                    fontsize=8, color=dp['color'])

    ax.set_xlabel(r'log([O III]/H$\beta$)')
    ax.set_ylabel(r'log([N II]/H$\alpha$)')
    ax.set_title('BPT-Diagramm: Ionisationsmechanismen')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.2)
    # Beschriftung der Zonen
    ax.text(-0.7, -0.9, 'SF-Ionisation', fontsize=10, color=GREEN,
            bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.text(0.6, 0.8, 'AGN/Schock-\nionisation', fontsize=10, color=RED,
            bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.legend(fontsize=8, loc='lower right')

    # --- Räumliche Verteilung der Ionisationsregionen ---
    ax = axes[1]
    # Schematische Karte
    theta = np.linspace(0, 2*np.pi, 100)
    # Galaxienscheibe (Ellipse)
    ax.add_patch(mpatches.Ellipse((0, 0), width=30, height=8, angle=0,
                                   fill=True, facecolor=LBLUE, edgecolor=BLUE,
                                   alpha=0.4, label='Galaktische Scheibe'))
    # Bikonaler Ausfluss (zwei Kegel)
    cone_angle = np.radians(20)
    r_cone = 20
    for sign, color, label in [(1, RED, 'NE-Ausfluss (rot)'), (-1, BLUE, 'SW-Ausfluss (blau)')]:
        x_cone = np.array([0, sign * r_cone * np.cos(np.radians(45) - cone_angle),
                           sign * r_cone * np.cos(np.radians(45) + cone_angle), 0])
        y_cone = np.array([0, sign * r_cone * np.sin(np.radians(45) - cone_angle),
                           sign * r_cone * np.sin(np.radians(45) + cone_angle), 0])
        ax.fill(x_cone, y_cone, alpha=0.3, color=color, label=label)
        ax.plot(x_cone, y_cone, color=color, lw=1.5)

    # Filamente
    for sign, col in [(1, RED), (-1, BLUE)]:
        for off in [-1, 0, 1]:
            ax.annotate('', xy=(sign * 15, sign * 15 + off * 2),
                       xytext=(sign * 2, sign * 2 + off * 0.5),
                       arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # AGN-Kern
    ax.scatter([0], [0], color='gold', s=300, zorder=10, label='AGN-Kern')
    # Skalierungsbalken
    ax.plot([8, 13], [-12, -12], 'k-', lw=2)
    ax.text(10.5, -13.5, '5 kpc', ha='center', fontsize=9)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-15, 22)
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_title('Schematische Karte der Emissions-\nregionen in VV 340a')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')

    # Farb-Legende für BPT
    handles = [
        mpatches.Patch(color=RED, label='Kern/AGN-dominiert'),
        mpatches.Patch(color=ORANGE, label='Scheibe'),
        mpatches.Patch(color=PURPLE, label='Seitliche Nebel'),
        mpatches.Patch(color=GREEN, label='Filamente (AGN/Schock)'),
    ]
    axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0][:2],
                   fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig('agn_plot4_ionisation.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 4 gespeichert: agn_plot4_ionisation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – Energiebilanz: AGN-Leuchtkraft, Jet, Ausfluss
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy_budget():
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle('Energiebilanz des AGN-Systems VV 340a',
                 fontsize=14, fontweight='bold')

    # --- Luminositätsvergleich ---
    ax = axes[0]
    labels = [
        r'$L_\mathrm{AGN,bol}$ [Ne V]',
        r'$L_\mathrm{AGN,bol}$ [O III]',
        r'$L_\mathrm{AGN,bol}$ (mm)',
        r'$L_\mathrm{AGN,bol}$ (X-ray)',
        r'$L_\mathrm{Edd}$',
        r'$\dot{E}_\mathrm{out}$ (Ausfluss)',
        r'$P_\mathrm{jet}$ (Ostjet)',
        r'$P_\mathrm{jet}$ (Westjet)',
    ]
    values_erg = [1.28e44, 1.1e44, 4e44, 2.1e43, 3.7e45, 1.0e43, 0.87e43, 0.68e43]
    colors_e = [RED, RED, RED, ORANGE, PURPLE, BLUE, GREEN, GREEN]
    markers_e = ['o', 's', '^', 'D', '*', 'p', 'h', 'h']

    y_pos = np.arange(len(labels))
    for i, (v, c, m) in enumerate(zip(values_erg, colors_e, markers_e)):
        ax.barh(i, np.log10(v), color=c, alpha=0.75, edgecolor='black', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'$\log_{10}(L\;\mathrm{[erg/s]})$')
    ax.set_title('Leuchtkraft- und Leistungsvergleich')
    ax.axvline(44, color=GRAY, ls='--', lw=1, alpha=0.7, label=r'$10^{44}$ erg/s')
    ax.axvline(43, color=GRAY, ls=':', lw=1, alpha=0.7, label=r'$10^{43}$ erg/s')
    ax.legend(fontsize=8)

    # --- Wirkungsgrad des Jet-Ausflusses ---
    ax = axes[1]
    eta_range = np.linspace(0.001, 0.3, 300)
    L_AGN = 1.28e44
    P_jet_model = eta_range * L_AGN
    eta_meas_E  = 0.87e43 / L_AGN
    eta_meas_W  = 0.68e43 / L_AGN

    ax.semilogy(eta_range * 100, P_jet_model, color=BLUE, lw=2.5,
                label=r'$P_\mathrm{jet} = \eta L_\mathrm{AGN,bol}$')
    ax.axhline(0.87e43, color=RED, ls='--', lw=2, label=r'Ostjet: $0.87 \times 10^{43}$ erg/s')
    ax.axhline(0.68e43, color=GREEN, ls='--', lw=2, label=r'Westjet: $0.68 \times 10^{43}$ erg/s')
    ax.axvline(eta_meas_E * 100, color=RED, ls=':', lw=1)
    ax.axvline(eta_meas_W * 100, color=GREEN, ls=':', lw=1)
    ax.fill_betweenx([0.68e43, 0.87e43], [eta_meas_W * 100], [eta_meas_E * 100],
                     alpha=0.2, color=ORANGE)
    ax.set_xlabel('Jet-Wirkungsgrad η [%]')
    ax.set_ylabel(r'$P_\mathrm{jet}$ [erg/s]')
    ax.set_title('Jet-Wirkungsgrad\n(Kopplung an AGN-Leuchtkraft)')
    ax.legend(fontsize=8)

    # --- Rückkopplungseffizienz: Ausfluss/AGN ---
    ax = axes[2]
    # Zeitentwicklung des Ausflusses (vereinfacht)
    t_Myr = np.linspace(0, 10, 300)
    M_dot_out = 19.4 * np.exp(-t_Myr / 3)   # M_sun/yr, abnehmend
    E_dot_out = 1.0e43 * np.exp(-t_Myr / 3) # erg/s

    ax2r = ax.twinx()
    l1, = ax.plot(t_Myr, M_dot_out, color=RED, lw=2.5, label=r'$\dot{M}_\mathrm{out}(t)$')
    l2, = ax2r.semilogy(t_Myr, E_dot_out, color=BLUE, lw=2.5, ls='--',
                        label=r'$\dot{E}_\mathrm{out}(t)$')
    ax.axhline(19.4, color=RED, ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Zeit [Myr]')
    ax.set_ylabel(r'$\dot{M}_\mathrm{out}$ [$M_\odot$/yr]', color=RED)
    ax2r.set_ylabel(r'$\dot{E}_\mathrm{out}$ [erg/s]', color=BLUE)
    ax.set_title('Zeitentwicklung des Ausflusses\n(exponentieller Abfall)')
    ax.tick_params(axis='y', labelcolor=RED)
    ax2r.tick_params(axis='y', labelcolor=BLUE)
    lns = [l1, l2]
    ax.legend(lns, [l.get_label() for l in lns], fontsize=9)

    plt.tight_layout()
    plt.savefig('agn_plot5_energie.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 5 gespeichert: agn_plot5_energie.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 – Schwarzloch-Masse, M-sigma-Relation
# ─────────────────────────────────────────────────────────────────────────────
def plot_mbh_sigma():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(r'Schwarzloch-Masse und $M_\mathrm{BH}$–$\sigma_*$-Relation in VV 340a',
                 fontsize=14, fontweight='bold')

    # --- M_BH - sigma Relation ---
    ax = axes[0]
    sigma = np.linspace(50, 400, 300)
    # Kormendy & Ho (2013) Relation: log(M_BH/M_sun) = 8.49 + 4.38 * log(sigma/200)
    log_M_BH_KH = 8.49 + 4.38 * np.log10(sigma / 200)
    M_BH_KH = 10**log_M_BH_KH

    # Scatter-Band
    ax.loglog(sigma, M_BH_KH, color=BLUE, lw=2.5, label='Kormendy & Ho (2013)')
    ax.fill_between(sigma,
                    10**(log_M_BH_KH - 0.3),
                    10**(log_M_BH_KH + 0.3),
                    alpha=0.15, color=BLUE, label=r'$\pm 0.3$ dex Streuung')

    # VV 340a Messung
    sigma_meas = 127
    M_BH_meas = 3.0e7
    M_BH_err_lo = 3.0e7 - max(3.0e7 - 5.21e7, 1e6)
    M_BH_err_hi = 5.21e7

    ax.errorbar(sigma_meas, M_BH_meas,
                xerr=16, yerr=[[M_BH_err_lo], [M_BH_err_hi]],
                fmt='o', color=RED, markersize=12, capsize=6, zorder=6,
                label=r'VV 340a: $\sigma_*=127\pm16$ km/s, $M_\mathrm{BH}=(3.0\pm5.2)\times10^7\,M_\odot$')

    # Weitere Referenzgalaxien (illustrativ)
    ref_sigma = [80, 150, 200, 300, 350]
    ref_M = [10**8.49 * (s/200)**4.38 for s in ref_sigma]
    ax.scatter(ref_sigma, ref_M, color=GRAY, s=60, alpha=0.6, zorder=4, label='Referenz-AGN')

    ax.set_xlabel(r'Sterngeschwindigkeitsdispersion $\sigma_*$ [km/s]')
    ax.set_ylabel(r'$M_\mathrm{BH}$ [$M_\odot$]')
    ax.set_title(r'$M_\mathrm{BH}$–$\sigma_*$-Relation')
    ax.legend(fontsize=8)
    ax.set_xlim(40, 500)
    ax.set_ylim(1e6, 1e11)

    # --- Schwarzloch-Parameter Übersicht ---
    ax = axes[1]
    # Schwarzschild-Radius: r_S = 2GM/c^2
    M_BH_vals = np.logspace(6, 10, 300)
    G = 6.674e-8  # cgs
    c = 3e10      # cm/s
    M_sun_g = 1.989e33
    r_S = 2 * G * M_BH_vals * M_sun_g / c**2  # cm
    r_S_AU = r_S / 1.496e13  # AU

    ax.loglog(M_BH_vals, r_S_AU, color=BLUE, lw=2.5, label='Schwarzschild-Radius $r_S$')
    ax.axvline(M_BH_meas, color=RED, ls='--', lw=2, label=r'$M_\mathrm{BH}$ VV 340a')
    r_S_meas = 2 * G * M_BH_meas * M_sun_g / c**2 / 1.496e13
    ax.scatter([M_BH_meas], [r_S_meas], color=RED, s=120, zorder=5)
    ax.annotate(fr'$r_S = {r_S_meas:.2f}$ AU',
                xy=(M_BH_meas, r_S_meas), xytext=(1e8, r_S_meas * 5),
                arrowprops=dict(arrowstyle='->', color=RED), fontsize=10)

    # Vergleich mit bekannten BHs
    known = {
        'Sgr A*': (4.1e6, 'o', GREEN),
        'M87*': (6.5e9, 's', PURPLE),
        'VV 340a': (M_BH_meas, '^', RED),
    }
    for name, (M, mk, col) in known.items():
        r_s = 2 * G * M * M_sun_g / c**2 / 1.496e13
        ax.scatter([M], [r_s], color=col, s=150, marker=mk, zorder=6, label=name)

    ax.set_xlabel(r'$M_\mathrm{BH}$ [$M_\odot$]')
    ax.set_ylabel(r'Schwarzschild-Radius $r_S$ [AU]')
    ax.set_title('Schwarzschild-Radius\nverschiedener Schwarzer Löcher')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('agn_plot6_mbh_sigma.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 6 gespeichert: agn_plot6_mbh_sigma.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 – Radiospektrum und Synchrotron-Emission
# ─────────────────────────────────────────────────────────────────────────────
def plot_radio_spectrum():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Radio-Kontinuumemission und Synchrotron-Physik in VV 340a',
                 fontsize=14, fontweight='bold')

    # --- Spektralindex ---
    ax = axes[0]
    nu = np.logspace(8, 11, 300)  # Hz
    nu_1 = 1.5e9  # Hz
    nu_6 = 6e9    # Hz
    S_1_E = 12.4e-3   # Jy, Ostjet bei 1.5 GHz
    S_6_E = 1.69e-3   # Jy, Ostjet bei 6 GHz
    S_1_W = 8.85e-3
    S_6_W = 1.14e-3

    alpha_E = np.log10(S_1_E / S_6_E) / np.log10(nu_6 / nu_1)
    alpha_W = np.log10(S_1_W / S_6_W) / np.log10(nu_6 / nu_1)

    S_E = S_1_E * (nu / nu_1)**(-alpha_E)
    S_W = S_1_W * (nu / nu_1)**(-alpha_W)

    ax.loglog(nu / 1e9, S_E * 1e3, color=RED, lw=2.5,
              label=fr'Ostjet ($\alpha_E = -{alpha_E:.2f}$)')
    ax.loglog(nu / 1e9, S_W * 1e3, color=BLUE, lw=2.5,
              label=fr'Westjet ($\alpha_W = -{alpha_W:.2f}$)')
    ax.scatter([nu_1 / 1e9, nu_6 / 1e9], [S_1_E * 1e3, S_6_E * 1e3],
               color=RED, s=100, zorder=5)
    ax.scatter([nu_1 / 1e9, nu_6 / 1e9], [S_1_W * 1e3, S_6_W * 1e3],
               color=BLUE, s=100, marker='s', zorder=5)
    ax.errorbar([nu_1/1e9], [S_1_E*1e3], yerr=1.27, fmt='none', color=RED, capsize=5)
    ax.errorbar([nu_6/1e9], [S_6_E*1e3], yerr=0.170, fmt='none', color=RED, capsize=5)
    ax.errorbar([nu_1/1e9], [S_1_W*1e3], yerr=0.940, fmt='none', color=BLUE, capsize=5)
    ax.errorbar([nu_6/1e9], [S_6_W*1e3], yerr=0.114, fmt='none', color=BLUE, capsize=5)
    ax.set_xlabel('Frequenz [GHz]')
    ax.set_ylabel('Flussdichte [mJy]')
    ax.set_title(f'Radio-Spektrum der Jets\n(Synchrotron, $\\alpha \\approx -1.4$)')
    ax.legend()
    ax.set_xlim(0.1, 100)

    # --- Synchrotron-Kühlzeit ---
    ax = axes[1]
    B_range = np.logspace(-2, 1, 300)  # µG
    # Synchrotron-Kühlzeit: t_cool ~ 8.35e9 / (B^2 * gamma) yr (vereinfacht)
    # Für relativistische Elektronen bei 6 GHz:
    # gamma ~ (nu / nu_B)^0.5, nu_B = eB/(2pi m_e c)
    e = 4.803e-10  # esu
    m_e = 9.109e-28  # g
    c = 3e10  # cm/s
    B_G = B_range * 1e-6  # Gauss
    nu_B = e * B_G / (2 * np.pi * m_e * c)
    gamma_6GHz = np.sqrt(6e9 / nu_B)
    # t_cool = (6 pi m_e c) / (sigma_T B^2 gamma)
    sigma_T = 6.652e-25  # cm^2
    t_cool_s = (6 * np.pi * m_e * c) / (sigma_T * B_G**2 * gamma_6GHz)
    t_cool_Myr = t_cool_s / (3.156e7 * 1e6)

    ax.loglog(B_range, t_cool_Myr, color=PURPLE, lw=2.5,
              label=r'$t_\mathrm{cool}$ (6 GHz Elektronen)')
    ax.axhline(0.82, color=GREEN, ls='--', lw=2, label=r'$P_\mathrm{prec} = 0.82$ Myr')
    ax.fill_between(B_range, 0.82 * 0.33, 0.82 * 1.67, alpha=0.2, color=GREEN)
    B_typical = 30  # µG, typischer Wert für ULIRGs
    ax.axvline(B_typical, color=ORANGE, ls=':', lw=2, label=fr'$B \approx {B_typical}\,\mu$G (typisch)')
    ax.set_xlabel(r'Magnetfeldstärke $B$ [µG]')
    ax.set_ylabel(r'Synchrotron-Kühlzeit [Myr]')
    ax.set_title('Synchrotron-Kühlzeit\nvs. Präzessionsperiode')
    ax.legend(fontsize=8)

    # --- Helicale Jet-Morphologie (S-Form) ---
    ax = axes[2]
    # Simulation der S-förmigen 6 GHz Morphologie
    t_jet = np.linspace(0, 4 * np.pi, 800)
    r_jet = t_jet / (4 * np.pi) * 18
    i = np.radians(71)
    psi = np.radians(33)
    phi_p = np.radians(14)

    # Helikale Bahn
    x3d = r_jet * np.cos(t_jet) * np.sin(phi_p)
    y3d = r_jet * np.sin(t_jet) * np.sin(phi_p)
    z3d = r_jet * np.cos(phi_p)

    # Projektion auf Himmelsebene
    xsky = x3d * np.cos(psi) - y3d * np.sin(psi)
    ysky = (x3d * np.sin(psi) + y3d * np.cos(psi)) * np.cos(i) + z3d * np.sin(i)

    # Intensität (simuliert durch Partikelanzahl in Bins)
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Farbkodierung nach Intensität (Gauss-Smoothing simuliert)
    intensity = np.exp(-0.05 * r_jet)

    sc = ax.scatter(xsky, ysky, c=intensity, cmap='hot', s=3, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Relative Radiointensität')
    ax.scatter([-xsky], [-ysky], c=intensity, cmap='cool', s=3, alpha=0.7)
    ax.scatter([0], [0], color='gold', s=200, zorder=6, label='AGN-Kern')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_title('Modell: S-förmige 6-GHz-\nRadiomorphologie (präzessierender Jet)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('agn_plot7_radio.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 7 gespeichert: agn_plot7_radio.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 – CO-Molekulargas: Scheibe vs. Ausfluss
# ─────────────────────────────────────────────────────────────────────────────
def plot_molecular_gas():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Molekulares Gas in VV 340a: CO(2-1)-Beobachtungen mit ALMA',
                 fontsize=14, fontweight='bold')

    # --- Moment 0: Integrierte Intensität (schematisch) ---
    ax = axes[0]
    x = np.linspace(-20, 20, 300)
    y = np.linspace(-15, 15, 300)
    X, Y = np.meshgrid(x, y)
    # Scheibe: elongierte Gauss
    disk = np.exp(-(X**2 / 60 + Y**2 / 8)) * 2
    # Kernkontinuum
    core = np.exp(-(X**2 + Y**2) / 1) * 0.5
    moment0 = disk + core
    im = ax.contourf(X, Y, moment0, levels=20, cmap='YlOrRd')
    ax.contour(X, Y, moment0, levels=6, colors='black', linewidths=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label='Jy km/s beam$^{-1}$')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_title('CO(2-1) Moment 0\n(Integrierte Intensität, ALMA-Modell)')
    ax.set_aspect('equal')

    # --- Moment 1: Geschwindigkeitskarte ---
    ax = axes[1]
    v_map = 300 * X / (np.sqrt(X**2 + Y**2) + 3) * np.exp(-(X**2 / 60 + Y**2 / 8))
    im2 = ax.contourf(X, Y, v_map, levels=20, cmap='RdBu_r')
    ax.contour(X, Y, v_map, levels=8, colors='black', linewidths=0.5, alpha=0.5)
    plt.colorbar(im2, ax=ax, label='LOS-Geschwindigkeit [km/s]')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_title('CO(2-1) Moment 1\n(Geschwindigkeitskarte – galaktische Rotation)')
    ax.set_aspect('equal')

    # --- Gasmassen und r21 ---
    ax = axes[2]
    # r21 Verteilung aus Literatur für (U)LIRGs
    np.random.seed(7)
    r21_ulirgs = np.random.normal(0.91, 0.18, 120)
    r21_ulirgs = r21_ulirgs[(r21_ulirgs > 0.3) & (r21_ulirgs < 1.5)]
    r21_vv340a = 0.75

    ax.hist(r21_ulirgs, bins=25, color=BLUE, alpha=0.6, edgecolor='black',
            label='(U)LIRGs (Literatur)')
    ax.axvline(r21_vv340a, color=RED, lw=3, ls='--', label=fr'VV 340a: $r_{{21}} = {r21_vv340a}$')
    ax.axvline(np.median(r21_ulirgs), color=GREEN, lw=2, ls=':',
               label=f'Median: {np.median(r21_ulirgs):.2f}')
    ax.set_xlabel(r'$r_{21} = S_\mathrm{CO(2-1)} / S_\mathrm{CO(1-0)}$')
    ax.set_ylabel('Häufigkeit')
    ax.set_title(r'CO-Linienfluss-Verhältnis $r_{21}$' + '\n(VV 340a vs. (U)LIRG-Stichprobe)')
    ax.legend()

    # Annotation: Gasmasse
    M_H2 = 1.70e10
    ax.text(0.62, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 5,
            fr'$M_{{H_2}} = {M_H2:.2e}\,M_\odot$', fontsize=10, color=RED,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('agn_plot8_co_gas.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 8 gespeichert: agn_plot8_co_gas.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 – Kosmologische Einordnung: Hubble-Entwicklung
# ─────────────────────────────────────────────────────────────────────────────
def plot_cosmological_context():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Kosmologische Einordnung: AGN-Rückkopplung und Galaxienentwicklung',
                 fontsize=14, fontweight='bold')

    # --- Hubble-Kosmologie: D_C(z) ---
    ax = axes[0]
    H0 = 69.3  # km/s/Mpc (aus dem Paper)
    Omega_L = 0.714
    Omega_m = 0.286
    c_kms = 3e5  # km/s

    z_range = np.linspace(0, 2, 400)

    def comoving_dist(z_arr):
        from scipy.integrate import quad
        def integrand(z):
            return 1 / np.sqrt(Omega_m * (1+z)**3 + Omega_L)
        dists = []
        for z in z_arr:
            d, _ = quad(integrand, 0, z)
            dists.append(c_kms / H0 * d)
        return np.array(dists)

    D_C = comoving_dist(z_range)

    # VV 340a: D_C = 157 Mpc -> z ?
    z_vv = 0.034  # Rotverschiebung VV 340a (bei 157 Mpc, WMAP9)

    ax.plot(z_range, D_C, color=BLUE, lw=2.5, label='Kovariante Distanz $D_C(z)$')
    ax.scatter([z_vv], [157], color=RED, s=150, zorder=5, label=f'VV 340a: $z={z_vv}$, $D_C=157$ Mpc')
    ax.axvline(z_vv, color=RED, ls='--', lw=1.5)
    ax.axhline(157, color=RED, ls=':', lw=1.5)
    ax.set_xlabel('Rotverschiebung $z$')
    ax.set_ylabel('Kovariante Distanz $D_C$ [Mpc]')
    ax.set_title(f'Kosmologisches Distanzmodell\n($H_0={H0}$ km/s/Mpc, WMAP9)')
    ax.legend()

    # --- AGN-Rückkopplungsmodell: SFR vs. AGN-Leistung ---
    ax = axes[1]
    L_AGN_range = np.logspace(42, 47, 300)  # erg/s
    # Empirische Relation: SFR ~ L_AGN^0.7 (vereinfacht)
    SFR_corr = 0.1 * (L_AGN_range / 1e44)**0.7  # M_sun/yr
    # Negatives Feedback: SFR_suppressed
    SFR_sup = SFR_corr * np.exp(-(np.log10(L_AGN_range) - 44)**2 / 2)

    ax.loglog(L_AGN_range, SFR_corr, color=BLUE, lw=2.5, ls='--',
              label='Positive Korrelation (beobachtet)')
    ax.loglog(L_AGN_range, SFR_sup, color=RED, lw=2.5,
              label='Mit AGN-Quenching')
    ax.scatter([1.28e44], [10], color=GREEN, s=150, zorder=5,
               label='VV 340a (SFR~10 $M_\odot$/yr)')
    ax.axvline(1.28e44, color=GREEN, ls=':', lw=1.5)
    ax.fill_between(L_AGN_range,
                    SFR_sup * 0.5, SFR_sup * 2,
                    alpha=0.15, color=RED, label='Unsicherheitsbereich')
    ax.set_xlabel(r'AGN-Bolometrische Leuchtkraft $L_\mathrm{AGN,bol}$ [erg/s]')
    ax.set_ylabel(r'Sternentstehungsrate SFR [$M_\odot$/yr]')
    ax.set_title('AGN-Rückkopplung: SFR vs. $L_\mathrm{AGN,bol}$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('agn_plot9_kosmologie.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 9 gespeichert: agn_plot9_kosmologie.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 – Überblick: Multi-Wellenlängen-Vergleich
# ─────────────────────────────────────────────────────────────────────────────
def plot_multiwavelength():
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle('Multi-Wellenlängen-Analyse von VV 340a: Vergleich aller Beobachtungen',
                 fontsize=14, fontweight='bold')

    x = np.linspace(-15, 15, 300)
    y = np.linspace(-12, 20, 300)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    # JWST MIRI [Ne V] 14 µm
    ax = axes[0, 0]
    NeV_NE = np.exp(-((X - 4)**2 + (Y - 4)**2) / 6) * 2
    NeV_SW1 = np.exp(-((X + 3)**2 + (Y + 3)**2) / 5)
    NeV_SW2 = np.exp(-((X + 5)**2 + (Y + 5)**2) / 5)
    NeV_core = np.exp(-r**2 / 0.5) * 3
    NeV_total = NeV_NE + NeV_SW1 + NeV_SW2 + NeV_core
    im = ax.contourf(X, Y, NeV_total, levels=15, cmap='hot')
    plt.colorbar(im, ax=ax, label='Intensität')
    ax.set_title('JWST/MIRI: [Ne V] 14 µm\n(Koronal-Emission, AGN-ionisiert)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')

    # Keck KCWI [O III]
    ax = axes[0, 1]
    OIII_disk = np.exp(-(X**2 / 80 + Y**2 / 12)) * 1.5
    OIII_out = np.exp(-((np.abs(X) - 7)**2 + (np.abs(Y) - 7)**2) / 15) * 0.8
    OIII_total = OIII_disk + OIII_out
    im2 = ax.contourf(X, Y, OIII_total, levels=15, cmap='Blues')
    plt.colorbar(im2, ax=ax, label='Intensität')
    ax.set_title('Keck/KCWI: [O III] 5007 Å\n(optische Emission, AGN-Nebel)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')

    # VLA 6 GHz
    ax = axes[0, 2]
    t_rad = np.linspace(0, 4*np.pi, 400)
    r_rad = t_rad / (4*np.pi) * 12
    xr = r_rad * np.cos(t_rad) * np.sin(np.radians(14))
    yr = r_rad * np.sin(t_rad) * np.sin(np.radians(14)) * np.cos(np.radians(71))
    ax.scatter(xr, yr, c=np.exp(-0.05*r_rad), cmap='hot', s=5, alpha=0.6)
    ax.scatter(-xr, -yr, c=np.exp(-0.05*r_rad), cmap='cool', s=5, alpha=0.6)
    ax.scatter([0], [0], color='gold', s=200, zorder=5)
    ax.set_title('VLA 6 GHz: Radio-Kontinuum\n(S-förmige Jet-Morphologie)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_aspect('equal')

    # ALMA CO(2-1)
    ax = axes[1, 0]
    CO_disk = np.exp(-(X**2 / 50 + Y**2 / 6)) * 2
    im3 = ax.contourf(X, Y, CO_disk, levels=15, cmap='YlGn')
    plt.colorbar(im3, ax=ax, label='Jy km/s')
    ax.set_title('ALMA: CO(2-1) Moment 0\n(Molekulares Gas, keine Ausflusskomponente)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')

    # Chandra X-ray
    ax = axes[1, 1]
    Xray_core = np.exp(-r**2 / 2) * 1.5
    Xray_out = np.exp(-((np.sqrt((np.abs(X) - 5)**2 + (np.abs(Y) - 5)**2))**2) / 10) * 0.4
    Xray_total = Xray_core + Xray_out
    im4 = ax.contourf(X, Y, Xray_total, levels=15, cmap='Purples')
    plt.colorbar(im4, ax=ax, label='Anzahl Photonen')
    ax.set_title('Chandra ACIS: Weich-X-ray (0.5–2 keV)\n(Thermisches Plasma, Jet-Schocks)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')

    # Überlagerung aller Wellenlängen
    ax = axes[1, 2]
    ax.contour(X, Y, NeV_total,    levels=3, colors=[RED],    linewidths=1.5, linestyles='-')
    ax.contour(X, Y, OIII_total,   levels=3, colors=[BLUE],   linewidths=1.5, linestyles='--')
    ax.contour(X, Y, CO_disk,      levels=3, colors=[GREEN],  linewidths=1.5, linestyles=':')
    ax.contour(X, Y, Xray_total,   levels=2, colors=[PURPLE], linewidths=1.5, linestyles='-.')

    # Jet-Achse
    jet_len = 14
    ax.annotate('', xy=(jet_len * np.cos(np.radians(45)), jet_len * np.sin(np.radians(45))),
                xytext=(-jet_len * np.cos(np.radians(45)), -jet_len * np.sin(np.radians(45))),
                arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
    ax.text(7, 8, 'Jet-Achse\nPA=45°', fontsize=9, color=ORANGE)

    handles = [
        mpatches.Patch(color=RED, label='[Ne V] 14 µm (JWST)'),
        mpatches.Patch(color=BLUE, label='[O III] 5007 Å (Keck)'),
        mpatches.Patch(color=GREEN, label='CO(2-1) (ALMA)'),
        mpatches.Patch(color=PURPLE, label='X-ray 0.5–2 keV (Chandra)'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper left')
    ax.scatter([0], [0], color='gold', s=200, zorder=6)
    ax.set_title('Multi-Wellenlängen-Überlagerung\n(Konturlinien aller Teleskope)')
    ax.set_xlabel('RA-Offset [arcsec]')
    ax.set_ylabel('Dec-Offset [arcsec]')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-12, 20)

    plt.tight_layout()
    plt.savefig('agn_plot10_multiwavelength.pdf', bbox_inches='tight')
    plt.close()
    print("Plot 10 gespeichert: agn_plot10_multiwavelength.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generiere alle AGN-Plots...")
    plot_eddington_ratio()
    plot_precessing_jet()
    plot_bicone_outflow()
    plot_ionization_bpt()
    plot_energy_budget()
    plot_mbh_sigma()
    plot_radio_spectrum()
    plot_molecular_gas()
    plot_cosmological_context()
    plot_multiwavelength()
    print("\nAlle 10 Plots erfolgreich generiert!")