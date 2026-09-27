#!/usr/bin/env python3
"""
Plots für Kapitel: Der Bereich des Nichts der Schwarzen Löcher – Quantenphysik an der Grenze des Seins
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as pat
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
import os

outdir = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# --------------------------------------------------------------------------
# Physikalische Konstanten
# --------------------------------------------------------------------------
c   = 3.0e8          # m/s
G   = 6.674e-11      # m³/(kg·s²)
hbar= 1.055e-34      # J·s
k_B = 1.381e-23      # J/K
M_sun = 1.989e30     # kg
h_planck = 6.626e-34 # J·s

# Planck-Einheiten
l_P = np.sqrt(hbar * G / c**3)  # ~1.616e-35 m
t_P = l_P / c                    # ~5.39e-44 s
m_P = np.sqrt(hbar * c / G)     # ~2.177e-8 kg
T_P = m_P * c**2 / k_B          # ~1.417e32 K


# --------------------------------------------------------------------------
# Plot 1: Hawking-Strahlung – Temperatur und Lebensdauer als Funktion der Masse
# --------------------------------------------------------------------------
def plot_hawking_strahlung():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hawking-Strahlung: Quantenphysik an der Grenze des Schwarzen Lochs',
                 fontsize=13, fontweight='bold', y=1.02)

    # --- Linkes Panel: Hawking-Temperatur ---
    ax = axes[0]
    M_arr = np.logspace(-10, 40, 1000) * M_sun  # in kg
    T_H = (hbar * c**3) / (8 * np.pi * G * M_arr * k_B)

    ax.loglog(M_arr / M_sun, T_H, color='crimson', lw=2.5, label='$T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$')

    # Referenzpunkte
    masses = {
        'Atomkern\n$(10^{-27}\\,\\mathrm{kg})$': 1e-27 / M_sun,
        'Planck-Masse\n$(m_P)$': m_P / M_sun,
        'Stern-BH\n$(10\\,M_\\odot)$': 10.0,
        'Galakt. BH\n$(10^6\\,M_\\odot)$': 1e6,
        'SGr A*\n$(4{\\times}10^6\\,M_\\odot)$': 4e6,
    }
    colors_ref = ['navy', 'darkgreen', 'darkorange', 'purple', 'brown']
    for (label, mass_frac), col in zip(masses.items(), colors_ref):
        mass_kg = mass_frac * M_sun
        T_val = (hbar * c**3) / (8 * np.pi * G * mass_kg * k_B)
        ax.plot(mass_frac, T_val, 'o', color=col, markersize=8, zorder=5)
        ax.annotate(label, xy=(mass_frac, T_val),
                    xytext=(mass_frac * 10, T_val * 3),
                    fontsize=7.5, color=col, ha='left',
                    arrowprops=dict(arrowstyle='->', color=col, lw=0.8))

    # Referenz-Temperaturbänder
    ax.axhspan(2.73, 2.75, alpha=0.25, color='cyan', label='CMB-Temperatur (2,73 K)')
    ax.axhspan(3e3, 6e3, alpha=0.15, color='yellow', label='Oberfläche Sonne (~5778 K)')

    ax.set_xlabel('Masse $M$ [$M_\\odot$]')
    ax.set_ylabel('Hawking-Temperatur $T_H$ [K]')
    ax.set_title('Hawking-Temperatur: Je kleiner das\nSchwarze Loch, desto heißer')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([1e-38, 1e10])
    ax.set_ylim([1e-20, 1e50])

    # --- Rechtes Panel: Wellenlänge der Hawking-Strahlung ---
    ax2 = axes[1]
    lambda_H = (8 * np.pi**2 * G * M_arr * k_B) / (hbar * c**4) * (hbar * c)
    # Wellenlänge der Hawking-Strahlung ~ proportional zum Schwarzschild-Radius
    r_s = 2 * G * M_arr / c**2
    lambda_H_proper = 4 * np.pi * r_s  # Wellenlänge ~ 4π r_s

    ax2.loglog(M_arr / M_sun, lambda_H_proper * 1e9, color='steelblue', lw=2.5,
               label='$\\lambda_H = 4\\pi r_s \\propto M$')
    ax2.loglog(M_arr / M_sun, np.ones_like(M_arr) * l_P * 1e9, 'g--', lw=1.5,
               label=f'Planck-Länge $\\ell_P \\approx 10^{{-26}}$ nm')

    # sichtbares Licht Band
    ax2.axhspan(380e-9 * 1e9, 780e-9 * 1e9, alpha=0.2, color='orange',
                label='Sichtbares Licht (380–780 nm)')

    ax2.set_xlabel('Masse $M$ [$M_\\odot$]')
    ax2.set_ylabel('Hawking-Wellenlänge $\\lambda_H$ [nm]')
    ax2.set_title('Wellenlänge der Hawking-Strahlung:\nProportional zum Schwarzschild-Radius')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_hawking_strahlung.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_hawking_strahlung.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 2: Planck-Skala-Vergleich – Größenordnungen der Quantenwirkung
# --------------------------------------------------------------------------
def plot_planck_skala():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim([-67, 30])
    ax.set_ylim([-1.5, 1.5])
    ax.axis('off')
    ax.set_title('Skalenvergleich: Von der Planck-Länge bis zur kosmischen Struktur –\n'
                 'Quantenphysik dominiert das Nichts der Schwarzen Löcher',
                 fontsize=13, fontweight='bold', pad=12)

    # Hauptachse
    ax.annotate('', xy=(28, 0), xytext=(-66, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(28.5, 0, 'lg(Länge/m)', va='center', fontsize=9)
    for x in range(-65, 28, 5):
        ax.plot([x, x], [-0.05, 0.05], 'k-', lw=1)
        ax.text(x, -0.13, str(x), ha='center', va='top', fontsize=7.5)

    objects = [
        (-35, 'Planck-Länge\n$\\ell_P \\approx 10^{-35}$ m\n(Grenze der\nQuantengravitation)', 'darkred', 'bottom', 0.5),
        (-30, 'Quanten-BH-Singularität\n$r_s^{\\min} \\approx 10^{-30}$ m\n(Planck-BH)', 'red', 'bottom', 0.85),
        (-15, 'Atomkern\n$r \\approx 10^{-15}$ m\n(Quantenphysik aktiv)', 'darkorange', 'bottom', 0.5),
        (-10, 'Atom\n$r \\approx 10^{-10}$ m\n(Quantenmechanik)', 'orange', 'bottom', 0.75),
        (0, 'Mensch\n$\\sim 1$ m\n(Klass. Physik)', 'mediumblue', 'bottom', 0.55),
        (7, 'Erde\n$R_\\oplus \\approx 10^7$ m\n(Klass. Physik,\nQM sehr schwach)', 'blue', 'bottom', 0.7),
        (11, 'Schwarzes Loch\n$r_s(10\\,M_\\odot)\\approx 10^{4}$ m\n(Übergangszone)', 'purple', 'bottom', 0.85),
        (20, 'Galaxie\n$\\sim 10^{22}$ m\n(Klass. GR)', 'darkgreen', 'top', 0.7),
        (26, 'Universum\n$d_H \\approx 10^{26}$ m\n(Kosmologie)', 'black', 'top', 0.55),
    ]

    for (logx, label, col, valign, ypos) in objects:
        yval = ypos if valign == 'bottom' else -ypos
        ax.plot([logx, logx], [0, yval * 0.9], color=col, lw=1.5, ls='--', alpha=0.7)
        ax.plot(logx, 0, 'o', color=col, markersize=10, zorder=5)
        va = 'bottom' if valign == 'bottom' else 'top'
        ax.text(logx, yval * 0.92, label, ha='center', va=va, fontsize=7.5,
                color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=col, lw=0.5))

    # Quantenbereich färben
    from matplotlib.patches import Rectangle as Rect
    ax.add_patch(Rect((-66, -1.35), 20, 2.7, alpha=0.12, facecolor='red', edgecolor='none'))
    ax.text(-56, 1.2, 'Quantendominierter Bereich\n(Nichts der Schwarzen Löcher)', color='darkred',
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red', lw=1.5))

    ax.add_patch(Rect((0, -1.35), 27, 2.7, alpha=0.07, facecolor='steelblue', edgecolor='none'))
    ax.text(13, -1.2, 'Klassisch-physikalischer Bereich (Erde, Galaxien)', color='steelblue',
            fontsize=9, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_planck_skala_vergleich.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_planck_skala_vergleich.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 3: Quantenfluktuation und Unschärferelation nahe der Singularität
# --------------------------------------------------------------------------
def plot_unschaerfe_singularitaet():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Heisenbergsche Unschärferelation und Quantenfluktuation:\n'
                 'Dominanz der Quantenphysik nahe der Singularität',
                 fontsize=12, fontweight='bold', y=1.02)

    # --- Links: Energiefluktuationen als Funktion des Abstands ---
    ax = axes[0]
    r_arr = np.logspace(-35, 5, 1000)  # in Metern (Abstand zur Singularität)

    # Unschärfe Δx ~ r, dann Δp >= ħ/(2r), Energie E >= (Δp)²/(2m) ~ ħ²/(8m r²)
    # Für Photon: E ~ ħc/r (Energieunschärfe)
    E_quantum = hbar * c / r_arr  # Energieunschärfe [J]
    E_quantum_eV = E_quantum / 1.602e-19  # in eV

    # Planck-Energie
    E_planck = m_P * c**2
    E_planck_eV = E_planck / 1.602e-19

    ax.loglog(r_arr, E_quantum_eV, color='crimson', lw=2.5,
              label='$\\Delta E \\approx \\hbar c / r$ (Quantenfluktuation)')
    ax.axhline(E_planck_eV, color='darkred', ls='--', lw=2, label=f'Planck-Energie $E_P \\approx 10^{{28}}$ eV')
    ax.axhline(0.511e6, color='orange', ls='--', lw=1.5, label='Elektron Ruheenergie (511 keV)')
    ax.axhline(938e6, color='purple', ls='--', lw=1.5, label='Proton Ruheenergie (938 MeV)')
    ax.axhline(1e12, color='steelblue', ls='--', lw=1.5, label='TeV (LHC-Bereich)')

    ax.axvline(l_P, color='darkred', ls=':', lw=2, alpha=0.7, label=f'Planck-Länge $\\ell_P$')
    ax.axvline(1e-15, color='gray', ls=':', lw=1.5, alpha=0.7, label='Atomkern-Radius')

    ax.fill_between(r_arr[r_arr < l_P * 10], E_quantum_eV[r_arr < l_P * 10],
                    1e-5, alpha=0.2, color='red', label='Quantengravitations-Regime')

    ax.set_xlabel('Abstand zur Singularität $r$ [m]')
    ax.set_ylabel('Energieunschärfe $\\Delta E$ [eV]')
    ax.set_title('Energieunschärfe divergiert zur Singularität:\n'
                 '$\\Delta E \\cdot \\Delta t \\geq \\hbar/2$ → Quanten dominieren')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([r_arr[0], r_arr[-1]])
    ax.set_ylim([1e-10, 1e40])

    # --- Rechts: Quantenmechanische Wirkung vs klassische Wirkung ---
    ax2 = axes[1]
    r_arr2 = np.logspace(-35, 12, 1000)

    # Klassische Wirkung ~ E * t ~ E * r/c ~ Mc²r/c = Mc*r für ein massives Objekt
    # Wir vergleichen S_klass = p*r mit hbar
    # Für BH: p = Mc, r = r_s = 2GM/c²
    # S_klass/hbar = Mc * r / hbar = Mc² * r / (hbar*c) = E*r/(hbar*c)
    # Quantum correction: wenn S<hbar, Quantum dominiert

    M_ref = 10 * M_sun
    S_klass = M_ref * c * r_arr2 / hbar  # normiert auf hbar

    ax2.loglog(r_arr2, S_klass, color='steelblue', lw=2.5,
               label='Klassische Wirkung $S_{\\mathrm{klass}}/\\hbar = Mc\\cdot r/\\hbar$')
    ax2.axhline(1, color='red', lw=2.5, ls='--',
                label='$S = \\hbar$ (Quanten-Grenze)')
    ax2.axhline(1e3, color='orange', lw=1.5, ls='--', label='$S \\sim 10^3 \\hbar$ (schwach quantal)')

    r_quantum_dom = hbar / (M_ref * c)
    ax2.axvline(r_quantum_dom, color='darkred', lw=2, ls=':',
                label=f'$r_{{\\mathrm{{quant}}}} = \\hbar/(Mc) \\approx {r_quantum_dom:.1e}$ m')

    ax2.fill_between(r_arr2[r_arr2 < r_quantum_dom * 1000],
                     np.ones(np.sum(r_arr2 < r_quantum_dom * 1000)) * 1e-5,
                     S_klass[r_arr2 < r_quantum_dom * 1000],
                     alpha=0.2, color='red')
    ax2.fill_betweenx([1, 1e60], r_quantum_dom * 1000, 1e12,
                      alpha=0.1, color='steelblue')

    ax2.text(1e-20, 0.01, 'Quantum-\ndominiert', fontsize=9, color='darkred',
             fontweight='bold', ha='center')
    ax2.text(1e3, 1e40, 'Klassisch\ndominiert', fontsize=9, color='steelblue',
             fontweight='bold', ha='center')

    ax2.set_xlabel('Abstand $r$ [m]')
    ax2.set_ylabel('Normierte Wirkung $S/\\hbar$')
    ax2.set_title('Übergang: Klassische → Quantenphysik\n'
                  'Wirkungsprinzip bestimmt den dominierenden Bereich')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([r_arr2[0], r_arr2[-1]])
    ax2.set_ylim([1e-10, 1e60])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_unschaerfe_singularitaet.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_unschaerfe_singularitaet.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 4: Vergleich Quantenwirkung Erde vs. Nichts der Schwarzen Löcher
# --------------------------------------------------------------------------
def plot_quanten_wirkung_vergleich():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quantenwirkung: Erde vs. Bereich des Nichts der Schwarzen Löcher',
                 fontsize=13, fontweight='bold', y=1.02)

    # --- Links: Radar-Plot als Balkendiagramm ---
    ax = axes[0]
    categories = [
        'Quantenfluktuation\n(rel. zu $\\hbar$)',
        'Vakuumenergie\n(rel. Planck)',
        'Wellenlängen-\n/Ortsunschärfe',
        'Dekohärenz-\nschutz',
        'Verschränkungs-\nstärke',
        'Raumzeit-\nKrümmung (rel.)',
    ]
    erde_values = [0.08, 0.05, 0.15, 0.2, 0.12, 0.02]
    nichts_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, erde_values, width, label='Erde (makroskop. Skala)',
                   color='steelblue', alpha=0.85, edgecolor='navy', lw=1)
    bars2 = ax.bar(x + width/2, nichts_values, width, label='Nichts des Schwarzen Lochs\n(Planck-Skala)',
                   color='crimson', alpha=0.85, edgecolor='darkred', lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, ha='center')
    ax.set_ylabel('Relative Quantenwirksamkeit (normiert auf Maximum = 1)', fontsize=9)
    ax.set_title('Quantenwirksamkeit im Vergleich:\nauf der Erde schwach, im Nichts maximal')
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.25])
    ax.axhline(1.0, color='red', ls='--', lw=1.5, alpha=0.6, label='Maximum')
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=7.5, color='navy')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=7.5, color='darkred')

    # --- Rechts: Korrespondenzprinzip – klassisch/quantal als Funktion der Wirkung ---
    ax2 = axes[1]

    S_over_hbar = np.logspace(-2, 12, 1000)
    # Quantenanteil ~ 1/(1 + S/ħ): einfache Interpolation
    quantum_fraction = 1.0 / (1.0 + S_over_hbar)
    classical_fraction = 1.0 - quantum_fraction

    ax2.semilogx(S_over_hbar, quantum_fraction * 100, color='crimson', lw=2.5,
                 label='Quantenanteil (%)')
    ax2.semilogx(S_over_hbar, classical_fraction * 100, color='steelblue', lw=2.5,
                 label='Klassischer Anteil (%)')
    ax2.axhline(50, color='gray', ls=':', lw=1.5, alpha=0.6)

    # Typische Bereiche
    regionen = [
        (0.01, 1, 'Quanten-\nBereich\n(Nichts)', 'crimson'),
        (1, 100, 'Übergangs-\nBereich', 'purple'),
        (1e4, 1e10, 'Klassischer\nBereich\n(Erde)', 'steelblue'),
    ]
    colors_r = ['mistyrose', '#f0e8ff', 'lightcyan']
    for (x1, x2, label, col), fc in zip(regionen, colors_r):
        ax2.axvspan(x1, x2, alpha=0.25, color=fc)
        ax2.text(np.sqrt(x1 * x2), 85, label, ha='center', fontsize=8.5,
                 color=col, fontweight='bold')

    # Markierungen für Erde und Schwarzes Loch
    ax2.axvline(1e10, color='steelblue', ls='--', lw=1.5, label='Erde: $S \\sim 10^{10}\\,\\hbar$')
    ax2.axvline(0.1, color='crimson', ls='--', lw=1.5, label='BH-Singularität: $S \\sim \\hbar$')

    ax2.set_xlabel('Wirkung $S / \\hbar$ (Korrespondenzprinzip)')
    ax2.set_ylabel('Anteil (%)')
    ax2.set_title('Korrespondenzprinzip: Quantensysteme werden\nklassisch für $S \\gg \\hbar$')
    ax2.legend(fontsize=8.5, loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_quanten_wirkung_vergleich.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_quanten_wirkung_vergleich.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 5: Lebensbedingungen – warum Leben im Nichts unmöglich ist
# --------------------------------------------------------------------------
def plot_lebensbedingungen():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Absolute Lebensfeindlichkeit des Nichts der Schwarzen Löcher:\n'
                 'Thermodynamische und quantenphysikalische Extreme',
                 fontsize=13, fontweight='bold', y=1.01)

    r_arr = np.logspace(-35, 20, 1000)  # Abstand zur Singularität in m

    # Hawking-Temperatur als Funktion (~T_H ~ 1/(BH-Masse) aber hier als Temperatur um Singularität)
    M_bh = 10 * M_sun  # 10 Sonnenmassen
    T_H = (hbar * c**3) / (8 * np.pi * G * M_bh * k_B)
    r_s = 2 * G * M_bh / c**2  # Schwarzschild-Radius

    # Strahlungstemperatur des Quantenvakuums als Funktion des Abstands zur Singularität
    # Unruh-Effekt: T_Unruh = a*hbar/(2*pi*c*k_B)
    # Beschleunigung nahe Ereignishorizont: a = c^4/(4GM) * ... ~ c^2/r_s für r~r_s
    a_arr = c**2 / np.maximum(r_arr, r_s)
    T_unruh = a_arr * hbar / (2 * np.pi * c * k_B)

    # --- Panel 1: Temperatur ---
    ax = axes[0, 0]
    ax.loglog(r_arr / r_s, T_unruh, color='crimson', lw=2.5, label='Unruh-Temperatur $T_{\\mathrm{Unruh}}$')
    ax.axhline(T_H, color='orange', lw=2, ls='--', label=f'Hawking-Temperatur $T_H \\approx {T_H:.1e}$ K')
    ax.axhline(T_P, color='darkred', lw=2, ls='-.', label=f'Planck-Temperatur $T_P \\approx 10^{{32}}$ K')
    ax.axhline(273, color='steelblue', lw=1.5, ls=':', label='Gefrierpunkt (273 K)')
    ax.axhline(310, color='green', lw=1.5, ls=':', label='Körpertemperatur (310 K)')

    ax.fill_between(r_arr / r_s, T_unruh, 1e-5,
                    where=(r_arr / r_s < 1.001), alpha=0.2, color='red')

    ax.set_xlabel('Abstand $r / r_s$ (normiert auf Schwarzschild-Radius)')
    ax.set_ylabel('Temperatur [K]')
    ax.set_title('Unruh-Temperatur nahe Ereignishorizont')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-10, 1e38])
    ax.set_xlim([0.9, 1e8])

    # --- Panel 2: Vakuumenergiediichte ---
    ax2 = axes[0, 1]
    # Casimir-Energiedichte ~ hbar*c/r^4
    rho_casimir = hbar * c / np.maximum(r_arr, l_P)**4  # J/m³
    rho_planck = hbar * c / l_P**4
    # Normale Atomphysik-Energiedichte
    rho_atom = 1e15  # J/m³ (Nuclear binding energy density)
    rho_Erde = 1e3   # J/m³ (chemische Energie Luft ~1 kJ/m³)

    ax2.loglog(r_arr, rho_casimir, color='purple', lw=2.5,
               label='Quantenvakuum-Energiedichte\n$\\rho_{\\mathrm{vak}} \\propto \\hbar c / r^4$')
    ax2.axhline(rho_planck, color='darkred', lw=2, ls='--',
                label=f'Planck-Energiedichte $\\sim 10^{{113}}$ J/m³')
    ax2.axhline(rho_atom, color='orange', lw=1.5, ls=':',
                label='Atomkern-Energiedichte ($10^{{15}}$ J/m³)')
    ax2.axhline(rho_Erde, color='steelblue', lw=1.5, ls=':',
                label='Chemische Energie (Erde)')

    ax2.fill_between(r_arr, rho_casimir, 1, where=(r_arr < 1e-15), alpha=0.2, color='red')

    ax2.set_xlabel('Abstand zur Singularität $r$ [m]')
    ax2.set_ylabel('Energiedichte [J/m³]')
    ax2.set_title('Quantenvakuum-Energiedichte:\nDivergenz zur Singularität')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([r_arr[0], r_arr[-1]])
    ax2.set_ylim([1e-30, 1e120])

    # --- Panel 3: Zeitdilatation – Zeit steht still ---
    ax3 = axes[1, 0]
    r_range = np.linspace(1.001, 20, 1000)  # r/r_s
    time_dilation = np.sqrt(1 - 1.0 / r_range)  # dt_eigen/dt_fern

    ax3.plot(r_range, time_dilation, color='darkgreen', lw=2.5,
             label='Eigenzeit / Fernzeit = $\\sqrt{1 - r_s/r}$')
    ax3.axhline(1.0, color='gray', ls='--', lw=1.5, alpha=0.7, label='Uhren gleich laufen')
    ax3.axvline(1.0, color='red', ls=':', lw=2, alpha=0.7, label='Ereignishorizont $r = r_s$')

    ax3.fill_between(r_range, 0, time_dilation, where=(time_dilation < 0.5),
                     alpha=0.15, color='red', label='Extreme Zeitdilatation')

    # Lebensrelevante Bereiche
    ax3.annotate('Ereignishorizont:\nZeit → 0\n(Biologische Prozesse\nizfroren)', xy=(1.05, 0.05),
                 xytext=(5, 0.15), fontsize=8.5, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax3.annotate('Erde: $r \\gg r_s$\nZeit normal\n(Leben möglich)', xy=(18, 0.997),
                 xytext=(12, 0.8), fontsize=8.5, color='darkgreen', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkgreen'))

    ax3.set_xlabel('Normierter Abstand $r/r_s$')
    ax3.set_ylabel('Relative Eigenzeit (Zeitdilatationsfaktor)')
    ax3.set_title('Gravitationszeitdilatation:\nZeit kommt zum Erliegen am Ereignishorizont')
    ax3.legend(fontsize=8.5, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Überlebenschancen-Plot ---
    ax4 = axes[1, 1]
    bedingungen = [
        'Temperatur\n($T > 10^{32}$ K)', 'Strahlungs-\ndruck', 'Zeitdilatation\n→ 0',
        'Gravit.-\nGradient', 'Quanten-\nFluktuation', 'Vakuum-\nenergie',
        'Kausal. Isolation\n(Informations-\nverlust)'
    ]
    lebbarkeit_nichts = [0, 0, 0, 0, 0, 0, 0]  # alles 0 im Nichts
    lebbarkeit_erde = [0.98, 0.99, 0.9999, 0.97, 0.85, 0.80, 1.0]

    x = np.arange(len(bedingungen))
    width = 0.35

    ax4.bar(x - width/2, lebbarkeit_erde, width, label='Erde',
            color='mediumseagreen', alpha=0.85, edgecolor='darkgreen', lw=1)
    ax4.bar(x + width/2, lebbarkeit_nichts, width, label='Nichts des Schwarzen Lochs',
            color='crimson', alpha=0.85, edgecolor='darkred', lw=1)

    ax4.set_xticks(x)
    ax4.set_xticklabels(bedingungen, fontsize=7.5)
    ax4.set_ylabel('Lebbarkeitsindex (1 = kompatibel, 0 = lethal)')
    ax4.set_title('Lebbarkeitsindex: Leben im Nichts\nist nach allen Kriterien unmöglich')
    ax4.legend(fontsize=9)
    ax4.set_ylim([0, 1.3])
    ax4.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax4.set_facecolor('#f8f8f8')
    ax4.grid(True, alpha=0.3, axis='y')

    ax4.text(3, 1.15, '☠ Im Nichts des Schwarzen Lochs ist Leben nach ALLEN 7 Kriterien absolut unmöglich ☠',
             ha='center', va='center', fontsize=9, color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_lebensbedingungen.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_lebensbedingungen.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 6: Der kosmische Quader – Schematische Darstellung des Universumrandes
# --------------------------------------------------------------------------
def plot_kosmischer_quader():
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('Das Ende des Universums: Schwarze Löcher treffen auf den kosmischen Quader –\n'
                 'Raumzeittopologie an der äußersten Grenze des Seins',
                 fontsize=13, fontweight='bold', y=1.02)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

    # --- Linkes Panel: 3D-Darstellung des kosmischen Quaders ---
    ax = fig.add_subplot(gs[0], projection='3d')

    # Quader-Ecken
    d = 1.0
    # 8 Ecken des Quaders
    corners = np.array([
        [0, 0, 0], [2*d, 0, 0], [2*d, 2*d, 0], [0, 2*d, 0],
        [0, 0, 2*d], [2*d, 0, 2*d], [2*d, 2*d, 2*d], [0, 2*d, 2*d]
    ])

    # Flächen des Quaders
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    face_colors = ['#E8E8FF', '#E8E8FF', '#D8D8EE', '#D8D8EE', '#C8C8DD', '#C8C8DD']

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as P3DC
    poly = P3DC([[corners[i] for i in f] for f in faces],
                facecolors=face_colors, edgecolors='navy', alpha=0.25, linewidth=1.5)
    ax.add_collection3d(poly)

    # Schwarze Löcher an den Ecken und Kanten des Quaders
    bh_positions = [
        [0.3, 0.3, 0.3], [1.7, 0.3, 0.3], [0.3, 1.7, 0.3], [1.7, 1.7, 0.3],
        [0.3, 0.3, 1.7], [1.7, 0.3, 1.7], [0.3, 1.7, 1.7], [1.7, 1.7, 1.7],
        [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [1.0, 1.7, 1.0], [1.0, 1.0, 1.7],
    ]
    for pos in bh_positions:
        ax.scatter(*pos, color='black', s=60, zorder=5)
        # Akkretionsscheibe-ähnliche Kreise
        theta = np.linspace(0, 2*np.pi, 30)
        r_disk = 0.15
        ax.plot(pos[0] + r_disk*np.cos(theta), pos[1] + r_disk*np.sin(theta),
                pos[2], color='orange', lw=1.0, alpha=0.7)

    # Galaxienfilamente
    np.random.seed(42)
    for _ in range(8):
        pts = np.random.uniform(0.2, 1.8, (10, 3))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='lightcoral', lw=0.8, alpha=0.5)

    ax.set_xlim([0, 2*d])
    ax.set_ylim([0, 2*d])
    ax.set_zlim([0, 2*d])
    ax.set_xlabel('$x$ [kosmisch]')
    ax.set_ylabel('$y$ [kosmisch]')
    ax.set_zlabel('$z$ [kosmisch]')
    ax.set_title('3D: Schwarze Löcher (schwarz)\nam Rand des kosmischen Quaders (blau)')
    ax.view_init(elev=20, azim=35)

    # --- Rechtes Panel: Schema des kosmischen Quaders in 2D ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_aspect('equal')
    ax2.set_xlim([-0.3, 8.3])
    ax2.set_ylim([-0.3, 8.3])
    ax2.axis('off')
    ax2.set_title('Schematische Querschnittsdarstellung des Universums\nmit dem kosmischen Quader als äußerer Grenze')

    # Quader-Rand
    rect = plt.Rectangle((0.5, 0.5), 7, 7, linewidth=3, edgecolor='navy', facecolor='#F0F0FF', zorder=1)
    ax2.add_patch(rect)
    ax2.text(4, 8.1, 'Kosmischer Quader (Universums-Rand)', ha='center', fontsize=10,
             color='navy', fontweight='bold')

    # Inneres des Universums
    inner_rect = plt.Rectangle((1.0, 1.0), 6, 6, linewidth=1, edgecolor='steelblue',
                                facecolor='#E8F4FF', alpha=0.5, zorder=2)
    ax2.add_patch(inner_rect)
    ax2.text(4, 6.6, 'Bekanntes Universum\n(Galaxien, Filamente, CMB)', ha='center',
             fontsize=8.5, color='steelblue', va='top')

    # Schwarze Löcher am Rand (wo Quanten herrschen)
    bh_rand_pos = [
        (0.75, 0.75), (4.0, 0.75), (7.25, 0.75),
        (0.75, 4.0), (7.25, 4.0),
        (0.75, 7.25), (4.0, 7.25), (7.25, 7.25),
    ]
    for bx, by in bh_rand_pos:
        bh_circ = Circle((bx, by), 0.25, color='black', zorder=5)
        ax2.add_patch(bh_circ)
        # Nichts-Bereich des BH
        nichts_circ = Circle((bx, by), 0.4, color='red', fill=False, lw=1.5, ls='--', zorder=4, alpha=0.7)
        ax2.add_patch(nichts_circ)
        # Quantenfeldpfeile
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            dx, dy = 0.15 * np.cos(angle), 0.15 * np.sin(angle)
            ax2.annotate('', xy=(bx + dx*2.5, by + dy*2.5), xytext=(bx + dx*3.5, by + dy*3.5),
                         arrowprops=dict(arrowstyle='->', color='crimson', lw=0.8), zorder=6)

    # Zentrales BH (supermassiv)
    central_circ = Circle((4, 4), 0.5, color='black', zorder=5)
    ax2.add_patch(central_circ)
    accent = Circle((4, 4), 0.8, color='darkorange', fill=False, lw=2.5, zorder=4)
    ax2.add_patch(accent)
    ax2.text(4, 3.5, 'Sgr A*\n$4 \\times 10^6 M_\\odot$', ha='center', fontsize=8,
             color='darkorange', fontweight='bold', zorder=6)

    # Quantenbereich-Shading an den Rändern
    for x0, y0, w, h in [(0.5, 0.5, 7, 0.5), (0.5, 7.0, 7, 0.5),
                          (0.5, 0.5, 0.5, 7), (7.0, 0.5, 0.5, 7)]:
        ax2.add_patch(plt.Rectangle((x0, y0), w, h, color='red', alpha=0.12, zorder=3))
    ax2.text(4, -0.1, 'Nichts-Bereich (quantum-dominiert, Quantenphysik voll aktiv, lebensfeindlich)',
             ha='center', fontsize=8.5, color='darkred', fontstyle='italic')

    # Legende
    legend_items = [
        mpatches.Patch(color='black', label='Schwarze Löcher'),
        mpatches.Patch(color='red', alpha=0.5, label='„Nichts" (Quantenbereich)'),
        mpatches.Patch(color='navy', alpha=0.5, label='Kosmischer Quader (Rand)'),
        mpatches.Patch(color='steelblue', alpha=0.4, label='Universum (klassisch)'),
    ]
    ax2.legend(handles=legend_items, loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_kosmischer_quader.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_kosmischer_quader.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 7: Zusammenfassung – Quantenphysik im Nichts vs. Erde (Stufendiagramm)
# --------------------------------------------------------------------------
def plot_quanten_stufendiagramm():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim([-1, 11])
    ax.set_ylim([-0.5, 11.0])
    ax.axis('off')
    ax.set_title('Das Wirkprinzip der Quantenphysik: Von der Erde bis zum Nichts der Schwarzen Löcher\n'
                 '– Stufenweiser Übergang von klassischer zu vollständiger Quantendominanz –',
                 fontsize=12, fontweight='bold', pad=10)

    stufen = [
        (0.0, 'Klassische\nPhysik\n(makroskop.)', '#4472C4',
         'Newtonsche Mechanik,\nKlassische Elektrodynamik\nThermodynamik\n$S \\gg \\hbar$',
         'Erde, Sonnensystem,\nGalaxien\n(Leben möglich!)'),
        (2.0, 'Quanten-\nmechanik\n(atomare\nSkala)', '#7030A0',
         'Schrödinger-Gleichung\nUnschärferelation\nWellenfunktionen\n$S \\sim 10^3 - 10^6\\,\\hbar$',
         'Atome, Moleküle,\nHalbleiter\n(Leben möglich!)'),
        (4.0, 'Quanten-\nfeldtheorie\n(subatomar)', '#C00000',
         'Dirac-Gleichung\nFeynman-Diagramme\nVirtuelle Teilchen\n$S \\sim 10 - 100\\,\\hbar$',
         'Quarks, Elektronen,\nPhotonen\n(Leben kaum möglich)'),
        (6.0, 'Quanten-\ngravitation\n(Planck-\nSkala)', '#FF0000',
         'Planck-Länge $\\ell_P$\nSchwarze Loch-Thermodynamik\nHawking-Strahlung\n$S \\sim \\hbar$',
         'Ereignishorizont\nNirgends im Alltag\n(Leben nicht möglich)'),
        (8.0, 'Vollständige\nQuantenherrschaft\n(Nichts des\nSchw. Lochs)', '#800000',
         'Raumzeit-Quantisierung\nSingularitätsphysik\nInformationsparadox\n$S \\to 0$',
         'Nichts innerhalb\ndes Schwarzen Lochs\nLeben UNMÖGLICH'),
    ]

    for i, (y, title, col, physik, ort) in enumerate(stufen):
        # Haupt-Box
        box = FancyBboxPatch((0, y), 2.5, 1.5, boxstyle='round,pad=0.1',
                              facecolor=col, edgecolor='black', lw=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(1.25, y + 0.75, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='none', edgecolor='none'))

        # Physik-Details
        phys_box = FancyBboxPatch((3.0, y), 3.5, 1.5, boxstyle='round,pad=0.08',
                                   facecolor='white', edgecolor=col, lw=1.5, alpha=0.9)
        ax.add_patch(phys_box)
        ax.text(4.75, y + 0.75, physik, ha='center', va='center', fontsize=8,
                color='black')

        # Ort-Details
        ort_box = FancyBboxPatch((7.0, y), 3.5, 1.5, boxstyle='round,pad=0.08',
                                  facecolor='white', edgecolor=col, lw=1.5, alpha=0.9)
        ax.add_patch(ort_box)
        ax.text(8.75, y + 0.75, ort, ha='center', va='center', fontsize=8,
                color='darkred' if i == 4 else 'black', fontweight='bold' if i == 4 else 'normal')

        # Pfeil nach oben (außer letztem)
        if i < len(stufen) - 1:
            ax.annotate('', xy=(1.25, y + 1.55), xytext=(1.25, y + 2.0 - 0.05),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Gradient auf der linken Seite
    gradient_x = -0.6
    for i, (y, _, col, _, _) in enumerate(stufen):
        rect = plt.Rectangle((gradient_x, y), 0.35, 1.5, facecolor=col, edgecolor='none', alpha=0.9)
        ax.add_patch(rect)

    ax.text(-0.42, 10.6, 'Zunehmende\nQuantendominanz →', ha='center', fontsize=8,
            color='darkred', rotation=0)

    # Kopfzeile – oberhalb aller Boxen (höchste Box endet bei y=9.5)
    ax.text(1.25, 9.8, 'Physikbereich', ha='center', fontsize=9, fontweight='bold', color='black')
    ax.text(4.75, 9.8, 'Wirkendes Prinzip', ha='center', fontsize=9, fontweight='bold', color='black')
    ax.text(8.75, 9.8, 'Domäne / Lebbarkeit', ha='center', fontsize=9, fontweight='bold', color='black')
    for xv in [0, 2.5, 3.0, 6.5, 7.0, 10.5]:
        ax.plot([xv, xv], [0, 10.3], 'k-', lw=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_quanten_stufendiagramm.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_quanten_stufendiagramm.pdf gespeichert")


# --------------------------------------------------------------------------
# Plot 8: Raumzeit-Struktur: Penrose-Diagramm des Schwarzen Lochs mit Quader-Rand
# --------------------------------------------------------------------------
def plot_raumzeit_struktur():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Raumzeit-Struktur des Schwarzen Lochs: Kausaldiagramm (Penrose-Carter)\n'
                 'und kosmische Einbettung in den Quader-Rand',
                 fontsize=12, fontweight='bold', y=1.02)

    # --- Links: Penrose-Carter-Diagramm ---
    ax = axes[0]
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.8, 2.2])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Penrose-Carter-Diagramm:\nKausalstruktur des Schwarzen Lochs')

    # Äußeres Universum (Region I)
    ext_pts = np.array([[0, 0], [1.2, 1.2], [0, 2.0], [-1.2, 1.2]])
    ext_poly = plt.Polygon(ext_pts, closed=True, facecolor='lightcyan', edgecolor='steelblue',
                           lw=1.5, alpha=0.7, zorder=2)
    ax.add_patch(ext_poly)
    ax.text(0, 0.8, 'Region I\n(Äußeres Universum)\nLeben möglich', ha='center', va='center',
            fontsize=9, color='steelblue', fontweight='bold')

    # Zukunfts-Singularität (Region II - Schwarzes Loch)
    bh_pts = np.array([[1.2, 1.2], [0, 2.0], [-1.2, 1.2], [0, 1.5]])
    bh_poly = plt.Polygon(np.array([[0, 2.0], [1.2, 1.2], [0, 1.5], [-1.2, 1.2]]),
                          closed=True, facecolor='#1a1a1a', edgecolor='black',
                          lw=2, alpha=0.95, zorder=3)
    ax.add_patch(bh_poly)
    ax.text(0, 1.7, 'Region II\n„Das Nichts"\nQuantenphysik dominiert\nLeben UNMÖGLICH',
            ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=5)

    # Vergangenheit-Singularität (Region IV - Weißes Loch)
    wh_poly = plt.Polygon(np.array([[0, 0.0], [1.2, -1.2], [0, -1.6], [-1.2, -1.2]]),
                          closed=True, facecolor='#2a0a0a', edgecolor='darkred',
                          lw=2, alpha=0.8, zorder=3)
    ax.add_patch(wh_poly)
    ax.text(0, -0.85, 'Region IV\n(Weißes Loch)\n[zeitgespiegelt]',
            ha='center', va='center', fontsize=8, color='salmon')

    # Zweites Universum Region III
    reg3_poly = plt.Polygon(np.array([[0, 0], [-1.2, -1.2], [-1.2, 1.2]]),
                            closed=True, facecolor='#E8F4E8', edgecolor='darkgreen',
                            lw=1.5, alpha=0.5, zorder=2)
    ax.add_patch(reg3_poly)
    ax.text(-0.6, 0, 'Region III\n(anderes\nUniversum?)', ha='center', va='center',
            fontsize=7.5, color='darkgreen')

    # Ereignishorizont (Lichtgeodäten)
    ax.plot([0, 1.2], [0, 1.2], 'r-', lw=2.5, zorder=4, label='Ereignishorizont')
    ax.plot([0, -1.2], [0, 1.2], 'r-', lw=2.5, zorder=4)
    ax.plot([0, 1.2], [0, -1.2], 'b--', lw=2, zorder=4, label='Weißes-Loch-Horizont')
    ax.plot([0, -1.2], [0, -1.2], 'b--', lw=2, zorder=4)

    # Singuläritätslinie
    ax.plot([-1.2, 1.2], [1.2, 1.2], 'k-', lw=4, zorder=6, label='Singularität (Nichts)')

    # Lichtkegelstruktur
    for x0, y0 in [(0, 0.5), (0.5, 0.6)]:
        ax.annotate('', xy=(x0 + 0.12, y0 + 0.12), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.2))
        ax.annotate('', xy=(x0 - 0.12, y0 + 0.12), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.2))

    ax.text(0, 2.1, '$i^+$ (zeitartige Zukunft)', ha='center', fontsize=8.5, color='black')
    ax.text(1.35, 1.25, '$\\mathscr{I}^+$', fontsize=10, color='steelblue')
    ax.text(-1.45, 1.25, '$\\mathscr{I}^+$', fontsize=10, color='steelblue')
    ax.legend(loc='lower right', fontsize=8.5)

    # --- Rechts: Kosmische Einbettung mit Quader ---
    ax2 = axes[1]
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 10])
    ax2.axis('off')
    ax2.set_title('Kosmische Einbettung:\nSchwarzes Loch am Quader-Rand des Universums')

    # Quader-Rand
    quader = plt.Rectangle((0.5, 0.5), 9, 9, linewidth=3.5, edgecolor='navy',
                            facecolor='#F0F0FF', zorder=1)
    ax2.add_patch(quader)

    # Universum
    inner = plt.Rectangle((1.5, 1.5), 7, 7, linewidth=1.5, edgecolor='steelblue',
                           facecolor='#E8F4FF', alpha=0.4, zorder=2)
    ax2.add_patch(inner)

    # Kosmische Filamente
    np.random.seed(7)
    for _ in range(12):
        x1, y1 = np.random.uniform(2, 8, 2)
        x2, y2 = np.random.uniform(2, 8, 2)
        ax2.plot([x1, x2], [y1, y2], color='lightcoral', lw=0.8, alpha=0.5, zorder=3)

    # Schwarze Löcher am Rand
    rand_bh = [(1.0, 1.0), (5.0, 1.0), (9.0, 1.0),
               (1.0, 5.0), (9.0, 5.0),
               (1.0, 9.0), (5.0, 9.0), (9.0, 9.0)]
    for bx, by in rand_bh:
        bh_c = Circle((bx, by), 0.35, color='black', zorder=6)
        ax2.add_patch(bh_c)
        # Nichts-Bereich
        n_c = Circle((bx, by), 0.6, color='red', fill=False, lw=2, ls='--', zorder=5, alpha=0.8)
        ax2.add_patch(n_c)
        # Quanten-Wellenmuster
        for r_wave in [0.8, 1.0]:
            w_c = Circle((bx, by), r_wave, color='crimson', fill=False, lw=0.8, ls=':', zorder=4, alpha=0.4)
            ax2.add_patch(w_c)

    # Zentrales BH
    c_bh = Circle((5, 5), 0.5, color='black', zorder=6)
    ax2.add_patch(c_bh)
    c_acc = Circle((5, 5), 0.9, color='orange', fill=False, lw=2.5, zorder=5)
    ax2.add_patch(c_acc)
    ax2.text(5, 4.2, 'Zentral-BH', ha='center', fontsize=8.5, color='darkorange', fontweight='bold')

    ax2.text(5, 9.8, '▲ Quader-Rand des Universums ▲', ha='center', fontsize=9,
             color='navy', fontweight='bold')
    ax2.text(5, 0.2, '▼ Quader-Rand des Universums ▼', ha='center', fontsize=9,
             color='navy', fontweight='bold')

    # Quantenzone Legende
    ax2.text(4.0, 0.9, 'Nichts-Bereiche = Quantenphysik, kein Leben', ha='center',
             fontsize=8, color='darkred', fontstyle='italic')

    legend_handles = [
        mpatches.Patch(color='black', label='Schwarzes Loch (Zentrum)'),
        mpatches.Patch(color='red', alpha=0.5, label='Nichts-Bereich (Quantenphysik)'),
        mpatches.Patch(color='navy', alpha=0.4, label='Kosmischer Quader (Universums-Rand)'),
    ]
    ax2.legend(handles=legend_handles, loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qn_raumzeit_struktur.pdf'), bbox_inches='tight')
    plt.close()
    print("qn_raumzeit_struktur.pdf gespeichert")


# --------------------------------------------------------------------------
# Alle Plots erzeugen
# --------------------------------------------------------------------------
if __name__ == '__main__':
    print("Erzeuge Plots für Kapitel: Nichts der Schwarzen Löcher ...")
    plot_hawking_strahlung()
    plot_planck_skala()
    plot_unschaerfe_singularitaet()
    plot_quanten_wirkung_vergleich()
    plot_lebensbedingungen()
    plot_kosmischer_quader()
    plot_quanten_stufendiagramm()
    plot_raumzeit_struktur()
    print("\nAlle 8 Plots erfolgreich gespeichert!")
