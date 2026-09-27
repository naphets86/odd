#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonische Ausbreitung akustischer Signale in symmetrischen Hörsälen
Matplotlib-Plots für die wissenschaftliche Arbeit
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle, Arc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.special import jn  # Bessel functions
import warnings
warnings.filterwarnings('ignore')

# Global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

FIGDIR = ' Lx) * np.cos(n * np.pi * Y / Ly)

    # Fundamental and first harmonics
    p = (room_mode(X, Y, Lx, Ly, 1, 0, 1.0) +
         room_mode(X, Y, Lx, Ly, 0, 1, 0.8) +
         room_mode(X, Y, Lx, Ly, 1, 1, 0.6) +
         room_mode(X, Y, Lx, Ly, 2, 1, 0.4))

    # Symmetrieachse
    ax = axes[0]
    im = ax.contourf(X, Y, p, levels=60, cmap='RdBu_r')
    ax.contour(X, Y, p, levels=12, colors='k', linewidths=0.4, alpha=0.5)
    ax.axvline(x=Lx/2, color='gold', lw=2, ls='--', label='Symmetrieachse')
    ax.axhline(y=Ly/2, color='limegreen', lw=2, ls='--', label='Symmetrieachse')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Schalldruckfeld p(x,y) — Modenüberlagerung')
    ax.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Schalldruck [Pa, normiert]')
    ax.set_aspect('equal')
    # Mark source
    ax.plot(Lx/2, 0.5, 'r*', ms=14, label='Quelle', zorder=5)

    # Pure mode (1,1) to show symmetry
    ax2 = axes[1]
    p11 = room_mode(X, Y, Lx, Ly, 2, 2, 1.0)
    im2 = ax2.contourf(X, Y, p11, levels=60, cmap='seismic')
    ax2.contour(X, Y, p11, levels=10, colors='k', linewidths=0.4, alpha=0.5)
    ax2.axvline(x=Lx/2, color='gold', lw=2, ls='--', label='Symmetrieachse x')
    ax2.axhline(y=Ly/2, color='limegreen', lw=2, ls='--', label='Symmetrieachse y')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Eigenmode (2,2) — Symmetrische Modenstruktur')
    ax2.legend(loc='upper right', fontsize=9)
    plt.colorbar(im2, ax=ax2, label='Schalldruck [Pa, normiert]')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot1_wellenfeld.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot1_wellenfeld.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 1 gespeichert.")

# ============================================================
# PLOT 2: Eigenfrequenzen des Quaderraums — Dispersionsrelation
# ============================================================
def plot2_eigenfrequenzen():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Abbildung 2: Eigenfrequenzen und Modendichte des rechteckigen Hörsaals',
                 fontsize=12, fontweight='bold')

    c = 343.0  # Schallgeschwindigkeit m/s
    Lx, Ly, Lz = 20.0, 12.0, 5.0

    freqs = []
    modes = []
    for m in range(0, 12):
        for n in range(0, 12):
            for l in range(0, 8):
                if m == 0 and n == 0 and l == 0:
                    continue
                f = (c / 2) * np.sqrt((m/Lx)**2 + (n/Ly)**2 + (l/Lz)**2)
                freqs.append(f)
                modes.append((m, n, l))

    freqs = np.array(freqs)
    freqs_sorted = np.sort(freqs)

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(freqs_sorted)))
    ax.scatter(range(len(freqs_sorted[:80])), freqs_sorted[:80],
               c=colors[:80], s=30, zorder=3)
    ax.set_xlabel('Modenindex $N$')
    ax.set_ylabel('Eigenfrequenz $f_{mnl}$ [Hz]')
    ax.set_title('Eigenfrequenzen $f_{mnl} = \\frac{c}{2}\\sqrt{(m/L_x)^2+(n/L_y)^2+(l/L_z)^2}$')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 80)

    # Weyl'sche Formel für Modendichte
    ax2 = axes[1]
    f_max = 1000
    f_range = np.linspace(10, f_max, 500)
    V = Lx * Ly * Lz
    S = 2*(Lx*Ly + Ly*Lz + Lx*Lz)
    L = 4*(Lx + Ly + Lz)
    # Weyl: N(f) ~ 4*pi*V/(3*c^3)*f^3
    N_weyl = (4*np.pi*V / (3*c**3)) * f_range**3 + (np.pi*S / (4*c**2)) * f_range**2 + (L/(8*c)) * f_range
    dN_df = np.gradient(N_weyl, f_range)

    ax2.fill_between(f_range, dN_df, alpha=0.4, color='steelblue', label='Modendichte $dN/df$')
    ax2.plot(f_range, dN_df, color='navy', lw=1.5)
    ax2b = ax2.twinx()
    ax2b.plot(f_range, N_weyl, color='crimson', lw=2, ls='--', label='$N(f)$ kumulativ (Weyl)')
    ax2b.set_ylabel('Kumulative Modenzahl $N(f)$', color='crimson')
    ax2b.tick_params(axis='y', labelcolor='crimson')
    ax2.set_xlabel('Frequenz $f$ [Hz]')
    ax2.set_ylabel('Modendichte $dN/df$ [1/Hz]', color='navy')
    ax2.set_title('Weylsche Modendichte: $N(f) \\approx \\frac{4\\pi V}{3c^3}f^3 + \\frac{\\pi S}{4c^2}f^2 + \\frac{L}{8c}f$')
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot2_eigenfrequenzen.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot2_eigenfrequenzen.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 2 gespeichert.")

# ============================================================
# PLOT 3: Symmetrische Greenssche Funktion + Spiegelquellen
# ============================================================
def plot3_greensfunction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Abbildung 3: Greensche Funktion und Spiegelquellenmethode im symmetrischen Raum",
                 fontsize=12, fontweight='bold')

    # Spiegelquellen im rechteckigen Raum
    Lx, Ly = 20.0, 12.0
    xs, ys = 10.0, 6.0  # Quelle im Zentrum (Symmetrie)
    xr, yr = 15.0, 9.0   # Empfänger

    x = np.linspace(0, Lx, 300)
    y = np.linspace(0, Ly, 300)
    X, Y = np.meshgrid(x, y)

    # Image sources (first order: 6 Spiegelquellen + Originalquelle)
    sources = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            sx = xs + 2*i*Lx
            sy = ys + 2*j*Ly
            sources.append((sx, sy, 1.0 / (np.abs(i)+np.abs(j)+1)**1.5))

    k = 2*np.pi * 200 / 343.0  # k bei 200 Hz

    p_total = np.zeros_like(X, dtype=complex)
    for (sx, sy, amp) in sources:
        r = np.sqrt((X-sx)**2 + (Y-sy)**2) + 1e-9
        p_total += amp * np.exp(1j*k*r) / r

    ax = axes[0]
    im = ax.contourf(X, Y, np.real(p_total), levels=60, cmap='RdBu_r')
    ax.contour(X, Y, np.real(p_total), levels=15, colors='k', linewidths=0.3, alpha=0.5)
    ax.plot(xs, ys, 'r*', ms=14, zorder=5, label=f'Quelle ({xs},{ys})')
    ax.plot(xr, yr, 'g^', ms=10, zorder=5, label=f'Empfänger ({xr},{yr})')
    ax.axvline(x=Lx/2, color='gold', lw=1.5, ls='--', alpha=0.8, label='Symmetrieachsen')
    ax.axhline(y=Ly/2, color='gold', lw=1.5, ls='--', alpha=0.8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Re[G(x,y)] bei f=200 Hz, Quelle im Zentrum')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    plt.colorbar(im, ax=ax, label='Schalldruck [Pa, normiert]')

    # SPL (dB) Verteilung
    ax2 = axes[1]
    SPL = 20 * np.log10(np.abs(p_total) / (np.max(np.abs(p_total))) + 1e-12)
    im2 = ax2.contourf(X, Y, SPL, levels=np.linspace(-40, 0, 60), cmap='inferno')
    ax2.plot(xs, ys, 'w*', ms=14, zorder=5, label='Quelle')
    ax2.plot(xr, yr, 'c^', ms=10, zorder=5, label='Empfänger')
    ax2.axvline(x=Lx/2, color='lime', lw=1.5, ls='--', alpha=0.8, label='Symmetrieachsen')
    ax2.axhline(y=Ly/2, color='lime', lw=1.5, ls='--', alpha=0.8)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Schalldruckpegel SPL(x,y) [dB rel. max]')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_xlim(0, Lx); ax2.set_ylim(0, Ly)
    plt.colorbar(im2, ax=ax2, label='SPL [dB]')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot3_greens.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot3_greens.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 3 gespeichert.")

# ============================================================
# PLOT 4: Nachhallzeit RT60 und Sabine-Formel
# ============================================================
def plot4_nachhall():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Abbildung 4: Nachhallzeit und Raumakustische Güte des symmetrischen Hörsaals',
                 fontsize=12, fontweight='bold')

    # Sabine: T60 = 0.161 * V / (alpha * S)
    V = 20.0 * 12.0 * 5.0   # Raumvolumen
    S_total = 2*(20*12 + 12*5 + 20*5)

    freqs = np.array([125, 250, 500, 1000, 2000, 4000])
    # Frequenzabhängige Absorptionsgrade für verschiedene Materialien
    alpha_hard = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05])   # Beton
    alpha_seat = np.array([0.15, 0.25, 0.40, 0.45, 0.45, 0.40])   # Bestuhlung
    alpha_carpet = np.array([0.05, 0.10, 0.20, 0.35, 0.45, 0.50]) # Teppich

    # Gewichteter Absorptionsgrad für den Hörsaal
    # Flächen: Boden (carpet), Decke (hard), Wände (mix)
    S_floor = 20.0 * 12.0
    S_ceil  = 20.0 * 12.0
    S_walls = S_total - 2*S_floor
    alpha_eff = (S_floor * alpha_carpet + S_ceil * alpha_hard + S_walls * alpha_seat) / S_total

    T60_sabine = 0.161 * V / (alpha_eff * S_total)
    # Eyring: T60 = -0.161*V / (S*ln(1-alpha))
    T60_eyring = -0.161 * V / (S_total * np.log(1 - alpha_eff + 1e-9))

    ax = axes[0]
    ax.semilogx(freqs, T60_sabine, 'b-o', lw=2, ms=8, label='Sabine $T_{60}$')
    ax.semilogx(freqs, T60_eyring, 'r--s', lw=2, ms=8, label='Eyring $T_{60}$')
    ax.axhline(y=0.8, color='green', ls=':', lw=2, label='Optimum Sprache (0.8 s)')
    ax.axhline(y=1.5, color='orange', ls=':', lw=2, label='Optimum Musik (1.5 s)')
    ax.fill_between([100, 5000], [0.6, 0.6], [1.0, 1.0], alpha=0.15, color='green', label='Akzeptanzbereich Sprache')
    ax.set_xlabel('Frequenz $f$ [Hz]')
    ax.set_ylabel('Nachhallzeit $T_{60}$ [s]')
    ax.set_title('Sabine vs. Eyring Nachhallzeit\n$T_{60}^{\\rm Sab} = \\frac{0{,}161\\,V}{\\bar{\\alpha}\\,S}$')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(freqs)
    ax.set_xticklabels([str(f) for f in freqs])

    # Impulsantwort-Abklingkurve (Schroeder-Integral)
    ax2 = axes[1]
    t = np.linspace(0, 2.5, 1000)
    T60_mid = T60_sabine[2]  # 500 Hz
    # Schroeder decay: L(t) = -60/T60 * t + noise
    decay = -60/T60_mid * t + np.random.normal(0, 0.8, len(t)).cumsum() * 0.02
    decay = np.minimum(0, decay)

    # EDT (Early Decay Time): slope from 0 to -10 dB
    # T20: -5 to -25 dB; T30: -5 to -35 dB
    ax2.plot(t, decay, color='steelblue', lw=1.5, label='Schroeder-Abklingkurve')
    # Fit lines
    mask_edt = (decay >= -10)
    if mask_edt.any():
        t_edt = t[mask_edt]
        ax2.plot([t_edt[0], t_edt[-1]], [decay[mask_edt][0], decay[mask_edt][-1]],
                 'r--', lw=2, label=f'EDT $\\approx {t_edt[-1]:.2f}$ s')
    ax2.axhline(-60, color='gray', ls=':', lw=1.5, label='-60 dB')
    ax2.axhline(-10, color='orange', ls=':', lw=1.2, alpha=0.7)
    ax2.axhline(-35, color='purple', ls=':', lw=1.2, alpha=0.7)
    ax2.set_xlabel('Zeit $t$ [s]')
    ax2.set_ylabel('Schalldruckpegel [dB]')
    ax2.set_title(f'Schroeder-Abklingkurve bei 500 Hz\n$T_{{60}} \\approx {T60_mid:.2f}$ s')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-70, 5)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot4_nachhall.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot4_nachhall.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 4 gespeichert.")

# ============================================================
# PLOT 5: Symmetrieanalyse — SPL-Profil entlang Achsen
# ============================================================
def plot5_symmetrie():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Abbildung 5: Symmetrische Schallpegelverteilung — Profilschnitte und Abweichungsanalyse',
                 fontsize=12, fontweight='bold')

    Lx, Ly = 20.0, 12.0
    xs, ys = Lx/2, 1.0  # Quelle an Symmetrieachse, vorne
    k_vals = [2*np.pi*f/343 for f in [125, 500, 1000, 2000]]
    labels = ['125 Hz', '500 Hz', '1000 Hz', '2000 Hz']
    colors_k = ['navy', 'steelblue', 'coral', 'crimson']

    # Längsschnitt y = Ly/2
    x_line = np.linspace(0.5, Lx-0.5, 300)

    ax = axes[0, 0]
    for k, lab, col in zip(k_vals, labels, colors_k):
        r = np.sqrt((x_line - xs)**2 + (Ly/2 - ys)**2) + 1e-6
        # Simple approximation: direct + first reflection
        r_refl = np.sqrt((x_line - xs)**2 + (Ly/2 + ys)**2) + 1e-6
        p = np.exp(1j*k*r)/r + 0.7*np.exp(1j*k*r_refl)/r_refl
        SPL = 20 * np.log10(np.abs(p) + 1e-12)
        SPL -= np.max(SPL)
        ax.plot(x_line, SPL, color=col, lw=1.8, label=lab)
    ax.axvline(x=Lx/2, color='gold', lw=2, ls='--', label='Symmetrieachse')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('SPL [dB, rel. max]')
    ax.set_title('Längsschnitt bei $y = L_y/2$')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Querschnitt x = Lx/2
    y_line = np.linspace(0.5, Ly-0.5, 300)
    ax2 = axes[0, 1]
    for k, lab, col in zip(k_vals, labels, colors_k):
        r = np.sqrt((Lx/2 - xs)**2 + (y_line - ys)**2) + 1e-6
        r_refl = np.sqrt((Lx/2 + xs - Lx)**2 + (y_line - ys)**2) + 1e-6
        p = np.exp(1j*k*r)/r + 0.7*np.exp(1j*k*r_refl)/r_refl
        SPL = 20 * np.log10(np.abs(p) + 1e-12)
        SPL -= np.max(SPL)
        ax2.plot(y_line, SPL, color=col, lw=1.8, label=lab)
    ax2.axvline(x=Ly/2, color='gold', lw=2, ls='--', label='Symmetrieachse')
    ax2.set_xlabel('y [m]')
    ax2.set_ylabel('SPL [dB, rel. max]')
    ax2.set_title('Querschnitt bei $x = L_x/2$')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Symmetriedeviation |SPL(x) - SPL(Lx-x)|
    ax3 = axes[1, 0]
    for k, lab, col in zip(k_vals, labels, colors_k):
        r_left = np.sqrt((x_line - xs)**2 + (Ly/2 - ys)**2) + 1e-6
        r_right = np.sqrt((Lx - x_line - xs)**2 + (Ly/2 - ys)**2) + 1e-6
        p_left = np.abs(np.exp(1j*k*r_left)/r_left)
        p_right = np.abs(np.exp(1j*k*r_right)/r_right)
        deviation = np.abs(20*np.log10(p_left+1e-12) - 20*np.log10(p_right+1e-12))
        ax3.plot(x_line[:150], deviation[:150], color=col, lw=1.5, label=lab)
    ax3.axhline(y=3.0, color='red', ls='--', lw=1.5, label='3 dB Toleranz')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('$|\\Delta SPL|$ [dB]')
    ax3.set_title('Symmetriedeviation $|\\mathrm{SPL}(x) - \\mathrm{SPL}(L_x - x)|$')
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Summed SPL across cross-section (seat rows) as function of distance from source
    ax4 = axes[1, 1]
    seat_rows = np.linspace(2, Lx-1, 20)
    y_seats = np.linspace(1, Ly-1, 12)
    SPL_seats_mean = []
    SPL_seats_std = []
    for xi in seat_rows:
        vals = []
        for yi in y_seats:
            r = np.sqrt((xi - xs)**2 + (yi - ys)**2) + 1e-6
            k = 2*np.pi*500/343
            p = np.abs(np.exp(1j*k*r)/r)
            vals.append(20*np.log10(p+1e-12))
        SPL_seats_mean.append(np.mean(vals))
        SPL_seats_std.append(np.std(vals))
    SPL_seats_mean = np.array(SPL_seats_mean)
    SPL_seats_std = np.array(SPL_seats_std)
    SPL_seats_mean -= np.max(SPL_seats_mean)

    ax4.errorbar(seat_rows, SPL_seats_mean, yerr=SPL_seats_std,
                 fmt='b-o', ms=5, capsize=4, label='Mittlerer SPL (500 Hz)')
    # Fit: inverse square law
    r_dist = np.sqrt((seat_rows - xs)**2 + (Ly/2 - ys)**2)
    spl_inv = -20*np.log10(r_dist/r_dist[0])
    ax4.plot(seat_rows, spl_inv - spl_inv[0] + SPL_seats_mean[0], 'r--', lw=2,
             label='Freifeldgesetz $-20\\lg(r/r_0)$')
    ax4.set_xlabel('Sitzreihen-Abstand von der Quelle [m]')
    ax4.set_ylabel('SPL [dB, rel. max]')
    ax4.set_title('SPL-Abfall über Sitzreihen (gemittelt über y)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot5_symmetrie.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot5_symmetrie.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 5 gespeichert.")

# ============================================================
# PLOT 6: Harmonische Frequenzreihen und Fourier-Zerlegung
# ============================================================
def plot6_harmonische():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Abbildung 6: Harmonische Signalstruktur und Fourier-Zerlegung akustischer Moden',
                 fontsize=12, fontweight='bold')

    t = np.linspace(0, 0.1, 10000)
    fs = 100000  # Abtastrate

    # Grundton f0 = 50 Hz (entspricht Raummode)
    f0 = 50.0
    harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
    amplitudes = [1.0, 0.6, 0.4, 0.25, 0.18, 0.12, 0.08, 0.05]

    # Zusammengesetztes Signal
    signal = sum(A * np.cos(2*np.pi*n*f0*t) for n, A in zip(harmonics, amplitudes))
    signal_fund = np.cos(2*np.pi*f0*t)

    ax = axes[0, 0]
    ax.plot(t*1000, signal_fund, color='navy', lw=1.5, alpha=0.7, label='Grundton $f_0$')
    ax.plot(t*1000, signal, color='crimson', lw=1.5, label='Überlagerung')
    ax.set_xlabel('Zeit $t$ [ms]')
    ax.set_ylabel('Amplitude [normiert]')
    ax.set_title('Zeitbereichssignal: Grundton + Harmonische')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 40)

    # Spektrum
    ax2 = axes[0, 1]
    N = len(signal)
    freq_ax = np.fft.rfftfreq(N, 1/fs)
    spec = np.abs(np.fft.rfft(signal)) / N * 2
    ax2.bar([n*f0 for n in harmonics], amplitudes,
            width=15, color=plt.cm.viridis(np.linspace(0, 1, len(harmonics))),
            alpha=0.85, label='Harmonische $nf_0$')
    ax2.plot(freq_ax[:500], spec[:500], 'k-', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Frequenz $f$ [Hz]')
    ax2.set_ylabel('Amplitude [normiert]')
    ax2.set_title('Spektrum: Harmonische Reihe $\\{n f_0\\}_{n=1}^{8}$')
    ax2.set_xlim(0, 450)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Parseval-Theorem: Energieverteilung
    ax3 = axes[1, 0]
    energies = [A**2 / 2 for A in amplitudes]
    total_energy = sum(energies)
    cum_energy = np.cumsum(energies) / total_energy * 100
    bars = ax3.bar(harmonics, [E/total_energy*100 for E in energies],
                   color=plt.cm.plasma(np.linspace(0.1, 0.9, len(harmonics))),
                   alpha=0.85, label='Energieanteil [%]')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(harmonics, cum_energy, 'r-o', ms=7, lw=2, label='Kumulierte Energie [%]')
    ax3_twin.axhline(90, color='gray', ls=':', lw=1.5, label='90 % Schwelle')
    ax3_twin.set_ylabel('Kumulierte Energie [%]', color='crimson')
    ax3_twin.tick_params(axis='y', labelcolor='crimson')
    ax3.set_xlabel('Harmonische Ordnung $n$')
    ax3.set_ylabel('Energieanteil [%]')
    ax3.set_title('Parseval: Energieverteilung auf Harmonische\n$E = \\sum_n \\frac{A_n^2}{2}$')
    ax3.set_xticks(harmonics)
    lines1, lbls1 = ax3.get_legend_handles_labels()
    lines2, lbls2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2, lbls1+lbls2, fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Phase response der Raumübertragungsfunktion
    ax4 = axes[1, 1]
    f_plot = np.linspace(20, 2000, 5000)
    # Model: Raumübertragungsfunktion mit Polen bei Eigenfrequenzen
    c, Lx_r = 343, 20.0
    f_room_modes = [c*n/(2*Lx_r) for n in range(1, 8)]
    H = np.ones(len(f_plot), dtype=complex)
    for fm in f_room_modes:
        Q = 15.0
        s = 1j * 2*np.pi*f_plot
        s_p = 1j * 2*np.pi*fm
        H += 0.3 * (s_p/Q) / (s**2 - 2*s_p/Q*s + s_p**2 + 1e-3)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)
    H_db -= np.max(H_db)
    phase = np.angle(H, deg=True)

    ax4.plot(f_plot, H_db, color='navy', lw=1.5, label='|H(f)| [dB]')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(f_plot, phase, color='coral', lw=1.0, alpha=0.7, label='Phase [°]')
    for fm in f_room_modes:
        ax4.axvline(x=fm, color='green', ls=':', lw=0.8, alpha=0.6)
    ax4_twin.set_ylabel('Phase [°]', color='coral')
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    ax4.set_xlabel('Frequenz $f$ [Hz]')
    ax4.set_ylabel('|H(f)| [dB]')
    ax4.set_title('Raumübertragungsfunktion $H(f)$\nPole bei Eigenfrequenzen (grün)')
    lines1, lbls1 = ax4.get_legend_handles_labels()
    lines2, lbls2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1+lines2, lbls1+lbls2, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot6_harmonische.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot6_harmonische.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 6 gespeichert.")

# ============================================================
# PLOT 7: Geometrische Akustik und Strahlendiagramm
# ============================================================
def plot7_strahlen():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Abbildung 7: Geometrische Akustik — Strahlengang und Symmetriereflexionen',
                 fontsize=12, fontweight='bold')

    Lx, Ly = 20.0, 12.0
    xs, ys = 10.0, 0.5  # Quelle auf Symmetrieachse

    ax = axes[0]
    ax.set_xlim(-0.5, Lx+0.5)
    ax.set_ylim(-0.5, Ly+0.5)
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='black', lw=2.5))
    # Symmetrieachse
    ax.axvline(x=Lx/2, color='gold', lw=2, ls='--', alpha=0.8, label='Symmetrieachse')

    np.random.seed(42)
    n_rays = 18
    angles = np.linspace(10, 170, n_rays) * np.pi / 180
    colors_r = plt.cm.tab20(np.linspace(0, 1, n_rays))

    for ang, col in zip(angles, colors_r):
        x0, y0 = xs, ys
        dx, dy = np.cos(ang), np.sin(ang)
        total_path = []
        total_path.append((x0, y0))
        for _ in range(6):
            # Find next wall
            t_min = 1e9
            # Check x=0, x=Lx, y=0, y=Ly
            if dx > 0: t_x = (Lx - x0) / dx
            elif dx < 0: t_x = -x0 / dx
            else: t_x = 1e9

            if dy > 0: t_y = (Ly - y0) / dy
            elif dy < 0: t_y = -y0 / dy
            else: t_y = 1e9

            t = min(t_x, t_y)
            x1 = x0 + t * dx
            y1 = y0 + t * dy
            total_path.append((x1, y1))
            if t == t_x: dx = -dx
            if t == t_y: dy = -dy
            x0, y0 = x1, y1

        xs_p = [p[0] for p in total_path]
        ys_p = [p[1] for p in total_path]
        ax.plot(xs_p, ys_p, color=col, lw=1.0, alpha=0.7)

    ax.plot(xs, ys, 'r*', ms=14, zorder=10, label='Schallquelle')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Strahlenakustik: Reflexionsmuster im Hörsaal')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # Laufzeitdifferenzen und Echodiagramm
    ax2 = axes[1]
    c = 343.0
    # Receiver at (15, 6)
    xr, yr = 15.0, 6.0
    # Compute path lengths: direct + image sources
    image_sources = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            sx = xs + 2*i*Lx
            sy = ys + 2*j*Ly
            r = np.sqrt((xr-sx)**2 + (yr-sy)**2)
            n_refl = abs(i) + abs(j)
            image_sources.append((r, n_refl))

    image_sources.sort()
    t_arrivals = [r/c for r, _ in image_sources[:40]]
    amplitudes_echo = [0.9**n for _, n in image_sources[:40]]

    # Convert to ms
    t_ms = np.array(t_arrivals) * 1000

    markerline, stemlines, baseline = ax2.stem(t_ms, amplitudes_echo,
                                                linefmt='steelblue', markerfmt='bo',
                                                basefmt='k-')
    plt.setp(stemlines, lw=1.0)
    plt.setp(markerline, ms=5)
    ax2.axvline(t_ms[0], color='red', lw=2, ls='--', label='Direktschall')
    ax2.axvspan(0, 50, alpha=0.1, color='green', label='Frühe Reflexionen (<50ms)')
    ax2.axvspan(50, max(t_ms)+5, alpha=0.1, color='orange', label='Nachhall (>50ms)')
    ax2.set_xlabel('Laufzeit $t$ [ms]')
    ax2.set_ylabel('Relative Amplitude')
    ax2.set_title('Echodiagramm: Spiegelquellen-Ankunftszeiten\n(Quelle: Symmetrieachse, Empfänger: (15,6) m)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 120)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot7_strahlen.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot7_strahlen.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 7 gespeichert.")

# ============================================================
# PLOT 8: Klanggüte-Kennzahlen (C50, D50, STI-Approximation)
# ============================================================
def plot8_kennzahlen():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Abbildung 8: Raumakustische Gütekriterien — C50, D50 und Deutlichkeit',
                 fontsize=12, fontweight='bold')

    # C50: Clarity = 10*log10(E_early / E_late)
    # Schroeder integral simulation
    np.random.seed(0)
    t = np.linspace(0, 2.5, 5000)
    T60 = 1.0
    # Impulsantwort h(t) ~ exp(-6.9*t/T60) * Gaussian noise
    h = np.exp(-6.9*t/T60) * (np.random.randn(len(t)) * 0.3 + np.cos(2*np.pi*500*t)*0.7)
    h2 = h**2
    dt = t[1]-t[0]
    t50_idx = np.searchsorted(t, 0.05)
    t80_idx = np.searchsorted(t, 0.08)

    E_early_50 = np.sum(h2[:t50_idx]) * dt
    E_late_50  = np.sum(h2[t50_idx:]) * dt
    C50 = 10 * np.log10(E_early_50 / (E_late_50 + 1e-12))
    D50 = E_early_50 / (np.sum(h2)*dt + 1e-12)

    E_early_80 = np.sum(h2[:t80_idx]) * dt
    E_late_80  = np.sum(h2[t80_idx:]) * dt
    C80 = 10 * np.log10(E_early_80 / (E_late_80 + 1e-12))

    ax = axes[0]
    ax.fill_between(t*1000, h2, where=(t < 0.05), alpha=0.7, color='steelblue', label=f'Früh (<50ms): C50={C50:.1f} dB')
    ax.fill_between(t*1000, h2, where=(t >= 0.05), alpha=0.5, color='coral', label=f'Spät (≥50ms)')
    ax.plot(t*1000, h2, 'k-', lw=0.5, alpha=0.4)
    ax.axvline(50, color='red', lw=2, ls='--', label='$t_{50}$ = 50 ms')
    ax.axvline(80, color='purple', lw=2, ls=':', label='$t_{80}$ = 80 ms')
    ax.set_xlabel('Zeit $t$ [ms]')
    ax.set_ylabel('$h^2(t)$ [normiert]')
    ax.set_title(f'Impulsantwort-Energie: C50={C50:.1f} dB, D50={D50:.2f}, C80={C80:.1f} dB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500)

    # C50 als Funktion des Abstands (symmetrischer Hörsaal)
    ax2 = axes[1]
    distances = np.linspace(1, 18, 40)
    c_speed = 343.0

    C50_vals = []
    D50_vals = []
    for d in distances:
        t_arr = d / c_speed  # Direkte Laufzeit
        # Simple model: direct-to-reverberant ratio
        # C50 ~ 10*log10(Q/(4*pi*d^2*R)) where R = room constant
        R = 50  # m^2 Sabine
        Q_src = 2  # Richtwirkung der Quelle
        DRR = Q_src / (4*np.pi*d**2 + 1e-9) * R
        C50_vals.append(10*np.log10(DRR + 1e-3))
        D50_vals.append(DRR / (DRR + 1))

    C50_vals = np.array(C50_vals)
    D50_vals = np.array(D50_vals)

    ax2.plot(distances, C50_vals, 'b-o', ms=5, lw=2, label='C50 [dB]')
    ax2b = ax2.twinx()
    ax2b.plot(distances, D50_vals*100, 'r--s', ms=5, lw=2, label='D50 [%]')
    ax2.axhline(0, color='gray', ls=':', lw=1.5, label='C50 = 0 dB (Grenze)')
    ax2.axhline(-3, color='orange', ls=':', lw=1.2, label='C50 = -3 dB')
    ax2.set_xlabel('Abstand zur Quelle $r$ [m]')
    ax2.set_ylabel('Klarheitsmaß C50 [dB]', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2b.set_ylabel('Deutlichkeit D50 [%]', color='crimson')
    ax2b.tick_params(axis='y', labelcolor='crimson')
    ax2.set_title('C50 und D50 als Funktion des Sitzabstands\n(symmetrischer Hörsaal)')
    ax2.grid(True, alpha=0.3)
    lines1, lbls1 = ax2.get_legend_handles_labels()
    lines2, lbls2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, lbls1+lbls2, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/plot8_kennzahlen.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/plot8_kennzahlen.png', bbox_inches='tight', dpi=120)
    plt.close()
    print("Plot 8 gespeichert.")

# ============================================================
# EXECUTE ALL
# ============================================================
if __name__ == '__main__':
    print("Generiere alle Plots...")
    plot1_wellenfeld()
    plot2_eigenfrequenzen()
    plot3_greensfunction()
    plot4_nachhall()
    plot5_symmetrie()
    plot6_harmonische()
    plot7_strahlen()
    plot8_kennzahlen()
    print("\nAlle 8 Plots erfolgreich gespeichert.")

