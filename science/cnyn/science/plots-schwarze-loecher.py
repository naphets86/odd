#!/usr/bin/env python3
"""Generate plots for the Schwarze-Loecher-Analogie chapter in cnyn.tex."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc, Circle, PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

OUTDIR = ''


# ── PLOT 1: Schematischer Vergleich Tiefenquelle vs. Schwarzes Loch ──────────

def make_bh_analogie_schema():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        'Strukturelle Analogie: Tiefenquelle (links) vs. Schwarzes Loch (rechts)',
        fontsize=13, fontweight='bold')

    # ---- Left: Tiefenquelle ----
    ax1.set_facecolor('#F0F4FF')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Tiefenquelle (Genesis 7:11)', fontsize=11, fontweight='bold', color='#1A3A7A')

    # Draw funnel
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, alpha in [(3.5, 0.08), (2.5, 0.12), (1.5, 0.18), (0.8, 0.30)]:
        ax1.plot(r * np.cos(theta), r * np.sin(theta), 'b-', alpha=alpha, lw=1.0)

    # Critical radius R0
    r0 = 1.0
    ax1.plot(r0 * np.cos(theta), r0 * np.sin(theta), 'r-', lw=2.0, label='$R_0$ (krit. Radius)')

    # Source at center
    ax1.add_patch(Circle((0, 0), 0.35, color='#1A3A7A', zorder=5))
    ax1.text(0, 0, 'TQ', color='white', ha='center', va='center', fontsize=9, fontweight='bold')

    # Flow arrows inward
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for a in angles:
        x1, y1 = 3.2 * np.cos(a), 3.2 * np.sin(a)
        x2, y2 = 0.7 * np.cos(a), 0.7 * np.sin(a)
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='#1A5F9E', lw=1.2))

    # Global network lines
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax1.plot([3.5 * np.cos(a), 4.8 * np.cos(a)],
                 [3.5 * np.sin(a), 4.8 * np.sin(a)],
                 '#888', lw=1.0, linestyle='--')
    ax1.text(0, -4.5, 'Globales Quellennetzwerk\n$C_{\\mathrm{max}}$-Kapazitätsgrenze',
             ha='center', va='center', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    ax1.text(-4.6, 3.8, '$R_0$', color='red', fontsize=10)

    # Labels
    ax1.text(0, 1.4, '$p_{\\mathrm{sog}} \\approx 10^7\\,\\mathrm{Pa}$',
             ha='center', fontsize=9, color='darkblue',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax1.text(0, -1.6, '$v_{\\mathrm{max}} \\approx 440\\,\\mathrm{m/s}$',
             ha='center', fontsize=9, color='darkblue',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # ---- Right: Black Hole ----
    ax2.set_facecolor('#0A0A1A')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Schwarzes Loch (Allgemeine Relativitätstheorie)', fontsize=11,
                  fontweight='bold', color='#FFD700')

    # Accretion disk rings
    for r, alpha in [(3.8, 0.12), (3.0, 0.20), (2.2, 0.30), (1.6, 0.45)]:
        color = mcolors.to_rgba('#FF8C00', alpha=alpha)
        ax2.plot(r * np.cos(theta), r * np.sin(theta) * 0.35,
                 color=color[:3], alpha=alpha * 2, lw=2.5)

    # Schwarzschild radius
    rs = 1.0
    ax2.plot(rs * np.cos(theta), rs * np.sin(theta), '-', color='#FFD700',
             lw=2.0, label=f'$r_s = 2GM/c^2$')

    # Black hole center
    ax2.add_patch(Circle((0, 0), 0.35, color='black', zorder=5, ec='#FFD700', lw=1.5))
    ax2.text(0, 0, 'SL', color='#FFD700', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # In-falling matter arrows
    for a in angles:
        x1, y1 = 3.2 * np.cos(a), 3.2 * np.sin(a)
        x2, y2 = 0.7 * np.cos(a), 0.7 * np.sin(a)
        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='#FFA500', lw=1.2))

    # Photon sphere
    ax2.plot(1.5 * np.cos(theta), 1.5 * np.sin(theta), ':', color='#88AAFF',
             lw=1.0, alpha=0.7)
    ax2.text(0, 1.7, 'Photonensphäre', color='#88AAFF', ha='center', fontsize=7.5)

    ax2.text(0, -4.5, 'Schwarzschild-Radius $r_s$\n(Ereignishorizont)',
             ha='center', va='center', fontsize=8.5, color='#FFD700',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#111133', alpha=0.9))
    ax2.text(3.8, 3.2, '$r_s$', color='#FFD700', fontsize=10)
    ax2.text(0, 1.25, '$v \\to c$ bei $r_s$', ha='center', fontsize=8.5, color='#FFA500',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A1A2A', alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_analogie_schema.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_analogie_schema.pdf')


# ── PLOT 2: Potentialfeldvergleich Φ ∝ -1/r ──────────────────────────────────

def make_bh_potentialfeld():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Strukturidentisches Gravitationspotential: $\Phi \propto -1/r$',
        fontsize=13, fontweight='bold')

    r = np.linspace(0.05, 5.0, 500)

    # Physical constants (normalized)
    Q_TQ = 2.0    # source strength Tiefenquelle
    GM_BH = 2.0   # GM for black hole
    r_s = 1.0     # Schwarzschild radius

    Phi_TQ = -Q_TQ / (4 * np.pi * r)
    Phi_BH = -GM_BH / r

    # ---- Left: Tiefenquelle Potential ----
    ax = axes[0]
    ax.set_facecolor('#F5F7FF')
    ax.plot(r, Phi_TQ, 'b-', lw=2.5, label=r'$\Phi_{\mathrm{TQ}}(r) = -\frac{Q}{4\pi r}$')
    ax.axvline(1.0, color='red', linestyle='--', lw=1.5, label='$R_0$ (krit. Radius)')
    ax.fill_between(r, Phi_TQ, 0, where=(r < 1.0), alpha=0.15, color='red',
                    label='Subkritische Zone')
    ax.fill_between(r, Phi_TQ, 0, where=(r >= 1.0), alpha=0.08, color='blue')
    ax.set_xlabel('$r / R_0$')
    ax.set_ylabel(r'$\Phi_{\mathrm{TQ}}(r)$ [normiert]')
    ax.set_title('Tiefenquelle: Quellenpotential\n(radiale Senkenströmung)', fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(-4.5, 0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.8, -4.0,
            r'$\nabla^2\Phi_{\mathrm{TQ}} = -\frac{Q}{\rho_f}\,\delta^3(\mathbf{r})$',
            fontsize=9, color='navy',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # ---- Right: Black Hole Potential ----
    ax = axes[1]
    ax.set_facecolor('#0F0F1F')
    ax.plot(r, Phi_BH, color='#FFA500', lw=2.5,
            label=r'$\Phi_{\mathrm{BH}}(r) = -\frac{GM}{r}$')
    ax.axvline(r_s, color='#FFD700', linestyle='--', lw=1.5, label='$r_s = 2GM/c^2$')
    ax.fill_between(r, Phi_BH, 0, where=(r < r_s), alpha=0.25, color='#FF4444',
                    label='Innere des Horizonts')
    ax.fill_between(r, Phi_BH, 0, where=(r >= r_s), alpha=0.06, color='#FFA500')
    ax.set_xlabel('$r / r_s$')
    ax.set_ylabel(r'$\Phi_{\mathrm{BH}}(r)$ [normiert]')
    ax.set_title('Schwarzes Loch: Newtonisches Potential\n(Schwarzschild-Näherung)',
                 fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(-4.5, 0.3)
    ax.tick_params(colors='#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')
    ax.xaxis.label.set_color('#CCCCCC')
    ax.title.set_color('#FFD700')
    ax.legend(fontsize=9, facecolor='#1A1A2A', labelcolor='#DDDDDD')
    ax.grid(True, alpha=0.2, color='#444')
    ax.text(2.0, -4.0,
            r'$\nabla^2\Phi_{\mathrm{BH}} = 4\pi G\rho$' + '\n' + r'(Newton-Approx.)',
            fontsize=9, color='#FFA500',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111133', alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_potentialfeld.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_potentialfeld.pdf')


# ── PLOT 3: Soggeschwindigkeit v(r) – Vergleich beider Systeme ───────────────

def make_bh_soggeschwindigkeit():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Strömungsgeschwindigkeiten: Tiefenquelle vs. Schwarzes Loch',
        fontsize=13, fontweight='bold')

    r = np.linspace(0.1, 6.0, 600)
    r0 = 1.0  # critical radius
    v0 = 1.0  # reference velocity
    c  = 1.0  # speed of light (normalized)
    rs = 1.0  # Schwarzschild radius

    v_TQ = v0 * (r0 / r)**2      # continuity: A·v = const → v ∝ 1/r²
    v_BH = c * np.sqrt(rs / r)   # free-fall: v ∝ r^{-1/2}

    # ---- Left: Tiefenquelle ----
    ax = axes[0]
    ax.set_facecolor('#F5F7FF')
    ax.plot(r, v_TQ, 'b-', lw=2.5, label=r'$v_{\mathrm{TQ}}(r) = v_0 (R_0/r)^2$')
    ax.axvline(r0, color='red', linestyle='--', lw=1.5, label='$r = R_0$ (krit. Radius)')
    ax.axhline(v0, color='green', linestyle=':', lw=1.3, alpha=0.7, label='$v = v_0$')
    ax.fill_between(r, v_TQ, 0, where=(r < r0), alpha=0.15, color='red',
                    label='Überschallzone')
    ax.set_xlabel('$r / R_0$')
    ax.set_ylabel('$v(r) / v_0$')
    ax.set_title('Tiefenquelle: Soggeschwindigkeit\n(Kontinuitätsgleichung)', fontsize=11)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(3.0, 4.2,
            r'$Q = 4\pi r^2 v(r) = \mathrm{const}$' + '\n' + r'$\Rightarrow v \propto r^{-2}$',
            fontsize=9, color='navy',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # ---- Right: Black Hole ----
    ax = axes[1]
    ax.set_facecolor('#0F0F1F')
    ax.plot(r, v_BH, color='#FFA500', lw=2.5,
            label=r'$v_{\mathrm{ff}}(r) = c\sqrt{r_s/r}$')
    ax.axvline(rs, color='#FFD700', linestyle='--', lw=1.5, label='$r = r_s$ (Horizont)')
    ax.axhline(c, color='#88FF88', linestyle=':', lw=1.3, alpha=0.8, label='$v = c$')
    ax.fill_between(r, v_BH, 0, where=(r < rs), alpha=0.25, color='#FF4444',
                    label='Hinter dem Horizont')
    ax.set_xlabel('$r / r_s$')
    ax.set_ylabel('$v(r) / c$', color='#CCCCCC')
    ax.set_title('Schwarzes Loch: Freifallgeschwindigkeit\n(Painlevé-Gullstrand-Koordinaten)',
                 fontsize=11)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.0)
    ax.tick_params(colors='#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')
    ax.xaxis.label.set_color('#CCCCCC')
    ax.title.set_color('#FFD700')
    ax.legend(fontsize=9, facecolor='#1A1A2A', labelcolor='#DDDDDD')
    ax.grid(True, alpha=0.2, color='#444')
    ax.text(3.0, 4.2,
            r'$v_{\mathrm{ff}} = \sqrt{2GM/r} = c\sqrt{r_s/r}$' + '\n' +
            r'$\Rightarrow v \propto r^{-1/2}$',
            fontsize=9, color='#FFA500',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111133', alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_soggeschwindigkeit.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_soggeschwindigkeit.pdf')


# ── PLOT 4: Ereignishorizont – Kritischer Radius R₀ vs. Schwarzschild-Radius ─

def make_bh_ereignishorizont():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Kritischer Radius $R_0$ (Tiefenquelle) vs. Schwarzschild-Radius $r_s$ (Schwarzes Loch)',
        fontsize=13, fontweight='bold')

    # ---- Left: Tiefenquelle – Critical radius as function of flow parameters ----
    ax = axes[0]
    ax.set_facecolor('#F5F7FF')
    Q_vals = np.linspace(0.5, 5.0, 200)   # Quellstärke
    rho_f  = 1000.0                         # Fluiddichte Wasser [kg/m³]
    v_c    = 440.0                          # kritische Geschwindigkeit [m/s] ~ Schallgeschwindigkeit
    # R0 from continuity + pressure: Q/(4pi R0^2) = v_c → R0 = sqrt(Q/(4pi v_c))
    R0 = np.sqrt(Q_vals / (4 * np.pi * v_c)) * 1e2   # scale for visibility
    for scale, label, col in [(1.0, '$v_c = 440\\,\\mathrm{m/s}$', 'blue'),
                               (0.75, '$v_c = 330\\,\\mathrm{m/s}$', 'green'),
                               (1.5, '$v_c = 660\\,\\mathrm{m/s}$', 'orange')]:
        R0_s = np.sqrt(Q_vals / (4 * np.pi * v_c * scale)) * 1e2
        ax.plot(Q_vals, R0_s, lw=2.0, label=label)
    ax.set_xlabel(r'Quellstärke $Q$ [normiert]')
    ax.set_ylabel(r'Kritischer Radius $R_0$ [normiert]')
    ax.set_title('Tiefenquelle: Abhängigkeit von\n$R_0$ von Quellstärke und Geschwindigkeit',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.15,
            r'$R_0 = \sqrt{\dfrac{Q}{4\pi\,v_c}}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11, color='navy',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    # ---- Right: Schwarzschild radius as function of mass ----
    ax = axes[1]
    ax.set_facecolor('#0F0F1F')
    M_sun = np.logspace(-1, 10, 400)   # mass in solar masses
    G_SI  = 6.674e-11
    c_SI  = 3e8
    M_sun_kg = 1.989e30
    r_s_km = 2 * G_SI * (M_sun * M_sun_kg) / c_SI**2 / 1e3   # in km
    ax.loglog(M_sun, r_s_km, color='#FFA500', lw=2.5,
              label=r'$r_s = 2GM/c^2$')
    # Mark known objects
    objects = [
        (10, 2 * G_SI * 10 * M_sun_kg / c_SI**2 / 1e3, 'Stellares SL\n(10 $M_\\odot$)', '#88AAFF'),
        (1e6, 2 * G_SI * 1e6 * M_sun_kg / c_SI**2 / 1e3, 'AGN\n($10^6 M_\\odot$)', '#FFD700'),
        (4e6, 2 * G_SI * 4e6 * M_sun_kg / c_SI**2 / 1e3, 'Sgr A*\n($4\\times10^6 M_\\odot$)',
         '#FF8888'),
        (1e9, 2 * G_SI * 1e9 * M_sun_kg / c_SI**2 / 1e3, 'M87*\n($10^9 M_\\odot$)', '#AAFFAA'),
    ]
    for m, rs_val, label, col in objects:
        ax.plot(m, rs_val, 'o', color=col, ms=8, zorder=5)
        ax.text(m * 1.5, rs_val * 0.5, label, color=col, fontsize=7.5)
    ax.set_xlabel('Masse $M$ [$M_\\odot$]', color='#CCCCCC')
    ax.set_ylabel('$r_s$ [km]', color='#CCCCCC')
    ax.set_title('Schwarzes Loch: Schwarzschild-Radius\nals Funktion der Masse', fontsize=11)
    ax.tick_params(colors='#CCCCCC')
    ax.title.set_color('#FFD700')
    ax.legend(fontsize=9, facecolor='#1A1A2A', labelcolor='#DDDDDD')
    ax.grid(True, alpha=0.2, color='#444', which='both')
    ax.text(0.05, 0.85,
            r'$r_s(M_\odot) \approx 2.95\,\mathrm{km}$',
            transform=ax.transAxes, color='#FFA500', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111133', alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_ereignishorizont.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_ereignishorizont.pdf')


# ── PLOT 5: Kapazitätsgrenze C_max vs. Eddington-Luminosität ─────────────────

def make_bh_kapazitaet():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Kapazitätsbegrenzte Ausbrüche: $C_{\max}$ (Tiefenquelle) vs. Eddington-Limit (Schwarzes Loch)',
        fontsize=13, fontweight='bold')

    t = np.linspace(0, 6, 2000)

    # ---- Left: Tiefenquelle cycles ----
    ax = axes[0]
    ax.set_facecolor('#F5F7FF')
    T_period = 5.0   # 5-day cycle
    tau_decay = 0.8
    C_max = 1.0      # capacity
    # Sawtooth: linear fill, exponential drain
    def tq_cycle(t, T=5.0, tau=0.8, phases=7):
        C = np.zeros_like(t)
        for k in range(phases):
            t0 = k * T
            t1 = t0 + 0.5 * T
            t2 = t0 + T
            mask_fill  = (t >= t0) & (t < t1)
            mask_drain = (t >= t1) & (t < t2)
            if mask_fill.any():
                C[mask_fill] = (t[mask_fill] - t0) / (0.5 * T)
            if mask_drain.any():
                C[mask_drain] = np.exp(-(t[mask_drain] - t1) / tau)
        return C

    C_t = tq_cycle(t * T_period)
    flux = np.gradient(np.maximum(0, -np.gradient(C_t, t)), t)
    flux = np.abs(flux)
    flux /= flux.max()

    ax.plot(t * T_period, C_t, 'b-', lw=2.0, label='Kapazität $C(t)/C_{\\mathrm{max}}$')
    ax.plot(t * T_period, flux * 0.9, 'r--', lw=1.5, alpha=0.8,
            label='Erosionsrate (Entleerungsphase)')
    ax.axhline(1.0, color='red', lw=1.2, linestyle=':', label='$C_{\\mathrm{max}}$')
    ax.set_xlabel('Zeit [Tage]')
    ax.set_ylabel('Relative Kapazität / Erosionsrate')
    ax.set_title('Tiefenquelle: Rhythmische Füllung\nund Entleerung über 40–50 Zyklen', fontsize=11)
    ax.set_xlim(0, t.max() * T_period)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.6,
            '$T_{\\mathrm{Zyklus}} \\approx 5\\,\\mathrm{Tage}$\n$N_{\\mathrm{ges}} = 40{-}50$',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # ---- Right: Eddington limit ----
    ax = axes[1]
    ax.set_facecolor('#0F0F1F')
    # Luminosity vs accretion rate
    M_dot = np.logspace(-10, -2, 400)  # solar masses / year
    eta = 0.1     # efficiency
    L_sun = 3.828e26  # W
    c_SI = 3e8
    M_sun = 1.989e30
    yr_s  = 3.156e7
    L_acc = eta * M_dot * (M_sun / yr_s) * c_SI**2 / L_sun   # in L_sun
    L_Edd_stellar = 1.26e4 * 10     # L_sun for 10 M_sun BH (L_Edd ≈ 1.26e4 M/M_sun)
    L_Edd_agn = 1.26e4 * 1e8
    ax.loglog(M_dot, L_acc, color='#FFA500', lw=2.5, label=r'$L_{\mathrm{acc}} = \eta \dot{M} c^2$')
    ax.axhline(L_Edd_stellar, color='#88AAFF', lw=1.5, linestyle='--',
               label='$L_{\\mathrm{Edd}}$ (10 $M_\\odot$)')
    ax.axhline(L_Edd_agn, color='#FFD700', lw=1.5, linestyle=':',
               label='$L_{\\mathrm{Edd}}$ ($10^8 M_\\odot$)')
    # Shade super-Eddington
    ax.fill_between(M_dot, L_Edd_stellar, L_acc,
                    where=(L_acc > L_Edd_stellar), alpha=0.25, color='red',
                    label='Supereddington-Regime')
    ax.set_xlabel(r'Akkretionsrate $\dot{M}$ [$M_\odot$/Jahr]', color='#CCCCCC')
    ax.set_ylabel('Leuchtkraft $L$ [$L_\\odot$]', color='#CCCCCC')
    ax.set_title('Schwarzes Loch: Eddington-Luminosität\n'
                 r'$L_{\mathrm{Edd}} = \frac{4\pi G M c}{\kappa}$', fontsize=11)
    ax.tick_params(colors='#CCCCCC')
    ax.title.set_color('#FFD700')
    ax.legend(fontsize=9, facecolor='#1A1A2A', labelcolor='#DDDDDD')
    ax.grid(True, alpha=0.2, color='#444', which='both')

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_kapazitaet_vergleich.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_kapazitaet_vergleich.pdf')


# ── PLOT 6: Periodizität – Erosionszyklen vs. QPO Frequenzen ─────────────────

def make_bh_periodik():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Rhythmische Periodizität: 5-Tage-Erosionszyklen vs. QPO-Frequenzen',
        fontsize=13, fontweight='bold')

    # ---- Left: Tiefenquelle frequency spectrum ----
    ax = axes[0]
    ax.set_facecolor('#F5F7FF')
    f_TQ = np.linspace(0, 20, 2000)   # frequency in 1/day
    # Main mode at f=1/5 day + harmonics
    spectrum_TQ = np.zeros_like(f_TQ)
    f0 = 1.0 / 5.0   # 5-day period
    for harmonic in range(1, 8):
        sigma = 0.02 * harmonic
        A = 1.0 / harmonic**1.5
        spectrum_TQ += A * np.exp(-0.5 * ((f_TQ - harmonic * f0) / sigma)**2)
    ax.plot(f_TQ, spectrum_TQ, 'b-', lw=1.5, label='Leistungsspektrum')
    ax.fill_between(f_TQ, spectrum_TQ, alpha=0.2, color='blue')
    ax.axvline(f0, color='red', lw=2.0, linestyle='--',
               label=f'$f_0 = 1/5\\,\\mathrm{{d}}^{{-1}}$ (Grundmode)')
    for h in range(2, 8):
        ax.axvline(h * f0, color='orange', lw=1.0, linestyle=':',
                   alpha=0.7, label='Harmonische' if h == 2 else '')
    ax.set_xlabel('Frequenz [d$^{-1}$]')
    ax.set_ylabel('Spektrale Leistungsdichte')
    ax.set_title('Tiefenquelle: Fourierspektrum\nder 5-Tage-Erosionszyklen', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95,
            r'$f_{\mathrm{TQ}} \propto \sqrt{\frac{C_{\max}}{\rho V L}}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # ---- Right: QPO frequency vs BH mass ----
    ax = axes[1]
    ax.set_facecolor('#0F0F1F')
    M_BH = np.logspace(0, 9, 400)   # solar masses
    # Keplerian orbital frequency at ISCO: f_ISCO ≈ 220 Hz / (M/M_sun) for Kerr
    f_QPO_Hz = 2200.0 / M_BH   # roughly: f_ISCO ∝ c³/(6√6 GM)
    f_QPO_day = f_QPO_Hz * 86400  # convert to day-1

    ax.loglog(M_BH, f_QPO_Hz, color='#FFA500', lw=2.5,
              label=r'$f_{\mathrm{ISCO}} \propto c^3 / (GM)$')
    # Mark known QPO objects
    qpo_objects = [
        (10, 220, 'GRS 1915+105\n(~10 $M_\\odot$)', '#88AAFF'),
        (1e6, 2.2e-4, 'AGN QPO\n($10^6 M_\\odot$)', '#FFD700'),
        (4e6, 5.5e-5, 'Sgr A*\n($4{\\times}10^6 M_\\odot$)', '#FF8888'),
    ]
    for m, f, label, col in qpo_objects:
        ax.plot(m, f, 'o', color=col, ms=9, zorder=5)
        ax.text(m * 1.5, f * 2.5, label, color=col, fontsize=7.5)
    ax.set_xlabel('Schwarze-Loch-Masse $M$ [$M_\\odot$]', color='#CCCCCC')
    ax.set_ylabel('QPO-Frequenz [Hz]', color='#CCCCCC')
    ax.set_title('Schwarzes Loch: Quasi-periodische Oszillationen\n'
                 r'$f_{\mathrm{QPO}} \propto c^3 / (GM)$', fontsize=11)
    ax.tick_params(colors='#CCCCCC')
    ax.title.set_color('#FFD700')
    ax.legend(fontsize=9, facecolor='#1A1A2A', labelcolor='#DDDDDD')
    ax.grid(True, alpha=0.2, color='#444', which='both')

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_periodik_vergleich.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_periodik_vergleich.pdf')


# ── PLOT 7: Kosmisches Netzwerk – kosmische Großstruktur als Tiefenquellen ────

def make_bh_kosmisches_netz():
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle(
        'Kosmische Großstruktur: Supermassive Schwarze Löcher als kosmische Tiefenquellen',
        fontsize=13, fontweight='bold')
    ax.set_facecolor('#050510')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    rng = np.random.default_rng(seed=17)

    # ---- Cosmic filaments (random lines with glow) ----
    n_filaments = 25
    for _ in range(n_filaments):
        x0, y0 = rng.uniform(0, 10, 2)
        length  = rng.uniform(1.5, 4.5)
        angle   = rng.uniform(0, 2 * np.pi)
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)
        # Glow effect: multiple widths
        for lw, alpha in [(3.5, 0.03), (1.8, 0.07), (0.6, 0.20)]:
            ax.plot([x0, x1], [y0, y1], '-', color='#AABBFF',
                    lw=lw, alpha=alpha, solid_capstyle='round')

    # ---- Void regions ----
    void_centers = [(2.5, 2.5), (7.0, 3.0), (4.5, 7.5), (8.0, 7.5), (1.5, 7.0)]
    for vx, vy in void_centers:
        ax.add_patch(Circle((vx, vy), rng.uniform(0.6, 1.1),
                            color='#050510', alpha=0.7, zorder=2))
        ax.text(vx, vy, 'Void', color='#334455', fontsize=7, ha='center', va='center',
                alpha=0.6)

    # ---- Supermassive black holes at filament nodes ----
    smbh_positions = [(1.0, 1.0), (4.0, 1.5), (8.5, 1.5), (2.5, 5.0),
                      (5.5, 4.5), (9.0, 5.5), (3.0, 8.8), (7.0, 9.0), (5.5, 8.0)]
    smbh_masses = [1e9, 3e8, 5e8, 1e8, 4e8, 2e9, 6e7, 8e8, 3e9]
    max_m = max(smbh_masses)
    for (sx, sy), sm in zip(smbh_positions, smbh_masses):
        r_disp = 0.12 + 0.22 * (sm / max_m)**0.4
        # Glow rings
        for mult, a_ring in [(3.0, 0.04), (2.0, 0.08), (1.3, 0.15), (1.0, 0.8)]:
            ax.add_patch(Circle((sx, sy), r_disp * mult,
                                color='#FFB830', alpha=a_ring, zorder=3))
        # Infall arrows
        n_arr = 6
        for ia in np.linspace(0, 2 * np.pi, n_arr, endpoint=False):
            ax.annotate('',
                        xy=(sx + r_disp * 1.2 * np.cos(ia),
                            sy + r_disp * 1.2 * np.sin(ia)),
                        xytext=(sx + r_disp * 2.8 * np.cos(ia),
                                sy + r_disp * 2.8 * np.sin(ia)),
                        arrowprops=dict(arrowstyle='->', color='#FFA500',
                                        lw=0.8, mutation_scale=10),
                        zorder=4)

    # Legend / annotation
    ax.text(5.0, 0.35, 'Kosmische Filamente', color='#AABBFF', fontsize=9,
            ha='center', alpha=0.9)
    for sx, sy in [(1.0, 1.0), (5.5, 4.5), (7.0, 9.0)]:
        ax.text(sx + 0.3, sy - 0.5, 'SMBH', color='#FFD700', fontsize=7.5, alpha=0.9)

    # Analogy box
    ax.text(5.0, 9.7,
            'SMBH als kosmische Senken: Materie fließt entlang der Filamente\n'
            'auf supermassive Schwarze Löcher zu — Analogon zu $C_{\\mathrm{max}}$ der Tiefenquellen',
            color='#CCDDFF', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A25', alpha=0.85))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'bh_kosmische_tiefenquellen.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_kosmische_tiefenquellen.pdf')


# ── PLOT 8: Formale Analogietafel ─────────────────────────────────────────────

def make_bh_formal_analogie():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor('white')
    ax.axis('off')
    fig.suptitle(
        'Formale Analogietafel: Tiefenquellen der Sintflut ↔ Schwarze Löcher',
        fontsize=14, fontweight='bold', y=0.97)

    # Table data: (property, TQ formula, BH formula, comment)
    rows = [
        ('Zentralobjekt', 'Tiefenquelle (TQ)', 'Schwarzes Loch (SL)',
         'Singuläre Senke im System'),
        ('Krit. Radius', r'$R_0$: Quellenöffnung',
         r'$r_s = 2GM/c^2$: Ereignishorizont',
         r'Kein Entkommen für $r < R_0$ bzw.\ $r_s$'),
        ('Potential', r'$\Phi_{\mathrm{TQ}}=-\frac{Q}{4\pi r}$',
         r'$\Phi_{\mathrm{BH}}=-\frac{GM}{r}$',
         r'Gleiche $-1/r$-Struktur'),
        ('Geschwindigkeit',
         r'$v_{\mathrm{TQ}}=v_0\!\left(\frac{R_0}{r}\right)^{\!2}$',
         r'$v_{\mathrm{ff}}=c\sqrt{r_s/r}$',
         r'Divergenz am krit.\ Radius'),
        ('Kapazitätsgrenze',
         r'$C_{\max}$: globale Systemkapazität',
         r'$L_{\mathrm{Edd}}=\frac{4\pi GMc}{\kappa}$',
         r'Obere Grenze des Energiedurchsatzes'),
        ('Periodizität',
         r'$T_{\mathrm{TQ}} \approx 5\,\mathrm{d}$: Erosionszyklus',
         r'$f_{\mathrm{QPO}} \propto c^3/(GM)$: Quasi-periodo.\ Oszill.',
         r'Rhythmische Pulsation des Systems'),
        ('Differentialgleichung',
         r'$\frac{d\psi_i}{dt} = -\alpha_i\sqrt{\psi_i}+\sum_j \beta_{ij}(\psi_j-\psi_i)$',
         r'$\frac{dL}{dt} = \eta \dot{M}c^2 - L/t_{\mathrm{dyn}}$',
         r'Gleiche mathematische Struktur'),
        ('Vernetzung',
         r'Globales Quellennetzwerk (Genesis 7:11)',
         r'Kosmisches Filament-SMBH-Netz',
         r'Hierarchisch vernetzt über Skalen'),
        ('Skala', r'$R_0 \sim 10^3\,\mathrm{km}$ (Erdmaßstab)',
         r'$r_s \sim 3\,\mathrm{km}\,(M_\odot)$',
         r'Skalierungsinvarianz der Physik'),
    ]

    col_headers = ['Eigenschaft', 'Tiefenquelle', 'Schwarzes Loch', 'Kommentar']
    col_colors  = ['#1A3A7A', '#1A6A3A', '#6A1A1A', '#4A3A6A']
    col_x = [0.01, 0.22, 0.52, 0.78]
    col_widths = [0.20, 0.29, 0.25, 0.21]
    row_height = 0.085
    header_y = 0.88

    # Draw column headers
    for cx, cw, ch, label in zip(col_x, col_widths, col_colors, col_headers):
        ax.add_patch(FancyBboxPatch(
            (cx, header_y - 0.005), cw - 0.005, row_height + 0.005,
            boxstyle='round,pad=0.005', facecolor=ch, edgecolor='white', lw=1.0,
            transform=ax.transAxes, clip_on=False))
        ax.text(cx + (cw - 0.005) / 2, header_y + row_height / 2, label,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    row_bg_colors = ['#EEF4FF', '#F4FFEE', '#FFF4EE', '#F4EEFF',
                     '#EEFFF4', '#FFEEFF', '#EEFFFF', '#FFEEEE', '#FFFFF0']

    for row_idx, (prop, tq_val, bh_val, comment) in enumerate(rows):
        y_pos = header_y - (row_idx + 1) * row_height - 0.01 * row_idx
        bg = row_bg_colors[row_idx % len(row_bg_colors)]
        for cx, cw in zip(col_x, col_widths):
            ax.add_patch(FancyBboxPatch(
                (cx, y_pos), cw - 0.005, row_height - 0.002,
                boxstyle='round,pad=0.003', facecolor=bg, edgecolor='#CCCCCC', lw=0.5,
                transform=ax.transAxes, clip_on=False))
        cell_items = [prop, tq_val, bh_val, comment]
        for col_i, (cx, cw, txt) in enumerate(zip(col_x, col_widths, cell_items)):
            ax.text(cx + (cw - 0.005) / 2, y_pos + row_height / 2, txt,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8.0, color='#1A1A2A', wrap=True)

    ax.text(0.5, 0.005,
            'Die formale Übereinstimmung der mathematischen Strukturen belegt eine tiefe physikalische Analogie\n'
            'zwischen den biblischen Tiefenquellen und den Schwarzen Löchern der modernen Astrophysik.',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8.5, color='#444',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFF0', alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTDIR + 'bh_formal_analogie.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ bh_formal_analogie.pdf')


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating Schwarze-Loecher-Analogie plots...')
    make_bh_analogie_schema()
    make_bh_potentialfeld()
    make_bh_soggeschwindigkeit()
    make_bh_ereignishorizont()
    make_bh_kapazitaet()
    make_bh_periodik()
    make_bh_kosmisches_netz()
    make_bh_formal_analogie()
    print('All plots generated successfully.')
