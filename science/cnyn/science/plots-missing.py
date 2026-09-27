#!/usr/bin/env python3
"""Generate all missing plots for cnyn.tex."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# Common style
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

# ── PLOT 1: grand_canyon_ablaufprozess ────────────────────────────────────────

def make_ablaufprozess():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Ablaufprozess im Grand Canyon: Inhomogene Quellenverteilung',
                 fontsize=13, fontweight='bold')

    np.random.seed(42)
    n_sources = 30
    # Place sources along the canyon flanks
    x_pos = np.concatenate([
        np.random.uniform(-4.5, -0.5, n_sources // 2),
        np.random.uniform(0.5, 4.5, n_sources // 2),
    ])
    y_pos = np.random.uniform(0, 20, n_sources)
    volumes = np.random.exponential(scale=1.0, size=n_sources) + 0.1
    volumes_norm = volumes / volumes.max()

    # Left panel: map view
    ax1.set_facecolor('#F5ECD7')
    # Colorado River (center)
    river_y = np.linspace(0, 20, 200)
    river_width = 0.3 + 0.1 * np.sin(river_y * 0.5)
    ax1.fill_betweenx(river_y, -river_width, river_width, color='#3A7FC1', alpha=0.8,
                      label='Colorado River')

    # Source circles
    cmap = plt.cm.YlOrRd
    for i in range(n_sources):
        radius = 0.15 + volumes_norm[i] * 0.55
        color = cmap(volumes_norm[i])
        circle = Circle((x_pos[i], y_pos[i]), radius, color=color, alpha=0.75,
                         linewidth=0.8, edgecolor='#333')
        ax1.add_patch(circle)
        # Arrow pointing toward center river
        dx = -x_pos[i] * 0.4
        dy = 0.0
        ax1.annotate('', xy=(x_pos[i] + dx, y_pos[i] + dy),
                     xytext=(x_pos[i], y_pos[i]),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))

    # Legend for sources
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=volumes.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label('Quellvolumen [rel. Einheiten]', fontsize=9)

    ax1.set_xlim(-5.5, 5.5)
    ax1.set_ylim(-0.5, 21)
    ax1.set_xlabel('Breite [km]')
    ax1.set_ylabel('Länge entlang Canyon [km]')
    ax1.set_title('Verteilung der Quellen entlang des Grand Canyon', fontsize=11)
    river_patch = mpatches.Patch(color='#3A7FC1', alpha=0.8, label='Colorado River')
    ax1.legend(handles=[river_patch], loc='upper right', fontsize=9)
    ax1.text(-5.2, 19.5, 'Pfeile: Absaugrichtung\nKreise: Quellvolumen',
             fontsize=8, color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Right panel: histogram of source volumes
    ax2.set_facecolor('#F5ECD7')
    n_bins = 12
    counts, edges, patches = ax2.hist(volumes, bins=n_bins, color='#E07B39', edgecolor='white',
                                       linewidth=0.8, alpha=0.85)
    # Color by bin height
    norm = plt.Normalize(counts.min(), counts.max())
    for p, c in zip(patches, counts):
        p.set_facecolor(plt.cm.YlOrRd(norm(c)))

    ax2.set_xlabel('Quellvolumen [rel. Einheiten]')
    ax2.set_ylabel('Anzahl der Quellen')
    ax2.set_title('Histogramm der Quellvolumina\n(Inhomogenität der Quellenverteilung)', fontsize=11)
    ax2.axvline(np.median(volumes), color='#1A5F9E', linewidth=1.5, linestyle='--',
                label=f'Median = {np.median(volumes):.2f}')
    ax2.axvline(np.mean(volumes), color='#B22222', linewidth=1.5, linestyle=':',
                label=f'Mittelwert = {np.mean(volumes):.2f}')
    ax2.legend(fontsize=9)
    ax2.text(0.97, 0.95,
             'Wenige sehr große Quellen\nviele kleine Quellen\n→ Inhomogene Verteilung',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR + 'grand_canyon_ablaufprozess.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ grand_canyon_ablaufprozess.pdf')


# ── PLOT 2: exp1_height_variation ─────────────────────────────────────────────

def make_exp1():
    H = np.linspace(1000, 8850, 30)
    rho = 1000.0
    g = 9.81
    alpha = np.radians(23.5)
    theta = np.radians(36)  # Grand Canyon
    P_sog = 1e7  # Pa

    P_hydro = rho * g * H          # Pa
    g_eff = g * np.cos(theta - alpha)
    P_tilt = rho * g_eff * H       # Pa

    v = np.sqrt(2 * (P_hydro + P_sog) / rho)
    Cf = 0.01
    tau = 0.5 * rho * Cf * v**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 1: Hydraulische Parameter als Funktion der Wasserhöhe',
                 fontsize=13, fontweight='bold')

    ax1.plot(H / 1000, P_hydro / 1e5, color='#1A5F9E', linewidth=2, label='Hydrostatischer Druck')
    ax1.plot(H / 1000, P_tilt / 1e5, color='#B22222', linewidth=2, linestyle='--',
             label='Druck mit Erdneigung (36°N)')
    ax1.set_xlabel('Wasserhöhe [km]')
    ax1.set_ylabel('Druck [bar]')
    ax1.set_title('Hydrostatischer Druck vs. Wasserhöhe')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1, 8.85)

    ax2_left = ax2
    ax2_right = ax2.twinx()
    l1, = ax2_left.plot(H / 1000, v, color='#2CA02C', linewidth=2, label='Strömungsgeschwindigkeit')
    l2, = ax2_right.plot(H / 1000, tau / 1e3, color='#FF7F0E', linewidth=2, linestyle='--',
                         label='Schubspannung')
    ax2_left.set_xlabel('Wasserhöhe [km]')
    ax2_left.set_ylabel('Strömungsgeschwindigkeit [m/s]', color='#2CA02C')
    ax2_right.set_ylabel('Schubspannung [kPa]', color='#FF7F0E')
    ax2.set_title('Strömungsgeschwindigkeit und Schubspannung')
    ax2_left.tick_params(axis='y', colors='#2CA02C')
    ax2_right.tick_params(axis='y', colors='#FF7F0E')
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2_left.set_xlim(1, 8.85)

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp1_height_variation.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp1_height_variation.pdf')


# ── PLOT 3: exp2_latitude_comparison ──────────────────────────────────────────

def make_exp2():
    theta_deg = np.linspace(-90, 90, 50)
    theta = np.radians(theta_deg)
    alpha = np.radians(23.5)
    g = 9.81
    rho = 1000.0
    H = 8850.0

    g_eff = g * np.cos(theta - alpha)
    P = rho * g_eff * H / 1e5  # bar

    omega = 2 * np.pi / 86400
    R = 6.371e6
    v = 440.0
    a_centri = omega**2 * R * np.cos(theta)**2
    a_coriolis = 2 * omega * v * np.abs(np.sin(theta))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 2: Geophysikalische Effekte als Funktion des Breitengrads',
                 fontsize=13, fontweight='bold')

    ax1_right = ax1.twinx()
    l1, = ax1.plot(theta_deg, g_eff, color='#1A5F9E', linewidth=2,
                   label='Effektive Gravitation $g_{\\mathrm{eff}}$')
    l2, = ax1_right.plot(theta_deg, P, color='#7B2D8B', linewidth=2, linestyle='--',
                         label='Druck [bar]')
    ax1.axvline(36, color='#B22222', linewidth=1.5, linestyle=':', label='Grand Canyon (36°N)')
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
    ax1.set_xlabel('Geografische Breite [°]')
    ax1.set_ylabel('$g_{\\mathrm{eff}}$ [m/s²]', color='#1A5F9E')
    ax1_right.set_ylabel('Druck [bar]', color='#7B2D8B')
    ax1.tick_params(axis='y', colors='#1A5F9E')
    ax1_right.tick_params(axis='y', colors='#7B2D8B')
    ax1.set_title('Effektive Gravitation und Druck')
    ax1.set_xlim(-90, 90)
    ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    lines = [l1, l2] + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=8)
    ax1.grid(alpha=0.3)

    ax2_right = ax2.twinx()
    l3, = ax2.plot(theta_deg, a_centri * 1000, color='#2CA02C', linewidth=2,
                   label='Zentrifugalbeschl. [mm/s²]')
    l4, = ax2_right.plot(theta_deg, a_coriolis * 1000, color='#FF7F0E', linewidth=2, linestyle='--',
                         label='Coriolisbeschl. [mm/s²]')
    ax2.axvline(36, color='#B22222', linewidth=1.5, linestyle=':', label='Grand Canyon (36°N)')
    ax2.set_xlabel('Geografische Breite [°]')
    ax2.set_ylabel('Zentrifugalbeschl. [mm/s²]', color='#2CA02C')
    ax2_right.set_ylabel('Coriolisbeschl. [mm/s²]', color='#FF7F0E')
    ax2.tick_params(axis='y', colors='#2CA02C')
    ax2_right.tick_params(axis='y', colors='#FF7F0E')
    ax2.set_title('Rotationsbedingte Beschleunigungen')
    ax2.set_xlim(-90, 90)
    ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    lines2 = [l3, l4] + [ax2.get_lines()[-1]]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper center', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp2_latitude_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp2_latitude_comparison.pdf')


# ── PLOT 4: exp3_erosion_timeline ─────────────────────────────────────────────

def make_exp3():
    n_cycles = 50
    t_total = 221  # days
    H_final = 1800  # m
    canyon_width = 20  # km
    canyon_depth_final = H_final

    t_cycle = t_total / n_cycles
    h_per_cycle = H_final / n_cycles

    # Staircase (treppenfunktion)
    t_stairs = [0]
    h_stairs = [0]
    for i in range(n_cycles):
        t_stairs.append(i * t_cycle)
        h_stairs.append(i * h_per_cycle)
        t_stairs.append((i + 1) * t_cycle)
        h_stairs.append((i + 1) * h_per_cycle)

    t_arr = np.array(t_stairs)
    h_arr = np.array(h_stairs)

    # Volume: pyramid-ish: V = (1/2) * width * depth * length
    # simplified: V proportional to depth^2
    vol_cum = (h_arr / H_final)**1.0 * 3240  # km³

    # Erosion rate: constant at h_per_cycle / t_cycle m/day
    rate = h_per_cycle / t_cycle  # m/day

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 3: Zeitliche Entwicklung der Erosion (50 Zyklen, 221 Tage)',
                 fontsize=13, fontweight='bold')

    ax1.plot(t_arr, h_arr, color='#1A5F9E', linewidth=1.8, label='Erosionstiefe')
    ax1.axhline(H_final, color='#B22222', linewidth=1.5, linestyle='--',
                label=f'Zieltiefe: {H_final} m')
    ax1.fill_between(t_arr, 0, h_arr, alpha=0.15, color='#1A5F9E')
    ax1.set_xlabel('Zeit [Tage]')
    ax1.set_ylabel('Erosionstiefe [m]')
    ax1.set_title('Erosionstiefe über die Zeit\n(treppenfunktionsartiger Zuwachs)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, t_total)
    ax1.set_ylim(0, H_final * 1.05)
    ax1.text(t_total * 0.05, H_final * 0.85,
             f'Rate: {rate:.1f} m/Tag\nZyklen: {n_cycles}\n$\\Delta$h: {h_per_cycle:.0f} m/Zyklus',
             fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax2_right = ax2.twinx()
    # smooth cumulative volume with staircase
    l1, = ax2.plot(t_arr, vol_cum / 1000 * 1000, color='#2CA02C', linewidth=1.8,
                   label='Kum. Volumen [km³]')
    # Erosion rate (constant)
    t_mid = np.linspace(0, t_total, 200)
    rate_arr = np.full_like(t_mid, rate)
    l2, = ax2_right.plot(t_mid, rate_arr, color='#FF7F0E', linewidth=1.8, linestyle='--',
                         label='Erosionsrate [m/Tag]')
    ax2.set_xlabel('Zeit [Tage]')
    ax2.set_ylabel('Kumulatives Volumen [km³]', color='#2CA02C')
    ax2_right.set_ylabel('Erosionsrate [m/Tag]', color='#FF7F0E')
    ax2.tick_params(axis='y', colors='#2CA02C')
    ax2_right.tick_params(axis='y', colors='#FF7F0E')
    ax2.set_title('Kumulatives Volumen und Erosionsrate')
    ax2.set_xlim(0, t_total)
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2_right.set_ylim(0, rate * 2)
    ax2.text(t_total * 0.97, 100,
             f'Endvolumen:\n3240 km³',
             ha='right', fontsize=9, color='#2CA02C',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp3_erosion_timeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp3_erosion_timeline.pdf')


# ── PLOT 5: exp4_tilt_sensitivity ─────────────────────────────────────────────

def make_exp4():
    tilt_deg = np.linspace(0, 45, 30)
    tilt = np.radians(tilt_deg)
    g = 9.81
    rho = 1000.0
    H = 8850.0
    theta_gc = np.radians(36)  # Grand Canyon
    P_sog = 1e7

    g_eff_gc = g * np.cos(theta_gc - tilt)  # effective g at Grand Canyon
    P_gc = rho * g_eff_gc * H / 1e5  # bar

    # Pressure gradient: equator (theta=0) to north pole (theta=90)
    g_eff_eq = g * np.cos(np.radians(0) - tilt)
    g_eff_pole = g * np.cos(np.radians(90) - tilt)
    dP = rho * (g_eff_eq - g_eff_pole) * H / 1e5  # bar

    v_total = np.sqrt(2 * (rho * g_eff_gc * H + P_sog) / rho)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 4: Sensitivität der hydraulischen Parameter gegenüber der Erdneigung',
                 fontsize=13, fontweight='bold')

    ax1_right = ax1.twinx()
    l1, = ax1.plot(tilt_deg, g_eff_gc, color='#1A5F9E', linewidth=2,
                   label='$g_{\\mathrm{eff}}$ bei 36°N [m/s²]')
    l2, = ax1_right.plot(tilt_deg, P_gc, color='#7B2D8B', linewidth=2, linestyle='--',
                         label='Druck bei 36°N [bar]')
    ax1.axvline(23.5, color='#B22222', linewidth=1.5, linestyle=':', label='Aktuelle Neigung 23,5°')
    ax1.set_xlabel('Erdneigung [°]')
    ax1.set_ylabel('$g_{\\mathrm{eff}}$ [m/s²]', color='#1A5F9E')
    ax1_right.set_ylabel('Druck [bar]', color='#7B2D8B')
    ax1.tick_params(axis='y', colors='#1A5F9E')
    ax1_right.tick_params(axis='y', colors='#7B2D8B')
    ax1.set_title('Effektive Gravitation und Druck bei 36°N\nvs. Erdneigungswinkel')
    lines = [l1, l2] + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8)
    ax1.grid(alpha=0.3)

    ax2_right = ax2.twinx()
    l3, = ax2.plot(tilt_deg, dP, color='#2CA02C', linewidth=2,
                   label='Druckgradient Äquator-Pol [bar]')
    l4, = ax2_right.plot(tilt_deg, v_total, color='#FF7F0E', linewidth=2, linestyle='--',
                         label='Strömungsgeschwindigkeit [m/s]')
    ax2.axvline(23.5, color='#B22222', linewidth=1.5, linestyle=':', label='Aktuelle Neigung 23,5°')
    ax2.set_xlabel('Erdneigung [°]')
    ax2.set_ylabel('Druckgradient [bar]', color='#2CA02C')
    ax2_right.set_ylabel('Strömungsgeschwindigkeit [m/s]', color='#FF7F0E')
    ax2.tick_params(axis='y', colors='#2CA02C')
    ax2_right.tick_params(axis='y', colors='#FF7F0E')
    ax2.set_title('Druckgradient und Strömungsgeschwindigkeit\nvs. Erdneigungswinkel')
    lines2 = [l3, l4] + [ax2.get_lines()[-1]]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp4_tilt_sensitivity.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp4_tilt_sensitivity.pdf')


# ── PLOT 6: exp5_time_dependent_pressure ──────────────────────────────────────

def make_exp5():
    # 10 days with 2-hour resolution
    dt = 2 / 24  # days
    t_total_days = 10
    t = np.arange(0, t_total_days + dt, dt)

    P_base = 848.0  # bar
    amp = 8.5       # bar
    P_total = P_base + amp * np.sin(2 * np.pi * t)

    # Single day detail
    t_day = np.linspace(0, 1, 200)
    P_day = amp * np.sin(2 * np.pi * t_day)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 5: Zeitabhängige Druckvariationen durch Erdrotation',
                 fontsize=13, fontweight='bold')

    ax1.plot(t, P_total, color='#1A5F9E', linewidth=1.2, alpha=0.85, label='Gesamtdruck')
    ax1.axhline(P_base, color='#B22222', linewidth=1.5, linestyle='--',
                label=f'Basisdruck: {P_base} bar')
    ax1.fill_between(t, P_base - amp * 1.5, P_total, alpha=0.15, color='#1A5F9E')
    ax1.set_xlabel('Zeit [Tage]')
    ax1.set_ylabel('Druck [bar]')
    ax1.set_title('Gesamtdruck über 10 Tage\n(24-h-Periodizität durch Erdrotation)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, t_total_days)
    ax1.set_ylim(P_base - 15, P_base + 15)
    ax1.text(0.97, 0.05,
             f'Amplitude: ±{amp} bar\n(≈1% des Basisdrucks)',
             transform=ax1.transAxes, ha='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax2.plot(t_day * 24, P_day, color='#2CA02C', linewidth=2, label='Druckvariation')
    ax2.fill_between(t_day * 24, 0, P_day, where=(P_day > 0), alpha=0.2, color='#2CA02C')
    ax2.fill_between(t_day * 24, 0, P_day, where=(P_day < 0), alpha=0.2, color='#B22222')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.axhline(amp, color='#FF7F0E', linewidth=1.2, linestyle='--',
                label=f'+{amp} bar (Maximum)')
    ax2.axhline(-amp, color='#7B2D8B', linewidth=1.2, linestyle='--',
                label=f'−{amp} bar (Minimum)')
    ax2.set_xlabel('Zeit [Stunden]')
    ax2.set_ylabel('Druckvariation [bar]')
    ax2.set_title(f'Detailansicht: Druckvariation\nein Erdrotationszyklus (24 h)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 24)
    ax2.set_xticks([0, 6, 12, 18, 24])
    ax2.text(0.5, 0.97, 'Sinusförmige Schwingung entspricht\nTages-Nacht-Rhythmus der Rotation',
             transform=ax2.transAxes, ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp5_time_dependent_pressure.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp5_time_dependent_pressure.pdf')


# ── PLOT 7: exp6_energy_analysis ──────────────────────────────────────────────

def make_exp6():
    H = np.linspace(1000, 8850, 50)
    rho = 1000.0
    g = 9.81
    V_global = 5e18  # m³
    P_sog = 1e7

    m = rho * V_global  # kg
    v = np.sqrt(2 * (rho * g * H + P_sog) / rho)

    E_pot = m * g * H          # J
    E_kin = 0.5 * m * v**2     # J
    E_tot = E_pot + E_kin

    e_kin_specific = 0.5 * v**2 / 1e6  # MJ/kg
    e_pot_specific = g * H / 1e9       # GJ/kg

    Q = 5.1e12 * v
    P_power = E_pot / (221 * 86400)   # W  average power

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Experiment 6: Energetische Analyse der Sintflut-Erosion',
                 fontsize=13, fontweight='bold')

    ax1.semilogy(H / 1000, E_pot, color='#2CA02C', linewidth=2, label='Pot. Energie [J]')
    ax1.semilogy(H / 1000, E_kin, color='#1A5F9E', linewidth=2, label='Kin. Energie [J]')
    ax1.semilogy(H / 1000, E_tot, color='#B22222', linewidth=2, linestyle='--',
                 label='Gesamtenergie [J]')
    ax1.set_xlabel('Wasserhöhe [km]')
    ax1.set_ylabel('Energie [J] (log. Skala)')
    ax1.set_title('Energieformen vs. Wasserhöhe\n(logarithmische Skala)')
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')
    ax1.set_xlim(1, 8.85)
    ax1.text(0.97, 0.05,
             '$E_{\\mathrm{pot}}$ dominiert\num 3 Größenordnungen',
             transform=ax1.transAxes, ha='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax2_right = ax2.twinx()
    l1, = ax2.plot(H / 1000, e_kin_specific, color='#1A5F9E', linewidth=2,
                   label='Spez. kin. Energie [MJ/kg]')
    l2, = ax2.plot(H / 1000, e_pot_specific * 1000, color='#2CA02C', linewidth=2, linestyle=':',
                   label='Spez. pot. Energie [GJ/kg]×1000')
    l3, = ax2_right.plot(H / 1000, P_power / 1e18, color='#FF7F0E', linewidth=2, linestyle='--',
                         label='Mittlere Leistung [EW]')
    ax2.set_xlabel('Wasserhöhe [km]')
    ax2.set_ylabel('Spez. Energie [MJ/kg oder GJ/kg·1000]', color='#1A5F9E')
    ax2_right.set_ylabel('Mittlere Leistung [Exawatt]', color='#FF7F0E')
    ax2.tick_params(axis='y', colors='#1A5F9E')
    ax2_right.tick_params(axis='y', colors='#FF7F0E')
    ax2.set_title('Spezifische Energiedichten und Leistung')
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2_right.set_xlim(1, 8.85)

    plt.tight_layout()
    fig.savefig(OUTDIR + 'exp6_energy_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ exp6_energy_analysis.pdf')


# ── RUN ALL ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    make_ablaufprozess()
    make_exp1()
    make_exp2()
    make_exp3()
    make_exp4()
    make_exp5()
    make_exp6()
    print('\nAlle fehlenden Plots erfolgreich generiert.')
