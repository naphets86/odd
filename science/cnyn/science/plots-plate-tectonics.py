#!/usr/bin/env python3
"""
Generate matplotlib plots for the continental plate formation section.
Topic: Biblical Flood as driving mechanism for continental drift and plate formation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc, FancyArrow
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Pangaea vs. heutige Kontinente – Vor/Nach-Vergleich
# ─────────────────────────────────────────────────────────────────────────────
def plot_pangaea_split():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Pangaea-Aufspaltung durch katastrophale Sintflut-Dynamik', fontsize=14, fontweight='bold')

    # Simplified Pangaea shape (left)
    ax = axes[0]
    ax.set_facecolor('#1a6690')
    ax.set_title('Vor der Sintflut: Pangaea (ein Superkontinent)', fontsize=11, fontweight='bold')

    # Pangaea as approx. polygon (combined supercontinent)
    pangaea = plt.Polygon([
        (-0.35, -0.6), (-0.5, -0.2), (-0.55, 0.2), (-0.4, 0.6),
        (-0.1, 0.8), (0.2, 0.75), (0.45, 0.5), (0.5, 0.0),
        (0.35, -0.4), (0.1, -0.75), (-0.15, -0.8)
    ], closed=True, facecolor='#8B6914', edgecolor='#4a3508', linewidth=2)
    ax.add_patch(pangaea)

    # Label regions
    ax.text(-0.1, 0.45, 'Laurasia\n(Nord)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2d5a27', alpha=0.7))
    ax.text(-0.05, -0.2, 'Gondwana\n(Süd)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#4a7a40', alpha=0.7))

    # Arrows showing compression forces during flood
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ax.annotate('', xy=(0.7*np.cos(rad), 0.7*np.sin(rad)),
                    xytext=(0.95*np.cos(rad), 0.95*np.sin(rad)),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0, -1.05, 'Sintflut-Wasserdruckkräfte', ha='center', fontsize=9, color='red', style='italic')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.15, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Modern continents (right) – simplified positions
    ax2 = axes[1]
    ax2.set_facecolor('#1a6690')
    ax2.set_title('Nach der Sintflut: Heutige Kontinentalverteilung', fontsize=11, fontweight='bold')

    continents = {
        'Nordamerika': ([(-0.9,-0.1),(-0.95,0.3),(-0.7,0.6),(-0.5,0.55),(-0.4,0.3),(-0.45,-0.1),(-0.65,-0.3)], '#8B6914'),
        'Südamerika': ([(-0.55,-0.2),(-0.6,-0.5),(-0.45,-0.85),(-0.3,-0.90),(-0.2,-0.7),(-0.25,-0.4),(-0.35,-0.15)], '#6B8E23'),
        'Europa':     ([(0.1,0.5),(0.0,0.7),(0.2,0.8),(0.35,0.75),(0.4,0.55),(0.25,0.45)], '#CD853F'),
        'Afrika':     ([(0.05,0.3),(0.0,0.0),(0.05,-0.5),(0.25,-0.7),(0.4,-0.5),(0.45,0.0),(0.35,0.35)], '#D2691E'),
        'Asien':      ([(0.4,0.4),(0.35,0.7),(0.6,0.85),(0.9,0.75),(1.0,0.5),(0.85,0.2),(0.6,0.15),(0.5,0.3)], '#A0522D'),
        'Australien': ([(0.65,-0.4),(0.6,-0.7),(0.85,-0.75),(0.95,-0.55),(0.9,-0.35)], '#DAA520'),
        'Antarktis':  ([(-0.4,-1.0),(0.0,-1.05),(0.4,-1.0),(0.3,-0.85),(0.0,-0.8),(-0.3,-0.85)], '#E0E0E0'),
    }

    for name, (coords, color) in continents.items():
        poly = plt.Polygon(coords, closed=True, facecolor=color, edgecolor='#333', linewidth=1.2, alpha=0.9)
        ax2.add_patch(poly)
        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])
        ax2.text(cx, cy, name, ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))

    # Arrows showing divergence directions
    divergence = [
        ((-0.4, 0.2), (-0.55, 0.2)),
        ((-0.3, 0.1), (-0.35, -0.1)),
        ((0.0, 0.5), (-0.15, 0.55)),
        ((0.3, -0.1), (0.05, -0.05)),
        ((0.7, 0.3), (0.8, 0.15)),
    ]
    for start, end in divergence:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5, alpha=0.8))

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.15, 1.0)
    ax2.set_aspect('equal')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('platte_pangaea_vergleich.pdf', bbox_inches='tight')
    plt.close()
    print('platte_pangaea_vergleich.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Runaway Subduction – Druckverteilung und Plattenverdrängung
# ─────────────────────────────────────────────────────────────────────────────
def plot_runaway_subduction():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Runaway Subduction: Sintflut-induzierte Plattendynamik', fontsize=14, fontweight='bold')

    # Left: subduction velocity over time
    ax = axes[0]
    t = np.linspace(0, 40, 400)   # days
    # velocity profile: exponential runaway then decay
    v_sub = 800 * np.exp(-((t - 10)**2) / 20) + 50 * np.exp(-t / 30)
    ax.fill_between(t, v_sub, alpha=0.3, color='red')
    ax.plot(t, v_sub, 'r-', linewidth=2, label='Subduktionsgeschwindigkeit (cm/Jahr äquiv.)')
    ax.axvline(x=10, color='darkred', linestyle='--', alpha=0.7, label='Maximum der Flut')
    ax.axvline(x=5, color='blue', linestyle=':', alpha=0.7, label='Beginn Rücksog')
    ax.set_xlabel('Sintflut-Zeitverlauf (Tage)')
    ax.set_ylabel('Relative Subduktionsrate (normiert)')
    ax.set_title('Zeitverlauf der Subduktionsgeschwindigkeit')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 40)

    # Right: cross-section of subduction zone
    ax2 = axes[1]
    ax2.set_facecolor('#87CEEB')
    ax2.set_title('Schematischer Querschnitt: Sintflut-Subduktionszone', fontsize=10)

    # Ocean floor
    x_ocean = np.linspace(0, 10, 200)
    y_ocean = -0.5 + 0.1 * np.sin(x_ocean * 0.8)
    ax2.fill_between(x_ocean, y_ocean, -3, color='#4682B4', alpha=0.5, label='Ozean')

    # Subducting plate
    x_sub = np.linspace(2, 8, 100)
    y_sub_top = -0.5 - (x_sub - 2) * 0.35
    y_sub_bot = y_sub_top - 0.3
    ax2.fill_between(x_sub, y_sub_top, y_sub_bot, color='#8B4513', alpha=0.9, label='Subduzierende Platte')

    # Overriding plate
    x_over = np.linspace(4, 10, 100)
    y_over = np.where(x_over < 5, -0.5, -0.5 + (x_over - 5) * 0.25)
    ax2.fill_between(x_over, y_over, y_over + 0.5, color='#6B8E23', alpha=0.9, label='Kontinentale Platte')

    # Mantle
    ax2.fill_between(x_ocean, -3, -2.5, color='#ff6b35', alpha=0.3, label='Heißer Mantel')

    # Arrows for movement
    ax2.annotate('', xy=(3, -1.5), xytext=(1.5, -0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax2.annotate('', xy=(8.5, 0.5), xytext=(7.5, 0.2),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))

    ax2.text(2.5, -2.2, 'Sintflut-Wassergewicht\nbeschleunigt Subduktion', fontsize=8,
             ha='center', color='#4682B4', fontweight='bold')
    ax2.text(5.5, 1.2, 'Gebirgsbildung\n(Orogenie)', fontsize=8, ha='center',
             color='darkgreen', fontweight='bold')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(-3, 1.8)
    ax2.legend(loc='upper left', fontsize=7)
    ax2.set_xlabel('Horizontale Distanz (schematisch)')
    ax2.set_ylabel('Tiefe (schematisch)')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('platte_subduktion.pdf', bbox_inches='tight')
    plt.close()
    print('platte_subduktion.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Sintflut-Wasserdruck als Antriebskraft der Plattenbewegung
# ─────────────────────────────────────────────────────────────────────────────
def plot_water_pressure_force():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Sintflut-Druckkräfte als Antrieb der Plattentektonik', fontsize=14, fontweight='bold')

    # 1) pressure vs depth
    ax = axes[0, 0]
    d = np.linspace(0, 4000, 500)   # depth in meters
    p = 1025 * 9.81 * d / 1e6        # MPa
    ax.plot(p, d, 'b-', linewidth=2)
    ax.invert_yaxis()
    ax.set_xlabel('Hydrostatischer Druck (MPa)')
    ax.set_ylabel('Wassertiefe (m)')
    ax.set_title('Sintflut: Druckzunahme mit Tiefe')
    ax.axhline(y=2000, color='red', linestyle='--', alpha=0.7, label='Typische Sintfluttiefe')
    ax.fill_betweenx(d, 0, p, alpha=0.2, color='blue')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) force on plate area
    ax2 = axes[0, 1]
    plate_areas = np.linspace(1e12, 1e14, 100)   # m²
    flood_depth_avg = 2000   # m
    pressure = 1025 * 9.81 * flood_depth_avg   # Pa
    force_N = pressure * plate_areas
    force_EJ = force_N * 1e-18   # exanewton-meters (not Joule, just scaling)
    ax2.loglog(plate_areas / 1e12, force_N / 1e18, 'r-', linewidth=2)
    ax2.set_xlabel('Plattenfläche (×10¹² m²)')
    ax2.set_ylabel('Druckkraft (×10¹⁸ N)')
    ax2.set_title('Gesamtdruckkraft auf Kontinentalplatten')
    current_plates = {'Pazifisch': 10.3, 'Nordamerikanisch': 7.59, 'Eurasisch': 6.78,
                      'Afrikanisch': 6.1, 'Antarktisch': 6.09, 'Indo-Australisch': 5.86}
    for name, area_1e12 in current_plates.items():
        f = pressure * area_1e12 * 1e12
        ax2.scatter(area_1e12, f / 1e18, zorder=5, s=50)
        ax2.annotate(name, (area_1e12, f / 1e18), textcoords='offset points',
                    xytext=(5, 3), fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3) Plate velocity simulation during flood
    ax3 = axes[1, 0]
    t = np.linspace(0, 370, 1000)   # days of flood year
    flood_active = (t >= 0) & (t <= 150)
    v_drift = np.zeros_like(t)
    v_drift[flood_active] = 50 * np.sin(np.pi * t[flood_active] / 150) * \
                             np.exp(-t[flood_active] / 80)
    v_drift[~flood_active] = 50 * np.exp(-(t[~flood_active] - 150) / 60) * 0.3

    ax3.fill_between(t, v_drift, alpha=0.3, color='orange')
    ax3.plot(t, v_drift, 'darkorange', linewidth=2, label='Plattendrift-Geschwindigkeit')
    ax3.axvspan(0, 150, alpha=0.1, color='blue', label='Aktive Flutphase')
    ax3.axvspan(150, 370, alpha=0.1, color='green', label='Rückzugsphase')
    ax3.set_xlabel('Sintflut-Tag')
    ax3.set_ylabel('Relative Driftgeschwindigkeit (cm/Jahr, normiert)')
    ax3.set_title('Plattenbewegungsgeschwindigkeit während der Sintflut')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 370)

    # 4) Heat flux map (schematic)
    ax4 = axes[1, 1]
    x = np.linspace(-180, 180, 360)
    y = np.linspace(-90, 90, 180)
    X, Y = np.meshgrid(x, y)
    # Simulate heat flux: higher at mid-ocean ridges and subduction zones
    heat = (40 + 80 * np.exp(-((np.abs(X) - 10)**2) / 1000) +
            60 * np.exp(-((np.abs(Y) - 45)**2) / 400) +
            30 * np.random.RandomState(42).rand(*X.shape))
    im = ax4.contourf(X, Y, heat, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax4, label='Wärmefluss (mW/m²)')
    ax4.set_title('Sintflut-induzierter Wärmefluss\n(Mantelkonvektion, schematisch)')
    ax4.set_xlabel('Längengrad')
    ax4.set_ylabel('Breitengrad')
    # Mark approximate plate boundaries
    ax4.axvline(x=-40, color='cyan', linewidth=1.5, alpha=0.7, linestyle='--', label='Plattengrenzen')
    ax4.axvline(x=60, color='cyan', linewidth=1.5, alpha=0.7, linestyle='--')
    ax4.axhline(y=0, color='cyan', linewidth=1, alpha=0.5, linestyle=':')
    ax4.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig('platte_druckkraefte.pdf', bbox_inches='tight')
    plt.close()
    print('platte_druckkraefte.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Plattentrennung – zeitlicher Ablauf der Kontinentaldrift
# ─────────────────────────────────────────────────────────────────────────────
def plot_continental_drift_timeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Zeitlicher Ablauf der Kontinentaldrift nach der Sintflut\n'
                 '(Baumgardner-Modell: katastrophale Plattentektonik)', fontsize=13, fontweight='bold')

    # Timeline events
    events = [
        (0,   'Sintflut\nbeginn', '#e74c3c'),
        (40,  'Maximale\nFluttiefe', '#c0392b'),
        (110, 'Rücksog\nbeginn', '#3498db'),
        (150, 'Hauptphase\nPlattenrift', '#9b59b6'),
        (220, 'Atlantik\nöffnet sich', '#1abc9c'),
        (370, 'Flut endet\n(Noahs Jahr)', '#27ae60'),
        (1000,'Post-Flut:\nStabilisierung', '#f39c12'),
    ]

    ax.set_facecolor('#f8f9fa')
    # Draw timeline
    ax.axhline(y=0.5, xmin=0.02, xmax=0.98, color='#2c3e50', linewidth=3, zorder=2)

    for i, (t, label, color) in enumerate(events):
        y_pos = 0.8 if i % 2 == 0 else 0.2
        ax.scatter(t, 0.5, s=120, color=color, zorder=5, edgecolors='white', linewidth=2)
        ax.plot([t, t], [0.5, y_pos], color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        ax.text(t, y_pos + (0.07 if y_pos > 0.5 else -0.07), label,
               ha='center', va='bottom' if y_pos > 0.5 else 'top',
               fontsize=8.5, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.85))

    # Phase shading
    ax.axvspan(0, 150, alpha=0.08, color='blue', label='Aktive Flutphase (150 Tage)')
    ax.axvspan(150, 370, alpha=0.08, color='green', label='Rückzugsphase (220 Tage)')
    ax.axvspan(370, 1200, alpha=0.08, color='orange', label='Post-Flut Stabilisierung')

    # Annotations for drift rates
    ax.annotate('Drift: bis zu\n10 m/Tag', xy=(220, 0.5),
                xytext=(280, 0.78),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5),
                fontsize=9, color='#9b59b6', fontweight='bold')
    ax.annotate('Drift heute:\n~2–7 cm/Jahr', xy=(800, 0.5),
                xytext=(650, 0.22),
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5),
                fontsize=9, color='#f39c12', fontweight='bold')

    ax.set_xlim(-30, 1200)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Tage ab Sintflutbeginn (logarithmisch extrapoliert)', fontsize=10)
    ax.set_yticks([])
    ax.legend(loc='center right', fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig('platte_drift_zeitlinie.pdf', bbox_inches='tight')
    plt.close()
    print('platte_drift_zeitlinie.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Mantelkonvektion und Wärmefluss – Sintflut-Modell
# ─────────────────────────────────────────────────────────────────────────────
def plot_mantle_convection():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle('Sintflut-beschleunigte Mantelkonvektion als Plattentektonik-Antrieb',
                 fontsize=13, fontweight='bold')

    # Left: convection cells schematic
    ax = axes[0]
    ax.set_facecolor('#1a0a00')
    ax.set_title('Mantelkonvektionszellen während der Sintflut', fontsize=10)

    # Layers
    crust_y = 0.85
    mantle_mid = 0.5
    core_y = 0.05

    # Color gradient for mantle
    mantle_grad = np.linspace(0, 1, 100).reshape(100, 1)
    ax.imshow(mantle_grad, extent=[0, 10, core_y, crust_y], aspect='auto',
             cmap='hot', alpha=0.4, origin='lower')

    # Crust
    ax.fill_between([0, 10], [crust_y, crust_y], [1.0, 1.0], color='#5D4037', alpha=0.9)
    ax.text(5, 0.93, 'Erdkruste / Lithosphäre', ha='center', color='white', fontsize=9, fontweight='bold')

    # Core
    ax.fill_between([0, 10], [0, 0], [core_y, core_y], color='#ff4500', alpha=0.8)
    ax.text(5, 0.025, 'Äußerer Kern (hochtemperatur)', ha='center', color='white', fontsize=8, fontweight='bold')

    # Convection arrows (upwelling and downwelling)
    arrow_style = dict(arrowstyle='->', color='yellow', lw=2)
    down_style = dict(arrowstyle='->', color='cyan', lw=2)

    # Upwelling currents
    for x_pos in [2, 6]:
        for y_start, y_end in zip([0.1, 0.35, 0.55], [0.3, 0.52, 0.75]):
            ax.annotate('', xy=(x_pos, y_end), xytext=(x_pos, y_start),
                       arrowprops=arrow_style)
        ax.text(x_pos, 0.42, 'Aufstieg\n(heiß)', ha='center', va='center', fontsize=7.5,
               color='yellow', fontweight='bold')

    # Downwelling currents
    for x_pos in [4, 8]:
        for y_start, y_end in zip([0.75, 0.52, 0.30], [0.57, 0.35, 0.12]):
            ax.annotate('', xy=(x_pos, y_end), xytext=(x_pos, y_start),
                       arrowprops=down_style)
        ax.text(x_pos, 0.42, 'Abstieg\n(kühl)', ha='center', va='center', fontsize=7.5,
               color='cyan', fontweight='bold')

    # Horizontal flow arrows at top and bottom
    for x_start, x_end, y in [(1.5, 3.5, 0.82), (5.5, 7.5, 0.82),
                               (2.5, 4.5, 0.1), (6.5, 8.5, 0.1)]:
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.8))

    ax.text(5, 1.05, '← Plattenbewegung durch Manteldrag →', ha='center', fontsize=9,
           color='orange', fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.1)
    ax.axis('off')

    # Right: temperature profile
    ax2 = axes[1]
    depth_km = np.linspace(0, 2900, 300)
    # Temperature profile: geotherm
    T_surface = 15
    T_base = 4000
    T_mantle = T_surface + (T_base - T_surface) * (1 - np.exp(-depth_km / 500))
    T_flood_excess = 200 * np.exp(-depth_km / 300) * np.exp(-(depth_km - 800)**2 / 100000)
    T_with_flood = T_mantle + T_flood_excess

    ax2.plot(T_mantle, depth_km, 'b--', linewidth=2, label='Normales Geotherm')
    ax2.plot(T_with_flood, depth_km, 'r-', linewidth=2.5, label='Sintflut-verstärktes Geotherm')
    ax2.fill_betweenx(depth_km, T_mantle, T_with_flood, alpha=0.2, color='red',
                      label='Sintflut-Wärmeüberschuss')

    ax2.invert_yaxis()
    ax2.set_xlabel('Temperatur (°C)')
    ax2.set_ylabel('Tiefe (km)')
    ax2.set_title('Temperaturprofil: Normal vs. Sintflut-Geotherm')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 4500)

    plt.tight_layout()
    plt.savefig('platte_mantelkonvektion.pdf', bbox_inches='tight')
    plt.close()
    print('platte_mantelkonvektion.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Sedimentschichten-Nachweis – Globale Verteilung
# ─────────────────────────────────────────────────────────────────────────────
def plot_sediment_global():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Globale Sedimentverteilung als Beleg für die Sintflut', fontsize=13, fontweight='bold')

    # Left: sediment thickness heatmap
    ax = axes[0]
    x = np.linspace(-180, 180, 360)
    y = np.linspace(-90, 90, 180)
    X, Y = np.meshgrid(x, y)
    rng = np.random.RandomState(42)

    # Simulate sediment thickness: thin on young ocean crust, thick at continental margins
    sed = (8 + 4 * np.exp(-((np.abs(X) - 90)**2) / 3000) +
           6 * np.exp(-((np.abs(Y) - 30)**2) / 500) +
           2 * rng.rand(*X.shape))
    # Thinner at mid-ocean ridges
    sed -= 5 * np.exp(-((np.abs(X) - 0)**2 + np.abs(X) * 0.5) / 1000)
    sed = np.clip(sed, 0, None)

    im = ax.contourf(X, Y, sed, levels=15, cmap='YlOrBr')
    plt.colorbar(im, ax=ax, label='Sedimentmächtigkeit (km, schematisch)')
    ax.set_title('Globale Sedimentmächtigkeit\n(Sintflut-Ablagerungen)')
    ax.set_xlabel('Längengrad')
    ax.set_ylabel('Breitengrad')
    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)

    # Right: bar chart of sediment volumes by region
    ax2 = axes[1]
    regions = ['Nordamerika', 'Südamerika', 'Europa', 'Afrika', 'Asien', 'Australien', 'Antarktis']
    # km³ (rough estimates, for illustration)
    total_sed = [6.8, 4.2, 3.1, 5.5, 9.3, 2.4, 1.7]
    flood_fraction = [0.72, 0.65, 0.58, 0.70, 0.68, 0.55, 0.80]  # fraction attributed to flood
    flood_vol = [t * f for t, f in zip(total_sed, flood_fraction)]
    post_vol  = [t * (1 - f) for t, f in zip(total_sed, flood_fraction)]

    bars1 = ax2.bar(regions, flood_vol, label='Sintflut-Sediment (geschätzt)', color='#e67e22', alpha=0.85)
    bars2 = ax2.bar(regions, post_vol, bottom=flood_vol, label='Post-Flut Sediment', color='#95a5a6', alpha=0.85)

    ax2.set_title('Sedimentvolumen nach Region\n(Sintflut-Anteil hervorgehoben)')
    ax2.set_ylabel('Sedimentvolumen (×10⁶ km³, schematisch)')
    ax2.set_xticklabels(regions, rotation=35, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, frac in zip(bars1, flood_fraction):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{frac*100:.0f}%', ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('platte_sediment_global.pdf', bbox_inches='tight')
    plt.close()
    print('platte_sediment_global.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Energiebilanz – Sintflut als Antrieb der Tektonik
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy_balance():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Energiebilanz: Sintflut-Energie und tektonische Umformungsprozesse',
                 fontsize=13, fontweight='bold')

    # 1) Energy comparison
    ax = axes[0, 0]
    processes = [
        'Sintflut\n(Gesamtenergie)', 'Jahreserdbeben\n(weltweit)',
        'Hiroshima-Bombe', 'Krakatau 1883', 'Chicxulub\nImpakt'
    ]
    energies_J = [1e32, 1e18, 6.3e13, 2e19, 4.2e23]
    colors_e = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.barh(processes, energies_J, color=colors_e, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Energie (Joule, logarithmisch)')
    ax.set_title('Vergleich Energiequellen (Sintflut dominiert)')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, energies_J):
        ax.text(val * 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1e} J', va='center', fontsize=7.5)

    # 2) Plate kinetic energy during flood
    ax2 = axes[0, 1]
    masses_kg = {'Pazifisch': 2.2e21, 'Nordamerikanisch': 1.6e21,
                 'Eurasisch': 1.4e21, 'Afrikanisch': 1.3e21}
    v_flood_ms = 0.1   # 10 cm per second during peak
    v_today_ms = 7e-11 # ~2 cm/year
    plates = list(masses_kg.keys())
    KE_flood = [0.5 * m * v_flood_ms**2 for m in masses_kg.values()]
    KE_today = [0.5 * m * v_today_ms**2 for m in masses_kg.values()]

    x_pos = np.arange(len(plates))
    ax2.bar(x_pos - 0.2, KE_flood, 0.35, label='Sintflut (10 cm/s)', color='red', alpha=0.8)
    ax2.bar(x_pos + 0.2, KE_today, 0.35, label='Heute (2 cm/Jahr)', color='blue', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plates, rotation=20, ha='right', fontsize=8)
    ax2.set_yscale('log')
    ax2.set_ylabel('Kinetische Energie (J, log)')
    ax2.set_title('Kinetische Plattenenergie: Sintflut vs. Heute')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # 3) Rift opening rate
    ax3 = axes[1, 0]
    t_days = np.linspace(0, 370, 1000)
    rift_rate = (200 * np.exp(-((t_days - 100)**2) / 3000) +
                 50 * np.exp(-t_days / 150))
    cumulative_rift = np.cumsum(rift_rate) * (370 / 1000)  # km accumulated
    ax3.fill_between(t_days, rift_rate, alpha=0.3, color='red')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_days, cumulative_rift, 'b-', linewidth=2, label='Kumulativ (km)')
    ax3.plot(t_days, rift_rate, 'r-', linewidth=2, label='Riftrate (km/Tag)')
    ax3.set_xlabel('Sintflut-Tag')
    ax3.set_ylabel('Riftöffnungsrate (km/Tag)', color='red')
    ax3_twin.set_ylabel('Kumulative Öffnung (km)', color='blue')
    ax3.set_title('Kontinentalrift-Öffnungsrate am Atlantik')
    ax3.legend(loc='upper left', fontsize=8)
    ax3_twin.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4) Flood impact pie chart
    ax4 = axes[1, 1]
    contributions = {
        'Hydrostatischer\nDruck': 35,
        'Thermische\nMantelkonvektion': 25,
        'Runaway\nSubduktion': 20,
        'Seismische\nFreisetzung': 12,
        'Sonstige\nKräfte': 8,
    }
    colors_pie = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#95a5a6']
    wedges, texts, autotexts = ax4.pie(
        contributions.values(),
        labels=contributions.keys(),
        autopct='%1.0f%%',
        colors=colors_pie,
        startangle=140,
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')
    ax4.set_title('Antriebskräfte der Sintflut-Tektonik\n(Energieanteile, schematisch)')

    plt.tight_layout()
    plt.savefig('platte_energiebilanz.pdf', bbox_inches='tight')
    plt.close()
    print('platte_energiebilanz.pdf ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: Fit-Nachweis – Küstenlinien-Kongruenz Pangaea
# ─────────────────────────────────────────────────────────────────────────────
def plot_coastline_fit():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Geometrische Kongruenz der Kontinentalränder – Beleg für ursprüngliche Einheit',
                 fontsize=12, fontweight='bold')

    def make_s_america(offset_x=0, offset_y=0):
        x = np.array([-0.2, -0.35, -0.4, -0.32, -0.2, -0.05, 0.05, 0.1, 0.05]) + offset_x
        y = np.array([0.5,  0.3,   -0.1, -0.5,  -0.8, -0.9,  -0.6, 0.0, 0.4]) + offset_y
        return x, y

    def make_africa(offset_x=0, offset_y=0):
        x = np.array([0.05, -0.05, -0.15, -0.1, 0.05, 0.25, 0.3, 0.2, 0.1]) + offset_x
        y = np.array([0.5,  0.3,   -0.05, -0.4, -0.7, -0.5, 0.0, 0.35, 0.6]) + offset_y
        return x, y

    for idx, (ax, title, gap) in enumerate(zip(axes,
        ['Heute: Atlantischer Ozean\ntrennt Südamerika und Afrika',
         'Zusammengesetzt:\nÜberlappungsnachweis',
         'Kongruenz-Analyse:\nFormübereinstimmung'],
        [0.8, 0.0, None])):

        ax.set_facecolor('#1a6690' if idx != 2 else 'white')
        ax.set_title(title, fontsize=9, fontweight='bold')

        if idx < 2:
            if idx == 0:
                sa_ox = -0.5 - gap / 2
                af_ox = 0.0 + gap / 2
            else:
                sa_ox = -0.3
                af_ox = -0.02
            sx, sy = make_s_america(sa_ox)
            ax.fill(sx, sy, color='#6B8E23', alpha=0.9, label='Südamerika')
            ax.fill(*make_africa(af_ox), color='#D2691E', alpha=0.9, label='Afrika')
            if idx == 0:
                # Ocean between them
                ax.text(0, -0.2, 'Atlantischer\nOzean\n(~4000 km)', ha='center', fontsize=8,
                       color='white', fontweight='bold')
            else:
                ax.text(0, -1.0, 'Deckungsgleich!\nNachweis gemeinsamer Herkunft', ha='center',
                       fontsize=8.5, color='yellow', fontweight='bold')
        else:
            # Correlation plot
            sa_east = np.array([-0.2, -0.05, 0.05, 0.1, 0.05])
            af_west = np.array([-0.15, -0.1, 0.05, 0.25, 0.3]) - 0.35
            corr_val = np.corrcoef(sa_east, af_west)[0, 1]
            lat = np.linspace(-40, 20, 50)
            offset_measured = 0.05 * np.random.RandomState(1).randn(50)
            ax.plot(lat, offset_measured, 'o', markersize=4, color='blue', alpha=0.6)
            ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label=f'Perfekte Übereinstimmung\nKorrelation r = {corr_val:.3f}')
            ax.fill_between(lat, -0.15, 0.15, alpha=0.1, color='green', label='±150 km Toleranzband')
            ax.set_xlabel('Breitengrad')
            ax.set_ylabel('Abweichung (normiert)')
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')

        ax.set_xlim(-1.2, 1.0)
        ax.set_ylim(-1.1, 0.8)
        if idx < 2:
            ax.legend(loc='upper right', fontsize=7.5)
            ax.axis('off' if idx < 2 else 'on')
    axes[2].set_xlim(-45, 25)

    plt.tight_layout()
    plt.savefig('platte_kuestenlinien.pdf', bbox_inches='tight')
    plt.close()
    print('platte_kuestenlinien.pdf ✓')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('Generiere Plattentektonik-Plots...')
    plot_pangaea_split()
    plot_runaway_subduction()
    plot_water_pressure_force()
    plot_continental_drift_timeline()
    plot_mantle_convection()
    plot_sediment_global()
    plot_energy_balance()
    plot_coastline_fit()
    print('\nAlle Plots erfolgreich erstellt.')
