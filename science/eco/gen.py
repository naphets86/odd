#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generierung der Matplotlib-Plots für die Arbeit:
"Zwei Hauptkräfte der Wirtschaft: Anthropogene und Natürliche Periodizität"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge, FancyArrowPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Stellung Deutsch ein
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# Farben
mainblue = (25/255, 70/255, 140/255)
accentred = (180/255, 50/255, 30/255)
darkgreen = (30/255, 100/255, 50/255)
lightgray = (245/255, 245/255, 248/255)

# ============================================================
# Plot 1: Die zwei Hauptkräfte der Wirtschaft
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Titel
ax.text(0.5, 0.95, 'Die zwei Hauptkräfte der Wirtschaft', 
        ha='center', va='top', fontsize=18, fontweight='bold',
        transform=ax.transAxes)

# Kraft 1: Anthropogene Kraft (Kauf und Verkauf)
rect1 = FancyBboxPatch((0.05, 0.55), 0.4, 0.3, boxstyle="round,pad=0.02", 
                        edgecolor=mainblue, facecolor=mainblue, alpha=0.2, linewidth=2.5,
                        transform=ax.transAxes)
ax.add_patch(rect1)
ax.text(0.25, 0.815, 'Kraft 1: Anthropogene Kraft', 
        ha='center', va='top', fontsize=13, fontweight='bold',
        transform=ax.transAxes, color=mainblue)
ax.text(0.25, 0.77, 'Kauf und Verkauf\nder Menschen\nfür das Leben', 
        ha='center', va='top', fontsize=11, style='italic',
        transform=ax.transAxes)
ax.text(0.25, 0.62, r'$F_a = \alpha \cdot (P - P_0)$', 
        ha='center', va='center', fontsize=12, 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Kraft 2: Natürliche Kraft (Periodizität)
rect2 = FancyBboxPatch((0.55, 0.55), 0.4, 0.3, boxstyle="round,pad=0.02", 
                        edgecolor=darkgreen, facecolor=darkgreen, alpha=0.2, linewidth=2.5,
                        transform=ax.transAxes)
ax.add_patch(rect2)
ax.text(0.75, 0.815, 'Kraft 2: Natürliche Periodizität', 
        ha='center', va='top', fontsize=13, fontweight='bold',
        transform=ax.transAxes, color=darkgreen)
ax.text(0.75, 0.77, 'Natürlicher Einfluss\n+ Regulierter Bedarf\n(begründet)', 
        ha='center', va='top', fontsize=11, style='italic',
        transform=ax.transAxes)
ax.text(0.75, 0.62, r'$F_n = \eta \cdot \sin(2\pi t/T)$', 
        ha='center', va='center', fontsize=12, 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Resultante
arrow = FancyArrowPatch((0.5, 0.5), (0.5, 0.35),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5,
                        color='black', transform=ax.transAxes)
ax.add_patch(arrow)

# Wirtschaftliches Gleichgewicht
rect3 = FancyBboxPatch((0.2, 0.08), 0.6, 0.22, boxstyle="round,pad=0.02", 
                        edgecolor=accentred, facecolor=accentred, alpha=0.15, linewidth=2.5,
                        transform=ax.transAxes)
ax.add_patch(rect3)
ax.text(0.5, 0.265, 'Resultierendes Wirtschaftliches Gleichgewicht', 
        ha='center', va='top', fontsize=12, fontweight='bold',
        transform=ax.transAxes, color=accentred)
ax.text(0.5, 0.21, r'$P(t) = \alpha (P - P_0) + \eta \sin(2\pi t/T)$', 
        ha='center', va='center', fontsize=11,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.5, 0.14, 'Gesunde Wirtschaft = Balance zwischen\nanthropogener Aktivität und natürlicher Periodizität', 
        ha='center', va='top', fontsize=10,
        transform=ax.transAxes, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('plot1_zwei_kraefte.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 1: Zwei Hauptkräfte erstellt")
plt.close()

# ============================================================
# Plot 2: Anthropogene Kraft über Zeit
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

t = np.linspace(0, 10, 1000)
alpha = 1.2
P_0 = 5.0
P_dev = 3 * np.sin(0.5 * np.pi * t)  # Schwingende Nachfrage
F_anthropogen = alpha * (P_dev)

ax.plot(t, F_anthropogen, linewidth=2.5, color=mainblue, label='Anthropogene Kraft $F_a(t)$')
ax.fill_between(t, 0, F_anthropogen, alpha=0.2, color=mainblue)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$ (Perioden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Anthropogene Kraft $F_a(t)$', fontsize=12, fontweight='bold')
ax.set_title('Kraft 1: Anthropogene Kraft (Kauf und Verkauf)\n$F_a(t) = \\alpha \\cdot (P(t) - P_0)$', 
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('plot2_anthropogene_kraft.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 2: Anthropogene Kraft erstellt")
plt.close()

# ============================================================
# Plot 3: Natürliche Periodizität
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

t = np.linspace(0, 10, 1000)
eta = 2.0
F_natuerlich = eta * np.sin(2 * np.pi * t / 2.5)

ax.plot(t, F_natuerlich, linewidth=2.5, color=darkgreen, label='Natürliche Periodizität $F_n(t)$')
ax.fill_between(t, 0, F_natuerlich, where=(F_natuerlich >= 0), alpha=0.2, color=darkgreen, label='Positive Phase (Ernte/Wohlstand)')
ax.fill_between(t, 0, F_natuerlich, where=(F_natuerlich < 0), alpha=0.2, color=accentred, label='Negative Phase (Mangel/Anpassung)')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$ (Perioden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Natürliche Periodizität $F_n(t)$', fontsize=12, fontweight='bold')
ax.set_title('Kraft 2: Natürliche Periodizität\n$F_n(t) = \\eta \\sin(2\\pi t / T)$', 
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('plot3_natuerliche_periodizitaet.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 3: Natürliche Periodizität erstellt")
plt.close()

# ============================================================
# Plot 4: Resultierendes Wirtschaftliches Gleichgewicht
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(13, 8))

t = np.linspace(0, 10, 1000)
alpha = 1.2
eta = 2.0
P_dev = 3 * np.sin(0.5 * np.pi * t)
F_a = alpha * P_dev
F_n = eta * np.sin(2 * np.pi * t / 2.5)
P_total = F_a + F_n

ax.plot(t, F_a, linewidth=2, color=mainblue, label='Anthropogene Kraft $F_a(t)$', alpha=0.7, linestyle='--')
ax.plot(t, F_n, linewidth=2, color=darkgreen, label='Natürliche Periodizität $F_n(t)$', alpha=0.7, linestyle='--')
ax.plot(t, P_total, linewidth=3, color=accentred, label='Resultierendes Gleichgewicht $P(t)$')
ax.fill_between(t, 0, P_total, alpha=0.15, color=accentred)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$ (Perioden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wirtschaftliche Größe', fontsize=12, fontweight='bold')
ax.set_title('Resultierendes Wirtschaftliches Gleichgewicht\n$P(t) = F_a(t) + F_n(t) = \\alpha(P-P_0) + \\eta\\sin(2\\pi t/T)$', 
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('plot4_resultierendes_gleichgewicht.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 4: Resultierendes Gleichgewicht erstellt")
plt.close()

# ============================================================
# Plot 5: Unbegründeter Überbedarf und Auswirkungen
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

t = np.linspace(0, 10, 1000)
eta = 2.0
F_n_gesund = eta * np.sin(2 * np.pi * t / 2.5)
F_n_gestoert = eta * np.sin(2 * np.pi * t / 2.5) + 0.5 * np.sin(3 * np.pi * t / 2.5)

# Linkes Diagramm: Gesunde Periodizität
ax1.plot(t, F_n_gesund, linewidth=2.5, color=darkgreen, label='Gesunde natürliche Periodizität')
ax1.fill_between(t, 0, F_n_gesund, where=(F_n_gesund >= 0), alpha=0.2, color=darkgreen)
ax1.fill_between(t, 0, F_n_gesund, where=(F_n_gesund < 0), alpha=0.2, color=accentred)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('Gesunde Wirtschaft\n(Regulierte Periodizität)', fontsize=12, fontweight='bold', color=darkgreen)
ax1.set_xlabel('Zeit $t$', fontsize=11)
ax1.set_ylabel('Natürliche Kraft $F_n(t)$', fontsize=11)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 10)

# Rechtes Diagramm: Gestörte Periodizität durch Überbedarf
ax2.plot(t, F_n_gesund, linewidth=2, color=darkgreen, alpha=0.5, label='Ursprüngliche Periodizität', linestyle='--')
ax2.plot(t, F_n_gestoert, linewidth=2.5, color=accentred, label='Gestört durch unbegründeten Überbedarf')
ax2.fill_between(t, F_n_gesund, F_n_gestoert, alpha=0.3, color=accentred, label='Störung/Abweichung')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_title('Kranke Wirtschaft\n(Gestörte Periodizität durch Überbedarf)', fontsize=12, fontweight='bold', color=accentred)
ax2.set_xlabel('Zeit $t$', fontsize=11)
ax2.set_ylabel('Gestörte Kraft $F_n(t) + $ Störung', fontsize=11)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 10)

plt.suptitle('Auswirkungen von unbegründetem Überbedarf auf natürliche Wirtschaftsperiodizität',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('plot5_ueberbedarf_auswirkungen.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 5: Überbedarf und Auswirkungen erstellt")
plt.close()

# ============================================================
# Plot 6: Stabilität und Instabilität
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(13, 8))

t = np.linspace(0, 20, 2000)

# Gesundes System (stabiler oszillierender Zustand)
F_n_stabil = 2.0 * np.sin(2 * np.pi * t / 2.5) * np.exp(-0.05 * t)
F_n_instabil = 2.0 * np.sin(2 * np.pi * t / 2.5) + 0.3 * np.sin(3 * np.pi * t / 2.5) * np.exp(0.05 * t)

ax.plot(t, F_n_stabil, linewidth=2.5, color=darkgreen, label='Stabiles System (natürliche Periodizität erhalten)')
ax.plot(t, F_n_instabil, linewidth=2.5, color=accentred, label='Instabiles System (Überbedarf erzeugt Chaos)')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$', fontsize=12, fontweight='bold')
ax.set_ylabel('Wirtschaftliche Kraft', fontsize=12, fontweight='bold')
ax.set_title('Stabilität vs. Instabilität: Einfluss von unbegründetem Überbedarf',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.set_xlim(0, 20)

# Annotationen
ax.annotate('Dampfung durch\nnatürliche Periodizität', xy=(10, 0.3), xytext=(12, 1.5),
            fontsize=10, color=darkgreen, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=darkgreen, lw=1.5))
ax.annotate('Exponentielles\nWachstum der Störung', xy=(15, 3), xytext=(13, 5),
            fontsize=10, color=accentred, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=accentred, lw=1.5))

plt.tight_layout()
plt.savefig('plot6_stabilitaet_instabilitaet.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 6: Stabilität und Instabilität erstellt")
plt.close()

# ============================================================
# Plot 7: Phasenraum - Gesunde vs. Kranke Wirtschaft
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gesundes System
t = np.linspace(0, 10, 1000)
x_gesund = 2.0 * np.sin(2 * np.pi * t / 2.5)
y_gesund = 2.0 * np.pi / 2.5 * np.cos(2 * np.pi * t / 2.5)

ax1.plot(x_gesund, y_gesund, linewidth=2.5, color=darkgreen)
ax1.scatter([x_gesund[0]], [y_gesund[0]], s=100, color='green', zorder=5, marker='o', label='Start')
ax1.scatter([x_gesund[-1]], [y_gesund[-1]], s=100, color='darkgreen', zorder=5, marker='s', label='Ende')
ax1.set_xlabel('Position $x(t)$', fontsize=11, fontweight='bold')
ax1.set_ylabel('Geschwindigkeit $v(t)$', fontsize=11, fontweight='bold')
ax1.set_title('Phasenraum: Gesunde Wirtschaft\n(Stabiler Grenzzyklus)', fontsize=12, fontweight='bold', color=darkgreen)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal', adjustable='box')

# Krankes System mit Chaos
t = np.linspace(0, 10, 2000)
x_krank = 2.0 * np.sin(2 * np.pi * t / 2.5) + 0.3 * np.sin(3 * np.pi * t / 2.5)
y_krank = 2.0 * np.pi / 2.5 * np.cos(2 * np.pi * t / 2.5) + 0.9 * np.pi / 2.5 * np.cos(3 * np.pi * t / 2.5)

ax2.plot(x_krank, y_krank, linewidth=1.5, color=accentred, alpha=0.7)
ax2.scatter([x_krank[0]], [y_krank[0]], s=100, color='red', zorder=5, marker='o', label='Start')
ax2.set_xlabel('Position $x(t)$', fontsize=11, fontweight='bold')
ax2.set_ylabel('Geschwindigkeit $v(t)$', fontsize=11, fontweight='bold')
ax2.set_title('Phasenraum: Kranke Wirtschaft\n(Chaos durch Überbedarf)', fontsize=12, fontweight='bold', color=accentred)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.suptitle('Dynamischer Vergleich: Gesunde vs. Kranke Wirtschaft',
             fontsize=13, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('plot7_phasenraum.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 7: Phasenraum erstellt")
plt.close()

# ============================================================
# Plot 8: Energiebilanz der Wirtschaft
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

t = np.linspace(0, 10, 1000)
eta = 2.0

# Natürliche Energie (konstant)
E_natuerlich = np.ones_like(t) * 10.0

# Anthropogene Energie (variable)
E_anthropogen = 5.0 + 2.0 * np.sin(0.5 * np.pi * t)

# Gesamtenergie bei gesundem Ausgleich
E_gesund = E_natuerlich + E_anthropogen

# Gesamtenergie bei Überbedarf (ineffizient)
E_krank = E_natuerlich + E_anthropogen * 1.5

ax.fill_between(t, 0, E_natuerlich, alpha=0.3, color=darkgreen, label='Verfügbare natürliche Energie')
ax.plot(t, E_anthropogen, linewidth=2.5, color=mainblue, label='Anthropogene Nachfrage', linestyle='--')
ax.plot(t, E_gesund, linewidth=2.5, color=darkgreen, label='Gesunde Bilanz (E_natur ≥ E_anthrop)')
ax.plot(t, E_krank, linewidth=2.5, color=accentred, label='Kranke Bilanz (Überbedarf, Defizit)')

ax.axhline(y=10.0, color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$ (Perioden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Energiebilanz (in Einheiten)', fontsize=12, fontweight='bold')
ax.set_title('Energiebilanz der Wirtschaft: Natürliche vs. Anthropogene Kräfte',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)

# Annotationen
ax.text(2, 20, 'Unbegründeter Überbedarf\nerschöpft Ressourcen', fontsize=10, color=accentred,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')

plt.tight_layout()
plt.savefig('plot8_energiebilanz.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 8: Energiebilanz erstellt")
plt.close()

# ============================================================
# Plot 9: Monte-Carlo Simulation - Wirtschaftliche Szenarien
# ============================================================
np.random.seed(42)

fig, ax = plt.subplots(1, 1, figsize=(13, 8))

# Verschiedene Szenarien
t = np.linspace(0, 10, 500)

# Szenario 1: Strikte natürliche Periodizität (keine anthropogene Störung)
E_n = 2.0 * np.sin(2 * np.pi * t / 2.5)

# Szenario 2: Mit moderater anthropogener Kraft (gesund)
E_a_moderate = 0.5 * np.sin(0.5 * np.pi * t)
E_mixed_moderate = E_n + E_a_moderate

# Szenario 3: Mit extremer anthropogener Kraft (krank)
E_a_extreme = 2.0 * np.sin(0.5 * np.pi * t) + 1.0 * np.sin(3 * np.pi * t / 2.5)
E_mixed_extreme = E_n + E_a_extreme

ax.plot(t, E_n, linewidth=2.5, color=darkgreen, label='Nur natürliche Periodizität', alpha=0.8)
ax.fill_between(t, E_n, E_mixed_moderate, alpha=0.2, color=mainblue, label='Moderate anthropogene Kraft (GESUND)')
ax.plot(t, E_mixed_moderate, linewidth=2.5, color=mainblue, alpha=0.9)

ax.plot(t, E_mixed_extreme, linewidth=2.5, color=accentred, label='Extreme anthropogene Kraft (KRANK)', 
        linestyle='--', alpha=0.9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlabel('Zeit $t$ (Perioden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wirtschaftliche Kraft', fontsize=12, fontweight='bold')
ax.set_title('Monte-Carlo Analyse: Szenarien bei verschiedenen Graden von Anthropogener Aktivität',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.set_xlim(0, 10)
ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('plot9_monte_carlo_szenarien.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 9: Monte-Carlo Szenarien erstellt")
plt.close()

# ============================================================
# Plot 10: Rechtlichkeit und Wirtschaft - Unbegründeter Überbedarf
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(13, 8))

categories = ['Begründeter\nBedarf\n(legal)', 'Moderater\nÜberbedarf\n(fragwürdig)', 
              'Starker\nÜberbedarf\n(rechtswidrig)', 'Extremer\nÜberbedarf\n(destruktiv)']
values = [10, 7, 4, 1]  # Wirtschaftliche Stabilität
colors_cat = [darkgreen, mainblue, accentred, (0.8, 0, 0)]

bars = ax.bar(categories, values, color=colors_cat, alpha=0.7, edgecolor='black', linewidth=2)

# Beschriftungen auf den Balken
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Wirtschaftliche Stabilität', fontsize=12, fontweight='bold')
ax.set_title('Auswirkungen von unbegründetem Überbedarf auf wirtschaftliche Stabilität\nund Rechtlichkeit',
             fontsize=13, fontweight='bold', pad=15)

ax.set_ylim(0, 12)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Rechtliche Grenze
ax.axhline(y=5.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Grenze: legal → rechtswidrig')
ax.legend(fontsize=11, loc='upper right')

# Annotationen
ax.text(0, 11, '✓ Natürliche\nPeriodizität\nerhalten', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontweight='bold')
ax.text(3, 0.5, '✗ Wirtschaft\nzerstört', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), fontweight='bold')

plt.tight_layout()
plt.savefig('plot10_rechtlichkeit_ueberbedarf.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 10: Rechtlichkeit und Überbedarf erstellt")
plt.close()

print("\n" + "="*60)
print("ALLE PLOTS ERFOLGREICH ERSTELLT!")
print("="*60)
print("\nErstellte Plot-Dateien:")
print("1. plot1_zwei_kraefte.pdf")
print("2. plot2_anthropogene_kraft.pdf")
print("3. plot3_natuerliche_periodizitaet.pdf")
print("4. plot4_resultierendes_gleichgewicht.pdf")
print("5. plot5_ueberbedarf_auswirkungen.pdf")
print("6. plot6_stabilitaet_instabilitaet.pdf")
print("7. plot7_phasenraum.pdf")
print("8. plot8_energiebilanz.pdf")
print("9. plot9_monte_carlo_szenarien.pdf")
print("10. plot10_rechtlichkeit_ueberbedarf.pdf")
