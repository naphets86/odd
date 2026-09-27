import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd

# Deutsches Locale für Text
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

# Farben und Styling
color_before = '#D62728'  # Rot
color_after = '#2CA02C'   # Grün
color_neutral = '#1F77B4' # Blau

# ============================================================
# PLOT 1: Risikokonzentrations-Index Vergleich
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

companies = ['Maschinenbau KG', 'Pharma-Handel KG', 'Immobilien KG Süd']
rci_before = [0.512, 0.608, 0.495]
rci_after = [0.334, 0.318, 0.327]
reduction = [34.8, 47.7, 34.0]

x = np.arange(len(companies))
width = 0.35

# Subplot 1: RCI Vergleich
bars1 = ax1.bar(x - width/2, rci_before, width, label='Baseline (Vor DRAV)', color=color_before, alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, rci_after, width, label='Nach 18 Monaten DRAV', color=color_after, alpha=0.8, edgecolor='black')

ax1.set_ylabel('Risikokonzentrations-Index (RCI)', fontsize=12, fontweight='bold')
ax1.set_title('Risikokonzentration vor und nach DRAV-Implementierung', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(companies, rotation=15, ha='right')
ax1.legend(loc='upper right', framealpha=0.95)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 0.7)

# Werte auf Balken
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: Reduktionsrate
colors_bars = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars3 = ax2.bar(companies, reduction, color=colors_bars, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Reduktion RCI (%)', fontsize=12, fontweight='bold')
ax2.set_title('Prozentuale Reduktion der Risikokonzentration', fontsize=13, fontweight='bold')
ax2.axhline(y=38.8, color='red', linestyle='--', linewidth=2, label='Durchschnitt (38.8%)', alpha=0.7)
ax2.set_xticklabels(companies, rotation=15, ha='right')
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 55)

# Werte auf Balken
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_1_rci_comparison.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 1: RCI-Vergleich erstellt")
plt.close()

# ============================================================
# PLOT 2: Governance-Transparenz-Index (GTI)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

gti_before = [23, 41, 18]
gti_after = [98, 100, 87]
companies_short = ['Masch.bau\nKG', 'Pharma\nHandel KG', 'Immobilien\nKG Süd']

x = np.arange(len(companies))
width = 0.35

# Subplot 1: GTI Vergleich
bars1 = ax1.bar(x - width/2, gti_before, width, label='Baseline (Vor DRAV)', color=color_before, alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, gti_after, width, label='Nach 18 Monaten DRAV', color=color_after, alpha=0.8, edgecolor='black')

ax1.set_ylabel('Governance-Transparenz-Index (%)', fontsize=12, fontweight='bold')
ax1.set_title('Dokumentation und Nachverfolgbarkeit von Risikotransfers', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(companies_short)
ax1.legend(loc='lower right', framealpha=0.95)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 110)

# Werte auf Balken
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: Improvement
improvement = [gti_after[i] - gti_before[i] for i in range(3)]
bars3 = ax2.bar(companies_short, improvement, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Verbesserung GTI (Prozentpunkte)', fontsize=12, fontweight='bold')
ax2.set_title('Transparenz-Verbesserung durch DRAV-Einführung', fontsize=13, fontweight='bold')
ax2.axhline(y=68, color='red', linestyle='--', linewidth=2, label='Durchschnitt (+68 pp)', alpha=0.7)
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 100)

# Werte auf Balken
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'+{int(height)} pp', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_2_gti_comparison.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 2: GTI-Vergleich erstellt")
plt.close()

# ============================================================
# PLOT 3: Risikovektor-Entwicklung (Maschinenbau KG)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Simulierte Daten: 4 Komplementäre + 2 Kommanditisten über 18 Monate
months = np.arange(0, 19, 3)
# Gesellschafter: K1, K2, K3, K4 (Kompl.), KM1, KM2 (Kommandit.)
rho_K1 = np.array([0.25, 0.23, 0.22, 0.20, 0.19, 0.18, 0.18])
rho_K2 = np.array([0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.25])
rho_K3 = np.array([0.18, 0.18, 0.19, 0.20, 0.21, 0.22, 0.22])
rho_K4 = np.array([0.22, 0.23, 0.23, 0.24, 0.24, 0.24, 0.23])
rho_KM1 = np.array([0.08, 0.09, 0.10, 0.11, 0.11, 0.11, 0.11])
rho_KM2 = np.array([0.07, 0.06, 0.04, 0.02, 0.01, 0.00, 0.01])

ax.plot(months, rho_K1, marker='o', linewidth=2.5, label='Kompl. 1', color='#d62728', markersize=8)
ax.plot(months, rho_K2, marker='s', linewidth=2.5, label='Kompl. 2', color='#1f77b4', markersize=8)
ax.plot(months, rho_K3, marker='^', linewidth=2.5, label='Kompl. 3', color='#2ca02c', markersize=8)
ax.plot(months, rho_K4, marker='D', linewidth=2.5, label='Kompl. 4', color='#ff7f0e', markersize=8)
ax.plot(months, rho_KM1, marker='*', linewidth=2.5, label='Kommandit. 1', color='#9467bd', markersize=12, linestyle='--')
ax.plot(months, rho_KM2, marker='x', linewidth=2.5, label='Kommandit. 2', color='#8c564b', markersize=10, linestyle='--')

ax.set_xlabel('Monate seit DRAV-Einführung', fontsize=12, fontweight='bold')
ax.set_ylabel('Risikoanteil (ρᵢ)', fontsize=12, fontweight='bold')
ax.set_title('Dynamische Risikovektorentwicklung: Maschinenbau KG über 18 Monate', fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.95, ncol=2, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(0, 0.30)
ax.set_xticks(months)

# Annotation für wichtige Ereignisse
ax.annotate('Erste Transfers', xy=(3, 0.22), xytext=(3, 0.27),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, color='red', fontweight='bold')
ax.annotate('Stabilisierung', xy=(12, 0.22), xytext=(12, 0.27),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5), fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('plot_3_risk_vector_evolution.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 3: Risikovektor-Entwicklung erstellt")
plt.close()

# ============================================================
# PLOT 4: DRAV-Transfer-Aktivität über Zeit
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Monatliche Transfer-Anzahl
months_data = np.arange(1, 19)
drav_count = np.array([3, 5, 7, 8, 9, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 1])
cumulative_drav = np.cumsum(drav_count)

# Subplot 1: Monatliche DRAVs
bars = ax1.bar(months_data, drav_count, color=color_neutral, alpha=0.8, edgecolor='black')
ax1.plot(months_data, drav_count, color='red', marker='o', linewidth=2, markersize=6, label='Trendlinie')
ax1.set_ylabel('Anzahl neuer DRAVs pro Monat', fontsize=12, fontweight='bold')
ax1.set_title('DRAV-Aktivität in den ersten 18 Monaten (Alle 3 KGs kombiniert)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')
ax1.set_xlim(0.5, 18.5)
ax1.set_xticks(months_data)

# Werte auf Balken
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8)

# Subplot 2: Kumulative DRAVs
ax2.fill_between(months_data, cumulative_drav, alpha=0.3, color=color_neutral)
ax2.plot(months_data, cumulative_drav, marker='o', linewidth=3, markersize=8, color=color_neutral, label='Kumulativ')
ax2.set_xlabel('Monate seit DRAV-Einführung', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kumulierte Anzahl DRAVs', fontsize=12, fontweight='bold')
ax2.set_title('Kumulative DRAV-Dokumentation', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left')
ax2.set_xlim(0.5, 18.5)
ax2.set_xticks(months_data)

# Wert am Ende
ax2.text(18, cumulative_drav[-1], f'  {cumulative_drav[-1]}\n  DRAVs\n  gesamt', 
         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plot_4_drav_activity.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 4: DRAV-Aktivität erstellt")
plt.close()

# ============================================================
# PLOT 5: Haftungskompatibilität-Constraints Visualisierung
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

# Beispiel-Risikoverteilungen und ihre Kompatibilität
configs = ['Start-\nkonfigur.', 'Transfer 1\n(unzulässig)', 'Transfer 2\n(zulässig)', 
           'Transfer 3\n(zulässig)', 'End-\nkonfigur.']
K1_values = [0.25, 0.20, 0.20, 0.18, 0.18]
K2_values = [0.20, 0.25, 0.25, 0.25, 0.25]
K3_values = [0.18, 0.18, 0.19, 0.22, 0.22]
K4_values = [0.22, 0.22, 0.23, 0.24, 0.23]
KM1_values = [0.08, 0.10, 0.10, 0.10, 0.11]
KM2_values = [0.07, 0.05, 0.03, 0.01, 0.01]
validity = ['✓', '✗', '✓', '✓', '✓']
validity_colors = ['green', 'red', 'green', 'green', 'green']

x = np.arange(len(configs))
width = 0.15

rects1 = ax.bar(x - 2*width, K1_values, width, label='Kompl. 1', color='#d62728', alpha=0.9, edgecolor='black')
rects2 = ax.bar(x - width, K2_values, width, label='Kompl. 2', color='#1f77b4', alpha=0.9, edgecolor='black')
rects3 = ax.bar(x, K3_values, width, label='Kompl. 3', color='#2ca02c', alpha=0.9, edgecolor='black')
rects4 = ax.bar(x + width, K4_values, width, label='Kompl. 4', color='#ff7f0e', alpha=0.9, edgecolor='black')
rects5 = ax.bar(x + 2*width, KM1_values, width, label='Kommandit. 1', color='#9467bd', alpha=0.7, edgecolor='black', linewidth=1.5)
rects6 = ax.bar(x + 3*width, KM2_values, width, label='Kommandit. 2', color='#8c564b', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Risikoanteil (ρᵢ)', fontsize=12, fontweight='bold')
ax.set_title('Haftungskompatibilität: Zulässige und unzulässige Risikotransfers', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.legend(loc='upper right', framealpha=0.95, ncol=2, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.32)

# Gültigkeitsindikatoren
for i, (valid, color) in enumerate(zip(validity, validity_colors)):
    if valid == '✓':
        marker = '✓'
        fontsize = 16
    else:
        marker = '✗'
        fontsize = 18
    ax.text(i, 0.30, marker, ha='center', va='bottom', fontsize=fontsize, 
            fontweight='bold', color=color)

# Text-Box mit Constraints
constraint_text = ('Constraints:\n'
                  '(C1) Komplementäre: ρᵢ monoton wachsend\n'
                  '(C2) Kommanditisten: nur mit Dokumentation\n'
                  '(C3) Keine negativen Risiken (ρᵢ ≥ 0)\n'
                  '(C4) Zeitgestempel und Dokumentation')
ax.text(0.02, 0.97, constraint_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        family='monospace')

plt.tight_layout()
plt.savefig('plot_5_constraints.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 5: Haftungskompatibilität-Constraints erstellt")
plt.close()

# ============================================================
# PLOT 6: Vergleich RCI-Gleichgewicht
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))

# Theoretischer vs. erreichter RCI
companies_full = ['Maschinenbau KG', 'Pharma-Handel KG', 'Immobilien KG Süd', 'Durchschnitt']
rci_theoretically_optimal = [1/6, 1/5, 1/4, (1/6 + 1/5 + 1/4) / 3]  # Gleichverteilte Baseline
rci_before_full = [0.512, 0.608, 0.495, 0.538]
rci_after_full = [0.334, 0.318, 0.327, 0.326]

x = np.arange(len(companies_full))
width = 0.25

bars1 = ax.bar(x - width, rci_theoretically_optimal, width, label='Theoretisch optimal\n(vollständige Gleichverteilung)', 
               color='#2ca02c', alpha=0.6, edgecolor='black')
bars2 = ax.bar(x, rci_before_full, width, label='Baseline (vor DRAV)', 
               color=color_before, alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, rci_after_full, width, label='Nach DRAV-Implementierung (18 Mo.)', 
               color=color_after, alpha=0.8, edgecolor='black')

ax.set_ylabel('Risikokonzentrations-Index (RCI)', fontsize=12, fontweight='bold')
ax.set_title('RCI-Entwicklung: Von Konzentration zu ausgeglichener Verteilung', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(companies_full, fontsize=11)
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.7)

# Werte auf Balken (nur für relevante)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Annotation für theoretisches Optimum
ax.annotate('Ideales Equilibrium\n(bei gleichmäßiger\nVerteilung)', xy=(3.3, rci_theoretically_optimal[-1]), 
            xytext=(3.3, 0.35),
            arrowprops=dict(arrowstyle='->', color='green', lw=2), fontsize=10, color='green', fontweight='bold',
            ha='center')

plt.tight_layout()
plt.savefig('plot_6_rci_equilibrium.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 6: RCI-Gleichgewicht erstellt")
plt.close()

print("\nAlle 6 Plots erfolgreich erstellt!")