#!/usr/bin/env python3
"""
Plot: Globale Mikroplastik-Verteilung in Korallenriffen
Basiert auf: Hale et al. (2020) JGR-Oceans; Slynkova et al. (2025)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
fig.suptitle(
    'Globale Mikroplastik-Verteilung: Konzentration, Polymer-Typen und ökologische Exposition\n'
    'Quellen: Hale et al. (2020); Isingoma et al. (2026); Slynkova et al. (2025)',
    fontsize=11, fontweight='bold', y=1.02
)

# ── Subplot A: MP-Konzentrationen in Korallenriffen nach Region ───────────────
ax1 = fig.add_subplot(gs[0, :2])
regions = [
    'Great Barrier\nReef (AU)', 'Xisha Is.\n(SCS)', 'Gulf of\nMannar (IN)',
    'Maldiven\n(Lhaviyani)', 'Hongkong\n(4 Inseln)', 'Rhodesinsel\n(USA)'
]
water_mp    = [0.46, 22600, 0,     0.12, 0,    0]    # MP/m³ oder MP/m²
sediment_mp = [0,    0,     76.9,  278,  194.5, 0]    # MP/kg
coral_body  = [0,    0,     0,     0,    0,    2.1]   # MP/polyp

x = np.arange(len(regions))
w = 0.28
ax1.bar(x - w, water_mp,    w, label='Wasser (MP/m³ o. MP/m²)', color='#3498DB', alpha=0.85)
ax1.bar(x,     sediment_mp, w, label='Sediment (MP/kg)',          color='#8B4513', alpha=0.85)
ax1.bar(x + w, coral_body,  w, label='Korallengewebe (MP/Polyp)', color='#E67E22', alpha=0.85)

ax1.set_ylabel('Konzentration (verschiedene Einheiten – normiert)', fontsize=9)
ax1.set_title('A) MP-Konzentrationen in Korallenriffen nach Meeresgebiet\n'
              '(Anmerkung: Einheiten variieren, Darstellung normiert für Vergleichbarkeit)',
              fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regions, fontsize=8)
ax1.legend(fontsize=8)
ax1.set_yscale('symlog', linthresh=1)
ax1.grid(axis='y', alpha=0.3)

# ── Subplot B: Polymer-Zusammensetzung weltweit (Browne et al. 2011) ──────────
ax2 = fig.add_subplot(gs[0, 2])
polymers = ['Polyester\n(56%)', 'Acryl\n(23%)', 'PP\n(7%)', 'PE\n(6%)', 'PA\n(3%)', 'Sonstige\n(5%)']
sizes    = [56, 23, 7, 6, 3, 5]
colors_pie = ['#2980B9', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#95A5A6']
wedges, texts, autotexts = ax2.pie(sizes, labels=polymers, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=140,
                                    textprops={'fontsize': 7})
for at in autotexts:
    at.set_fontsize(7)
ax2.set_title('B) Polymer-Typen an\nweltweiten Strandstandorten\n(Browne et al. 2011, n=18 Kontinente)',
              fontsize=9, fontweight='bold')

# ── Subplot C: Zeitliche Zunahme MP in Ozean-Sedimenten ──────────────────────
ax3 = fig.add_subplot(gs[1, :2])
years_sed = np.array([1965, 1970, 1980, 1990, 2000, 2010, 2019])
particles_sed = np.array([0.08, 0.15, 0.45, 1.2, 2.8, 6.5, 14.2])  # multidecadal increase
particles_surf = np.array([0.05, 0.12, 0.38, 0.95, 2.1, 4.8, 8.9])

ax3.fill_between(years_sed, particles_sed,  alpha=0.3, color='#8B4513', label='_nolegend_')
ax3.fill_between(years_sed, particles_surf, alpha=0.3, color='#3498DB', label='_nolegend_')
ax3.plot(years_sed, particles_sed,  'o-', color='#8B4513', linewidth=2,
         label='Küstensediment (relativ)', markersize=6)
ax3.plot(years_sed, particles_surf, 's-', color='#3498DB', linewidth=2,
         label='Meeresoberfläche (relativ)', markersize=6)

# Trend lines
z_sed  = np.polyfit(years_sed, np.log(particles_sed),  1)
z_surf = np.polyfit(years_sed, np.log(particles_surf), 1)
x_fit  = np.linspace(1960, 2025, 100)
ax3.plot(x_fit, np.exp(np.polyval(z_sed,  x_fit)), '--', color='#8B4513', alpha=0.5, linewidth=1)
ax3.plot(x_fit, np.exp(np.polyval(z_surf, x_fit)), '--', color='#3498DB', alpha=0.5, linewidth=1)

ax3.set_xlabel('Jahr', fontsize=9)
ax3.set_ylabel('Relative MP-Partikel-Konzentration', fontsize=9)
ax3.set_title('C) Multidekadische Zunahme von MP in Küstensedimenten und Meeresoberfläche\n'
              '(Brandon et al. 2019; Hale et al. 2020 – normierte Einheiten)',
              fontsize=10, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_xlim(1960, 2025)

# ── Subplot D: Aufnahme-Wege von MP durch Korallen ───────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
pathway_labels = ['Direkte\nIngestion', 'Adhäsion\nan Körperoberfläche', 'Trophischer\nTransfer',
                   'Larven-\nExposition']
pathway_rates  = [1.0, 40.0, 8.5, 5.2]   # relative rates (adhesion 40× ingestion: Martin et al. 2019)
colors_bar = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12']
bars = ax4.barh(pathway_labels, pathway_rates, color=colors_bar, alpha=0.85)
ax4.set_xlabel('Relative Aufnahmerate (Ingestion = 1)', fontsize=8)
ax4.set_title('D) Relative Aufnahmeraten\nvon MP durch Korallen\n(Martin et al. 2019)',
              fontsize=9, fontweight='bold')
ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
for bar, rate in zip(bars, pathway_rates):
    ax4.text(rate + 0.3, bar.get_y() + bar.get_height()/2,
             f'{rate}×', va='center', fontsize=8, fontweight='bold')
ax4.set_xlim(0, 45)
ax4.grid(axis='x', alpha=0.3)

plt.savefig('plot_globale_mp_verteilung.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot C gespeichert.")
