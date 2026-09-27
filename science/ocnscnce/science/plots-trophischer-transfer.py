#!/usr/bin/env python3
"""
Plot: Trophischer Transfer von Mikroplastik in Korallenriffen
Basiert auf: Isingoma et al. (2026), Hale et al. (2020)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(
    'Trophischer Transfer von Mikroplastik in Korallenriffen und bioakkumulative Effekte\n'
    'Quellen: Isingoma et al. (2026); Hale et al. (2020); Bisht & Negi (2020)',
    fontsize=11, fontweight='bold', y=1.02
)

# ── Subplot A: Bioakkumulationsfaktoren entlang der Nahrungskette ─────────────
ax = axes[0]
trophic_levels = ['Phytoplankton\n(Produzenten)', 'Zooplankton\n(Primärkons.)',
                  'Korallen\n(Sekundärkons.)', 'Korallenriff-\nFische',
                  'Prädatorische\nFische', 'Mensch\n(top predator)']
# Relative MP concentrations (Bioakkumulationsfaktoren, relative to seawater)
# Based on trophic magnification factor concepts
mp_conc  = [1.0, 3.2, 8.5, 18.7, 42.3, 89.6]  # Bioakkumulation relativ zu Meerwasser
cu_conc  = [1.0, 2.1, 5.8, 12.4, 28.9, 61.2]  # Cu-Bioakkumulation (MPs als Vektoren)

x = np.arange(len(trophic_levels))
w = 0.35
ax.bar(x - w/2, mp_conc, w, label='MP-Partikel (relative Konzentration)',
       color='#3498DB', alpha=0.85)
ax.bar(x + w/2, cu_conc, w, label='Cu(II) via MP-Vektor (relative Konzentration)',
       color='#E74C3C', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(trophic_levels, fontsize=7.5, rotation=15, ha='right')
ax.set_ylabel('Relative Konzentration (Meerwasser = 1)', fontsize=9)
ax.set_title('A) Bioakkumulation von MP-Partikeln und Cu(II)\n'
             'entlang der trophischen Kette im Korallenriff\n'
             '(Trophische Magnifikationsfaktoren, konzeptuell)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Add bioconcentration factor labels
for i, (m, c) in enumerate(zip(mp_conc, cu_conc)):
    ax.text(i - w/2, m + 0.8, f'×{m:.1f}', ha='center', fontsize=7, color='#2980B9')
    ax.text(i + w/2, c + 0.8, f'×{c:.1f}', ha='center', fontsize=7, color='#C0392B')

# ── Subplot B: MP-Konzentrations-Wirkungs-Beziehungen für Korallenarten ───────
ax = axes[1]
# Dose-response data from multiple studies
conc_ppm = np.array([0, 0.1, 1, 5, 10, 30, 50, 75, 100])  # mg/L

# Based on Reichert et al. 2018, 2019; Mendrik et al. 2021; Tirpitz et al. 2025
def dose_response(c, ec50, hill_coef):
    return 100 / (1 + (ec50 / (c + 0.001))**hill_coef)

# Different species with different sensitivities
poc_verr  = dose_response(conc_ppm, ec50=12, hill_coef=1.8)   # P. verrucosa - most sensitive
acr_cerv  = dose_response(conc_ppm, ec50=18, hill_coef=1.5)   # A. cervicornis
sty_pist  = dose_response(conc_ppm, ec50=25, hill_coef=1.4)   # S. pistillata
por_lutea = dose_response(conc_ppm, ec50=45, hill_coef=1.2)   # P. lutea - more resistant

ax.plot(conc_ppm, poc_verr,  'r-o',  linewidth=2.5, markersize=5,
        label='P. verrucosa (sensitiv)', zorder=5)
ax.plot(conc_ppm, acr_cerv,  'b-s',  linewidth=2, markersize=5,
        label='A. cervicornis')
ax.plot(conc_ppm, sty_pist,  'g-^',  linewidth=2, markersize=5,
        label='S. pistillata')
ax.plot(conc_ppm, por_lutea, 'm-D',  linewidth=2, markersize=5,
        label='P. lutea (resistenter)')

# Mark EC50 lines
for ec50, col in zip([12, 18, 25, 45], ['red', 'blue', 'green', 'purple']):
    ax.axvline(x=ec50, color=col, linestyle=':', alpha=0.4, linewidth=1)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
ax.text(80, 51, 'EC₅₀-Linie', fontsize=8, color='gray')
ax.fill_betweenx([0, 100], 0, 2, alpha=0.1, color='green', label='Umweltrelevante\nKonzentration (<2 mg/L)')

ax.set_xlabel('Mikroplastik-Konzentration (mg/L)', fontsize=9)
ax.set_ylabel('Effektstärke (% betroffene Korallen)', fontsize=9)
ax.set_title('B) Konzentrations-Wirkungs-Beziehungen\nverschiedener Korallenarten\n'
             '(Reichert et al. 2018, 2019; Tirpitz et al. 2025)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-1, 102)
ax.set_ylim(-2, 105)

plt.tight_layout()
plt.savefig('plot_trophischer_transfer.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot E gespeichert.")
