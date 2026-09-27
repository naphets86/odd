"""
Plot 2: Atmosphärischer Wasserkreislauf – Knotengleichung der Erde
Werte nach IPCC AR6 / Trenberth et al. 2011 (km³/Jahr)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('#F0F5FF')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#EAF3FF')

# Himmel (hellblau)
sky = mpatches.FancyBboxPatch((0, 4.5), 14, 3.5, boxstyle='square',
                               fc='#B8D8F8', ec='none', zorder=0)
ax.add_patch(sky)
# Erde/Boden (braun)
ground = mpatches.FancyBboxPatch((0, 0), 14, 4.0, boxstyle='square',
                                  fc='#D4E8C2', ec='none', zorder=0)
ax.add_patch(ground)
# Ozean
ocean = mpatches.FancyBboxPatch((0, 0), 6.5, 3.8, boxstyle='square',
                                 fc='#5B9BD5', ec='none', zorder=1)
ax.add_patch(ocean)

ax.text(3.2, 1.9, 'Ozean', fontsize=15, ha='center', va='center',
        color='white', fontweight='bold', zorder=3)
ax.text(3.2, 1.3, '502.800 km³/a\nVerdunstung', fontsize=8.5,
        ha='center', va='center', color='#DDEEFF', zorder=3)

ax.text(10.2, 1.9, 'Land', fontsize=15, ha='center', va='center',
        color='#4A3000', fontweight='bold', zorder=3)
ax.text(10.2, 1.3, '74.200 km³/a\nEvapotranspiration', fontsize=8.5,
        ha='center', va='center', color='#5A4000', zorder=3)

# Wolken
def cloud(ax, cx, cy, r=0.5, col='#FFFFFF', zorder=4):
    for dx, dy, ri in [(0,0,r),(r*0.7,r*0.25,r*0.75),(-r*0.7,r*0.25,r*0.75),
                        (0, r*0.55, r*0.7)]:
        c = plt.Circle((cx+dx, cy+dy), ri, color=col, zorder=zorder)
        ax.add_patch(c)

cloud(ax, 3.5, 6.2, r=0.65, col='#CFE5FF')
cloud(ax, 10.5, 6.2, r=0.65, col='#CFE5FF')
cloud(ax, 7.0, 6.5, r=0.55, col='#E0EEFF')

ax.text(3.5, 7.4, '458.000 km³/a\nNiederschlag Ozean', fontsize=8,
        ha='center', color='#19468C', fontweight='bold')
ax.text(10.5, 7.4, '119.000 km³/a\nNiederschlag Land', fontsize=8,
        ha='center', color='#19468C', fontweight='bold')

# Pfeile Verdunstung (blau, aufwärts)
def arrow(ax, x1, y1, x2, y2, col, label, lpos='mid', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                mutation_scale=16,
                                connectionstyle='arc3,rad=0.0'))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx+0.15, my, label, fontsize=8.5, color=col, va='center')

# Verdunstung Ozean → Wolke
ax.annotate('', xy=(3.5, 5.55), xytext=(3.5, 3.8),
            arrowprops=dict(arrowstyle='->', color='#B4321E', lw=2.5,
                            mutation_scale=16))
ax.text(3.75, 4.7, '502.800 km³/a\nVerdunstung', fontsize=8,
        color='#B4321E', va='center')

# Verdunstung Land → Wolke
ax.annotate('', xy=(10.5, 5.55), xytext=(10.5, 3.8),
            arrowprops=dict(arrowstyle='->', color='#B4321E', lw=2.5,
                            mutation_scale=16))
ax.text(10.75, 4.7, '74.200 km³/a\nEvapotranspiration', fontsize=8,
        color='#B4321E', va='center')

# Niederschlag Ozean (hellblau, abwärts)
ax.annotate('', xy=(2.8, 3.8), xytext=(2.8, 5.6),
            arrowprops=dict(arrowstyle='->', color='#19468C', lw=2.5,
                            mutation_scale=16))

# Niederschlag Land
ax.annotate('', xy=(9.8, 3.8), xytext=(9.8, 5.6),
            arrowprops=dict(arrowstyle='->', color='#19468C', lw=2.5,
                            mutation_scale=16))

# Horizontaler Transport Ozean → Land
ax.annotate('', xy=(9.2, 5.0), xytext=(4.7, 5.0),
            arrowprops=dict(arrowstyle='->', color='#5B2D8E', lw=2.5,
                            mutation_scale=16,
                            connectionstyle='arc3,rad=-0.2'))
ax.text(7.0, 5.25, '44.800 km³/a\nTransport Ozean→Land',
        fontsize=8.5, ha='center', color='#5B2D8E', fontweight='bold')

# Oberflächenabfluss Land → Ozean
ax.annotate('', xy=(6.5, 2.2), xytext=(8.0, 2.2),
            arrowprops=dict(arrowstyle='->', color='#005F8C', lw=2.5,
                            mutation_scale=16))
ax.text(7.2, 2.45, '44.800 km³/a\nAbfluss', fontsize=8.5,
        ha='center', color='#005F8C')

# Titel
ax.text(7.0, 7.85, 'Atmosphärische Knotengleichung der Erde',
        fontsize=13, ha='center', va='top', fontweight='bold',
        color='#19468C')

# Knotengleichung als Text
ax.text(7.0, 0.55,
        r'$\oint_{\partial \Omega} \rho_w\, \mathbf{v} \cdot \mathrm{d}\mathbf{A} = \dot{E} - \dot{P} \;\Rightarrow\; \dot{E}_\mathrm{global} = \dot{P}_\mathrm{global} = 577{.}000\;\mathrm{km}^3/\mathrm{a}$',
        fontsize=10, ha='center', va='center', color='#19468C',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#19468C', lw=1.2))

# Legende
leg_entries = [
    mpatches.Patch(color='#B4321E', label='Verdunstung / Evaporation'),
    mpatches.Patch(color='#19468C', label='Niederschlag / Precipitation'),
    mpatches.Patch(color='#5B2D8E', label='Atmosphärischer Transport'),
    mpatches.Patch(color='#005F8C', label='Oberflächenabfluss'),
]
ax.legend(handles=leg_entries, loc='lower right', fontsize=8.5,
          framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig('plot2_wasserkreislauf.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot 2 gespeichert.")
