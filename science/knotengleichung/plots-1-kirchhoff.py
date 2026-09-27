"""
Plot 1: Kirchhoffsches Stromgesetz – Knotendiagramm
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('white')

# ---------- linkes Bild: Knotendiagramm ----------
ax = axes[0]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Elektrischer Knoten\n$\\sum I_\\mathrm{zu} = \\sum I_\\mathrm{ab}$',
             fontsize=13, fontweight='bold', color='#19468C')

# Zentralknoten
node = plt.Circle((0, 0), 0.22, color='#19468C', zorder=5)
ax.add_patch(node)
ax.text(0, 0, 'K', ha='center', va='center', fontsize=12,
        color='white', fontweight='bold', zorder=6)

# Pfeile & Beschriftungen
arrows = [
    # (x0, y0, dx, dy, Farbe, Label, seite)
    (-2.8, 0.8,   2.0, -0.6, '#B4321E', '$I_1 = 4\\,\\mathrm{A}$', 'in'),
    (-2.8, -0.8,  2.0,  0.6, '#B4321E', '$I_2 = 2\\,\\mathrm{A}$', 'in'),
    (0, 2.8,    0.0, -2.4, '#B4321E', '$I_3 = 1\\,\\mathrm{A}$', 'in'),
    (2.8,  0.8, -2.0, -0.6, '#1E6432', '$I_4 = 3\\,\\mathrm{A}$', 'out'),
    (2.8, -0.8, -2.0,  0.6, '#1E6432', '$I_5 = 2\\,\\mathrm{A}$', 'out'),
    (0,  -2.8,  0.0,  2.4, '#1E6432', '$I_6 = 2\\,\\mathrm{A}$', 'out'),
]
for (x0, y0, dx, dy, col, lbl, side) in arrows:
    ax.annotate('', xy=(x0 + dx * 0.8, y0 + dy * 0.8),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=2.2, mutation_scale=18))
    lx = x0 + dx * 0.35
    ly = y0 + dy * 0.35
    ha = 'right' if x0 < 0 else ('left' if x0 > 0 else 'center')
    ax.text(lx, ly, lbl, fontsize=10, color=col, ha=ha, va='bottom')

# Legende
in_p  = mpatches.Patch(color='#B4321E', label='Zustrom $\\Sigma=7\\,\\mathrm{A}$')
out_p = mpatches.Patch(color='#1E6432', label='Abstrom $\\Sigma=7\\,\\mathrm{A}$')
ax.legend(handles=[in_p, out_p], loc='lower center', fontsize=9,
          framealpha=0.9, edgecolor='#cccccc')

# ---------- rechtes Bild: Gleichgewichtsbalken ----------
ax2 = axes[1]
cats  = ['$I_1$', '$I_2$', '$I_3$', '$I_4$', '$I_5$', '$I_6$']
vals  = [4, 2, 1, -3, -2, -2]
colors = ['#B4321E' if v > 0 else '#1E6432' for v in vals]

bars = ax2.bar(cats, vals, color=colors, edgecolor='#555555', linewidth=0.8)
ax2.axhline(0, color='#19468C', linewidth=1.5, linestyle='--')
ax2.set_title('Stromstärken am Knoten\n$\\sum_{k} I_k = 0$',
              fontsize=13, fontweight='bold', color='#19468C')
ax2.set_ylabel('Stromstärke [A]', fontsize=11)
ax2.set_xlabel('Zweig', fontsize=11)
ax2.set_ylim(-4.5, 5.5)
ax2.grid(axis='y', alpha=0.3, linestyle=':')

# Vorzeichen-Annotation
total_in  = sum(v for v in vals if v > 0)
total_out = abs(sum(v for v in vals if v < 0))
ax2.text(0.5, 0.97,
         f'Zustrom: +{total_in} A\nAbstrom: −{total_out} A\nNettostrom: 0 A',
         transform=ax2.transAxes, va='top', ha='center',
         fontsize=10, color='#19468C',
         bbox=dict(boxstyle='round,pad=0.4', fc='#EEF3FF', ec='#19468C', lw=1))

for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             v + (0.15 if v > 0 else -0.35),
             f'{v:+d} A', ha='center', va='bottom', fontsize=9,
             color='#333333')

plt.tight_layout(pad=2.0)
plt.savefig('plot1_kirchhoff.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot 1 gespeichert.")
