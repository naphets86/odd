import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mainblue  = '#19468C'
accentred = '#B4321E'
darkgreen = '#1E6432'
orange    = '#D08020'
purple    = '#6A2E8C'

labels = [
    r'$L_1$: Liebe zum Leben',
    r'$L_2$: Liebe zur Freiheit',
    r'$L_3$: Liebe zur Leichtigkeit',
    r'$L_4$: Vernunftquotient',
    r'$L_5$: Schuldresistenz'
]

val_1990 = np.array([0.88, 0.91, 0.85, 0.86, 0.89])
val_2024 = np.array([0.49, 0.52, 0.38, 0.40, 0.26])
erosion  = val_1990 - val_2024  # absolute Erosion
erosion_pct = (erosion / val_1990) * 100  # relative Erosion in %

colors = [mainblue, darkgreen, orange, purple, accentred]

x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Linkes Diagramm: Absolute Erosion ---
bars1 = ax1.bar(x, erosion, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels([f'$L_{i+1}$' for i in range(5)], fontsize=12)
ax1.set_ylabel('Absolute Erosion $\Delta L_i$ (1990–2024)', fontsize=11)
ax1.set_title('Absolute Werterosion', fontsize=12, fontweight='bold', color=mainblue)
ax1.set_ylim(0, 0.75)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, val in zip(bars1, erosion):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

# --- Rechtes Diagramm: Relative Erosion (%) ---
bars2 = ax2.bar(x, erosion_pct, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels([f'$L_{i+1}$' for i in range(5)], fontsize=12)
ax2.set_ylabel('Relative Erosion $e_i$ (%) gegenüber 1990', fontsize=11)
ax2.set_title('Relative Werterosion (%)', fontsize=12, fontweight='bold', color=mainblue)
ax2.set_ylim(0, 80)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, val in zip(bars2, erosion_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

# Legende
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(5)]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

fig.suptitle('Erosion liberaler Grundwerte von 1990 bis 2024\n(absolut und relativ)', 
             fontsize=13, fontweight='bold', color=mainblue, y=1.01)

plt.tight_layout()
plt.savefig('plot3_erosion.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 3 gespeichert.")
