import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(42)
years = np.array([1990, 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020, 2023, 2024])

L1 = np.array([0.88, 0.87, 0.85, 0.84, 0.81, 0.78, 0.74, 0.71, 0.67, 0.62, 0.56, 0.51, 0.49])
L2 = np.array([0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.79, 0.76, 0.72, 0.67, 0.61, 0.55, 0.52])
L3 = np.array([0.85, 0.83, 0.80, 0.78, 0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38])
L4 = np.array([0.86, 0.84, 0.82, 0.80, 0.77, 0.73, 0.69, 0.64, 0.59, 0.54, 0.48, 0.43, 0.40])
L5 = np.array([0.89, 0.86, 0.82, 0.76, 0.70, 0.63, 0.56, 0.50, 0.44, 0.38, 0.32, 0.28, 0.26])

data = np.vstack([L1, L2, L3, L4, L5])
corr = np.corrcoef(data)

labels = [r'$L_1$', r'$L_2$', r'$L_3$', r'$L_4$', r'$L_5$']
full_labels = [
    r'$L_1$: Leben',
    r'$L_2$: Freiheit',
    r'$L_3$: Leichtigkeit',
    r'$L_4$: Vernunft',
    r'$L_5$: Schuldresistenz'
]

mainblue  = '#19468C'
accentred = '#B4321E'

# Eigene Colormap: tiefblau → weiß → tiefrot
cmap = mcolors.LinearSegmentedColormap.from_list(
    'bluewhitered',
    [(0.0, '#B4321E'), (0.5, '#FFFFFF'), (1.0, '#19468C')]
)

fig, ax = plt.subplots(figsize=(7.0, 6.2))
im = ax.imshow(corr, cmap=cmap, vmin=0.8, vmax=1.0)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(full_labels, fontsize=9.5, rotation=30, ha='right')
ax.set_yticklabels(full_labels, fontsize=9.5)

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label('Pearson-Korrelationskoeffizient $r$', fontsize=10)

for i in range(5):
    for j in range(5):
        val = corr[i, j]
        color = 'white' if val > 0.97 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=color)

ax.set_title('Korrelationsmatrix der Wertindizes $L_i(t)$\n(Pearson, $n = 13$ Datenpunkte, 1990–2024)',
             fontsize=11.5, fontweight='bold', color=mainblue, pad=12)

plt.tight_layout()
plt.savefig('plot4_korrelation.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 4 gespeichert.")
