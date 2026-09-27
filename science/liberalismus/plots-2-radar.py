import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

mainblue  = '#19468C'
accentred = '#B4321E'

categories = [
    'Liebe\nzum Leben',
    'Liebe zur\nFreiheit',
    'Liebe zur\nLeichtigkeit',
    'Vernunft-\nquotient',
    'Schuld-\nresistenz'
]
N = len(categories)

values_1995 = np.array([0.87, 0.89, 0.82, 0.83, 0.84])
values_2024 = np.array([0.49, 0.52, 0.38, 0.40, 0.26])

# Winkel
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

v1 = values_1995.tolist() + [values_1995[0]]
v2 = values_2024.tolist() + [values_2024[0]]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

ax.plot(angles, v1, color=mainblue,  lw=2.5, ls='-',  label='1995 (Ausgangswert)')
ax.fill(angles, v1, color=mainblue,  alpha=0.15)

ax.plot(angles, v2, color=accentred, lw=2.5, ls='--', label='2024 (aktueller Wert)')
ax.fill(angles, v2, color=accentred, alpha=0.20)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0,2', '0,4', '0,6', '0,8', '1,0'], fontsize=8.5, color='gray')
ax.set_ylim(0, 1.0)

ax.set_title('Liberale Grundwerte: Vergleich 1995 vs. 2024\n(Radardiagramm, normierte Wertindizes)', 
             fontsize=12, fontweight='bold', color=mainblue, pad=25)

ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.15), fontsize=10.5, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.45)

plt.tight_layout()
plt.savefig('plot2_radar.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 2 gespeichert.")
