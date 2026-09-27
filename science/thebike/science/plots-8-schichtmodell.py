"""
Plot 8: Schematischer Querschnitt des Thermoumschlags (Schichtmodell)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 7)   # oder sogar 6.5 je nach Geschmack
ax.set_ylim(0, 8)
ax.axis('off')

# Schichten von innen nach außen (x-Positionen)
schichten = [
    {'x': 1.0, 'w': 1.5, 'farbe': '#3A7DC9', 'label': 'Li-Ion-Akkupack\n(Wärmequelle)', 'text_x': 1.75},
    {'x': 2.5, 'w': 0.4, 'farbe': '#CCCCFF', 'label': 'Wärmeleitmatte\n(0.5 mm)', 'text_x': 2.7},
    {'x': 2.9, 'w': 0.6, 'farbe': '#B43220', 'label': 'PCM-Schicht\n(RT18, 5 mm)', 'text_x': 3.2},
    {'x': 3.5, 'w': 0.8, 'farbe': '#19468C', 'label': 'Aerogel-Kern\n(8 mm)', 'text_x': 3.9},
    {'x': 4.3, 'w': 0.3, 'farbe': '#88AADD', 'label': 'Reflektor\n(Alu-Folie, 0.2 mm)', 'text_x': 4.45},
    {'x': 4.6, 'w': 0.4, 'farbe': '#AADDAA', 'label': 'Neopren-Außenschicht\n(4 mm)', 'text_x': 4.8},
    {'x': 5.0, 'w': 0.3, 'farbe': '#88AA88', 'label': 'Wetterschutz-\ntextil', 'text_x': 5.15},
]

y0, y1 = 2.5, 6.5

for s in schichten:
    rect = mpatches.FancyBboxPatch((s['x'], y0), s['w'], y1-y0,
                                    boxstyle='square,pad=0',
                                    facecolor=s['farbe'], edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    # Pfeil + Label
    x_mid = s['x'] + s['w']/2
    ax.annotate(s['label'], xy=(x_mid, y1), xytext=(x_mid, y1+0.7),
                ha='center', va='bottom', fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

# Umgebung
ax.text(6.5, 4.5, 'Umgebungsluft\n$T_{\\mathrm{amb}}$', ha='center', fontsize=11, color='gray')
ax.annotate('', xy=(5.4, 4.5), xytext=(6.2, 4.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Temperaturgradient-Pfeil
ax.annotate('Wärmegradient $\\Delta T$', xy=(1.75, y0-0.3), xytext=(5.15, y0-0.3),
            ha='center', fontsize=10, color='#B43220',
            arrowprops=dict(arrowstyle='<->', color='#B43220', lw=1.5))

# U-Gesamt
ax.text(3.25, y0-1.0,
        '$U_{\\mathrm{ges}} = \\left(\\sum_i \\frac{d_i}{\\lambda_i}\\right)^{-1} \\approx 0.80\\,\\mathrm{W/(m^2 K)}$',
        ha='center', fontsize=11)

ax.set_title(
    'Schematischer Querschnitt des Thermoumschlags für E-Bike-Akkus',
    fontsize=13,
    pad=10
)
ax.annotate(
    s['label'],
    xy=(x_mid, y1),
    xytext=(x_mid, y1+0.9),  # etwas mehr Abstand
    fontsize=8
)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Platz für Titel lassen

plt.savefig('plot8_schichtmodell.pdf', dpi=300)
print("Plot 8 gespeichert")
