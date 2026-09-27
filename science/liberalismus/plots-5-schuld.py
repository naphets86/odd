import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

years = np.array([1990, 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020, 2023, 2024])

# Schuldgeständnis-Kultur-Index (SK): steigend (0 = keine Schuld-Rhetorik, 1 = maximale Schuld-Kultur)
SK = 1.0 - np.array([0.89, 0.86, 0.82, 0.76, 0.70, 0.63, 0.56, 0.50, 0.44, 0.38, 0.32, 0.28, 0.26])

# Vernunft-Erosion: E4 = 1 - L4
E4 = 1.0 - np.array([0.86, 0.84, 0.82, 0.80, 0.77, 0.73, 0.69, 0.64, 0.59, 0.54, 0.48, 0.43, 0.40])

# Gesamterosionsindex
L1 = np.array([0.88, 0.87, 0.85, 0.84, 0.81, 0.78, 0.74, 0.71, 0.67, 0.62, 0.56, 0.51, 0.49])
L2 = np.array([0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.79, 0.76, 0.72, 0.67, 0.61, 0.55, 0.52])
L3 = np.array([0.85, 0.83, 0.80, 0.78, 0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38])
L4 = np.array([0.86, 0.84, 0.82, 0.80, 0.77, 0.73, 0.69, 0.64, 0.59, 0.54, 0.48, 0.43, 0.40])
L5 = np.array([0.89, 0.86, 0.82, 0.76, 0.70, 0.63, 0.56, 0.50, 0.44, 0.38, 0.32, 0.28, 0.26])
erosion_total = 1.0 - (L1+L2+L3+L4+L5)/5.0

mainblue  = '#19468C'
accentred = '#B4321E'
purple    = '#6A2E8C'
darkgray  = '#3C3C46'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Linkes Diagramm: SK und E4 im Zeitverlauf ---
ax = axes[0]
ax.plot(years, SK, color=accentred, lw=2.5, marker='v', ms=6,
        label=r'Schuld-Kultur-Index $SK(t) = 1 - L_5(t)$')
ax.plot(years, E4, color=purple, lw=2.5, marker='D', ms=6,
        label=r'Vernunft-Erosion $E_4(t) = 1 - L_4(t)$')
ax.plot(years, erosion_total, color=darkgray, lw=2.5, ls='--', marker='*', ms=7,
        label=r'Gesamt-Erosionsindex $\bar{E}(t)$')

ax.set_xlim(1989, 2025)
ax.set_ylim(0.0, 0.85)
ax.set_xlabel('Jahr', fontsize=11)
ax.set_ylabel('Erosionsgrad', fontsize=11)
ax.set_title('Schuld-Kultur-Index und Vernunft-Erosion\n(1990–2024)', 
             fontsize=11.5, fontweight='bold', color=mainblue)
ax.legend(fontsize=9.0, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Phasenbeschriftungen
ax.axvspan(1990, 2004, alpha=0.07, color=mainblue, label='_Phase I')
ax.axvspan(2004, 2015, alpha=0.07, color=purple, label='_Phase II')
ax.axvspan(2015, 2024, alpha=0.07, color=accentred, label='_Phase III')
ax.text(1995, 0.78, 'Phase I', fontsize=9, color=mainblue, alpha=0.7, fontstyle='italic')
ax.text(2007, 0.78, 'Phase II', fontsize=9, color=purple, alpha=0.7, fontstyle='italic')
ax.text(2018, 0.78, 'Phase III', fontsize=9, color=accentred, alpha=0.7, fontstyle='italic')

# --- Rechtes Diagramm: SK vs. E4 Scatter ---
ax2 = axes[1]
sc = ax2.scatter(E4, SK, c=years, cmap='RdYlBu_r', s=70, zorder=3, edgecolors='white', lw=0.6)

# Regressionsgerade
m, b = np.polyfit(E4, SK, 1)
x_fit = np.linspace(E4.min(), E4.max(), 100)
ax2.plot(x_fit, m * x_fit + b, color=darkgray, lw=1.8, ls='--', alpha=0.8,
         label=f'Regression: $SK = {m:.2f} \\cdot E_4 + {b:.2f}$')

corr_coef = np.corrcoef(E4, SK)[0, 1]
ax2.text(0.05, 0.93, f'$r = {corr_coef:.3f}$', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', color=accentred)

# Jahresbeschriftung für ausgewählte Punkte
for yr, x, y in zip(years, E4, SK):
    if yr in [1990, 1999, 2008, 2017, 2024]:
        ax2.annotate(str(yr), (x, y), textcoords='offset points',
                     xytext=(6, 4), fontsize=8.5, color=darkgray)

cbar2 = plt.colorbar(sc, ax=ax2)
cbar2.set_label('Jahr', fontsize=9.5)
ax2.set_xlabel(r'Vernunft-Erosion $E_4(t)$', fontsize=11)
ax2.set_ylabel(r'Schuld-Kultur-Index $SK(t)$', fontsize=11)
ax2.set_title(r'Korrelation: $E_4(t)$ vs. $SK(t)$'+'\n(Streudiagramm mit Regression)', 
              fontsize=11.5, fontweight='bold', color=mainblue)
ax2.legend(fontsize=9, framealpha=0.9, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plot5_schuld.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 5 gespeichert.")
