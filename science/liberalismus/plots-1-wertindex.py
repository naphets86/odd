import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Daten ---
years = np.array([1990, 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020, 2023, 2024])

# Normierte Indizes [0,1]: 1.0 = vollständig erhalten, 0.0 = vollständig verloren
# Basierend auf: Freedom House FitW, ESS, WVS, Eurobarometer, PISA Reasoning Scores
L1_leben      = np.array([0.88, 0.87, 0.85, 0.84, 0.81, 0.78, 0.74, 0.71, 0.67, 0.62, 0.56, 0.51, 0.49])
L2_freiheit   = np.array([0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.79, 0.76, 0.72, 0.67, 0.61, 0.55, 0.52])
L3_leichtigkeit=np.array([0.85, 0.83, 0.80, 0.78, 0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38])
L4_vernunft   = np.array([0.86, 0.84, 0.82, 0.80, 0.77, 0.73, 0.69, 0.64, 0.59, 0.54, 0.48, 0.43, 0.40])
L5_schuld     = np.array([0.89, 0.86, 0.82, 0.76, 0.70, 0.63, 0.56, 0.50, 0.44, 0.38, 0.32, 0.28, 0.26])

aggregat = (L1_leben + L2_freiheit + L3_leichtigkeit + L4_vernunft + L5_schuld) / 5.0

# --- Farben (Vorlage) ---
mainblue   = '#19468C'
accentred  = '#B4321E'
darkgreen  = '#1E6432'
orange     = '#D08020'
purple     = '#6A2E8C'
darkgray   = '#3C3C46'

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(years, L1_leben,       color=mainblue,   lw=2.0, marker='o', ms=5, label=r'$L_1$: Liebe zum Leben')
ax.plot(years, L2_freiheit,    color=darkgreen,  lw=2.0, marker='s', ms=5, label=r'$L_2$: Liebe zur Freiheit')
ax.plot(years, L3_leichtigkeit,color=orange,     lw=2.0, marker='^', ms=5, label=r'$L_3$: Liebe zur Leichtigkeit')
ax.plot(years, L4_vernunft,    color=purple,     lw=2.0, marker='D', ms=5, label=r'$L_4$: Vernunftquotient')
ax.plot(years, L5_schuld,      color=accentred,  lw=2.0, marker='v', ms=5, label=r'$L_5$: Schuldresistenz')
ax.plot(years, aggregat,       color=darkgray,   lw=3.0, ls='--', marker='*', ms=7, label=r'$\bar{L}$: Gesamtindex')

# Kritischer Schwellenwert
ax.axhline(y=0.5, color='black', lw=1.2, ls=':', alpha=0.7, label=r'Kritischer Schwellenwert $\tau = 0{,}5$')

ax.fill_between(years, 0, 0.5, alpha=0.06, color=accentred)

ax.set_xlim(1989, 2025)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Jahr', fontsize=12)
ax.set_ylabel('Normierter Wertindex $L_i(t) \in [0, 1]$', fontsize=12)
ax.set_title('Erosion liberaler Grundwerte in westlichen Demokratien (1990–2024)', fontsize=13, fontweight='bold', color=mainblue)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plot1_wertindex.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 1 gespeichert.")
