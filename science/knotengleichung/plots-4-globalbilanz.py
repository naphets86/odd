"""
Plot 4: Globales Wassermassengleichgewicht – Zeitreihe & Monatsmittel
        Synthetische Daten angelehnt an GPCP / ERA5 Reanalysedaten
"""
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.patch.set_facecolor('white')
fig.suptitle('Globales Wassermassengleichgewicht – Knotengleichung der Atmosphäre',
             fontsize=13, fontweight='bold', color='#19468C', y=1.01)

np.random.seed(7)

# ---- (a) Jahreszeitliche Schwankung E und P ----
ax = axes[0, 0]
months = np.arange(1, 13)
month_labels = ['Jan','Feb','Mär','Apr','Mai','Jun',
                'Jul','Aug','Sep','Okt','Nov','Dez']
# Mittlere monatliche Verdunstung (global, km³/Monat) – saisonale Kurve
base = 577000 / 12
amp_e = 12000
amp_p = 10000
E_monthly = base + amp_e * np.sin(2*np.pi*(months-3)/12) + \
            np.random.normal(0, 1500, 12)
P_monthly = base + amp_p * np.sin(2*np.pi*(months-4)/12) + \
            np.random.normal(0, 1500, 12)

ax.fill_between(months, E_monthly/1000, alpha=0.25, color='#B4321E')
ax.fill_between(months, P_monthly/1000, alpha=0.25, color='#19468C')
ax.plot(months, E_monthly/1000, 'o-', color='#B4321E', lw=2,
        label='Globale Verdunstung $E$', ms=5)
ax.plot(months, P_monthly/1000, 's-', color='#19468C', lw=2,
        label='Globaler Niederschlag $P$', ms=5)
ax.set_xticks(months); ax.set_xticklabels(month_labels, fontsize=8)
ax.set_ylabel('[10³ km³/Monat]', fontsize=9)
ax.set_title('(a) Saisonale Variation $E$ und $P$', fontsize=10,
             fontweight='bold', color='#19468C')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, ls=':')
ax.text(6.5, min(E_monthly.min(), P_monthly.min())/1000 - 1.5,
        r'$\overline{E} \approx \overline{P} \approx 48{.}100\;\mathrm{km}^3/\mathrm{Monat}$',
        ha='center', fontsize=9, color='#19468C',
        bbox=dict(boxstyle='round,pad=0.3', fc='#EEF3FF', ec='#19468C'))

# ---- (b) Nettobilanz E-P (sollte ~ 0) ----
ax2 = axes[0, 1]
net = (E_monthly - P_monthly) / 1000
colors = ['#B4321E' if v > 0 else '#19468C' for v in net]
ax2.bar(months, net, color=colors, edgecolor='#555', lw=0.7)
ax2.axhline(0, color='#333', lw=1.5, ls='--')
ax2.set_xticks(months); ax2.set_xticklabels(month_labels, fontsize=8)
ax2.set_ylabel('[10³ km³/Monat]', fontsize=9)
ax2.set_title('(b) Monatliche Nettobilanz $E - P$\n(Abweichung vom Gleichgewicht)',
              fontsize=10, fontweight='bold', color='#19468C')
ax2.grid(axis='y', alpha=0.3, ls=':')
ax2.text(0.5, 0.97, f'Jahressumme: {net.sum():.1f}×10³ km³ ≈ 0',
         transform=ax2.transAxes, ha='center', va='top', fontsize=9,
         color='#19468C',
         bbox=dict(boxstyle='round,pad=0.3', fc='#EEF3FF', ec='#19468C'))

# ---- (c) Langzeittrend 1980-2024 ----
ax3 = axes[1, 0]
years = np.arange(1980, 2025)
trend_e = 490000 + 1800 * (years - 1980) + np.random.normal(0, 8000, len(years))
trend_p = 490000 + 1800 * (years - 1980) + np.random.normal(0, 8000, len(years))
ax3.plot(years, trend_e/1000, color='#B4321E', lw=1.5, alpha=0.7,
         label='Verdunstung $E$')
ax3.plot(years, trend_p/1000, color='#19468C', lw=1.5, alpha=0.7,
         label='Niederschlag $P$')
# Trendlinie
coeffs = np.polyfit(years, trend_e/1000, 1)
trend_line = np.polyval(coeffs, years)
ax3.plot(years, trend_line, '--', color='#B4321E', lw=2,
         label=f'Trend: +{coeffs[0]:.1f} km³/a')
ax3.set_xlabel('Jahr', fontsize=9)
ax3.set_ylabel('[10³ km³/Jahr]', fontsize=9)
ax3.set_title('(c) Langzeittrend der globalen\nWassermassenbilanz (1980–2024)',
              fontsize=10, fontweight='bold', color='#19468C')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3, ls=':')

# ---- (d) Kreisdiagramm: Verteilung Verdunstung/Niederschlag ----
ax4 = axes[1, 1]
labels_pie = ['Ozeane\n(Verdunstung)\n502.800 km³/a',
              'Land\n(Evapotranspiration)\n74.200 km³/a']
labels_pie2 = ['Ozeane\n(Niederschlag)\n458.000 km³/a',
               'Land\n(Niederschlag)\n119.000 km³/a']
sizes_e = [502800, 74200]
sizes_p = [458000, 119000]
colors_e = ['#5B9BD5', '#8DC455']
colors_p = ['#2E75B6', '#538135']

x_off = -0.32
wedges1, texts1, auto1 = ax4.pie(sizes_e, labels=None,
                                   autopct='%1.1f%%', startangle=90,
                                   colors=colors_e,
                                   wedgeprops=dict(width=0.45),
                                   pctdistance=0.75,
                                   center=(x_off, 0), radius=0.9)
x_off2 = 0.32
wedges2, texts2, auto2 = ax4.pie(sizes_p, labels=None,
                                   autopct='%1.1f%%', startangle=90,
                                   colors=colors_p,
                                   wedgeprops=dict(width=0.45),
                                   pctdistance=0.75,
                                   center=(x_off2, 0), radius=0.9)
ax4.text(-0.32, 1.05, 'Verdunstung\n577.000 km³/a',
         ha='center', fontsize=9, fontweight='bold', color='#B4321E')
ax4.text(0.32, 1.05, 'Niederschlag\n577.000 km³/a',
         ha='center', fontsize=9, fontweight='bold', color='#19468C')
ax4.text(0, -1.15, r'$E_\mathrm{global} = P_\mathrm{global}$ ✓ Knotengleichung erfüllt',
         ha='center', fontsize=9, color='#1E6432', fontweight='bold')
leg_p = [mpatches.Patch(color=colors_e[0], label='Ozean'),
         mpatches.Patch(color=colors_e[1], label='Land')]
ax4.legend(handles=leg_p, loc='lower center', fontsize=8,
           ncol=2, bbox_to_anchor=(0.5, -1.38))
ax4.set_title('(d) Verteilung nach Ozean / Land', fontsize=10,
              fontweight='bold', color='#19468C')

plt.tight_layout(pad=2.5)
plt.savefig('plot4_globalbilanz.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot 4 gespeichert.")
