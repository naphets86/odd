import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data: approximate global HIV statistics (UNAIDS estimates)
years = np.array([1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
                  2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
                  2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
                  2020,2021,2022,2023])

# New infections per year (millions)
new_infections = np.array([1.9,2.1,2.4,2.7,2.9,3.1,3.3,3.2,3.1,3.0,
                            3.0,2.9,2.8,2.7,2.6,2.5,2.3,2.2,2.1,2.0,
                            1.9,1.8,1.7,1.7,1.7,1.7,1.7,1.6,1.6,1.5,
                            1.5,1.5,1.3,1.3])

# AIDS-related deaths per year (millions)
deaths = np.array([0.3,0.4,0.5,0.7,0.8,1.1,1.2,1.3,1.4,1.6,
                   1.7,1.8,1.9,2.0,2.1,2.2,2.1,2.1,2.0,1.9,
                   1.7,1.6,1.4,1.3,1.2,1.1,1.0,0.95,0.88,0.82,
                   0.77,0.73,0.63,0.63])

# People living with HIV (millions)
plhiv = np.array([8,9.7,11.5,13.4,15.5,17.6,19.6,21.5,23.3,25.0,
                  26.8,28.5,30.0,31.4,32.7,33.9,34.8,35.5,35.9,36.0,
                  36.0,36.3,36.7,37.0,37.2,37.5,37.7,37.9,38.0,38.0,
                  38.4,38.6,39.0,39.9])

fig, axes = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('Globale HIV/AIDS-Epidemie: 1990–2023', fontsize=15, fontweight='bold', y=0.98)

# Top: New infections and deaths
ax1 = axes[0]
ax1.fill_between(years, new_infections, alpha=0.25, color='#1946a0', label='_nolegend_')
ax1.plot(years, new_infections, '-o', color='#1946a0', linewidth=2.0, markersize=4,
         label='Neuinfektionen pro Jahr (Mio.)')
ax1.fill_between(years, deaths, alpha=0.25, color='#b4321e', label='_nolegend_')
ax1.plot(years, deaths, '-s', color='#b4321e', linewidth=2.0, markersize=4,
         label='AIDS-bedingte Todesfälle pro Jahr (Mio.)')

ax1.axvline(x=1996, color='#1e6432', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.text(1996.4, 2.9, 'cART\nEinführung\n1996', color='#1e6432', fontsize=8.5, va='top')
ax1.axvline(x=2004, color='#8B6914', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.text(2004.4, 2.6, 'PEPFAR\n2004', color='#8B6914', fontsize=8.5, va='top')

ax1.set_ylabel('Millionen Personen', fontsize=11)
ax1.set_xlabel('Jahr', fontsize=11)
ax1.legend(loc='upper right', fontsize=9.5)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1989, 2024)
ax1.set_ylim(0, 3.8)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Bottom: People living with HIV
ax2 = axes[1]
ax2.fill_between(years, plhiv, alpha=0.2, color='#6a0dad')
ax2.plot(years, plhiv, '-D', color='#6a0dad', linewidth=2.0, markersize=4,
         label='Menschen mit HIV (Mio.)')
ax2.axhline(y=39.9, color='#6a0dad', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.text(1990.5, 40.3, '39,9 Mio. (2023)', color='#6a0dad', fontsize=9)

ax2.set_ylabel('Millionen Personen', fontsize=11)
ax2.set_xlabel('Jahr', fontsize=11)
ax2.legend(loc='upper left', fontsize=9.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1989, 2024)
ax2.set_ylim(0, 46)

# Annotate cumulative deaths
ax2.text(2015, 5, '≈ 40 Mio. kumulierte\nTodesfälle seit 1981',
         fontsize=10, color='#b4321e',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0ee', edgecolor='#b4321e', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('home/claude/plot1_epidemic.png', dpi=150, bbox_inches='tight')
print("Plot 1 saved.")
