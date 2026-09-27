#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig, axes = plt.subplots(1,2,figsize=(13,5))

# --- Links: CPU-Nachfrage-Prognose ---
ax = axes[0]
years_hist  = np.array([2022,2023,2024,2025,2026])
cpu_demand  = np.array([1.0, 1.3, 1.8, 2.9, 4.5])   # normiert
years_proj  = np.array([2026,2027,2028,2029,2030,2031,2032])
cpu_proj_11 = np.array([4.5, 6.2, 9.1, 13.0, 18.5, 26.0, 36.0])  # 1:1 Trend
cpu_proj_14 = np.array([4.5, 5.3, 6.8,  8.5, 11.0, 14.5, 19.0])  # 1:4 Trend

ax.plot(years_hist, cpu_demand, 'k-o', lw=2.5, ms=8, label='Historisch', zorder=5)
ax.plot(years_proj, cpu_proj_11,'#1f77b4','--',lw=2.2,
        marker='D',ms=7,label='Prognose: 1:1-Trend (AMD/Meta)')
ax.plot(years_proj, cpu_proj_14,'#ff7f0e','--',lw=2.2,
        marker='s',ms=7,label='Prognose: 1:4-Trend (klassisch)')

ax.fill_between(years_proj, cpu_proj_14, cpu_proj_11, alpha=0.12, color='#1f77b4',
                label='Unterschied Szenarien')
ax.axvline(2026, color='gray', ls=':', lw=1.3, label='Gegenwart (2026)')

ax.set_xlabel('Jahr', fontsize=11)
ax.set_ylabel('Normierte CPU-Nachfrage (KI-Infrastruktur)', fontsize=11)
ax.set_title('Abbildung 5a: CPU-Nachfrage-Prognose\nnach CPU:GPU-Ratio-Szenario', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)
ax.grid(True,alpha=0.3,linestyle='--')
ax.set_xticks(list(years_hist)+list(years_proj[1:]))
ax.tick_params(axis='x',rotation=30)

# --- Rechts: Θ(n³)-Skalierung Gesamt-Topologie ---
ax2 = axes[1]
n = np.linspace(2,100,300)

configs_2 = {
    '1:1  $n_T=2n$': (lambda n: (2*n)**3, '#1f77b4','-'),
    '1:2  $n_T=3n$': (lambda n: (3*n)**3, '#2ca02c','--'),
    '1:4  $n_T=5n$': (lambda n: (5*n)**3, '#9467bd','-.'),
    '1:8  $n_T=9n$': (lambda n: (9*n)**3, '#d62728',':'),
}
for lbl,(fn,col,ls) in configs_2.items():
    ax2.loglog(n, fn(n), color=col, lw=2.2, ls=ls, label=lbl)

# Annotation Schnittpunkte
n50 = 50
for lbl,(fn,col,ls) in configs_2.items():
    ax2.scatter([n50],[fn(n50)],color=col,s=60,zorder=5)

ax2.set_xlabel('Knotenanzahl $n$ (je Einheit)', fontsize=11)
ax2.set_ylabel('$\\Theta(n_T^3)$ (log-log)', fontsize=11)
ax2.set_title('Abbildung 5b: Gesamt-Komplexität $\\Theta(n_T^3)$\nnach Rack-Konfiguration', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True,alpha=0.3,which='both',linestyle='--')

plt.tight_layout()
plt.savefig('plots/plot5_forecast_complexity.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 OK")

