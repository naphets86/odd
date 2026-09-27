#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

data = [
    (2019, 8,  "DGX A100 (1:8)",           "#d62728", "s"),
    (2021, 8,  "DGX H100 (1:8)",            "#ff7f0e", "s"),
    (2022, 4,  "Std. Server (1:4)",          "#8c564b", "^"),
    (2024, 2,  "Nvidia GB200 NVL2 (1:2)",   "#2ca02c", "D"),
    (2025, 1,  "Meta Catalina (1:1)",        "#1f77b4", "o"),
    (2026, 1,  "AMD Venice 1:1",             "#9467bd", "o"),
    (2026, 4,  "AMD Helios (1:4)",           "#e377c2", "P"),
]

# Trend
tx = np.array([2019, 2021, 2022, 2024, 2025])
ty = np.array([8,    8,    4,    2,    1   ])
z  = np.polyfit(tx, np.log(ty), 1)
xs = np.linspace(2018.5, 2027, 200)
ax.plot(xs, np.exp(np.poly1d(z)(xs)), 'k--', alpha=0.35, lw=1.8, label='Exponentieller Trend')

offsets = [( 6, 6),(-6,-16),(6,6),(6,6),(-90,6),(6,-16),(6,6)]
for i,(yr,ratio,label,color,marker) in enumerate(data):
    ax.scatter(yr, ratio, s=180, color=color, marker=marker, zorder=5,
               edgecolors='white', linewidths=1.5)
    ox,oy = offsets[i]
    ax.annotate(label,(yr,ratio),textcoords='offset points',xytext=(ox,oy),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle='->',color=color,lw=0.8) if abs(ox)>8 else None)

ax.set_xlabel('Jahr', fontsize=12)
ax.set_ylabel('GPU-Anzahl pro CPU', fontsize=12)
ax.set_title('Abbildung 1: Evolution des CPU:GPU-Verhältnisses\nin KI-Infrastruktur (2019–2026)', fontsize=13, fontweight='bold')
ax.set_xticks(range(2019,2028))
ax.set_yticks([1,2,4,8])
ax.set_yticklabels(['1:1','1:2','1:4','1:8'])
ax.set_ylim(0.4,11)
ax.set_xlim(2018.3,2027.5)
ax.grid(True,alpha=0.3,linestyle='--')
ax.legend(fontsize=9,loc='upper right')
plt.tight_layout()
plt.savefig('plots/plot1_ratio_evolution.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 OK")

