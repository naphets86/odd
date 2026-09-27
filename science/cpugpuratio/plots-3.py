#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(11,6))

platforms = {
    'DGX A100 (1:8)':       dict(bw=2000,  tflops=312,   ratio=8, c='#d62728', m='s'),
    'DGX H100 (1:8)':       dict(bw=3350,  tflops=989,   ratio=8, c='#ff7f0e', m='s'),
    'GB200 NVL2 (1:2)':     dict(bw=8000,  tflops=4500,  ratio=2, c='#2ca02c', m='D'),
    'Meta Catalina (1:1)':  dict(bw=4000,  tflops=2250,  ratio=1, c='#1f77b4', m='o'),
    'AMD Helios (1:4)':     dict(bw=12800, tflops=2900,  ratio=4, c='#9467bd', m='P'),
    'AMD Venice 1:1':       dict(bw=3200,  tflops=1450,  ratio=1, c='#e377c2', m='o'),
    'GB200 NVL72 (1:2)':    dict(bw=288000,tflops=162000,ratio=2, c='#17becf', m='^'),
}

size_map = {1:220, 2:160, 4:120, 8:90}
offsets  = {
    'DGX A100 (1:8)':      ( 8,  4),
    'DGX H100 (1:8)':      ( 8, -14),
    'GB200 NVL2 (1:2)':    ( 8,  4),
    'Meta Catalina (1:1)': ( 8,  4),
    'AMD Helios (1:4)':    ( 8,  4),
    'AMD Venice 1:1':      ( 8,-14),
    'GB200 NVL72 (1:2)':   ( 8,  4),
}

for name, sp in platforms.items():
    x = np.log10(sp['bw'])
    y = np.log10(sp['tflops'])
    ax.scatter(x, y, s=size_map[sp['ratio']]*1.5,
               color=sp['c'], marker=sp['m'], zorder=5,
               edgecolors='white', linewidths=1.2, label=f"{name}")
    ox,oy = offsets[name]
    ax.annotate(name,(x,y),textcoords='offset points',xytext=(ox,oy),
                fontsize=8,color=sp['c'])

# Isochoren arith. Intensität
xr = np.linspace(3.0, 5.6, 200)
for ai,ls,alpha in [(0.5,':',0.4),(1.0,'--',0.4),(2.0,'-.',0.4)]:
    ax.plot(xr, xr+np.log10(ai), 'gray', ls=ls, alpha=alpha, lw=1,
            label=f'Arith. Intensität ×{ai}')

ax.set_xlabel('$\\log_{{10}}$(Speicherbandbreite [GB/s])', fontsize=12)
ax.set_ylabel('$\\log_{{10}}$(Rechenleistung [TFLOP/s])', fontsize=12)
ax.set_title('Abbildung 3: Speicherbandbreite vs. Rechenleistung\nnach Plattform und CPU:GPU-Verhältnis', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/plot3_memory_compute.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 OK")

