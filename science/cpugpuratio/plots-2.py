#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1,2,figsize=(12,5))

# --- Links: Theta(n^3) pro Konfiguration ---
ax = axes[0]
k_vals = np.arange(2, 32)
configs = {
    '1:1  $n=2k$': (2, '#1f77b4', '-'),
    '1:2  $n=3k$': (3, '#2ca02c', '--'),
    '1:4  $n=5k$': (5, '#ff7f0e', '-.'),
    '1:8  $n=9k$': (9, '#d62728', ':'),
}
for label,(factor,color,ls) in configs.items():
    n = k_vals * factor
    ax.semilogy(k_vals, n**3, color=color, lw=2.2, ls=ls, label=label)

ax.set_xlabel('Basisgröße $k$ (Knoten)', fontsize=11)
ax.set_ylabel('$\\Theta(n^3)$ Operationen (log)', fontsize=11)
ax.set_title('Subgraph-Isomorphismus-Komplexität\nnach CPU:GPU-Konfiguration', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')

# --- Rechts: Workload-Effizienz ---
ax2 = axes[1]
ratios_lbl = ['1:1','1:2','1:4','1:8']
x = np.arange(4)
w = 0.22
# Modellwerte: Inferenz, Training, Memory
inf_eff  = [1.00, 0.87, 0.65, 0.42]
train_eff= [0.55, 0.72, 0.95, 0.88]
mem_eff  = [0.90, 0.85, 0.70, 0.50]

b1 = ax2.bar(x-w,   inf_eff,   w, label='Inferenz (Prefill)',  color='#1f77b4', alpha=0.88)
b2 = ax2.bar(x,     train_eff, w, label='Training (Batch)',    color='#ff7f0e', alpha=0.88)
b3 = ax2.bar(x+w,   mem_eff,   w, label='Speicher-Effizienz', color='#2ca02c', alpha=0.88)

for bars in [b1,b2,b3]:
    for b in bars:
        h=b.get_height()
        ax2.text(b.get_x()+b.get_width()/2, h+0.01, f'{h:.2f}',
                 ha='center',va='bottom',fontsize=7)

ax2.set_xticks(x); ax2.set_xticklabels(ratios_lbl, fontsize=11)
ax2.set_xlabel('CPU:GPU-Verhältnis', fontsize=11)
ax2.set_ylabel('Normierte Effizienz [0,1]', fontsize=11)
ax2.set_title('Workload-spezifische Effizienz\nnach CPU:GPU-Konfiguration', fontsize=11, fontweight='bold')
ax2.set_ylim(0,1.18)
ax2.legend(fontsize=9)
ax2.grid(True,alpha=0.3,linestyle='--',axis='y')

plt.tight_layout()
plt.savefig('plots/plot2_complexity_efficiency.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 OK")

