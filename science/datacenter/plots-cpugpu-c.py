#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Abbildung 9: Speicherbandbreite, Rechenleistung und Signatur-Analyse der Hardware-Konfigurationen\n'
    'Roofline-Modell, Signatur-Arrays und Komplexitätsvergleich',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40,
                       top=0.90, bottom=0.08, left=0.07, right=0.97)

# ── Plot 1: Roofline Memory BW vs Compute ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])

platforms = {
    'DGX A100 (1:8)':        {'bw': 2000, 'flops': 5000,  'ratio': '1:8',  'color': '#e76f51', 'marker': 'v'},
    'NVIDIA GB200 NVL2 (1:2)':{'bw': 4800, 'flops': 20000, 'ratio': '1:2',  'color': '#2a9d8f', 'marker': 's'},
    'AMD Helios (1:4)':       {'bw': 3200, 'flops': 12000, 'ratio': '1:4',  'color': '#B4321E', 'marker': '^'},
    'Meta Catalina (1:1)':    {'bw': 3600, 'flops': 8000,  'ratio': '1:1',  'color': '#19468C', 'marker': 'D'},
    'AMD Venice (1:1)':       {'bw': 4000, 'flops': 9500,  'ratio': '1:1',  'color': '#1a7a3c', 'marker': 'o'},
    'NVL72 Blackwell (1:4)':  {'bw': 7200, 'flops': 72000, 'ratio': '1:4',  'color': '#9b2226', 'marker': '*'},
    'Grace-Blackwell (1:1)':  {'bw': 8000, 'flops': 15000, 'ratio': '1:1',  'color': '#0077b6', 'marker': 'P'},
}

bw_range = np.logspace(3, 4.1, 200)
for ai, label in [(0.5, '0.5'), (1.0, '1.0'), (2.0, '2.0'), (5.0, '5.0')]:
    ax1.loglog(bw_range, ai * bw_range, '--', color='gray', alpha=0.3, lw=1)
    ax1.text(bw_range[-1]*0.85, ai * bw_range[-1]*1.05,
             f'AI={label}', fontsize=7.5, color='gray', alpha=0.8)

for name, p in platforms.items():
    ax1.scatter(p['bw'], p['flops'], s=180, color=p['color'],
                marker=p['marker'], zorder=5, label=name)
    ax1.annotate(name.split('(')[0].strip(), (p['bw'], p['flops']),
                 textcoords='offset points', xytext=(6, 4), fontsize=7.5,
                 color=p['color'])

# Trend arrow: from 1:8 to 1:1
ax1.annotate('', xy=(4000, 9500), xytext=(2000, 5000),
             arrowprops=dict(arrowstyle='->', color='purple', lw=2.5,
                             connectionstyle='arc3,rad=-0.2'))
ax1.text(2800, 8000, 'Trend:\n1:8 → 1:1', fontsize=9, color='purple',
         fontweight='bold', ha='center')

ax1.set_xlabel('Speicherbandbreite [GB/s]')
ax1.set_ylabel('Rechenleistung [TFLOPS]')
ax1.set_title('Roofline-Modell: Speicherbandbreite vs. Rechenleistung\nrealer KI-Plattformen (log-log, Isochronen konstanter arithmetischer Intensität)')
ax1.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.9)

# ── Plot 2: Signatur-Arrays ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])

configs_sig = {
    '1:1 ($n=2$)':   [2, 5],
    '1:2 ($n=3$)':   [6, 13, 19],
    '1:4 ($n=5$)':   [30, 61, 93, 125, 157],
    '1:8 ($n=9$)':   [510, 1021, 1533, 2045, 2557, 3069, 3581, 4093, 4605],
}
colors_sig = ['#19468C', '#2a9d8f', '#B4321E', '#e76f51']
markers_sig = ['o', 's', '^', 'D']

for (label, sigs), col, mk in zip(configs_sig.items(), colors_sig, markers_sig):
    ax2.plot(range(len(sigs)), sigs, marker=mk, color=col, lw=2, ms=7,
             label=label)
    for idx, sv in enumerate(sigs):
        ax2.annotate(str(sv), (idx, sv), textcoords='offset points',
                     xytext=(3, 4), fontsize=6.5, color=col)

ax2.set_xlabel('Spaltenindex $j$')
ax2.set_ylabel('Signaturwert $\\sigma_j$')
ax2.set_title('Signatur-Arrays $\\boldsymbol{\\sigma}(G_r)$\nnach CPU:GPU-Konfiguration')
ax2.legend(fontsize=8, framealpha=0.9)
ax2.set_yscale('log')

# ── Plot 3: Komplexität n^3 nach Konfiguration ─────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
config_list  = ['1:1', '1:2', '1:4', '1:8']
q_list       = [1,     2,     4,     8    ]
colors_c     = ['#19468C', '#2a9d8f', '#B4321E', '#e76f51']
k_cpus       = 10

n_t  = [k_cpus * (1 + q) for q in q_list]
ops  = [(nt**3) for nt in n_t]
bars = ax3.bar(config_list, ops, color=colors_c, alpha=0.85, edgecolor='white', lw=1.5)
for bar, op, nt in zip(bars, ops, n_t):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
             f'$n_T={nt}$\n$\\Theta({op:,.0f})$',
             ha='center', va='bottom', fontsize=8, fontweight='bold')
ax3.set_xlabel('CPU:GPU-Konfiguration')
ax3.set_ylabel('Operationen $\\Theta(n_T^3)$')
ax3.set_title(f'Subgraph Algorithmus Komplexität\n($k={k_cpus}$ CPU-Einheiten)')
ax3.set_yscale('log')

# ── Plot 4: Bandbreite-Anforderung nach Workload + Config ─────────────
ax4 = fig.add_subplot(gs[1, 1])
workloads = ['LLM-\nTraining', 'LLM-\nInferenz', 'KI-\nBildverarbeitung', 'HPC-\nSimulation']
bw_req   = [800, 300, 150, 500]  # GB/s required by workload
# Available BW per config
bw_avail = {
    '1:1': 3800, '1:2': 2400, '1:4': 1600, '1:8': 800
}
x = np.arange(len(workloads))
w = 0.18
for i, (cfg, bw) in enumerate(bw_avail.items()):
    ax4.bar(x + (i-1.5)*w, [bw]*len(workloads), w,
            alpha=0.4, label=f'{cfg} (avail.)', color=colors_c[i])
ax4.bar(x, bw_req, 0.08, color='black', alpha=0.85, label='Workload-Bedarf', zorder=5)
ax4.set_xticks(x); ax4.set_xticklabels(workloads)
ax4.set_xlabel('Workload-Typ')
ax4.set_ylabel('Speicherbandbreite [GB/s]')
ax4.set_title('Speicherbandbreite:\nAngebot vs. Nachfrage nach Konfiguration')
ax4.legend(fontsize=7.5, ncol=2)

# ── Plot 5: Energieeffizienz Vergleich ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
config_labels = ['1:1', '1:2', '1:4', '1:8']
flops_per_watt = [12.5, 15.8, 18.2, 14.3]  # TFLOPS/kW
pue_optimal    = [1.30, 1.25, 1.20, 1.22]

ax5b = ax5.twinx()
bars_e = ax5.bar(config_labels, flops_per_watt, color=colors_c, alpha=0.7,
                 label='TFLOPS/kW')
ax5b.plot(config_labels, pue_optimal, 'ko-', ms=8, lw=2, label='PUE (optimal)')
ax5b.set_ylabel('PUE (optimal)', color='black')
ax5.set_xlabel('CPU:GPU-Konfiguration')
ax5.set_ylabel('TFLOPS / kW', color='#333')
ax5.set_title('Energieeffizienz nach Konfiguration\nTFLOPS/kW und optimales PUE')
for bar, v in zip(bars_e, flops_per_watt):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v:.1f}',
             ha='center', fontsize=9, fontweight='bold')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1+lines2, labels1+labels2, fontsize=8)

plt.savefig('plot_cpugpu_c.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_c.pdf saved")


