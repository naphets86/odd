#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Abbildung 12: Subgraph-Scheduling Integration und Ausblick CPU:GPU-Optimierung\n'
    'Dynamische Konfigurationsauswahl, Konvergenzanalyse und 2030-Roadmap Bielefeld',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40,
                       top=0.90, bottom=0.08, left=0.07, right=0.97)

# ── Plot 1: Scheduling-Effizienz mit q-Parameter ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bw_range = np.linspace(0, 400, 200)

def sched_eff(bw, q):
    bw_sat = 120 * q
    return min(0.98, (1 - np.exp(-bw / bw_sat)) * (1 + 0.02 / (1 + q/2)))

for q, col, lbl in [(1,'#19468C','q=1 (1:1)'), (2,'#2a9d8f','q=2 (1:2)'),
                     (4,'#B4321E','q=4 (1:4)'), (8,'#e76f51','q=8 (1:8)')]:
    eff_v = [sched_eff(b, q) * 100 for b in bw_range]
    ax1.plot(bw_range, eff_v, lw=2.5, color=col, label=f'$q={q}$')

ax1.axvline(100, color='green',  ls='-.', lw=1.5, alpha=0.7, label='100GbE (jetzt)')
ax1.axvline(400, color='purple', ls='-.', lw=1.5, alpha=0.7, label='400GbE (2028)')
ax1.axhline(95,  color='gray',   ls='--', lw=1.5, alpha=0.5, label='SLA 95%')
ax1.set_xlabel('Netzwerkbandbreite [Gbps]')
ax1.set_ylabel('Scheduling-Effizienz [%]')
ax1.set_title('Scheduling-Effizienz vs. BW\nnach CPU:GPU-Verhältnis $q$')
ax1.legend(fontsize=8, ncol=2)

# ── Plot 2: Konvergenz Matching-Qualität mit q ────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
iters = np.arange(0, 51)
for q, col, lbl in [(1,'#19468C','1:1'), (2,'#2a9d8f','1:2'),
                     (4,'#B4321E','1:4'), (8,'#e76f51','1:8')]:
    k_conv = 0.18 * (1 + 0.5/q)
    mq = 100 * (1 - np.exp(-k_conv * iters))
    ax2.plot(iters, mq, lw=2.5, color=col, label=f'Config {lbl}')

ax2.axhline(95, color='gray', ls='--', lw=1.5, alpha=0.6, label='SLA-Schwelle (95%)')
ax2.set_xlabel('Iterationen')
ax2.set_ylabel('Matching-Qualität [%]')
ax2.set_title('Matching-Konvergenz\nnach CPU:GPU-Konfiguration')
ax2.legend(fontsize=8)

# ── Plot 3: Dynamische Konfigurationsauswahl Simulation ──────────────
ax3 = fig.add_subplot(gs[0, 2])
np.random.seed(42)
t_sim = np.arange(0, 24*7)  # 7 Tage stündlich
inf_load = 0.4 + 0.35*np.sin(2*np.pi*t_sim/24) + 0.05*np.random.randn(len(t_sim))
inf_load = np.clip(inf_load, 0, 1)

# Optimal config based on inference fraction
def opt_q(xi):
    if xi > 0.75: return 1
    elif xi > 0.50: return 2
    elif xi > 0.25: return 4
    else: return 8

opt_configs = np.array([opt_q(xi) for xi in inf_load])
config_eff  = []
for xi, q in zip(inf_load, opt_configs):
    ei = {1:1.0, 2:0.87, 4:0.72, 8:0.42}
    et = {1:0.48, 2:0.72, 4:0.95, 8:0.88}
    config_eff.append(ei[q]*xi + et[q]*(1-xi))

static_eff_14 = [0.72*xi + 0.95*(1-xi) for xi in inf_load]

ax3_b = ax3.twinx()
ax3.fill_between(t_sim, inf_load*100, alpha=0.25, color='#19468C', label='Inf-Lastanteil')
ax3_b.plot(t_sim, config_eff, color='#1a7a3c', lw=2, label='Dynamisch (opt.)')
ax3_b.plot(t_sim, static_eff_14, color='#B4321E', lw=1.5, ls='--', label='Statisch 1:4')
ax3.set_xlabel('Zeit [h] (7 Tage)'); ax3.set_ylabel('Inferenz-Last [%]')
ax3_b.set_ylabel('Effizienz (normiert)')
ax3.set_title('Dynamische Konfigurationsauswahl\nim Tagesverlauf (Simulation)')
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_b.get_legend_handles_labels()
ax3.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='lower right')

# ── Plot 4: PUE-Verbesserung durch Config-Wahl ────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
server_util = np.linspace(10, 100, 100)
for q, col, lbl in [(1,'#19468C','1:1'), (2,'#2a9d8f','1:2'),
                     (4,'#B4321E','1:4'), (8,'#e76f51','1:8')]:
    # PUE ~ 1.4 - 0.2*(util/100) - 0.05*log(1+q)/log(9)
    pue = 1.5 - 0.25*(server_util/100) - 0.08*np.log1p(q)/np.log(9)
    ax4.plot(server_util, pue, lw=2.5, color=col, label=f'Config {lbl}')

ax4.axhline(1.2, color='green', ls='--', lw=2, label='Ideales PUE = 1.2')
ax4.set_xlabel('Serverauslastung [%]'); ax4.set_ylabel('PUE')
ax4.set_title('PUE nach CPU:GPU-Konfiguration\nund Serverauslastung')
ax4.legend(fontsize=8); ax4.set_ylim(1.0, 1.7)

# ── Plot 5: Roadmap BIE 2026–2030 ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_xlim(2025.5, 2030.8); ax5.set_ylim(-0.5, 4.5)
ax5.axis('off')
ax5.set_title('T-Systems Bielefeld Roadmap\nCPU:GPU-Optimierung 2026–2030', fontweight='bold')

milestones = [
    (2026.0, 3.5, '2026 Q1\nSubgraph-Pilot\n(1:4, 800 GPU)', '#19468C'),
    (2026.8, 2.5, '2026 Q4\nHybrid-Modus\n1:2 & 1:4', '#2a9d8f'),
    (2027.5, 3.5, '2027 Q2\n400GbE\nUpgrade', '#B4321E'),
    (2028.2, 2.5, '2028 Q1\n1:1 Pilotcluster\n(AMD Venice)', '#9b2226'),
    (2029.0, 3.5, '2029 Q3\nDynamisches\nScheduling v2', '#1a7a3c'),
    (2030.0, 2.5, '2030 Q1\n12.000 GPU\nVollbetrieb', '#19468C'),
]
y_line = 1.5
ax5.plot([2026, 2030.5], [y_line, y_line], 'k-', lw=3, alpha=0.3, zorder=1)
for yr, y_pos, label, col in milestones:
    ax5.plot(yr, y_line, 's', color=col, ms=12, zorder=5)
    ax5.plot([yr, yr], [y_line, y_pos], color=col, lw=1.5, ls=':', alpha=0.7)
    ax5.text(yr, y_pos + 0.15, label, ha='center', va='bottom', fontsize=8,
             color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=col+'22', edgecolor=col, lw=1))

ax5.text(2028, 0.5,
         'Subgraph Algorithmus ermöglicht\ndynamische Config-Auswahl bei jeder Phase',
         ha='center', fontsize=9, style='italic', color='#555',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff', edgecolor='#19468C', lw=1))

# ── Plot 6: Vergleich Effizienzgewinn-Potenzial erweitert ─────────────
ax6 = fig.add_subplot(gs[1, 2])
categories = ['GPU-\nAuslastung', 'Netzwerk-\nOpt.', 'Kühlung\n(PUE)',
              'Storage-\nTiering', 'CPU:GPU-\nRatio-Opt.', 'Dynamisches\nScheduling', 'Gesamt']
values_base   = [18.5, 12.3, 15.7, 8.2,  0.0,  0.0,  31.4]
values_cpugpu = [18.5, 12.3, 15.7, 8.2, 11.8, 14.2,  43.2]
colors_b = ['#19468C','#2a9d8f','#1a7a3c','#e76f51',
            '#B4321E','#9b2226','#6a0dad']

x = np.arange(len(categories))
bars1 = ax6.barh(x - 0.2, values_base,   0.35, label='Basisanalyse (Abb. 6)',
                 color=[c+'99' for c in colors_b], edgecolor='white')
bars2 = ax6.barh(x + 0.2, values_cpugpu, 0.35, label='Inkl. CPU:GPU-Opt.',
                 color=colors_b, alpha=0.9, edgecolor='white')

for bar, v in zip(bars2, values_cpugpu):
    if v > 0:
        ax6.text(v + 0.3, bar.get_y()+bar.get_height()/2, f'{v}%',
                 va='center', fontsize=8.5, color=colors_b[int(bar.get_y()+0.5) if False else bars2.index(bar)
                                                           if False else list(bars2).index(bar)],
                 fontweight='bold')

ax6.set_yticks(x); ax6.set_yticklabels(categories, fontsize=8.5)
ax6.set_xlabel('Effizienzgewinn / Energieersparnis [%]')
ax6.set_title('Erweitertes Optimierungspotenzial\ninkl. CPU:GPU-Ratio-Optimierung')
ax6.legend(fontsize=8, loc='lower right')
ax6.set_xlim(0, 50)

plt.savefig('plot_cpugpu_f.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_f.pdf saved")


