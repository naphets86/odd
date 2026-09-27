#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Abbildung 7: CPU:GPU-Verhältnis-Evolution und T-Systems Bielefeld Projektion\n'
    'Historische Entwicklung, Hardware-Konfigurationsvergleich und Bielefeld-Roadmap',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       top=0.90, bottom=0.08, left=0.07, right=0.97)

# ── Plot 1: Evolution CPU:GPU ratio 2019–2030 ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
years_hist   = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
ratios_hist  = [8,    6,    4,    4,    2,    2,    1,    1   ]
systems_hist = ['DGX A100\n1:8', '', 'DGX A100\n1:4', '',
                'GB200 NVL2\n1:2', '', 'Meta Catalina\n1:1', 'AMD Venice\n1:1']

years_proj  = [2026, 2027, 2028, 2029, 2030]
# Scenario 1: stay at 1:1
ratio_s1 = [1, 1, 1, 0.8, 0.5]
# Scenario 2: hybrid re-consolidation
ratio_s2 = [1, 1.5, 2, 2.5, 3]
# T-Systems Bielefeld specific
ratio_bie = [8, 8, 6, 4, 4, 2, 1.5, 1, 1, 1.2, 1.5]
years_bie  = [2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2030]

t = np.linspace(2019, 2026, 200)
lam = 0.35
ratio_exp = 8 * np.exp(-lam * (t - 2019))

ax1.plot(t, ratio_exp, '--', color='#aaa', lw=1.5, label='Exponentialtrend $8e^{-0.35(t-2019)}$', zorder=1)
ax1.plot(years_hist, ratios_hist, 'o-', color='#19468C', lw=2.5, ms=8,
         label='Reale Systeme (historisch)', zorder=3)
ax1.plot(years_proj, ratio_s1, 's--', color='#1a7a3c', lw=2, ms=7,
         label='Szenario S1: 1:1-Dominanz', zorder=2)
ax1.plot(years_proj, ratio_s2, '^--', color='#B4321E', lw=2, ms=7,
         label='Szenario S2: Hybrid-Konsolidierung', zorder=2)
ax1.plot(years_bie, ratio_bie, 'D-', color='#E07000', lw=2, ms=7,
         label='T-Systems Bielefeld (projiziert)', zorder=4)

for yr, rt, lbl in zip(years_hist, ratios_hist, systems_hist):
    if lbl:
        ax1.annotate(lbl, (yr, rt), textcoords='offset points', xytext=(0, 12),
                     ha='center', fontsize=7.5, color='#19468C',
                     arrowprops=dict(arrowstyle='->', color='#19468C', lw=0.8))

ax1.axvline(2026, color='purple', ls=':', lw=1.5, alpha=0.7, label='Heute (2026)')
ax1.set_xlabel('Jahr')
ax1.set_ylabel('GPU:CPU-Verhältnis (q)')
ax1.set_title('Historische Evolution & Prognose CPU:GPU-Verhältnis in KI-RZ')
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.set_xlim(2018.5, 2030.5)
ax1.set_ylim(-0.2, 9)
ax1.set_xticks(range(2019, 2031))
ax1.tick_params(axis='x', rotation=30)

# ── Plot 2: Anzahl CPUs Bielefeld 2024–2030 ──────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
years_b = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
gpu_b   = [200,  400,  800, 2000, 5000, 8000, 12000]
# CPUs: 1:8 → 1:1 transition → cpu = gpu * ratio_factor
cpu_b   = [25,   60,  180,  700, 2000, 4000,  8000]
cpu_b2  = [25,   60,  180,  300,  600,  800,  1200]  # 1:4 scenario

ax2.fill_between(years_b, cpu_b2, cpu_b, alpha=0.2, color='#1a7a3c', label='CPU-Spanne')
ax2.plot(years_b, cpu_b,  'o-', color='#1a7a3c', lw=2, ms=6, label='CPUs (1:1-Szenario)')
ax2.plot(years_b, cpu_b2, 's-', color='#1a7a3c', lw=2, ms=6, ls='--', label='CPUs (1:4-Szenario)')
ax2.plot(years_b, gpu_b,  '^-', color='#B4321E', lw=2, ms=6, label='GPUs (KI)')
ax2.set_xlabel('Jahr')
ax2.set_ylabel('Serveranzahl')
ax2.set_title('CPU-Bedarf T-Systems Bielefeld\nnach CPU:GPU-Szenario')
ax2.legend(fontsize=8, framealpha=0.9)
ax2.tick_params(axis='x', rotation=30)

# ── Plot 3: Workload-Effizienz nach Konfiguration ─────────────────────
ax3 = fig.add_subplot(gs[1, 0])
configs = ['1:1', '1:2', '1:4', '1:8']
eff_inf   = [1.00, 0.87, 0.72, 0.42]
eff_train = [0.48, 0.72, 0.95, 0.88]
eff_mixed = [0.74, 0.80, 0.84, 0.65]

x = np.arange(len(configs))
w = 0.25
bars1 = ax3.bar(x - w, eff_inf,   w, label='Inferenz',  color='#19468C', alpha=0.85)
bars2 = ax3.bar(x,     eff_train, w, label='Training',  color='#B4321E', alpha=0.85)
bars3 = ax3.bar(x + w, eff_mixed, w, label='Gemischt',  color='#1a7a3c', alpha=0.85)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=8)
ax3.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(configs)
ax3.set_xlabel('CPU:GPU-Konfiguration')
ax3.set_ylabel('Normierte Effizienz')
ax3.set_title('Workload-spezifische Effizienz\nnach Konfiguration (normiert)')
ax3.legend(fontsize=8)
ax3.set_ylim(0, 1.15)

# ── Plot 4: Pareto-Frontier Inference vs Training ─────────────────────
ax4 = fig.add_subplot(gs[1, 1])
pareto_inf   = [1.00, 0.87, 0.72, 0.48]
pareto_train = [0.48, 0.72, 0.95, 0.88]
colors_p = ['#19468C', '#2a9d8f', '#B4321E', '#e76f51']
labels_p = ['1:1', '1:2', '1:4', '1:8']

ax4.scatter(pareto_inf, pareto_train, c=colors_p, s=200, zorder=5)
# Pareto frontier
pareto_pts = sorted(zip(pareto_inf, pareto_train), key=lambda x: x[0])
ax4.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
         'k--', lw=1.5, alpha=0.4, label='Pareto-Frontier')

for i, (xi, yi) in enumerate(zip(pareto_inf, pareto_train)):
    ax4.annotate(f'$G_{{\\text{{{labels_p[i]}}}}}$', (xi, yi),
                 textcoords='offset points', xytext=(8, 5), fontsize=10,
                 color=colors_p[i], fontweight='bold')

ax4.axhline(0.90, color='#1a7a3c', ls=':', lw=1.5, alpha=0.7, label='SLA-Training (90%)')
ax4.axvline(0.85, color='#19468C', ls=':', lw=1.5, alpha=0.7, label='SLA-Inferenz (85%)')
ax4.set_xlabel('Inferenz-Effizienz (normiert)')
ax4.set_ylabel('Training-Effizienz (normiert)')
ax4.set_title('Pareto-Frontier: Inferenz vs.\nTraining-Effizienz')
ax4.legend(fontsize=8)
ax4.set_xlim(0.3, 1.1); ax4.set_ylim(0.3, 1.1)
# Best Compromise
ax4.annotate('Optimaler\nKompromiss', (0.87, 0.72),
             textcoords='offset points', xytext=(-70, -30),
             fontsize=8.5, color='#2a9d8f',
             arrowprops=dict(arrowstyle='->', color='#2a9d8f'))

# ── Plot 5: Algorithmus-Komplexität nach Konfiguration ────────────────
ax5 = fig.add_subplot(gs[1, 2])
n_values = np.linspace(5, 100, 200)
k = 10  # racks
for ratio_label, q, color_c in [('1:1', 1, '#19468C'), ('1:2', 2, '#2a9d8f'),
                                  ('1:4', 4, '#B4321E'), ('1:8', 8, '#e76f51')]:
    nt = k * (1 + q) * (n_values / 10)
    ops = nt**3
    ax5.loglog(n_values, ops, lw=2, label=f'$G_{{1:{ratio_label.split(":")[1]}}}$', color=color_c)

# Mark RZ-Bielefeld relevant sizes
for label, n_mark, q_mark in [('ToR\n(8)', 8, 1), ('Rack\n(32)', 32, 4), ('DC\n(512)', 512, 8)]:
    ax5.axvline(n_mark, color='gray', ls=':', lw=1, alpha=0.5)

ax5.set_xlabel('Knotenanzahl $n$')
ax5.set_ylabel('Operationen $\\Theta(n_T^3)$')
ax5.set_title('Subgraph Algorithmus Komplexität\nnach CPU:GPU-Konfiguration')
ax5.legend(fontsize=8)

plt.savefig('plot_cpugpu_a.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_a.pdf saved")


