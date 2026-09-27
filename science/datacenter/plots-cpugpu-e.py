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
    'Abbildung 11: CPU-Nachfrageprognose 2026–2032 und Wirtschaftlichkeitsanalyse\n'
    'T-Systems Bielefeld — Szenarienvergleich nach CPU:GPU-Verhältnis',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40,
                       top=0.90, bottom=0.08, left=0.07, right=0.97)

years = np.array([2026, 2027, 2028, 2029, 2030, 2031, 2032])
gpu_proj = np.array([800, 2000, 5000, 8000, 12000, 16000, 22000])

# CPU counts per scenario
cpu_11 = gpu_proj // 1
cpu_12 = gpu_proj // 2
cpu_14 = gpu_proj // 4
cpu_18 = gpu_proj // 8

# ── Plot 1: CPU-Bedarf absolut ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(years, cpu_18, cpu_11, alpha=0.12, color='#19468C', label='Unsicherheitsband')
ax1.plot(years, cpu_11, 'o-', color='#19468C', lw=2.5, ms=7, label='Szenario 1:1')
ax1.plot(years, cpu_12, 's-', color='#2a9d8f', lw=2,   ms=7, label='Szenario 1:2')
ax1.plot(years, cpu_14, '^-', color='#B4321E', lw=2,   ms=7, label='Szenario 1:4')
ax1.plot(years, cpu_18, 'D-', color='#e76f51', lw=2,   ms=7, label='Szenario 1:8')
ax1.plot(years, gpu_proj, 'k--', lw=2, alpha=0.5, label='GPU-Einheiten')
ax1.set_xlabel('Jahr'); ax1.set_ylabel('Serveranzahl')
ax1.set_title('CPU-Bedarf nach Szenario\n(absolut, T-Systems Bielefeld)')
ax1.legend(fontsize=8, framealpha=0.9)
ax1.tick_params(axis='x', rotation=30)

# ── Plot 2: Kostendifferenz (CAPEX) ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
# Kosten pro CPU ~15k EUR, pro GPU ~25k EUR
cost_cpu = 15_000
cost_gpu = 25_000
cost_11 = (cpu_11 * cost_cpu + gpu_proj * cost_gpu) / 1e6
cost_12 = (cpu_12 * cost_cpu + gpu_proj * cost_gpu) / 1e6
cost_14 = (cpu_14 * cost_cpu + gpu_proj * cost_gpu) / 1e6
cost_18 = (cpu_18 * cost_cpu + gpu_proj * cost_gpu) / 1e6

ax2.stackplot(years, cost_11 - cost_18, labels=['Mehrkosten 1:1 vs. 1:8'],
              colors=['#19468C'], alpha=0.2)
ax2.plot(years, cost_11, 'o-', color='#19468C', lw=2.5, ms=7, label='CAPEX 1:1')
ax2.plot(years, cost_12, 's-', color='#2a9d8f', lw=2,   ms=7, label='CAPEX 1:2')
ax2.plot(years, cost_14, '^-', color='#B4321E', lw=2,   ms=7, label='CAPEX 1:4')
ax2.plot(years, cost_18, 'D-', color='#e76f51', lw=2,   ms=7, label='CAPEX 1:8')
ax2.set_xlabel('Jahr'); ax2.set_ylabel('CAPEX [Mio. EUR]')
ax2.set_title('CAPEX-Prognose nach Szenario\n(CPU + GPU Hardware, T-Systems BIE)')
ax2.legend(fontsize=8)
ax2.tick_params(axis='x', rotation=30)

# ── Plot 3: TCO (CAPEX + OPEX) ────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
# OPEX: power cost proportional to #servers * 0.5kW * 0.1 EUR/kWh * 8760h/yr
kw_cpu = 0.5; kw_gpu = 1.5
eur_kwh = 0.15
opex_11 = ((cpu_11*kw_cpu + gpu_proj*kw_gpu)*eur_kwh*8760)/1e6
opex_12 = ((cpu_12*kw_cpu + gpu_proj*kw_gpu)*eur_kwh*8760)/1e6
opex_14 = ((cpu_14*kw_cpu + gpu_proj*kw_gpu)*eur_kwh*8760)/1e6
opex_18 = ((cpu_18*kw_cpu + gpu_proj*kw_gpu)*eur_kwh*8760)/1e6

tco_11 = cost_11 + opex_11
tco_12 = cost_12 + opex_12
tco_14 = cost_14 + opex_14
tco_18 = cost_18 + opex_18

ax3.plot(years, tco_11, 'o-', color='#19468C', lw=2.5, ms=7, label='TCO 1:1')
ax3.plot(years, tco_12, 's-', color='#2a9d8f', lw=2,   ms=7, label='TCO 1:2')
ax3.plot(years, tco_14, '^-', color='#B4321E', lw=2,   ms=7, label='TCO 1:4')
ax3.plot(years, tco_18, 'D-', color='#e76f51', lw=2,   ms=7, label='TCO 1:8')
ax3.fill_between(years, tco_18, tco_11, alpha=0.1, color='#19468C')
ax3.set_xlabel('Jahr'); ax3.set_ylabel('TCO [Mio. EUR/Jahr]')
ax3.set_title('TCO-Prognose (CAPEX+OPEX)\nnach CPU:GPU-Szenario')
ax3.legend(fontsize=8)
ax3.tick_params(axis='x', rotation=30)

# ── Plot 4: Subgraph Algorithmus-Laufzeit nach q bei DC-Skala ─────────
ax4 = fig.add_subplot(gs[1, 0])
q_values = [1, 2, 4, 8, 16]
n_dc = 512
for n_scale, label, col in [(8,'ToR (8 GPU)','#e76f51'),
                              (32,'Rack (32)','#B4321E'),
                              (128,'Pod (128)','#2a9d8f'),
                              (512,'DC (512)','#19468C')]:
    ops = [((n_scale * (1 + q))**3) for q in q_values]
    ax4.semilogy(q_values, ops, 'o-', lw=2, ms=7, label=label, color=col)
ax4.set_xlabel('GPU:CPU-Verhältnis $q$')
ax4.set_ylabel('Operationen $\\Theta(n_T^3)$ (log)')
ax4.set_title('Subgraph-Laufzeit vs. $q$\nnach Rechenzentrum-Skala')
ax4.legend(fontsize=8); ax4.set_xticks(q_values)

# ── Plot 5: Optimale Konfiguration nach Workload-Mix ──────────────────
ax5 = fig.add_subplot(gs[1, 1])
x_inf = np.linspace(0, 1, 100)
x_train = 1 - x_inf

def score(q, xi, xt):
    ei = {1:1.0, 2:0.87, 4:0.72, 8:0.42}
    et = {1:0.48, 2:0.72, 4:0.95, 8:0.88}
    return ei[q]*xi + et[q]*xt

for q, col, lbl in [(1,'#19468C','q=1 (1:1)'), (2,'#2a9d8f','q=2 (1:2)'),
                     (4,'#B4321E','q=4 (1:4)'), (8,'#e76f51','q=8 (1:8)')]:
    ax5.plot(x_inf*100, score(q, x_inf, x_train), lw=2.5, color=col, label=lbl)

# Intersection points
for q1, q2, col1, col2 in [(1,2,'#19468C','#2a9d8f'), (2,4,'#2a9d8f','#B4321E'),
                             (4,8,'#B4321E','#e76f51')]:
    ei1,et1 = {1:1.0,2:0.87,4:0.72,8:0.42}[q1], {1:0.48,2:0.72,4:0.95,8:0.88}[q1]
    ei2,et2 = {1:1.0,2:0.87,4:0.72,8:0.42}[q2], {1:0.48,2:0.72,4:0.95,8:0.88}[q2]
    # score1 = score2: (ei1-et1)*xi + et1 = (ei2-et2)*xi + et2
    denom = (ei1-et1) - (ei2-et2)
    if abs(denom) > 1e-9:
        xi_cross = (et2 - et1) / denom
        if 0 < xi_cross < 1:
            ax5.axvline(xi_cross*100, color='gray', ls=':', lw=1.5, alpha=0.6)
            ax5.text(xi_cross*100+0.5, 0.50, f'{xi_cross*100:.0f}%', fontsize=8, color='gray')

ax5.set_xlabel('Anteil Inferenz-Workload [%]')
ax5.set_ylabel('Gewichtete Effizienz')
ax5.set_title('Optimale Konfiguration\nnach Workload-Mix (Inferenz vs. Training)')
ax5.legend(fontsize=8, loc='center right')
ax5.set_xlim(0, 100); ax5.set_ylim(0.4, 1.05)
ax5.text(10, 0.97, '← Training\ndominiert', ha='left', fontsize=8, color='#555')
ax5.text(85, 0.97, 'Inferenz\ndominiert →', ha='right', fontsize=8, color='#555')

# ── Plot 6: Marktanteile CPU-Plattform 2024–2030 ─────────────────────
ax6 = fig.add_subplot(gs[1, 2])
years_m = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
share_intel = [55, 48, 40, 32, 25, 20, 18]
share_amd   = [30, 35, 38, 42, 48, 52, 55]
share_arm   = [10, 12, 16, 20, 22, 23, 22]
share_other = [5,  5,  6,  6,  5,  5,  5 ]

ax6.stackplot(years_m,
              share_intel, share_amd, share_arm, share_other,
              labels=['Intel Xeon', 'AMD EPYC', 'ARM (Neoverse)', 'Andere'],
              colors=['#0071C5', '#ED1C24', '#1a7a3c', '#999'],
              alpha=0.85)
ax6.set_xlabel('Jahr'); ax6.set_ylabel('Marktanteil CPU-Server [%]')
ax6.set_title('Prognose Marktanteile\nCPU-Server im KI-RZ-Segment')
ax6.legend(loc='lower left', fontsize=8, framealpha=0.95)
ax6.set_ylim(0, 100); ax6.tick_params(axis='x', rotation=30)

plt.savefig('plot_cpugpu_e.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_e.pdf saved")


