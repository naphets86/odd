#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Abbildung 10: Einbettungshierarchie und formale Subgraph-Beziehungen\n'
    '$G_{1:1} \\hookrightarrow G_{1:2} \\hookrightarrow G_{1:4} \\hookrightarrow G_{1:8}$',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.40,
                       top=0.90, bottom=0.08, left=0.06, right=0.97)

# ── Plot 1: Hierarchie-Diagramm (Hasse) ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10); ax1.set_ylim(-1, 4)
ax1.axis('off')
ax1.set_title('Einbettungshierarchie der Hardware-Topologiegraphen\n'
              '$G_{1:1} \\hookrightarrow G_{1:2} \\hookrightarrow G_{1:4} \\hookrightarrow G_{1:8}$'
              ' (Satz 1, Beweisstruktur)', fontsize=11, fontweight='bold', color='#19468C')

configs = [
    (1.5, 1.5, '$G_{1:1}$\n$n=2$\n$|E|=2$', '#19468C'),
    (3.5, 1.5, '$G_{1:2}$\n$n=3$\n$|E|=6$', '#2a9d8f'),
    (5.8, 1.5, '$G_{1:4}$\n$n=5$\n$|E|=20$', '#B4321E'),
    (8.5, 1.5, '$G_{1:8}$\n$n=9$\n$|E|=72$', '#e76f51'),
]

for x, y, label, col in configs:
    box = FancyBboxPatch((x-0.85, y-0.75), 1.7, 1.5,
                          boxstyle='round,pad=0.1', linewidth=2,
                          edgecolor=col, facecolor=col+'22')
    ax1.add_patch(box)
    ax1.text(x, y, label, ha='center', va='center', fontsize=10,
             color=col, fontweight='bold')

embeddings = [
    (2.35, 1.5, 2.65, 1.5, '$\\phi_1$: $c_1\\mapsto c_1$\n$\\gamma_1\\mapsto\\gamma_1$'),
    (4.65, 1.5, 4.95, 1.5, '$\\phi_2$: +$\\gamma_2,\\gamma_3$\nmapped'),
    (6.65, 1.5, 7.65, 1.5, '$\\phi_3$: +$\\gamma_5,..,\\gamma_8$\nmapped'),
]
for x1, y1, x2, y2, lbl in embeddings:
    ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    mx = (x1 + x2) / 2
    ax1.text(mx, y1 + 0.5, lbl, ha='center', va='bottom', fontsize=8.5,
             color='#333', style='italic')

ax1.text(5, -0.5,
         'Jede Einbettung $\\phi_k: V_{G_{r_k}} \\to V_{G_{r_{k+1}}}$ erhält alle Kanten: '
         '$(u,v) \\in E_{r_k} \\Rightarrow (\\phi_k(u), \\phi_k(v)) \\in E_{r_{k+1}}$',
         ha='center', fontsize=10, color='#555', style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff', edgecolor='#19468C', lw=1))

# ── Plot 2: Paarweise LCS-Heatmap ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
configs_list = ['$G_{1:1}$', '$G_{1:2}$', '$G_{1:4}$', '$G_{1:8}$']
# LCS matrix (max LCS over all rotations for each pair)
lcs_matrix = np.array([
    [2, 2, 2, 2],
    [0, 3, 3, 3],
    [0, 0, 5, 5],
    [0, 0, 0, 9],
], dtype=float)
subgraph_exists = lcs_matrix >= 2

im2 = ax2.imshow(lcs_matrix, cmap='YlOrRd', vmin=0, vmax=9, aspect='auto')
ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
ax2.set_xticklabels(configs_list, fontsize=9)
ax2.set_yticklabels(configs_list, fontsize=9)
ax2.set_title('LCS-Werte (max. über Rotationen)\n$G_r \\subseteq G_{r\'}$ iff LCS $\\geq 2$')

for i in range(4):
    for j in range(4):
        val = int(lcs_matrix[i, j])
        marker = '✓' if subgraph_exists[i, j] else '✗'
        color_txt = 'white' if lcs_matrix[i,j] > 5 else '#333'
        ax2.text(j, i, f'{val}\n{marker}', ha='center', va='center',
                 fontsize=11, color=color_txt, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='LCS-Länge')

# ── Plot 3: Signatur-Berechnung Schritt für Schritt (G_{1:4}) ────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Signatur-Berechnung für $G_{1:4}$ ($n=5$)\nSchritt-für-Schritt Nachweis')

n = 5
A = np.array([[0,1,1,1,1],
              [1,0,1,1,1],
              [1,1,0,1,1],
              [1,1,1,0,1],
              [1,1,1,1,0]])
node_labels = ['$c_1$', '$\\gamma_1$', '$\\gamma_2$', '$\\gamma_3$', '$\\gamma_4$']

lines = ['Adjazenzmatrix $A(G_{1:4})$:']
lines.append('')
header = '      ' + '  '.join(node_labels)
lines.append(header)
for i in range(n):
    row = '  '.join(str(A[i,j]) for j in range(n))
    lines.append(f'{node_labels[i]:8s} {row}')
lines.append('')
lines.append('Signaturberechnung: $\\sigma_j = \\sum_i A_{ij}\\cdot2^i + j\\cdot2^5$')
lines.append('')
sigmas = []
for j in range(n):
    col_val = sum(A[i,j] * (2**i) for i in range(n))
    sigma_j = col_val + j * (2**n)
    sigmas.append(sigma_j)
    lines.append(f'  $\\sigma_{j}$ = {col_val} + {j}·{2**n} = {sigma_j}')
lines.append('')
lines.append(f'$\\boldsymbol{{\\sigma}}(G_{{1:4}}) = {sigmas}$')

y_pos = 0.98
for line in lines:
    ax3.text(0.02, y_pos, line, transform=ax3.transAxes,
             fontsize=8.5, va='top', fontfamily='monospace',
             color='#19468C' if 'sigma' in line.lower() or 'Sigma' in line or
                               'Adjazenz' in line or 'Signatur' in line else '#333')
    y_pos -= 0.068
    if y_pos < 0.0:
        break

# ── Plot 4: CPU-Bedarf pro Konfiguration im RZ Bielefeld ─────────────
ax4 = fig.add_subplot(gs[1, 2])

gpu_counts = [200, 500, 1000, 2000, 5000, 10000]
for cfg, q, col in [('1:1', 1, '#19468C'), ('1:2', 2, '#2a9d8f'),
                     ('1:4', 4, '#B4321E'), ('1:8', 8, '#e76f51')]:
    cpu_needed = [g // q for g in gpu_counts]
    ax4.semilogy(gpu_counts, cpu_needed, 'o-', color=col, lw=2, ms=7, label=f'Config {cfg}')

ax4.axvline(500, color='gray', ls=':', lw=1.5, alpha=0.7, label='BIE Phase 1')
ax4.axvline(12000, color='#888', ls='--', lw=1.5, alpha=0.7, label='BIE 2030')
ax4.set_xlabel('Anzahl GPUs im RZ')
ax4.set_ylabel('Benötigte CPUs (log)')
ax4.set_title('CPU-Bedarf nach GPU-Anzahl\nund Konfiguration (T-Systems BIE)')
ax4.legend(fontsize=8, ncol=2)

plt.savefig('plot_cpugpu_d.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_d.pdf saved")

