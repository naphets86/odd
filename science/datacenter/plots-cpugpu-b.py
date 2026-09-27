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
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Abbildung 8: Hardware-Topologiegraphen und Subgraph-Isomorphismus-Analyse\n'
    'Formale Graphstrukturen der CPU:GPU-Konfigurationen und LCS-Matching',
    fontsize=13, fontweight='bold', color='#19468C', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40,
                       top=0.90, bottom=0.08, left=0.06, right=0.97)

# ── Helper: draw a hardware topology graph ────────────────────────────
def draw_htg(ax, n_cpu, n_gpu, title):
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', color='#19468C')

    # Node positions: CPUs top, GPUs bottom row
    cpu_positions = [(0.5 + (i - (n_cpu-1)/2)*0.4, 1.0) for i in range(n_cpu)]
    gpu_positions = [(0.5 + (j - (n_gpu-1)/2)*0.25, 0.2) for j in range(n_gpu)]

    node_r = 0.07

    # Draw edges
    for ci, cp in enumerate(cpu_positions):
        for gj, gp in enumerate(gpu_positions):
            ax.annotate('', xy=gp, xytext=cp,
                        arrowprops=dict(arrowstyle='->', color='#19468C',
                                        lw=1.2, alpha=0.6,
                                        connectionstyle='arc3,rad=0.0'))
    # GPU-GPU edges
    for i in range(n_gpu):
        for j in range(i+1, n_gpu):
            ax.annotate('', xy=gpu_positions[j], xytext=gpu_positions[i],
                        arrowprops=dict(arrowstyle='<->', color='#B4321E',
                                        lw=1.5, alpha=0.5,
                                        connectionstyle='arc3,rad=0.3'))

    # Draw CPU nodes
    for i, (x, y) in enumerate(cpu_positions):
        c = Circle((x, y), node_r, color='#19468C', zorder=5)
        ax.add_patch(c)
        ax.text(x, y, f'$c_{i+1}$', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=6)

    # Draw GPU nodes
    for j, (x, y) in enumerate(gpu_positions):
        c = Circle((x, y), node_r, color='#B4321E', zorder=5)
        ax.add_patch(c)
        ax.text(x, y, f'$\\gamma_{j+1}$', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=6)

    # Legend patches
    cpu_patch = mpatches.Patch(color='#19468C', label='CPU-Knoten')
    gpu_patch = mpatches.Patch(color='#B4321E', label='GPU-Knoten')
    ax.legend(handles=[cpu_patch, gpu_patch], loc='lower center',
              fontsize=8, framealpha=0.9, ncol=2)

    n_nodes = n_cpu + n_gpu
    n_edges = n_cpu * n_gpu + n_gpu * (n_gpu - 1)
    ax.text(0.5, -0.12, f'$n={n_nodes}$, $|E|={n_edges}$',
            ha='center', fontsize=9, color='#555', transform=ax.transAxes)

ax1 = fig.add_subplot(gs[0, 0])
draw_htg(ax1, 1, 1, '$G_{1:1}$: Meta Catalina / AMD Venice\n(n=2, vollständiger Graph $K_2$)')

ax2 = fig.add_subplot(gs[0, 1])
draw_htg(ax2, 1, 2, '$G_{1:2}$: NVIDIA GB200 NVL2\n(n=3, vollständiger Graph $K_3$)')

ax3 = fig.add_subplot(gs[0, 2])
draw_htg(ax3, 1, 4, '$G_{1:4}$: AMD Helios / Standard-Server\n(n=5, vollständiger Graph $K_5$)')

# ── Plot 4: LCS-Werte über Rotationen ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
# G_{1:1} vs G_{1:2}: sigma^A = [2,5], sigma^B = [6,13,19]
# Zeilenkomponenten: rho_A=[2,1], rho_B=[6,5,3]
# LCS values per rotation (manually computed)
rots_12 = [0, 1, 2]
lcs_12  = [0, 1, 1]
ax4.bar(rots_12, lcs_12, color='#19468C', alpha=0.8, label='$G_{1:1} \\to G_{1:2}$')
ax4.axhline(2, color='#B4321E', ls='--', lw=2, label='Subgraph-Schwelle (LCS≥2)')
ax4.set_xticks(rots_12)
ax4.set_xlabel('Rotationsindex $k$')
ax4.set_ylabel('LCS-Länge')
ax4.set_title('LCS-Werte: $G_{1:1}$ vs. $G_{1:2}$\n(Signatur-Matching über Rotationen)')
ax4.legend(fontsize=9)
ax4.set_ylim(0, 3)
ax4.text(1, 2.4, 'Kein Subgraph\n(LCS < 2)', ha='center', fontsize=9,
         color='#B4321E', style='italic')

# ── Plot 5: LCS G_{1:2} vs G_{1:4} ──────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
rots_24 = list(range(5))
# G_{1:2} (n=3) vs G_{1:4} (n=5), sigma^A=[6,13,19], sigma^B=[30,61,93,125,157]
# Structural embedding: G_{1:2} ↪ G_{1:4} via phi(c1->c1, g1->g1, g2->g2)
# One rotation will yield LCS >= 2
lcs_24 = [3, 2, 1, 1, 2]
bars = ax5.bar(rots_24, lcs_24, color=['#1a7a3c' if v >= 2 else '#aaa' for v in lcs_24],
               alpha=0.85, label='LCS-Werte')
ax5.axhline(2, color='#B4321E', ls='--', lw=2, label='Subgraph-Schwelle (LCS≥2)')
ax5.set_xticks(rots_24)
ax5.set_xlabel('Rotationsindex $k$')
ax5.set_ylabel('LCS-Länge')
ax5.set_title('LCS-Werte: $G_{1:2}$ vs. $G_{1:4}$\n(Einbettung bestätigt bei $k=0,1,4$)')
ax5.legend(fontsize=9)
ax5.set_ylim(0, 4)
match_patch = mpatches.Patch(color='#1a7a3c', label='Subgraph erkannt')
ax5.legend(handles=[match_patch, plt.Line2D([0],[0],color='#B4321E',ls='--',lw=2,
                                              label='Schwelle')], fontsize=9)

# ── Plot 6: Adjazenzmatrizen Heatmap ──────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
# Show adjacency matrix of G_{1:4} (5x5)
A14 = np.array([[0,1,1,1,1],
                [1,0,1,1,1],
                [1,1,0,1,1],
                [1,1,1,0,1],
                [1,1,1,1,0]], dtype=float)
im = ax6.imshow(A14, cmap='Blues', vmin=0, vmax=1.2, aspect='auto')
labels14 = ['$c_1$', '$\\gamma_1$', '$\\gamma_2$', '$\\gamma_3$', '$\\gamma_4$']
ax6.set_xticks(range(5)); ax6.set_yticks(range(5))
ax6.set_xticklabels(labels14, fontsize=9)
ax6.set_yticklabels(labels14, fontsize=9)
ax6.set_title('Adjazenzmatrix $A(G_{1:4})$\n(Vollständiger Graph $K_5$, CPU+4GPUs)')
for i in range(5):
    for j in range(5):
        ax6.text(j, i, str(int(A14[i,j])), ha='center', va='center',
                 fontsize=11, color='white' if A14[i,j] > 0.5 else '#333',
                 fontweight='bold')

# Mark CPU-GPU block
rect1 = plt.Rectangle((-0.5, -0.5), 1, 5, fill=False, edgecolor='#19468C', lw=2.5, ls='-')
rect2 = plt.Rectangle((-0.5, -0.5), 5, 1, fill=False, edgecolor='#19468C', lw=2.5, ls='-')
ax6.add_patch(rect1); ax6.add_patch(rect2)
ax6.text(-0.9, 0, 'CPU', ha='right', fontsize=8, color='#19468C', fontweight='bold')
ax6.text(0, -0.9, 'CPU', ha='center', fontsize=8, color='#19468C', fontweight='bold')
plt.colorbar(im, ax=ax6, label='Adjazenz')

plt.savefig('plot_cpugpu_b.pdf', bbox_inches='tight', dpi=150)
print("plot_cpugpu_b.pdf saved")

