#!/usr/bin/env python3
"""Generate all matplotlib figures for the matrix compactness paper."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg as la
import os

OUT = 'figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'text.usetex': False,
})

BLUE  = '#19468C'
RED   = '#B4321E'
GREEN = '#1E6432'
ORANGE= '#C86400'
LGRAY = '#F5F5F8'

# ── Figure 1: Relationship complexity comparison ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

ns = np.arange(1, 21)

ax = axes[0]
ax.plot(ns, ns,        color=GREEN,  lw=2.5, label=r'Abbildung $f:A\to B$  ($n$)')
ax.plot(ns, ns*(ns-1), color=ORANGE, lw=2.5, label=r'Gerichteter Graph  ($n(n-1)$)')
ax.plot(ns, ns**2,     color=RED,    lw=2.5, linestyle='--', label=r'Matrix $n\times n$  ($n^2$)')
ax.set_xlabel('Elementanzahl $n$')
ax.set_ylabel('Anzahl möglicher Relationen')
ax.set_title('Relationale Kapazität im Vergleich')
ax.legend(fontsize=9)
ax.set_xlim(1, 20)
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

ax = axes[1]
ratios_map_to_graph = ns*(ns-1) / ns
ratios_graph_to_mat = ns**2 / (ns*(ns-1)+1e-9)
ax.plot(ns, ratios_map_to_graph, color=ORANGE, lw=2.5,
        label=r'Graph / Abbildung = $n-1$')
ax.axhline(1, color=GREEN, lw=1.5, linestyle=':')
ax.set_xlabel('Elementanzahl $n$')
ax.set_ylabel('Verhältnis')
ax.set_title('Wachstumsverhältnisse')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

ax = axes[2]
densities = [0.2, 0.5, 0.8, 1.0]
colors_d  = [BLUE, GREEN, ORANGE, RED]
for d, c in zip(densities, colors_d):
    ax.plot(ns, d * ns**2, color=c, lw=2, label=f'Dichte $\\rho={d}$')
ax.set_xlabel('Elementanzahl $n$')
ax.set_ylabel('Tatsächliche Kantenanzahl')
ax.set_title('Matrixdichte vs. realisierte Relationen')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

fig.suptitle('Abbildung 1 – Relationale Kapazität: Abbildung, Graph, Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig01_relational_capacity.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig01_relational_capacity.png', bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ── Figure 2: Adjacency matrix <-> graph isomorphism visual ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n = 5
A = np.array([
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
], dtype=float)

ax = axes[0]
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1.2)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f'$v_{i+1}$' for i in range(n)])
ax.set_yticklabels([f'$v_{i+1}$' for i in range(n)])
for i in range(n):
    for j in range(n):
        ax.text(j, i, str(int(A[i,j])), ha='center', va='center',
                fontsize=14, color='white' if A[i,j] > 0.5 else 'black', fontweight='bold')
ax.set_title('Adjazenzmatrix $A \\in \\{0,1\\}^{5\\times 5}$')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(n)}

for i in range(n):
    for j in range(n):
        if A[i,j]:
            xi, yi = pos[i]
            xj, yj = pos[j]
            ax.annotate('', xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8,
                                        shrinkA=14, shrinkB=14))

for i in range(n):
    x, y = pos[i]
    ax.plot(x, y, 'o', markersize=22, color=BLUE, alpha=0.85)
    ax.text(x, y, f'$v_{i+1}$', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Induzierter gerichteter Graph $\\mathcal{G}(A)$')

# count edges
m = int(A.sum())
ax.text(0, -1.42, f'$|E| = {m}$ Kanten, $|V| = {n}$ Knoten',
        ha='center', fontsize=9, color='gray')

fig.suptitle('Abbildung 2 – Isomorphie: Adjazenzmatrix und gerichteter Graph', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig02_adjacency_graph.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig02_adjacency_graph.png', bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ── Figure 3: Information density: entries as relation slots ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ns = np.arange(1, 31)

ax = axes[0]
ax.fill_between(ns, ns**2, alpha=0.25, color=RED)
ax.plot(ns, ns**2, color=RED, lw=2.5, label='Matrix: $n^2$ Einträge')
ax.fill_between(ns, ns,    alpha=0.25, color=GREEN)
ax.plot(ns, ns, color=GREEN, lw=2.5, linestyle='--', label='Abbildung: $n$ Einträge')
ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Anzahl Einträge')
ax.set_title('Informationsdichte: Einträge als Relationsslots')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

ax = axes[1]
cmap = plt.cm.RdYlGn_r
ns2d = np.arange(2, 16)
densities_rel = (ns2d**2) / (ns2d**2)  # =1 always, reference
bar_vals = ns2d**2
bars = ax.bar(ns2d, bar_vals, color=[cmap(i/len(ns2d)) for i in range(len(ns2d))], edgecolor='white')
ax.plot(ns2d, ns2d, color=BLUE, lw=2, linestyle='--', label='Lineare Referenz $n$')
ax.set_xlabel('Matrixdimension $n$')
ax.set_ylabel('Gesamteinträge $n^2$')
ax.set_title('Quadratisches Wachstum der Matrixeinträge')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor(LGRAY)

fig.suptitle('Abbildung 3 – Quadratische Informationsdichte der Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig03_information_density.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig03_information_density.png', bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ── Figure 4: Powers of adjacency matrix = walk counting ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

A = np.array([
    [0,1,1,0,0],
    [0,0,1,1,0],
    [0,0,0,1,1],
    [1,0,0,0,1],
    [0,1,0,0,0],
], dtype=float)

for k, ax in enumerate(axes.flat):
    power = k  # A^0 ... A^5
    Ak = np.linalg.matrix_power(A.astype(int), power)
    im = ax.imshow(Ak, cmap='YlOrRd')
    ax.set_title(f'$A^{power}$' + (': Einheitsmatrix' if power==0 else f': Wege der Länge {power}'))
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'$v_{i+1}$' for i in range(5)], fontsize=8)
    ax.set_yticklabels([f'$v_{i+1}$' for i in range(5)], fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(Ak[i,j]), ha='center', va='center',
                    fontsize=9, color='black' if Ak[i,j] < Ak.max()*0.6 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Abbildung 4 – Matrixpotenzen $A^k$ zählen gerichtete Wege der Länge $k$', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig04_matrix_powers.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig04_matrix_powers.png', bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ── Figure 5: Spectrum of adjacency matrix ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Random adjacency matrices of increasing size
rng = np.random.default_rng(42)

for idx, n in enumerate([5, 10, 20]):
    ax = axes[idx]
    # Erdos-Renyi p=0.4
    p = 0.4
    A = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(A, 0)
    eigs = np.linalg.eigvals(A)
    ax.scatter(eigs.real, eigs.imag, color=BLUE, s=60, zorder=5, edgecolors='white', lw=0.5)
    theta = np.linspace(0, 2*np.pi, 300)
    r = np.sqrt(n*p*(1-p))
    ax.plot(r*np.cos(theta), r*np.sin(theta), color=RED, lw=1.5, linestyle='--',
            label=f'Kreisgesetz $r={r:.2f}$')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('Re($\\lambda$)')
    ax.set_ylabel('Im($\\lambda$)')
    ax.set_title(f'Spektrum $\\sigma(A)$, $n={n}$, $p={p}$')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(LGRAY)

fig.suptitle('Abbildung 5 – Eigenwertspektrum der Adjazenzmatrix (Erdős–Rényi)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig05_spectrum.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig05_spectrum.png', bbox_inches='tight')
plt.close()
print("Fig 5 done")

# ── Figure 6: Structural hierarchy: set -> function -> matrix ────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_facecolor(LGRAY)

boxes = [
    (1.5, 3.0, 'Menge $X$\n$n$ Elemente\n0 Relationen', BLUE,    0.3),
    (4.0, 3.0, 'Abbildung $f: A\\to B$\n$n$ Relationen\n(linear)', GREEN,   0.3),
    (6.5, 3.0, 'Gerichteter Graph\n$G=(V,E)$\nbis $n(n-1)$ Kanten', ORANGE, 0.3),
    (9.0, 3.0, 'Matrix $A\\in\\mathbb{R}^{n\\times n}$\n$n^2$ Einträge\n(maximal kompakt)', RED,  0.3),
]

prev_x = None
for x, y, label, color, alpha in boxes:
    fancy = FancyBboxPatch((x-1.1, y-0.9), 2.2, 1.8,
                           boxstyle='round,pad=0.1',
                           facecolor=color, alpha=0.18, edgecolor=color, lw=2)
    ax.add_patch(fancy)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            color=color, fontweight='bold')
    if prev_x is not None:
        ax.annotate('', xy=(x-1.15, y), xytext=(prev_x+1.15, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    prev_x = x

# Complexity labels under arrows
for xi, label in [(2.75, '+linear\nRelationen'), (5.25, '+quadratisch\nRelationen'), (7.75, 'vollständig\nkompakt')]:
    ax.text(xi, 2.3, label, ha='center', va='top', fontsize=8, color='gray', style='italic')

ax.text(5.0, 5.4, 'Strukturhierarchie der mathematischen Darstellungen',
        ha='center', va='top', fontsize=13, fontweight='bold', color='black')
ax.text(5.0, 4.8, 'Von der einfachen Menge zur maximal kompakten Matrix',
        ha='center', va='top', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(f'{OUT}/fig06_hierarchy.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig06_hierarchy.png', bbox_inches='tight')
plt.close()
print("Fig 6 done")

# ── Figure 7: Completeness analysis – K_n vs. sparse ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ns = np.arange(2, 15)

ax = axes[0]
ax.plot(ns, ns**2,       color=RED,   lw=2.5, label='Vollmatrix $n^2$')
ax.plot(ns, ns*(ns-1),   color=ORANGE,lw=2.5, label='Max. Digraph $n(n-1)$')
ax.plot(ns, ns*(ns-1)//2,color=GREEN, lw=2.5, label='Unger. vollst. $\\binom{n}{2}$')
ax.fill_between(ns, ns*(ns-1)//2, ns**2, alpha=0.12, color=RED,
                label='Differenz (Diagonale + Richtung)')
ax.set_xlabel('$n$')
ax.set_ylabel('Anzahl Relationen')
ax.set_title('Relationale Vollständigkeit')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

ax = axes[1]
# Show sparsity vs density trade-off for a fixed n=10
n = 10
densities = np.linspace(0, 1, 100)
storage   = densities * n**2
linear    = np.full_like(densities, n)
ax.fill_between(densities, linear, storage, alpha=0.2, color=BLUE)
ax.plot(densities, storage, color=BLUE,  lw=2.5, label='Matrix (Speicher $\\rho n^2$)')
ax.axhline(n, color=GREEN, lw=2, linestyle='--', label=f'Abbildung: konstant $n={n}$')
ax.axvline(1.0, color=RED, lw=1.5, linestyle=':', label='$\\rho=1$ (vollständig)')
ax.set_xlabel('Dichte $\\rho$')
ax.set_ylabel('Speicherbedarf (Einträge)')
ax.set_title(f'Dichte-Speicher-Relation, $n={n}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

fig.suptitle('Abbildung 7 – Vollständigkeit und Sparsität von Matrizenstrukturen', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig07_completeness.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig07_completeness.png', bbox_inches='tight')
plt.close()
print("Fig 7 done")

# ── Figure 8: Subgraph algorithm and SCC illustration ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# A graph with 2 SCCs
# SCC1: {0,1,2}, SCC2: {3,4}
A_scc = np.array([
    [0,1,0,0,0],
    [0,0,1,1,0],
    [1,0,0,0,0],
    [0,0,0,0,1],
    [0,0,0,1,0],
], dtype=float)

scc_colors = [BLUE, RED, BLUE, GREEN, GREEN]
scc_labels = ['SCC 1','SCC 1','SCC 1','SCC 2','SCC 2']

ax = axes[0]
im = ax.imshow(A_scc, cmap='Blues')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f'$v_{i+1}$' for i in range(5)])
ax.set_yticklabels([f'$v_{i+1}$' for i in range(5)])
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(int(A_scc[i,j])), ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if A_scc[i,j]>0.5 else 'black')
# Highlight SCC blocks
import matplotlib.patches as patches
rect1 = patches.Rectangle((-0.5,-0.5), 3, 3, linewidth=3, edgecolor=BLUE,  facecolor='none')
rect2 = patches.Rectangle(( 2.5, 2.5), 2, 2, linewidth=3, edgecolor=GREEN, facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.set_title('Adjazenzmatrix mit 2 SCCs')
ax.text(1.0, -0.85, 'SCC 1', fontsize=9, color=BLUE,  ha='center')
ax.text(3.5, -0.85, 'SCC 2', fontsize=9, color=GREEN, ha='center')

ax = axes[1]
theta = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/2
pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(5)}

for i in range(5):
    for j in range(5):
        if A_scc[i,j]:
            xi, yi = pos[i]
            xj, yj = pos[j]
            col = BLUE if (i<3 and j<3) else (GREEN if (i>=3 and j>=3) else ORANGE)
            ax.annotate('', xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2,
                                        shrinkA=14, shrinkB=14))

scc_node_colors = [BLUE, BLUE, BLUE, GREEN, GREEN]
for i in range(5):
    x, y = pos[i]
    ax.plot(x, y, 'o', markersize=24, color=scc_node_colors[i], alpha=0.85)
    ax.text(x, y, f'$v_{i+1}$', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')

ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Graph $\\mathcal{G}(A)$ mit SCCs (farbkodiert)')

blue_patch  = mpatches.Patch(color=BLUE,  label='SCC 1: $\\{v_1,v_2,v_3\\}$')
green_patch = mpatches.Patch(color=GREEN, label='SCC 2: $\\{v_4,v_5\\}$')
ax.legend(handles=[blue_patch, green_patch], loc='lower center', fontsize=9)

fig.suptitle('Abbildung 8 – SCC-Zerlegung: Matrix und Graph', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig08_scc.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig08_scc.png', bbox_inches='tight')
plt.close()
print("Fig 8 done")

# ── Figure 9: Spectral radius and graph connectivity ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rng2 = np.random.default_rng(7)
ns_spec = range(3, 25)
rho_list = []
lambda1_list = []
for n in ns_spec:
    A = (rng2.random((n,n)) < 0.4).astype(float)
    np.fill_diagonal(A, 0)
    ev = np.abs(np.linalg.eigvals(A))
    rho_list.append(np.max(ev))
    lambda1_list.append(np.max(ev))

ax = axes[0]
ax.plot(list(ns_spec), rho_list, 'o-', color=BLUE, lw=2, markersize=5)
ax.plot(list(ns_spec), [0.4*(n-1) for n in ns_spec], color=RED, lw=1.5,
        linestyle='--', label='Theoret. Schranke $p(n-1)$')
ax.set_xlabel('Knotenanzahl $n$')
ax.set_ylabel('Spektralradius $\\rho(A)$')
ax.set_title('Spektralradius von Zufallsadjazenzmatrizen')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

ax = axes[1]
# Frobenius norm vs. spectral radius
frobenius = [np.sqrt(np.sum((rng2.random((n,n))<0.4).astype(float)**2)) for n in ns_spec]
ax.scatter(list(ns_spec), rho_list, color=BLUE,  s=50, label='Spektralradius $\\rho(A)$')
ax.scatter(list(ns_spec), [f/(n**0.5) for f,n in zip(frobenius,ns_spec)],
           color=RED, s=50, marker='^', label='Frobeniusnorm / $\\sqrt{n}$')
ax.set_xlabel('Knotenanzahl $n$')
ax.set_ylabel('Normwert')
ax.set_title('Spektralradius vs. Frobeniusnorm')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor(LGRAY)

fig.suptitle('Abbildung 9 – Spektralradius und Graphkonnektivität', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig09_spectral_radius.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig09_spectral_radius.png', bbox_inches='tight')
plt.close()
print("Fig 9 done")

# ── Figure 10: Matrix as universal language across disciplines ────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_facecolor('#FAFAFA')

# Central matrix box
cx, cy = 5.0, 5.0
fancy_center = FancyBboxPatch((cx-1.2, cy-0.7), 2.4, 1.4,
                              boxstyle='round,pad=0.15',
                              facecolor=RED, alpha=0.25, edgecolor=RED, lw=3)
ax.add_patch(fancy_center)
ax.text(cx, cy, 'Matrix\n$A \\in \\mathbb{R}^{n \\times n}$',
        ha='center', va='center', fontsize=12, fontweight='bold', color=RED)

# Surrounding domains
domains = [
    (5.0, 8.8, 'Systemtheorie\n$\\dot{x}=Ax$',           BLUE),
    (8.5, 7.0, 'Graphentheorie\n$G=(V,E)$',              GREEN),
    (8.5, 3.0, 'Numerik\n$Ax=b$',                        ORANGE),
    (5.0, 1.2, 'Statistik\n$\\Sigma$ (Kovarianz)',        '#8B4C8C'),
    (1.5, 3.0, 'Quantenmechanik\n$H|\\psi\\rangle$',     '#2C7BB6'),
    (1.5, 7.0, 'Netzwerktheorie\n$L_{\\rm Laplace}$',    '#D7191C'),
]

for dx, dy, label, color in domains:
    fancy = FancyBboxPatch((dx-1.1, dy-0.55), 2.2, 1.1,
                           boxstyle='round,pad=0.1',
                           facecolor=color, alpha=0.15, edgecolor=color, lw=2)
    ax.add_patch(fancy)
    ax.text(dx, dy, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=color)
    ax.annotate('', xy=(cx + 0.9*(dx-cx)/np.hypot(dx-cx,dy-cy),
                        cy + 0.7*(dy-cy)/np.hypot(dx-cx,dy-cy)),
                xytext=(dx - 1.1*(dx-cx)/np.hypot(dx-cx,dy-cy),
                        dy - 0.6*(dy-cy)/np.hypot(dx-cx,dy-cy)),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=1.5, alpha=0.7))

ax.text(5.0, 9.7, 'Abbildung 10 – Die Matrix als universelle Sprache der Wissenschaft',
        ha='center', va='top', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/fig10_universal.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig10_universal.png', bbox_inches='tight')
plt.close()
print("Fig 10 done")

print("\nAll figures generated successfully!")
