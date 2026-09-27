"""
plots_hierarchical.py
Erzeugt Plots 11-20 für das neue Kapitel:
"Hierarchische Subgraph-DFS-Analyse von Sternenclustern im Weltall"
in stars.tex.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Arc
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'text.usetex': False,
})

OUTDIR = ''  # PDFs are saved in the working directory (same location as stars.tex)

# Minimum distance threshold (m) to prevent gravitational singularities in
# visualizations. ~0.00033 pc -- smaller than any stellar diameter in our model.
MIN_DISTANCE_THRESHOLD = 1e13

np.random.seed(2026)

# ── Globale Hilfsfunktionen (Subgraph Algorithmus) ───────────────────────────

def compute_signature(matrix):
    n = matrix.shape[0]
    sigs = []
    for col in range(n):
        col_vec = matrix[:, col]
        row_sig = sum(2**i for i in range(n) if col_vec[i] == 1)
        sigs.append(row_sig + col * (2**n))
    return sigs


def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def subgraph_check(A, B):
    """Prüft ob A ein (zyklischer) Subgraph von B ist."""
    nA, nB = A.shape[0], B.shape[0]
    sigA = compute_signature(A)
    sigB = compute_signature(B)
    wA, wB = 2**nA, 2**nB
    rA = [s % wA for s in sigA]
    rB = [s % wB for s in sigB]
    for rot in range(nB):
        rotB = rB[rot:] + rB[:rot]
        if nA == nB:
            if lcs_length(rA, rotB) >= 2:
                return True, rot
        else:
            for start in range(nB - nA + 1):
                win = rotB[start:start+nA]
                if lcs_length(rA, win) >= 2:
                    return True, rot
    return False, -1


def gravity_matrix(positions, masses):
    """Gravitationsstärke-Matrix (normiert auf einheitenlose Werte)."""
    G = 6.674e-11
    n = len(positions)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                f = G * masses[i] * masses[j] / r**2
                W[i, j] = W[j, i] = f
    return W


def adj_from_gravity(W, threshold_frac=0.3):
    """Adjazenzmatrix: Kante wenn Gravitation > threshold * max."""
    if W.max() == 0:
        return np.zeros_like(W, dtype=int)
    thr = threshold_frac * W.max()
    A = (W > thr).astype(int)
    np.fill_diagonal(A, 0)
    return A


def make_cluster(center, n, spread, mass_range=(0.5, 2.0)):
    pos = center + np.random.randn(n, 2) * spread
    masses = np.random.uniform(*mass_range, n) * 2e30
    return pos, masses


def cluster_binding_energy(positions, masses):
    G = 6.674e-11
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                E -= G * masses[i] * masses[j] / r
    return E


def lcs_matrix_between_clusters(adjs):
    """LCS-Matrix zwischen allen Clusteradjazenzmatrizen."""
    N = len(adjs)
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            nA, nB = adjs[i].shape[0], adjs[j].shape[0]
            wA, wB = 2**nA, 2**nB
            sA = compute_signature(adjs[i])
            sB = compute_signature(adjs[j])
            rA = [s % wA for s in sA]
            rB = [s % wB for s in sB]
            L[i, j] = lcs_length(rA, rB)
    return L


# ── Cluster-Daten auf drei Hierarchieebenen ──────────────────────────────────

# Ebene 0 (Einzelsterne – kleinste Skala, 3-6 Sterne pro Micro-Cluster)
micro_clusters_pos = []
micro_clusters_m = []
SCALE = 1e15  # 1e15 m ≈ 0.033 pc

# 9 Micro-Cluster
centers_micro = [
    np.array([0.0,  0.0]),
    np.array([4.0,  1.0]),
    np.array([8.0,  0.5]),
    np.array([1.0, -4.0]),
    np.array([5.0, -4.5]),
    np.array([9.0, -4.0]),
    np.array([1.5,  4.5]),
    np.array([5.5,  5.0]),
    np.array([9.0,  4.5]),
]
sizes_micro = [5, 4, 5, 4, 5, 4, 4, 5, 4]

for i, (c, s) in enumerate(zip(centers_micro, sizes_micro)):
    p, m = make_cluster(c * SCALE, s, 0.5 * SCALE, (0.5, 1.5))
    micro_clusters_pos.append(p)
    micro_clusters_m.append(m)

# Ebene 1 (Meso-Cluster: Gruppen von je 3 Micro-Clustern)
meso_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Ebene 2 (Makro-Supercluster: alle Meso-Cluster)
# (enthält alle 9 Micro-Cluster)

# Adjazenzmatrizen für alle Micro-Cluster
micro_adjs = []
for p, m in zip(micro_clusters_pos, micro_clusters_m):
    W = gravity_matrix(p, m)
    A = adj_from_gravity(W, 0.2)
    micro_adjs.append(A)

# Adjazenzmatrizen für Meso-Cluster (kombinierte Adjazenzmatrix)
meso_adjs = []
for group in meso_groups:
    all_p = np.vstack([micro_clusters_pos[i] for i in group])
    all_m = np.concatenate([micro_clusters_m[i] for i in group])
    W = gravity_matrix(all_p, all_m)
    A = adj_from_gravity(W, 0.25)
    meso_adjs.append(A)

# Adjazenzmatrix für Supercluster (alle 9 Micro-Cluster)
super_p = np.vstack(micro_clusters_pos)
super_m = np.concatenate(micro_clusters_m)
W_super = gravity_matrix(super_p, super_m)
A_super = adj_from_gravity(W_super, 0.3)

# Meta-Graph: Knoten = Meso-Cluster (3 Stück), Kanten = Subgraph-Beziehungen
N_meta = 3
meta_edges = {}
meta_lcs = np.zeros((N_meta, N_meta))
for i in range(N_meta):
    for j in range(N_meta):
        if i != j:
            ok, rot = subgraph_check(meso_adjs[i], meso_adjs[j])
            nA = meso_adjs[i].shape[0]
            nB = meso_adjs[j].shape[0]
            wA, wB = 2**nA, 2**nB
            rA = [s % wA for s in compute_signature(meso_adjs[i])]
            rB = [s % wB for s in compute_signature(meso_adjs[j])]
            meta_lcs[i, j] = lcs_length(rA, rB)
            if ok:
                meta_edges[(i, j)] = rot

# ══════════════════════════════════════════════════════════════════════════════
# Plot 11: Dreistufige Hierarchie der Sterncluster
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Dreistufige Hierarchie der Sterncluster: Mikro → Meso → Supercluster',
             fontsize=14, fontweight='bold')

colors_micro = ['#1f77b4', '#ff7f0e', '#2ca02c',
                '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22']
markers_micro = ['o', 's', '^', 'D', 'P', 'X', 'h', '*', 'v']

# Ebene 0: Micro-Cluster (alle 9)
ax = axes[0]
for idx, (p, m) in enumerate(zip(micro_clusters_pos, micro_clusters_m)):
    px = p[:, 0] / SCALE
    py = p[:, 1] / SCALE
    sizes_plot = m / m.max() * 120 + 20
    ax.scatter(px, py, s=sizes_plot, c=colors_micro[idx],
               marker=markers_micro[idx], label=f'$\\mathcal{{C}}^{{(0)}}_{{{idx+1}}}$',
               edgecolors='k', linewidths=0.5, zorder=5)
    # Gravitationskanten innerhalb des Micro-Clusters
    W = gravity_matrix(p, m)
    Wmax = W.max() if W.max() > 0 else 1
    for ii in range(len(p)):
        for jj in range(ii+1, len(p)):
            if W[ii, jj] > 0.15 * Wmax:
                ax.plot([px[ii], px[jj]], [py[ii], py[jj]],
                        '-', color=colors_micro[idx], alpha=0.3,
                        linewidth=0.8, zorder=2)
ax.set_title('Ebene 0: Micro-Cluster\n(Individuelle Sterne)', fontsize=11)
ax.set_xlabel('x [Einh. $10^{15}$ m]')
ax.set_ylabel('y [Einh. $10^{15}$ m]')
ax.legend(fontsize=6, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Ebene 1: Meso-Cluster (3 Gruppen)
ax = axes[1]
meso_colors = ['#1f77b4', '#d62728', '#2ca02c']
for gi, group in enumerate(meso_groups):
    for idx in group:
        p = micro_clusters_pos[idx]
        m = micro_clusters_m[idx]
        px = p[:, 0] / SCALE
        py = p[:, 1] / SCALE
        sizes_plot = m / m.max() * 80 + 15
        ax.scatter(px, py, s=sizes_plot, c=meso_colors[gi],
                   alpha=0.7, edgecolors='k', linewidths=0.5, zorder=5)
    # Hüllkreis um Meso-Cluster
    all_p = np.vstack([micro_clusters_pos[i] for i in group]) / SCALE
    cx, cy = all_p.mean(axis=0)
    r = np.max(np.linalg.norm(all_p - [cx, cy], axis=1)) + 0.8
    circle = plt.Circle((cx, cy), r, fill=False, linestyle='--',
                         color=meso_colors[gi], linewidth=2, alpha=0.8)
    ax.add_patch(circle)
    ax.text(cx, cy - r - 0.5, f'$\\mathcal{{M}}_{{{gi+1}}}$',
            ha='center', va='top', fontsize=11, color=meso_colors[gi],
            fontweight='bold')
ax.set_title('Ebene 1: Meso-Cluster\n(Gruppen von Micro-Clustern)', fontsize=11)
ax.set_xlabel('x [Einh. $10^{15}$ m]')
ax.set_ylabel('y [Einh. $10^{15}$ m]')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Ebene 2: Supercluster (alle zusammen)
ax = axes[2]
for gi, group in enumerate(meso_groups):
    for idx in group:
        p = micro_clusters_pos[idx]
        m = micro_clusters_m[idx]
        px = p[:, 0] / SCALE
        py = p[:, 1] / SCALE
        sizes_plot = m / m.max() * 60 + 10
        ax.scatter(px, py, s=sizes_plot, c=meso_colors[gi],
                   alpha=0.6, edgecolors='k', linewidths=0.3, zorder=5)
# Verbindungslinien zwischen Meso-Cluster-Zentren
meso_centers = []
for gi, group in enumerate(meso_groups):
    all_p = np.vstack([micro_clusters_pos[i] for i in group]) / SCALE
    meso_centers.append(all_p.mean(axis=0))
for gi in range(3):
    for gj in range(gi+1, 3):
        ax.plot([meso_centers[gi][0], meso_centers[gj][0]],
                [meso_centers[gi][1], meso_centers[gj][1]],
                'k-', linewidth=1.5, alpha=0.4, zorder=3)
for gi, (cx, cy) in enumerate(meso_centers):
    ax.scatter([cx], [cy], s=300, c=meso_colors[gi], marker='*',
               edgecolors='k', linewidths=1, zorder=10)
    ax.text(cx + 0.3, cy + 0.3, f'Zentrum $\\mathcal{{M}}_{{{gi+1}}}$',
            fontsize=8, color=meso_colors[gi])
# Gesamthülle
all_p_super = np.vstack(micro_clusters_pos) / SCALE
cx_s, cy_s = all_p_super.mean(axis=0)
r_s = np.max(np.linalg.norm(all_p_super - [cx_s, cy_s], axis=1)) + 1.2
circle_s = plt.Circle((cx_s, cy_s), r_s, fill=False, linestyle='-',
                       color='purple', linewidth=2.5, alpha=0.6)
ax.add_patch(circle_s)
ax.text(cx_s, cy_s - r_s - 0.7, '$\\mathcal{S}$ (Supercluster)',
        ha='center', fontsize=11, color='purple', fontweight='bold')
ax.set_title('Ebene 2: Supercluster\n(Gesamthierarchie)', fontsize=11)
ax.set_xlabel('x [Einh. $10^{15}$ m]')
ax.set_ylabel('y [Einh. $10^{15}$ m]')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot11_hierarchie_ebenen.pdf', bbox_inches='tight')
plt.close()
print("Plot 11 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 12: Meta-Graph Konstruktion
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Meta-Graph Konstruktion: Meso-Cluster als Knoten, Subgraph-Kanten',
             fontsize=13, fontweight='bold')

# Linkes Panel: LCS-Matrix
ax = axes[0]
lcs_mat = lcs_matrix_between_clusters(meso_adjs)
im = ax.imshow(lcs_mat, cmap='Blues', aspect='auto', vmin=0)
ax.set_xticks(range(N_meta))
ax.set_yticks(range(N_meta))
labels_m = [f'$\\mathcal{{M}}_{i+1}$' for i in range(N_meta)]
ax.set_xticklabels(labels_m, fontsize=12)
ax.set_yticklabels(labels_m, fontsize=12)
ax.set_title('LCS-Matrix zwischen\nMeso-Clustern', fontsize=12)
plt.colorbar(im, ax=ax, label='LCS-Wert')
for i in range(N_meta):
    for j in range(N_meta):
        ax.text(j, i, f'{lcs_mat[i,j]:.0f}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if lcs_mat[i,j] > lcs_mat.max()*0.6 else 'black')

# Rechtes Panel: Meta-Graph
ax = axes[1]
# Knoten des Meta-Graphen
node_pos_meta = np.array([
    [0.5, 0.85],  # M1
    [0.15, 0.2],  # M2
    [0.85, 0.2],  # M3
])
node_colors_meta = ['#1f77b4', '#d62728', '#2ca02c']
node_labels_meta = ['$\\mathcal{M}_1$', '$\\mathcal{M}_2$', '$\\mathcal{M}_3$']
node_sizes_meta = [sizes_micro[i]*200 for i in range(3)]

for idx, (nc, nl, ns) in enumerate(zip(node_colors_meta, node_labels_meta, node_sizes_meta)):
    circle = plt.Circle(node_pos_meta[idx], 0.12, color=nc, alpha=0.85, zorder=5)
    ax.add_patch(circle)
    ax.text(node_pos_meta[idx][0], node_pos_meta[idx][1], nl,
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', zorder=6)
    n_stars = sum(sizes_micro[g] for g in meso_groups[idx])
    ax.text(node_pos_meta[idx][0], node_pos_meta[idx][1] - 0.17,
            f'$n={n_stars}$ Sterne', ha='center', fontsize=9, color=nc)

# Kanten des Meta-Graphen
edge_found = False
for (i, j), rot in meta_edges.items():
    xi, yi = node_pos_meta[i]
    xj, yj = node_pos_meta[j]
    dx, dy = xj - xi, yj - yi
    ax.annotate('', xy=(xj - 0.12 * dx/np.hypot(dx,dy),
                         yj - 0.12 * dy/np.hypot(dx,dy)),
                xytext=(xi + 0.12 * dx/np.hypot(dx,dy),
                         yi + 0.12 * dy/np.hypot(dx,dy)),
                arrowprops=dict(arrowstyle='->', color='#e67e22',
                                lw=2.5, mutation_scale=18))
    mx, my = (xi + xj)/2, (yi + yj)/2
    ax.text(mx, my + 0.05, f'rot={rot}', fontsize=8, color='#e67e22',
            ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   alpha=0.8, edgecolor='#e67e22'))
    edge_found = True

if not edge_found:
    # Zeige auch Nicht-Kanten mit LCS-Werten
    for i in range(N_meta):
        for j in range(i+1, N_meta):
            xi, yi = node_pos_meta[i]
            xj, yj = node_pos_meta[j]
            dx, dy = xj - xi, yj - yi
            d = np.hypot(dx, dy)
            ax.plot([xi + 0.12*dx/d, xj - 0.12*dx/d],
                    [yi + 0.12*dy/d, yj - 0.12*dy/d],
                    '--', color='gray', alpha=0.4, linewidth=1.5)
            mx, my = (xi + xj)/2, (yi + yj)/2
            ax.text(mx, my + 0.04, f'LCS={lcs_mat[i,j]:.0f}', fontsize=8,
                    color='gray', ha='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title('Meta-Graph $\\mathcal{M}$:\nMeso-Cluster als Knoten', fontsize=12)
ax.axis('off')

# Legende
legend_elems = [
    mpatches.Patch(color='#1f77b4', label='$\\mathcal{M}_1$: Cluster 1-3'),
    mpatches.Patch(color='#d62728', label='$\\mathcal{M}_2$: Cluster 4-6'),
    mpatches.Patch(color='#2ca02c', label='$\\mathcal{M}_3$: Cluster 7-9'),
    mpatches.Patch(color='#e67e22', label='Subgraph-Kante (LCS≥2)'),
]
ax.legend(handles=legend_elems, loc='lower center', fontsize=9,
          bbox_to_anchor=(0.5, -0.08), ncol=2)

plt.tight_layout()
plt.savefig(OUTDIR + 'plot12_metagraph.pdf', bbox_inches='tight')
plt.close()
print("Plot 12 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 13: HS-DFS-Wald auf dem Meta-Graphen
# ══════════════════════════════════════════════════════════════════════════════

# Simuliere HS-DFS auf 5 Clustern für eine detailliertere Demonstration
np.random.seed(42)
n_demo_clusters = 5
demo_clusters = []
demo_names = ['$G_1$', '$G_2$', '$G_3$', '$G_4$', '$G_5$']
demo_sizes = [6, 5, 6, 4, 5]
demo_centers = [np.array([0.0, 0.0]), np.array([5e15, 2e15]),
                np.array([10e15, 0.0]), np.array([2e15, -5e15]),
                np.array([7e15, -5e15])]

for ctr, sz in zip(demo_centers, demo_sizes):
    p, m = make_cluster(ctr, sz, 1.2e15)
    W = gravity_matrix(p, m)
    A = adj_from_gravity(W, 0.2)
    demo_clusters.append(A)

# Baue Meta-Graph (Subgraph-Relation)
demo_meta = {i: [] for i in range(n_demo_clusters)}
demo_lcs = np.zeros((n_demo_clusters, n_demo_clusters))
for i in range(n_demo_clusters):
    for j in range(n_demo_clusters):
        if i != j:
            nA, nB = demo_clusters[i].shape[0], demo_clusters[j].shape[0]
            wA, wB = 2**nA, 2**nB
            rA = [s % wA for s in compute_signature(demo_clusters[i])]
            rB = [s % wB for s in compute_signature(demo_clusters[j])]
            demo_lcs[i, j] = lcs_length(rA, rB)
            ok, _ = subgraph_check(demo_clusters[i], demo_clusters[j])
            if ok:
                demo_meta[i].append(j)

# DFS auf Meta-Graph
visited = [False] * n_demo_clusters
discovery = [0] * n_demo_clusters
finish = [0] * n_demo_clusters
parent = [-1] * n_demo_clusters
timer = [0]
tree_edges = []
back_edges = []
forward_edges = []
cross_edges = []
color = ['white'] * n_demo_clusters  # white=unvisited, gray=active, black=done

def dfs_visit(u):
    color[u] = 'gray'
    timer[0] += 1
    discovery[u] = timer[0]
    for v in demo_meta[u]:
        if color[v] == 'white':
            parent[v] = u
            tree_edges.append((u, v))
            dfs_visit(v)
        elif color[v] == 'gray':
            back_edges.append((u, v))
        elif discovery[u] < discovery[v]:
            forward_edges.append((u, v))
        else:
            cross_edges.append((u, v))
    color[u] = 'black'
    timer[0] += 1
    finish[u] = timer[0]

for u in range(n_demo_clusters):
    if color[u] == 'white':
        dfs_visit(u)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('HS-DFS-Wald auf dem Meta-Graphen der Sterncluster',
             fontsize=13, fontweight='bold')

# Linkes Panel: Vollständiger Meta-Graph
ax = axes[0]
ax.set_title('Meta-Graph mit allen Subgraph-Kanten', fontsize=12)
node_xy = np.array([
    [0.5, 0.85],
    [0.15, 0.55],
    [0.85, 0.55],
    [0.25, 0.15],
    [0.75, 0.15],
])
demo_colors_node = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
for idx in range(n_demo_clusters):
    circ = plt.Circle(node_xy[idx], 0.09, color=demo_colors_node[idx], zorder=5, alpha=0.9)
    ax.add_patch(circ)
    ax.text(node_xy[idx][0], node_xy[idx][1], demo_names[idx],
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', zorder=6)
    ax.text(node_xy[idx][0], node_xy[idx][1] - 0.13,
            f'd={discovery[idx]}, f={finish[idx]}',
            ha='center', fontsize=8, color='#555')

for i in range(n_demo_clusters):
    for j in demo_meta[i]:
        xi, yi = node_xy[i]
        xj, yj = node_xy[j]
        dx, dy = xj - xi, yj - yi
        d = np.hypot(dx, dy)
        col = '#e67e22'
        if (i, j) in tree_edges:
            col = '#27ae60'
        elif (i, j) in back_edges:
            col = '#c0392b'
        elif (i, j) in forward_edges:
            col = '#8e44ad'
        elif (i, j) in cross_edges:
            col = '#16a085'
        ax.annotate('', xy=(xj - 0.09*dx/d, yj - 0.09*dy/d),
                    xytext=(xi + 0.09*dx/d, yi + 0.09*dy/d),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.0,
                                   mutation_scale=16))

# Legende
leg_elems = [
    mpatches.Patch(color='#27ae60', label='Baum-Kante'),
    mpatches.Patch(color='#c0392b', label='Rückwärts-Kante'),
    mpatches.Patch(color='#8e44ad', label='Vorwärts-Kante'),
    mpatches.Patch(color='#16a085', label='Quer-Kante'),
]
ax.legend(handles=leg_elems, loc='upper left', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Rechtes Panel: DFS-Wald (nur Baumkanten)
ax = axes[1]
ax.set_title('DFS-Wald (Tiefensuche-Bäume)', fontsize=12)
for idx in range(n_demo_clusters):
    circ = plt.Circle(node_xy[idx], 0.09, color=demo_colors_node[idx], zorder=5, alpha=0.9)
    ax.add_patch(circ)
    ax.text(node_xy[idx][0], node_xy[idx][1], demo_names[idx],
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', zorder=6)
    ax.text(node_xy[idx][0], node_xy[idx][1] - 0.13,
            f'd={discovery[idx]}, f={finish[idx]}',
            ha='center', fontsize=8, color='#555')
    # Wurzel-Markierung
    if parent[idx] == -1:
        ax.text(node_xy[idx][0], node_xy[idx][1] + 0.13,
                'Wurzel', ha='center', fontsize=8, color='darkgreen',
                fontweight='bold')

for (u, v) in tree_edges:
    xi, yi = node_xy[u]
    xj, yj = node_xy[v]
    dx, dy = xj - xi, yj - yi
    d = np.hypot(dx, dy)
    ax.annotate('', xy=(xj - 0.09*dx/d, yj - 0.09*dy/d),
                xytext=(xi + 0.09*dx/d, yi + 0.09*dy/d),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5,
                               mutation_scale=18))

# Nicht-Baumkanten als gestrichelt
for (u, v) in back_edges + forward_edges + cross_edges:
    xi, yi = node_xy[u]
    xj, yj = node_xy[v]
    dx, dy = xj - xi, yj - yi
    d = np.hypot(dx, dy) if np.hypot(dx, dy) > 0 else 1
    ax.plot([xi + 0.09*dx/d, xj - 0.09*dx/d],
            [yi + 0.09*dy/d, yj - 0.09*dy/d],
            '--', color='#bdc3c7', linewidth=1.0, alpha=0.6)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# DFS-Statistiken als Text
stats_text = (f'Baum-Kanten: {len(tree_edges)}\n'
              f'Rückwärts-Kanten: {len(back_edges)}\n'
              f'Vorwärts-Kanten: {len(forward_edges)}\n'
              f'Quer-Kanten: {len(cross_edges)}\n'
              f'Anzahl DFS-Bäume: {sum(1 for p in parent if p == -1)}')
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
        va='top', ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

plt.tight_layout()
plt.savefig(OUTDIR + 'plot13_hsdfs_wald.pdf', bbox_inches='tight')
plt.close()
print("Plot 13 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 14: LCS-Heatmap zwischen allen Ebenen (Micro, Meso, Super)
# ══════════════════════════════════════════════════════════════════════════════

# Wähle je 3 repräsentative Micro-Cluster, alle 3 Meso-Cluster, + Supercluster
rep_adjs = micro_adjs[:3] + meso_adjs + [A_super]
rep_names = ([f'$\\mathcal{{C}}^{{(0)}}_{i+1}$' for i in range(3)] +
             [f'$\\mathcal{{M}}_{i+1}$' for i in range(3)] +
             ['$\\mathcal{S}$'])

N_rep = len(rep_adjs)
lcs_full = np.zeros((N_rep, N_rep))
for i in range(N_rep):
    for j in range(N_rep):
        nA, nB = rep_adjs[i].shape[0], rep_adjs[j].shape[0]
        wA, wB = 2**nA, 2**nB
        rA = [s % wA for s in compute_signature(rep_adjs[i])]
        rB = [s % wB for s in compute_signature(rep_adjs[j])]
        lcs_full[i, j] = lcs_length(rA, rB)

fig, ax = plt.subplots(figsize=(9, 7))
cmap = LinearSegmentedColormap.from_list('lcs', ['#ffffff', '#2980b9', '#1a252f'])
im = ax.imshow(lcs_full, cmap=cmap, aspect='auto')
ax.set_xticks(range(N_rep))
ax.set_yticks(range(N_rep))
ax.set_xticklabels(rep_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(rep_names, fontsize=10)
ax.set_title('LCS-Ähnlichkeitsmatrix zwischen allen Hierarchieebenen\n'
             '(Micro-Cluster, Meso-Cluster, Supercluster)', fontsize=12)
plt.colorbar(im, ax=ax, label='LCS-Länge')
for i in range(N_rep):
    for j in range(N_rep):
        ax.text(j, i, f'{lcs_full[i,j]:.0f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if lcs_full[i,j] > lcs_full.max()*0.5 else '#2c3e50')

# Trennlinien zwischen Ebenen
ax.axhline(2.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.axhline(5.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(2.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(5.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.text(1.0, -0.7, 'Micro', ha='center', fontsize=10, color='red', fontweight='bold')
ax.text(4.0, -0.7, 'Meso', ha='center', fontsize=10, color='red', fontweight='bold')
ax.text(6.0, -0.7, 'Super', ha='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTDIR + 'plot14_lcs_hierarchie_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Plot 14 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 15: Gravitationspotential-Hierarchie (3 Ebenen)
# ══════════════════════════════════════════════════════════════════════════════

def potential_field_2d(grid_x, grid_y, positions, masses):
    G = 6.674e-11
    phi = np.zeros_like(grid_x)
    for p, m in zip(positions, masses):
        dx = grid_x - p[0]
        dy = grid_y - p[1]
        r = np.sqrt(dx**2 + dy**2)
        r = np.where(r < MIN_DISTANCE_THRESHOLD, MIN_DISTANCE_THRESHOLD, r)
        phi -= G * m / r
    return phi

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Gravitationspotential-Hierarchie der Sterncluster',
             fontsize=13, fontweight='bold')

x_range = np.linspace(-2*SCALE, 12*SCALE, 200)
y_range = np.linspace(-8*SCALE, 8*SCALE, 200)
gx, gy = np.meshgrid(x_range, y_range)

for level, (ax, title, use_groups) in enumerate(zip(
        axes,
        ['Ebene 0: Micro-Cluster\n(einzelne Sterncluster)',
         'Ebene 1: Meso-Cluster\n(kombinierte Gruppen)',
         'Ebene 2: Supercluster\n(gesamte Hierarchie)'],
        [range(3), [0, 3, 6], [0]])):

    if level == 0:
        plot_pos = np.vstack(micro_clusters_pos[:3])
        plot_m = np.concatenate(micro_clusters_m[:3])
    elif level == 1:
        plot_pos = np.vstack([micro_clusters_pos[i] for g in meso_groups for i in g[:2]])
        plot_m = np.concatenate([micro_clusters_m[i] for g in meso_groups for i in g[:2]])
    else:
        plot_pos = np.vstack(micro_clusters_pos)
        plot_m = np.concatenate(micro_clusters_m)

    phi = potential_field_2d(gx, gy, plot_pos, plot_m)
    phi_plot = np.log10(-phi + 1e-40)

    im = ax.contourf(gx / SCALE, gy / SCALE, phi_plot, levels=30, cmap='plasma_r')
    ax.contour(gx / SCALE, gy / SCALE, phi_plot, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Sternpositionen einzeichnen
    for gi, group in enumerate(meso_groups):
        if level < 2:
            plot_group = group[:3] if level == 0 else group
        else:
            plot_group = group
        if level == 0 and gi > 0:
            continue
        for idx in plot_group:
            p = micro_clusters_pos[idx] / SCALE
            ax.scatter(p[:, 0], p[:, 1], s=20, c='white', marker='*',
                       edgecolors='k', linewidths=0.3, zorder=10, alpha=0.8)

    plt.colorbar(im, ax=ax, label=r'$\log_{10}(-\Phi)$')
    ax.set_xlabel('x [$10^{15}$ m]')
    ax.set_ylabel('y [$10^{15}$ m]')
    ax.set_title(title, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot15_potential_hierarchie.pdf', bbox_inches='tight')
plt.close()
print("Plot 15 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 16: Subgraph-Erkennung zwischen Hierarchieebenen
# ══════════════════════════════════════════════════════════════════════════════

# Berechne Subgraph-Verhältnisse: Welche Micro-Cluster sind Subgraphen
# der Meso-Cluster? Welche Meso-Cluster sind Subgraphen des Superclusters?
results_micro_in_meso = np.zeros((9, 3), dtype=int)
for i in range(9):
    for j in range(3):
        ok, _ = subgraph_check(micro_adjs[i], meso_adjs[j])
        results_micro_in_meso[i, j] = 1 if ok else 0

results_meso_in_super = np.zeros(3, dtype=int)
for j in range(3):
    ok, _ = subgraph_check(meso_adjs[j], A_super)
    results_meso_in_super[j] = 1 if ok else 0

# LCS-Werte Micro→Meso
lcs_micro_meso = np.zeros((9, 3))
for i in range(9):
    for j in range(3):
        nA, nB = micro_adjs[i].shape[0], meso_adjs[j].shape[0]
        wA, wB = 2**nA, 2**nB
        rA = [s % wA for s in compute_signature(micro_adjs[i])]
        rB = [s % wB for s in compute_signature(meso_adjs[j])]
        lcs_micro_meso[i, j] = lcs_length(rA, rB)

# LCS-Werte Meso→Super
lcs_meso_super = np.zeros(3)
for j in range(3):
    nA, nB = meso_adjs[j].shape[0], A_super.shape[0]
    wA, wB = 2**nA, 2**nB
    rA = [s % wA for s in compute_signature(meso_adjs[j])]
    rB = [s % wB for s in compute_signature(A_super)]
    lcs_meso_super[j] = lcs_length(rA, rB)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Subgraph-Erkennung zwischen Hierarchieebenen',
             fontsize=13, fontweight='bold')

# Linkes Panel: Micro → Meso
ax = axes[0]
im0 = ax.imshow(lcs_micro_meso, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks(range(3))
ax.set_xticklabels([f'$\\mathcal{{M}}_{j+1}$' for j in range(3)], fontsize=12)
ax.set_yticks(range(9))
ax.set_yticklabels([f'$\\mathcal{{C}}^{{(0)}}_{i+1}$' for i in range(9)], fontsize=10)
ax.set_title('LCS: Micro-Cluster → Meso-Cluster\n(Subgraph-Einbettungstiefe)', fontsize=11)
plt.colorbar(im0, ax=ax, label='LCS-Länge')
for i in range(9):
    for j in range(3):
        marker = '✓' if results_micro_in_meso[i, j] else '·'
        col = 'white' if lcs_micro_meso[i,j] > lcs_micro_meso.max()*0.6 else '#2c3e50'
        ax.text(j, i, f'{lcs_micro_meso[i,j]:.0f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=col)

# Trennlinien zwischen Meso-Gruppen
ax.axhline(2.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.6)
ax.axhline(5.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.6)

# Rechtes Panel: Meso → Super + Balkendiagramm
ax = axes[1]
bars = ax.bar(range(3), lcs_meso_super, color=['#1f77b4', '#d62728', '#2ca02c'],
              alpha=0.8, edgecolor='k', linewidth=1)
ax.axhline(2, color='orange', linewidth=2, linestyle='--',
           label='Subgraph-Schwelle (LCS=2)')
for j, (bar, val) in enumerate(zip(bars, lcs_meso_super)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.0f}',
            ha='center', fontsize=12, fontweight='bold')
    status = '✓ Subgraph' if results_meso_in_super[j] else '✗ kein Subgraph'
    color = 'green' if results_meso_in_super[j] else 'red'
    ax.text(bar.get_x() + bar.get_width()/2, val/2,
            status, ha='center', va='center', fontsize=9, color=color,
            fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels([f'$\\mathcal{{M}}_{j+1}$ → $\\mathcal{{S}}$' for j in range(3)],
                   fontsize=11)
ax.set_ylabel('LCS-Länge')
ax.set_title('LCS: Meso-Cluster → Supercluster\n(Hierarchische Subgraph-Einbettung)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.4)
ax.set_ylim(0, lcs_meso_super.max() + 1.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot16_subgraph_hierarchie.pdf', bbox_inches='tight')
plt.close()
print("Plot 16 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 17: Energie-Hierarchie (Bindungsenergie auf 3 Ebenen)
# ══════════════════════════════════════════════════════════════════════════════

# Berechne Bindungsenergien auf allen Ebenen
micro_energies = []
for p, m in zip(micro_clusters_pos, micro_clusters_m):
    micro_energies.append(abs(cluster_binding_energy(p, m)))

meso_energies = []
for group in meso_groups:
    all_p = np.vstack([micro_clusters_pos[i] for i in group])
    all_m = np.concatenate([micro_clusters_m[i] for i in group])
    meso_energies.append(abs(cluster_binding_energy(all_p, all_m)))

super_energy = abs(cluster_binding_energy(super_p, super_m))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Bindungsenergie-Hierarchie der Sterncluster', fontsize=13, fontweight='bold')

# Linkes Panel: Energie aller Micro-Cluster nach Meso-Gruppe
ax = axes[0]
x_pos = np.arange(9)
bar_colors = ([colors_micro[i] for i in range(3)] +
              [colors_micro[i+3] for i in range(3)] +
              [colors_micro[i+6] for i in range(3)])
energies_J = np.array(micro_energies)
bars = ax.bar(x_pos, energies_J / 1e28, color=bar_colors, alpha=0.8,
              edgecolor='k', linewidth=0.8)
# Meso-Gruppen markieren
for gi, group in enumerate(meso_groups):
    ax.axvspan(group[0] - 0.5, group[-1] + 0.5, alpha=0.08,
               color=meso_colors[gi])
    mean_e = np.mean(energies_J[[g for g in group]]) / 1e28
    ax.hlines(mean_e, group[0] - 0.5, group[-1] + 0.5,
              colors=meso_colors[gi], linewidth=2, linestyle='--',
              label=f'$\\mathcal{{M}}_{gi+1}$: $\\bar{{E}}$={mean_e:.1f}')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'$\\mathcal{{C}}^{{(0)}}_{i+1}$' for i in range(9)],
                   rotation=45, ha='right', fontsize=9)
ax.set_ylabel(r'$|E_{\rm bind}|$ [$10^{28}$ J]')
ax.set_title('Bindungsenergie der Micro-Cluster\n(gruppiert nach Meso-Ebene)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.4)

# Rechtes Panel: Energieskala über alle Ebenen
ax = axes[1]
level_names = ([f'$\\mathcal{{C}}^{{(0)}}_{i+1}$' for i in range(9)] +
               [f'$\\mathcal{{M}}_{j+1}$' for j in range(3)] +
               ['$\\mathcal{S}$'])
level_energies = np.array(micro_energies + meso_energies + [super_energy])
level_colors = (bar_colors + list(meso_colors) + ['purple'])

ax.barh(range(len(level_names)), level_energies / 1e28,
        color=level_colors, alpha=0.8, edgecolor='k', linewidth=0.5)
ax.set_yticks(range(len(level_names)))
ax.set_yticklabels(level_names, fontsize=9)
ax.set_xlabel(r'$|E_{\rm bind}|$ [$10^{28}$ J]')
ax.set_title('Hierarchische Energieskala\n(alle Ebenen)', fontsize=11)
ax.axhline(8.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='Meso/Super-Grenze')
ax.axhline(11.5, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax.grid(axis='x', alpha=0.4)

# Energiegewinn durch Hierarchisierung
e_micro_sum = sum(micro_energies) / 1e28
e_super = super_energy / 1e28
ax.text(0.98, 0.98,
        f'$\\Sigma E_{{\\rm micro}}$: {e_micro_sum:.0f}$\\times 10^{{28}}$ J\n'
        f'$E_{{\\rm super}}$: {e_super:.0f}$\\times 10^{{28}}$ J\n'
        f'Verhältnis: {e_super/e_micro_sum:.1f}$\\times$',
        transform=ax.transAxes, va='top', ha='right',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot17_energie_hierarchie.pdf', bbox_inches='tight')
plt.close()
print("Plot 17 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 18: Komplexitätsvergleich: Einstufig vs. Hierarchisch
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Komplexitätsvergleich: Direkter vs. Hierarchischer HS-DFS',
             fontsize=13, fontweight='bold')

n_values = np.arange(5, 105, 5)

# Direkte Analyse: O(N^2 * n^3) für N Cluster mit je n Sternen
# Modell: jeder Cluster hat durchschnittlich n/5 Sterne (d.h. 5 Sterne pro Cluster)
def complexity_direct(n, N=None):
    if N is None:
        N = max(1, n // 5)  # Annahme: ~5 Sterne pro Cluster
    return N**2 * n**3


# Hierarchische Analyse: O(K^2 * (n/K)^3 + K^3) mit K Meta-Knoten
# Modell: K Gruppen, jede mit n/K Sternen; Meta-Graph mit K Knoten
def complexity_hierarchical(n, K=3):
    n0 = max(1, n // K)
    return K**2 * n0**3 + K**3


# LCS: O(n^2) pro Paar, N Cluster mit je n/5 Sternen
def complexity_lcs_direct(n, N=None):
    if N is None:
        N = max(1, n // 5)  # Annahme: ~5 Sterne pro Cluster
    return N**2 * n**2

comp_direct = np.array([complexity_direct(n) for n in n_values])
comp_hier = np.array([complexity_hierarchical(n) for n in n_values])
comp_lcs = np.array([complexity_lcs_direct(n) for n in n_values])

ax = axes[0]
ax.semilogy(n_values, comp_direct, 'b-o', label='Direkter Subgraph-Check: $O(N^2 n^3)$',
            markersize=4)
ax.semilogy(n_values, comp_hier, 'r-s', label='Hierarchisch: $O(K^2(n/K)^3 + K^3)$',
            markersize=4)
ax.semilogy(n_values, comp_lcs, 'g-^', label='LCS-Vergleich: $O(N^2 n^2)$',
            markersize=4)
ax.set_xlabel('Gesamtzahl Sterne $n$')
ax.set_ylabel('Rechenschritte (log-Skala)')
ax.set_title('Komplexität vs. Sternanzahl', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4, which='both')

# Speedup-Faktor der Hierarchie
ax = axes[1]
speedup = comp_direct / np.maximum(comp_hier, 1)
ax.plot(n_values, speedup, 'purple', linewidth=2.5, marker='D', markersize=4)
ax.fill_between(n_values, 1, speedup, alpha=0.2, color='purple')
ax.axhline(1, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Gesamtzahl Sterne $n$')
ax.set_ylabel('Speedup-Faktor')
ax.set_title('Hierarchischer Speedup-Faktor\n(Direkt / Hierarchisch)', fontsize=11)
ax.grid(True, alpha=0.4)

# Annotationen
max_speedup_idx = np.argmax(speedup)
ax.annotate(f'Max. Speedup: {speedup[max_speedup_idx]:.0f}×\nbei n={n_values[max_speedup_idx]}',
            xy=(n_values[max_speedup_idx], speedup[max_speedup_idx]),
            xytext=(n_values[max_speedup_idx] - 20, speedup[max_speedup_idx] * 0.7),
            arrowprops=dict(arrowstyle='->', color='purple'),
            fontsize=10, color='purple',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot18_komplexitaet_vergleich.pdf', bbox_inches='tight')
plt.close()
print("Plot 18 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 19: DFS-Traversal mit Zeitstempeln und Kanten-Klassifikation
# ══════════════════════════════════════════════════════════════════════════════

# Detaillierte Visualisierung der DFS-Traversal auf dem 5-Knoten-Meta-Graphen
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('HS-DFS Traversal: Schritt-für-Schritt auf dem Meta-Graphen',
             fontsize=13, fontweight='bold')

# DFS-Schritte simulieren (5 Knoten, Reihenfolge des DFS)
node_pos_dfs = {
    0: (0.5, 0.85),
    1: (0.15, 0.55),
    2: (0.85, 0.55),
    3: (0.25, 0.15),
    4: (0.75, 0.15),
}
demo_colors_dfs = {
    'unbesucht': '#ecf0f1',
    'aktiv': '#f39c12',
    'abgeschlossen': '#2ecc71',
}

# Schrittfolge für Demo-DFS
dfs_steps = [
    {'visited': [], 'active': [], 'done': [], 'title': 'Schritt 0: Start'},
    {'visited': [0], 'active': [0], 'done': [], 'title': 'Schritt 1: Besuche $G_1$'},
    {'visited': [0, 1], 'active': [0, 1], 'done': [], 'title': 'Schritt 2: Rekursion $G_2$'},
    {'visited': [0, 1, 3], 'active': [0, 1, 3], 'done': [], 'title': 'Schritt 3: Rekursion $G_4$'},
    {'visited': [0, 1, 3], 'active': [0, 1], 'done': [3], 'title': 'Schritt 4: $G_4$ fertig'},
    {'visited': [0, 1, 3], 'active': [0], 'done': [3, 1], 'title': 'Schritt 5: $G_2$ fertig'},
]

for step_idx, (ax, step) in enumerate(zip(axes.flat[:6], dfs_steps)):
    ax.set_title(step['title'], fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.axis('off')

    # Kanten zuerst
    for i in range(n_demo_clusters):
        for j in demo_meta[i]:
            xi, yi = node_pos_dfs[i]
            xj, yj = node_pos_dfs[j]
            dx, dy = xj - xi, yj - yi
            d = np.hypot(dx, dy) if np.hypot(dx, dy) > 0 else 1
            col = '#bdc3c7'
            lw = 1.0
            if (i, j) in tree_edges and i in step['visited'] and j in step['visited']:
                col = '#27ae60'
                lw = 2.5
            ax.annotate('', xy=(xj - 0.09*dx/d, yj - 0.09*dy/d),
                        xytext=(xi + 0.09*dx/d, yi + 0.09*dy/d),
                        arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                       mutation_scale=14))

    # Knoten
    for nid in range(n_demo_clusters):
        xy = node_pos_dfs[nid]
        if nid in step['done']:
            c = demo_colors_dfs['abgeschlossen']
            ec = '#27ae60'
        elif nid in step['active']:
            c = demo_colors_dfs['aktiv']
            ec = '#e67e22'
        else:
            c = demo_colors_dfs['unbesucht']
            ec = '#7f8c8d'
        circ = plt.Circle(xy, 0.09, color=c, ec=ec, linewidth=2, zorder=5)
        ax.add_patch(circ)
        ax.text(xy[0], xy[1], demo_names[nid], ha='center', va='center',
                fontsize=12, fontweight='bold', color='#2c3e50', zorder=6)
        ts = f'd={discovery[nid]}, f={finish[nid]}' if nid in step['visited'] else ''
        ax.text(xy[0], xy[1] - 0.14, ts, ha='center', fontsize=8, color='#555')

    # Legende (nur im ersten Panel)
    if step_idx == 0:
        for label, col, ec in [('Unbesucht', '#ecf0f1', '#7f8c8d'),
                                ('Aktiv (grau)', '#f39c12', '#e67e22'),
                                ('Fertig (schwarz)', '#2ecc71', '#27ae60')]:
            ax.scatter([], [], s=200, c=col, edgecolors=ec, linewidths=1.5,
                       label=label)
        ax.legend(loc='lower right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot19_dfs_traversal.pdf', bbox_inches='tight')
plt.close()
print("Plot 19 gespeichert.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 20: Energiepotentiale der Hierarchie – Gewinnungspotentiale
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 6))
fig.suptitle('Energiegewinnungspotentiale aus hierarchischen Sterncluster-Strukturen',
             fontsize=13, fontweight='bold')

# Panel 1: Gravitationsenergie vs. Hierarchietiefe
ax1 = fig.add_subplot(1, 3, 1)
depths = np.array([0, 1, 2])
energies_per_depth = [
    np.mean(micro_energies) / 1e28,
    np.mean(meso_energies) / 1e28,
    super_energy / 1e28
]
depth_labels = ['Micro\n(Ebene 0)', 'Meso\n(Ebene 1)', 'Super\n(Ebene 2)']
bars_d = ax1.bar(depths, energies_per_depth,
                  color=['#3498db', '#e74c3c', '#9b59b6'],
                  alpha=0.85, edgecolor='k', linewidth=1.2, width=0.5)
ax1.set_xticks(depths)
ax1.set_xticklabels(depth_labels, fontsize=10)
ax1.set_ylabel(r'$|E_{\rm bind}|$ [$10^{28}$ J]')
ax1.set_title('Bindungsenergie\npro Hierarchieebene', fontsize=11)
for bar, val in zip(bars_d, energies_per_depth):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.4)

# Virialenergie-Annotation
e_kin_micro = np.mean(micro_energies) / 2e28
ax1.text(0.98, 0.98,
         f'Virialtheorem:\n$E_{{\\rm kin}} = -\\frac{{1}}{{2}}E_{{\\rm pot}}$\n'
         f'$E_{{\\rm kin,Micro}}$ ≈ {e_kin_micro:.1f}$\\times 10^{{28}}$ J',
         transform=ax1.transAxes, va='top', ha='right', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Panel 2: Kumulierter Energiegewinn durch Substruktur-Auflösung
ax2 = fig.add_subplot(1, 3, 2)
# Modell: E_release = E_super - sum(E_meso_i) bei "Auflösung"
e_micro_individual = np.array(micro_energies) / 1e28
e_meso_individual = np.array(meso_energies) / 1e28
e_super_val = super_energy / 1e28

e_gain_meso_from_micro = e_meso_individual - np.array([
    sum(e_micro_individual[i] for i in group) for group in meso_groups
])
e_gain_super_from_meso = e_super_val - sum(e_meso_individual)

categories = (['$\\mathcal{M}_1$', '$\\mathcal{M}_2$', '$\\mathcal{M}_3$'] +
              ['$\\mathcal{S}$'])
gains = list(e_gain_meso_from_micro) + [e_gain_super_from_meso]
colors_gain = (['#e74c3c', '#e67e22', '#f1c40f'] + ['#9b59b6'])

bars_g = ax2.bar(range(len(categories)), gains,
                  color=colors_gain, alpha=0.85, edgecolor='k', linewidth=0.8)
ax2.axhline(0, color='k', linewidth=1)
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_ylabel(r'$\Delta E_{\rm bind}$ [$10^{28}$ J]')
ax2.set_title('Gravitationsenergie-Gewinn\nbei Cluster-Zusammenfusion', fontsize=11)
for bar, val in zip(bars_g, gains):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.03 if val >= 0 else -0.08),
             f'{val:+.2f}', ha='center', fontsize=10)
ax2.grid(axis='y', alpha=0.4)

# Panel 3: Theoretischer Kardashev-Index durch Hierarchienutzung
ax3 = fig.add_subplot(1, 3, 3)
kardashev_energies = {
    'Typ I\n(~10^{26} W)': 1e26,
    'Typ II\n(~10^{33} W)': 1e33,
    'Typ III\n(~10^{43} W)': 1e43,
}
# Extrahierbare Leistung aus Cluster-Hierarchie (über 1 Myr)
t_myr = 1e6 * 3.156e7  # 1 Myr in Sekunden
extracted_powers = {
    'Micro\n(1 Cluster)': abs(e_gain_meso_from_micro[0]) * 1e28 / t_myr,
    'Meso\n(1 Gruppe)': abs(e_gain_super_from_meso) * 1e28 / (3 * t_myr),
    'Super\n(gesamte\nHierarchie)': e_super_val * 1e28 / (10 * t_myr),
}

k_names = list(extracted_powers.keys())
k_powers = list(extracted_powers.values())
colors_k = ['#3498db', '#e74c3c', '#9b59b6']

ax3.bar(range(len(k_names)), np.log10(np.maximum(k_powers, 1e1)),
        color=colors_k, alpha=0.85, edgecolor='k', linewidth=1)
# Kardashev-Referenzlinien
for kname, ke in kardashev_energies.items():
    ax3.axhline(np.log10(ke), linestyle='--', alpha=0.6, linewidth=1.5,
                label=kname.replace('\n', ' '))
ax3.set_xticks(range(len(k_names)))
ax3.set_xticklabels(k_names, fontsize=9)
ax3.set_ylabel(r'$\log_{10}(P)$ [W]')
ax3.set_title('Extrahierbare Leistung\n(Kardashev-Einordnung)', fontsize=11)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(axis='y', alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR + 'plot20_energie_kardashev.pdf', bbox_inches='tight')
plt.close()
print("Plot 20 gespeichert.")

print("\nAlle Plots (11-20) erfolgreich gespeichert!")
