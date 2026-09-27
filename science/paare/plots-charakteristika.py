#!/usr/bin/env python3
"""
Plots 7-11 fuer: Wesentliche Charakteristika natuerlicher Paare
Einfachheit, Harmonie, Formalitaet, Verbindung
Stephan Epp, Universitaet Bielefeld, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from itertools import combinations

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
ORANGE    = '#C06000'
PURPLE    = '#5A0A8A'
GRAY      = '#707080'
GOLD      = '#B8860B'
OUT = '/home/claude/'

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def kolmogorov_approx(n_nodes, n_edges):
    """Approximierte Beschreibungslaenge K(G) = |V| + |E| (vereinfacht)."""
    return n_nodes + n_edges

def spectral_symmetry(A):
    """Spektrale Symmetrie rho: Anteil paariger Eigenwerte (bipartit -> rho=1)."""
    eigvals = np.sort(np.linalg.eigvalsh(A))
    mirror  = eigvals + eigvals[::-1]
    E_total = np.sum(np.abs(eigvals))
    if E_total < 1e-12:
        return 0.0
    return float(1.0 - np.sum(np.abs(mirror)) / (2.0 * E_total))

def graph_energy(A):
    """Graphenergie E(G) = Sum |lambda_i|."""
    return float(np.sum(np.abs(np.linalg.eigvalsh(A))))

def matching_number(G):
    return len(nx.max_weight_matching(G, maxcardinality=True))

def bridge_fraction(G):
    if G.number_of_edges() == 0:
        return 0.0
    bridges = list(nx.bridges(G))
    return len(bridges) / G.number_of_edges()

def connected_pair_fraction(G):
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    total = n * (n - 1) / 2
    reachable = sum(1 for u, v in combinations(G.nodes(), 2)
                    if nx.has_path(G, u, v))
    return reachable / total

def axiom_scores(G, A):
    """Gewichtete Axiom-Erfuellung (Gewichte: A1=2, A2=2, A3=3, A4=1, A5=2)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    connected   = nx.is_connected(G)
    acyclic     = nx.is_forest(G) if connected else False
    bipartite   = nx.is_bipartite(G)
    a1 = 2 if n == 2 else 0   # |V| = 2
    a2 = 2 if m == 1 else 0   # |E| = 1
    a3 = 3 if connected else 0
    a4 = 1 if acyclic   else 0
    a5 = 2 if bipartite else 0
    return np.array([a1, a2, a3, a4, a5], dtype=float) / np.array([2,2,3,1,2], dtype=float)

def compute_sigma(A, j):
    """Signatur sigma_j gemaess Subgraph Algorithmus (Epp 2026)."""
    n = A.shape[0]
    val = sum(int(A[i, j]) * (2**i) for i in range(n)) + j * (2**n)
    return val

def vier_charakteristika(G, A):
    """Berechne die vier normalisierten Charakteristika (alle in [0,1])."""
    n  = G.number_of_nodes()
    m  = G.number_of_edges()
    nu = matching_number(G)
    # 1. Einfachheit S = K(K12) / K(G) falls verbunden, sonst 0
    K_pair = kolmogorov_approx(2, 1)
    K_G    = kolmogorov_approx(n, m)
    conn   = nx.is_connected(G)
    S = (K_pair / K_G) if (conn and K_G > 0) else 0.0
    S = min(S, 1.0)
    # 2. Harmonie H = 2*nu / |V|
    H = 2 * nu / n if n > 0 else 0.0
    # 3. Formalitaet F = gewichtete Axiom-Erfuellung
    ax = axiom_scores(G, A)
    F  = float(np.mean(ax))  # already in [0,1] per axiom
    # 4. Verbindung V = bridge_fraction * connected_fraction * (2/n)
    bf  = bridge_fraction(G)
    cpf = connected_pair_fraction(G)
    V   = bf * cpf * (2.0 / n) if n >= 2 else 0.0
    V   = min(V, 1.0)
    return S, H, F, V

# ─────────────────────────────────────────────────────────────────────────────
# Graphen definieren
# ─────────────────────────────────────────────────────────────────────────────
A_k12 = np.array([[0,1],[1,0]], dtype=float)
A_p3  = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
A_k3  = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
A_c4  = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=float)
A_k4  = np.ones((4,4), dtype=float) - np.eye(4, dtype=float)
A_e2  = np.zeros((2,2), dtype=float)   # leerer 2-Graph
A_p4  = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=float)  # Pfad P4

G_k12 = nx.from_numpy_array(A_k12)
G_p3  = nx.from_numpy_array(A_p3)
G_k3  = nx.from_numpy_array(A_k3)
G_c4  = nx.from_numpy_array(A_c4)
G_k4  = nx.from_numpy_array(A_k4)
G_e2  = nx.from_numpy_array(A_e2)
G_p4  = nx.from_numpy_array(A_p4)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 7: Einfachheit – Minimale Struktur
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

graphs_info = [
    ('$K_{1,1}$\n(Paar)',    2, 1),
    ('$P_3$\n(Pfad-3)',      3, 2),
    ('$K_3$\n(Dreieck)',     3, 3),
    ('$C_4$\n(Kreis-4)',     4, 4),
    ('$P_4$\n(Pfad-4)',      4, 3),
    ('$K_4$\n(Komplett-4)',  4, 6),
    ('$S_4$\n(Stern-4)',     4, 3),
]
labels = [g[0] for g in graphs_info]
nv     = [g[1] for g in graphs_info]
ne     = [g[2] for g in graphs_info]
K_vals = [kolmogorov_approx(v, e) for v, e in zip(nv, ne)]
x = np.arange(len(labels))

# (a) Beschreibungslaenge K(G)
ax = axes[0]
bar_colors = [MAINBLUE if i == 0 else GRAY for i in range(len(labels))]
bars = ax.bar(x, K_vals, color=bar_colors, edgecolor='white', linewidth=1.2, alpha=0.88)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Beschreibungslaenge $K(G) = |V| + |E|$')
ax.set_title('(a) Kolmogorov-Beschreibungslaenge $K(G)$')
ax.axhline(K_vals[0], color=MAINBLUE, linestyle='--', lw=1.5, alpha=0.7,
           label=f'$K(K_{{1,1}}) = {K_vals[0]}$ (Minimum)')
for bar, val in zip(bars, K_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.08, str(val),
            ha='center', fontsize=10, fontweight='bold',
            color=MAINBLUE if val == K_vals[0] else GRAY)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(K_vals) + 1.5)

# (b) (|V|, |E|) Raum
ax = axes[1]
v_vals = np.array(nv, dtype=float)
e_vals = np.array(ne, dtype=float)
ax.scatter(v_vals[1:], e_vals[1:], color=GRAY, s=120, zorder=3, alpha=0.8)
ax.scatter([v_vals[0]], [e_vals[0]], color=MAINBLUE, s=220, zorder=5,
           marker='*', label='$K_{1,1}$ (Minimum)')
# Contour lines K = const
v_grid = np.linspace(1.5, 4.8, 200)
for k_const, alpha in [(3, 0.8), (5, 0.5), (7, 0.3), (9, 0.2)]:
    e_line = k_const - v_grid
    mask = (e_line >= 0) & (e_line <= v_grid * (v_grid - 1) / 2)
    ax.plot(v_grid[mask], e_line[mask], '--', color=ORANGE, alpha=alpha, lw=1.2)
    if np.any(mask):
        mid = np.where(mask)[0][len(np.where(mask)[0])//2]
        ax.text(v_grid[mid], e_line[mid] + 0.1, f'$K={k_const}$',
                fontsize=8, color=ORANGE, alpha=alpha+0.1)
for i, lbl in enumerate(labels):
    clean = lbl.replace('\n', ' ')
    ax.annotate(clean, (v_vals[i], e_vals[i]),
                textcoords='offset points', xytext=(6, 4), fontsize=8,
                color=MAINBLUE if i == 0 else GRAY)
ax.set_xlabel('Knotenanzahl $|V|$')
ax.set_ylabel('Kantenanzahl $|E|$')
ax.set_title('(b) Strukturraum $(|V|, |E|)$ mit Iso-Komplexitaets-Linien')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(1.5, 5.0); ax.set_ylim(-0.3, 7)

fig.suptitle('Abbildung 7: Einfachheit – $K_{1,1}$ als minimal-komplexer zusammenhaengender Graph',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot7_einfachheit.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 gespeichert.")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 8: Harmonisch – Spektrale Symmetrie
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

spec_graphs = [
    ('$K_{1,1}$', A_k12, MAINBLUE),
    ('$P_3$',     A_p3,  DARKGREEN),
    ('$K_3$',     A_k3,  ACCENTRED),
    ('$C_4$',     A_c4,  ORANGE),
]

# (a) Eigenwert-Spektren nebeneinander
ax = axes[0]
offsets = np.linspace(0, 3, len(spec_graphs))
for idx, (name, A, col) in enumerate(spec_graphs):
    eigvals = np.sort(np.linalg.eigvalsh(A))
    for ev in eigvals:
        ax.plot([offsets[idx] - 0.15, offsets[idx] + 0.15], [ev, ev],
                color=col, lw=3.5, solid_capstyle='round', alpha=0.9)
        ax.plot(offsets[idx], ev, 'o', color=col, ms=7, zorder=5)
    ax.text(offsets[idx], min(eigvals) - 0.35, name,
            ha='center', fontsize=11, color=col, fontweight='bold')
ax.axhline(0, color=GRAY, lw=0.8, linestyle='--', alpha=0.5)
ax.set_xticks([])
ax.set_ylabel('Eigenwert $\\lambda_i$')
ax.set_title('(a) Eigenwertspektren der Adjazenzmatrix')
ax.set_xlim(-0.5, 3.5)
ax.text(0, 1.15, '$\\rho=1$', ha='center', fontsize=10, color=MAINBLUE, fontweight='bold')
ax.text(1, np.sqrt(2)+0.15, '$\\rho=1$', ha='center', fontsize=10, color=DARKGREEN, fontweight='bold')
ax.text(2, 2.15, '$\\rho<1$', ha='center', fontsize=10, color=ACCENTRED, fontweight='bold')
ax.text(3, 2.15, '$\\rho=1$', ha='center', fontsize=10, color=ORANGE, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
handles = [mpatches.Patch(color=c, label=n) for n, _, c in spec_graphs]
ax.legend(handles=handles, fontsize=9, loc='lower right')

# (b) Spektrale Symmetrie rho und Graphenergie E/|V|
ax = axes[1]
all_graphs_spec = [
    ('$K_{1,1}$', A_k12, MAINBLUE),
    ('$P_3$',     A_p3,  DARKGREEN),
    ('$K_3$',     A_k3,  ACCENTRED),
    ('$C_4$',     A_c4,  ORANGE),
    ('$P_4$',     A_p4,  PURPLE),
    ('$K_4$',     A_k4,  GRAY),
]
names_s  = [g[0] for g in all_graphs_spec]
rho_vals = [spectral_symmetry(g[1]) for g in all_graphs_spec]
eng_vals = [graph_energy(g[1]) / g[1].shape[0] for g in all_graphs_spec]
colors_s = [g[2] for g in all_graphs_spec]
x2 = np.arange(len(names_s))

ax2b = ax.twinx()
bars1 = ax.bar(x2 - 0.2, rho_vals, 0.35, color=colors_s, alpha=0.85,
               label='Spektrale Symmetrie $\\rho(G)$')
bars2 = ax2b.bar(x2 + 0.2, eng_vals, 0.35, color=colors_s, alpha=0.40,
                 hatch='//', label='Graphenergie $\\mathcal{E}(G)/|V|$')
ax.set_xticks(x2); ax.set_xticklabels(names_s, fontsize=10)
ax.set_ylabel('Spektrale Symmetrie $\\rho(G)$', color=MAINBLUE)
ax2b.set_ylabel('Graphenergie $\\mathcal{E}(G)/|V|$', color=GRAY)
ax.set_ylim(0, 1.35); ax2b.set_ylim(0, 2.0)
ax.set_title('(b) Spektrale Symmetrie und normierte Graphenergie')
ax.axhline(1.0, color=MAINBLUE, linestyle='--', lw=1.2, alpha=0.5)
h1 = mpatches.Patch(color=MAINBLUE, alpha=0.85, label='$\\rho(G)$ (Symmetrie)')
h2 = mpatches.Patch(color=GRAY,     alpha=0.40, hatch='//', label='$\\mathcal{E}/|V|$ (Energie)')
ax.legend(handles=[h1, h2], fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, (r, e) in enumerate(zip(rho_vals, eng_vals)):
    ax.text(i - 0.2, r + 0.03, f'{r:.2f}', ha='center', fontsize=8,
            color=MAINBLUE, fontweight='bold' if r == 1.0 else 'normal')

fig.suptitle('Abbildung 8: Harmonisch – Spektrale Symmetrie und Graphenergie natuerlicher Paare',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot8_harmonisch_spektral.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 gespeichert.")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 9: Formal – Axiome und Signatur
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Axiom-Erfuellungs-Heatmap
axiom_graphs = [
    ('$K_{1,1}$\n(Paar)',    G_k12, A_k12),
    ('$E_2$\n(leer)',         G_e2,  A_e2),
    ('$P_3$\n(Pfad-3)',       G_p3,  A_p3),
    ('$K_3$\n(Dreieck)',      G_k3,  A_k3),
    ('$P_4$\n(Pfad-4)',       G_p4,  A_p4),
    ('$C_4$\n(Kreis-4)',      G_c4,  A_c4),
    ('$K_4$\n(Komplett)',     G_k4,  A_k4),
]
axiom_labels = ['A1\n$|V|=2$', 'A2\n$|E|=1$', 'A3\nVerbunden',
                'A4\nKreisfrei', 'A5\nBipartit']
matrix = np.array([axiom_scores(G, A) for _, G, A in axiom_graphs])

ax = axes[0]
cmap = mcolors.LinearSegmentedColormap.from_list('gh', [ACCENTRED, 'white', DARKGREEN])
im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(axiom_labels, fontsize=9)
ax.set_yticks(range(len(axiom_graphs)))
ax.set_yticklabels([g[0] for g in axiom_graphs], fontsize=10)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=10, color='white' if (val > 0.7 or val < 0.3) else 'black',
                fontweight='bold' if val == 1.0 else 'normal')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Axiom-Erfuellungsgrad')
ax.set_title('(a) Axiom-Erfuellungsmatrix\n($K_{1,1}$ erfullt alle Axiome maximal)')
# Highlight K₁,₁ row
for j in range(5):
    ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, 1,
                 fill=False, edgecolor=MAINBLUE, lw=2.5))

# (b) Signatur-Raum (sigma_0, sigma_1)
ax = axes[1]
# Alle einfachen Graphen auf n=2 Knoten: E2 und K12
sig_2 = {}
for name, A in [('$E_2$ (leer)', A_e2), ('$K_{1,1}$ (Paar)', A_k12)]:
    s0 = compute_sigma(A, 0)
    s1 = compute_sigma(A, 1)
    sig_2[name] = (s0, s1)

# Alle einfachen Graphen auf n=3 Knoten
A_e3   = np.zeros((3,3))
A_p3x  = np.array([[0,1,0],[1,0,0],[0,0,0]])  # ein Paar + Isolierter
sigs_3 = {
    '$E_3$':           (compute_sigma(A_e3,  0), compute_sigma(A_e3,  1)),
    '$K_{1,1}+$ iso.': (compute_sigma(A_p3x, 0), compute_sigma(A_p3x, 1)),
    '$P_3$':           (compute_sigma(A_p3,  0), compute_sigma(A_p3,  1)),
    '$K_3$':           (compute_sigma(A_k3,  0), compute_sigma(A_k3,  1)),
}

# Plot n=2
for name, (s0, s1) in sig_2.items():
    col = MAINBLUE if 'Paar' in name else ACCENTRED
    ms  = 220 if 'Paar' in name else 100
    mk  = '*' if 'Paar' in name else 'o'
    ax.scatter([s0], [s1], color=col, s=ms, marker=mk, zorder=5)
    ax.annotate(name, (s0, s1), textcoords='offset points',
                xytext=(8, 4), fontsize=9, color=col, fontweight='bold')

# Plot n=3
for name, (s0, s1) in sigs_3.items():
    ax.scatter([s0], [s1], color=GRAY, s=80, marker='s', zorder=4, alpha=0.7)
    ax.annotate(name, (s0, s1), textcoords='offset points',
                xytext=(6, 4), fontsize=8, color=GRAY)

ax.set_xlabel('Signatur $\\sigma_0$')
ax.set_ylabel('Signatur $\\sigma_1$')
ax.set_title('(b) Signaturraum $(\\sigma_0, \\sigma_1)$\n$K_{1,1}$ eindeutig bei $(2,\\,5)$')
ax.annotate('$(2,5)$\n$\\longleftarrow$ eindeutige\nPaar-Signatur',
            xy=(2, 5), xytext=(4, 3.5),
            arrowprops=dict(arrowstyle='->', color=MAINBLUE, lw=1.8),
            fontsize=9, color=MAINBLUE)
ax.grid(alpha=0.3)

fig.suptitle('Abbildung 9: Formal – Axiomerfuellung und Signaturraum des Paares $K_{1,1}$',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot9_formal.pdf', bbox_inches='tight')
plt.close()
print("Plot 9 gespeichert.")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 10: Verbindend – Bruecken und Distanzminimierung
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Vor und nach der Paarung
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(a) Vor und nach Paarbildung: Distanz $\\infty \\to 1$')
# Vor der Paarung: zwei isolierte Knoten
c1 = plt.Circle((0.18, 0.62), 0.07, color=ACCENTRED, zorder=4)
c2 = plt.Circle((0.40, 0.62), 0.07, color=ACCENTRED, zorder=4)
ax.add_patch(c1); ax.add_patch(c2)
ax.text(0.18, 0.62, '$a$', ha='center', va='center', fontsize=12,
        color='white', fontweight='bold', zorder=5)
ax.text(0.40, 0.62, '$b$', ha='center', va='center', fontsize=12,
        color='white', fontweight='bold', zorder=5)
ax.text(0.29, 0.80, 'Vor Paarung', ha='center', fontsize=10,
        fontweight='bold', color=ACCENTRED)
ax.text(0.29, 0.50, '$d(a,b) = \\infty$', ha='center', fontsize=10,
        color=ACCENTRED)
ax.text(0.29, 0.44, '$H(G) = 0$', ha='center', fontsize=10, color=ACCENTRED)
ax.text(0.29, 0.38, 'Kein emergentes\nPhaenomen', ha='center',
        fontsize=9, color=GRAY)
# Nach der Paarung
c3 = plt.Circle((0.62, 0.62), 0.07, color=MAINBLUE, zorder=4)
c4 = plt.Circle((0.84, 0.62), 0.07, color=MAINBLUE, zorder=4)
ax.add_patch(c3); ax.add_patch(c4)
ax.text(0.62, 0.62, '$a$', ha='center', va='center', fontsize=12,
        color='white', fontweight='bold', zorder=5)
ax.text(0.84, 0.62, '$b$', ha='center', va='center', fontsize=12,
        color='white', fontweight='bold', zorder=5)
ax.annotate('', xy=(0.77, 0.62), xytext=(0.69, 0.62),
            arrowprops=dict(arrowstyle='->', color=DARKGREEN, lw=3.5))
ax.text(0.73, 0.70, '$e$', ha='center', fontsize=10,
        color=DARKGREEN, fontweight='bold')
ax.text(0.73, 0.80, 'Nach Paarung', ha='center', fontsize=10,
        fontweight='bold', color=MAINBLUE)
ax.text(0.73, 0.50, '$d(a,b) = 1$', ha='center', fontsize=10, color=MAINBLUE)
ax.text(0.73, 0.44, '$H(G) = 1$', ha='center', fontsize=10, color=MAINBLUE)
ax.text(0.73, 0.38, 'Emergente Verbindung\n(Harmonie)', ha='center',
        fontsize=9, color=DARKGREEN)
# Pfeil zwischen den zwei Zustaenden
ax.annotate('', xy=(0.54, 0.62), xytext=(0.48, 0.62),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=2.0))
ax.text(0.51, 0.67, 'Paarung', ha='center', fontsize=9, color=GRAY)
ax.axvline(0.51, color=GRAY, lw=0.5, linestyle=':', alpha=0.5)

# (b) Groesseres Netzwerk: Bruecken-Hervorhebung
ax = axes[1]
# Konstruiere ein Netzwerk aus 4 Paaren, mit Verbindungskanten
G_net = nx.Graph()
G_net.add_edges_from([(0,1),(2,3),(4,5),(6,7)])  # Paar-Kanten (Bruecken)
G_net.add_edges_from([(1,2),(3,4),(5,6)])          # Verbindungskanten
pos = {0:(0,1.5), 1:(1,1.5), 2:(2,1.5), 3:(3,1.5),
       4:(3,0.0), 5:(2,0.0), 6:(1,0.0), 7:(0,0.0)}
bridg = list(nx.bridges(G_net))
non_b = [e for e in G_net.edges() if e not in bridg and (e[1],e[0]) not in bridg]
bc = nx.edge_betweenness_centrality(G_net)
# Draw
nx.draw_networkx_nodes(G_net, pos, ax=ax, node_color=MAINBLUE,
                       node_size=600, alpha=0.92)
nx.draw_networkx_labels(G_net, pos, ax=ax, font_color='white',
                        font_size=9, font_weight='bold')
nx.draw_networkx_edges(G_net, pos, ax=ax, edgelist=bridg,
                       edge_color=DARKGREEN, width=3.5, alpha=0.9,
                       label='Bruecke (Paar-Kante)')
nx.draw_networkx_edges(G_net, pos, ax=ax, edgelist=non_b,
                       edge_color=GRAY, width=1.8, alpha=0.6,
                       style='dashed', label='Verbindungskante')
edge_labels = {e: f'{bc[e]:.2f}' for e in G_net.edges()}
nx.draw_networkx_edge_labels(G_net, pos, edge_labels=edge_labels,
                             ax=ax, font_size=8, font_color=ORANGE)
ax.set_title('(b) Brueckenanalyse: Paar-Kanten (gruen) als\nkritische Verbindungselemente\n'
             '(Betweenness-Zentralitaet in Orange)')
ax.axis('off')
h_b = mpatches.Patch(color=DARKGREEN, label='Bruecke / Paar-Kante')
h_n = mpatches.Patch(color=GRAY,      label='Verbindungskante')
ax.legend(handles=[h_b, h_n], fontsize=9, loc='lower center')

fig.suptitle('Abbildung 10: Verbindend – Paarung als strukturschaffende Verbindung und Bruecke',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot10_verbindend.pdf', bbox_inches='tight')
plt.close()
print("Plot 10 gespeichert.")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 11: Zusammenfassung – Radar der vier Charakteristika
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

radar_graphs = [
    ('$K_{1,1}$ (Paar)',   G_k12, A_k12, MAINBLUE,  'o-',  3.0),
    ('$K_3$ (Dreieck)',    G_k3,  A_k3,  ACCENTRED, 's--', 2.2),
    ('$P_4$ (Pfad-4)',     G_p4,  A_p4,  DARKGREEN, '^:',  2.2),
    ('$E_2$ (leer)',        G_e2,  A_e2,  GRAY,      'D-.', 2.0),
]

char_names = ['Einfachheit\n$S(G)$', 'Harmonie\n$H(G)$',
              'Formalitaet\n$F(G)$', 'Verbindung\n$V(G)$']

N_ax = 4
angles = np.linspace(0, 2 * np.pi, N_ax, endpoint=False).tolist()
angles += angles[:1]

# Radar
ax_radar = fig.add_axes([0.04, 0.08, 0.45, 0.82], polar=True)
for name, G, A, col, ls, lw in radar_graphs:
    S, H, F, V = vier_charakteristika(G, A)
    vals = [S, H, F, V]
    vals += vals[:1]
    ax_radar.plot(angles, vals, ls, color=col, lw=lw, label=name, alpha=0.9)
    ax_radar.fill(angles, vals, alpha=0.07, color=col)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(char_names, fontsize=10)
ax_radar.set_ylim(0, 1.15)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=8)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.50, 1.15), fontsize=9)
ax_radar.grid(True, alpha=0.4)
ax_radar.set_title('(a) Radar: Alle 4 Charakteristika\n$K_{1,1}$ erfullt alle maximal',
                   fontsize=11, fontweight='bold', pad=14)

# Balkendiagramm nebeneinander
ax = axes[1]
char_short = ['$S(G)$', '$H(G)$', '$F(G)$', '$V(G)$']
n_g = len(radar_graphs)
x_g = np.arange(N_ax)
width = 0.18
for i, (name, G, A, col, _, _) in enumerate(radar_graphs):
    S, H, F, V = vier_charakteristika(G, A)
    vals = [S, H, F, V]
    xpos = x_g + (i - n_g/2 + 0.5) * width
    bars = ax.bar(xpos, vals, width, color=col, alpha=0.85,
                  label=name, edgecolor='white', linewidth=0.8)
ax.set_xticks(x_g)
ax.set_xticklabels(char_short, fontsize=12)
ax.set_ylabel('Charakteristikum-Wert $\\in [0,1]$')
ax.set_title('(b) Balkenvergleich aller 4 Charakteristika')
ax.axhline(1.0, color=GRAY, linestyle='--', lw=1.0, alpha=0.5)
ax.set_ylim(0, 1.25)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Abbildung 11: Zusammenfassung – Die vier Charakteristika im Vergleich\n'
             '($K_{1,1}$ ist das einzige Graphelement, das alle vier Maxima gleichzeitig erreicht)',
             fontsize=11, fontweight='bold')
plt.savefig(OUT + 'plot11_charakteristika_radar.pdf', bbox_inches='tight')
plt.close()
print("Plot 11 gespeichert.")

print("\nAlle 5 Plots (7-11) erfolgreich gespeichert.")

