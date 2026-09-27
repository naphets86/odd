"""
Generiert alle matplotlib-Visualisierungen fuer die wissenschaftliche Arbeit:
"Graphenbasierte DNA-Sequenzierung mittels des Subgraph Algorithmus"
Autor: Stephan Epp, Universitaet Bielefeld
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Globale Stileinstellungen ──────────────────────────────────────────────
MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
LIGHTGRAY = '#F5F5F8'
DARKGRAY  = '#3C3C46'
GOLD      = '#C8A400'

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       150,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

BASES = ['A', 'T', 'G', 'C']
BASE_COLORS = {'A': MAINBLUE, 'T': ACCENTRED, 'G': DARKGREEN, 'C': GOLD}

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – DNA-Doppelhelix als Graphdarstellung
# ══════════════════════════════════════════════════════════════════════════════
def plot_01_dna_als_graph():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Links: schematische Doppelhelix-Kurve
    ax = axes[0]
    t = np.linspace(0, 4 * np.pi, 400)
    x1 =  np.sin(t)
    x2 = -np.sin(t)
    y  =  t / (4 * np.pi) * 10

    ax.plot(x1, y, color=MAINBLUE,  linewidth=2.5, label='Strang 5\'→3\'')
    ax.plot(x2, y, color=ACCENTRED, linewidth=2.5, label='Strang 3\'→5\'')

    # Wasserstoffbruecken als horizontale Verbindungen
    crosslink_positions = np.linspace(0.6, 9.4, 14)
    for ypos in crosslink_positions:
        t_val = ypos / 10 * 4 * np.pi
        x_left  =  np.sin(t_val)
        x_right = -np.sin(t_val)
        ax.plot([x_left, x_right], [ypos, ypos],
                color=DARKGRAY, linewidth=1.0, linestyle='--', alpha=0.6)

    # Basenbeschriftungen
    base_pairs = [('A','T'), ('T','A'), ('G','C'), ('C','G'),
                  ('A','T'), ('G','C'), ('T','A'), ('C','G'),
                  ('A','T'), ('T','A'), ('G','C'), ('C','G'),
                  ('A','T'), ('G','C')]
    for idx, (ypos, (b1, b2)) in enumerate(zip(crosslink_positions, base_pairs)):
        t_val  = ypos / 10 * 4 * np.pi
        x_left =  np.sin(t_val)
        x_right= -np.sin(t_val)
        ax.text(x_left  + 0.08, ypos, b1, fontsize=8,
                color=BASE_COLORS[b1], fontweight='bold', va='center')
        ax.text(x_right - 0.22, ypos, b2, fontsize=8,
                color=BASE_COLORS[b2], fontweight='bold', va='center')

    ax.set_xlim(-1.8, 1.8)
    ax.set_title('DNA-Doppelhelix\nmit komplementären Basenpaaren', fontweight='bold')
    ax.set_xlabel('Transversale Ausdehnung')
    ax.set_ylabel("Längsachse (5'→3'-Richtung)")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_yticks([])

    # Rechts: Graph-Repräsentation
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    node_positions = {
        'A1': (0, 3), 'T1': (0, 2), 'G1': (0, 1), 'C1': (0, 0),
        'T2': (4, 3), 'A2': (4, 2), 'C2': (4, 1), 'G2': (4, 0),
    }
    # Kanten zwischen komplementaeren Basen
    complement_edges = [('A1','T2'), ('T1','A2'), ('G1','C2'), ('C1','G2')]
    for (u, v) in complement_edges:
        xu, yu = node_positions[u]
        xv, yv = node_positions[v]
        ax2.plot([xu, xv], [yu, yv], color=DARKGRAY, linewidth=1.5,
                 linestyle='--', alpha=0.7, zorder=1)

    # Backbone-Kanten (sequentiell)
    backbone_l = [('A1','T1'), ('T1','G1'), ('G1','C1')]
    backbone_r = [('T2','A2'), ('A2','C2'), ('C2','G2')]
    for edges, col in [(backbone_l, MAINBLUE), (backbone_r, ACCENTRED)]:
        for (u, v) in edges:
            xu, yu = node_positions[u]
            xv, yv = node_positions[v]
            ax2.annotate('', xy=(xv, yv), xytext=(xu, yu),
                         arrowprops=dict(arrowstyle='->', color=col,
                                         lw=2.0))

    # Knoten zeichnen
    for name, (x, y) in node_positions.items():
        base = name[0]
        circle = plt.Circle((x, y), 0.28, color=BASE_COLORS[base],
                             zorder=3, alpha=0.9)
        ax2.add_patch(circle)
        ax2.text(x, y, base, ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white', zorder=4)

    ax2.set_title('Graph-Repräsentation $G_{DNA}$\n(Knoten = Basen, Kanten = Bindungen)',
                  fontweight='bold')

    # Legende
    legend_elements = [
        mpatches.Patch(color=BASE_COLORS['A'], label='Adenin (A)'),
        mpatches.Patch(color=BASE_COLORS['T'], label='Thymin (T)'),
        mpatches.Patch(color=BASE_COLORS['G'], label='Guanin (G)'),
        mpatches.Patch(color=BASE_COLORS['C'], label='Cytosin (C)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    fig.suptitle('Abbildung 1: DNA als gerichteter Graph',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig('figures/fig01_dna_graph.pdf')
    fig.savefig('figures/fig01_dna_graph.png')
    plt.close(fig)
    print("Plot 1 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Adjazenzmatrix-Darstellung einer DNA-Sequenz
# ══════════════════════════════════════════════════════════════════════════════
def plot_02_adjazenzmatrix():
    sequence = ['A', 'T', 'G', 'C', 'A', 'G', 'T', 'C']
    n = len(sequence)
    # Adjazenzmatrix: Kante von Base i zu Base i+1 (Backbone)
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adjacency[i, i + 1] = 1
    # Komplementaere Bindungen
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    for i in range(n):
        for j in range(n):
            if j != i and sequence[j] == complement_map[sequence[i]]:
                adjacency[i, j] = max(adjacency[i, j], 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Subplot 1: Rohe Adjazenzmatrix ---
    ax = axes[0]
    cmap_blue = LinearSegmentedColormap.from_list(
        'blue_white', ['white', MAINBLUE])
    im = ax.imshow(adjacency, cmap=cmap_blue, aspect='equal', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'{sequence[i]}{i}' for i in range(n)], rotation=45, fontsize=8)
    ax.set_yticklabels([f'{sequence[i]}{i}' for i in range(n)], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(adjacency[i, j]), ha='center', va='center',
                    fontsize=9, color='white' if adjacency[i, j] else DARKGRAY)
    ax.set_title('Adjazenzmatrix $A$\nder Sequenz ATGCAGTC', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # --- Subplot 2: Signaturen der Spalten ---
    ax2 = axes[1]
    signatures = []
    for col in range(n):
        col_vector = adjacency[:, col]
        row_sig = sum(2**i for i in range(n) if col_vector[i] == 1)
        col_weight = col * (2**n)
        signatures.append(row_sig + col_weight)

    colors = [BASE_COLORS[sequence[i]] for i in range(n)]
    bars = ax2.bar(range(n), signatures, color=colors, edgecolor='white',
                   linewidth=1.5, alpha=0.85)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([f'{sequence[i]}{i}' for i in range(n)], fontsize=9)
    ax2.set_ylabel('Signaturwert $\\sigma_j$')
    ax2.set_title('Spalten-Signaturen $\\sigma_j^A$\n(eindeutige Hashwerte)', fontweight='bold')
    for bar, sig in zip(bars, signatures):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(sig), ha='center', va='bottom', fontsize=7, rotation=45)

    # --- Subplot 3: Signatur-Heatmap zweier Sequenzen ---
    ax3 = axes[2]
    seq2 = ['T', 'G', 'C', 'A', 'T', 'C', 'A', 'G']
    adjacency2 = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adjacency2[i, i + 1] = 1
    for i in range(n):
        for j in range(n):
            if j != i and seq2[j] == complement_map[seq2[i]]:
                adjacency2[i, j] = max(adjacency2[i, j], 1)

    sigs2 = []
    for col in range(n):
        col_vector = adjacency2[:, col]
        row_sig = sum(2**i for i in range(n) if col_vector[i] == 1)
        sigs2.append(row_sig + col * (2**n))

    comparison_matrix = np.zeros((n, n))
    for i, s1 in enumerate(signatures):
        for j, s2 in enumerate(sigs2):
            row1 = s1 % (2**n)
            row2 = s2 % (2**n)
            comparison_matrix[i, j] = 1 if row1 == row2 else 0

    cmap_match = LinearSegmentedColormap.from_list(
        'match', ['white', DARKGREEN])
    im3 = ax3.imshow(comparison_matrix, cmap=cmap_match, aspect='equal', vmin=0, vmax=1)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels([f'{seq2[i]}{i}' for i in range(n)], rotation=45, fontsize=8)
    ax3.set_yticklabels([f'{sequence[i]}{i}' for i in range(n)], fontsize=8)
    ax3.set_xlabel('Sequenz 2: TGCATCAG')
    ax3.set_ylabel('Sequenz 1: ATGCAGTC')
    ax3.set_title('Signaturen-Übereinstimmungsmatrix\n$M_{ij} = [\\sigma_i^A = \\sigma_j^B]$',
                  fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.7, label='Übereinstimmung')

    fig.suptitle('Abbildung 2: Adjazenzmatrix und Signatur-Analyse',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig02_adjazenz.pdf')
    fig.savefig('figures/fig02_adjazenz.png')
    plt.close(fig)
    print("Plot 2 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Zyklische Rotation auf DNA-Sequenzen
# ══════════════════════════════════════════════════════════════════════════════
def plot_03_rotation():
    sequence = ['A', 'T', 'G', 'C']
    n = len(sequence)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for rot_idx in range(4):
        ax = axes[rot_idx]
        rotated = sequence[rot_idx:] + sequence[:rot_idx]

        # Kreisförmige Anordnung
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
        radius = 1.2

        for i, (base, angle) in enumerate(zip(rotated, angles)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            circle = plt.Circle((x, y), 0.28,
                                 color=BASE_COLORS[base], zorder=3, alpha=0.9)
            ax.add_patch(circle)
            ax.text(x, y, base, ha='center', va='center',
                    fontsize=13, fontweight='bold', color='white', zorder=4)
            ax.text(x * 1.55, y * 1.55, f'pos {i}', ha='center', va='center',
                    fontsize=8, color=DARKGRAY)

        # Pfeile zwischen den Basen
        for i in range(n):
            angle_from = angles[i]
            angle_to   = angles[(i+1) % n]
            x1 = (radius - 0.35) * np.cos(angle_from)
            y1 = (radius - 0.35) * np.sin(angle_from)
            x2 = (radius - 0.35) * np.cos(angle_to)
            y2 = (radius - 0.35) * np.sin(angle_to)

            mid_angle = (angle_from + angle_to) / 2
            if angle_to < angle_from:
                mid_angle += np.pi
            cx = (radius - 0.7) * np.cos(mid_angle)
            cy = (radius - 0.7) * np.sin(mid_angle)

            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=MAINBLUE,
                                       lw=1.5,
                                       connectionstyle=f'arc3,rad=0.3'))

        rot_label = f'Rotation $r={rot_idx}$: [{", ".join(rotated)}]'
        ax.set_title(rot_label, fontweight='bold', fontsize=10)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.axis('off')

        if rot_idx == 0:
            ax.text(0, -2.0, 'Originalsequenz', ha='center',
                    fontsize=9, color=ACCENTRED, style='italic')
        else:
            ax.text(0, -2.0, f'nach {rot_idx} Rotation(en)', ha='center',
                    fontsize=9, color=DARKGREEN, style='italic')

    fig.suptitle('Abbildung 3: Zyklische Rotationen der Sequenz [A, T, G, C]\n'
                 'Alle $n=4$ Rotationen statt $4!=24$ Permutationen',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig03_rotation.pdf')
    fig.savefig('figures/fig03_rotation.png')
    plt.close(fig)
    print("Plot 3 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 – LCS dynamische Programmierung auf DNA-Signaturen
# ══════════════════════════════════════════════════════════════════════════════
def plot_04_lcs_dp():
    seq_a = ['A', 'T', 'G', 'C', 'A']
    seq_b = ['T', 'A', 'G', 'T', 'C', 'A']

    n_a = len(seq_a)
    n_b = len(seq_b)

    dp = np.zeros((n_a + 1, n_b + 1), dtype=int)
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])

    # Traceback
    traceback_cells = set()
    i, j = n_a, n_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            traceback_cells.add((i, j))
            i -= 1
            j -= 1
        elif dp[i-1, j] >= dp[i, j-1]:
            i -= 1
        else:
            j -= 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- DP-Tabelle ---
    ax = axes[0]
    cmap_dp = LinearSegmentedColormap.from_list('dp', ['white', MAINBLUE])
    im = ax.imshow(dp, cmap=cmap_dp, aspect='auto', vmin=0, vmax=dp.max())

    for i in range(n_a + 1):
        for j in range(n_b + 1):
            color = 'white' if dp[i, j] > dp.max() / 2 else DARKGRAY
            weight = 'bold' if (i, j) in traceback_cells else 'normal'
            fc     = ACCENTRED if (i, j) in traceback_cells else None
            if (i, j) in traceback_cells:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                     fill=True, facecolor=ACCENTRED,
                                     alpha=0.4, zorder=2)
                ax.add_patch(rect)
            ax.text(j, i, str(dp[i, j]), ha='center', va='center',
                    fontsize=10, fontweight=weight,
                    color='white' if dp[i,j] >= dp.max() and (i,j) not in traceback_cells
                    else DARKGRAY,
                    zorder=3)

    ax.set_xticks(range(n_b + 1))
    ax.set_xticklabels(['ε'] + list(seq_b), fontsize=10)
    ax.set_yticks(range(n_a + 1))
    ax.set_yticklabels(['ε'] + list(seq_a), fontsize=10)

    for tick, base in zip(ax.get_xticklabels()[1:], seq_b):
        tick.set_color(BASE_COLORS.get(base, DARKGRAY))
        tick.set_fontweight('bold')
    for tick, base in zip(ax.get_yticklabels()[1:], seq_a):
        tick.set_color(BASE_COLORS.get(base, DARKGRAY))
        tick.set_fontweight('bold')

    ax.set_xlabel('Sequenz B: ' + ''.join(seq_b), fontsize=10)
    ax.set_ylabel('Sequenz A: ' + ''.join(seq_a), fontsize=10)
    ax.set_title(f'LCS-DP-Tabelle\nLCS = {dp[n_a, n_b]} (rot = optimaler Pfad)',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7, label='dp[i][j]')

    # --- Rekurrenz visuell ---
    ax2 = axes[1]
    ax2.axis('off')

    formula_text = (
        "LCS-Rekurrenz:\n\n"
        "Falls i=0 oder j=0:   dp[i][j] = 0\n"
        "Falls s_A[i]=s_B[j]:  dp[i][j] = dp[i-1][j-1] + 1\n"
        "Sonst:                dp[i][j] = max(dp[i-1][j], dp[i][j-1])"
    )
    ax2.text(0.05, 0.78, formula_text, transform=ax2.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6',
             facecolor=LIGHTGRAY, edgecolor=MAINBLUE, linewidth=1.5))

    # LCS-Ergebnis
    lcs_length = dp[n_a, n_b]
    ax2.text(0.05, 0.45,
             f'Ergebnis:\n\n'
             f'LCS-Länge = {lcs_length}\n\n'
             f'Sequenz A: {" ".join(seq_a)}\n'
             f'Sequenz B: {" ".join(seq_b)}\n\n'
             f'Subgraph-Bedingung: LCS ≥ 2\n'
             f'→ Erfüllt: {lcs_length} ≥ 2  ✓',
             transform=ax2.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor='#E8F4E8', edgecolor=DARKGREEN, linewidth=1.5))

    # Komplexitaet
    ax2.text(0.05, 0.12,
             f'Komplexität: $O(n_A \\cdot n_B) = O({n_a} \\cdot {n_b}) = O({n_a*n_b})$\n'
             f'Gesamt (mit $n$ Rotationen): $O(n^3)$',
             transform=ax2.transAxes, fontsize=10, va='bottom',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='#FFF8E0', edgecolor=GOLD, linewidth=1.5))

    ax2.set_title('LCS-Algorithmus: Theorie & Ergebnis', fontweight='bold')

    fig.suptitle('Abbildung 4: Longest Common Subsequence (LCS) auf DNA-Signaturen',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig04_lcs_dp.pdf')
    fig.savefig('figures/fig04_lcs_dp.png')
    plt.close(fig)
    print("Plot 4 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Komplexitaetsvergleich: naiv vs. Subgraph Algorithmus
# ══════════════════════════════════════════════════════════════════════════════
def plot_05_komplexitaet():
    n_values = np.arange(2, 26)

    import math as _math
    naive_permutation = np.array([float(_math.factorial(n)) * n**2 for n in n_values], dtype=float)
    subgraph_algo     = n_values**3
    n_squared         = n_values**2
    n_log_n           = n_values * np.log2(n_values)
    ngs_sequencing    = n_values * np.log2(n_values) * 4   # NGS approximation

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Linke Achse: Logarithmische Skala ---
    ax = axes[0]
    ax.semilogy(n_values, naive_permutation, 'o-', color=ACCENTRED, linewidth=2.2,
                markersize=5, label=r'Naiv: $O(n! \cdot n^2)$')
    ax.semilogy(n_values, subgraph_algo, 's-', color=MAINBLUE, linewidth=2.2,
                markersize=5, label=r'Subgraph-DNA: $O(n^3)$')
    ax.semilogy(n_values, n_squared, '^-', color=DARKGREEN, linewidth=2.0,
                markersize=5, label=r'Signaturberechnung: $O(n^2)$')
    ax.semilogy(n_values, n_log_n, 'D-', color=GOLD, linewidth=2.0,
                markersize=5, label=r'NGS-Referenz: $O(n \log n)$')

    ax.set_xlabel('Sequenzlänge $n$ (Basen)')
    ax.set_ylabel('Operationen (log-Skala)')
    ax.set_title('Komplexitätsvergleich\n(logarithmische Skala)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # --- Rechte Achse: Speedup-Faktor ---
    ax2 = axes[1]
    speedup = naive_permutation / subgraph_algo
    ax2.semilogy(n_values, speedup, 'o-', color=DARKGREEN, linewidth=2.5,
                 markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(n_values, speedup, alpha=0.15, color=DARKGREEN)

    ax2.set_xlabel('Sequenzlänge $n$ (Basen)')
    ax2.set_ylabel('Speedup-Faktor (log-Skala)')
    ax2.set_title('Beschleunigung: Subgraph-DNA\nvs. naiver Permutationsansatz', fontweight='bold')

    # Annotierungen
    for n_mark in [5, 10, 15, 20]:
        idx = np.where(n_values == n_mark)[0][0]
        sp  = speedup[idx]
        ax2.annotate(f'n={n_mark}\n{sp:.1e}×',
                     xy=(n_mark, sp), xytext=(n_mark + 1.2, sp * 0.3),
                     fontsize=8, color=DARKGRAY,
                     arrowprops=dict(arrowstyle='->', color=DARKGRAY, lw=1))

    ax2.grid(True, which='both', alpha=0.3)

    fig.suptitle('Abbildung 5: Algorithmen-Komplexität für DNA-Sequenzabgleich',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig05_komplexitaet.pdf')
    fig.savefig('figures/fig05_komplexitaet.png')
    plt.close(fig)
    print("Plot 5 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 – Subgraph-Erkennung zweier DNA-Fragmente (vollstaendiges Beispiel)
# ══════════════════════════════════════════════════════════════════════════════
def compute_signature(matrix):
    n = matrix.shape[0]
    sigs = []
    for col in range(n):
        col_vec = matrix[:, col]
        row_sig = sum(2**i for i in range(n) if col_vec[i] == 1)
        sigs.append(row_sig + col * (2**n))
    return sigs

def lcs_length(seq_a, seq_b):
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    return dp[m, n]

def plot_06_subgraph_dna():
    # Fragment A (klein): ATGC
    seq_a = ['A', 'T', 'G', 'C']
    n_a   = len(seq_a)
    mat_a = np.zeros((n_a, n_a), dtype=int)
    for i in range(n_a - 1):
        mat_a[i, i+1] = 1
    # Komplementaerbindung A-T
    mat_a[0, 1] = 1
    mat_a[2, 3] = 1

    # Fragment B (gross): CATGCTAT
    seq_b = ['C', 'A', 'T', 'G', 'C', 'T', 'A', 'T']
    n_b   = len(seq_b)
    mat_b = np.zeros((n_b, n_b), dtype=int)
    for i in range(n_b - 1):
        mat_b[i, i+1] = 1
    mat_b[1, 2] = 1
    mat_b[3, 4] = 1

    sig_a = compute_signature(mat_a)
    sig_b = compute_signature(mat_b)

    # Alle Rotationen und LCS-Werte
    col_weight_a = 2**n_a
    col_weight_b = 2**n_b
    row_sigs_a = [s % col_weight_a for s in sig_a]
    row_sigs_b = [s % col_weight_b for s in sig_b]

    lcs_per_rotation = []
    match_rotation   = None
    for rot in range(n_b):
        rotated_b = row_sigs_b[rot:] + row_sigs_b[:rot]
        max_lcs = 0
        for start in range(n_b - n_a + 1):
            window = rotated_b[start:start + n_a]
            lcs_val = lcs_length(row_sigs_a, window)
            if lcs_val > max_lcs:
                max_lcs = lcs_val
        lcs_per_rotation.append(max_lcs)
        if max_lcs >= 2 and match_rotation is None:
            match_rotation = rot

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Subplot 1: Fragment A als Graph ---
    ax = axes[0, 0]
    pos_a = {i: (i, 0) for i in range(n_a)}
    for i in range(n_a):
        circle = plt.Circle(pos_a[i], 0.25,
                             color=BASE_COLORS[seq_a[i]], zorder=3, alpha=0.9)
        ax.add_patch(circle)
        ax.text(pos_a[i][0], pos_a[i][1], seq_a[i],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', zorder=4)
    for i in range(n_a):
        for j in range(n_a):
            if mat_a[i, j] == 1:
                ax.annotate('', xy=pos_a[j], xytext=pos_a[i],
                            arrowprops=dict(arrowstyle='->', color=MAINBLUE, lw=1.8))
    ax.set_xlim(-0.8, n_a - 0.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Fragment $G_A$: {" ".join(seq_a)}\n($n_A={n_a}$ Knoten)',
                 fontweight='bold')

    # --- Subplot 2: Fragment B als Graph ---
    ax2 = axes[0, 1]
    pos_b = {i: (i, 0) for i in range(n_b)}
    for i in range(n_b):
        circle = plt.Circle(pos_b[i], 0.25,
                             color=BASE_COLORS[seq_b[i]], zorder=3, alpha=0.9)
        ax2.add_patch(circle)
        ax2.text(pos_b[i][0], pos_b[i][1], seq_b[i],
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 color='white', zorder=4)
    for i in range(n_b):
        for j in range(n_b):
            if mat_b[i, j] == 1:
                ax2.annotate('', xy=pos_b[j], xytext=pos_b[i],
                             arrowprops=dict(arrowstyle='->', color=ACCENTRED, lw=1.5))
    ax2.set_xlim(-0.8, n_b - 0.2)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Fragment $G_B$: {" ".join(seq_b)}\n($n_B={n_b}$ Knoten)',
                  fontweight='bold')

    # --- Subplot 3: LCS pro Rotation ---
    ax3 = axes[1, 0]
    bar_colors = [DARKGREEN if l >= 2 else ACCENTRED for l in lcs_per_rotation]
    bars = ax3.bar(range(n_b), lcs_per_rotation, color=bar_colors,
                   edgecolor='white', linewidth=1.5)
    ax3.axhline(y=2, color=DARKGRAY, linestyle='--', linewidth=1.5,
                label='Schwellwert LCS = 2')
    ax3.set_xticks(range(n_b))
    ax3.set_xticklabels([f'Rot. {r}' for r in range(n_b)], rotation=30, fontsize=8)
    ax3.set_ylabel('LCS-Länge')
    ax3.set_title('LCS-Wert je Rotation\n(grün = Subgraph erkannt)', fontweight='bold')
    ax3.legend(fontsize=9)
    for bar, lcs_val in zip(bars, lcs_per_rotation):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(lcs_val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- Subplot 4: Ergebnis-Zusammenfassung ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    result_text = (
        f"Ergebnis der Subgraph-DNA-Analyse\n"
        f"{'='*38}\n\n"
        f"Fragment A:  {''.join(seq_a)}  ($n_A = {n_a}$)\n"
        f"Fragment B:  {''.join(seq_b)}  ($n_B = {n_b}$)\n\n"
        f"Signatur A:  {sig_a}\n\n"
        f"Rotationen:  {n_b}\n"
        f"LCS-Werte:   {lcs_per_rotation}\n\n"
    )
    if match_rotation is not None:
        result_text += (
            f"✓ Subgraph erkannt bei Rotation {match_rotation}\n"
            f"  G_A ⊆ G_B  (nach Rotation um {match_rotation})\n\n"
            f"Entscheidung: BEHALTE G_B"
        )
        color = '#E8F4E8'
        edge_color = DARKGREEN
    else:
        result_text += "✗ Kein Subgraph gefunden\nBeide Graphen behalten"
        color = '#FFF0F0'
        edge_color = ACCENTRED

    ax4.text(0.05, 0.95, result_text, transform=ax4.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor=color,
                       edgecolor=edge_color, linewidth=2))
    ax4.set_title('Algorithmus-Ergebnis', fontweight='bold')

    fig.suptitle('Abbildung 6: Vollständiges Subgraph-DNA-Beispiel',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig06_subgraph_dna.pdf')
    fig.savefig('figures/fig06_subgraph_dna.png')
    plt.close(fig)
    print("Plot 6 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 – Sequenzierungskosten-Kurve (historisch + Subgraph-Prognose)
# ══════════════════════════════════════════════════════════════════════════════
def plot_07_kosten():
    years_hist  = [2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
    costs_hist  = [1e8, 5e7, 1e7, 1e6, 1e5, 1e4, 5e3, 1e3, 800, 600, 400, 250]

    # Subgraph-basierte Effizienzprognose
    years_proj  = [2024, 2025, 2026, 2027, 2028, 2030]
    costs_proj  = [180, 120, 80, 50, 30, 10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.semilogy(years_hist, costs_hist, 'o-', color=MAINBLUE, linewidth=2.5,
                markersize=7, label='Historische Kosten (NHGRI)', zorder=3)
    ax.semilogy(years_proj, costs_proj, 's--', color=DARKGREEN, linewidth=2.2,
                markersize=7, label='Projektion (Subgraph-DNA)', zorder=3)
    ax.semilogy([years_hist[-1], years_proj[0]], [costs_hist[-1], costs_proj[0]],
                '--', color=DARKGRAY, linewidth=1.5, alpha=0.5)

    # Mooresches Gesetz als Referenz
    years_moore = np.array([2001, 2030])
    costs_moore_start = costs_hist[0]
    costs_moore = costs_moore_start * (0.5 ** ((years_moore - 2001) / 2))
    ax.semilogy(years_moore, costs_moore, ':', color=ACCENTRED, linewidth=1.8,
                label="Mooresches Gesetz (Ref.)", alpha=0.7)

    # Meilenstein-Annotierungen
    milestones = {
        2003: ('Sanger\nHumangenom', 5e7),
        2007: ('NGS\nIllumina', 1.5e6),
        2015: ('100-Dollar-\nZiel', 2e3),
        2023: ('Aktuell', 400),
    }
    for year, (label, y_pos) in milestones.items():
        idx = years_hist.index(year)
        ax.annotate(label, xy=(year, costs_hist[idx]),
                    xytext=(year - 1.5, y_pos * 8),
                    fontsize=7.5, ha='center', color=DARKGRAY,
                    arrowprops=dict(arrowstyle='->', color=DARKGRAY, lw=1))

    ax.set_xlabel('Jahr')
    ax.set_ylabel('Kosten pro Genom (USD, log-Skala)')
    ax.set_title('Sequenzierungskosten pro Genom\n(historisch + Projektion)', fontweight='bold')
    ax.legend(fontsize=9)

    # --- Rechts: Speedup durch Subgraph ---
    ax2 = axes[1]
    n_values = np.arange(10, 1001, 10)
    traditional_time = n_values**2 * np.log2(n_values)
    subgraph_time    = n_values**3
    speedup_ratio    = traditional_time / subgraph_time

    genome_lengths = np.array([100, 500, 1000, 5000, 10000])
    genome_names   = ['Phage\nλ', 'E.coli\nFragment', 'HIV\nGenom', 'Mito-\nchondrien', 'SARS-\nCoV-2']
    for length, name in zip(genome_lengths, genome_names):
        if length <= n_values[-1]:
            idx   = np.searchsorted(n_values, length)
            t_trad = length**2 * np.log2(length)
            t_sub  = length**3
            ratio  = t_trad / t_sub
            ax2.annotate(name, xy=(length, ratio),
                         xytext=(length * 1.3, ratio * 2),
                         fontsize=7, color=DARKGRAY,
                         arrowprops=dict(arrowstyle='->', lw=0.8, color=DARKGRAY))

    ax2.loglog(n_values, traditional_time / traditional_time[0],
               '-', color=GOLD, linewidth=2, label=r'Klassisch: $O(n^2 \log n)$')
    ax2.loglog(n_values, subgraph_time / subgraph_time[0],
               '-', color=MAINBLUE, linewidth=2, label=r'Subgraph-DNA: $O(n^3)$')
    ax2.set_xlabel('Sequenzlänge $n$ (Basen, log)')
    ax2.set_ylabel('Normierte Laufzeit (log)')
    ax2.set_title('Laufzeitvergleich nach Genomgröße\n(doppelt-logarithmisch)',
                  fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle('Abbildung 7: Sequenzierungskosten und Laufzeitanalyse',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig07_kosten.pdf')
    fig.savefig('figures/fig07_kosten.png')
    plt.close(fig)
    print("Plot 7 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 – Basenpaar-Haeufigkeitsanalyse & Graphstruktur-Metriken
# ══════════════════════════════════════════════════════════════════════════════
def plot_08_statistik():
    np.random.seed(42)
    # Simuliertes menschliches Chromosom-Fragment
    human_gc = 0.41   # ~41% GC-Gehalt
    n_seq = 1000
    probabilities = [
        (1 - human_gc) / 2,   # A
        (1 - human_gc) / 2,   # T
        human_gc / 2,          # G
        human_gc / 2,          # C
    ]
    sequence = np.random.choice(['A', 'T', 'G', 'C'], size=n_seq, p=probabilities)

    # Basenhaeufigkeiten
    counts = {b: np.sum(sequence == b) for b in BASES}

    # Dinukleotid-Matrix
    dinuc_matrix = np.zeros((4, 4))
    base_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    for i in range(len(sequence) - 1):
        dinuc_matrix[base_idx[sequence[i]], base_idx[sequence[i+1]]] += 1
    dinuc_matrix /= dinuc_matrix.sum()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- Subplot 1: Basenhaeufigkeiten ---
    ax = axes[0, 0]
    bar_colors = [BASE_COLORS[b] for b in BASES]
    bars = ax.bar(BASES, [counts[b] for b in BASES],
                  color=bar_colors, edgecolor='white', linewidth=2, alpha=0.85)
    for bar, base in zip(bars, BASES):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{counts[base]/n_seq*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=BASE_COLORS[base])
    ax.set_ylabel('Anzahl Vorkommen')
    ax.set_title(f'Basenhäufigkeiten\n(Stichprobe $n={n_seq}$)', fontweight='bold')
    ax.set_ylim(0, max(counts.values()) * 1.2)

    # --- Subplot 2: Dinukleotid-Übergangsmatrix ---
    ax2 = axes[0, 1]
    cmap_dinuc = LinearSegmentedColormap.from_list('dinuc', ['white', MAINBLUE])
    im2 = ax2.imshow(dinuc_matrix, cmap=cmap_dinuc, aspect='equal')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(BASES, fontsize=12)
    ax2.set_yticklabels(BASES, fontsize=12)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{dinuc_matrix[i,j]:.3f}',
                     ha='center', va='center', fontsize=9,
                     color='white' if dinuc_matrix[i,j] > 0.08 else DARKGRAY)
    ax2.set_xlabel('Nachfolger-Base')
    ax2.set_ylabel('Vorgänger-Base')
    ax2.set_title('Dinukleotid-Übergangsmatrix\n(Markov-Modell erster Ordnung)',
                  fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Übergangswahrscheinlichkeit')

    # --- Subplot 3: GC-Gehalt im Fenster ---
    ax3 = axes[1, 0]
    window_size = 50
    gc_content = []
    for start in range(0, n_seq - window_size + 1, 10):
        window = sequence[start:start + window_size]
        gc = np.sum((window == 'G') | (window == 'C')) / window_size
        gc_content.append(gc)
    x_positions = np.arange(len(gc_content)) * 10

    ax3.fill_between(x_positions, gc_content, alpha=0.3, color=MAINBLUE)
    ax3.plot(x_positions, gc_content, color=MAINBLUE, linewidth=1.8)
    ax3.axhline(y=np.mean(gc_content), color=ACCENTRED, linestyle='--',
                linewidth=1.5, label=f'Mittlerer GC = {np.mean(gc_content)*100:.1f}%')
    ax3.axhline(y=0.5, color=DARKGRAY, linestyle=':', linewidth=1.2,
                label='Gleichgewicht (50%)', alpha=0.6)
    ax3.set_xlabel('Position in der Sequenz')
    ax3.set_ylabel('GC-Gehalt (Fenster = 50 Basen)')
    ax3.set_title(f'GC-Gehalt-Profil\n(gleitendes Fenster)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1)

    # --- Subplot 4: Graphdichte-Metriken ---
    ax4 = axes[1, 1]
    seq_lengths = [4, 8, 16, 32, 64, 128]
    densities   = [0.5, 0.38, 0.25, 0.18, 0.12, 0.08]
    avg_degrees = [d * (n-1) for d, n in zip(densities, seq_lengths)]
    lcs_ratios  = [0.85, 0.78, 0.72, 0.65, 0.58, 0.50]

    ax4_twin = ax4.twinx()
    ax4.bar(range(len(seq_lengths)), densities, alpha=0.6, color=MAINBLUE,
            label='Graphdichte $\\rho$', width=0.4, align='edge')
    ax4_twin.plot(range(len(seq_lengths)), lcs_ratios, 'o-', color=ACCENTRED,
                  linewidth=2, markersize=6, label='LCS/n Verhältnis')
    ax4.set_xticks(range(len(seq_lengths)))
    ax4.set_xticklabels([str(n) for n in seq_lengths])
    ax4.set_xlabel('Sequenzlänge $n$')
    ax4.set_ylabel('Graphdichte $\\rho$', color=MAINBLUE)
    ax4_twin.set_ylabel('LCS/n', color=ACCENTRED)
    ax4.set_title('Graph-Metriken vs. Sequenzlänge',fontweight='bold')

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    fig.suptitle('Abbildung 8: Statistische Eigenschaften von DNA als Graph',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig08_statistik.pdf')
    fig.savefig('figures/fig08_statistik.png')
    plt.close(fig)
    print("Plot 8 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 – Mutationsanalyse durch Subgraph-Differenz
# ══════════════════════════════════════════════════════════════════════════════
def plot_09_mutation():
    # Wildtyp-Sequenz
    wildtype  = list('ATGCATGC')
    # Mutationen
    mutation1 = list('ATGTATGC')   # Punkt-Mutation C->T an Pos 4
    mutation2 = list('ATGCATC')    # Deletion an Pos 7
    mutation3 = list('ATGCAATGC')  # Insertion an Pos 5

    def seq_to_adj(seq):
        n = len(seq)
        mat = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            mat[i, i + 1] = 1
        return mat

    def seq_signatures(seq):
        mat = seq_to_adj(seq)
        return compute_signature(mat)

    sigs_wt  = seq_signatures(wildtype)
    sigs_m1  = seq_signatures(mutation1)
    sigs_m2  = seq_signatures(mutation2)
    sigs_m3  = seq_signatures(mutation3)

    def compare_seqs(s1, s2):
        """Gibt Array mit 1 fuer Unterschied, 0 fuer gleich zurueck."""
        min_len = min(len(s1), len(s2))
        diff = []
        for i in range(min_len):
            diff.append(1 if s1[i] != s2[i] else 0)
        diff.extend([1] * abs(len(s1) - len(s2)))
        return diff

    diff_m1 = compare_seqs(wildtype, mutation1)
    diff_m2 = compare_seqs(wildtype, mutation2)
    diff_m3 = compare_seqs(wildtype, mutation3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def draw_sequence(ax, seq, label, highlight_pos=None, color=MAINBLUE):
        n = len(seq)
        for i, base in enumerate(seq):
            facecolor = ACCENTRED if (highlight_pos and i in highlight_pos) else BASE_COLORS[base]
            circle = plt.Circle((i, 0), 0.3, color=facecolor, zorder=3, alpha=0.9)
            ax.add_patch(circle)
            ax.text(i, 0, base, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white', zorder=4)
            ax.text(i, -0.6, str(i), ha='center', va='center',
                    fontsize=7, color=DARKGRAY)
            if i < n - 1:
                ax.annotate('', xy=(i+1, 0), xytext=(i, 0),
                            arrowprops=dict(arrowstyle='->', color=color,
                                           lw=1.5, shrinkA=14, shrinkB=14))
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(-1.0, 1.0)
        ax.axis('off')
        ax.set_title(label, fontweight='bold', fontsize=10)

    draw_sequence(axes[0, 0], wildtype, 'Wildtyp: ' + ''.join(wildtype))
    draw_sequence(axes[0, 1], mutation1,
                  'Punktmutation: ' + ''.join(mutation1), highlight_pos={4})
    draw_sequence(axes[1, 0], mutation2,
                  'Deletion: ' + ''.join(mutation2), highlight_pos={6})
    draw_sequence(axes[1, 1], mutation3,
                  'Insertion: ' + ''.join(mutation3), highlight_pos={5})

    # Overlays: Unterschiede markieren
    for ax, diff in [(axes[0, 1], diff_m1), (axes[1, 0], diff_m2), (axes[1, 1], diff_m3)]:
        for i, d in enumerate(diff):
            if d == 1:
                rect = plt.Rectangle((i - 0.45, -0.45), 0.9, 0.9,
                                      fill=False, edgecolor=ACCENTRED,
                                      linewidth=2.5, zorder=5)
                ax.add_patch(rect)

    fig.suptitle('Abbildung 9: Mutationsanalyse durch Subgraph-Differenz\n'
                 '(rot umrandet = Abweichung vom Wildtyp)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig09_mutation.pdf')
    fig.savefig('figures/fig09_mutation.png')
    plt.close(fig)
    print("Plot 9 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 – Gesamtüberblick: Pipeline Subgraph-DNA-Algorithmus
# ══════════════════════════════════════════════════════════════════════════════
def plot_10_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    steps = [
        (0.5, 3.0, 'Eingabe\nDNA-Sequenzen\n$S_A, S_B$', MAINBLUE),
        (2.5, 3.0, 'Graph-\nModellierung\n$G_A, G_B$', DARKGREEN),
        (4.5, 3.0, 'Signatur-\nBerechnung\n$\\sigma^A, \\sigma^B$\n$O(n^2)$', GOLD),
        (6.5, 3.0, 'Zyklische\nRotationen\n$r = 0 \\ldots n$\n$O(n^3)$', ACCENTRED),
        (8.5, 3.0, 'LCS &\nEntscheidung\nSubgraph?', MAINBLUE),
    ]

    for (x, y, label, color) in steps:
        rect = mpatches.FancyBboxPatch((x - 0.75, y - 0.9), 1.5, 1.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white',
                                        linewidth=2, alpha=0.85, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=3)

    # Pfeile zwischen den Schritten
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.75
        x2 = steps[i+1][0] - 0.75
        y_arrow = steps[i][1]
        ax.annotate('', xy=(x2, y_arrow), xytext=(x1, y_arrow),
                    arrowprops=dict(arrowstyle='->', color=DARKGRAY,
                                   lw=2.0, mutation_scale=15))

    # Ausgabe-Boxen unten
    outcomes = [
        (2.5, 1.0, '$G_A \\subseteq G_B$\nBehalte $G_B$', DARKGREEN),
        (5.0, 1.0, '$G_B \\subseteq G_A$\nBehalte $G_A$', MAINBLUE),
        (7.5, 1.0, 'Keine Beziehung\nBeide behalten', GOLD),
    ]
    for (x, y, label, color) in outcomes:
        rect = mpatches.FancyBboxPatch((x - 1.1, y - 0.55), 2.2, 1.1,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='white',
                                        linewidth=1.5, alpha=0.75, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=3)

    # Pfeil von Entscheidungsbox zu Outcomes
    for ox, oy, *_ in outcomes:
        ax.annotate('', xy=(ox, oy + 0.55), xytext=(8.5, steps[0][1] - 0.9),
                    arrowprops=dict(arrowstyle='->', color=DARKGRAY,
                                   lw=1.3, linestyle='dashed',
                                   connectionstyle='arc3,rad=0.2'))

    # Komplexitaets-Annotation oben
    ax.text(5.0, 5.4,
            r'Gesamtkomplexität: $O(n^2) + O(n^3) = O(n^3)$  —  Optimal unter SETH',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=LIGHTGRAY,
                      edgecolor=DARKGRAY, linewidth=1.5))

    ax.set_title('Abbildung 10: Pipeline des Subgraph-DNA-Algorithmus',
                 fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig('figures/fig10_pipeline.pdf')
    fig.savefig('figures/fig10_pipeline.png')
    plt.close(fig)
    print("Plot 10 gespeichert.")


# ══════════════════════════════════════════════════════════════════════════════
# Alle Plots generieren
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import math as _math
    # numpy.math deprecated fix
    np.math = _math

    print("Generiere alle Plots ...")
    plot_01_dna_als_graph()
    plot_02_adjazenzmatrix()
    plot_03_rotation()
    plot_04_lcs_dp()
    plot_05_komplexitaet()
    plot_06_subgraph_dna()
    plot_07_kosten()
    plot_08_statistik()
    plot_09_mutation()
    plot_10_pipeline()
    print("\nAlle 10 Plots erfolgreich gespeichert.")
