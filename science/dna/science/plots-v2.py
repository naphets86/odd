"""
Neue Plots fuer das erweiterte Kapitel:
  - Metagenomik (Abb. 11-15)
  - Erweiterter Ausblick (Abb. 16)
Autor: Stephan Epp, Universitaet Bielefeld
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings, math
warnings.filterwarnings('ignore')

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
LIGHTGRAY = '#F5F5F8'
DARKGRAY  = '#3C3C46'
GOLD      = '#C8A400'
PURPLE    = '#6A1E8C'
TEAL      = '#0E6B6B'

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

BASE_COLORS = {'A': MAINBLUE, 'T': ACCENTRED, 'G': DARKGREEN, 'C': GOLD}

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────
def compute_signature(matrix):
    n = matrix.shape[0]
    sigs = []
    for col in range(n):
        col_vec = matrix[:, col]
        row_sig = sum(2**i for i in range(n) if col_vec[i] == 1)
        sigs.append(row_sig + col * (2**n))
    return sigs

def lcs_len(seq_a, seq_b):
    m, n = len(seq_a), len(seq_b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 11 – Metagenomik-Ueberblick: Reads aus mehreren Organismen
# ─────────────────────────────────────────────────────────────────────────────
def plot_11_meta_ueberblick():
    np.random.seed(7)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Links: schematische Probe mit Reads ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Umweltprobe als Ellipse
    probe_ellipse = mpatches.Ellipse((5, 4), 8, 5.5,
                                      facecolor='#EAF4FF', edgecolor=MAINBLUE,
                                      linewidth=2, zorder=1)
    ax.add_patch(probe_ellipse)
    ax.text(5, 7.1, 'Umweltprobe (z.\,B. Meerwasser, Boden)', ha='center',
            fontsize=10, fontweight='bold', color=MAINBLUE)

    # Organismen als farbige Kreise
    organisms = [
        (2.5, 5.2, 0.45, ACCENTRED,  'Bakterium A'),
        (4.5, 5.8, 0.38, DARKGREEN,  'Bakterium B'),
        (6.8, 5.0, 0.50, PURPLE,     'Archaeon C'),
        (3.5, 2.8, 0.42, TEAL,       'Virus D'),
        (6.5, 2.8, 0.35, GOLD,       'Pilz E'),
    ]
    for (cx, cy, r, col, label) in organisms:
        circ = plt.Circle((cx, cy), r, color=col, alpha=0.7, zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, label[:3], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)

    # Reads als kleine Rechtecke (Sequenz-Fragmente)
    read_data = [
        (1.2, 4.0, ACCENTRED),  (2.0, 3.5, ACCENTRED),
        (3.8, 4.5, DARKGREEN),  (5.0, 3.0, DARKGREEN),
        (6.2, 4.2, PURPLE),     (7.5, 5.5, PURPLE),
        (3.0, 2.2, TEAL),       (5.8, 2.1, GOLD),
        (7.0, 3.5, GOLD),       (4.8, 2.5, TEAL),
    ]
    for (rx, ry, col) in read_data:
        rect = mpatches.FancyBboxPatch((rx-0.55, ry-0.18), 1.1, 0.36,
                                        boxstyle='round,pad=0.04',
                                        facecolor=col, alpha=0.5, zorder=2)
        ax.add_patch(rect)

    ax.set_title('Metagenomische Umweltprobe:\nMischung vieler Organismen',
                 fontweight='bold')

    # --- Rechts: Read-Verteilung nach Organismus ---
    ax2 = axes[1]
    org_labels = ['Bakterium A', 'Bakterium B', 'Archaeon C', 'Virus D', 'Pilz E',
                  'Unbekannt']
    read_counts = [3840, 2210, 1650, 890, 430, 980]
    colors_bar  = [ACCENTRED, DARKGREEN, PURPLE, TEAL, GOLD, DARKGRAY]
    bars = ax2.barh(org_labels, read_counts, color=colors_bar,
                    edgecolor='white', linewidth=1.5, alpha=0.85)
    for bar, count in zip(bars, read_counts):
        ax2.text(bar.get_width() + 40, bar.get_y() + bar.get_height()/2,
                 f'{count:,}', va='center', fontsize=9)
    ax2.set_xlabel('Anzahl sequenzierter Reads')
    ax2.set_title('Read-Verteilung in der Probe\n(gesamt: 10\u202f000 Reads)',
                  fontweight='bold')
    total = sum(read_counts)
    ax2.set_xlim(0, max(read_counts) * 1.25)

    fig.suptitle('Abbildung 11: Metagenomische Probe – Reads aus mehreren Organismen',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig11_meta_ueberblick.pdf')
    fig.savefig('figures/fig11_meta_ueberblick.png')
    plt.close(fig)
    print("Plot 11 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 12 – Metagenom-Graph: Multi-Organismus-Adjazenzmatrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_12_meta_graph():
    np.random.seed(42)

    # Drei kurze Reads (simuliert)
    reads = {
        'R1 (Bakt. A)': list('ATGCAT'),
        'R2 (Bakt. B)': list('GCATGC'),
        'R3 (Archaeon)': list('ATCGTA'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap_blue = LinearSegmentedColormap.from_list('bw', ['white', MAINBLUE])

    for ax, (label, seq) in zip(axes, reads.items()):
        n = len(seq)
        mat = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            mat[i, i+1] = 1
        # Komplementaerbindungen
        comp = {'A':'T','T':'A','G':'C','C':'G'}
        for i in range(n):
            for j in range(n):
                if j != i and seq[j] == comp[seq[i]]:
                    mat[i, j] = max(mat[i, j], 1)

        im = ax.imshow(mat, cmap=cmap_blue, aspect='equal', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        tick_labels = [f'{seq[i]}{i}' for i in range(n)]
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                        fontsize=9,
                        color='white' if mat[i, j] else DARKGRAY)
        ax.set_title(f'Read {label}\n{" ".join(seq)}', fontweight='bold')

    fig.suptitle('Abbildung 12: Adjazenzmatrizen dreier metagenomischer Reads',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig12_meta_graph.pdf')
    fig.savefig('figures/fig12_meta_graph.png')
    plt.close(fig)
    print("Plot 12 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 13 – Subgraph-basierte Read-Deduplizierung
# ─────────────────────────────────────────────────────────────────────────────
def plot_13_dedup():
    np.random.seed(3)

    # Simuliere einen Pool von n Reads mit Redundanz
    all_reads = [
        'ATGC', 'ATGCAT', 'GCTA', 'ATGCATGC',
        'TGCA', 'GCTAGC', 'ATGC', 'CATG',
        'ATGCATGCTA', 'TGCAT', 'GCTA', 'ATGC',
    ]

    def seq_to_row_sigs(seq):
        n = len(seq)
        mat = np.zeros((n, n), dtype=int)
        for i in range(n-1):
            mat[i, i+1] = 1
        sigs = compute_signature(mat)
        return [s % (2**n) for s in sigs]

    def is_subgraph(sigs_a, sigs_b):
        na, nb = len(sigs_a), len(sigs_b)
        if na > nb:
            return False
        for rot in range(nb):
            rotated = sigs_b[rot:] + sigs_b[:rot]
            for start in range(nb - na + 1):
                window = rotated[start:start+na]
                if lcs_len(sigs_a, window) >= 2:
                    return True
        return False

    # Deduplizierungs-Algorithmus
    retained = []
    eliminated = []
    for read in all_reads:
        sigs_new = seq_to_row_sigs(read)
        dominated = False
        to_remove = []
        for kept in retained:
            sigs_kept = seq_to_row_sigs(kept)
            if is_subgraph(sigs_new, sigs_kept):
                dominated = True
                break
            elif is_subgraph(sigs_kept, sigs_new):
                to_remove.append(kept)
        for r in to_remove:
            retained.remove(r)
            eliminated.append(r)
        if not dominated:
            retained.append(read)
        else:
            eliminated.append(read)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Links: Vorher/Nachher Balken ---
    ax = axes[0]
    categories = ['Eingehende\nReads', 'Nach\nDeduplizierung', 'Eliminiert']
    values = [len(all_reads), len(retained), len(eliminated)]
    bar_colors = [MAINBLUE, DARKGREEN, ACCENTRED]
    bars = ax.bar(categories, values, color=bar_colors,
                  edgecolor='white', linewidth=2, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    reduction = (1 - len(retained)/len(all_reads)) * 100
    ax.text(0.5, 0.92,
            f'Reduktion: {reduction:.0f}% der Reads eliminiert',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4E8',
                      edgecolor=DARKGREEN))
    ax.set_ylabel('Anzahl Reads')
    ax.set_title('Subgraph-basierte Redundanz-Elimination\n(Vorher vs. Nachher)',
                 fontweight='bold')
    ax.set_ylim(0, len(all_reads) * 1.25)

    # --- Rechts: Read-Längen und Inklusions-Baum ---
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)

    # Baum der Subgraph-Beziehungen
    nodes = {
        'ATGCATGCTA': (5, 7.0),
        'ATGCATGC':   (3, 5.5),
        'GCTAGC':     (7, 5.5),
        'ATGCAT':     (2, 4.0),
        'CATG':       (5, 4.0),
        'GCTA':       (7.5, 4.0),
        'ATGC':       (1.5, 2.5),
        'TGCA':       (4, 2.5),
        'TGCAT':      (6, 2.5),
    }
    edges_tree = [
        ('ATGCATGCTA', 'ATGCATGC'),
        ('ATGCATGCTA', 'GCTAGC'),
        ('ATGCATGC',   'ATGCAT'),
        ('ATGCATGC',   'CATG'),
        ('GCTAGC',     'GCTA'),
        ('ATGCAT',     'ATGC'),
        ('ATGCAT',     'TGCAT'),
        ('CATG',       'TGCA'),
    ]
    for (u, v) in edges_tree:
        xu, yu = nodes[u]
        xv, yv = nodes[v]
        ax2.annotate('', xy=(xv, yv+0.25), xytext=(xu, yu-0.25),
                     arrowprops=dict(arrowstyle='->', color=DARKGRAY, lw=1.5))

    for name, (x, y) in nodes.items():
        in_retained = name in retained
        col = DARKGREEN if in_retained else ACCENTRED
        rect = mpatches.FancyBboxPatch((x-0.85, y-0.28), 1.7, 0.56,
                                        boxstyle='round,pad=0.05',
                                        facecolor=col, alpha=0.8, zorder=3)
        ax2.add_patch(rect)
        ax2.text(x, y, name, ha='center', va='center',
                 fontsize=7.5, fontweight='bold', color='white', zorder=4)

    legend_elements = [
        mpatches.Patch(color=DARKGREEN, label='Behalten (maximal)'),
        mpatches.Patch(color=ACCENTRED, label='Eliminiert (Subgraph)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0), fontsize=9)
    ax2.set_title('Subgraph-Inklusionsbaum der Reads', fontweight='bold')

    fig.suptitle('Abbildung 13: Subgraph-basierte Read-Deduplizierung im Metagenom',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig13_dedup.pdf')
    fig.savefig('figures/fig13_dedup.png')
    plt.close(fig)
    print("Plot 13 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 14 – Komplexitaet metagenomischer Assemblierung
# ─────────────────────────────────────────────────────────────────────────────
def plot_14_meta_komplex():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Links: Laufzeit als Funktion der Read-Anzahl M ---
    ax = axes[0]
    m_values = np.logspace(2, 7, 100)  # 100 bis 10^7 Reads

    # Paarweiser naiver Vergleich: O(M^2 * n^2) mit n=150 (Read-Laenge)
    naive_pairwise = m_values**2 * 150**2

    # Subgraph-Ansatz: O(M * n^3) dank Deduplizierung
    subgraph_approach = m_values * 150**3

    # Nach Deduplizierung (nur ~10% verbleiben): O(0.1*M * n^3)
    after_dedup = 0.1 * m_values * 150**3

    ax.loglog(m_values, naive_pairwise / naive_pairwise[0],
              '-', color=ACCENTRED, linewidth=2.2,
              label=r'Naiv paarweise: $O(M^2 \cdot n^2)$')
    ax.loglog(m_values, subgraph_approach / subgraph_approach[0],
              '-', color=MAINBLUE, linewidth=2.2,
              label=r'Subgraph: $O(M \cdot n^3)$')
    ax.loglog(m_values, after_dedup / after_dedup[0],
              '--', color=DARKGREEN, linewidth=2.2,
              label=r'Nach Dedup (10\%): $O(0{,}1 M \cdot n^3)$')

    # Kreuzungspunkt
    cross_idx = np.argmin(np.abs(naive_pairwise - subgraph_approach))
    ax.axvline(x=m_values[cross_idx], color=DARKGRAY, linestyle=':',
               linewidth=1.5, alpha=0.7)
    ax.text(m_values[cross_idx]*1.5, 1e2,
            f'Übergang\n$M \\approx {m_values[cross_idx]:.0f}$',
            fontsize=8, color=DARKGRAY)

    ax.set_xlabel('Anzahl Reads $M$ (log)')
    ax.set_ylabel('Normierte Laufzeit (log)')
    ax.set_title('Laufzeit der Metagenom-Assemblierung\nvs. Read-Anzahl $M$',
                 fontweight='bold')
    ax.legend(fontsize=9)

    # --- Rechts: Skalierung mit Organismenanzahl ---
    ax2 = axes[1]
    k_values = np.arange(1, 51)

    # Speicherbedarf: O(K * n^2) fuer K Organismen je n=150
    memory_naive = k_values * 150**2 * 8  # 8 Bytes pro int
    memory_subgraph = k_values * 150 * 8  # nur Signatur-Arrays

    ax2.semilogy(k_values, memory_naive / 1e6,
                 'o-', color=ACCENTRED, linewidth=2, markersize=4,
                 label=r'Adjazenzmatrizen: $O(K \cdot n^2)$')
    ax2.semilogy(k_values, memory_subgraph / 1e3,
                 's-', color=DARKGREEN, linewidth=2, markersize=4,
                 label=r'Signatur-Arrays: $O(K \cdot n)$')

    ax2.set_xlabel('Anzahl Organismen $K$')
    ax2.set_ylabel('Speicher (log-Skala, skaliert)')
    ax2.set_title('Speicherbedarf vs. Organismenanzahl\n(log-Skala)',
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.fill_between(k_values,
                     memory_naive / 1e6,
                     memory_subgraph / 1e3,
                     alpha=0.08, color=MAINBLUE,
                     label='Einsparung')

    fig.suptitle('Abbildung 14: Komplexitätsanalyse der metagenomischen Assemblierung',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig14_meta_komplex.pdf')
    fig.savefig('figures/fig14_meta_komplex.png')
    plt.close(fig)
    print("Plot 14 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 15 – Taxonomische Klassifikation via Subgraph-Signaturen
# ─────────────────────────────────────────────────────────────────────────────
def plot_15_taxonomie():
    np.random.seed(17)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Links: Dendrogram-artiger Baum mit LCS-Distanzen ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')

    # Taxonomischer Baum (vereinfacht)
    tree_nodes = {
        'Leben':       (5.0, 7.5, DARKGRAY,  14),
        'Prokaryoten': (2.5, 6.2, MAINBLUE,  11),
        'Eukaryoten':  (7.5, 6.2, DARKGREEN, 11),
        'Bakterien':   (1.5, 4.8, MAINBLUE,   9),
        'Archaeen':    (3.5, 4.8, TEAL,        9),
        'Pilze':       (6.2, 4.8, DARKGREEN,   9),
        'Tiere':       (8.8, 4.8, ACCENTRED,   9),
        'E. coli':     (0.8, 3.2, MAINBLUE,    8),
        'B. subtilis': (2.2, 3.2, MAINBLUE,    8),
        'M. jannaschii':(3.5, 3.2, TEAL,       8),
        'S. cerevisiae':(6.2, 3.2, DARKGREEN,  8),
        'H. sapiens':  (8.8, 3.2, ACCENTRED,   8),
    }
    tree_edges = [
        ('Leben', 'Prokaryoten'), ('Leben', 'Eukaryoten'),
        ('Prokaryoten', 'Bakterien'), ('Prokaryoten', 'Archaeen'),
        ('Eukaryoten', 'Pilze'), ('Eukaryoten', 'Tiere'),
        ('Bakterien', 'E. coli'), ('Bakterien', 'B. subtilis'),
        ('Archaeen', 'M. jannaschii'),
        ('Pilze', 'S. cerevisiae'), ('Tiere', 'H. sapiens'),
    ]

    # LCS-Aehnlichkeiten als Kantengewichte (simuliert)
    edge_weights = {
        ('Leben', 'Prokaryoten'): 0.55,
        ('Leben', 'Eukaryoten'): 0.60,
        ('Prokaryoten', 'Bakterien'): 0.72,
        ('Prokaryoten', 'Archaeen'): 0.68,
        ('Eukaryoten', 'Pilze'): 0.78,
        ('Eukaryoten', 'Tiere'): 0.82,
        ('Bakterien', 'E. coli'): 0.91,
        ('Bakterien', 'B. subtilis'): 0.88,
        ('Archaeen', 'M. jannaschii'): 0.93,
        ('Pilze', 'S. cerevisiae'): 0.95,
        ('Tiere', 'H. sapiens'): 0.96,
    }

    for (u, v) in tree_edges:
        xu, yu, *_ = tree_nodes[u]
        xv, yv, *_ = tree_nodes[v]
        weight = edge_weights.get((u, v), 0.7)
        lw = 1 + weight * 3
        ax.plot([xu, xv], [yu, yv], color=DARKGRAY,
                linewidth=lw, alpha=0.5, zorder=1)
        mx, my = (xu+xv)/2, (yu+yv)/2
        ax.text(mx + 0.15, my, f'{weight:.2f}', fontsize=7,
                color=DARKGRAY, style='italic')

    for name, (x, y, col, fs) in tree_nodes.items():
        r = 0.38 if fs >= 11 else 0.30
        circ = plt.Circle((x, y), r, color=col, alpha=0.85, zorder=3)
        ax.add_patch(circ)
        short = name.split('.')[0][:3] if '.' in name else name[:4]
        ax.text(x, y, short, ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)
        ax.text(x, y - r - 0.22, name, ha='center', va='top',
                fontsize=6.5, color=DARKGRAY)

    ax.set_title('Taxonomischer Baum mit LCS-Ähnlichkeitsgewichten\n'
                 '(Kantenbreite proportional zur LCS-Ähnlichkeit)',
                 fontweight='bold')

    # --- Rechts: Heatmap paarweiser LCS-Abstände ---
    ax2 = axes[1]
    species = ['E. coli', 'B. subtil.', 'M. jann.', 'S. cerev.', 'H. sap.']
    k = len(species)
    # Simulierte normierte LCS-Distanzen (symmetrisch)
    dist_matrix = np.array([
        [1.00, 0.88, 0.42, 0.31, 0.28],
        [0.88, 1.00, 0.40, 0.33, 0.27],
        [0.42, 0.40, 1.00, 0.38, 0.35],
        [0.31, 0.33, 0.38, 1.00, 0.61],
        [0.28, 0.27, 0.35, 0.61, 1.00],
    ])

    cmap_sim = LinearSegmentedColormap.from_list('sim', ['white', MAINBLUE])
    im = ax2.imshow(dist_matrix, cmap=cmap_sim, aspect='equal', vmin=0, vmax=1)
    ax2.set_xticks(range(k))
    ax2.set_yticks(range(k))
    ax2.set_xticklabels(species, rotation=30, ha='right', fontsize=9)
    ax2.set_yticklabels(species, fontsize=9)
    for i in range(k):
        for j in range(k):
            ax2.text(j, i, f'{dist_matrix[i,j]:.2f}',
                     ha='center', va='center', fontsize=10,
                     color='white' if dist_matrix[i,j] > 0.6 else DARKGRAY,
                     fontweight='bold' if i==j else 'normal')
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Normierte LCS-Ähnlichkeit')
    ax2.set_title('Paarweise LCS-Ähnlichkeitsmatrix\nfür 5 Modellorganismen',
                  fontweight='bold')

    fig.suptitle('Abbildung 15: Taxonomische Klassifikation via Subgraph-LCS-Ähnlichkeiten',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/fig15_taxonomie.pdf')
    fig.savefig('figures/fig15_taxonomie.png')
    plt.close(fig)
    print("Plot 15 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 16 – Erweiterter Ausblick: Roadmap aller zukuenftigen Richtungen
# ─────────────────────────────────────────────────────────────────────────────
def plot_16_roadmap():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Zentraler Kern
    center_ellipse = mpatches.Ellipse((7.5, 5.0), 3.2, 1.6,
                                       facecolor=MAINBLUE, edgecolor='white',
                                       linewidth=2, alpha=0.9, zorder=3)
    ax.add_patch(center_ellipse)
    ax.text(7.5, 5.0, 'Subgraph-DNA\nAlgorithmus', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # Ausblick-Knoten (x, y, Farbe, Titel, Untertitel)
    outlook_nodes = [
        (2.0, 8.5, ACCENTRED,  'Parallelisierung',      'GPU / $n$ Prozessoren\n$O(n^2)$ bei $n$ Cores'),
        (6.0, 8.5, DARKGREEN,  'RNA-Analyse',            'Sekundärstruktur\n$\\Sigma_{RNA}=\\{A,U,G,C\\}$'),
        (10.5, 8.5, PURPLE,    'Epigenetik',             'Methylierungs-\ngraph $G_{epi}$'),
        (13.5, 7.0, TEAL,      'Proteomik',              'Aminosäure-\nGraph, AlphaFold'),
        (13.5, 3.5, GOLD,      'Quantenalgorithmen',     'Grover-basierte\nRotationssuche'),
        (10.5, 1.5, ACCENTRED, 'Fuzzy-Matching',         'Hamming $\\leq k$\n$\\varepsilon$-LCS'),
        (5.5, 1.5, DARKGREEN,  'Metagenomik',            'Multi-Organismus\nDeduplizierung'),
        (2.0, 1.5, MAINBLUE,   'Phylogenetik',           'Evolutionsbäume\nLCS-Distanzmatrix'),
        (0.8, 5.0, PURPLE,     'Gewichtete Graphen',     'Bindungsstärken\n$w: E \\to \\mathbb{R}_{>0}$'),
        (2.5, 3.2, TEAL,       'Nanoporen-Integration',  'Echtzeit-\nSequenzierung'),
        (12.5, 5.5, ACCENTRED, 'FPGA-Beschleunigung',    'Hardware-\nSignaturberechnung'),
        (5.0, 3.0, GOLD,       'Approximation',          'PTAS für\n$k$-Subgraph-Cover'),
    ]

    for (x, y, col, title, subtitle) in outlook_nodes:
        rect = mpatches.FancyBboxPatch((x-1.2, y-0.72), 2.4, 1.44,
                                        boxstyle='round,pad=0.1',
                                        facecolor=col, edgecolor='white',
                                        linewidth=1.5, alpha=0.82, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y+0.22, title, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=4)
        ax.text(x, y-0.28, subtitle, ha='center', va='center',
                fontsize=7, color='white', alpha=0.92, zorder=4)

        # Verbindungslinie zum Kern
        ax.annotate('', xy=(7.5, 5.0), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=1.4, alpha=0.55,
                                   connectionstyle='arc3,rad=0.12'))

    ax.text(7.5, 9.6, 'Abbildung 16: Erweiterter Ausblick – Forschungsrichtungen des Subgraph-DNA-Algorithmus',
            ha='center', va='center', fontsize=12, fontweight='bold', color=DARKGRAY)

    fig.tight_layout()
    fig.savefig('figures/fig16_roadmap.pdf')
    fig.savefig('figures/fig16_roadmap.png')
    plt.close(fig)
    print("Plot 16 gespeichert.")


if __name__ == '__main__':
    print("Generiere neue Plots (11–16) ...")
    plot_11_meta_ueberblick()
    plot_12_meta_graph()
    plot_13_dedup()
    plot_14_meta_komplex()
    plot_15_taxonomie()
    plot_16_roadmap()
    print("Alle neuen Plots gespeichert.")

