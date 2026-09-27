#!/usr/bin/env python3
"""
Generates all matplotlib plots for the TENTRIS/Subgraph Algorithm paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BLUE = '#19468C'
RED  = '#B4321E'
GREEN= '#1E6432'
GRAY = '#3C3C46'
LGRAY= '#F5F5F8'

# ─────────────────────────────────────────────────────────────
# Plot 1: Laufzeitkomplexität – Vergleich verschiedener Algorithmen
# ─────────────────────────────────────────────────────────────
def plot1_complexity():
    fig, ax = plt.subplots(figsize=(8, 5))
    n = np.arange(1, 51)

    ax.semilogy(n, n**2,         color=GREEN,  lw=2,   ls='--',  label=r'$O(n^2)$ – Signaturberechnung')
    ax.semilogy(n, n**3,         color=BLUE,   lw=2.5, ls='-',   label=r'$O(n^3)$ – Subgraph Algorithmus (TENTRIS)')
    ax.semilogy(n, n**3*np.log(n+1), color=RED, lw=2, ls='-.',  label=r'$O(n^3 \log n)$ – Naiver Ansatz')
    fact = [1]
    for i in range(1, 16):
        fact.append(fact[-1]*i)
    n_fact = np.arange(1, 16)
    ax.semilogy(n_fact, np.array(fact[:15])*np.array(n_fact)**2,
                color=GRAY, lw=1.5, ls=':', label=r'$O(n!\cdot n^2)$ – Vollständige Permutation')

    ax.set_xlabel('Anzahl der Knoten $n$')
    ax.set_ylabel('Laufzeit (log-Skala, relative Einheiten)')
    ax.set_title('Laufzeitkomplexität verschiedener Subgraph-Algorithmen')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(1, 50)
    ax.set_ylim(1, 1e14)
    ax.grid(True, which='both', alpha=0.3)
    ax.fill_between(n, n**3, n**3*np.log(n+1), alpha=0.08, color=BLUE)
    fig.tight_layout()
    fig.savefig('plot1_complexity.pdf')
    fig.savefig('plot1_complexity.png')
    plt.close(fig)
    print("Plot 1 gespeichert.")

# ─────────────────────────────────────────────────────────────
# Plot 2: Hypertrie-Struktur – Tiefenverteilung der Knoten
# ─────────────────────────────────────────────────────────────
def plot2_hypertrie():
    # Daten aus dem Folienbild: 8 Tripel aus id(s), id(p), id(o)
    triples = [
        (1,2,3),(1,2,4),(3,2,4),(3,2,5),
        (4,2,3),(4,2,5),(3,6,7),(5,6,7)
    ]
    depths = {'H(3)':1, 'H(2)':0, 'H(1)':0}

    # Simuliere Hypertrie-Aufbau: Zähle Knoten je Ebene
    h3_nodes = set()
    h2_nodes = set()
    h1_nodes = set()
    for s,p,o in triples:
        h3_nodes.add(s)
        h2_nodes.add((s,p))
        h1_nodes.add((s,p,o))

    levels = ['H(3)\n(Wurzel)', 'H(2)\n(Mitte)', 'H(1)\n(Blatt)']
    counts = [len(h3_nodes), len(h2_nodes), len(h1_nodes)]
    colors = [BLUE, RED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Balkendiagramm
    bars = axes[0].bar(levels, counts, color=colors, edgecolor='white', linewidth=1.2, width=0.5)
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(cnt), ha='center', va='bottom', fontweight='bold')
    axes[0].set_title('Knotenanzahl je Hypertrie-Ebene\n(8 RDF-Tripel)')
    axes[0].set_ylabel('Anzahl eindeutiger Knoten')
    axes[0].set_ylim(0, max(counts)+2)
    axes[0].grid(axis='y', alpha=0.3)

    # Prädikat-Verteilung
    pred_count = {}
    for s,p,o in triples:
        pred_count[p] = pred_count.get(p,0)+1
    preds = [f'p={k}' for k in sorted(pred_count)]
    vals  = [pred_count[k] for k in sorted(pred_count)]
    axes[1].pie(vals, labels=preds, colors=[BLUE, RED],
                autopct='%1.0f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':1.5})
    axes[1].set_title('Verteilung der Prädikate\nin den 8 RDF-Tripeln')

    fig.suptitle('Hypertrie-Struktur: Statistik der Beispieldaten (TENTRIS)', fontweight='bold')
    fig.tight_layout()
    fig.savefig('plot2_hypertrie.pdf')
    fig.savefig('plot2_hypertrie.png')
    plt.close(fig)
    print("Plot 2 gespeichert.")

# ─────────────────────────────────────────────────────────────
# Plot 3: SPARQL → Einstein-Summation: Tensor-Operand-Größen
# ─────────────────────────────────────────────────────────────
def plot3_sparql_tensor():
    # Simuliere die drei Operanden aus der Folie:
    # T[1,2,:] , T[:,2,:] , T[:,6,7]
    # Zeige Speicher- und Kardinalitätswachstum mit n
    n_vals = np.array([10, 50, 100, 500, 1000, 5000, 10000])

    # Sparse-Anteil: reale KG haben ~0.001% Dichte
    density = 0.001
    t_size  = density * n_vals**3          # Tensor-Größe
    t12     = density * n_vals             # T[1,2,:] – ein Schnitt
    t_star2 = density * n_vals**2          # T[:,2,:] – zwei freie Dim.
    t_s67   = density * n_vals             # T[:,6,7] – ein Schnitt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].loglog(n_vals, t_size,  color=GRAY,  lw=1.5, ls=':', label='Vollständiger Tensor $T$')
    axes[0].loglog(n_vals, t_star2, color=RED,   lw=2,   ls='-.', label=r'$T[:,2,:]$ – Matrix-Schnitt')
    axes[0].loglog(n_vals, t12,     color=BLUE,  lw=2,   ls='-',  label=r'$T[1,2,:]$ – Vektor-Schnitt')
    axes[0].loglog(n_vals, t_s67,   color=GREEN, lw=2,   ls='--', label=r'$T[:,6,7]$ – Vektor-Schnitt')
    axes[0].set_xlabel('Anzahl Entitäten $n$')
    axes[0].set_ylabel('Nicht-Null-Einträge (log-Skala)')
    axes[0].set_title('Tensor-Operandgröße in Abhängigkeit von $n$\n(Dichte = 0.1%)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which='both', alpha=0.3)

    # Einsteinsumme Auswertungszeit (simuliert)
    t_naive   = t_star2**1.5                # naiv: Matrix-Vektor
    t_tentris = t12 * np.log(t12+1) + t_s67 * np.log(t_s67+1)  # index-gestützt
    axes[1].loglog(n_vals, t_naive,   color=RED,  lw=2, label='Naive Auswertung')
    axes[1].loglog(n_vals, t_tentris, color=BLUE, lw=2, ls='--', label='TENTRIS (Hypertrie)')
    axes[1].fill_between(n_vals, t_tentris, t_naive, alpha=0.1, color=BLUE)
    axes[1].set_xlabel('Anzahl Entitäten $n$')
    axes[1].set_ylabel('Relative Auswertungszeit (log)')
    axes[1].set_title('Auswertungszeit: Naive vs. TENTRIS')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which='both', alpha=0.3)

    fig.suptitle('SPARQL-zu-Einstein-Summation: Tensoroperanden und Auswertungseffizienz', fontweight='bold')
    fig.tight_layout()
    fig.savefig('plot3_sparql_tensor.pdf')
    fig.savefig('plot3_sparql_tensor.png')
    plt.close(fig)
    print("Plot 3 gespeichert.")

# ─────────────────────────────────────────────────────────────
# Plot 4: Subgraph Algorithmus auf Wissensgraphen – Signatur-Demo
# ─────────────────────────────────────────────────────────────
def plot4_signatures():
    # Demonstriere Signatur-Werte für eine 5×5 Adjazenzmatrix
    np.random.seed(42)
    n = 5
    A = np.triu(np.random.randint(0,2,(n,n)),1)  # Obere Dreiecksmatrix

    # Signaturen berechnen
    sigs = []
    for j in range(n):
        val = sum(A[i,j]*(2**i) for i in range(n)) + j*(2**n)
        sigs.append(val)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Adjazenzmatrix visualisieren
    im = axes[0].imshow(A, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Adjazenzmatrix $A$ ($5\\times5$)')
    axes[0].set_xlabel('Spalte $j$')
    axes[0].set_ylabel('Zeile $i$')
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, str(int(A[i,j])), ha='center', va='center',
                         color='white' if A[i,j]>0.5 else GRAY, fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))

    # Spalten-Vektoren
    for j in range(n):
        col = A[:, j]
        axes[1].barh([f'$j={j}$, Bit {i}' for i in range(n)],
                      col * (2**np.arange(n)), left=0,
                      color=[BLUE if x else LGRAY for x in col],
                      edgecolor='white', height=0.15+(j*0.12))
    axes[1].set_xlim(0, 20)
    axes[1].set_title('Bit-Gewichtung je Spalte')
    axes[1].set_xlabel('Wert $A_{ij}\\cdot 2^i$')

    # Signatur-Werte
    colors = [BLUE, RED, GREEN, GRAY, '#8B0057']
    bars = axes[2].bar([f'$\\sigma_{j}$' for j in range(n)], sigs,
                       color=colors[:n], edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, sigs):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].set_title('Berechnete Signaturen $\\sigma_j$')
    axes[2].set_ylabel('Signaturwert')
    axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle('Signaturberechnung des Subgraph Algorithmus auf RDF-Wissensgraphen', fontweight='bold')
    fig.tight_layout()
    fig.savefig('plot4_signatures.pdf')
    fig.savefig('plot4_signatures.png')
    plt.close(fig)
    print("Plot 4 gespeichert.")

# ─────────────────────────────────────────────────────────────
# Plot 5: LCS-Länge vs. Rotationsindex – zyklische Analyse
# ─────────────────────────────────────────────────────────────
def lcs_length(a, b):
    m, n2 = len(a), len(b)
    dp = [[0]*(n2+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n2+1):
            if a[i-1]==b[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    return dp[m][n2]

def plot5_lcs_rotation():
    np.random.seed(7)
    n = 10
    A = np.triu(np.random.randint(0,2,(n,n)),1)
    B = np.triu(np.random.randint(0,2,(n,n)),1)
    # Füge Subgraph-Relation ein: Kopiere erste 5 Spalten
    B[:,:5] = A[:,:5]

    def sig(mat):
        return [sum(mat[i,j]*(2**i) for i in range(n))+j*(2**n) for j in range(n)]

    sigA = sig(A)
    sigB = sig(B)

    lcs_vals = []
    for r in range(n):
        rotB = sigB[r:] + sigB[:r]
        lcs_vals.append(lcs_length(sigA, rotB))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(range(n), lcs_vals, color=[GREEN if v>=2 else RED for v in lcs_vals],
                edgecolor='white', linewidth=1.2)
    axes[0].axhline(2, color=GRAY, ls='--', lw=1.5, label='Schwelle LCS = 2')
    axes[0].set_xlabel('Rotationsindex $r$')
    axes[0].set_ylabel('LCS-Länge')
    axes[0].set_title('LCS-Länge je Rotation\n(Grün = Subgraph erkannt)')
    axes[0].set_xticks(range(n))
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # DP-Tabelle für beste Rotation
    best_r = int(np.argmax(lcs_vals))
    rotB = sigB[best_r:] + sigB[:best_r]
    m2 = len(sigA)
    dp = [[0]*(m2+1) for _ in range(m2+1)]
    for i in range(1,m2+1):
        for j in range(1,m2+1):
            if sigA[i-1]==rotB[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    dp_arr = np.array(dp)
    im2 = axes[1].imshow(dp_arr, cmap='YlOrRd', aspect='auto')
    axes[1].set_title(f'DP-Tabelle für Rotation $r={best_r}$ (LCS={lcs_vals[best_r]})')
    axes[1].set_xlabel('Index $\\sigma^B$ (rotiert)')
    axes[1].set_ylabel('Index $\\sigma^A$')
    plt.colorbar(im2, ax=axes[1], label='LCS-Teilwert')

    fig.suptitle('LCS-Analyse über alle zyklischen Rotationen', fontweight='bold')
    fig.tight_layout()
    fig.savefig('plot5_lcs_rotation.pdf')
    fig.savefig('plot5_lcs_rotation.png')
    plt.close(fig)
    print("Plot 5 gespeichert.")

# ─────────────────────────────────────────────────────────────
# Plot 6: Speicher- und Skalierungsverhalten
# ─────────────────────────────────────────────────────────────
def plot6_memory():
    n = np.logspace(1, 4, 100)

    mem_adj   = n**2 * 4 / 1024**2          # Adjazenzmatrix in MB (int32)
    mem_sig   = n * 8 / 1024**2             # Signatur-Array in MB (int64)
    mem_hyper = 0.001 * n**3 * 8 / 1024**2  # Hypertrie (1% Dichte)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].loglog(n, mem_adj,  color=BLUE,  lw=2,   label='Adjazenzmatrix $O(n^2)$')
    axes[0].loglog(n, mem_sig,  color=GREEN, lw=2,   ls='--', label='Signatur-Array $O(n)$')
    axes[0].loglog(n, mem_hyper,color=RED,   lw=2,   ls='-.', label='Hypertrie (Dichte=0.1%)')
    axes[0].set_xlabel('Anzahl Knoten/Entitäten $n$')
    axes[0].set_ylabel('Speicher (MB, log-Skala)')
    axes[0].set_title('Speicherbedarf in Abhängigkeit von $n$')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].axvline(1000, color=GRAY, ls=':', lw=1, alpha=0.6)
    axes[0].text(1100, 0.001, '$n=1000$', color=GRAY, fontsize=8)

    # Skalierungseffizienz TENTRIS vs. relationale DB
    n2 = np.array([100, 500, 1000, 2000, 5000, 10000])
    t_relational = (n2/100)**3        # kubisch skalierend
    t_tentris    = (n2/100)**2.1      # TENTRIS näherungsweise n^2.1
    t_subgraph   = (n2/100)**3        # Subgraph Algorithmus n^3

    axes[1].plot(n2, t_relational, color=RED,   lw=2,   marker='s', ms=5, label='Relationale DB (Join)')
    axes[1].plot(n2, t_tentris,    color=BLUE,  lw=2,   marker='o', ms=5, label='TENTRIS (Hypertrie)')
    axes[1].plot(n2, t_subgraph,   color=GREEN, lw=2,   marker='^', ms=5, ls='--', label='Subgraph Alg. $O(n^3)$')
    axes[1].set_xlabel('Anzahl Entitäten $n$')
    axes[1].set_ylabel('Relative Laufzeit (normiert auf $n=100$)')
    axes[1].set_title('Skalierungsvergleich: TENTRIS vs. relationale DB')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    fig.suptitle('Speicher- und Skalierungsanalyse: TENTRIS und Subgraph Algorithmus', fontweight='bold')
    fig.tight_layout()
    fig.savefig('plot6_memory.pdf')
    fig.savefig('plot6_memory.png')
    plt.close(fig)
    print("Plot 6 gespeichert.")

# Run all
plot1_complexity()
plot2_hypertrie()
plot3_sparql_tensor()
plot4_signatures()
plot5_lcs_rotation()
plot6_memory()
print("Alle Plots erfolgreich generiert.")

