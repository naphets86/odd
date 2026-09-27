#!/usr/bin/env python3
"""
Plots für: Transformer-Architekturen als Subgraph-Isomorphismus-Problem
Stephan Epp, 2026
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ─── Plot 1: Transformer-Architektur als Graph ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

nodes = {
    'Input': (5, 7),
    'Embed': (5, 6),
    'Q': (2.5, 4.5), 'K': (5, 4.5), 'V': (7.5, 4.5),
    'Attn': (5, 3),
    'FF': (5, 1.8),
    'Output': (5, 0.5),
}
colors = {
    'Input': '#1946a2', 'Embed': '#1946a2',
    'Q': '#b43218', 'K': '#b43218', 'V': '#b43218',
    'Attn': '#1e6432', 'FF': '#1e6432', 'Output': '#1946a2',
}
for name, (x, y) in nodes.items():
    circle = plt.Circle((x, y), 0.45, color=colors[name], zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)

edges = [
    ('Input','Embed'), ('Embed','Q'), ('Embed','K'), ('Embed','V'),
    ('Q','Attn'), ('K','Attn'), ('V','Attn'), ('Attn','FF'), ('FF','Output')
]
for (a, b) in edges:
    x1, y1 = nodes[a]; x2, y2 = nodes[b]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))

ax.set_title('Plot 1: Transformer-Architektur als gerichteter Graph $G_T$', pad=12)
legend_elements = [
    mpatches.Patch(color='#1946a2', label='Token/Embedding/Output'),
    mpatches.Patch(color='#b43218', label='Attention-Köpfe (Q, K, V)'),
    mpatches.Patch(color='#1e6432', label='Feed-Forward / Attention'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('plot1_transformer_graph.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 erstellt")

# ─── Plot 2: Subgraph-Isomorphismus Transformer → Subgraph ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax in axes:
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis('off')

# Linker Graph: Vollständiger Transformer (G)
ax = axes[0]
ax.set_title('Vollständiger Transformer $G_T$', fontsize=11)
pos_G = {'x1':(1,5),'x2':(3,5),'x3':(5,5),'h1':(1,3),'h2':(3,3),'h3':(5,3),'y':(3,1)}
for n,(x,y) in pos_G.items():
    c = '#1946a2' if n.startswith('x') or n=='y' else '#b43218'
    ax.add_patch(plt.Circle((x,y),0.4,color=c,zorder=3))
    ax.text(x,y,n,ha='center',va='center',color='white',fontsize=8,fontweight='bold',zorder=4)
for src,dst in [('x1','h1'),('x2','h2'),('x3','h3'),('x1','h2'),('x2','h1'),('x2','h3'),('x3','h2'),('h1','y'),('h2','y'),('h3','y')]:
    x1,y1=pos_G[src]; x2,y2=pos_G[dst]
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#555',lw=1.2))

# Rechter Graph: Subgraph-Pattern (G')
ax = axes[1]
ax.set_title("Subgraph-Pattern $G_S \\subseteq G_T$", fontsize=11)
pos_S = {'x1':(2,5),'x2':(4,5),'h1':(2,3),'h2':(4,3),'y':(3,1)}
for n,(x,y) in pos_S.items():
    c = '#1e6432' if n.startswith('h') else '#1946a2'
    ax.add_patch(plt.Circle((x,y),0.4,color=c,zorder=3))
    ax.text(x,y,n,ha='center',va='center',color='white',fontsize=8,fontweight='bold',zorder=4)
for src,dst in [('x1','h1'),('x2','h2'),('x1','h2'),('h1','y'),('h2','y')]:
    x1,y1=pos_S[src]; x2,y2=pos_S[dst]
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#b43218',lw=1.5))
axes[1].text(3, 0.2, r'$G_S \subseteq G_T$ via Subgraph Algorithmus', ha='center', fontsize=9, color='#b43218')

fig.suptitle('Plot 2: Subgraph-Isomorphismus zwischen Transformer und Pattern', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('plot2_subgraph_isomorphismus.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 erstellt")

# ─── Plot 3: Signaturberechnung Attention-Matrix ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

n = 5
np.random.seed(42)
A = np.random.randint(0, 2, (n, n)).astype(float)
A = (A + A.T > 0).astype(float)
np.fill_diagonal(A, 0)

sigmas = []
for j in range(n):
    s = sum(A[i, j] * (2**i) for i in range(n)) + j * (2**n)
    sigmas.append(s)

im = axes[0].imshow(A, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Attention-Adjazenzmatrix $A$')
axes[0].set_xlabel('Spalte $j$'); axes[0].set_ylabel('Zeile $i$')
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, int(A[i,j]), ha='center', va='center',
                     color='white' if A[i,j]>0.5 else 'black', fontsize=10)

axes[1].bar(range(n), sigmas, color='#1946a2', edgecolor='white', linewidth=0.5)
axes[1].set_title('Signaturen $\\sigma_j$ der Attention-Matrix')
axes[1].set_xlabel('Spaltenindex $j$')
axes[1].set_ylabel('Signaturwert $\\sigma_j$')
axes[1].set_xticks(range(n))
for i, v in enumerate(sigmas):
    axes[1].text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=9, color='#1946a2')

fig.suptitle('Plot 3: Signaturberechnung für Transformer-Attention-Matrix', fontsize=12)
plt.tight_layout()
plt.savefig('plot3_signaturen_attention.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 erstellt")

# ─── Plot 4: Zyklische Rotation der Signatursequenz ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
sigmas_base = [5, 18, 36, 48, 72]
n = len(sigmas_base)
colors_bar = ['#1946a2','#b43218','#1e6432','#8B4513','#666']

for rot in range(5):
    ax = axes[rot // 3][rot % 3]
    rotated = sigmas_base[rot:] + sigmas_base[:rot]
    bars = ax.bar(range(n), rotated, color=colors_bar, edgecolor='white')
    ax.set_title(f'Rotation $r={rot}$', fontsize=10)
    ax.set_xlabel('Position'); ax.set_ylabel('$\\sigma$')
    ax.set_xticks(range(n))
    for i, v in enumerate(rotated):
        ax.text(i, v+0.5, str(v), ha='center', va='bottom', fontsize=8)

axes[1][2].axis('off')
axes[1][2].text(0.5, 0.5, f'Nur $n={n}$ Rotationen\nstatt $n! = {120}$\nPermutationen',
                ha='center', va='center', fontsize=12, transform=axes[1][2].transAxes,
                bbox=dict(boxstyle='round', facecolor='#e8f0fe', edgecolor='#1946a2'))

fig.suptitle('Plot 4: Zyklische Rotation der Transformer-Signatursequenz', fontsize=12)
plt.tight_layout()
plt.savefig('plot4_rotation_signaturen.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 erstellt")

# ─── Plot 5: LCS-DP-Tabelle (Attention-Pattern vs. Query) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

seq_A = [5, 18, 36, 48]
seq_B = [18, 5, 48, 36, 72]
m, n_s = len(seq_A), len(seq_B)
dp = [[0]*(n_s+1) for _ in range(m+1)]
for i in range(1, m+1):
    for j in range(1, n_s+1):
        if seq_A[i-1] == seq_B[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

dp_arr = np.array(dp)
im = axes[0].imshow(dp_arr, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(n_s+1))
axes[0].set_xticklabels(['-']+[str(x) for x in seq_B], fontsize=8)
axes[0].set_yticks(range(m+1))
axes[0].set_yticklabels(['-']+[str(x) for x in seq_A], fontsize=8)
axes[0].set_xlabel('$\\sigma^B$ (Transformer-Query)')
axes[0].set_ylabel('$\\sigma^A$ (Pattern)')
axes[0].set_title('LCS-DP-Tabelle')
for i in range(m+1):
    for j in range(n_s+1):
        axes[0].text(j, i, dp[i][j], ha='center', va='center',
                     color='white' if dp[i][j] > 2 else 'black', fontsize=9)
plt.colorbar(im, ax=axes[0], label='LCS-Länge')

lcs_lengths = [dp[m][j] for j in range(n_s+1)]
axes[1].plot(range(n_s+1), lcs_lengths, 'o-', color='#1946a2', linewidth=2, markersize=6)
axes[1].axhline(y=2, color='#b43218', linestyle='--', label='Schwellenwert (LCS $\\geq$ 2)')
axes[1].fill_between(range(n_s+1), lcs_lengths, 2,
                     where=[l >= 2 for l in lcs_lengths], alpha=0.2, color='#1e6432',
                     label='Subgraph erkannt')
axes[1].set_xlabel('Position in $\\sigma^B$')
axes[1].set_ylabel('LCS-Länge')
axes[1].set_title('LCS-Verlauf und Subgraph-Schwellenwert')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Plot 5: LCS-Berechnung für Transformer-Attention-Pattern-Matching', fontsize=12)
plt.tight_layout()
plt.savefig('plot5_lcs_dp_tabelle.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 erstellt")

# ─── Plot 6: Komplexitätsvergleich ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ns = np.arange(2, 25)

import math
naive = np.array([math.factorial(n) * n**2 if n <= 12 else float('inf') for n in ns], dtype=float)
subgraph = ns**3
attention_naive = 2**ns
attention_sg = ns**2 * np.log2(ns)

ax1 = axes[0]
mask = naive < 1e15
ax1.semilogy(ns[mask], naive[mask], 's--', color='#b43218', label='Naiv: $O(n! \\cdot n^2)$', linewidth=2)
ax1.semilogy(ns, subgraph, 'o-', color='#1946a2', label='Subgraph: $O(n^3)$', linewidth=2)
ax1.set_xlabel('Graphgröße $n$ (Attention-Köpfe)')
ax1.set_ylabel('Operationen (log)')
ax1.set_title('Komplexitätsvergleich: Naiv vs. Subgraph')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ns2 = np.arange(2, 50)
ax2.semilogy(ns2, ns2**3, 'o-', color='#1946a2', label='Subgraph $O(n^3)$', linewidth=2)
ax2.semilogy(ns2, 2**ns2, 's--', color='#b43218', label='Exponentiell $O(2^n)$', linewidth=2)
ax2.semilogy(ns2, ns2**2 * np.log2(ns2), '^-.', color='#1e6432', label='$O(n^2 \\log n)$', linewidth=2)
ax2.set_xlabel('Modellgröße $n$')
ax2.set_ylabel('Laufzeit (log)')
ax2.set_title('Skalierung für LLM-Analyse')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

fig.suptitle('Plot 6: Komplexitätsanalyse des Subgraph Algorithmus für Transformer', fontsize=12)
plt.tight_layout()
plt.savefig('plot6_komplexitaet.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 erstellt")

# ─── Plot 7: Hierarchische Subgraph-Struktur (GPT → BERT → Pattern) ──────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

levels = [
    {'label': 'GPT-4 / LLaMA\n(Vollständiges LLM)', 'x': 5, 'y': 7, 'r': 0.7, 'c': '#1946a2'},
    {'label': 'Transformer-Block\n(Multi-Head Attention)', 'x': 5, 'y': 5, 'r': 0.65, 'c': '#1946a2'},
    {'label': 'Self-Attention\nSubgraph $G_A$', 'x': 2.5, 'y': 3, 'r': 0.6, 'c': '#b43218'},
    {'label': 'Feed-Forward\nSubgraph $G_F$', 'x': 7.5, 'y': 3, 'r': 0.6, 'c': '#b43218'},
    {'label': 'Query-Signatur\n$\\sigma_Q$', 'x': 1.5, 'y': 1, 'r': 0.5, 'c': '#1e6432'},
    {'label': 'Key-Signatur\n$\\sigma_K$', 'x': 3.5, 'y': 1, 'r': 0.5, 'c': '#1e6432'},
    {'label': 'Value-Signatur\n$\\sigma_V$', 'x': 6.5, 'y': 1, 'r': 0.5, 'c': '#1e6432'},
    {'label': 'Output-Signatur\n$\\sigma_O$', 'x': 8.5, 'y': 1, 'r': 0.5, 'c': '#1e6432'},
]
for l in levels:
    ax.add_patch(plt.Circle((l['x'], l['y']), l['r'], color=l['c'], zorder=3, alpha=0.9))
    ax.text(l['x'], l['y'], l['label'], ha='center', va='center',
            color='white', fontsize=7, fontweight='bold', zorder=4)

edges_h = [(5,7,5,5),(5,5,2.5,3),(5,5,7.5,3),(2.5,3,1.5,1),(2.5,3,3.5,1),(7.5,3,6.5,1),(7.5,3,8.5,1)]
for x1,y1,x2,y2 in edges_h:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color='#888',lw=1.5))

legend_e = [
    mpatches.Patch(color='#1946a2', label='LLM / Transformer-Block'),
    mpatches.Patch(color='#b43218', label='Attention / FF Subgraphen'),
    mpatches.Patch(color='#1e6432', label='Signatur-Primitive'),
]
ax.legend(handles=legend_e, loc='upper left', fontsize=9)
ax.set_title('Plot 7: Hierarchische Subgraph-Zerlegung eines LLM', pad=12)
plt.tight_layout()
plt.savefig('plot7_hierarchie_llm.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 erstellt")

# ─── Plot 8: Empirischer Laufzeitvergleich ────────────────────────────────────
import time, random

def compute_signatures(A):
    n = len(A)
    return [sum(A[i][j] * (2**i) for i in range(n)) + j * (2**n) for j in range(n)]

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
    return dp[m][n]

def subgraph_time(n, reps=3):
    times = []
    for _ in range(reps):
        A = [[random.randint(0,1) for _ in range(n)] for _ in range(n)]
        B = [[random.randint(0,1) for _ in range(n)] for _ in range(n)]
        t0 = time.time()
        sa = compute_signatures(A)
        sb = compute_signatures(B)
        for r in range(n):
            sb_rot = sb[r:] + sb[:r]
            lcs(sa, sb_rot)
        times.append(time.time() - t0)
    return np.mean(times)

sizes = [3, 4, 5, 6, 7, 8, 9, 10]
measured = [subgraph_time(n) for n in sizes]
theoretical = [s**3 * 1e-7 for s in sizes]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(sizes, measured, 'o-', color='#1946a2', linewidth=2, markersize=7, label='Gemessen')
axes[0].plot(sizes, theoretical, 's--', color='#b43218', linewidth=2, markersize=5, label='$c \\cdot n^3$ (theoretisch)')
axes[0].set_xlabel('Graphgröße $n$')
axes[0].set_ylabel('Laufzeit (Sekunden)')
axes[0].set_title('Empirische Laufzeit vs. theoretische Schranke')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Log-Log Plot
axes[1].loglog(sizes, measured, 'o-', color='#1946a2', linewidth=2, markersize=7, label='Gemessen')
axes[1].loglog(sizes, [s**3 * 1e-7 for s in sizes], 's--', color='#b43218', linewidth=2, label='$O(n^3)$')
axes[1].loglog(sizes, [s**2 * 1e-6 for s in sizes], '^-.', color='#1e6432', linewidth=2, label='$O(n^2)$')
axes[1].set_xlabel('Graphgröße $n$ (log)')
axes[1].set_ylabel('Laufzeit (log)')
axes[1].set_title('Log-Log-Darstellung der Skalierung')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

fig.suptitle('Plot 8: Empirische Laufzeitmessung des Subgraph Algorithmus auf Transformer-Graphen', fontsize=11)
plt.tight_layout()
plt.savefig('plot8_laufzeit_empirisch.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 erstellt")
print("\nAlle 8 Plots erfolgreich erstellt!")
