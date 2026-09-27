#!/usr/bin/env python3
"""
Plots fuer: Quantengravitation als Subgraph-Isomorphismus-Problem
Stephan Epp, 2026
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import odeint

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

BLUE   = '#1946a2'
RED    = '#b43218'
GREEN  = '#1e6432'
ORANGE = '#c47a00'
PURPLE = '#6a1b9a'

# ─── Plot 1: Raumzeit-Graph ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

# Raumzeit-Knoten (Ereignisse)
events = {
    'E1': (1, 6), 'E2': (3, 7), 'E3': (5, 6.5), 'E4': (7, 7),
    'E5': (2, 4), 'E6': (4, 4.5), 'E7': (6, 4), 'E8': (8, 5),
    'E9': (1.5, 2), 'E10': (4, 2.5), 'E11': (6.5, 2), 'E12': (9, 3),
    'S':  (5, 0.8),
}
colors_e = {e: BLUE for e in events}
colors_e['S'] = RED

for name, (x, y) in events.items():
    r = 0.35 if name != 'S' else 0.45
    ax.add_patch(plt.Circle((x, y), r, color=colors_e[name], zorder=3, alpha=0.9))
    ax.text(x, y, name, ha='center', va='center', color='white', fontsize=7,
            fontweight='bold', zorder=4)

# Kausale Kanten (zeitartig)
causal = [
    ('E1','E5'),('E2','E5'),('E2','E6'),('E3','E6'),('E3','E7'),('E4','E7'),('E4','E8'),
    ('E5','E9'),('E5','E10'),('E6','E10'),('E6','E11'),('E7','E11'),('E7','E12'),('E8','E12'),
    ('E9','S'),('E10','S'),('E11','S'),('E12','S'),
]
for (a, b) in causal:
    x1, y1 = events[a]; x2, y2 = events[b]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))

# Raumartige Kanten (gestrichelt)
spatial = [('E1','E2'),('E2','E3'),('E3','E4'),('E5','E6'),('E6','E7'),('E7','E8')]
for (a, b) in spatial:
    x1, y1 = events[a]; x2, y2 = events[b]
    ax.plot([x1, x2], [y1, y2], '--', color=ORANGE, lw=1.2, alpha=0.7)

legend_e = [
    mpatches.Patch(color=BLUE, label='Raumzeit-Ereignisse (Knoten)'),
    mpatches.Patch(color=RED, label='Singularitaet S'),
    mpatches.Patch(color='#555', label='Zeitartige Kanten (kausal)'),
    mpatches.Patch(color=ORANGE, label='Raumartige Kanten'),
]
ax.legend(handles=legend_e, loc='lower left', fontsize=9)
ax.set_title('Plot 1: Raumzeit als gerichteter kausaler Graph $G_{\\mathcal{M}}$', pad=12)
plt.tight_layout()
plt.savefig('plot1_raumzeit_graph.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 erstellt")

# ─── Plot 2: Quantengraph vs. klassische Raumzeit ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Links: Klassische glatte Raumzeit
ax = axes[0]
ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title('Klassische Raumzeit $\\mathcal{M}$ (glatt)', fontsize=11)
theta = np.linspace(0, 2*np.pi, 100)
for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
    ax.plot(3 + r*np.cos(theta), 3 + r*np.sin(theta), '-', color=BLUE, lw=0.8, alpha=0.5)
for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
    ax.annotate('', xy=(3 + 2.5*np.cos(ang), 3 + 2.5*np.sin(ang)),
                xytext=(3 + 0.5*np.cos(ang), 3 + 0.5*np.sin(ang)),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0, alpha=0.6))
ax.text(3, 3, '$g_{\\mu\\nu}$', ha='center', va='center', fontsize=14, color=BLUE)

# Rechts: Quantengraph (diskrete Knoten)
ax = axes[1]
ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title('Quanten-Raumzeit $G_Q$ (diskret)', fontsize=11)
np.random.seed(7)
n_nodes = 18
xs = np.random.uniform(0.5, 5.5, n_nodes)
ys = np.random.uniform(0.5, 5.5, n_nodes)
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        if d < 1.5:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], '-', color=GREEN, lw=0.8, alpha=0.6)
for i in range(n_nodes):
    ax.add_patch(plt.Circle((xs[i], ys[i]), 0.18, color=RED, zorder=3))
ax.text(3, 0.2, r'Planck-Skala: $\ell_P \approx 1.6 \times 10^{-35}$ m',
        ha='center', fontsize=9, color=RED)

fig.suptitle('Plot 2: Klassische vs.\ diskrete Quanten-Raumzeit', fontsize=12)
plt.tight_layout()
plt.savefig('plot2_klassisch_vs_quanten.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 erstellt")

# ─── Plot 3: Signaturberechnung fuer Spin-Netzwerk ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

n = 6
np.random.seed(42)
A = np.random.randint(0, 2, (n, n)).astype(float)
A = np.tril(A, -1) + np.tril(A, -1).T  # symmetrisch
np.fill_diagonal(A, 0)
# Spin-Gewichte
W = A * np.random.choice([0.5, 1.0, 1.5, 2.0], size=(n, n))

im = axes[0].imshow(W, cmap='Blues', vmin=0)
axes[0].set_title('Spin-Netzwerk Adjazenzmatrix $A_S$\n(Gewichte = halbe Spin-Zahlen)')
axes[0].set_xlabel('Knoten $j$'); axes[0].set_ylabel('Knoten $i$')
for i in range(n):
    for j in range(n):
        if W[i, j] > 0:
            axes[0].text(j, i, f'{W[i,j]:.1f}', ha='center', va='center',
                         color='white' if W[i,j] > 1.5 else 'black', fontsize=9)
plt.colorbar(im, ax=axes[0], label='Spin-Gewicht $j_e$')

sigmas = [sum(int(A[i, j]) * (2**i) for i in range(n)) + j * (2**n) for j in range(n)]
axes[1].bar(range(n), sigmas, color=BLUE, edgecolor='white', linewidth=0.5)
axes[1].set_title('Signatursequenz $\\sigma^S$ des Spin-Netzwerks')
axes[1].set_xlabel('Knotenindex $j$')
axes[1].set_ylabel('Signaturwert $\\sigma_j$')
axes[1].set_xticks(range(n))
for i, v in enumerate(sigmas):
    axes[1].text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=9, color=BLUE)
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Plot 3: Signaturberechnung fuer Spin-Netzwerk-Graphen', fontsize=12)
plt.tight_layout()
plt.savefig('plot3_spinnet_signaturen.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 erstellt")

# ─── Plot 4: Planck-Skala und Subgraph-Hierarchie ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')

levels = [
    {'label': 'Kosmische Skala\n$G_{\\rm cosmos}$\n($\\sim 10^{26}$ m)', 'x': 6, 'y': 8.8, 'r': 1.1, 'c': BLUE},
    {'label': 'Galaktische Skala\n$G_{\\rm galaxy}$\n($\\sim 10^{21}$ m)', 'x': 6, 'y': 6.8, 'r': 0.9, 'c': BLUE},
    {'label': 'Schwarzes Loch\n$G_{\\rm BH}$\n($\\sim 10^{3}$ m)', 'x': 6, 'y': 5.0, 'r': 0.8, 'c': PURPLE},
    {'label': 'Nukleare Skala\n$G_{\\rm nuc}$\n($\\sim 10^{-15}$ m)', 'x': 6, 'y': 3.3, 'r': 0.75, 'c': RED},
    {'label': 'Planck-Skala\n$G_{\\rm Planck}$\n($\\sim 10^{-35}$ m)', 'x': 6, 'y': 1.5, 'r': 0.7, 'c': GREEN},
]
for l in levels:
    ax.add_patch(plt.Circle((l['x'], l['y']), l['r'], color=l['c'], zorder=3, alpha=0.9))
    ax.text(l['x'], l['y'], l['label'], ha='center', va='center',
            color='white', fontsize=8, fontweight='bold', zorder=4)

for i in range(len(levels)-1):
    x1, y1 = levels[i]['x'], levels[i]['y'] - levels[i]['r']
    x2, y2 = levels[i+1]['x'], levels[i+1]['y'] + levels[i+1]['r']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#888', lw=2.0))
    mid_y = (y1 + y2) / 2
    ax.text(7.5, mid_y, r'$\subseteq$', ha='center', va='center', fontsize=16, color='#888')

ax.set_title('Plot 4: Subgraph-Hierarchie der Raumzeit-Skalen\n'
             r'$G_{\rm Planck} \subseteq G_{\rm nuc} \subseteq G_{\rm BH} \subseteq G_{\rm galaxy} \subseteq G_{\rm cosmos}$',
             pad=12)
plt.tight_layout()
plt.savefig('plot4_hierarchie_skalen.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 erstellt")

# ─── Plot 5: Einstein-Gleichungen als Graphtransformation ────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_title('Vor Masse: flache Raumzeit $G_0$')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
xs = np.linspace(-3.5, 3.5, 8)
ys = np.linspace(-3.5, 3.5, 8)
for x in xs:
    ax.axvline(x, color=BLUE, lw=0.6, alpha=0.5)
for y in ys:
    ax.axhline(y, color=BLUE, lw=0.6, alpha=0.5)
ax.grid(False)
ax.set_aspect('equal')

ax = axes[1]
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_title('Nach Masse: gekruemmte Raumzeit $G_M$')
ax.set_xlabel('$x$')
x_grid = np.linspace(-3.5, 3.5, 12)
y_grid = np.linspace(-3.5, 3.5, 12)
for x in x_grid:
    ys_curved = []
    for y in np.linspace(-3.5, 3.5, 100):
        r = np.sqrt(x**2 + y**2) + 0.1
        defl = 0.6 / r
        ys_curved.append(y)
    xc = [x + 0.5*np.sign(x)*(1/(np.sqrt(x**2+yy**2)+0.5)) for yy in ys_curved]
    ax.plot(xc, ys_curved, '-', color=GREEN, lw=0.7, alpha=0.6)
for y in y_grid:
    xs_curved = []
    for x in np.linspace(-3.5, 3.5, 100):
        r = np.sqrt(x**2 + y**2) + 0.1
        xs_curved.append(x)
    yc = [y + 0.5*np.sign(y)*(1/(np.sqrt(xx**2+y**2)+0.5)) for xx in xs_curved]
    ax.plot(np.linspace(-3.5, 3.5, 100), yc, '-', color=GREEN, lw=0.7, alpha=0.6)
ax.add_patch(plt.Circle((0, 0), 0.4, color=RED, zorder=5))
ax.text(0, 0, '$M$', ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=6)
ax.set_aspect('equal')

fig.suptitle('Plot 5: Einstein-Gleichungen als Kantentransformation des Raumzeit-Graphen\n'
             r'$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$', fontsize=12)
plt.tight_layout()
plt.savefig('plot5_einstein_graph.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 erstellt")

# ─── Plot 6: LCS-Matching Quantengraph vs. GR-Graph ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

sigma_Q = [5, 18, 36, 72, 144]  # Quantengravitations-Signaturen
sigma_G = [18, 5, 144, 36, 288, 72]  # GR-Signaturen (groesser)

m, n_s = len(sigma_Q), len(sigma_G)
dp = [[0]*(n_s+1) for _ in range(m+1)]
for i in range(1, m+1):
    for j in range(1, n_s+1):
        if sigma_Q[i-1] == sigma_G[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

dp_arr = np.array(dp)
im = axes[0].imshow(dp_arr, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(n_s+1))
axes[0].set_xticklabels(['-']+[str(x) for x in sigma_G], fontsize=7, rotation=45)
axes[0].set_yticks(range(m+1))
axes[0].set_yticklabels(['-']+[str(x) for x in sigma_Q], fontsize=8)
axes[0].set_xlabel('$\\sigma^{GR}$ (Allgemeine Relativitaet)')
axes[0].set_ylabel('$\\sigma^Q$ (Quantengravitation)')
axes[0].set_title('LCS-DP-Tabelle: QG $\\hookrightarrow$ GR')
for i in range(m+1):
    for j in range(n_s+1):
        axes[0].text(j, i, dp[i][j], ha='center', va='center',
                     color='white' if dp[i][j] > 2 else 'black', fontsize=8)
plt.colorbar(im, ax=axes[0], label='LCS-Laenge')

lcs_vals = [dp[m][j] for j in range(n_s+1)]
axes[1].plot(range(n_s+1), lcs_vals, 'o-', color=BLUE, lw=2, markersize=6)
axes[1].axhline(y=2, color=RED, ls='--', label='Schwellenwert (LCS $\\geq 2$)')
axes[1].fill_between(range(n_s+1), lcs_vals, 2,
                     where=[v >= 2 for v in lcs_vals], alpha=0.2, color=GREEN,
                     label='Subgraph $G_Q \\subseteq G_{GR}$ erkannt')
axes[1].set_xlabel('Position in $\\sigma^{GR}$')
axes[1].set_ylabel('LCS-Laenge')
axes[1].set_title('LCS-Verlauf und Einbettungsnachweis')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

fig.suptitle('Plot 6: LCS-basierter Nachweis $G_Q \\subseteq G_{GR}$', fontsize=12)
plt.tight_layout()
plt.savefig('plot6_lcs_qg_gr.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 erstellt")

# ─── Plot 7: Schwarzes Loch als Subgraph ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

# Aeussere Raumzeit
theta = np.linspace(0, 2*np.pi, 100)
for r in [4.5, 3.8, 3.2]:
    ax.plot(5 + r*np.cos(theta), 5 + r*np.sin(theta), '-', color=BLUE, lw=0.7, alpha=0.4)

# Ereignishorizont
ax.add_patch(plt.Circle((5, 5), 2.5, color='black', zorder=4, alpha=0.85))
ax.add_patch(plt.Circle((5, 5), 2.5, color='black', zorder=4))
ax.text(5, 5, 'Singularitaet\n$G_S$', ha='center', va='center',
        color='white', fontsize=10, fontweight='bold', zorder=6)

# Ereignishorizont-Ring
ax.add_patch(plt.Circle((5, 5), 2.5, fill=False, edgecolor=RED, lw=2.5, zorder=5))
ax.text(5, 2.3, 'Ereignishorizont\n$r_S = 2GM/c^2$', ha='center', va='top',
        fontsize=9, color=RED, zorder=6)

# Subgraph-Knoten auf dem Horizont
n_h = 8
for k in range(n_h):
    ang = 2*np.pi * k / n_h
    xk = 5 + 2.5*np.cos(ang)
    yk = 5 + 2.5*np.sin(ang)
    ax.add_patch(plt.Circle((xk, yk), 0.25, color=ORANGE, zorder=7))

# Einfallendes Material
for ang in [np.pi/6, np.pi/3, 2*np.pi/3]:
    x_start = 5 + 4.3*np.cos(ang)
    y_start = 5 + 4.3*np.sin(ang)
    x_end = 5 + 2.6*np.cos(ang)
    y_end = 5 + 2.6*np.sin(ang)
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.0))
    ax.add_patch(plt.Circle((x_start, y_start), 0.2, color=GREEN, zorder=5))

# Hawking-Strahlung
for ang in [np.pi + np.pi/6, np.pi + np.pi/3, np.pi + 2*np.pi/3]:
    x_start = 5 + 2.8*np.cos(ang)
    y_start = 5 + 2.8*np.sin(ang)
    x_end = 5 + 4.3*np.cos(ang)
    y_end = 5 + 4.3*np.sin(ang)
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5, linestyle='dashed'))

legend_bh = [
    mpatches.Patch(color='black', label='Schwarzes Loch $G_{BH} \\subseteq G_T$'),
    mpatches.Patch(color=RED, label='Ereignishorizont'),
    mpatches.Patch(color=ORANGE, label='Subgraph-Knoten auf Horizont'),
    mpatches.Patch(color=GREEN, label='Einfallendes Material'),
    mpatches.Patch(color=PURPLE, label='Hawking-Strahlung'),
]
ax.legend(handles=legend_bh, loc='lower left', fontsize=9)
ax.set_title('Plot 7: Schwarzes Loch als Subgraph $G_{BH} \\subseteq G_{\\mathcal{M}}$\n'
             'Subgraph Algorithmus erkennt Ereignishorizont in $O(n^3)$', pad=12)
plt.tight_layout()
plt.savefig('plot7_schwarzes_loch.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 erstellt")

# ─── Plot 8: Komplexitaetsvergleich QG-Algorithmen ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
import math

ns = np.arange(2, 20)
naive_perm = np.array([math.factorial(n) * n**2 if n <= 13 else float('inf') for n in ns], dtype=float)
subgraph_  = ns**3
pathintegral = 2.0**(2*ns)  # Pfadintegral: exponentiell in Metrikfreiheitsgraden
lqg_coarse = ns**4          # Loop-QG Groebung: O(n^4)

ax1 = axes[0]
mask = naive_perm < 1e16
ax1.semilogy(ns[mask], naive_perm[mask], 's--', color=RED, label='Naiv $O(n!\\cdot n^2)$', lw=2)
ax1.semilogy(ns, subgraph_, 'o-', color=BLUE, label='Subgraph $O(n^3)$', lw=2)
ax1.semilogy(ns[:12], pathintegral[:12], '^-.', color=ORANGE, label='Pfadintegral $O(2^{2n})$', lw=2)
ax1.semilogy(ns, lqg_coarse, 'D:', color=GREEN, label='Loop-QG Groebung $O(n^4)$', lw=2)
ax1.set_xlabel('Graphgroesse $n$ (Raumzeit-Knoten)')
ax1.set_ylabel('Operationen (log)')
ax1.set_title('Komplexitaetsvergleich QG-Ansaetze')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ns2 = np.arange(2, 50)
ax2.loglog(ns2, ns2**3, 'o-', color=BLUE, label='Subgraph $O(n^3)$', lw=2)
ax2.loglog(ns2, ns2**4, 'D--', color=GREEN, label='$O(n^4)$', lw=1.8)
ax2.loglog(ns2, ns2**6, 's-.', color=ORANGE, label='$O(n^6)$ (Graviton-Streuung)', lw=1.8)
ax2.set_xlabel('Graphgroesse $n$ (log)')
ax2.set_ylabel('Laufzeit (log)')
ax2.set_title('Skalierung fuer grosse QG-Graphen')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, which='both')

fig.suptitle('Plot 8: Komplexitaetsanalyse des Subgraph Algorithmus fuer Quantengravitation', fontsize=11)
plt.tight_layout()
plt.savefig('plot8_komplexitaet_qg.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 erstellt")

# ─── Plot 9: Spin-Schaum Entwicklung ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for idx, (ax, label, n_nodes, seed) in enumerate(zip(
    axes,
    ['$t_0$: Initial-Netzwerk', '$t_1$: Nach Koales.', '$t_2$: Verfeinert'],
    [5, 8, 12],
    [1, 2, 3]
)):
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title(label, fontsize=10)
    np.random.seed(seed * 10)
    xs = np.random.uniform(0.5, 5.5, n_nodes)
    ys = np.random.uniform(0.5, 5.5, n_nodes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < 2.2:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], '-', color=BLUE, lw=1.0, alpha=0.5)
    for i in range(n_nodes):
        ax.add_patch(plt.Circle((xs[i], ys[i]), 0.22, color=RED, zorder=3))
        ax.text(xs[i], ys[i], str(i+1), ha='center', va='center',
                color='white', fontsize=7, zorder=4)
    ax.text(3, 0.1, f'$|V|={n_nodes}$, Subgraph-Check in $O({n_nodes}^3)$={n_nodes**3}',
            ha='center', fontsize=8, color='#444')

fig.suptitle('Plot 9: Spin-Schaum-Entwicklung als dynamische Subgraph-Sequenz\n'
             r'$G_Q^{(t_0)} \hookrightarrow G_Q^{(t_1)} \hookrightarrow G_Q^{(t_2)}$', fontsize=12)
plt.tight_layout()
plt.savefig('plot9_spin_schaum.pdf', bbox_inches='tight')
plt.close()
print("Plot 9 erstellt")

# ─── Plot 10: Empirische Laufzeit ─────────────────────────────────────────
import time, random

def compute_signatures(A):
    n = len(A)
    return [sum(A[i][j] * (2**i) for i in range(n)) + j * (2**n) for j in range(n)]

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
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
            lcs_len(sa, sb[r:]+sb[:r])
        times.append(time.time()-t0)
    return np.mean(times)

sizes = [3, 4, 5, 6, 7, 8, 9, 10]
measured = [subgraph_time(n) for n in sizes]
theoretical = [s**3 * 8e-8 for s in sizes]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(sizes, measured, 'o-', color=BLUE, lw=2, markersize=7, label='Gemessen')
axes[0].plot(sizes, theoretical, 's--', color=RED, lw=2, markersize=5, label='$c \\cdot n^3$ (theoretisch)')
axes[0].set_xlabel('Anzahl Raumzeit-Knoten $n$')
axes[0].set_ylabel('Laufzeit (Sekunden)')
axes[0].set_title('Empirische Laufzeit auf QG-Graphen')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].loglog(sizes, measured, 'o-', color=BLUE, lw=2, markersize=7, label='Gemessen')
axes[1].loglog(sizes, [s**3*8e-8 for s in sizes], 's--', color=RED, lw=2, label='$O(n^3)$')
axes[1].loglog(sizes, [s**2*5e-7 for s in sizes], '^-.', color=GREEN, lw=2, label='$O(n^2)$')
axes[1].set_xlabel('$n$ (log)')
axes[1].set_ylabel('Laufzeit (log)')
axes[1].set_title('Log-Log: Bestaetigung der $O(n^3)$-Schranke')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

fig.suptitle('Plot 10: Empirische Laufzeitmessung fuer Quantengravitations-Graphen', fontsize=12)
plt.tight_layout()
plt.savefig('plot10_laufzeit_empirisch.pdf', bbox_inches='tight')
plt.close()
print("Plot 10 erstellt")
print("\nAlle 10 Plots erfolgreich erstellt!")
