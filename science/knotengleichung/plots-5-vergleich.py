"""
Plot 5: Vergleich elektrische und atmosphärische Knotengleichung
        + Graphtheoretische Modellierung mit Subgraph Algorithmus
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.patch.set_facecolor('white')
fig.suptitle('Universelle Knotengleichung: Elektrotechnik – Atmosphäre – Graphmodell',
             fontsize=12, fontweight='bold', color='#19468C')

# ---- (a) Elektrischer Knoten ----
ax = axes[0]
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(a) Elektrischer Knoten\nKirchhoff 1845', fontsize=10,
             fontweight='bold', color='#19468C')

node_circle = plt.Circle((0, 0), 0.28, color='#19468C', zorder=5)
ax.add_patch(node_circle)
ax.text(0, 0, 'K', ha='center', va='center', color='white',
        fontweight='bold', fontsize=12, zorder=6)

in_arrows = [(-2, 0.7, 1.55, -0.55, '4 A'), (-2, -0.7, 1.55, 0.55, '2 A'),
             (0, 2, 0, -1.6, '1 A')]
out_arrows = [(2, 0.7, -1.55, -0.55, '3 A'), (2, -0.7, -1.55, 0.55, '2 A'),
              (0, -2, 0, 1.6, '2 A')]

for (x0, y0, dx, dy, lbl) in in_arrows:
    ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#B4321E', lw=2.2,
                                mutation_scale=15))
    ax.text(x0 + dx*0.4, y0 + dy*0.4, lbl, fontsize=9, color='#B4321E',
            ha='center')

for (x0, y0, dx, dy, lbl) in out_arrows:
    ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#1E6432', lw=2.2,
                                mutation_scale=15))
    ax.text(x0 + dx*0.4, y0 + dy*0.4, lbl, fontsize=9, color='#1E6432',
            ha='center')

ax.text(0, -2.4, r'$\sum I_\mathrm{in} = \sum I_\mathrm{out} = 7\,\mathrm{A}$',
        ha='center', fontsize=10, color='#19468C',
        bbox=dict(boxstyle='round', fc='#EEF3FF', ec='#19468C'))

# ---- (b) Atmosphärischer Knoten ----
ax2 = axes[1]
ax2.set_xlim(-2.5, 2.5); ax2.set_ylim(-2.5, 2.5)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('(b) Atmosphärischer Knoten\nKontinuitätsgleichung', fontsize=10,
              fontweight='bold', color='#19468C')

cloud_c = plt.Circle((0, 0), 0.32, color='#5B9BD5', zorder=5)
ax2.add_patch(cloud_c)
ax2.text(0, 0, '☁', ha='center', va='center', fontsize=16, zorder=6)

atm_in = [(-2, 0.5, 1.6, -0.4, 'E=502.8\nkm³/a'), (0, 2, 0, -1.6, 'E=74.2\nkm³/a')]
atm_out = [(2, 0.5, -1.6, -0.4, 'P=458\nkm³/a'), (2, -0.8, -1.6, 0.6, 'P=119\nkm³/a'),
           (0, -2, 0, 1.6, 'T=44.8\nkm³/a')]

for (x0, y0, dx, dy, lbl) in atm_in:
    ax2.annotate('', xy=(x0+dx*0.85, y0+dy*0.85), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle='->', color='#B4321E', lw=2.2,
                                 mutation_scale=15))
    ax2.text(x0+dx*0.4, y0+dy*0.4+0.15, lbl, fontsize=7.5, color='#B4321E',
             ha='center')

for (x0, y0, dx, dy, lbl) in atm_out:
    ax2.annotate('', xy=(x0+dx*0.85, y0+dy*0.85), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle='->', color='#1E6432', lw=2.2,
                                 mutation_scale=15))
    ax2.text(x0+dx*0.4, y0+dy*0.4+0.15, lbl, fontsize=7.5, color='#1E6432',
             ha='center')

ax2.text(0, -2.4, r'$E_\mathrm{global} = P_\mathrm{global} = 577{.}000\,\mathrm{km}^3/\mathrm{a}$',
         ha='center', fontsize=9, color='#19468C',
         bbox=dict(boxstyle='round', fc='#EEF3FF', ec='#19468C'))

# ---- (c) Graphmodell mit nx ----
ax3 = axes[2]
ax3.set_title('(c) Graphmodell: Flussnetz\nSubgraph Algorithmus (Epp 2026)', fontsize=10,
              fontweight='bold', color='#19468C')

G = nx.DiGraph()
nodes = {
    'Ozean': (0.5, 0.8),
    'Atmo': (0.5, 0.2),
    'Land': (0.85, 0.5),
    'Fluss': (0.65, 0.35),
    'Gletscher': (0.15, 0.5),
}
G.add_nodes_from(nodes.keys())
edges = [('Ozean', 'Atmo', {'label': 'E=502.8', 'w': 3}),
         ('Land', 'Atmo', {'label': 'E=74.2', 'w': 2}),
         ('Atmo', 'Ozean', {'label': 'P=458', 'w': 2.5}),
         ('Atmo', 'Land', {'label': 'P=119', 'w': 2}),
         ('Land', 'Fluss', {'label': 'Abfl.', 'w': 1.5}),
         ('Fluss', 'Ozean', {'label': '44.8', 'w': 1.5}),
         ('Gletscher', 'Fluss', {'label': 'Schmelze', 'w': 1}),
         ('Ozean', 'Gletscher', {'label': 'Akk.', 'w': 1}),
]
for (u, v, d) in edges:
    G.add_edge(u, v, **d)

pos = nodes
node_colors = {'Ozean': '#5B9BD5', 'Atmo': '#A8D4F5',
               'Land': '#8DC455', 'Fluss': '#2E75B6', 'Gletscher': '#BDD7EE'}
nc = [node_colors[n] for n in G.nodes()]
ns = [1800 if n in ['Ozean', 'Atmo'] else 1200 for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax3, node_color=nc, node_size=ns, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax3, font_size=8, font_weight='bold')
nx.draw_networkx_edges(G, pos, ax=ax3, edge_color='#19468C',
                       width=[d['w'] for _, _, d in G.edges(data=True)],
                       arrows=True, arrowsize=18, alpha=0.8,
                       connectionstyle='arc3,rad=0.12')
edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax3,
                              font_size=7, label_pos=0.35,
                              bbox=dict(boxstyle='round,pad=0.2',
                                        fc='white', ec='none', alpha=0.7))
ax3.text(0.5, 0.02, r'Knotengleichung: $\sum_k f_k = 0\;\forall$ Knoten',
         transform=ax3.transAxes, ha='center', fontsize=9, color='#19468C',
         bbox=dict(boxstyle='round,pad=0.3', fc='#EEF3FF', ec='#19468C'))
ax3.axis('off')

plt.tight_layout(pad=2.0)
plt.savefig('plot5_vergleich.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot 5 gespeichert.")

