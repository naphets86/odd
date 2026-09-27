import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Plot 2: Subgraph-Matching API-Graphen ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

def draw_graph(ax, adj, labels, title, color='#2980b9'):
    n = len(labels)
    angles = [2*np.pi*i/n for i in range(n)]
    pos = {i: (np.cos(a), np.sin(a)) for i,a in enumerate(angles)}
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                x1,y1 = pos[i]; x2,y2 = pos[j]
                ax.annotate('', xy=(x2*0.82, y2*0.82), xytext=(x1*0.82, y1*0.82),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    for i,(lbl) in enumerate(labels):
        x,y = pos[i]
        circ = plt.Circle((x,y), 0.18, color=color, ec='#2c3e50', lw=1.5, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=4)

# G: REST → Backend (3 Knoten)
adj_G = [[0,1,0],[0,0,1],[0,0,0]]
draw_graph(axes[0], adj_G, ['REST','B2B','Pay'], 'Graph G (REST-Kette)', '#2980b9')

# G': Internal API Graph (5 Knoten, enthält G als Subgraph)
adj_Gp = [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]]
draw_graph(axes[1], adj_Gp, ['REST','B2B','Pay','DB','Auth'],
           "Graph G' (Erweitertes Netz)", '#27ae60')

# Signaturen als Balkendiagramm
ax = axes[2]
sigs_G  = [5, 18, 36]
sigs_Gp = [5, 18, 36, 64, 80]
x1 = np.arange(len(sigs_G))
x2 = np.arange(len(sigs_Gp))
ax.bar(x1, sigs_G, color='#2980b9', alpha=0.8, label='G')
ax.bar(x2+0.15, sigs_Gp[:len(sigs_Gp)], color='#27ae60', alpha=0.7, label="G'", width=0.4)
ax.set_xticks(range(max(len(sigs_G), len(sigs_Gp))))
ax.set_xticklabels([f'σ{i}' for i in range(max(len(sigs_G), len(sigs_Gp)))], fontsize=9)
ax.set_ylabel('Signaturwert')
ax.set_title('Signatur-Vergleich: G ⊆ G\'', fontsize=10, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Subgraph Algorithmus auf API-Graphen', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_api_subgraph.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 2 done")


