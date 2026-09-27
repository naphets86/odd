import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---- Fig 1: Subgraph-Repräsentation eines Dreiecks als Graph ----
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

def draw_polygon_graph(ax, n, title, color):
    angles = [np.pi/2 + 2*np.pi*k/n for k in range(n)]
    pts = [(np.cos(a), np.sin(a)) for a in angles]
    # Kanten
    for i in range(n):
        j = (i+1) % n
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                'k-', linewidth=1.8, zorder=1)
    # Knoten
    for i, p in enumerate(pts):
        ax.plot(*p, 'o', color=color, markersize=14, zorder=2)
        ax.text(p[0]*1.22, p[1]*1.22, f'$v_{i}$', ha='center', va='center', fontsize=11)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold')

draw_polygon_graph(axes[0], 3, 'Dreieck $T_3$\n($n=3$, $|E|=3$)', '#1946a0')
draw_polygon_graph(axes[1], 5, 'Pentagon $P_5$\n($n=5$, $|E|=5$)', '#b43220')
draw_polygon_graph(axes[2], 6, 'Hexagon $H_6$\n($n=6$, $|E|=6$)', '#1e6432')

plt.tight_layout()
plt.savefig('fig_poly_graphs.pdf', bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ---- Fig 2: Adjazenzmatrix des Dreiecks und Pentagons ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

def draw_adj_matrix(ax, n, title):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i+1) % n] = 1
        A[(i+1) % n, i] = 1
    im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(A[i,j]), ha='center', va='center',
                    fontsize=13, color='white' if A[i,j] else '#444')
    labels = [f'$v_{k}$' for k in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

draw_adj_matrix(axes[0], 3, 'Adjazenzmatrix $A_{T_3}$ (Dreieck)')
draw_adj_matrix(axes[1], 5, 'Adjazenzmatrix $A_{P_5}$ (Pentagon)')
plt.tight_layout()
plt.savefig('fig_adj_matrices.pdf', bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ---- Fig 3: Signaturen und Rotationen ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

def draw_rotation(ax, sigs, rot, title):
    n = len(sigs)
    rotated = sigs[rot:] + sigs[:rot]
    colors = ['#1946a0' if i < n-rot else '#b43220' for i in range(n)]
    bars = ax.bar(range(n), rotated, color=colors, edgecolor='black', linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars, rotated)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'$\\sigma_{i}$' for i in range(n)])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(rotated)*1.2 + 1)
    ax.set_ylabel('Signaturwert')

# Signaturen für Dreieck (n=3)
sigs_tri = [5, 19, 34]  # Beispielwerte
for idx, rot in enumerate([0, 1, 2, 3]):
    r = idx % 3
    draw_rotation(axes[idx//2][idx%2], sigs_tri, r,
                  f'Rotation $r={r}$ der Dreieck-Signaturen')

plt.suptitle('Zyklische Rotationen der Signaturfolge (Dreieck $T_3$)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_rotations.pdf', bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ---- Fig 4: Effizienzvergleich Polygon-Typen ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ns = np.arange(3, 13)
# Effizienzmaß
eta = (ns/2 * np.sin(2*np.pi/ns)) / (2*(ns-2))
axes[0].bar(ns, eta, color=['#1946a0' if n==3 else ('#b43220' if n==5 else '#aaa') for n in ns],
            edgecolor='black', linewidth=0.8)
axes[0].set_xlabel('Eckenzahl $n$', fontsize=11)
axes[0].set_ylabel(r'$\eta(n)$', fontsize=11)
axes[0].set_title(r'Effizienzmaß $\eta(n)$ nach Polygon-Typ', fontsize=11, fontweight='bold')
axes[0].set_xticks(ns)
axes[0].axhline(y=eta[0], color='#1946a0', linestyle='--', linewidth=1, alpha=0.6, label='$\eta(3)$ Optimum')
axes[0].legend(fontsize=9)
for i, (n, e) in enumerate(zip(ns, eta)):
    axes[0].text(n, e+0.005, f'{e:.3f}', ha='center', fontsize=7.5, fontweight='bold')

# Pentagon zu Dreiecke Zerlegung: Verhältnis Vertices/Fläche
n_arr = np.arange(3, 13)
# Dreiecke nach Zerlegung
tri_count = n_arr - 2
axes[1].bar(n_arr, tri_count,
            color=['#1946a0' if n==3 else ('#b43220' if n==5 else '#aaa') for n in n_arr],
            edgecolor='black', linewidth=0.8)
axes[1].set_xlabel('Eckenzahl $n$', fontsize=11)
axes[1].set_ylabel('Anzahl Dreiecke nach Zerlegung', fontsize=11)
axes[1].set_title('Dreiecksanzahl $n-2$ bei Fächerzerlegung', fontsize=11, fontweight='bold')
axes[1].set_xticks(n_arr)
for n, t in zip(n_arr, tri_count):
    axes[1].text(n, t+0.05, str(t), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_efficiency.pdf', bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ---- Fig 5: Subgraph-Test Pentagon -> Dreieck ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Pentagon
ax = axes[0]
ax.set_title('Pentagon $P_5$\n(Eingabe-Graph)', fontsize=10, fontweight='bold')
angles5 = [np.pi/2 + 2*np.pi*k/5 for k in range(5)]
pts5 = [(np.cos(a), np.sin(a)) for a in angles5]
for i in range(5):
    j = (i+1)%5
    ax.plot([pts5[i][0], pts5[j][0]], [pts5[i][1], pts5[j][1]], 'k-', lw=1.8)
for i, p in enumerate(pts5):
    ax.plot(*p, 'o', color='#b43220', markersize=12, zorder=3)
    ax.text(p[0]*1.3, p[1]*1.3, f'$v_{i}$', ha='center', va='center', fontsize=10)
ax.set_xlim(-1.7,1.7); ax.set_ylim(-1.7,1.7); ax.set_aspect('equal'); ax.axis('off')

# Subgraph-Relation: Dreieck in Pentagon?
ax = axes[1]
ax.set_title('Subgraph-Test:\n$T_3 \\subseteq P_5$?', fontsize=10, fontweight='bold')
for i in range(5):
    j = (i+1)%5
    ax.plot([pts5[i][0], pts5[j][0]], [pts5[i][1], pts5[j][1]], 'k-', lw=1.2, alpha=0.3)
# highlight triangle v0, v1, v2
tri_idx = [0,1,2]
for i in range(3):
    j = tri_idx[(i+1)%3]
    ii = tri_idx[i]
    ax.plot([pts5[ii][0], pts5[j][0]], [pts5[ii][1], pts5[j][1]], 'b-', lw=2.5)
for i, p in enumerate(pts5):
    col = '#1946a0' if i in tri_idx else '#aaa'
    ax.plot(*p, 'o', color=col, markersize=12, zorder=3)
    ax.text(p[0]*1.3, p[1]*1.3, f'$v_{i}$', ha='center', va='center', fontsize=10)
ax.text(0, -1.55, 'Nicht direkt: kein gemeinsamer Zyklus', ha='center', fontsize=8.5, style='italic', color='#b43220')
ax.set_xlim(-1.7,1.7); ax.set_ylim(-1.7,1.7); ax.set_aspect('equal'); ax.axis('off')

# Fächerzerlegung
ax = axes[2]
ax.set_title('Fächerzerlegung:\n$P_5 \\to 3 \\cdot T_3$', fontsize=10, fontweight='bold')
colors_fan = ['#cce4ff', '#ffd6d0', '#d4f0d4']
tris = [(0,1,2),(0,2,3),(0,3,4)]
for (a,b,c), col in zip(tris, colors_fan):
    tri = plt.Polygon([pts5[a], pts5[b], pts5[c]], closed=True,
                       facecolor=col, edgecolor='#1946a0', linewidth=1.5, zorder=1)
    ax.add_patch(tri)
for i, p in enumerate(pts5):
    ax.plot(*p, 'o', color='#1946a0', markersize=12, zorder=3)
    ax.text(p[0]*1.3, p[1]*1.3, f'$v_{i}$', ha='center', va='center', fontsize=10)
ax.text(0,-1.55,'3 Dreiecke, kein neuer Vertex', ha='center', fontsize=8.5, style='italic', color='#1946a0')
ax.set_xlim(-1.7,1.7); ax.set_ylim(-1.7,1.7); ax.set_aspect('equal'); ax.axis('off')

plt.tight_layout()
plt.savefig('fig_subgraph_penta.pdf', bbox_inches='tight')
plt.close()
print("Fig 5 done")

print("All figures done.")
