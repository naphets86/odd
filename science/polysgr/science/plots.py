import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---- Figure 1: Triangle tessellation vs quad tessellation ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
ax.set_title("Dreiecks-Tessellierung (3-Polygon)", fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        if (i+j) % 2 == 0:
            tri = plt.Polygon([[i,j],[i+1,j],[i,j+1]], closed=True, fill=True,
                               facecolor='#a8d8ea', edgecolor='#2c3e50', linewidth=0.8)
        else:
            tri = plt.Polygon([[i+1,j],[i+1,j+1],[i,j+1]], closed=True, fill=True,
                               facecolor='#f7cac9', edgecolor='#2c3e50', linewidth=0.8)
        ax.add_patch(tri)
ax.set_xlim(-0.1, 5.1)
ax.set_ylim(-0.1, 5.1)
ax.set_aspect('equal')
ax.axis('off')
ax.text(2.5, -0.3, "50 Dreiecke, 36 Vertices", ha='center', fontsize=9, style='italic')

ax = axes[1]
ax.set_title("Vierecks-Tessellierung (4-Polygon)", fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        rect = patches.Rectangle((i, j), 1, 1, linewidth=0.8,
                                   edgecolor='#2c3e50', facecolor='#b5ead7')
        ax.add_patch(rect)
ax.set_xlim(-0.1, 5.1)
ax.set_ylim(-0.1, 5.1)
ax.set_aspect('equal')
ax.axis('off')
ax.text(2.5, -0.3, "25 Vierecke, 36 Vertices", ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('fig_tessellation.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 done")

# ---- Figure 2: Vertex-sharing efficiency ----
fig, ax = plt.subplots(figsize=(8, 4))
n_polys = np.arange(1, 101)
tri_verts = 2 * n_polys + 2  # strip
quad_verts = n_polys + 1
isolated_tri = 3 * n_polys
isolated_quad = 4 * n_polys

ax.plot(n_polys, isolated_tri, 'r--', label=r'Isolierte Dreiecke: $3n$', linewidth=1.5)
ax.plot(n_polys, isolated_quad, 'b--', label=r'Isolierte Vierecke: $4n$', linewidth=1.5)
ax.plot(n_polys, tri_verts, 'r-', label=r'Dreiecksstreifen: $2n+2$', linewidth=2)
ax.plot(n_polys, quad_verts, 'b-', label=r'Vierecksstreifen: $n+1$', linewidth=2)
ax.set_xlabel("Anzahl Polygone $n$", fontsize=11)
ax.set_ylabel("Anzahl Vertices", fontsize=11)
ax.set_title("Vertex-Komplexität: Strip vs. Isoliert", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
plt.tight_layout()
plt.savefig('fig_vertex_complexity.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 done")

# ---- Figure 3: Pentagon subdivision ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

def pentagon_points(cx, cy, r):
    angles = [np.pi/2 + 2*np.pi*k/5 for k in range(5)]
    return [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]

ax = axes[0]
ax.set_title("Pentagon (5-Eck)", fontsize=12, fontweight='bold')
pts = pentagon_points(0, 0, 1)
pent = plt.Polygon(pts, closed=True, fill=True, facecolor='#ffd700', edgecolor='#2c3e50', linewidth=1.5)
ax.add_patch(pent)
for p in pts:
    ax.plot(*p, 'ko', markersize=5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.axis('off')
ax.text(0, -1.25, "1 Pentagon, 5 Vertices", ha='center', fontsize=9, style='italic')

ax = axes[1]
ax.set_title("Pentagon → 3 Dreiecke", fontsize=12, fontweight='bold')
pts = pentagon_points(0, 0, 1)
colors = ['#a8d8ea', '#f7cac9', '#b5ead7']
triangles = [(0,1,2), (0,2,3), (0,3,4)]
for i, (a,b,c) in enumerate(triangles):
    tri = plt.Polygon([pts[a], pts[b], pts[c]], closed=True, fill=True,
                       facecolor=colors[i], edgecolor='#2c3e50', linewidth=1.2)
    ax.add_patch(tri)
for p in pts:
    ax.plot(*p, 'ko', markersize=5)
# Draw diagonals
ax.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]], 'k--', linewidth=0.8)
ax.plot([pts[0][0], pts[3][0]], [pts[0][1], pts[3][1]], 'k--', linewidth=0.8)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.axis('off')
ax.text(0, -1.25, "3 Dreiecke, 5 Vertices (fan)", ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('fig_pentagon.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 done")

# ---- Figure 4: Curvature approximation ----
fig, ax = plt.subplots(figsize=(8, 4))
theta = np.linspace(0, np.pi, 500)
x_true = np.cos(theta)
y_true = np.sin(theta)

ax.plot(x_true, y_true, 'k-', linewidth=2.5, label='Exakter Halbkreis')
for n, color, ls in [(4, 'red', '--'), (8, 'blue', '-.'), (16, 'green', ':')]:
    angles = np.linspace(0, np.pi, n+1)
    xp = np.cos(angles)
    yp = np.sin(angles)
    ax.plot(xp, yp, color=color, linestyle=ls, linewidth=1.5,
            marker='o', markersize=4, label=f'{n} Dreiecke')

ax.set_aspect('equal')
ax.set_xlabel('$x$', fontsize=11)
ax.set_ylabel('$y$', fontsize=11)
ax.set_title("Krümmungsapproximation durch Dreiecks-Polygonzug", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_curvature.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 done")
