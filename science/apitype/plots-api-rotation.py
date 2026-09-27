import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 4, figsize=(15, 7))

# API-Adjazenzmatrix: Open API (4 Knoten: REST, SOAP, GraphQL, Client)
A = np.array([
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,0]
], dtype=float)

labels = ['REST','SOAP','GQL','Client']
n = 4

def draw_adj(ax, mat, lbl, title, highlight=None):
    n = mat.shape[0]
    angles = [2*np.pi*i/n - np.pi/2 for i in range(n)]
    pos = [(np.cos(a)*0.8, np.sin(a)*0.8) for a in angles]
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.axis('off')
    ax.set_title(title, fontsize=9, fontweight='bold')
    for i in range(n):
        for j in range(n):
            if mat[i,j]:
                col = '#e74c3c' if (highlight and (i,j) in highlight) else '#95a5a6'
                ax.annotate('', xy=(pos[j][0]*0.8, pos[j][1]*0.8),
                            xytext=(pos[i][0]*0.8, pos[i][1]*0.8),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    for i in range(n):
        col = '#2980b9'
        c = plt.Circle(pos[i], 0.22, color=col, ec='#1a252f', lw=1.2, zorder=3)
        ax.add_patch(c)
        ax.text(pos[i][0], pos[i][1], lbl[i], ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=4)

# Original + 3 rotations (permute columns/rows)
rotations = [A, np.roll(A, 1, axis=1), np.roll(A, 2, axis=1), np.roll(A, 3, axis=1)]
rot_labels = [
    [labels[i] for i in range(4)],
    [labels[(i-1)%4] for i in range(4)],
    [labels[(i-2)%4] for i in range(4)],
    [labels[(i-3)%4] for i in range(4)],
]
titles = ['Original (k=0)','Rotation k=1','Rotation k=2','Rotation k=3']
for i, (mat, lbl, t) in enumerate(zip(rotations, rot_labels, titles)):
    draw_adj(axes[0][i], mat, lbl, t)

# Signatur-Berechnung visualisiert
sig_vals = []
for rot_mat in rotations:
    sigs = []
    for j in range(n):
        col = rot_mat[:, j]
        s = sum(int(col[i]) * (2**i) for i in range(n)) + j * (2**n)
        sigs.append(s)
    sig_vals.append(sigs)

x = np.arange(n)
colors_rot = ['#2980b9','#27ae60','#e67e22','#8e44ad']
for i, (sigs, col, t) in enumerate(zip(sig_vals, colors_rot, titles)):
    ax = axes[1][i]
    ax.bar(x, sigs, color=col, alpha=0.85, edgecolor='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ{j}' for j in range(n)], fontsize=8)
    ax.set_title(f'Signaturen {t}', fontsize=8, fontweight='bold')
    ax.set_ylabel('Wert' if i==0 else '')
    for j,v in enumerate(sigs):
        ax.text(j, v+0.5, str(v), ha='center', va='bottom', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Zyklische Rotation des Open-API-Graphen und Signaturberechnung',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_api_rotation.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 4 done")

