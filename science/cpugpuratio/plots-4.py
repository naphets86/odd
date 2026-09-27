#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1,2,figsize=(13,5.5))

# --- Links: Topologie-Graphen ---
ax = axes[0]
ax.set_xlim(-0.3,3.8); ax.set_ylim(-0.3,3.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('Abbildung 4a: Hardware-Topologien als Subgraph-Instanzen', fontsize=11, fontweight='bold')

def node(ax, x, y, lbl, col, r=0.19):
    ax.add_patch(plt.Circle((x,y),r,color=col,zorder=5))
    ax.text(x,y,lbl,ha='center',va='center',fontsize=8.5,fontweight='bold',color='white',zorder=6)

def edge(ax,x1,y1,x2,y2,col='#555'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=1.6))

# 1:1
ax.text(0.5,3.3,'G_{1:1}',fontsize=10,ha='center',fontweight='bold',color='#1f77b4')
node(ax,0.15,2.75,'C','#1f77b4'); node(ax,0.85,2.75,'G','#ff7f0e')
edge(ax,0.34,2.75,0.66,2.75,'#1f77b4')

# 1:2
ax.text(2.5,3.3,'G_{1:2}',fontsize=10,ha='center',fontweight='bold',color='#2ca02c')
node(ax,2.5,2.8,'C','#2ca02c')
node(ax,1.9,2.2,'G','#ff7f0e'); node(ax,3.1,2.2,'G','#ff7f0e')
edge(ax,2.37,2.62,2.06,2.38,'#2ca02c')
edge(ax,2.63,2.62,2.94,2.38,'#2ca02c')

# 1:4
ax.text(1.75,1.7,'G_{1:4}',fontsize=10,ha='center',fontweight='bold',color='#9467bd')
node(ax,1.75,1.35,'C','#9467bd')
gpu_pos=[(0.7,0.55),(1.25,0.55),(2.25,0.55),(2.8,0.55)]
for gx,gy in gpu_pos:
    node(ax,gx,gy,'G','#ff7f0e')
    edge(ax,1.75,1.16,gx,gy+0.19,'#9467bd')

# Subgraph-Relation Annotation
ax.annotate('',xy=(0.85,2.6),xytext=(1.9,2.2),
            arrowprops=dict(arrowstyle='-[',color='gray',lw=1.0,linestyle='dashed'))
ax.text(1.4,2.47,'$G_{1:1}\\subseteq G_{1:2}$',fontsize=8,color='gray',ha='center')

# --- Rechts: LCS-Matching-Verlauf über Rotationen ---
ax2 = axes[1]
rotations = np.arange(0,8)
# Simulierter LCS-Wert für G_(1:1) gegen rotierte G_(1:2), G_(1:4)
lcs_12 = [2,3,2,1,3,2,1,2]
lcs_14 = [1,2,1,2,1,3,2,1]
lcs_11 = [4,3,4,4,3,4,3,4]  # gegen sich selbst (max)

ax2.plot(rotations,lcs_11,'#1f77b4',lw=2.2,marker='o',ms=7,label='LCS($G_{1:1}$, $G_{1:1}$)')
ax2.plot(rotations,lcs_12,'#2ca02c',lw=2.2,marker='s',ms=7,label='LCS($G_{1:1}$, $G_{1:2}$)')
ax2.plot(rotations,lcs_14,'#9467bd',lw=2.2,marker='^',ms=7,label='LCS($G_{1:1}$, $G_{1:4}$)')
ax2.axhline(2,color='red',lw=1.3,ls='--',label='Schwellwert LCS $\\geq 2$')

ax2.set_xlabel('Rotationsindex $r$', fontsize=11)
ax2.set_ylabel('LCS-Länge', fontsize=11)
ax2.set_title('Abbildung 4b: LCS-Matching über zyklische Rotationen\n(Subgraph-Erkennung)', fontsize=11, fontweight='bold')
ax2.set_xticks(rotations)
ax2.set_ylim(0,5.5)
ax2.legend(fontsize=9)
ax2.grid(True,alpha=0.3,linestyle='--')

plt.tight_layout()
plt.savefig('plots/plot4_topology_lcs.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 OK")

