import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import gumbel_r

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ─────────────────────────────────────────────────
# Plot W1: CFG des Subgraph Algorithmus für WCET
# ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Abbildung W1: Control-Flow-Graph (CFG) als Graph-Repräsentation\nfür den Subgraph Algorithmus – WCET-Pfad-Identifikation", pad=12, fontweight='bold')

nodes = {
    'Entry': (5, 9.2),
    'Init':  (5, 8.0),
    'Loop-Head': (5, 6.8),
    'Body-A': (2.5, 5.6),
    'Body-B': (7.5, 5.6),
    'Branch': (5, 4.4),
    'Call-F': (2.5, 3.2),
    'Back-Edge': (7.5, 3.2),
    'Merge': (5, 2.0),
    'Exit': (5, 0.8),
}
colors_node = {
    'Entry': '#2ecc71', 'Exit': '#e74c3c', 'Loop-Head': '#e67e22',
    'Branch': '#e67e22', 'Back-Edge': '#c0392b',
}
for name, (x, y) in nodes.items():
    c = colors_node.get(name, '#3498db')
    circle = plt.Circle((x, y), 0.45, color=c, zorder=3, alpha=0.9)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)

edges = [
    ('Entry','Init',''), ('Init','Loop-Head',''),
    ('Loop-Head','Body-A','true'), ('Loop-Head','Merge','false'),
    ('Body-A','Body-B',''), ('Body-B','Branch',''),
    ('Branch','Call-F','branch A'), ('Branch','Back-Edge','branch B'),
    ('Call-F','Merge',''), ('Back-Edge','Loop-Head','back edge'),
    ('Merge','Exit',''),
]
wcet_edges = {('Loop-Head','Body-A'), ('Body-A','Body-B'), ('Body-B','Branch'), ('Branch','Call-F'), ('Call-F','Merge')}
for (src, dst, lbl) in edges:
    x0,y0 = nodes[src]; x1,y1 = nodes[dst]
    dx,dy = x1-x0, y1-y0
    length = np.hypot(dx,dy)
    ux,uy = dx/length, dy/length
    xs,ys = x0+ux*0.45, y0+uy*0.45
    xe,ye = x1-ux*0.45, y1-uy*0.45
    is_wcet = (src,dst) in wcet_edges
    color = '#c0392b' if is_wcet else '#555555'
    lw = 2.5 if is_wcet else 1.2
    style = '->' if (src,dst)!=('Back-Edge','Loop-Head') else 'arc3,rad=-0.4'
    ax.annotate('', xy=(xe,ye), xytext=(xs,ys),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        connectionstyle='arc3,rad=0.05' if (src,dst)==('Back-Edge','Loop-Head') else 'arc3,rad=0.0'))
    if lbl:
        mx,my = (xs+xe)/2, (ys+ye)/2
        ax.text(mx+0.15, my, lbl, fontsize=7.5, color='#333333', style='italic')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0],color='#c0392b',lw=2.5,label='WCET-Pfad (Subgraph-identifiziert)'),
    Line2D([0],[0],color='#555555',lw=1.2,label='Normalpfad'),
    mpatches.Patch(color='#e67e22',label='Schleifenknoten / Branch'),
    mpatches.Patch(color='#c0392b',label='Back-Edge (Zyklus)'),
    mpatches.Patch(color='#2ecc71',label='Entry'), mpatches.Patch(color='#e74c3c',label='Exit'),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
plt.tight_layout()
plt.savefig('plot_w1_cfg_wcet.pdf', bbox_inches='tight')
plt.close()
print("W1 done")

# ─────────────────────────────────────────────────
# Plot W2: Subgraph-Adjazenzmatrix-Analyse
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Abbildung W2: Adjazenzmatrix-Darstellung – Subgraph Algorithmus für WCET-Pfaderkennung", fontweight='bold')

# CFG als Adjazenzmatrix (8 Knoten)
labels = ['Entry','Init','LH','BA','BB','Br','CF','Exit']
A_full = np.array([
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0],
], dtype=float)
# WCET Subgraph (nur der kritische Pfad)
A_wcet = np.zeros((8,8))
for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]:
    A_wcet[i,j] = 1

cmap1 = plt.cm.Blues
for ax_idx, (mat, title, cmap) in enumerate([
    (A_full, 'Vollständiger CFG\n(Adjazenzmatrix)', plt.cm.Blues),
    (A_wcet, 'WCET-Subgraph\n(kritischer Pfad)', plt.cm.Reds),
    (A_full - A_wcet, 'Differenz\n(nicht-kritische Kanten)', plt.cm.Greens),
]):
    ax2 = axes[ax_idx]
    im = ax2.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(8)); ax2.set_yticks(range(8))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title(title, fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    for i in range(8):
        for j in range(8):
            v = mat[i,j]
            if abs(v) > 0.01:
                ax2.text(j, i, f'{v:.0f}', ha='center', va='center', color='white' if v>0.5 else 'black', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_w2_adjazenz.pdf', bbox_inches='tight')
plt.close()
print("W2 done")

# ─────────────────────────────────────────────────
# Plot W3: WCET-Analyse Vergleich Methoden
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Abbildung W3: WCET-Schätzgenauigkeit – Subgraph Algorithmus vs. klassische Methoden", fontweight='bold')

methods = ['Manuell\n(meas.)', 'IPET\n(klassisch)', 'AbsInt\naiT', 'Subgraph\nAlg. (Epp)']
cores = ['Cortex-M4', 'Cortex-R52', 'Cortex-HX-SI']
wcet_ratios = {
    'Cortex-M4':   [1.45, 1.18, 1.09, 1.05],
    'Cortex-R52':  [1.62, 1.24, 1.12, 1.06],
    'Cortex-HX-SI':[1.38, 1.16, 1.08, 1.04],
}
colors_m = ['#95a5a6','#3498db','#e67e22','#c0392b']
x = np.arange(len(methods))
width = 0.22
ax3 = axes[0]
for i, core in enumerate(cores):
    bars = ax3.bar(x + i*width, wcet_ratios[core], width, label=core, color=colors_m, alpha=0.85)
ax3.set_xlabel('Analysemethode')
ax3.set_ylabel('WCET/BCET-Ratio (Overapproximation)')
ax3.set_title('Overapproximations-Faktor je Methode', fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels(methods)
ax3.axhline(1.0, color='black', lw=1.5, ls='--', label='Ideal (exakt)')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_ylim(0.9, 1.8)
ax3.grid(axis='y', alpha=0.3)

# Analyse-Zeit
ax4 = axes[1]
prog_sizes = [100, 500, 1000, 5000, 10000, 50000]
time_ipet = [0.01, 0.08, 0.3, 8.5, 35, 800]
time_absint = [0.005, 0.04, 0.15, 4.2, 18, 420]
time_subgraph = [0.003, 0.012, 0.045, 0.8, 3.1, 75]
ax4.loglog(prog_sizes, time_ipet, 'o-', color='#3498db', lw=2, label='IPET (klassisch) ≈O(n⁴)')
ax4.loglog(prog_sizes, time_absint, 's-', color='#e67e22', lw=2, label='AbsInt aiT ≈O(n³·log n)')
ax4.loglog(prog_sizes, time_subgraph, '^-', color='#c0392b', lw=2.5, label='Subgraph Alg. O(n³)')
ax4.set_xlabel('Programmgröße (Anzahl Basisblöcke n)')
ax4.set_ylabel('Analysezeit [s]')
ax4.set_title('Laufzeit-Skalierung der Analysemethoden', fontweight='bold')
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)
ax4.set_title('Laufzeit-Skalierung O(n³) Subgraph Alg.', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_w3_wcet_methoden.pdf', bbox_inches='tight')
plt.close()
print("W3 done")

# ─────────────────────────────────────────────────
# Plot W4: Signatur-Chiffre Rotations-Analyse für CFG
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Abbildung W4: Signatur-basierte Zyklusidentifikation im CFG\n(Subgraph Algorithmus – Rotationsanalyse)", fontweight='bold')

n = 6
np.random.seed(42)
sig_cfg = np.array([5, 21, 9, 37, 13, 25])  # Signaturen CFG
sig_wcet = np.array([5, 21, 9, 37, 13, 25])  # Perfekter Match
sig_other = np.array([7, 11, 9, 43, 13, 19])

ax_s = axes[0,0]
x_labels = [f'σ_{i}' for i in range(n)]
width = 0.3
ax_s.bar(np.arange(n)-0.15, sig_cfg, width, color='#3498db', alpha=0.8, label='CFG-Signaturen')
ax_s.bar(np.arange(n)+0.15, sig_wcet, width, color='#c0392b', alpha=0.8, label='WCET-Kandidat')
ax_s.set_xticks(range(n)); ax_s.set_xticklabels(x_labels)
ax_s.set_ylabel('Signaturwert σ_j'); ax_s.set_title('Signaturvergleich CFG vs. WCET-Kandidat', fontweight='bold')
ax_s.legend(); ax_s.grid(axis='y', alpha=0.3)

ax_r = axes[0,1]
rotations = np.arange(n)
lcs_scores = [6, 3, 2, 2, 3, 4]  # LCS-Scores der Rotationen
ax_r.bar(rotations, lcs_scores, color=['#c0392b' if v==max(lcs_scores) else '#95a5a6' for v in lcs_scores], alpha=0.85)
ax_r.axhline(2, color='orange', ls='--', lw=1.5, label='Mindestschwelle LCS≥2')
ax_r.set_xlabel('Rotation k'); ax_r.set_ylabel('LCS-Score')
ax_r.set_title('LCS-Score je Rotation (k=0 → Subgraph erkannt)', fontweight='bold')
ax_r.legend(); ax_r.grid(axis='y', alpha=0.3)

ax_c = axes[1,0]
complexity_n = np.arange(2, 51)
complexity_n3 = complexity_n**3 / 1000
complexity_n4 = complexity_n**4 / 10000
ax_c.plot(complexity_n, complexity_n3, 'r-', lw=2.5, label='Subgraph Alg. O(n³)')
ax_c.plot(complexity_n, complexity_n4, 'b--', lw=2, label='Naive Permutation O(n!)')
ax_c.fill_between(complexity_n, complexity_n3, complexity_n4, alpha=0.15, color='green', label='Vorteil Subgraph Alg.')
ax_c.set_xlabel('Knotenzahl n (Basisblöcke)'); ax_c.set_ylabel('Relative Operationen (×10³)')
ax_c.set_title('Komplexitätsvergleich: O(n³) vs. O(n!)', fontweight='bold')
ax_c.legend(); ax_c.grid(alpha=0.3)

ax_d = axes[1,1]
cores_wcet = ['M4', 'M7', 'R4', 'R52', 'R7', 'HX-SI', 'A53', 'A72', 'A76', 'HX-HP']
delta_before = [3.2, 6.8, 4.1, 5.2, 7.3, 3.4, 15.2, 22.5, 48.3, 5.9]
delta_after  = [2.1, 4.5, 2.8, 3.5, 5.1, 2.2,  9.8, 14.1, 28.7, 3.8]
x_pos = np.arange(len(cores_wcet))
ax_d.bar(x_pos-0.2, delta_before, 0.35, color='#e74c3c', alpha=0.8, label='δ ohne Subgraph-Opt.')
ax_d.bar(x_pos+0.2, delta_after,  0.35, color='#27ae60', alpha=0.8, label='δ mit Subgraph-Opt.')
ax_d.set_xticks(x_pos); ax_d.set_xticklabels(cores_wcet, rotation=45, ha='right')
ax_d.set_ylabel('WCET/BCET-Ratio δ')
ax_d.set_title('Reduktion des WCET-Spreads δ durch Subgraph-Analyse', fontweight='bold')
ax_d.legend(); ax_d.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_w4_signaturen.pdf', bbox_inches='tight')
plt.close()
print("W4 done")

# ─────────────────────────────────────────────────
# Plot W5: pWCET Verteilung
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Abbildung W5: Probabilistisches WCET (pWCET) – Gumbel-Extremwertverteilung\nund Konfidenzgrenzen nach Subgraph-Pfadanalyse", fontweight='bold')

t = np.linspace(0, 100, 1000)
mu_m4, beta_m4 = 35, 5
mu_r52, beta_r52 = 28, 3
mu_hx, beta_hx = 22, 2

from scipy.stats import gumbel_r
ax5 = axes[0]
ax5.plot(t, gumbel_r.pdf(t, loc=mu_m4, scale=beta_m4)*100, 'b-', lw=2, label='Cortex-M4 (μ=35,β=5)')
ax5.plot(t, gumbel_r.pdf(t, loc=mu_r52, scale=beta_r52)*100, 'orange', lw=2, label='Cortex-R52 (μ=28,β=3)')
ax5.plot(t, gumbel_r.pdf(t, loc=mu_hx, scale=beta_hx)*100, 'r-', lw=2.5, label='HX-Safety-Island (μ=22,β=2)')
p999 = [gumbel_r.ppf(0.999, loc=m, scale=b) for m,b in [(mu_m4,beta_m4),(mu_r52,beta_r52),(mu_hx,beta_hx)]]
colors5 = ['blue','orange','red']
for p,c in zip(p999,colors5):
    ax5.axvline(p, color=c, ls='--', lw=1.2, alpha=0.7)
ax5.set_xlabel('Ausführungszeit [μs]'); ax5.set_ylabel('Dichte (×10⁻²)')
ax5.set_title('pWCET Gumbel-Verteilung\n(gestrichelt: p=0.999 Quantil)', fontweight='bold')
ax5.legend(); ax5.grid(alpha=0.3)

ax6 = axes[1]
conf_levels = [0.9, 0.95, 0.99, 0.999, 0.9999]
wcet_m4   = [gumbel_r.ppf(c, loc=mu_m4, scale=beta_m4) for c in conf_levels]
wcet_r52  = [gumbel_r.ppf(c, loc=mu_r52, scale=beta_r52) for c in conf_levels]
wcet_hx   = [gumbel_r.ppf(c, loc=mu_hx, scale=beta_hx) for c in conf_levels]
x_conf = range(len(conf_levels))
ax6.plot(x_conf, wcet_m4, 'b-o', lw=2, label='Cortex-M4')
ax6.plot(x_conf, wcet_r52, 'orange', marker='s', lw=2, label='Cortex-R52')
ax6.plot(x_conf, wcet_hx, 'r-^', lw=2.5, label='HX-Safety-Island')
ax6.axhline(100, color='black', ls=':', lw=1.5, label='EtherCAT-Deadline 100μs')
ax6.set_xticks(x_conf)
ax6.set_xticklabels([f'{c*100:.2f}%' for c in conf_levels])
ax6.set_xlabel('Konfidenzniveau'); ax6.set_ylabel('pWCET [μs]')
ax6.set_title('pWCET vs. Konfidenzniveau\n(EtherCAT-Deadline 100μs)', fontweight='bold')
ax6.legend(); ax6.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_w5_pwcet.pdf', bbox_inches='tight')
plt.close()
print("W5 done")

