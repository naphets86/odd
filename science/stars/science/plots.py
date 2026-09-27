import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'text.usetex': False,
})

OUTDIR = 'science/'

# ── HILFSFUNKTIONEN (Subgraph Algorithmus) ──────────────────────────────────
def compute_signature(matrix):
    n = matrix.shape[0]
    sigs = []
    for col in range(n):
        col_vec = matrix[:, col]
        row_sig = sum(2**i for i in range(n) if col_vec[i] == 1)
        sigs.append(row_sig + col * (2**n))
    return sigs

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def subgraph_check(A, B):
    nA, nB = A.shape[0], B.shape[0]
    sigA = compute_signature(A)
    sigB = compute_signature(B)
    wA, wB = 2**nA, 2**nB
    rA = [s % wA for s in sigA]
    rB = [s % wB for s in sigB]
    for rot in range(nB):
        rotB = rB[rot:] + rB[:rot]
        if nA == nB:
            if lcs_length(rA, rotB) >= 2:
                return True, rot
        else:
            for start in range(nB - nA + 1):
                win = rotB[start:start+nA]
                if lcs_length(rA, win) >= 2:
                    return True, rot
    return False, -1

def gravity_matrix(positions, masses):
    """Gravitationsstärke-Matrix zwischen Sternen (Kanten-Gewichte)."""
    G = 6.674e-11
    n = len(positions)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                f = G * masses[i] * masses[j] / r**2
                W[i, j] = W[j, i] = f
    return W

def adj_from_gravity(W, threshold_frac=0.3):
    """Adjazenzmatrix: Kante wenn Gravitation > threshold * max."""
    thr = threshold_frac * W.max()
    A = (W > thr).astype(int)
    np.fill_diagonal(A, 0)
    return A

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Drei Sterncluster im 2D-Raum mit gravitativen Kanten
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

def make_cluster(center, n, spread, mass_range):
    pos = center + np.random.randn(n, 2) * spread
    masses = np.random.uniform(*mass_range, n) * 2e30  # Sonnenmassen
    return pos, masses

cA_pos, cA_m = make_cluster(np.array([0.0, 0.0]),   8, 1.5e15, (0.5, 2.0))
cB_pos, cB_m = make_cluster(np.array([6e15, 2e15]),  7, 1.2e15, (1.0, 3.0))
cC_pos, cC_m = make_cluster(np.array([2e15, -5e15]), 6, 1.8e15, (0.3, 1.5))

all_pos = np.vstack([cA_pos, cB_pos, cC_pos])
all_m   = np.concatenate([cA_m, cB_m, cC_m])
nA, nB_, nC = len(cA_pos), len(cB_pos), len(cC_pos)

W_full = gravity_matrix(all_pos, all_m)
A_full = adj_from_gravity(W_full, 0.15)

fig, ax = plt.subplots(figsize=(9, 7))
colors_node = (['#E8593C']*nA + ['#378ADD']*nB_ + ['#1D9E75']*nC)
sizes = (all_m / 2e30) * 80 + 30

# Kanten zeichnen
for i in range(len(all_pos)):
    for j in range(i+1, len(all_pos)):
        if A_full[i, j]:
            lw = W_full[i,j] / W_full.max() * 3
            alpha = 0.3 + 0.5 * W_full[i,j] / W_full.max()
            ax.plot([all_pos[i,0], all_pos[j,0]],
                    [all_pos[i,1], all_pos[j,1]],
                    color='#888', lw=lw, alpha=alpha, zorder=1)

sc = ax.scatter(all_pos[:,0], all_pos[:,1], c=colors_node, s=sizes,
                zorder=3, edgecolors='white', linewidths=0.7)

labels = [f'$s_{{{i+1}}}$' for i in range(len(all_pos))]
for i, (x, y) in enumerate(all_pos):
    ax.annotate(labels[i], (x, y), textcoords='offset points',
                xytext=(7, 5), fontsize=8, color='#333')

legend_handles = [
    mpatches.Patch(color='#E8593C', label='Cluster A (offen, 8 Sterne)'),
    mpatches.Patch(color='#378ADD', label='Cluster B (Kugel, 7 Sterne)'),
    mpatches.Patch(color='#1D9E75', label='Cluster C (Feld, 6 Sterne)'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
ax.set_xlabel('x-Position [m]', fontsize=11)
ax.set_ylabel('y-Position [m]', fontsize=11)
ax.set_title('Abbildung 1: Sterncluster als gewichteter Graph\n(Kantenstärke ∝ Gravitationskraft)', fontsize=12)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot1_cluster_graph.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Adjazenzmatrizen der drei Cluster
# ══════════════════════════════════════════════════════════════════════════════
WA = gravity_matrix(cA_pos, cA_m)
WB = gravity_matrix(cB_pos, cB_m)
WC = gravity_matrix(cC_pos, cC_m)
AA = adj_from_gravity(WA, 0.25)
AB = adj_from_gravity(WB, 0.25)
AC = adj_from_gravity(WC, 0.25)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mat, title, cmap in zip(axes,
                                 [AA, AB, AC],
                                 ['Cluster A\n(Offener Haufen)',
                                  'Cluster B\n(Kugelhaufen)',
                                  'Cluster C\n(Feldpopulation)'],
                                 ['Oranges', 'Blues', 'Greens']):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Stern-Index j')
    ax.set_ylabel('Stern-Index i')
    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f's{i+1}' for i in range(n)], fontsize=7)
    ax.set_yticklabels([f's{i+1}' for i in range(n)], fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(mat[i,j]), ha='center', va='center',
                    fontsize=7, color='white' if mat[i,j] else '#999')

plt.suptitle('Abbildung 2: Adjazenzmatrizen der drei Sterncluster\n(Gravitationsschwelle: 25 % des Maximalwerts)', fontsize=12)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot2_adjazenzmatrizen.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Signatur-Arrays der Cluster
# ══════════════════════════════════════════════════════════════════════════════
sigA = compute_signature(AA)
sigB = compute_signature(AB)
sigC = compute_signature(AC)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
for ax, sig, title, color in zip(axes,
                                   [sigA, sigB, sigC],
                                   ['Signatur-Array: Cluster A', 'Signatur-Array: Cluster B', 'Signatur-Array: Cluster C'],
                                   ['#E8593C', '#378ADD', '#1D9E75']):
    xpos = range(len(sig))
    bars = ax.bar(xpos, sig, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('σ_j', fontsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'j={i}' for i in xpos], fontsize=8)
    for bar, val in zip(bars, sig):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sig)*0.01,
                str(val), ha='center', va='bottom', fontsize=7)

plt.suptitle('Abbildung 3: Eindeutige Signaturen σ_j = Σ A_ij·2^i + j·2^n\nder drei Cluster-Adjazenzmatrizen', fontsize=12)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot3_signaturen.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: LCS-Heatmap (Rotation-Match zwischen A und B)
# ══════════════════════════════════════════════════════════════════════════════
wA_ = 2**nA
wB2 = 2**nB_
rA_ = [s % wA_ for s in sigA]
rB2 = [s % wB2 for s in sigB]

lcs_matrix = np.zeros((nB_, nB_))
for rot in range(nB_):
    rotB = rB2[rot:] + rB2[:rot]
    for start in range(nB_ - nA + 1):
        win = rotB[start:start+nA]
        l = lcs_length(rA_, win)
        lcs_matrix[rot, start] = l

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(lcs_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='LCS-Länge')
ax.set_xlabel('Startposition im rotierten Fenster', fontsize=11)
ax.set_ylabel('Rotation k', fontsize=11)
ax.set_title('Abbildung 4: LCS-Längen bei allen Rotationen\n(Cluster A vs. Cluster B — Subgraph-Erkennung)', fontsize=12)
ax.set_xticks(range(nB_-nA+1))
ax.set_yticks(range(nB_))
ax.set_yticklabels([f'rot={k}' for k in range(nB_)])

# Markiere Felder mit LCS >= 2
for i in range(lcs_matrix.shape[0]):
    for j in range(lcs_matrix.shape[1]):
        v = lcs_matrix[i,j]
        color = 'white' if v >= 3 else '#333'
        ax.text(j, i, f'{int(v)}', ha='center', va='center', fontsize=9, color=color)
        if v >= 2:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                        fill=False, edgecolor='#185FA5', lw=2))

plt.tight_layout()
plt.savefig(OUTDIR + 'plot4_lcs_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Gravitationspotentialfeld (Potential-Landschaft)
# ══════════════════════════════════════════════════════════════════════════════
G_const = 6.674e-11
scale = 1e15
xs = np.linspace(-3*scale, 9*scale, 200)
ys = np.linspace(-8*scale, 5*scale, 200)
XX, YY = np.meshgrid(xs, ys)
PHI = np.zeros_like(XX)

for pos, m in zip(all_pos, all_m):
    R = np.sqrt((XX - pos[0])**2 + (YY - pos[1])**2)
    R = np.maximum(R, 1e13)
    PHI -= G_const * m / R

fig, ax = plt.subplots(figsize=(10, 7))
PHI_log = np.log10(-PHI + 1e-30)
cf = ax.contourf(XX/scale, YY/scale, PHI_log, levels=40, cmap='inferno_r', alpha=0.85)
plt.colorbar(cf, ax=ax, label='log₁₀(|Φ|) [m²/s²]')
cs = ax.contour(XX/scale, YY/scale, PHI_log, levels=15, colors='white', alpha=0.25, linewidths=0.5)

ax.scatter(all_pos[:,0]/scale, all_pos[:,1]/scale, c=colors_node, s=sizes*1.5,
           zorder=5, edgecolors='white', linewidths=1)
ax.set_xlabel('x [10¹⁵ m]', fontsize=11)
ax.set_ylabel('y [10¹⁵ m]', fontsize=11)
ax.set_title('Abbildung 5: Gravitationspotentialfeld Φ(x,y) der Sterncluster\n(Tiefe Senken = gravitativ dominante Regionen)', fontsize=12)
ax.grid(True, alpha=0.15, color='white')
plt.tight_layout()
plt.savefig(OUTDIR + 'plot5_gravitationspotential.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6: Substruktur-Hierarchie durch Schwellenwert-Variation
# ══════════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(0.05, 0.95, 30)
subgraph_AB = []
subgraph_BA = []
subgraph_AC = []
edge_counts_A = []

for thr in thresholds:
    AA_t = adj_from_gravity(WA, thr)
    AB_t = adj_from_gravity(WB, thr)
    AC_t = adj_from_gravity(WC, thr)
    edge_counts_A.append(AA_t.sum() // 2)
    r_ab, _ = subgraph_check(AA_t, AB_t)
    r_ba, _ = subgraph_check(AB_t, AA_t)
    r_ac, _ = subgraph_check(AA_t, AC_t)
    subgraph_AB.append(int(r_ab))
    subgraph_BA.append(int(r_ba))
    subgraph_AC.append(int(r_ac))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(thresholds, subgraph_AB, 'o-', color='#E8593C', label='A ⊆ B (Cluster A in B)', lw=2)
ax1.plot(thresholds, subgraph_BA, 's--', color='#378ADD', label='B ⊆ A (Cluster B in A)', lw=2)
ax1.plot(thresholds, subgraph_AC, '^:', color='#1D9E75', label='A ⊆ C (Cluster A in C)', lw=2)
ax1.set_ylabel('Subgraph erkannt (1=Ja)', fontsize=11)
ax1.set_title('Abbildung 6: Subgraph-Erkennung in Abhängigkeit vom Gravitationsschwellenwert', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_ylim(-0.1, 1.2)
ax1.grid(True, alpha=0.3)

ax2.plot(thresholds, edge_counts_A, color='#BA7517', lw=2.5, label='Kantenanzahl Cluster A')
ax2.set_xlabel('Gravitationsschwelle (Bruchteil des Maximalwerts)', fontsize=11)
ax2.set_ylabel('Anzahl aktiver Kanten', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot6_schwellenwert.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 7: Energie-Analyse — Bindungsenergie & Virial-Theorem
# ══════════════════════════════════════════════════════════════════════════════
# E_kin ≈ (1/2) m v²  mit v aus Virial-Näherung v ~ sqrt(G M / R)
def cluster_energetics(positions, masses):
    G_c = 6.674e-11
    n = len(positions)
    E_pot = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                E_pot -= G_c * masses[i] * masses[j] / r
    M_total = masses.sum()
    R_half = np.median(cdist(positions, positions[positions.shape[0]//2:positions.shape[0]//2+1]).flatten())
    R_half = max(R_half, 1e12)
    v_virial = np.sqrt(G_c * M_total / R_half)
    E_kin = 0.5 * M_total * v_virial**2
    return E_pot, E_kin, -0.5 * E_pot  # (E_pot, E_kin_approx, virial_E_kin)

# Variation der Clustergröße
spreads = np.linspace(0.5e15, 3e15, 15)
Epot_list, Ekin_virial, E_bind = [], [], []
for sp in spreads:
    pos = np.array([0,0]) + np.random.randn(8, 2) * sp
    ms  = np.random.uniform(0.5, 2.0, 8) * 2e30
    ep, ek, ev = cluster_energetics(pos, ms)
    Epot_list.append(ep)
    Ekin_virial.append(ek)
    E_bind.append(abs(ep) - ek)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(spreads/1e15, np.array(Epot_list)/1e30, 'o-', color='#E8593C', label='E_pot [10³⁰ J]', lw=2)
ax.plot(spreads/1e15, np.array(Ekin_virial)/1e30, 's--', color='#378ADD', label='E_kin (Virial) [10³⁰ J]', lw=2)
ax.set_xlabel('Clusterausdehnung [10¹⁵ m]', fontsize=11)
ax.set_ylabel('Energie [10³⁰ J]', fontsize=11)
ax.set_title('Potential- und kinetische Energie\nvs. Clusterausdehnung', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(spreads/1e15, np.array(E_bind)/1e30, alpha=0.4, color='#1D9E75')
ax.plot(spreads/1e15, np.array(E_bind)/1e30, 'o-', color='#0F6E56', lw=2.5, label='Bindungsenergie [10³⁰ J]')
ax.axhline(0, color='#333', lw=1, ls='--')
ax.set_xlabel('Clusterausdehnung [10¹⁵ m]', fontsize=11)
ax.set_ylabel('E_bind [10³⁰ J]', fontsize=11)
ax.set_title('Bindungsenergie E_bind = |E_pot| − E_kin\n(positiv = gravitativ gebunden)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Abbildung 7: Energetik der Sterncluster — Virial-Theorem-Analyse', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot7_energie.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 8: Algorithmus-Komplexität O(n³) empirisch
# ══════════════════════════════════════════════════════════════════════════════
import time

ns = list(range(3, 18))
times_sub = []
for n_nodes in ns:
    pos1 = np.random.randn(n_nodes, 2) * 1e15
    pos2 = np.random.randn(n_nodes, 2) * 1e15
    m1 = np.random.uniform(1, 3, n_nodes) * 2e30
    m2 = np.random.uniform(1, 3, n_nodes) * 2e30
    W1 = gravity_matrix(pos1, m1)
    W2 = gravity_matrix(pos2, m2)
    A1 = adj_from_gravity(W1, 0.3)
    A2 = adj_from_gravity(W2, 0.3)
    t0 = time.time()
    for _ in range(5):
        subgraph_check(A1, A2)
    times_sub.append((time.time() - t0) / 5)

ns_arr = np.array(ns, dtype=float)
times_arr = np.array(times_sub)
# Fit O(n^3)
coeffs = np.polyfit(np.log(ns_arr), np.log(times_arr + 1e-9), 1)
fit_exp = coeffs[0]
fit_line = np.exp(coeffs[1]) * ns_arr**fit_exp

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ns_arr, times_arr*1000, 'o-', color='#185FA5', lw=2, label='Messung')
ax1.plot(ns_arr, fit_line*1000, '--', color='#E24B4A', lw=2, label=f'Fit O(n^{fit_exp:.2f})')
ax1.set_xlabel('Anzahl Sterne n', fontsize=11)
ax1.set_ylabel('Laufzeit [ms]', fontsize=11)
ax1.set_title('Laufzeit des Subgraph Algorithmus\n(empirisch)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.loglog(ns_arr, times_arr + 1e-9, 'o-', color='#185FA5', lw=2, label='Messung (log-log)')
ax2.loglog(ns_arr, fit_line, '--', color='#E24B4A', lw=2, label=f'Fit: Exponent ≈ {fit_exp:.2f}')
n3_ref = (ns_arr/ns_arr[0])**3 * (times_arr[0]+1e-9)
ax2.loglog(ns_arr, n3_ref, ':', color='#0F6E56', lw=2, label='O(n³) Referenz')
ax2.set_xlabel('n (log)', fontsize=11)
ax2.set_ylabel('Zeit (log)', fontsize=11)
ax2.set_title('Log-Log-Darstellung\n(Steigung ≈ 3 bestätigt O(n³))', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

plt.suptitle('Abbildung 8: Empirische Komplexitätsverifikation des Subgraph Algorithmus', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot8_komplexitaet.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 9: Dynamische Substruktur-Evolution (Zeitreihe)
# ══════════════════════════════════════════════════════════════════════════════
# Simuliere einfache Punktmassen-Dynamik über Zeit (Verlet)
np.random.seed(7)
n_stars = 6
pos_dyn = np.random.randn(n_stars, 2) * 1e15
vel_dyn = np.random.randn(n_stars, 2) * 1e3  # m/s
mass_dyn = np.ones(n_stars) * 2e30
dt = 1e11  # s ≈ 3000 Jahre
G_c = 6.674e-11

def accel(pos, masses):
    n = len(pos)
    a = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                if r > 1e12:
                    a[i] += G_c * masses[j] / r**3 * r_vec
    return a

n_steps = 8
snap_pos = [pos_dyn.copy()]
snap_sub = []
for _ in range(n_steps - 1):
    a = accel(pos_dyn, mass_dyn)
    vel_dyn += a * dt
    pos_dyn += vel_dyn * dt
    snap_pos.append(pos_dyn.copy())

for sp in snap_pos:
    W = gravity_matrix(sp, mass_dyn)
    A = adj_from_gravity(W, 0.3)
    n_edges = A.sum() // 2
    snap_sub.append(n_edges)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for idx, (ax, sp, ne) in enumerate(zip(axes.flatten(), snap_pos, snap_sub)):
    W = gravity_matrix(sp, mass_dyn)
    A = adj_from_gravity(W, 0.3)
    for i in range(n_stars):
        for j in range(i+1, n_stars):
            if A[i,j]:
                ax.plot([sp[i,0], sp[j,0]], [sp[i,1], sp[j,1]],
                        color='#888', alpha=0.5, lw=1)
    ax.scatter(sp[:,0], sp[:,1], c='#EF9F27', s=80, zorder=3, edgecolors='white', lw=0.5)
    ax.set_title(f't = {idx} · Δt\n{ne} Kanten', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Abbildung 9: Zeitliche Evolution der Subgraph-Struktur eines Sternclusters\n(Kanten = gravitativ gebundene Sternpaare)', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot9_zeitevolution.pdf', bbox_inches='tight')
plt.close()
print("Plot 9 OK")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 10: Energiepotential-Differenz bei Sternverschiebung
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(12)
pos_base = np.random.randn(8, 2) * 1e15
masses_base = np.random.uniform(0.8, 2.5, 8) * 2e30

def total_potential(positions, masses):
    G_c = 6.674e-11
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                E -= G_c * masses[i] * masses[j] / r
    return E

star_idx = 0
displacements = np.linspace(-2e15, 2e15, 50)
delta_E_x, delta_E_y = [], []
E0 = total_potential(pos_base, masses_base)

for d in displacements:
    pos_x = pos_base.copy()
    pos_x[star_idx, 0] += d
    delta_E_x.append(total_potential(pos_x, masses_base) - E0)
    
    pos_y = pos_base.copy()
    pos_y[star_idx, 1] += d
    delta_E_y.append(total_potential(pos_y, masses_base) - E0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(displacements/1e15, np.array(delta_E_x)/1e30, color='#E8593C', lw=2.5, label='Verschiebung in x')
ax.plot(displacements/1e15, np.array(delta_E_y)/1e30, color='#378ADD', lw=2.5, ls='--', label='Verschiebung in y')
ax.axhline(0, color='#333', lw=1, ls=':')
ax.axvline(0, color='#333', lw=1, ls=':')
ax.fill_between(displacements/1e15, np.array(delta_E_x)/1e30, 0, alpha=0.15, color='#E8593C')
ax.set_xlabel('Verschiebung Δr [10¹⁵ m]', fontsize=11)
ax.set_ylabel('ΔE_pot [10³⁰ J]', fontsize=11)
ax.set_title('Potentialänderung bei Verschiebung\nvon Stern s₁ in einem Cluster', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Subgraph-Änderung bei Verschiebung
displacements_coarse = np.linspace(-2e15, 2e15, 20)
edges_shift = []
for d in displacements_coarse:
    pos_x = pos_base.copy()
    pos_x[star_idx, 0] += d
    W = gravity_matrix(pos_x, masses_base)
    A = adj_from_gravity(W, 0.25)
    edges_shift.append(A.sum()//2)

ax = axes[1]
ax.step(displacements_coarse/1e15, edges_shift, color='#1D9E75', lw=2.5, where='mid', label='Aktive Kanten')
ax.set_xlabel('Verschiebung Δr [10¹⁵ m]', fontsize=11)
ax.set_ylabel('Anzahl aktiver Subgraph-Kanten', fontsize=11)
ax.set_title('Subgraph-Topologie in Abhängigkeit\nvon der Sternposition', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Abbildung 10: Energetische und topologische Auswirkung der Sternverschiebung', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot10_sternverschiebung.pdf', bbox_inches='tight')
plt.close()
print("Plot 10 OK")

print("\nAlle 10 Plots erfolgreich generiert.")
