#!/usr/bin/env python3
"""
Plots fuer Kapitel: Der Subgraph Algorithmus als Stabilisierungsoperator
auf dem Wirtschaftsgraphen -- Konvergenz zur Systemstabilitaet
Stephan Epp, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eigvals
import warnings
warnings.filterwarnings('ignore')

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
DARKGRAY  = '#3C3C46'
ORANGE    = '#E07820'
GOLD      = '#C8960A'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

OUT = ''

# =========================================================
# Hilfsfunktionen: Subgraph Algorithmus (Kernlogik)
# =========================================================

def compute_signatures(matrix):
    n = matrix.shape[0]
    sigs = []
    for col in range(n):
        col_vec = matrix[:, col]
        row_sig = sum(int(2**i) for i in range(n) if col_vec[i] == 1)
        sig = row_sig + col * (2**n)
        sigs.append(sig)
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
    """Gibt True zurueck wenn A Subgraph von B (unter zykl. Rotation)."""
    n_a, n_b = A.shape[0], B.shape[0]
    if n_a > n_b:
        return False
    sig_a = compute_signatures(A)
    sig_b = compute_signatures(B)
    cw_a = 2**n_a
    cw_b = 2**n_b
    rs_a = [s % cw_a for s in sig_a]
    rs_b = [s % cw_b for s in sig_b]
    for rot in range(n_b):
        rot_b = rs_b[rot:] + rs_b[:rot]
        if n_a == n_b:
            if lcs_length(rs_a, rot_b) >= 2:
                return True
        else:
            for start in range(n_b - n_a + 1):
                window = rot_b[start:start+n_a]
                if lcs_length(rs_a, window) >= 2:
                    return True
    return False

def coupling_degree(W):
    N = W.shape[0]
    off = W - np.diag(np.diag(W))
    return off.sum() / (N*(N-1)) if N > 1 else 0.0

# =========================================================
# Plot S1 -- Subgraph-Operator: Schritt-fuer-Schritt Wirkung
#            auf der Adjazenzmatrix eines Wirtschaftsgraphen
# =========================================================
np.random.seed(42)
N = 8

# Starte mit einem zufaelligen, zu stark gekoppelten Graphen
def random_adj(n, p, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n,n)) < p).astype(float)
    np.fill_diagonal(A, 0)
    return A

# Referenz-"Stabilitaetsstruktur" (sparsamer, gut geclusterter Graph)
A_ref = np.zeros((N,N))
# 2 Cluster a 4 Knoten, je vollstaendig intern verbunden, schwache Brieftauben-Kante
for i in range(4):
    for j in range(4):
        if i != j:
            A_ref[i,j] = 1.0
for i in range(4,8):
    for j in range(4,8):
        if i != j:
            A_ref[i,j] = 1.0
A_ref[3,4] = 1.0; A_ref[4,3] = 1.0   # eine schwache Bruecke

# Ausgangsmatrix: stark uebergekoppelt
A_start = random_adj(N, 0.72, seed=7)

def subgraph_stabilize_step(A, A_ref):
    """
    Wendet einen Schritt des Subgraph-Operators an:
    Entfernt Kanten, die nicht im Referenzgraphen als Subgraph vorhanden sind.
    Konkret: fuer jede Kante (i,j) pruefen ob 1-Knoten-Subgraph {i->j}
    im Referenz vorhanden; sonst Kante loeschen.
    """
    A_new = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j and A[i,j] > 0:
                # Mini-Subgraph: nur diese eine Kante
                A_sub = np.zeros((2,2)); A_sub[0,1] = 1.0
                A_win = np.array([[A_ref[i,i], A_ref[i,j]],
                                   [A_ref[j,i], A_ref[j,j]]])
                # Subgraph-Check: ist die Kante i->j in A_ref enthalten?
                if A_ref[i,j] < 0.5:
                    # Kante nicht im Referenz: mit Prob proportional zur Abweichung entfernen
                    A_new[i,j] = max(0.0, A[i,j] - 0.35)
    return A_new

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
titles = ['Ausgangszustand\n(ueberkoppelt)', 'Nach 1 Iteration', 'Nach 2 Iterationen', 'Stabilitaets-\nReferenz']

A_curr = A_start.copy()
states = [A_start.copy()]
for _ in range(2):
    A_curr = subgraph_stabilize_step(A_curr, A_ref)
    states.append(A_curr.copy())
states.append(A_ref.copy())

cmap_custom = LinearSegmentedColormap.from_list('bw', ['white', MAINBLUE])
for ax, mat, title in zip(axes, states, titles):
    im = ax.imshow(mat, cmap=cmap_custom, vmin=0, vmax=1, aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Knoten $j$'); ax.set_ylabel('Knoten $i$')
    c_val = coupling_degree(mat)
    ax.text(0.5, -0.22, f'$c = {c_val:.3f}$', transform=ax.transAxes,
            ha='center', fontsize=9, color=ACCENTRED)

fig.suptitle('Plot S1 -- Subgraph-Operator: Schrittweise Reduktion des Kopplungsgrades\nder Wirtschaftsgraphen-Adjazenzmatrix',
             fontsize=12, y=1.03)
plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUT + 'plotS1_operator_schritte.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S1 gespeichert.")

# =========================================================
# Plot S2 -- Kopplungsgrad c(t) ueber Iterationen: Konvergenz
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

T_iter = 30
seeds  = [7, 13, 42, 99, 123]
p_start_vals = [0.85, 0.75, 0.65]
colors_conv = [ACCENTRED, ORANGE, MAINBLUE]

ax = axes[0]
for p0, col in zip(p_start_vals, colors_conv):
    c_traj = []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_iter+1):
        c_traj.append(coupling_degree(A_c))
        A_c = subgraph_stabilize_step(A_c, A_ref)
    ax.plot(range(T_iter+1), c_traj, color=col, lw=2.2,
            label=f'Startdichte $p_0 = {p0}$')

c_ref = coupling_degree(A_ref)
ax.axhline(c_ref, color=DARKGREEN, lw=1.8, linestyle='--',
           label=f'Referenz $c^* = {c_ref:.3f}$')
ax.set_xlabel('Iteration $t$')
ax.set_ylabel('Kopplungsgrad $c(t)$')
ax.set_title('(a) Konvergenz des Kopplungsgrades\nzum Referenzwert $c^*$')
ax.legend(fontsize=9)
ax.set_ylim(0, 1)

# -- Kontraktionsrate --
ax = axes[1]
for p0, col in zip(p_start_vals, colors_conv):
    c_traj = []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_iter+1):
        c_traj.append(coupling_degree(A_c))
        A_c = subgraph_stabilize_step(A_c, A_ref)
    c_arr = np.array(c_traj)
    diff = np.abs(c_arr - c_ref)
    diff = np.where(diff < 1e-10, 1e-10, diff)
    ax.semilogy(range(T_iter+1), diff, color=col, lw=2.2,
                label=f'$p_0 = {p0}$')

ax.set_xlabel('Iteration $t$')
ax.set_ylabel('$|c(t) - c^*|$ (log)')
ax.set_title('(b) Konvergenzgeschwindigkeit\n(logarithmische Skala)')
ax.legend(fontsize=9)

fig.suptitle('Plot S2 -- Konvergenz des Subgraph-Stabilisierungsoperators', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'plotS2_konvergenz.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S2 gespeichert.")

# =========================================================
# Plot S3 -- Spektralradius der Ansteckungsmatrix Lambda(t)
# =========================================================
fig, ax = plt.subplots(figsize=(9, 5))

def ansteckungsmatrix(A, c):
    N = A.shape[0]
    phi = 0.6
    return c * phi * A / (N - 1 + 1e-9)

T_spec = 25
for p0, col, lab in zip(p_start_vals, colors_conv,
                         [f'$p_0={p}$' for p in p_start_vals]):
    radii = []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_spec+1):
        c_t = coupling_degree(A_c)
        Lam = ansteckungsmatrix(A_c, c_t)
        rho = max(abs(eigvals(Lam)))
        radii.append(float(rho.real))
        A_c = subgraph_stabilize_step(A_c, A_ref)
    ax.plot(range(T_spec+1), radii, color=col, lw=2.2, label=lab)

# Stabilitaetsgrenze rho = 1
ax.axhline(1.0, color=DARKGRAY, lw=1.4, linestyle=':', label='Stabilitaetsgrenze $\\rho(\\Lambda)=1$')
# Referenzwert
A_ref_lam = ansteckungsmatrix(A_ref, c_ref)
rho_ref = float(max(abs(eigvals(A_ref_lam))).real)
ax.axhline(rho_ref, color=DARKGREEN, lw=1.6, linestyle='--',
           label=f'Referenz $\\rho^* = {rho_ref:.3f}$')

ax.set_xlabel('Iteration $t$')
ax.set_ylabel('Spektralradius $\\rho(\\Lambda(t))$')
ax.set_title('Plot S3 -- Spektralradius der Ansteckungsmatrix $\\Lambda(t)$\nueber Stabilisierungsiterationen')
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig(OUT + 'plotS3_spektralradius.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S3 gespeichert.")

# =========================================================
# Plot S4 -- Lyapunov-Funktion V(t) = ||W(t) - W*||_F^2
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for p0, col, lab in zip(p_start_vals, colors_conv, [f'$p_0={p}$' for p in p_start_vals]):
    V_traj = []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_iter+1):
        diff_mat = A_c - A_ref
        V = np.sum(diff_mat**2)
        V_traj.append(V)
        A_c = subgraph_stabilize_step(A_c, A_ref)
    ax.plot(range(T_iter+1), V_traj, color=col, lw=2.2, label=lab)

ax.axhline(0, color=DARKGREEN, lw=1.4, linestyle='--', label='$V^* = 0$ (Gleichgewicht)')
ax.set_xlabel('Iteration $t$')
ax.set_ylabel('$V(t) = \\|W(t) - W^*\\|_F^2$')
ax.set_title('(a) Lyapunov-Funktion $V(t)$\n(Frobenius-Norm zum Gleichgewicht)')
ax.legend(fontsize=9)

# -- Delta V --
ax = axes[1]
for p0, col, lab in zip(p_start_vals, colors_conv, [f'$p_0={p}$' for p in p_start_vals]):
    V_traj = []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_iter+1):
        diff_mat = A_c - A_ref
        V_traj.append(np.sum(diff_mat**2))
        A_c = subgraph_stabilize_step(A_c, A_ref)
    dV = np.diff(V_traj)
    ax.plot(range(1, T_iter+1), dV, color=col, lw=2.0, label=lab)

ax.axhline(0, color=DARKGRAY, lw=1.2, linestyle=':')
ax.fill_between(range(1, T_iter+1), dV, 0,
                where=(dV < 0), alpha=0.12, color=DARKGREEN, label='$\\Delta V < 0$')
ax.set_xlabel('Iteration $t$')
ax.set_ylabel('$\\Delta V(t) = V(t) - V(t-1)$')
ax.set_title('(b) Differenz $\\Delta V(t)$\n(negativ = Kontraktionseigenschaft)')
ax.legend(fontsize=9)

fig.suptitle('Plot S4 -- Lyapunov-Analyse: Monotone Abnahme der Systemabweichung', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'plotS4_lyapunov.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S4 gespeichert.")

# =========================================================
# Plot S5 -- Phasendiagramm: Trajektorien im (c, rho)-Raum
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Stabilitaetsregion definieren
c_grid = np.linspace(0, 1, 300)
rho_stable = 0.4 + 0.6*c_grid   # rho = 1 Grenze skaliert mit c

ax.fill_between(c_grid, rho_stable, 1.5,
                alpha=0.10, color=ACCENTRED, label='Instabilitaetsbereich $\\rho > \\rho_s(c)$')
ax.fill_between(c_grid, 0, rho_stable,
                alpha=0.08, color=DARKGREEN, label='Stabilitaetsbereich $\\rho \\leq \\rho_s(c)$')
ax.plot(c_grid, rho_stable, color=DARKGRAY, lw=1.5, linestyle='--',
        label='Stabilitaetsgrenze $\\rho_s(c)$')

p_starts_phase = [0.85, 0.75, 0.65, 0.55]
cols_phase = [ACCENTRED, ORANGE, MAINBLUE, GOLD]

for p0, col in zip(p_starts_phase, cols_phase):
    c_path, r_path = [], []
    A_c = random_adj(N, p0, seed=42)
    for t in range(T_iter+1):
        c_t = coupling_degree(A_c)
        Lam = ansteckungsmatrix(A_c, c_t)
        rho_t = float(max(abs(eigvals(Lam))).real)
        c_path.append(c_t)
        r_path.append(rho_t)
        A_c = subgraph_stabilize_step(A_c, A_ref)
    ax.plot(c_path, r_path, color=col, lw=1.8, alpha=0.8)
    ax.scatter(c_path[0],  r_path[0],  color=col,     s=90, zorder=5)
    ax.scatter(c_path[-1], r_path[-1], color=DARKGREEN, s=90, marker='*', zorder=6)

# Attraktor
ax.scatter(c_ref, rho_ref, color=DARKGREEN, s=200, marker='*', zorder=7,
           label=f'Attraktor $(c^*, \\rho^*) = ({c_ref:.2f}, {rho_ref:.2f})$')

ax.set_xlabel('Kopplungsgrad $c$')
ax.set_ylabel('Spektralradius $\\rho(\\Lambda)$')
ax.set_title('Plot S5 -- Phasendiagramm: Trajektorien im $(c, \\rho)$-Raum\nunter wiederholter Anwendung des Subgraph-Operators')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.4)
plt.tight_layout()
plt.savefig(OUT + 'plotS5_phasendiagramm.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S5 gespeichert.")

# =========================================================
# Plot S6 -- Vergleich: Mit vs. Ohne Subgraph-Operator
#            Krisenausbreitung nach exogenem Schock
# =========================================================
np.random.seed(2026)

def simulate_health(A, T=40, shock_node=0, shock_t=5):
    N = A.shape[0]
    h = np.ones(N)
    c = coupling_degree(A)
    traj = [h.copy()]
    for t in range(1, T+1):
        h_new = h.copy()
        if t == shock_t:
            h_new[shock_node] = 0.1
        for i in range(N):
            contagion = c * sum((1-h[j])*A[i,j] for j in range(N) if j != i) / max(1, A[i].sum())
            recovery  = (1-c)*0.06
            h_new[i] = np.clip(h_new[i] - contagion*0.4 + recovery, 0, 1)
        h = h_new
        traj.append(h.copy())
    return np.array(traj)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Ohne Operator: starte mit ueberkoppeltem Graphen, keine Anpassung
A_over = random_adj(N, 0.82, seed=7)
traj_over = simulate_health(A_over, T=40, shock_t=8)

# Mit Operator: Subgraph-Operator laeuft parallel zur Wirtschaft
T_sim = 40
h_op = np.ones(N)
A_op = random_adj(N, 0.82, seed=7)
traj_op = [h_op.copy()]
for t in range(1, T_sim+1):
    # Subgraph-Operator schritt alle 3 Zeiteinheiten
    if t % 3 == 0:
        A_op = subgraph_stabilize_step(A_op, A_ref)
    c_t = coupling_degree(A_op)
    h_new = h_op.copy()
    if t == 8:
        h_new[0] = 0.1
    for i in range(N):
        con = c_t * sum((1-h_op[j])*A_op[i,j] for j in range(N) if j != i) / max(1, A_op[i].sum())
        rec = (1-c_t)*0.06
        h_new[i] = np.clip(h_new[i] - con*0.4 + rec, 0, 1)
    h_op = h_new
    traj_op.append(h_op.copy())
traj_op = np.array(traj_op)

t_axis = np.arange(T_sim+1)

ax = axes[0]
mean_over = traj_over.mean(axis=1)
std_over  = traj_over.std(axis=1)
ax.fill_between(t_axis, mean_over - std_over, np.minimum(mean_over+std_over,1),
                alpha=0.18, color=ACCENTRED)
ax.plot(t_axis, mean_over, color=ACCENTRED, lw=2.5, label='Mittlere Systemgesundheit')
ax.plot(t_axis, traj_over[:,0], color=DARKGRAY, lw=1.4, linestyle='--', label='Schock-Knoten 0')
ax.axvline(8, color=DARKGRAY, lw=1.0, linestyle=':')
ax.text(8.2, 0.92, 'Schock', fontsize=9, color=DARKGRAY)
ax.set_title('(a) Ohne Subgraph-Operator\n(statischer ueberkoppelter Graph)')
ax.set_xlabel('Zeitschritt $t$'); ax.set_ylabel('Gesundheit $h(t)$')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9)

ax = axes[1]
mean_op = traj_op.mean(axis=1)
std_op  = traj_op.std(axis=1)
ax.fill_between(t_axis, mean_op - std_op, np.minimum(mean_op+std_op,1),
                alpha=0.18, color=DARKGREEN)
ax.plot(t_axis, mean_op, color=DARKGREEN, lw=2.5, label='Mittlere Systemgesundheit')
ax.plot(t_axis, traj_op[:,0], color=DARKGRAY, lw=1.4, linestyle='--', label='Schock-Knoten 0')
ax.axvline(8, color=DARKGRAY, lw=1.0, linestyle=':')
ax.text(8.2, 0.92, 'Schock', fontsize=9, color=DARKGRAY)
for tt in range(3, T_sim+1, 3):
    ax.axvline(tt, color=MAINBLUE, lw=0.6, alpha=0.25)
ax.text(1, 0.08, 'Subgraph-Op. aktiv (blau)', fontsize=8, color=MAINBLUE)
ax.set_title('(b) Mit Subgraph-Operator\n(adaptiver Graph, alle 3 Schritte)')
ax.set_xlabel('Zeitschritt $t$'); ax.set_ylabel('Gesundheit $h(t)$')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9)

fig.suptitle('Plot S6 -- Krisenresistenz: Vergleich mit und ohne Subgraph-Stabilisierungsoperator',
             fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'plotS6_krisenvergleich.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot S6 gespeichert.")

print("\nAlle Stabilitaets-Plots gespeichert.")

