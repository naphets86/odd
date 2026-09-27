#!/usr/bin/env python3
"""
Plots fuer Kapitel 10: Dynamische Graphen -- Zeitvariante Wirtschaftszellen
Stephan Epp, 2026
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigvals, expm
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
DARKGRAY  = '#3C3C46'
ORANGE    = '#E07820'
GOLD      = '#C8960A'
PURPLE    = '#6B2D8B'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

OUT = ''
np.random.seed(2026)

# ============================================================
# Hilfsfunktionen
# ============================================================
def coupling_degree(W):
    N = W.shape[0]
    off = W - np.diag(np.diag(W))
    denom = N*(N-1)
    return float(off.sum()/denom) if denom > 0 else 0.0

def make_cluster_adj(N, K, c_intra, c_inter, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((N,N))
    sz = N // K
    for k in range(K):
        for i in range(k*sz, (k+1)*sz):
            for j in range(k*sz, (k+1)*sz):
                if i != j:
                    W[i,j] = c_intra if rng.random() < c_intra else 0.0
    for i in range(N):
        for j in range(N):
            if i != j and W[i,j] == 0:
                if rng.random() < c_inter:
                    W[i,j] = c_inter
    np.fill_diagonal(W, 0)
    return np.clip(W, 0, 1)

# ============================================================
# Plot D1 -- Zeitliche Evolution der Gewichtsmatrix W(t)
#            unter drei Regime-Typen
# ============================================================
N, T = 8, 60
t_axis = np.arange(T+1)

# Regime 1: Graduelle Kopplung (Globalisierungsphase)
def regime_globalisierung(t, W0, rate=0.008):
    W = W0.copy()
    for _ in range(t):
        noise = np.random.uniform(-0.01, 0.03, W.shape)
        W = np.clip(W + rate + noise, 0, 1)
        np.fill_diagonal(W, 0)
    return W

# Regime 2: Schockdekopplung (Krisendekopplung)
def regime_krise(t, W0, shock_t=20, decay=0.025):
    W = W0.copy()
    for tt in range(t):
        if tt >= shock_t:
            W = np.clip(W - decay, 0, 1)
        else:
            noise = np.random.uniform(-0.005, 0.01, W.shape)
            W = np.clip(W + noise, 0, 1)
        np.fill_diagonal(W, 0)
    return W

# Regime 3: Zyklische Fluktuation (Konjunkturzyklen)
def c_zyklisch(t_arr, c0=0.4, amp=0.25, omega=0.18):
    return c0 + amp * np.sin(omega * t_arr)

W0 = make_cluster_adj(N, 2, 0.55, 0.08, seed=7)

c_glob  = []
c_krise = []
c_zykl  = c_zyklisch(t_axis)

W_g = W0.copy(); W_k = W0.copy()
for t in range(T+1):
    c_glob.append(coupling_degree(W_g))
    c_krise.append(coupling_degree(W_k))
    if t < T:
        noise_g = np.random.uniform(-0.01, 0.025, W_g.shape)
        W_g = np.clip(W_g + 0.007 + noise_g, 0, 1); np.fill_diagonal(W_g, 0)
        if t >= 20:
            W_k = np.clip(W_k - 0.022, 0, 1)
        else:
            noise_k = np.random.uniform(-0.005, 0.01, W_k.shape)
            W_k = np.clip(W_k + noise_k, 0, 1)
        np.fill_diagonal(W_k, 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(t_axis, c_glob,  color=ACCENTRED, lw=2.3, label='Globalisierungsregime')
ax.plot(t_axis, c_krise, color=MAINBLUE,  lw=2.3, label='Krisendekopplung ($t^*=20$)')
ax.plot(t_axis, c_zykl,  color=DARKGREEN, lw=2.3, linestyle='--', label='Zyklisches Regime')
ax.axvline(20, color=DARKGRAY, lw=1.1, linestyle=':', alpha=0.7)
ax.text(20.5, 0.78, 'Schock $t^*$', fontsize=9, color=DARKGRAY)
ax.set_xlabel('Zeitschritt $t$'); ax.set_ylabel('Kopplungsgrad $c(t)$')
ax.set_title('(a) Trajektorien des Kopplungsgrades\nunter verschiedenen Regimes')
ax.set_ylim(0, 1); ax.legend(fontsize=9.5)

# Phasendarstellung c(t) vs. c'(t)
ax = axes[1]
dc_glob  = np.gradient(c_glob)
dc_krise = np.gradient(c_krise)
dc_zykl  = np.gradient(c_zykl)
ax.plot(c_glob,  dc_glob,  color=ACCENTRED, lw=1.8, label='Globalisierung')
ax.plot(c_krise, dc_krise, color=MAINBLUE,  lw=1.8, label='Krise')
ax.plot(c_zykl,  dc_zykl,  color=DARKGREEN, lw=1.8, linestyle='--', label='Zyklisch')
ax.scatter(c_glob[0],  dc_glob[0],  color=ACCENTRED, s=70, zorder=5)
ax.scatter(c_krise[0], dc_krise[0], color=MAINBLUE,  s=70, zorder=5)
ax.scatter(c_zykl[0],  dc_zykl[0],  color=DARKGREEN, s=70, zorder=5)
ax.axhline(0, color=DARKGRAY, lw=0.8, alpha=0.5)
ax.set_xlabel('$c(t)$'); ax.set_ylabel("$\\dot{c}(t)$")
ax.set_title('(b) Phasenporträt $c(t)$ vs.\ $\\dot{c}(t)$\n(Pfeile: Zeitrichtung)')
ax.legend(fontsize=9.5)

fig.suptitle('Plot D1 -- Zeitliche Evolution des Kopplungsgrades unter drei Regime-Typen',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD1_regimes.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D1 gespeichert.")

# ============================================================
# Plot D2 -- Zeitvariante Spektraleigenschaften: Floquet-Analyse
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Monodromiematrix: Produkt der Uebergangsmatrizen ueber eine Periode
T_period = 35   # Laenge einer zyklischen Periode
omega_fl = 2*np.pi / T_period
N_fl = 6

def W_t(t, N, amp=0.25, c0=0.4):
    """Zeitvariante Gewichtsmatrix (zyklisch)."""
    c = c0 + amp * np.sin(omega_fl * t)
    W = np.ones((N,N)) * c
    np.fill_diagonal(W, 0)
    # Clusterbias
    for i in range(N//2):
        for j in range(N//2):
            if i != j: W[i,j] = min(1, c + 0.15)
    for i in range(N//2, N):
        for j in range(N//2, N):
            if i != j: W[i,j] = min(1, c + 0.15)
    return W

# Floquet-Multiplikatoren (Eigenwerte der Monodromiematrix)
dt = 0.5
Phi = np.eye(N_fl)
for ti in range(int(T_period/dt)):
    A_ti = -0.5 * W_t(ti*dt, N_fl) + 0.05 * np.eye(N_fl)
    Phi = expm(A_ti * dt) @ Phi

floquet_eigs = eigvals(Phi)
ax = axes[0]
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color=DARKGRAY, lw=1.2,
        linestyle='--', alpha=0.5, label='Einheitskreis')
ax.scatter(floquet_eigs.real, floquet_eigs.imag,
           color=MAINBLUE, s=120, zorder=5, label='Floquet-Multiplikatoren')
for e in floquet_eigs:
    ax.annotate(f'  {abs(e):.3f}', (e.real, e.imag), fontsize=8, color=DARKGRAY)
ax.set_xlabel('Re($\\mu_i$)'); ax.set_ylabel('Im($\\mu_i$)')
ax.set_title('(a) Floquet-Multiplikatoren\n$|\\mu_i| < 1$ $\\Rightarrow$ asympt. stabil')
ax.set_aspect('equal'); ax.axhline(0, lw=0.7, color='k', alpha=0.3)
ax.axvline(0, lw=0.7, color='k', alpha=0.3)
ax.legend(fontsize=9)

# Zeitvarianter Spektralradius
ax = axes[1]
t_fl = np.linspace(0, 2*T_period, 200)
rho_t = []
for tt in t_fl:
    Wt = W_t(tt, N_fl)
    c_t = coupling_degree(Wt)
    Lam = c_t * 0.6 * Wt / (N_fl-1)
    rho_t.append(float(max(abs(eigvals(Lam))).real))
ax.plot(t_fl, rho_t, color=MAINBLUE, lw=2.3, label='$\\varrho(\\Lambda(t))$')
ax.axhline(1.0, color=ACCENTRED, lw=1.5, linestyle='--', label='Stabilitätsgrenze')
ax.fill_between(t_fl, rho_t, 1.0, where=np.array(rho_t) > 1.0,
                alpha=0.15, color=ACCENTRED, label='Instabilitätsphase')
ax.fill_between(t_fl, rho_t, 1.0, where=np.array(rho_t) <= 1.0,
                alpha=0.08, color=DARKGREEN)
ax.set_xlabel('Zeit $t$'); ax.set_ylabel('$\\varrho(\\Lambda(t))$')
ax.set_title('(b) Zeitvarianter Spektralradius\n(2 Perioden)')
ax.legend(fontsize=9)

fig.suptitle('Plot D2 -- Floquet-Theorie: Spektrale Stabilität zeitvarianter Wirtschaftsgraphen',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD2_floquet.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D2 gespeichert.")

# ============================================================
# Plot D3 -- Stochastische Differentialgleichung: Ito-Dynamik
#            des Kopplungsgrades
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ito-SDE: dc = mu_c(c) dt + sigma_c(c) dW
# Mean-reverting: mu_c = kappa*(c* - c),  sigma_c = sigma*sqrt(c*(1-c))
c_star = 0.30
kappa  = 0.18
sigma_sde = 0.12
dt_sde = 0.1
T_sde  = 80
N_paths = 5
t_sde = np.arange(0, T_sde + dt_sde, dt_sde)

ax = axes[0]
colors_paths = [ACCENTRED, MAINBLUE, DARKGREEN, ORANGE, PURPLE]
all_paths = []
for k, col in enumerate(colors_paths):
    c0 = np.random.uniform(0.1, 0.85)
    path = [c0]
    c_curr = c0
    for _ in t_sde[1:]:
        mu  = kappa * (c_star - c_curr)
        sig = sigma_sde * np.sqrt(max(c_curr*(1-c_curr), 1e-6))
        dW  = np.random.normal(0, np.sqrt(dt_sde))
        c_curr = np.clip(c_curr + mu*dt_sde + sig*dW, 0.01, 0.99)
        path.append(c_curr)
    path = np.array(path)
    all_paths.append(path)
    ax.plot(t_sde, path, color=col, lw=1.6, alpha=0.75,
            label=f'Pfad {k+1} ($c_0={path[0]:.2f}$)')

ax.axhline(c_star, color=DARKGRAY, lw=2, linestyle='--',
           label=f'Gleichgewicht $c^* = {c_star}$')
ax.set_xlabel('Zeit $t$'); ax.set_ylabel('$c(t)$')
ax.set_title('(a) Ito-Pfade des Kopplungsgrades\n(Mean-Reversion mit Diffusion)')
ax.legend(fontsize=8.5); ax.set_ylim(0, 1)

# Stationaere Verteilung (Beta-Approximation)
ax = axes[1]
all_vals = np.concatenate([p[200:] for p in all_paths])
ax.hist(all_vals, bins=35, density=True, color=MAINBLUE, alpha=0.5,
        label='Empirische Verteilung')
from scipy.stats import beta as beta_dist
# Beta-Fit
a_fit = c_star * (c_star*(1-c_star)/sigma_sde**2 - 1)
b_fit = (1-c_star) * (c_star*(1-c_star)/sigma_sde**2 - 1)
a_fit = max(a_fit, 0.5); b_fit = max(b_fit, 0.5)
x_b = np.linspace(0.01, 0.99, 200)
ax.plot(x_b, beta_dist.pdf(x_b, a_fit, b_fit),
        color=ACCENTRED, lw=2.3, label=f'Beta$({a_fit:.1f},{b_fit:.1f})$-Approx.')
ax.axvline(c_star, color=DARKGRAY, lw=1.8, linestyle='--',
           label=f'$c^* = {c_star}$')
ax.set_xlabel('$c$'); ax.set_ylabel('Dichte $p_\\infty(c)$')
ax.set_title('(b) Stationäre Verteilung $p_\\infty(c)$\n(Langzeitverhalten)')
ax.legend(fontsize=9)

fig.suptitle('Plot D3 -- Stochastische Kopplung: Ito-SDE-Dynamik des Kopplungsgrades',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD3_sde.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D3 gespeichert.")

# ============================================================
# Plot D4 -- Strukturbrüche: Sprungprozesse im Wirtschaftsgraphen
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compound Poisson-Prozess fuer Kopplungssprunge
T_jump = 100
lam_jump = 0.06   # Sprungrate (Ereignisse pro Zeiteinheit)
mu_jump  = -0.18  # Mittlere Sprunggroesse (negativ = Dekopplung bei Krisen)
sig_jump = 0.10
dt_j = 0.5
t_j = np.arange(0, T_jump, dt_j)

def simulate_jump_diffusion(c0, T, dt, lam, mu_j, sig_j, kappa=0.12, c_star=0.35,
                             sigma_d=0.06, seed=0):
    np.random.seed(seed)
    path = [c0]
    c = c0
    for _ in t_j[1:]:
        # Diffusionsanteil
        dW = np.random.normal(0, np.sqrt(dt))
        drift = kappa * (c_star - c)
        diff  = sigma_d * dW
        # Sprunganteil (Poisson)
        n_jumps = np.random.poisson(lam * dt)
        jump = sum(np.random.normal(mu_j, sig_j) for _ in range(n_jumps))
        c = np.clip(c + drift*dt + diff + jump, 0.02, 0.98)
        path.append(c)
    return np.array(path)

ax = axes[0]
for seed, col, lab in [(0, MAINBLUE, 'Pfad 1'),
                        (1, ACCENTRED, 'Pfad 2'),
                        (2, DARKGREEN, 'Pfad 3')]:
    p = simulate_jump_diffusion(0.55, T_jump, dt_j, lam_jump, mu_jump,
                                sig_jump, seed=seed)
    ax.plot(t_j, p, color=col, lw=1.7, alpha=0.8, label=lab)
ax.axhline(0.35, color=DARKGRAY, lw=1.6, linestyle='--', label='$c^* = 0.35$')
ax.set_xlabel('Zeit $t$'); ax.set_ylabel('$c(t)$')
ax.set_title('(a) Sprung-Diffusions-Prozess\n(Krisenereignisse als Poisson-Sprünge)')
ax.legend(fontsize=9); ax.set_ylim(0, 1)

# Wartezeitverteilung zwischen Strukturbruchen
ax = axes[1]
N_sim = 5000
wait_times = np.random.exponential(1/lam_jump, N_sim)
ax.hist(wait_times, bins=50, density=True, color=MAINBLUE, alpha=0.5,
        label='Simulierte Wartezeiten')
x_exp = np.linspace(0, 60, 300)
ax.plot(x_exp, lam_jump * np.exp(-lam_jump * x_exp),
        color=ACCENTRED, lw=2.3,
        label=f'Exp$({lam_jump})$-Verteilung')
ax.set_xlabel('Wartezeit $\\tau$ bis nächstem Ereignis')
ax.set_ylabel('Dichte')
ax.set_title('(b) Wartezeiten zwischen Strukturbrüchen\n(Exponentialverteilung)')
ax.legend(fontsize=9)

fig.suptitle('Plot D4 -- Strukturbrüche: Sprung-Diffusions-Dynamik im Wirtschaftsgraphen',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD4_sprungprozess.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D4 gespeichert.")

# ============================================================
# Plot D5 -- Zeitvariante Resilienz: R(t) unter dynamischer Kopplung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

T_res = 80
t_res = np.linspace(0, T_res, 400)

# Drei Szenarien
def c_of_t(t, regime):
    if regime == 'globalisierung':
        return np.clip(0.25 + 0.008*t + 0.04*np.sin(0.3*t), 0, 1)
    elif regime == 'stabil':
        return 0.30 + 0.05*np.sin(0.18*t + 0.5)
    elif regime == 'krise':
        return np.where(t < 30, 0.35 + 0.005*t,
               np.where(t < 50, 0.35 + 0.005*30 - 0.04*(t-30),
                        0.35 + 0.005*30 - 0.04*20 + 0.012*(t-50)))

def resilience_of_c(c, alpha=3.5, c0=0.55):
    return np.exp(-alpha*(c-c0)**2) * (1 - 0.6*c)

ax = axes[0]
for reg, col, lab in [('globalisierung', ACCENTRED, 'Globalisierungsregime'),
                       ('stabil', DARKGREEN,  'Stabiles Regime'),
                       ('krise',  MAINBLUE,   'Krisenzyklus')]:
    c_t = c_of_t(t_res, reg)
    R_t = resilience_of_c(c_t)
    ax.plot(t_res, R_t, color=col, lw=2.2, label=lab)
    ax.plot(t_res, c_t, color=col, lw=1.3, linestyle=':', alpha=0.5)

ax.set_xlabel('Zeit $t$')
ax.set_ylabel('$R(t)$ (durchgezogen) / $c(t)$ (gepunktet)')
ax.set_title('(a) Zeitvariante Resilienz $R(t)$\n(gepunktet: zugehöriger $c(t)$-Verlauf)')
ax.legend(fontsize=9.5)

# Integrale Resilienz (Zeitdurchschnitt)
ax = axes[1]
T_vals = np.linspace(5, T_res, 100)
for reg, col, lab in [('globalisierung', ACCENTRED, 'Globalisierung'),
                       ('stabil', DARKGREEN, 'Stabil'),
                       ('krise',  MAINBLUE,  'Krise')]:
    int_res = []
    for Tv in T_vals:
        t_v = np.linspace(0, Tv, 200)
        c_v = c_of_t(t_v, reg)
        R_v = resilience_of_c(c_v)
        int_res.append(np.trapezoid(R_v, t_v) / Tv)
    ax.plot(T_vals, int_res, color=col, lw=2.2, label=lab)

ax.set_xlabel('Beobachtungshorizont $T$')
ax.set_ylabel('Integrale Resilienz $\\bar{R}(T)$')
ax.set_title('(b) Integrale Resilienz $\\bar{R}(T) = \\frac{1}{T}\\int_0^T R(t)\\,dt$')
ax.legend(fontsize=9.5)

fig.suptitle('Plot D5 -- Zeitvariante Resilienz unter dynamischer Kopplung',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD5_resilienz_dynamisch.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D5 gespeichert.")

# ============================================================
# Plot D6 -- Subgraph-Operator auf dynamischem Graphen:
#            Tracking des Referenzpfades c*(t)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Zeitvarianter Referenzpfad c*(t) (Pareto-optimal unter sich aendernden Bedingungen)
T_track = 60
t_tr = np.arange(T_track+1)
c_star_t = 0.28 + 0.10*np.sin(2*np.pi*t_tr/T_track) + 0.04*np.cos(4*np.pi*t_tr/T_track)

np.random.seed(42)
# Ohne adaptiven Operator: c(t) folgt eigenem Regime
c_no_op = np.zeros(T_track+1)
c_no_op[0] = 0.70
for t in range(1, T_track+1):
    noise = np.random.normal(0, 0.03)
    c_no_op[t] = np.clip(c_no_op[t-1] * 0.97 + 0.01 + noise, 0, 1)

# Mit adaptivem Subgraph-Operator: verfolgt c*(t)
c_with_op = np.zeros(T_track+1)
c_with_op[0] = 0.70
delta_adapt = 0.12
for t in range(1, T_track+1):
    err = c_with_op[t-1] - c_star_t[t-1]
    noise = np.random.normal(0, 0.015)
    correction = -delta_adapt * np.sign(err) * min(abs(err), 0.15)
    c_with_op[t] = np.clip(c_with_op[t-1] + correction + noise*0.3, 0.05, 0.95)

ax = axes[0]
ax.plot(t_tr, c_star_t,   color=DARKGRAY,   lw=2.2, linestyle='--', label='Referenzpfad $c^*(t)$')
ax.plot(t_tr, c_no_op,    color=ACCENTRED,  lw=2.0, label='Ohne Operator')
ax.plot(t_tr, c_with_op,  color=DARKGREEN,  lw=2.0, label='Mit adapt. Subgraph-Operator')
ax.fill_between(t_tr,
                np.minimum(c_no_op, c_star_t),
                np.maximum(c_no_op, c_star_t),
                alpha=0.12, color=ACCENTRED)
ax.fill_between(t_tr,
                np.minimum(c_with_op, c_star_t),
                np.maximum(c_with_op, c_star_t),
                alpha=0.12, color=DARKGREEN)
ax.set_xlabel('Zeit $t$'); ax.set_ylabel('$c(t)$')
ax.set_title('(a) Tracking des zeitvarianten Referenzpfads\ndurch adaptiven Subgraph-Operator')
ax.legend(fontsize=9.5)

# Tracking-Fehler
ax = axes[1]
err_no  = np.abs(c_no_op   - c_star_t)
err_op  = np.abs(c_with_op - c_star_t)
ax.plot(t_tr, err_no, color=ACCENTRED, lw=2.0, label='Tracking-Fehler (ohne Operator)')
ax.plot(t_tr, err_op, color=DARKGREEN, lw=2.0, label='Tracking-Fehler (mit Operator)')
ax.fill_between(t_tr, err_op, err_no, alpha=0.12, color=MAINBLUE,
                label='Verbesserung durch Operator')
ax.set_xlabel('Zeit $t$')
ax.set_ylabel('$|c(t) - c^*(t)|$')
ax.set_title('(b) Tracking-Fehler $|c(t) - c^*(t)|$')
ax.legend(fontsize=9.5)

fig.suptitle('Plot D6 -- Adaptives Tracking des Pareto-Referenzpfades durch den Subgraph-Operator',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT + 'plotD6_tracking.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D6 gespeichert.")

# ============================================================
# Plot D7 -- Zeitvariante Netzwerktopologie: Snapshot-Sequenz
# ============================================================
import matplotlib.cm as cm

fig = plt.figure(figsize=(14, 4))
gs  = GridSpec(1, 4, figure=fig, wspace=0.05)

N_snap = 8
t_snaps = [0, 20, 40, 60]
labels  = ['$t=0$\n(Gleichgewicht)', '$t=20$\n(Schock)', '$t=40$\n(Erholung)', '$t=60$\n(Neues GGW)']

def W_snap(t, N):
    rng = np.random.default_rng(seed=int(t*7+3))
    c_base = 0.30 + 0.35 * np.exp(-0.015*t) * abs(np.sin(0.08*t + 0.5))
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                cluster_bonus = 0.20 if (i < N//2) == (j < N//2) else 0
                w = c_base + cluster_bonus + rng.uniform(-0.05, 0.05)
                W[i,j] = np.clip(w, 0, 1)
    np.fill_diagonal(W, 0)
    return W

cmap_w = LinearSegmentedColormap.from_list('coupling', ['white', '#19468C'])
for idx, (t_s, lab) in enumerate(zip(t_snaps, labels)):
    ax = fig.add_subplot(gs[idx])
    W_s = W_snap(t_s, N_snap)
    im = ax.imshow(W_s, cmap=cmap_w, vmin=0, vmax=1, aspect='auto')
    ax.set_title(lab, fontsize=10)
    c_val = coupling_degree(W_s)
    ax.set_xlabel(f'$c={c_val:.3f}$', fontsize=9, color=ACCENTRED)
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.02).set_label('$w_{ij}(t)$', fontsize=10)
fig.suptitle('Plot D7 -- Snapshot-Sequenz der Gewichtsmatrix $W(t)$:\nzeitvariante Netzwerktopologie',
             fontsize=12, y=1.04)
plt.savefig(OUT + 'plotD7_snapshots.pdf', format='pdf', bbox_inches='tight')
plt.close(); print("Plot D7 gespeichert.")

print("\nAlle D-Plots erfolgreich gespeichert.")
