#!/usr/bin/env python3
"""
Plots für: Dezentrale Wirtschaftszellen – Resilienzmechanismen
           durch optimale Entkopplung und positives Risikoexposure
Stephan Epp, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from scipy.stats import norm, expon
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ---- Farb-Schema (passend zur LaTeX-Vorlage) ----
MAINBLUE   = '#19468C'
ACCENTRED  = '#B4321E'
DARKGREEN  = '#1E6432'
DARKGRAY   = '#3C3C46'
LIGHTGRAY  = '#F5F5F8'
ORANGE     = '#E07820'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

OUTDIR = ''

# =========================================================
# Plot 1 – Kopplungsgrad vs. Systemresilienz (Phasendiagramm)
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))

coupling = np.linspace(0, 1, 400)

def resilience(c, alpha=3.5, c0=0.55):
    return np.exp(-alpha * (c - c0)**2) * (1 - 0.6 * c)

def systemic_risk(c):
    return 1 / (1 + np.exp(-10*(c - 0.55)))

res  = resilience(coupling)
risk = systemic_risk(coupling)

ax.plot(coupling, res,  color=MAINBLUE,  lw=2.5, label='Systemresilienz $R(c)$')
ax.plot(coupling, risk, color=ACCENTRED, lw=2.5, linestyle='--', label='Systemisches Risiko $\Psi(c)$')

ax.axvline(0.55, color=DARKGRAY, lw=1.2, linestyle=':', alpha=0.7)
ax.text(0.57, 0.82, 'Kritischer Kopplungspunkt\n$c^* = 0{,}55$',
        fontsize=9, color=DARKGRAY)

ax.fill_betweenx([0, 1], 0, 0.35,  alpha=0.07, color=DARKGREEN,
                 label='Optimale Dekopplungszone')
ax.fill_betweenx([0, 1], 0.70, 1.0, alpha=0.07, color=ACCENTRED,
                 label='Hochrisiko-Zone (Monokultur)')

ax.set_xlabel('Kopplungsgrad $c \in [0, 1]$')
ax.set_ylabel('Normierter Wert')
ax.set_title('Plot 1 – Systemresilienz und systemisches Risiko\nin Abhängigkeit vom Kopplungsgrad')
ax.legend(loc='upper right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot1_kopplung_resilienz.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 1 gespeichert.")

# =========================================================
# Plot 2 – Netzwerktopologie: Zentralisiert vs. Dezentralisiert
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# -- Zentralisiertes Netz (Stern) --
ax = axes[0]
n_periph = 8
center = np.array([0.5, 0.5])
angles = np.linspace(0, 2*np.pi, n_periph, endpoint=False)
periphery = np.column_stack([0.5 + 0.38*np.cos(angles),
                              0.5 + 0.38*np.sin(angles)])

for p in periphery:
    ax.annotate('', xy=p, xytext=center,
                arrowprops=dict(arrowstyle='-', color=ACCENTRED, lw=1.8, alpha=0.6))

ax.scatter(*center, s=300, color=ACCENTRED, zorder=5, label='Zentrum')
ax.scatter(periphery[:,0], periphery[:,1], s=120, color=MAINBLUE, zorder=5,
           label='Wirtschaftszellen')
ax.set_title('(a) Maximal gekoppeltes Netz\n(Monolithische Weltwirtschaft)', fontsize=11)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.legend(loc='lower right', fontsize=8.5)

# -- Dezentralisiertes Netz (Cluster) --
ax = axes[1]
np.random.seed(42)
cluster_centers = [(0.22, 0.75), (0.75, 0.75), (0.22, 0.25), (0.75, 0.25)]
colors_cl = [MAINBLUE, DARKGREEN, ORANGE, ACCENTRED]
all_nodes = []
node_colors = []
for (cx, cy), col in zip(cluster_centers, colors_cl):
    for _ in range(4):
        x = cx + np.random.uniform(-0.11, 0.11)
        y = cy + np.random.uniform(-0.11, 0.11)
        all_nodes.append((x, y))
        node_colors.append(col)

nodes = np.array(all_nodes)

# intra-cluster edges
for ci in range(4):
    idxs = list(range(ci*4, ci*4+4))
    for i in idxs:
        for j in idxs:
            if i < j:
                ax.plot([nodes[i,0], nodes[j,0]],
                        [nodes[i,1], nodes[j,1]],
                        color=node_colors[i], lw=1.2, alpha=0.55)

# sparse inter-cluster links (positive risk coupling)
inter = [(0, 4), (7, 8), (4, 12), (2, 13)]
for (a, b) in inter:
    ax.plot([nodes[a,0], nodes[b,0]], [nodes[a,1], nodes[b,1]],
            color=DARKGRAY, lw=1.5, alpha=0.4, linestyle='--')

ax.scatter(nodes[:,0], nodes[:,1], s=110, c=node_colors, zorder=5)
ax.set_title('(b) Dezentrales Netz mit\nentkoppelten Wirtschaftszellen', fontsize=11)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

patches = [mpatches.Patch(color=c, label=f'Zelle {i+1}')
           for i,(c,_) in enumerate(zip(colors_cl, cluster_centers))]
ax.legend(handles=patches, loc='lower right', fontsize=8.5)

fig.suptitle('Plot 2 – Netzwerktopologien im Vergleich', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot2_netzwerktopologie.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 2 gespeichert.")

# =========================================================
# Plot 3 – Schockausbreitung: gekoppelt vs. entkoppelt
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))

t = np.linspace(0, 10, 500)

def shock_coupled(t):
    return np.clip(0.9 * np.exp(-0.1*t) + 0.7 * np.sin(1.2*t) * np.exp(-0.2*t), 0, 1)

def shock_decoupled(t, delay=2.5, damping=0.7):
    result = np.zeros_like(t)
    mask = t >= delay
    result[mask] = damping * 0.35 * np.exp(-0.6*(t[mask]-delay)) * \
                   np.abs(np.sin(1.2*(t[mask]-delay)))
    return result

coupled   = shock_coupled(t)
decoupled = shock_decoupled(t)

ax.plot(t, coupled,   color=ACCENTRED, lw=2.5, label='Maximal gekoppeltes System')
ax.plot(t, decoupled, color=DARKGREEN, lw=2.5, linestyle='--',
        label='Dezentrales System (entkoppelt)')

ax.axvline(0, color=DARKGRAY, lw=1, linestyle=':')
ax.axvline(2.5, color=DARKGRAY, lw=1, linestyle=':', alpha=0.5)
ax.text(0.08, 0.92, 'Schock-\nUrsprung', fontsize=8.5, color=DARKGRAY)
ax.text(2.58, 0.38, 'Verzögerter\nEintritt', fontsize=8.5, color=DARKGREEN)

ax.set_xlabel('Zeit $t$')
ax.set_ylabel('Relative Wirtschaftsschädigung $\delta(t)$')
ax.set_title('Plot 3 – Schockausbreitung im gekoppelten vs. entkoppelten System')
ax.legend()
ax.set_xlim(0, 10); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot3_schockausbreitung.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 3 gespeichert.")

# =========================================================
# Plot 4 – Diversifikationsgewinn (Portfolio-Analogie)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# -- 4a: Varianz-Reduktion durch Entkopplung --
ax = axes[0]
n_cells = np.arange(1, 51)
rho_high = 0.8   # hohe Korrelation (gekoppelt)
rho_low  = 0.05  # niedrige Korrelation (entkoppelt)
sigma2   = 1.0

var_high = sigma2 * (1/n_cells + (1 - 1/n_cells)*rho_high)
var_low  = sigma2 * (1/n_cells + (1 - 1/n_cells)*rho_low)

ax.plot(n_cells, var_high, color=ACCENTRED, lw=2.5,
        label=f'Hohe Korrelation ($\\rho={rho_high}$, gekoppelt)')
ax.plot(n_cells, var_low,  color=DARKGREEN, lw=2.5, linestyle='--',
        label=f'Niedrige Korrelation ($\\rho={rho_low}$, entkoppelt)')
ax.axhline(rho_high, color=ACCENTRED, lw=1, linestyle=':', alpha=0.5)
ax.axhline(rho_low,  color=DARKGREEN, lw=1, linestyle=':', alpha=0.5)

ax.set_xlabel('Anzahl Wirtschaftszellen $N$')
ax.set_ylabel('Portfoliovarianz $\sigma^2_P$')
ax.set_title('(a) Varianzreduktion durch\nZellendiversifikation')
ax.legend(fontsize=9)

# -- 4b: Erwarteter Nutzen unter Risikoaversion --
ax = axes[1]
lam = np.linspace(0, 5, 300)  # Risikoaversion
mu_c, mu_d = 0.05, 0.04       # Erwartete Rendite
sig_c, sig_d = 0.30, 0.10     # Standardabw.

EU_coupled   = mu_c - lam * sig_c**2 / 2
EU_decoupled = mu_d - lam * sig_d**2 / 2

cross = (mu_c - mu_d) / ((sig_c**2 - sig_d**2)/2)

ax.plot(lam, EU_coupled,   color=ACCENTRED, lw=2.5, label='Gekoppelt (höhere Rendite, höheres Risiko)')
ax.plot(lam, EU_decoupled, color=DARKGREEN, lw=2.5, linestyle='--',
        label='Entkoppelt (geringere Rendite, geringeres Risiko)')
ax.axvline(cross, color=DARKGRAY, lw=1.5, linestyle=':')
ax.text(cross+0.1, -0.02, f'$\\lambda^* \\approx {cross:.2f}$', fontsize=9, color=DARKGRAY)
ax.axhline(0, color='black', lw=0.7, alpha=0.4)

ax.set_xlabel('Risikoaversionsparameter $\\lambda$')
ax.set_ylabel('Erwarteter Nutzen $EU(\\lambda)$')
ax.set_title('(b) Erwarteter Nutzen in Abhängigkeit\nvon der Risikoaversion')
ax.legend(fontsize=9)

fig.suptitle('Plot 4 – Portfoliotheoretische Analyse dezentraler Wirtschaftszellen', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot4_diversifikation.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 4 gespeichert.")

# =========================================================
# Plot 5 – Simulation: Krisenausbreitung über Zeitschritte
# =========================================================
np.random.seed(7)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
T = 30
N_cells_list = [5, 10, 20]
coupling_list = [0.9, 0.5, 0.1]  # Hochgekoppelt, Mittel, Entkoppelt
colors_sim = [ACCENTRED, ORANGE, DARKGREEN]
labels_sim = ['Hoch gekoppelt ($c=0.9$)', 'Mittel ($c=0.5$)', 'Entkoppelt ($c=0.1$)']

def simulate_crisis(N, c, T, seed=42):
    np.random.seed(seed)
    health = np.ones(N)
    health[0] = 0.0   # Initialschock in Zelle 0
    history = [health.copy()]
    for _ in range(T):
        new_health = health.copy()
        for i in range(N):
            # Ansteckung durch Nachbarn
            for j in range(N):
                if i != j:
                    contagion = c * (1 - health[j]) * np.random.uniform(0, 1)
                    new_health[i] = max(0, new_health[i] - contagion / N)
            # Erholung
            new_health[i] = min(1, new_health[i] + (1-c)*0.04)
        health = new_health
        history.append(health.copy())
    return np.array(history)

for ax, N, c, col, lab in zip(axes, N_cells_list, coupling_list, colors_sim, labels_sim):
    hist = simulate_crisis(N, c, T, seed=42)
    mean_health = hist.mean(axis=1)
    std_health  = hist.std(axis=1)
    t_sim = np.arange(T+1)

    ax.fill_between(t_sim, mean_health - std_health,
                    np.minimum(mean_health + std_health, 1),
                    alpha=0.2, color=col)
    ax.plot(t_sim, mean_health, color=col, lw=2.5, label='Mittlere Gesundheit')
    ax.plot(t_sim, hist[:,0], color=DARKGRAY, lw=1.2, linestyle='--',
            alpha=0.7, label='Zelle 0 (Ursprung)')

    ax.set_title(lab, fontsize=10)
    ax.set_xlabel('Zeitschritt $t$')
    ax.set_ylabel('Systemgesundheit $h(t)$')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)

fig.suptitle('Plot 5 – Monte-Carlo-Simulation der Krisenausbreitung\nbei verschiedenen Kopplungsgraden',
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot5_simulation.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 5 gespeichert.")

# =========================================================
# Plot 6 – Positives Risiko: Produktionsauslagerung als Hedge
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# -- 6a: Ertrag Inland vs. Ausland (negativ korreliert) --
ax = axes[0]
t_ec = np.linspace(0, 4*np.pi, 500)
domestic = np.sin(t_ec) * 0.6 + 0.5
foreign  = -np.sin(t_ec) * 0.4 + 0.55
combined = 0.5*domestic + 0.5*foreign

ax.plot(t_ec, domestic, color=ACCENTRED, lw=2, alpha=0.85, label='Inländische Produktion')
ax.plot(t_ec, foreign,  color=MAINBLUE,  lw=2, alpha=0.85, linestyle='--',
        label='Ausgelagerte Produktion (andere Zelle)')
ax.plot(t_ec, combined, color=DARKGREEN, lw=2.5, label='Kombinierter Ertrag (Portfolio)')

ax.axhline(combined.mean(), color=DARKGREEN, lw=1, linestyle=':', alpha=0.6)
ax.set_xlabel('Konjunkturzyklen (normiert)')
ax.set_ylabel('Relativer Produktionsertrag')
ax.set_title('(a) Hedge durch ausgelagerte\nProduktionsstätten')
ax.legend(fontsize=9)

# -- 6b: Value at Risk Vergleich --
ax = axes[1]
x = np.linspace(-0.5, 1.2, 1000)
mu1, sig1 = 0.05, 0.25   # Mono (maximale Kopplung)
mu2, sig2 = 0.05, 0.10   # Dezentral (entkoppelt)
mu3, sig3 = 0.06, 0.14   # Dezentral + positives Risiko

pdf1 = norm.pdf(x, mu1, sig1)
pdf2 = norm.pdf(x, mu2, sig2)
pdf3 = norm.pdf(x, mu3, sig3)

ax.fill_between(x, pdf1, alpha=0.18, color=ACCENTRED)
ax.fill_between(x, pdf2, alpha=0.18, color=DARKGREEN)
ax.fill_between(x, pdf3, alpha=0.18, color=MAINBLUE)
ax.plot(x, pdf1, color=ACCENTRED, lw=2, label='Monolithisch ($\\mu=5\\%$, $\\sigma=25\\%$)')
ax.plot(x, pdf2, color=DARKGREEN, lw=2, linestyle='--',
        label='Entkoppelt ($\\mu=5\\%$, $\\sigma=10\\%$)')
ax.plot(x, pdf3, color=MAINBLUE, lw=2, linestyle=':',
        label='Entkoppelt + pos. Risiko ($\\mu=6\\%$, $\\sigma=14\\%$)')

# VaR-Linien (5% Quantil)
for mu, sig, col in [(mu1,sig1,ACCENTRED),(mu2,sig2,DARKGREEN),(mu3,sig3,MAINBLUE)]:
    var = norm.ppf(0.05, mu, sig)
    ax.axvline(var, color=col, lw=1.2, linestyle='-.', alpha=0.7)

ax.set_xlabel('Jahresrendite')
ax.set_ylabel('Wahrscheinlichkeitsdichte')
ax.set_title('(b) Renditedichte und Value-at-Risk\n(gestrichelt: VaR 5%-Niveau)')
ax.legend(fontsize=8.5)

fig.suptitle('Plot 6 – Positives Risiko durch Produktionsauslagerung in andere Wirtschaftszellen',
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + 'plot6_positives_risiko.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 6 gespeichert.")

# =========================================================
# Plot 7 – Währungsunion vs. dezentrale Währungen
# =========================================================
fig, ax = plt.subplots(figsize=(9, 5))

years = np.arange(0, 30)
np.random.seed(2026)

def gdp_path(initial, trend, vol, shock_year, shock_size, coupling, seed=0):
    np.random.seed(seed)
    path = [initial]
    for t in range(1, 30):
        shock = shock_size * np.exp(-coupling*(t - shock_year)) if (shock_year is not None and t >= shock_year) else 0
        noise = np.random.normal(0, vol)
        path.append(path[-1] * (1 + trend + noise - max(0, shock)))
    return np.array(path)

# Währungsunion: hohe Kopplung → gemeinsamer Schock pflanzt sich fort
union_a = gdp_path(100, 0.025, 0.015, 10, 0.06, 0.9, seed=1)
union_b = gdp_path(100, 0.020, 0.018, 10, 0.07, 0.9, seed=2)
union_c = gdp_path(100, 0.030, 0.012, 10, 0.05, 0.9, seed=3)

# Dezentrales System: entkoppelt → unterschiedliche Schockzeiten
decent_a = gdp_path(100, 0.025, 0.015, 10,  0.06, 0.15, seed=1)
decent_b = gdp_path(100, 0.020, 0.018, 16,  0.04, 0.15, seed=2)
decent_c = gdp_path(100, 0.030, 0.012, None,0.0,  0.15, seed=3)  # kein Schock

for path, col, ls in [(union_a, ACCENTRED, '-'),
                       (union_b, ACCENTRED, '--'),
                       (union_c, ACCENTRED, ':')]:
    ax.plot(years, path, color=col, lw=1.8, alpha=0.75, linestyle=ls)

for path, col, ls in [(decent_a, DARKGREEN, '-'),
                       (decent_b, DARKGREEN, '--'),
                       (decent_c, DARKGREEN, ':')]:
    ax.plot(years, path, color=col, lw=1.8, alpha=0.75, linestyle=ls)

ax.axvline(10, color=DARKGRAY, lw=1.2, linestyle=':', alpha=0.6)
ax.text(10.3, 108, 'Externer Schock\n$t=10$', fontsize=9, color=DARKGRAY)

p1 = mpatches.Patch(color=ACCENTRED, label='Währungsunion (3 Mitgliedsstaaten, $c=0.9$)')
p2 = mpatches.Patch(color=DARKGREEN, label='Dezentrale Währungen (3 Zellen, $c=0.15$)')
ax.legend(handles=[p1, p2], fontsize=9.5)

ax.set_xlabel('Jahr $t$')
ax.set_ylabel('BIP-Index (Basis = 100)')
ax.set_title('Plot 7 – BIP-Entwicklung: Währungsunion vs. dezentrale Währungen\nbei exogenem Schock')
plt.tight_layout()
plt.savefig(OUTDIR + 'plot7_waehrung.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 7 gespeichert.")

# =========================================================
# Plot 8 – Optimaler Kopplungsgrad: Pareto-Frontier
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))

c_range = np.linspace(0.01, 0.99, 300)

# Wachstum: moderate Kopplung optimal (Handel)
def growth(c):
    return 0.04 * c**0.5 * (1 - 0.3*c) + 0.01

# Resilienz: abnehmend mit Kopplung
def resil(c):
    return 0.9 * (1 - c)**0.7 + 0.05

g = growth(c_range)
r = resil(c_range)

# Pareto-Frontier (Parametrisch)
sc = ax.scatter(r, g, c=c_range, cmap='RdYlGn_r', s=15, zorder=3)

# Optimum annotieren
pareto_score = g + r
opt_idx = np.argmax(pareto_score)
ax.scatter(r[opt_idx], g[opt_idx], color='gold', edgecolor='black',
           s=200, zorder=5, label=f'Pareto-Optimum ($c^*={c_range[opt_idx]:.2f}$)')
ax.annotate(f'  $c^* = {c_range[opt_idx]:.2f}$', (r[opt_idx], g[opt_idx]),
            fontsize=10, color='black')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Kopplungsgrad $c$', fontsize=10)

ax.set_xlabel('Systemresilienz $R$')
ax.set_ylabel('Wirtschaftswachstum $g$')
ax.set_title('Plot 8 – Pareto-Frontier: Abwägung zwischen\nWachstum und Resilienz')
ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR + 'plot8_pareto.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Plot 8 gespeichert.")

print("\nAlle 8 Plots erfolgreich gespeichert.")
