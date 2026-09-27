"""
Erzeugung aller Plots fuer die wissenschaftliche Arbeit:
"Genetische Prädisposition und Stressbelastung als interagierende
Determinanten des Krebsausbruchs: Ein mathematisches Modellframework"
Autor: Stephan Epp, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit  # sigmoid
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ─── Globale Farbpalette ──────────────────────────────────────────────────────
C_BLUE   = "#1A3A5C"
C_RED    = "#8B1A1A"
C_GREEN  = "#1A5C2A"
C_ORANGE = "#C45C00"
C_PURPLE = "#4B0082"
C_TEAL   = "#006666"
C_GRAY   = "#404040"
C_BG     = "#F8F8F6"

PLOT_DIR = ""

# ─── Modellparameter ──────────────────────────────────────────────────────────
ALPHA = 4.0   # Gewicht genetische Prädisposition
BETA  = 4.0   # Gewicht Stressbelastung
GAMMA = 3.0   # Interaktionsterm
THETA = 4.5   # Schwellenwert
P_CRIT = 0.5  # kritische Ausbruchswahrscheinlichkeit

def P(G, S, alpha=ALPHA, beta=BETA, gamma=GAMMA, theta=THETA):
    """Krebsausbruch-Wahrscheinlichkeit P(G,S) = sigma(alpha*G + beta*S + gamma*G*S - theta)"""
    z = alpha * G + beta * S + gamma * G * S - theta
    return expit(z)

def argmin_S_for_G(G_val, p_target=P_CRIT):
    """Berechnet S* sodass P(G_val, S*) = p_target (Kompensationsschwelle)."""
    # sigma(alpha*G + beta*S + gamma*G*S - theta) = p_target
    # alpha*G + (beta + gamma*G)*S - theta = logit(p_target)
    from scipy.special import logit
    L = logit(p_target)
    denom = BETA + GAMMA * G_val
    if denom < 1e-12:
        return np.nan
    S_star = (L - ALPHA * G_val + THETA) / denom
    return np.clip(S_star, 0, 1)

# Gitter
N = 400
G_arr = np.linspace(0, 1, N)
S_arr = np.linspace(0, 1, N)
GG, SS = np.meshgrid(G_arr, S_arr)
PP = P(GG, SS)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Heatmap P(G,S)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=C_BG)
ax.set_facecolor(C_BG)

cmap_custom = cm.RdYlGn_r
img = ax.contourf(G_arr, S_arr, PP, levels=100, cmap=cmap_custom)
cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$P(G,S)$  –  Ausbruchswahrscheinlichkeit', fontsize=11, color=C_GRAY)
cbar.ax.tick_params(colors=C_GRAY)

# Kritische Konturlinie p = 0.5
cs = ax.contour(G_arr, S_arr, PP, levels=[P_CRIT], colors=[C_BLUE], linewidths=2.2, linestyles='--')
ax.clabel(cs, fmt={P_CRIT: r'$P=0{,}5$'}, inline=True, fontsize=11, colors=C_BLUE)

ax.set_xlabel(r'Genetische Prädisposition $G \in [0,1]$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'Stressbelastung $S \in [0,1]$', fontsize=12, color=C_GRAY)
ax.set_title('Heatmap der Krebsausbruch-Wahrscheinlichkeit $P(G,S)$', fontsize=13, color=C_BLUE, pad=12)
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)

# Annotationen
ax.text(0.08, 0.88, 'Kein Ausbruch\n(trotz hoher Veranlagung)', fontsize=9,
        color='white', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_GREEN, alpha=0.7),
        transform=ax.transAxes)
ax.text(0.82, 0.15, 'Ausbruch\n(trotz geringer Veranlagung)', fontsize=9,
        color='white', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_RED, alpha=0.7),
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_heatmap.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_heatmap.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Iso-Wahrscheinlichkeits-Kurven (Isocontours)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=C_BG)
ax.set_facecolor(C_BG)

levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors_iso = [cm.RdYlGn_r(v) for v in levels]

cs2 = ax.contour(G_arr, S_arr, PP, levels=levels, colors=colors_iso, linewidths=1.8)
fmt = {lev: f'$P={lev:.1f}$' for lev in levels}
ax.clabel(cs2, fmt=fmt, inline=True, fontsize=9)

# Kritische Linie hervorheben
cs_crit = ax.contour(G_arr, S_arr, PP, levels=[0.5], colors=[C_BLUE], linewidths=2.8, linestyles='-')
ax.clabel(cs_crit, fmt={0.5: r'Kritische Schwelle $P=0{,}5$'}, inline=True, fontsize=10, colors=C_BLUE)

ax.set_xlabel(r'Genetische Prädisposition $G$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'Stressbelastung $S$', fontsize=12, color=C_GRAY)
ax.set_title('Iso-Wahrscheinlichkeits-Kurven $C_p = \\{(G,S) : P(G,S) = p\\}$', fontsize=13, color=C_BLUE, pad=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)

# Pfeile: Kompensationsrichtungen
ax.annotate('', xy=(0.15, 0.55), xytext=(0.55, 0.15),
            arrowprops=dict(arrowstyle='<->', color=C_PURPLE, lw=1.8))
ax.text(0.38, 0.3, 'Kompensationsrichtung', fontsize=9, color=C_PURPLE, rotation=-40, va='center')

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_isocontours.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_isocontours.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: 3D-Oberflächenplot P(G,S)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7), facecolor=C_BG)
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_facecolor(C_BG)

N3 = 80
G3 = np.linspace(0, 1, N3)
S3 = np.linspace(0, 1, N3)
GG3, SS3 = np.meshgrid(G3, S3)
PP3 = P(GG3, SS3)

surf = ax3d.plot_surface(GG3, SS3, PP3, cmap=cm.RdYlGn_r, alpha=0.92,
                          linewidth=0, antialiased=True, rstride=1, cstride=1)

# Kritische Ebene P = 0.5
GGf, SSf = np.meshgrid([0, 1], [0, 1])
PPf = np.full_like(GGf, 0.5, dtype=float)
ax3d.plot_surface(GGf, SSf, PPf, alpha=0.22, color=C_BLUE)

cbar3 = fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.1)
cbar3.set_label(r'$P(G,S)$', fontsize=11, color=C_GRAY)
cbar3.ax.tick_params(colors=C_GRAY)

ax3d.set_xlabel(r'$G$  (Prädisposition)', fontsize=10, color=C_GRAY, labelpad=8)
ax3d.set_ylabel(r'$S$  (Stress)', fontsize=10, color=C_GRAY, labelpad=8)
ax3d.set_zlabel(r'$P(G,S)$', fontsize=10, color=C_GRAY, labelpad=8)
ax3d.set_title('3D-Oberfläche der Ausbruchswahrscheinlichkeit', fontsize=12, color=C_BLUE, pad=15)
ax3d.view_init(elev=28, azim=-55)
ax3d.tick_params(colors=C_GRAY, labelsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_3d_surface.pdf', dpi=180, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_3d_surface.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Querschnitte P vs S für feste G-Werte
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.set_facecolor(C_BG)

G_fixed_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors_G = [C_GREEN, "#2E8B57", C_ORANGE, "#CC6600", C_RED, "#5C0000"]
S_range = np.linspace(0, 1, 500)

for g_val, col in zip(G_fixed_vals, colors_G):
    p_vals = P(g_val, S_range)
    lw = 2.5 if g_val in [0.0, 1.0] else 1.8
    ax.plot(S_range, p_vals, color=col, lw=lw,
            label=f'$G = {g_val:.1f}$')
    # S* Schnittpunkt mit p=0.5
    S_star = argmin_S_for_G(g_val)
    if 0 <= S_star <= 1:
        ax.plot(S_star, 0.5, 'o', color=col, ms=6, zorder=5)

ax.axhline(0.5, color=C_BLUE, lw=1.5, ls='--', label=r'Kritische Schwelle $P=0{,}5$')
ax.axhspan(0, 0.5, alpha=0.06, color=C_GREEN)
ax.axhspan(0.5, 1.0, alpha=0.06, color=C_RED)
ax.text(0.85, 0.25, 'Kein Ausbruch', fontsize=10, color=C_GREEN, alpha=0.8)
ax.text(0.85, 0.75, 'Ausbruch', fontsize=10, color=C_RED, alpha=0.8)

ax.set_xlabel(r'Stressbelastung $S$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'$P(G,S)$  –  Ausbruchswahrscheinlichkeit', fontsize=12, color=C_GRAY)
ax.set_title('Querschnitte $P(G_0, S)$ für feste Prädispositionswerte $G_0$', fontsize=12, color=C_BLUE, pad=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)
ax.grid(True, alpha=0.25, color=C_GRAY)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_cross_sections_G.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_cross_sections_G.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Querschnitte P vs G für feste S-Werte
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.set_facecolor(C_BG)

S_fixed_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors_S = [C_TEAL, "#008080", "#6699CC", "#3366AA", C_PURPLE, "#2B0057"]
G_range = np.linspace(0, 1, 500)

for s_val, col in zip(S_fixed_vals, colors_S):
    p_vals = P(G_range, s_val)
    lw = 2.5 if s_val in [0.0, 1.0] else 1.8
    ax.plot(G_range, p_vals, color=col, lw=lw,
            label=f'$S = {s_val:.1f}$')

ax.axhline(0.5, color=C_BLUE, lw=1.5, ls='--', label=r'Kritische Schwelle $P=0{,}5$')
ax.axhspan(0, 0.5, alpha=0.06, color=C_GREEN)
ax.axhspan(0.5, 1.0, alpha=0.06, color=C_RED)
ax.text(0.02, 0.25, 'Kein Ausbruch', fontsize=10, color=C_GREEN, alpha=0.8)
ax.text(0.02, 0.75, 'Ausbruch', fontsize=10, color=C_RED, alpha=0.8)

ax.set_xlabel(r'Genetische Prädisposition $G$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'$P(G,S)$  –  Ausbruchswahrscheinlichkeit', fontsize=12, color=C_GRAY)
ax.set_title('Querschnitte $P(G, S_0)$ für feste Stressbelastungswerte $S_0$', fontsize=12, color=C_BLUE, pad=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)
ax.grid(True, alpha=0.25, color=C_GRAY)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_cross_sections_S.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_cross_sections_S.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Phasendiagramm (Ausbruch vs. kein Ausbruch)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=C_BG)
ax.set_facecolor(C_BG)

# Binäre Klassifikation
PP_binary = (PP >= P_CRIT).astype(float)
ax.contourf(G_arr, S_arr, PP_binary, levels=[-0.5, 0.5, 1.5],
            colors=[C_GREEN, C_RED], alpha=0.38)

# Trennlinie
cs_sep = ax.contour(G_arr, S_arr, PP, levels=[P_CRIT], colors=[C_BLUE], linewidths=2.5)
ax.clabel(cs_sep, fmt={P_CRIT: r'Trennkurve $P=0{,}5$'}, inline=True, fontsize=10, colors=C_BLUE)

# Beispielpunkte
example_pts = [
    (0.85, 0.15, 'Hohe Veranlagung,\ngeringer Stress\n→ kein Ausbruch', C_GREEN, 'v'),
    (0.15, 0.85, 'Geringe Veranlagung,\nhoher Stress\n→ Ausbruch', C_RED, '^'),
    (0.7,  0.65, 'Beide erhöht\n→ Ausbruch', C_RED, 's'),
    (0.3,  0.25, 'Beide gering\n→ kein Ausbruch', C_GREEN, 'D'),
]

for gp, sp, label, col, marker in example_pts:
    ax.scatter(gp, sp, color=col, s=90, marker=marker, zorder=10, edgecolors='white', lw=1.2)
    offsets = {'v': (-0.02, 0.06), '^': (0.03, 0.03), 's': (0.04, -0.10), 'D': (0.04, 0.04)}
    dx, dy = offsets[marker]
    ax.text(gp + dx, sp + dy, label, fontsize=8, color=col,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax.set_xlabel(r'Genetische Prädisposition $G$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'Stressbelastung $S$', fontsize=12, color=C_GRAY)
ax.set_title('Phasendiagramm: Ausbruchsregion $\Omega$ und komplementäre Region $\Omega^c$',
             fontsize=12, color=C_BLUE, pad=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=C_RED, alpha=0.5, label='Ausbruchsregion $\\Omega$'),
                Patch(facecolor=C_GREEN, alpha=0.5, label='Sichere Region $\\Omega^c$')]
ax.legend(handles=legend_elems, fontsize=10, loc='lower right', framealpha=0.85)
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)
ax.grid(True, alpha=0.2, color=C_GRAY)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_phase_diagram.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_phase_diagram.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Kompensationskurve S*(G) für verschiedene p-Zielwerte
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.set_facecolor(C_BG)

p_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors_p   = [cm.RdYlGn_r(v) for v in p_targets]
G_comp = np.linspace(0, 1, 500)

for p_t, col in zip(p_targets, colors_p):
    from scipy.special import logit
    L = logit(p_t)
    # S*(G) = (L - alpha*G + theta) / (beta + gamma*G)
    S_star_arr = (L - ALPHA * G_comp + THETA) / (BETA + GAMMA * G_comp)
    S_star_arr = np.clip(S_star_arr, 0, 1)
    ax.plot(G_comp, S_star_arr, color=col, lw=1.8, label=f'$p={p_t:.1f}$')

ax.set_xlabel(r'Genetische Prädisposition $G$', fontsize=12, color=C_GRAY)
ax.set_ylabel(r'Kritische Stressbelastung $S^*(G)$', fontsize=12, color=C_GRAY)
ax.set_title(r'Kompensationskurven $S^*(G; p)$ – Maximale Stresstoleranz', fontsize=12, color=C_BLUE, pad=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=9, loc='upper right', framealpha=0.85, ncol=2, title='Zielwahrsch.')
ax.tick_params(colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)
ax.grid(True, alpha=0.25, color=C_GRAY)

plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_compensation_curves.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_compensation_curves.pdf gespeichert.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: Sensitivitätsanalyse – Partielle Ableitungen
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=C_BG)

# dP/dG  und  dP/dS
# P = sigma(z), z = alpha*G + beta*S + gamma*G*S - theta
# P' = P*(1-P)*(dz/dG) bzw. (dz/dS)
dPdG = PP * (1 - PP) * (ALPHA + GAMMA * SS)
dPdS = PP * (1 - PP) * (BETA  + GAMMA * GG)

for ax_i, data, title, cmap_str in zip(
        axes,
        [dPdG, dPdS],
        [r'$\partial P / \partial G$ – Sensitivität bzgl. Prädisposition',
         r'$\partial P / \partial S$ – Sensitivität bzgl. Stress'],
        ['Blues', 'Oranges']):
    ax_i.set_facecolor(C_BG)
    img_i = ax_i.contourf(G_arr, S_arr, data, levels=80, cmap=cmap_str)
    cbar_i = fig.colorbar(img_i, ax=ax_i, fraction=0.046, pad=0.04)
    cbar_i.ax.tick_params(colors=C_GRAY)
    ax_i.contour(G_arr, S_arr, PP, levels=[0.5], colors=[C_BLUE], linewidths=1.8, linestyles='--')
    ax_i.set_xlabel(r'$G$', fontsize=11, color=C_GRAY)
    ax_i.set_ylabel(r'$S$', fontsize=11, color=C_GRAY)
    ax_i.set_title(title, fontsize=11, color=C_BLUE, pad=8)
    ax_i.tick_params(colors=C_GRAY)
    for spine in ax_i.spines.values():
        spine.set_color(C_GRAY)

fig.suptitle('Sensitivitätsanalyse: Partielle Ableitungen von $P(G,S)$',
             fontsize=13, color=C_BLUE, y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR + 'plot_sensitivity.pdf', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("plot_sensitivity.pdf gespeichert.")

print("\n✓ Alle 8 Plots erfolgreich gespeichert.")