"""
Plot 3: Kontinuitätsgleichung – Divergenzfeld des atmosphärischen
        Wasserdampftransports und lokale Knotengleichungen
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MultipleLocator

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

# ---- links: Vektorfeld + Divergenz ----
ax = axes[0]
x = np.linspace(-3, 3, 22)
y = np.linspace(-3, 3, 22)
X, Y = np.meshgrid(x, y)

# Simulierter Feuchtefluss: Quellen (Ozean) links, Senken (Land) rechts
Fx = 0.6 * np.tanh(X) + 0.4 * np.sin(0.7 * Y)
Fy = 0.3 * np.cos(0.5 * X) - 0.2 * np.tanh(Y)

# Numerische Divergenz
dFx_dx = np.gradient(Fx, x, axis=1)
dFy_dy = np.gradient(Fy, y, axis=0)
div = dFx_dx + dFy_dy

cont = ax.contourf(X, Y, div, levels=20, cmap='RdBu_r', alpha=0.85)
cbar = plt.colorbar(cont, ax=ax, shrink=0.82)
cbar.set_label('Divergenz $\\nabla \\cdot (\\rho_w \\mathbf{v})$\n[kg m$^{-2}$ s$^{-1}$]',
               fontsize=9)

skip = 2
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
          Fx[::skip, ::skip], Fy[::skip, ::skip],
          color='#19468C', alpha=0.75, scale=8, width=0.004)

ax.set_title('Divergenzfeld des Feuchteflusses\n$\\nabla \\cdot (\\rho_w \\mathbf{v})$ > 0: Quelle | < 0: Senke',
             fontsize=11, fontweight='bold', color='#19468C')
ax.set_xlabel('Longitude [normiert]', fontsize=10)
ax.set_ylabel('Latitude [normiert]', fontsize=10)
ax.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.5)
ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5)

# Knotenmarkierungen
for (cx, cy, lbl) in [(-1.8, 1.5, 'Knoten A\n(Ozean)'),
                       (1.8, -1.5, 'Knoten B\n(Land)')]:
    ax.plot(cx, cy, 'o', ms=10, color='#B4321E', zorder=5)
    ax.text(cx, cy + 0.35, lbl, ha='center', fontsize=8.5,
            color='#B4321E', fontweight='bold')

# ---- rechts: Knotengleichung pro Gitterzelle ----
ax2 = axes[1]

# Zeige E - P = div(qv) für 12 Beispielgitterzellen
np.random.seed(42)
n_cells = 12
labels = [f'Zelle {i+1}' for i in range(n_cells)]
E = np.random.uniform(1.5, 4.5, n_cells)
transport = np.random.uniform(-1.0, 1.0, n_cells)
P = E + transport                                  # P = E + div(qv)

x_pos = np.arange(n_cells)
w = 0.28
b1 = ax2.bar(x_pos - w, E,       w, color='#B4321E', alpha=0.85, label='Evaporation $E$')
b2 = ax2.bar(x_pos,     P,       w, color='#19468C', alpha=0.85, label='Niederschlag $P$')
b3 = ax2.bar(x_pos + w, E - P,   w, color='#5B2D8E', alpha=0.85, label='$E - P$ (Netto)')

ax2.axhline(0, color='#333', lw=1.2)
ax2.set_title('Lokale Knotenbilanz pro Gitterzelle\n$E_i - P_i = \\nabla \\cdot (q\\mathbf{v})_i$',
              fontsize=11, fontweight='bold', color='#19468C')
ax2.set_xlabel('Gitterzelle', fontsize=10)
ax2.set_ylabel('[mm/d]', fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{i+1}' for i in range(n_cells)], fontsize=8)
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, ls=':')

# Globale Summe
global_net = np.sum(E - P)
ax2.text(0.98, 0.97,
         f'Globale Summe $\\sum(E-P)$:\n{global_net:.2f} mm/d $\\approx 0$',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         color='#5B2D8E',
         bbox=dict(boxstyle='round,pad=0.4', fc='#F0EBF8', ec='#5B2D8E'))

plt.tight_layout(pad=2.0)
plt.savefig('plot3_kontinuitaet.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot 3 gespeichert.")
