import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('HIV: Replikationszyklus und Basisreproduktionszahl $R_0$',
             fontsize=14, fontweight='bold')

# ---- Left: R0 sensitivity analysis ----
ax1 = axes[0]
# R0 = beta * p / (c * (delta + mu_I))
# Vary beta (transmission), p (production), c (clearance), delta (infected cell death)

beta_range  = np.linspace(1e-9, 5e-8, 300)
p_vals      = [100, 300, 600]     # production rate
c_default   = 3
delta_def   = 0.7
mu_I_def    = 0.01
T_ss_def    = 1e6   # target cells at steady state (cells/mL)

colors_p = ['#1946a0', '#1e6432', '#b4321e']
for pv, col in zip(p_vals, colors_p):
    R0 = beta_range * pv * T_ss_def / (c_default * (delta_def + mu_I_def))
    ax1.plot(beta_range * 1e8, R0, color=col, linewidth=2.2, label=f'$p={pv}$ Kopien/Zelle/Tag')

ax1.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.9)
ax1.text(0.5, 1.15, '$R_0 = 1$ (Epidemieschwelle)', fontsize=9, color='gray')

ax1.fill_between(beta_range * 1e8, 0, 1, alpha=0.07, color='green')
ax1.fill_between(beta_range * 1e8, 1, 25, alpha=0.07, color='red')
ax1.text(0.5, 0.3, 'Aussterben\n($R_0 < 1$)', fontsize=9, color='#1e6432')
ax1.text(2.5, 18, 'Ausbreitung\n($R_0 > 1$)', fontsize=9, color='#b4321e')

# Mark typical HIV R0 range
ax1.axvspan(1.5, 3.5, alpha=0.12, color='orange', label='Typischer HIV $\\beta$-Bereich')

ax1.set_xlabel('Infektionsrate $\\beta$ ($\\times 10^{-8}$ mL/(Kopie·Tag))', fontsize=10)
ax1.set_ylabel('Basisreproduktionszahl $R_0$', fontsize=10)
ax1.set_title('$R_0$-Sensitivitätsanalyse', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 25)
ax1.set_xlim(0, 5)

# ---- Right: ART drug class timeline ----
ax2 = axes[1]
ax2.set_xlim(1985, 2030)
ax2.set_ylim(-0.5, 7.5)

drug_classes = [
    (1987, 'NRTI\n(Zidovudin, AZT)', '#1946a0', 0),
    (1995, 'PI\n(Protease-Inhibitoren)', '#1e6432', 1),
    (1996, 'NNRTI\n(Nevirapin, Efavirenz)', '#8B6914', 2),
    (2003, 'Fusionsinhibitoren\n(Enfuvirtid)', '#6a0dad', 3),
    (2007, 'Integrase-Inhibitoren\n(Raltegravir)', '#b4321e', 4),
    (2007, 'CCR5-Antagonisten\n(Maraviroc)', '#007b8a', 5),
    (2018, 'Long-acting ART\n(Cabotegravir/Rilpivirin)', '#5a5a8a', 6),
    (2022, 'Lenacapavir\n(Capsid-Inhibitor)', '#3a7a3a', 7),
]

for year, name, col, ypos in drug_classes:
    ax2.barh(ypos, 2027 - year, left=year, height=0.5,
             color=col, alpha=0.75, edgecolor='white', linewidth=0.8)
    ax2.text(year + 0.5, ypos, f'{year}', fontsize=8, va='center',
             color='white', fontweight='bold')
    ax2.text(2028, ypos, name, fontsize=8.5, va='center', color=col, fontweight='bold')

# Key events
events = [
    (1996, 'cART\n(Dreifach-Kombination)'),
    (2012, 'PrEP\n(Truvada)'),
    (2022, 'U=U\n(undetectable=untransmittable)'),
]
for yr, ev in events:
    ax2.axvline(x=yr, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.text(yr, 7.2, ev, fontsize=7.5, color='gray', ha='center', va='center',
             rotation=0, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.6, pad=0.3))

ax2.set_xlabel('Jahr', fontsize=10)
ax2.set_title('Zeitleiste: ART-Entwicklung und Wirkstoffklassen', fontsize=11, fontweight='bold')
ax2.set_yticks([])
ax2.set_xlim(1985, 2042)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('home/claude/plot6_r0_art.png', dpi=150, bbox_inches='tight')
print("Plot 6 saved.")
