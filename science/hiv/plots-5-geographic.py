import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

regions = [
    'Östliches &\nSüdliches Afrika',
    'Westliches &\nZentralafrka',
    'Westeuropa\n& Nordamerika',
    'Asien-Pazifik',
    'Lateinamerika',
    'Osteuropa &\nZentralasien',
    'Mittlerer\nOsten & N.-Afrika',
    'Karibik',
]

# People living with HIV (millions, 2023 estimates, UNAIDS)
plhiv     = np.array([20.8, 4.8, 2.3, 6.5, 2.2, 1.7, 0.37, 0.33])
# ART coverage (%)
art_cov   = np.array([82, 71, 87, 60, 73, 50, 28, 71])
# New infections (thousands, 2023)
new_inf   = np.array([530, 170, 37, 300, 100, 120, 20, 18])
# Viral suppression among those on ART (%)
vs        = np.array([90, 85, 92, 80, 83, 72, 68, 83])

x = np.arange(len(regions))
width = 0.35

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Globale HIV-Last: Regionale Verteilung (2023)\nQuelle: UNAIDS World AIDS Day Report 2023',
             fontsize=13, fontweight='bold')

# Plot 1: PLHIV
ax1 = axes[0, 0]
bars = ax1.bar(x, plhiv, color='#1946a0', alpha=0.82, edgecolor='white', linewidth=0.8)
ax1.set_title('Menschen mit HIV (Millionen)', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regions, rotation=30, ha='right', fontsize=7.5)
ax1.set_ylabel('Millionen', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, plhiv):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: ART coverage
ax2 = axes[0, 1]
colors_art = ['#1e6432' if c >= 80 else '#e07b00' if c >= 60 else '#b4321e' for c in art_cov]
bars2 = ax2.bar(x, art_cov, color=colors_art, alpha=0.82, edgecolor='white', linewidth=0.8)
ax2.axhline(y=95, color='#1946a0', linestyle='--', linewidth=2, alpha=0.8, label='95% UNAIDS-Ziel')
ax2.set_title('ART-Abdeckung (%)', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(regions, rotation=30, ha='right', fontsize=7.5)
ax2.set_ylabel('Prozent (%)', fontsize=10)
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8.5)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, art_cov):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 3: New infections
ax3 = axes[1, 0]
bars3 = ax3.bar(x, new_inf, color='#b4321e', alpha=0.82, edgecolor='white', linewidth=0.8)
ax3.set_title('Neuinfektionen 2023 (Tausend)', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(regions, rotation=30, ha='right', fontsize=7.5)
ax3.set_ylabel('Tausend', fontsize=10)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, new_inf):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}k', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 4: Viral suppression
ax4 = axes[1, 1]
colors_vs = ['#1e6432' if v >= 90 else '#e07b00' if v >= 75 else '#b4321e' for v in vs]
bars4 = ax4.bar(x, vs, color=colors_vs, alpha=0.82, edgecolor='white', linewidth=0.8)
ax4.axhline(y=95, color='#1946a0', linestyle='--', linewidth=2, alpha=0.8, label='95%-Ziel')
ax4.set_title('Virussuppression unter ART (%)', fontsize=11, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(regions, rotation=30, ha='right', fontsize=7.5)
ax4.set_ylabel('Prozent (%)', fontsize=10)
ax4.set_ylim(0, 105)
ax4.legend(fontsize=8.5)
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars4, vs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('home/claude/plot5_geographic.png', dpi=150, bbox_inches='tight')
print("Plot 5 saved.")
