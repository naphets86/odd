import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm

# ── Figure 1: Farbverlauf-Balken ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))
colors = [
    "#5C3A1E",   # S1 Dunkelbraun
    "#8B5E3C",   # S2 Mittelbraun
    "#B8864E",   # S3 Hellbraun/Gelbbraun
    "#D4A96A",   # S4 Cremebraun
    "#E8D5A3",   # S5 Cremeweiß/Gelblichweiß
    "#F0EAD6",   # S6 Weißlichgrau
    "#FAFAF8",   # S7 Reinweiß
]
labels = [
    "S1\nDunkelbraun",
    "S2\nMittelbraun",
    "S3\nHellbraun",
    "S4\nCremebraun",
    "S5\nCremeweiß",
    "S6\nWeißlichgrau",
    "S7\nReinweiß",
]
for i, (c, l) in enumerate(zip(colors, labels)):
    rect = mpatches.FancyBboxPatch((i*1.5, 0.2), 1.2, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor='#444444', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(i*1.5 + 0.6, 0.1, l, ha='center', va='top', fontsize=8)

ax.set_xlim(-0.2, 10.7)
ax.set_ylim(-0.15, 1.0)
ax.axis('off')
ax.set_title('Abbildung 1: Sieben Fellfarb-Stufen S1–S7 (Braunbär → Eisbär)',
             fontsize=11, pad=8)
plt.tight_layout()
plt.savefig(' (1.0 + np.exp((t - 240000) / 60000))
ax.plot(t/1000, p, color='#5C3A1E', linewidth=2.5, label=r'$p(t)$ — Pigmentindex')
# Markierungen für S1–S7
stage_t = [0, 60000, 150000, 240000, 330000, 420000, 470000]
stage_p = [1.0/(1.0+np.exp((s-240000)/60000)) for s in stage_t]
stage_colors = ["#5C3A1E","#8B5E3C","#B8864E","#D4A96A","#E8D5A3","#F0EAD6","#FAFAF8"]
for i,(st,sp,sc) in enumerate(zip(stage_t,stage_p,stage_colors)):
    ax.scatter(st/1000, sp, color=sc, edgecolor='#333333', s=100, zorder=5)
    ax.text(st/1000, sp+0.04, f'S{i+1}', ha='center', fontsize=8)

ax.set_xlabel('Zeit (in Tausend Jahren vor heute)', fontsize=10)
ax.set_ylabel('Pigmentindex $p(t) \in [0,1]$', fontsize=10)
ax.set_title('Abbildung 2: Logistischer Rückgang des Melanin-Pigmentindex', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('home/claude/fig3_fitness.pdf', bbox_inches='tight')
plt.close()
print("fig3 done")

# ── Figure 4: Mutations-Akkumulation (Galton-Watson-artig) ────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
gens = np.arange(0, 3001, 50)
# Erwartete fixierte Allele: n_fix(g) ~ g * mu * N_e (vereinfacht linear mit Sättigung)
mu = 1e-5
Ne = 1000
n_fix = Ne * (1 - np.exp(-mu * gens))
ax.plot(gens, n_fix, color='darkgreen', linewidth=2.5)
ax.set_xlabel('Generation', fontsize=10)
ax.set_ylabel('Akkumulierte fixierte Allele', fontsize=10)
ax.set_title('Abbildung 4: Fixierung vorteilhafter Allele nach Wright-Fisher-Modell', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/fig4_allele.pdf', bbox_inches='tight')
plt.close()
print("fig4 done")

print("Alle Figuren generiert.")
