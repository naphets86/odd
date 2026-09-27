#!/usr/bin/env python3
"""
Plot: Kupfer-Dynamik unter Ozeanversauerung und Mikroplastik-Einfluss
Basiert auf: Poersch et al. (2026) - Water Air Soil Pollut 237:771
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Kupfer-Dynamik unter Ozeanversauerung und Mikroplastik-Einfluss\n'
    'Quelle: Poersch et al. (2026) – Water Air Soil Pollut 237:771',
    fontsize=12, fontweight='bold', y=1.01
)

# ── Subplot A: Gemessene Cu(II)-Konzentrationen ───────────────────────────────
ax = axes[0, 0]
categories = ['pH 7.88\nohne MP', 'pH 7.88\nmit MP', 'pH 7.59\nohne MP', 'pH 7.59\nmit MP']
initial = [3.936, 3.936, 2.514, 2.514]
final   = [2.359, 2.011, 3.428, 2.357]
initial_err = [1.146, 1.146, 0.395, 0.395]
final_err   = [0.858, 0.465, 1.003, 0.140]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, initial, w, yerr=initial_err, capsize=4,
            label='Initial (t=0h)', color='#3498DB', alpha=0.85)
b2 = ax.bar(x + w/2, final,   w, yerr=final_err,   capsize=4,
            label='Final (t=48h)', color='#E74C3C', alpha=0.85)
ax.set_ylabel('Gelöstes Cu(II) (μg/L)', fontsize=9)
ax.set_title('A) Gemessene Cu(II)-Konzentrationen\n(initial vs. final, Mittelwert ± SE)',
             fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 6.5)

# ── Subplot B: Differenz Initial–Final (Cu-Verfügbarkeit) ────────────────────
ax = axes[0, 1]
# From paper Figure 2 – positive = less Cu in water; negative = more Cu
diff_no_mp  = [1.577, -0.914]   # pH 7.88, pH 7.59
diff_mp     = [1.925, 0.157]
ph_labels   = ['pH 7.88\n(moderate)', 'pH 7.59\n(intensiv)']
x = np.arange(2)
w = 0.35
colors_no_mp = ['#27AE60' if d > 0 else '#E74C3C' for d in diff_no_mp]
colors_mp    = ['#27AE60' if d > 0 else '#E74C3C' for d in diff_mp]
for i, (d1, d2) in enumerate(zip(diff_no_mp, diff_mp)):
    ax.bar(i - w/2, d1, w, color='#2980B9' if d1 > 0 else '#E74C3C', alpha=0.85,
           label='Ohne MP' if i == 0 else '')
    ax.bar(i + w/2, d2, w, color='#27AE60' if d2 > 0 else '#C0392B', alpha=0.85,
           label='Mit MP' if i == 0 else '')
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Δ[Cu] = C_initial – C_final (μg/L)', fontsize=9)
ax.set_title('B) Cu(II)-Verfügbarkeitsänderung (Δ[Cu])\n(positiv = Cu adsorbiert, negativ = Cu freigesetzt)',
             fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ph_labels, fontsize=9)
ax.legend(['Ohne Mikroplastik', 'Mit Mikroplastik'], fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.annotate('Cu freigesetzt\n(Versauerung)', xy=(0.75, -0.85), fontsize=8,
            color='#C0392B', ha='center')

# ── Subplot C: Cu-Speziation bei verschiedenen pH-Werten ─────────────────────
ax = axes[1, 0]
species  = ['CuCO₃(aq)', 'CuOH⁺', 'Cu²⁺\n(frei)', 'CuHCO₃⁺', 'Sonstige']
ph788    = [90.24, 4.51, 2.60, 1.90, 0.75]
ph759    = [88.92, 4.36, 4.99, 0.98, 0.75]
x = np.arange(len(species))
w = 0.35
ax.bar(x - w/2, ph788, w, label='pH 7.88 (moderat)', color='#3498DB', alpha=0.85)
ax.bar(x + w/2, ph759, w, label='pH 7.59 (intensiv)', color='#E74C3C', alpha=0.85)
ax.set_ylabel('Anteil an gelöstem Cu(II) (%)', fontsize=9)
ax.set_title('C) Cu(II)-Speziation bei verschiedenen pH-Werten\n(Visual MINTEQ v3.1, Poersch et al. 2026)',
             fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(species, fontsize=9)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.annotate('↑ Cu²⁺ steigt\nbei Versauerung', xy=(2.3, 5.2), fontsize=8, color='red')

# ── Subplot D: Karbonat-System-Parameter ──────────────────────────────────────
ax = axes[1, 1]
param_names = ['TA\n(μmol/kg)', 'DIC\n(μmol/kg)', 'pCO₂\n(μatm)', 'CO₃²⁻\n(μmol/kg)',
               'HCO₃⁻\n(μmol/kg)']
ph788_vals = [2377, 2257, 1024, 99.8, 2133]
ph759_vals = [2409, 2396, 1731, 65.9, 2248]
# Normalise for visualization
norm_788 = np.array(ph788_vals, dtype=float)
norm_759 = np.array(ph759_vals, dtype=float)
x = np.arange(len(param_names))
w = 0.35

# Use secondary axis for pCO2 
ax.bar(x - w/2, norm_788, w, label='pH 7.88', color='#2980B9', alpha=0.85)
ax.bar(x + w/2, norm_759, w, label='pH 7.59', color='#E74C3C', alpha=0.85)
ax.set_ylabel('Konzentration (μmol/kg) / pCO₂ (μatm)', fontsize=8)
ax.set_title('D) Karbonat-Systemparameter\nbei beiden Versauerungsszenarien (CO₂SYS v3.0)',
             fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(param_names, fontsize=8)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Add delta labels
for i, (v1, v2) in enumerate(zip(ph788_vals, ph759_vals)):
    delta = ((v2 - v1) / v1) * 100
    sign = '+' if delta > 0 else ''
    ax.text(i, max(v1, v2) + 30, f'{sign}{delta:.1f}%', ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('plot_kupfer_ozeanversauerung.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Plot B gespeichert.")
