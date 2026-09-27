import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── Plot 1: ASIL Risk Matrix ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
severity   = ['S1\n(leicht)', 'S2\n(mittel)', 'S3\n(lebensbedrohlich)']
exposure   = ['E1\n(sehr selten)', 'E2\n(selten)', 'E3\n(gelegentlich)', 'E4\n(häufig)']
# ASIL classification (rows=severity 0-2, cols=exposure 0-3)
asil = [
    ['QM', 'QM', 'QM', 'A'],
    ['QM', 'A',  'B',  'C'],
    ['A',  'B',  'C',  'D'],
]
colors = {'QM': '#d0e8d0', 'A': '#fffacd', 'B': '#ffd580', 'C': '#ffaa55', 'D': '#ff6666'}
for i, row in enumerate(asil):
    for j, val in enumerate(row):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[val], ec='white', lw=2))
        ax.text(j+0.5, i+0.5, val, ha='center', va='center', fontsize=13, fontweight='bold')
ax.set_xlim(0, 4); ax.set_ylim(0, 3)
ax.set_xticks([x+0.5 for x in range(4)]); ax.set_xticklabels(exposure)
ax.set_yticks([y+0.5 for y in range(3)]); ax.set_yticklabels(severity)
ax.set_xlabel('Exposition (E)')
ax.set_ylabel('Schwere (S) – Kontrollierbarkeit C3 angenommen')
ax.set_title('ASIL-Risikomatrix (Controllability C3)')
patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
ax.legend(handles=patches, loc='upper left', fontsize=9, title='ASIL-Stufe')
plt.tight_layout()
plt.savefig('plot_asil_matrix.pdf')
plt.close()
print("Plot 1 done")

# ── Plot 2: V-Model Lifecycle ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
left_labels  = ['Konzeptphase\n(Item Definition)',
                 'Systementwicklung\n(System Level)',
                 'Hardware-Entwicklung',
                 'Software-Entwicklung',
                 'Implementierung\n(Coding)']
right_labels = ['ASIL-Bewertung\n(Safety Case)',
                'Item-Integration\n& Test',
                'HW-Integration\n& Test',
                'SW-Integration\n& Test']
n_left  = len(left_labels)
n_right = len(right_labels)
lx = [0.5]*n_left;  ly = list(range(n_left, 0, -1))
rx = [3.5]*n_right; ry = list(range(n_right+1, 1, -1))
for i, (x, y, lbl) in enumerate(zip(lx, ly, left_labels)):
    ax.plot(x, y, 's', color='#1946a0', ms=14)
    ax.text(x-0.08, y, lbl, ha='right', va='center', fontsize=8)
for i, (x, y, lbl) in enumerate(zip(rx, ry, right_labels)):
    ax.plot(x, y, 's', color='#c03010', ms=14)
    ax.text(x+0.08, y, lbl, ha='left', va='center', fontsize=8)
# diagonal lines
for i in range(min(n_left, n_right)):
    ax.plot([lx[i], rx[i]], [ly[i], ry[i]], '--', color='gray', lw=1)
# horizontal bottom
ax.plot([lx[-1], 2.0], [ly[-1], ly[-1]], '-', color='#1946a0', lw=2)
ax.plot([2.0, rx[-1]], [ry[-1], ry[-1]], '-', color='#c03010', lw=2)
ax.plot(2.0, ly[-1], 'o', color='green', ms=14)
ax.text(2.0, ly[-1]+0.1, 'Implementierung', ha='center', va='bottom', fontsize=8)
ax.set_xlim(0, 4); ax.set_ylim(0.5, n_left+0.7)
ax.axis('off')
ax.set_title('ISO 26262 V-Modell Entwicklungsprozess', fontsize=13)
ax.annotate('Dekomposition', xy=(0.5, n_left), xytext=(0.5, n_left+0.4),
            ha='center', fontsize=9, color='#1946a0',
            arrowprops=dict(arrowstyle='->', color='#1946a0'))
ax.annotate('Integration & Verifikation', xy=(3.5, n_right+1), xytext=(3.5, n_right+1.4),
            ha='center', fontsize=9, color='#c03010',
            arrowprops=dict(arrowstyle='->', color='#c03010'))
plt.tight_layout()
plt.savefig('plot_v_model.pdf')
plt.close()
print("Plot 2 done")

# ── Plot 3: Failure Rate vs. ASIL ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
asil_levels = ['QM', 'ASIL A', 'ASIL B', 'ASIL C', 'ASIL D']
# Max tolerable random hardware failure rate (FIT = failures per 10^9 hours)
fit_max  = [float('inf'), 1000, 100, 10, 1]
fit_vals = [10000, 1000, 100, 10, 1]
colors2  = ['#d0e8d0', '#fffacd', '#ffd580', '#ffaa55', '#ff6666']
bars = ax.bar(asil_levels, fit_vals, color=colors2, edgecolor='black', linewidth=1.2)
ax.set_yscale('log')
ax.set_ylabel('Max. Ausfallrate [FIT = 10$^{-9}$/h]')
ax.set_xlabel('ASIL-Stufe')
ax.set_title('Maximale zufällige Hardwareausfallrate je ASIL-Stufe')
for bar, val in zip(bars, fit_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val*1.4, f'{val} FIT',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim(0.1, 1e5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plot_failure_rate.pdf')
plt.close()
print("Plot 3 done")

# ── Plot 4: Safety Mechanism Coverage ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
mechanisms = ['Watchdog\nTimer', 'CRC\nPrüfsumme', 'ECC\nSpeicher', 'Redundante\nKanäle',
              'Plausibilitäts-\nprüfung', 'FMEA/\nFTA']
dc_values = [85, 90, 99, 99, 75, 95]  # Diagnostic Coverage in %
colors3 = ['#4472c4']*6
bars = ax.barh(mechanisms, dc_values, color=colors3, edgecolor='white', linewidth=0.8)
ax.set_xlabel('Diagnoseabdeckung (DC) [%]')
ax.set_title('Diagnoseabdeckung typischer Sicherheitsmechanismen')
ax.set_xlim(0, 110)
for bar, val in zip(bars, dc_values):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val}%',
            va='center', fontsize=10, fontweight='bold')
ax.axvline(60, color='orange', linestyle='--', label='Min. ASIL A/B (60%)')
ax.axvline(90, color='red', linestyle='--', label='Min. ASIL C/D (90%)')
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('plot_dc_coverage.pdf')
plt.close()
print("Plot 4 done")

# ── Plot 5: Development Cost vs. ASIL Level ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
asil_x  = [0, 1, 2, 3, 4]
rel_cost = [1.0, 1.5, 2.5, 5.0, 10.0]
ax.plot(asil_x, rel_cost, 'o-', color='#c03010', lw=2.5, ms=10, markerfacecolor='white',
        markeredgewidth=2)
ax.fill_between(asil_x, rel_cost, alpha=0.15, color='#c03010')
ax.set_xticks(asil_x)
ax.set_xticklabels(['QM', 'ASIL A', 'ASIL B', 'ASIL C', 'ASIL D'])
ax.set_ylabel('Relativer Entwicklungsaufwand')
ax.set_xlabel('ASIL-Stufe')
ax.set_title('Entwicklungsaufwand in Abhängigkeit der ASIL-Stufe')
for x, y in zip(asil_x, rel_cost):
    ax.annotate(f'{y}×', (x, y), textcoords='offset points',
                xytext=(5, 8), fontsize=10, color='#c03010', fontweight='bold')
ax.grid(linestyle='--', alpha=0.5)
ax.set_ylim(0, 12)
plt.tight_layout()
plt.savefig('plot_cost_asil.pdf')
plt.close()
print("Plot 5 done")

print("All plots generated successfully.")
