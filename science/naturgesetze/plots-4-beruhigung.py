import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(99)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# --- Subplot 1: Beruhigungsindex als Funktion der Kontemplationszeit ---
t_kontemp = np.linspace(0, 100, 1000)
# Beruhigungsindex steigt mit Betrachtung der Naturgesetze
beta_calm = 0.05
B0 = 0.1
beruhigung = 1.0 - (1.0 - B0) * np.exp(-beta_calm * t_kontemp)

# Angstreduktion
fear_initial = 0.85
fear_reduced = fear_initial * (1.0 - beruhigung * 0.85)

ax1.plot(t_kontemp, beruhigung, color='#1946A0', linewidth=2.5, label='Beruhigungsindex $\\beta(t)$')
ax1.plot(t_kontemp, fear_reduced, color='#B4321E', linewidth=2.5, linestyle='--',
         label='Angstmaß $\\Phi(t)$ (reduziert)')
ax1.fill_between(t_kontemp, beruhigung, alpha=0.15, color='#1946A0')
ax1.fill_between(t_kontemp, fear_reduced, alpha=0.15, color='#B4321E')
ax1.axhline(1.0, color='#1946A0', linestyle=':', alpha=0.5)
ax1.set_xlabel('Kontemplationszeit $\\tau$ [willk. Einh.]', fontsize=11)
ax1.set_ylabel('Normierter Wert', fontsize=11)
ax1.set_title('Beruhigungsindex $\\beta(\\tau)$ und\nAngstreduktion', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F5F5F8')

# --- Subplot 2: Phasendiagramm Angst vs. Konsequenzmaß ---
kappa_range = np.linspace(0.05, 1.0, 200)
# Stability regions
fear_high = 0.9 * (1 - kappa_range)**0.6
fear_low = 0.4 * (1 - kappa_range)**0.6

ax2.fill_between(kappa_range, fear_high, 1.0, alpha=0.25, color='#B4321E', label='Instabilitätszone (Teufel)')
ax2.fill_between(kappa_range, 0, fear_low, alpha=0.25, color='#1946A0', label='Stabilitätszone (Frieden)')
ax2.fill_between(kappa_range, fear_low, fear_high, alpha=0.2, color='#888', label='Übergangszone')
ax2.plot(kappa_range, fear_high, color='#B4321E', linewidth=2.0)
ax2.plot(kappa_range, fear_low, color='#1946A0', linewidth=2.0)

# Trajectory
traj_kappa = np.array([0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.88])
traj_fear = np.array([0.92, 0.78, 0.61, 0.45, 0.28, 0.17, 0.08])
ax2.plot(traj_kappa, traj_fear, 'ko-', markersize=6, linewidth=1.8,
         label='Trajektorie (Kontemplation)')
for i, (k, f) in enumerate(zip(traj_kappa, traj_fear)):
    ax2.annotate(f'$t_{i}$', (k, f), xytext=(k+0.02, f+0.04), fontsize=8)

ax2.set_xlabel('Konsequenzmaß $\\kappa$', fontsize=11)
ax2.set_ylabel('Angstmaß $\\Phi$', fontsize=11)
ax2.set_title('Phasendiagramm:\nStabilität durch Naturgesetz-Kontemplation', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F5F5F8')

# --- Subplot 3: Vergleich Beruhigungsquellen ---
sources = ['Naturgesetze\n(dauerhaft)', 'Musik\n(temp.)', 'Gespräch\n(temp.)', 'Schlaf\n(temp.)', 'Philosophie\n(mittel)', 'Andere\nMenschen\n(unsicher)']
durability = [1.00, 0.35, 0.28, 0.42, 0.61, 0.22]
effectiveness = [0.95, 0.62, 0.55, 0.70, 0.66, 0.38]

x_pos = np.arange(len(sources))
width = 0.35
bars1 = ax3.bar(x_pos - width/2, durability, width, label='Dauerhaftigkeit', color='#1946A0', alpha=0.85)
bars2 = ax3.bar(x_pos + width/2, effectiveness, width, label='Wirksamkeit', color='#1E7840', alpha=0.85)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(sources, fontsize=8.5)
ax3.set_ylabel('Normierter Index', fontsize=11)
ax3.set_title('Beruhigungsquellen:\nDauerhaftigkeit und Wirksamkeit', fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1.15)
for bar, val in zip(bars1, durability):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
for bar, val in zip(bars2, effectiveness):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_facecolor('#F5F5F8')

fig.suptitle('Abbildung 4: Das Beruhigungstheorem — Kontemplation der Naturgesetze als einzige dauerhafte Angstreduktion',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig('plot4_beruhigung.pdf', bbox_inches='tight', dpi=200)
plt.savefig('plot4_beruhigung.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 4 gespeichert.")
