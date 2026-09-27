import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(123)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.42)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# --- Subplot 1: Todesangst als Funktion des Inkonsequenzgrades ---
kappa = np.linspace(0.001, 1.0, 1000)  # Konsequenzmaß
# Fear model: Angst(kappa) = A0 * (1 - kappa)^alpha / kappa^beta
alpha_param = 0.8
beta_param = 0.3
A0 = 10.0
angst = A0 * (1.0 - kappa)**alpha_param / (kappa**beta_param + 0.001)
angst = angst / angst.max()

ax1.plot(kappa, angst, color='#B4321E', linewidth=2.5)
ax1.fill_between(kappa, angst, alpha=0.18, color='#B4321E')

# Markierungen
kappa_nl = 1.0
angst_nl = 0.0
kappa_avg_human = 0.312
angst_avg_human = np.interp(kappa_avg_human, kappa, angst)

ax1.scatter([kappa_nl], [angst_nl], s=120, color='#1946A0', zorder=5, label=f'Naturgesetz ($\\kappa=1$)')
ax1.scatter([kappa_avg_human], [angst_avg_human], s=120, color='#B4321E', zorder=5,
            label=f'Ø Mensch ($\\kappa\\approx 0.31$)')

ax1.annotate('Naturgesetz\n$\\Phi=0$', xy=(kappa_nl, angst_nl), xytext=(0.72, 0.15),
             arrowprops=dict(arrowstyle='->', color='#1946A0'), fontsize=9, color='#1946A0')
ax1.annotate(f'Ø Mensch\n$\\Phi\\approx{angst_avg_human:.2f}$', xy=(kappa_avg_human, angst_avg_human),
             xytext=(0.1, 0.65), arrowprops=dict(arrowstyle='->', color='#B4321E'), fontsize=9, color='#B4321E')

ax1.set_xlabel('Konsequenzmaß $\\kappa$', fontsize=11)
ax1.set_ylabel('Normiertes Angstmaß $\\Phi(\\kappa)$', fontsize=11)
ax1.set_title('Todesangst als Funktion\ndes Konsequenzmaßes', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F5F5F8')

# --- Subplot 2: Populationsdichte der Inkonsequenz ---
n_people = 5000
kappa_pop = np.random.beta(2.5, 5.5, n_people)  # Beta-verteilung: linksskew
angst_pop = A0 * (1.0 - kappa_pop)**alpha_param / (kappa_pop**beta_param + 0.001)
angst_pop = angst_pop / (A0 * (1.0 - 0.001)**alpha_param / (0.001**beta_param + 0.001)) 

ax2.scatter(kappa_pop, angst_pop, s=5, alpha=0.3, color='#B4321E', rasterized=True)
ax2.plot(kappa, angst, color='#1946A0', linewidth=2.0, label='Theoret. Kurve $\\Phi(\\kappa)$')
ax2.axvline(np.mean(kappa_pop), color='#333', linestyle='--', linewidth=1.5,
            label=f'Pop.-Mittel $\\bar{{\\kappa}}={np.mean(kappa_pop):.3f}$')
ax2.set_xlabel('Konsequenzmaß $\\kappa$', fontsize=11)
ax2.set_ylabel('Angstmaß $\\Phi$', fontsize=11)
ax2.set_title('Populationsverteilung\n($N=5000$ Individuen)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F5F5F8')

# --- Subplot 3: Zeitliche Dynamik der Angst über Lebensspanne ---
age = np.linspace(0, 80, 800)
# Fear increases as death approaches, modulated by inconsistency
base_awareness = 1 / (1 + np.exp(-(age - 40) / 10))  # Sigmoid: becomes aware of death
kappa_life = 0.35 + 0.25 * np.exp(-age / 30) - 0.1 * (age / 80)**2
kappa_life = np.clip(kappa_life, 0.1, 0.95)
fear_life = base_awareness * (1.5 - kappa_life)

kappa_life2 = 0.65 + 0.1 * np.exp(-age / 20) * np.sin(age / 5)
kappa_life2 = np.clip(kappa_life2, 0.4, 0.95)
fear_life2 = base_awareness * (1.5 - kappa_life2)

ax3.plot(age, fear_life, color='#B4321E', linewidth=2.2, label='Inkonsequente Person ($\\bar{\\kappa}=0.35$)')
ax3.plot(age, fear_life2, color='#1E7840', linewidth=2.2, label='Konsequente Person ($\\bar{\\kappa}=0.65$)')
ax3.fill_between(age, fear_life, fear_life2, alpha=0.15, color='#888',
                 label='Differenz durch $\\Delta\\kappa$')
ax3.set_xlabel('Lebensalter [Jahre]', fontsize=11)
ax3.set_ylabel('Todesangstmaß $\\Phi(t)$', fontsize=11)
ax3.set_title('Todesangst über die\nLebensspanne', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('#F5F5F8')

fig.suptitle('Abbildung 3: Formale Modellierung der Todesangst $\\Phi$ als Funktion des Inkonsequenzmaßes $\\kappa$',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig('plot3_todesangst.pdf', bbox_inches='tight', dpi=200)
plt.savefig('plot3_todesangst.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 3 gespeichert.")
