import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(2024)

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

# --- Subplot 1: Der Teufel als divergente Geisteskraft ---
t = np.linspace(0, 20, 5000)
# Devil: maximally contradictory - alternates, diverges, never settles
devil_func = np.sin(5 * t) * np.cos(7 * t + 1.3) * (1 + 0.3 * t) + \
             0.2 * np.sin(23 * t) + 0.4 * np.cumsum(np.random.randn(5000)) / 100
# Normalize to show chaos
devil_func = devil_func / np.max(np.abs(devil_func))

# Natural law: constant
natural = np.ones_like(t) * 0.8

ax1.plot(t, devil_func, color='#8B0000', linewidth=1.5, alpha=0.85,
         label='Teufel: $D(t)$ — maximal widersprüchlich')
ax1.plot(t, natural, color='#1946A0', linewidth=2.5, label='Naturgesetz: $N(t) = \\text{const}$')
ax1.axhline(0, color='#666', linestyle=':', linewidth=1.0)
ax1.fill_between(t, devil_func, alpha=0.12, color='#8B0000')
ax1.set_xlabel('Zeit $t$', fontsize=11)
ax1.set_ylabel('Wirkungsstärke', fontsize=11)
ax1.set_title('Der Teufel als maximal widersprüchliche Geisteskraft im Vergleich zum Naturgesetz', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F5F5F8')

# Annotation
ax1.annotate('Kein Grenzwert:\n$\\lim_{t\\to\\infty} D(t)$ existiert nicht',
             xy=(17, devil_func[-300]), xytext=(13, 0.5),
             arrowprops=dict(arrowstyle='->', color='#8B0000'),
             fontsize=9, color='#8B0000')

# --- Subplot 2: Spektralanalyse des Teufels ---
from numpy.fft import fft, fftfreq
N = 5000
yf = np.abs(fft(devil_func))[:N//2]
xf = fftfreq(N, 20.0/N)[:N//2]
ax2.semilogy(xf[1:200], yf[1:200], color='#8B0000', linewidth=1.5, alpha=0.8)
ax2.semilogy([0, xf[199]], [1, 1], color='#1946A0', linewidth=2.0, linestyle='--',
             label='Naturgesetz (DC)')
ax2.set_xlabel('Frequenz [Hz]', fontsize=11)
ax2.set_ylabel('Amplitude (log)', fontsize=11)
ax2.set_title('Spektrum: Teufel\n(breitbandig) vs. Naturgesetz (DC)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F5F5F8')

# --- Subplot 3: Konsequenz-Inkonsequenz Zyklus (Hölle) ---
theta = np.linspace(0, 4 * np.pi, 500)
# Spiral towards Hölle (increasing fear, decreasing consistency)
r_hell = 1.0 + 0.3 * theta
x_hell = r_hell * np.cos(theta)
y_hell = r_hell * np.sin(theta)

# Spiral towards Frieden (converging to natural law)
r_peace = 3.0 * np.exp(-0.15 * theta)
x_peace = r_peace * np.cos(theta) + 5
y_peace = r_peace * np.sin(theta)

ax3.plot(x_hell, y_hell, color='#B4321E', linewidth=2.0, label='Spirale: Hölle')
ax3.plot(x_peace, y_peace, color='#1E7840', linewidth=2.0, label='Spirale: Frieden')
ax3.scatter([0], [0], s=100, color='#B4321E', zorder=5)
ax3.scatter([5], [0], s=100, color='#1E7840', zorder=5)
ax3.annotate('Hölle\n(Divergenz)', xy=(0, 0), xytext=(1.5, -4),
             arrowprops=dict(arrowstyle='->', color='#B4321E'), fontsize=9, color='#B4321E')
ax3.annotate('Frieden\n(Konvergenz)', xy=(5, 0), xytext=(6.5, 3),
             arrowprops=dict(arrowstyle='->', color='#1E7840'), fontsize=9, color='#1E7840')
ax3.set_aspect('equal')
ax3.set_title('Trajektorien:\nHölle (Div.) vs. Frieden (Konv.)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('#F5F5F8')

# --- Subplot 4: Konsequenzmaß verschiedener Entitäten ---
entities = ['Teufel', 'Mensch\n(min)', 'Ø Mensch', 'Mensch\n(max)', 'Sonne', 'Gravitation', 'Licht-\ngeschw.']
kappa_e = [0.001, 0.05, 0.312, 0.72, 0.998, 1.000, 1.000]
colors_e = ['#8B0000', '#C44', '#B4321E', '#E08060', '#FFB000', '#1946A0', '#1946A0']

bars = ax4.barh(entities, kappa_e, color=colors_e, edgecolor='white', linewidth=0.8)
ax4.axvline(1.0, color='#1946A0', linestyle='--', linewidth=1.5, alpha=0.6)
for bar, val in zip(bars, kappa_e):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
             va='center', fontsize=9, fontweight='bold')
ax4.set_xlabel('Konsequenzmaß $\\kappa$', fontsize=11)
ax4.set_title('$\\kappa$-Ranking aller\nbetrachteten Entitäten', fontsize=11, fontweight='bold')
ax4.set_xlim(0, 1.12)
ax4.grid(True, axis='x', alpha=0.3)
ax4.set_facecolor('#F5F5F8')

# --- Subplot 5: Hölle als konsequente Antwort ---
inkonsequenz = np.linspace(0, 1, 200)
# Hell probability = consequence of inconsistency
hoelle_prob = inkonsequenz**0.7
frieden_prob = (1 - inkonsequenz)**0.7
norm_factor = hoelle_prob + frieden_prob
hoelle_n = hoelle_prob / norm_factor
frieden_n = frieden_prob / norm_factor

ax5.fill_between(inkonsequenz, hoelle_n, alpha=0.4, color='#B4321E', label='Hölle (konsequente Folge)')
ax5.fill_between(inkonsequenz, frieden_n, alpha=0.4, color='#1946A0', label='Frieden')
ax5.plot(inkonsequenz, hoelle_n, color='#B4321E', linewidth=2.0)
ax5.plot(inkonsequenz, frieden_n, color='#1946A0', linewidth=2.0)
ax5.axvline(0.312, color='#888', linestyle='--', linewidth=1.5,
            label=f'Ø Mensch ($\\bar{{\\kappa}}\\approx 0.31$)')
ax5.set_xlabel('Inkonsequenzgrad $1 - \\kappa$', fontsize=11)
ax5.set_ylabel('Normierter Anteil', fontsize=11)
ax5.set_title('Konsequente Folge:\nHölle vs. Frieden', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_facecolor('#F5F5F8')

fig.suptitle('Abbildung 5: Der Teufel als maximal widersprüchliche Kraft — Formale Analyse der Konsequenz (Hölle) inkonsequenten Handelns',
             fontsize=12, fontweight='bold', y=0.99)

plt.savefig('plot5_teufel.pdf', bbox_inches='tight', dpi=200)
plt.savefig('plot5_teufel.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 5 gespeichert.")
