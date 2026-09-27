import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

np.random.seed(42)

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# --- Subplot 1: Erdanziehungskraft (konsequent) ---
t = np.linspace(0, 10, 500)
g_const = 9.81 * np.ones_like(t)
ax1.plot(t, g_const, color='#1946A0', linewidth=2.5, label='Erdanziehungskraft $g(t)$')
ax1.fill_between(t, 9.80, 9.82, alpha=0.15, color='#1946A0')
ax1.set_xlabel('Zeit $t$ [s]', fontsize=11)
ax1.set_ylabel('$g$ [m/s²]', fontsize=11)
ax1.set_title('Erdanziehungskraft: Konsequenz $\\kappa = 1{,}000$', fontsize=11, fontweight='bold')
ax1.set_ylim(9.75, 9.87)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F5F5F8')

# --- Subplot 2: Menschliches Verhalten (inkonsequent) ---
behavior = 5.0 + np.cumsum(np.random.normal(0, 0.25, 500))
behavior = np.clip(behavior, 0, 10)
ax2.plot(t, behavior, color='#B4321E', linewidth=1.8, alpha=0.9, label='Menschliches Verhalten $b(t)$')
ax2.axhline(np.mean(behavior), color='#666', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mittelwert $\\bar{{b}} = {np.mean(behavior):.2f}$')
ax2.fill_between(t, behavior - np.std(behavior), behavior + np.std(behavior), alpha=0.15, color='#B4321E')
ax2.set_xlabel('Zeit $t$ [willk. Einh.]', fontsize=11)
ax2.set_ylabel('Verhaltenswert $b(t)$', fontsize=11)
kappa_h = 1.0 / (1.0 + np.std(behavior))
ax2.set_title(f'Menschliches Verhalten: Konsequenz $\\kappa \\approx {kappa_h:.3f}$', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F5F5F8')

# --- Subplot 3: Konsequenzmaß Vergleich (Balken) ---
categories = ['Erdanziehung\n$g$', 'Lichtgeschwind.\n$c$', 'Plancksches\n$h$', 'Elektronspin\n$s$', 'Mensch\n(Ø)', 'Mensch\n(max)']
kappa_vals = [1.000, 1.000, 1.000, 1.000, 0.312, 0.601]
colors_bar = ['#1946A0']*4 + ['#B4321E', '#E07060']
bars = ax3.bar(categories, kappa_vals, color=colors_bar, edgecolor='white', linewidth=0.8, width=0.6)
ax3.axhline(1.0, color='#1946A0', linestyle='--', linewidth=1.5, alpha=0.5, label='Maximale Konsequenz')
ax3.set_ylabel('Konsequenzmaß $\\kappa$', fontsize=11)
ax3.set_title('Konsequenzmaß $\\kappa$ im Vergleich', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 1.12)
for bar, val in zip(bars, kappa_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_facecolor('#F5F5F8')

# --- Subplot 4: Verteilung der Inkonsequenz ---
x = np.linspace(-4, 4, 400)
sigma_nl = 0.02
sigma_h = 1.0
pdf_nl = norm.pdf(x, 0, sigma_nl)
pdf_h = norm.pdf(x, 0, sigma_h)
ax4.plot(x, pdf_nl / pdf_nl.max(), color='#1946A0', linewidth=2.5, label=f'Naturgesetz ($\\sigma={sigma_nl}$)')
ax4.plot(x, pdf_h / pdf_h.max(), color='#B4321E', linewidth=2.5, label=f'Mensch ($\\sigma={sigma_h}$)')
ax4.fill_between(x, pdf_nl / pdf_nl.max(), alpha=0.2, color='#1946A0')
ax4.fill_between(x, pdf_h / pdf_h.max(), alpha=0.15, color='#B4321E')
ax4.set_xlabel('Abweichung vom Sollwert $\\Delta$', fontsize=11)
ax4.set_ylabel('Normierte Wahrscheinlichkeitsdichte', fontsize=11)
ax4.set_title('Dichteverteilung der Verhaltensabweichung', fontsize=11, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_facecolor('#F5F5F8')

fig.suptitle('Abbildung 1: Formaler Konsequenzvergleich — Naturgesetze vs. Menschliches Verhalten',
             fontsize=13, fontweight='bold', y=0.98)

plt.savefig('plot1_konsequenz.pdf', bbox_inches='tight', dpi=200)
plt.savefig('plot1_konsequenz.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 1 gespeichert.")
