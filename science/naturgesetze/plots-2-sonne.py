import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(7)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# --- Subplot 1: Solare Einstrahlung über 365 Tage ---
days = np.linspace(0, 365, 3650)
# True solar constant (consistent)
S0 = 1361.0  # W/m²
solar_output = S0 * (1 + 0.033 * np.cos(2 * np.pi * days / 365.25))  # slight elliptical orbit

# Earth receiving side (day/night + clouds)
day_night = np.maximum(0, np.sin(2 * np.pi * days))
cloud_factor = 0.5 + 0.3 * np.sin(2 * np.pi * days / 30) + 0.1 * np.random.randn(3650)
cloud_factor = np.clip(cloud_factor, 0.2, 1.0)
received = solar_output * day_night * cloud_factor

ax1.plot(days, solar_output, color='#FFB000', linewidth=2.5, label='Solare Abstrahlung $S(t)$ [konstant]', zorder=3)
ax1.fill_between(days, received, alpha=0.35, color='#1946A0', label='Irdischer Empfang $E(t)$ [variabel]')
ax1.axhline(S0, color='#B4321E', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Solarkonstante $S_0 = {S0}$ W/m²')
ax1.set_xlabel('Tag des Jahres', fontsize=11)
ax1.set_ylabel('Strahlungsintensität [W/m²]', fontsize=11)
ax1.set_title('Konsequenz der solaren Strahlung: Die Sonne hört nicht auf zu scheinen', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F5F5F8')
ax1.set_xlim(0, 365)

# Annotationen
ax1.annotate('Wolken', xy=(90, 350), xytext=(105, 600),
             arrowprops=dict(arrowstyle='->', color='#555'),
             fontsize=10, color='#1946A0')
ax1.annotate('Nacht', xy=(180, 0), xytext=(185, 500),
             arrowprops=dict(arrowstyle='->', color='#555'),
             fontsize=10, color='#444')

# --- Subplot 2: Sonnenkonstante über Milliarden Jahre ---
years_gy = np.linspace(0, 10, 1000)  # billion years
# Sun's luminosity evolution (standard solar model)
L_solar = (1 / (1 + 0.4 * (1 - years_gy/10.0)))  # normalized
ax2.plot(years_gy, L_solar, color='#FFB000', linewidth=2.5)
ax2.axvline(4.6, color='#B4321E', linestyle='--', linewidth=1.5, label='Heute ($\\approx 4{,}6$ Mrd. Jahre)')
ax2.fill_between(years_gy, L_solar, alpha=0.2, color='#FFB000')
ax2.set_xlabel('Zeit [Mrd. Jahre]', fontsize=11)
ax2.set_ylabel('Normierte Leuchtkraft $L/L_\\odot$', fontsize=11)
ax2.set_title('Solare Leuchtkraftentwicklung\n(geolog. Konsequenz)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F5F5F8')

# --- Subplot 3: Vergleich Sonnenkonstanz vs. menschliche Aktivität ---
t_24h = np.linspace(0, 24, 2400)
sun_horizon = np.maximum(0, np.sin(np.pi * (t_24h - 6) / 12))
human_activity = 0.2 + 0.6 * np.maximum(0, np.sin(np.pi * (t_24h - 7) / 13)) + \
                 0.08 * np.random.randn(2400)
human_activity = np.clip(human_activity, 0, 1)

ax3.plot(t_24h, sun_horizon, color='#FFB000', linewidth=2.5, label='Sonneneinstrahlung (normiert)')
ax3.plot(t_24h, human_activity, color='#B4321E', linewidth=1.5, alpha=0.8, label='Menschl. Aktivität (normiert)')
ax3.set_xlabel('Tageszeit [h]', fontsize=11)
ax3.set_ylabel('Normierter Wert', fontsize=11)
ax3.set_title('Tagesrhythmus:\nSonne vs. Mensch', fontsize=11, fontweight='bold')
ax3.set_xticks([0, 6, 12, 18, 24])
ax3.set_xticklabels(['0h', '6h', '12h', '18h', '24h'])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('#F5F5F8')

fig.suptitle('Abbildung 2: Die konsequente Sonne — Strahlungsphysik als Zeugin der Ordnung',
             fontsize=13, fontweight='bold', y=0.99)

plt.savefig('plot2_sonne.pdf', bbox_inches='tight', dpi=200)
plt.savefig('plot2_sonne.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 2 gespeichert.")
