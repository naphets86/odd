import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Front radar: 77 GHz FMCW, ±30° (FOV 60°), range 150m
theta_front = np.linspace(-np.pi/6, np.pi/6, 300)
r_front = np.ones(300) * 150
ax.fill_between(theta_front, 0, r_front, alpha=0.35, color='#193c8c', label='Front-Radar 77 GHz (±30°, 150 m)')
ax.plot(theta_front, r_front, color='#193c8c', linewidth=2)
ax.plot([0, -np.pi/6], [0, 150], color='#193c8c', linewidth=1.5, linestyle='--')
ax.plot([0, np.pi/6], [0, 150], color='#193c8c', linewidth=1.5, linestyle='--')

# Front radar short range (wide): ±60°, range 30m
theta_front_wide = np.linspace(-np.pi/3, np.pi/3, 300)
r_front_wide = np.ones(300) * 30
ax.fill_between(theta_front_wide, 0, r_front_wide, alpha=0.25, color='#3a9ad9', label='Front-Radar Nahbereich (±60°, 30 m)')
ax.plot(theta_front_wide, r_front_wide, color='#3a9ad9', linewidth=2)

# Rear radar: ±75° (FOV 150°), range 50m, pointing backwards (pi direction)
theta_rear = np.linspace(np.pi - np.pi*75/180, np.pi + np.pi*75/180, 300)
r_rear = np.ones(300) * 50
ax.fill_between(theta_rear, 0, r_rear, alpha=0.35, color='#b4321e', label='Rear-Radar 77 GHz (±75°, 50 m)')
ax.plot(theta_rear, r_rear, color='#b4321e', linewidth=2)
ax.plot([0, np.pi - np.pi*75/180], [0, 50], color='#b4321e', linewidth=1.5, linestyle='--')
ax.plot([0, np.pi + np.pi*75/180], [0, 50], color='#b4321e', linewidth=1.5, linestyle='--')

# Vehicle representation (center)
ax.plot(0, 0, 's', color='#1e641e', markersize=14, zorder=5, label='Fahrzeug')

# Annotate AEB zone
ax.annotate('AEB-Zone\n(Notbremsung)', xy=(0, 75), xytext=(np.pi/4, 110),
            fontsize=9, color='#193c8c',
            arrowprops=dict(arrowstyle='->', color='#193c8c', lw=1.2),
            ha='center')

# Annotate rear zone
ax.annotate('Rückfahrassistent\nTotwinkel', xy=(np.pi, 30), xytext=(np.pi - np.pi/3, 80),
            fontsize=9, color='#b4321e',
            arrowprops=dict(arrowstyle='->', color='#b4321e', lw=1.2),
            ha='center')

# Range circles
for r_val, label in [(30, '30 m'), (50, '50 m'), (100, '100 m'), (150, '150 m')]:
    ax.plot(np.linspace(0, 2*np.pi, 360), [r_val]*360, 'k-', alpha=0.12, linewidth=0.8)
    ax.text(np.pi/2 * 0.35, r_val + 4, label, fontsize=7.5, color='gray', ha='center')

ax.set_rmax(165)
ax.set_rticks([])
ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
ax.set_xticklabels(['Vorne\n(0°)', '45°', 'Rechts\n(90°)', '135°', 'Hinten\n(180°)', '225°', 'Links\n(270°)', '315°'], fontsize=9)
ax.grid(True, alpha=0.2)

ax.set_title('Abbildung 1: Radarabdeckung des Zwei-Sensor-Systems\n'
             'Front-Radar (AEB, ISA) und Rear-Radar (Rückfahrassistent, Totwinkel)',
             fontsize=11, fontweight='bold', pad=20)

legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9,
                   framealpha=0.9, edgecolor='#193c8c')

plt.tight_layout()
plt.savefig('home/claude/plot1_radar_coverage.png', format='png', bbox_inches='tight', dpi=150)
print("Plot 1 saved")
