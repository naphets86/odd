import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# --- Left: Braking distance vs speed ---
ax = axes[0]
v_kmh = np.linspace(10, 130, 300)
v_ms = v_kmh / 3.6

# Reaction time human: 1.0s, AEB: 0.15s
t_human = 1.0
t_aeb = 0.15
mu = 0.75  # friction coefficient dry road
g = 9.81
a_brake = mu * g  # ~7.36 m/s^2

d_human_react = v_ms * t_human
d_aeb_react   = v_ms * t_aeb
d_brake       = v_ms**2 / (2 * a_brake)

d_total_human = d_human_react + d_brake
d_total_aeb   = d_aeb_react   + d_brake

ax.fill_between(v_kmh, d_total_aeb, d_total_human,
                alpha=0.3, color='#b4321e', label='Einsparung durch AEB')
ax.plot(v_kmh, d_total_human, color='#b4321e', linewidth=2.5, label='Mensch (Reaktion 1,0 s)')
ax.plot(v_kmh, d_total_aeb,   color='#193c8c', linewidth=2.5, label='AEB (Reaktion 0,15 s)')
ax.plot(v_kmh, d_brake,       color='#1e641e', linewidth=1.8, linestyle='--', label='Physikalischer Bremsweg')

# Annotate key speeds
for v_ref, color in [(30, 'gray'), (50, 'darkorange'), (100, 'purple')]:
    v_ms_ref = v_ref / 3.6
    d_h = v_ms_ref * t_human + v_ms_ref**2 / (2*a_brake)
    d_a = v_ms_ref * t_aeb   + v_ms_ref**2 / (2*a_brake)
    ax.annotate(f'{v_ref} km/h\nΔ={d_h-d_a:.1f} m',
                xy=(v_ref, (d_h+d_a)/2), fontsize=7.5,
                color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec=color))

ax.set_xlabel('Geschwindigkeit [km/h]', fontsize=11)
ax.set_ylabel('Anhalteweg [m]', fontsize=11)
ax.set_title('(a) Anhalteweg: Mensch vs. AEB\nbei μ = 0,75 (Trockenasphalt)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(10, 130)
ax.set_ylim(0, 140)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xticks(np.arange(10, 131, 20))

# --- Right: TTC (Time-to-Collision) and AEB trigger ---
ax2 = axes[1]
t_vals = np.linspace(0, 3.5, 500)

# Scenario: ego at 50 km/h, obstacle stationary
v_ego = 50 / 3.6
v_obj = 0
d0 = 45  # initial distance

dist = np.maximum(d0 - v_ego * t_vals, 0)
ttc = np.where(dist > 0.1, dist / v_ego, 0)

# AEB trigger at TTC=1.5s, full brake
ttc_threshold = 1.5
t_trigger = (d0 - v_ego * ttc_threshold) / v_ego
t_trigger_idx = np.argmin(np.abs(t_vals - t_trigger))

# Braking phase
a_b = -7.5  # m/s^2
t_brake = t_vals[t_trigger_idx:]
v_brake = np.maximum(v_ego + a_b*(t_brake - t_trigger), 0)
# position after trigger
pos_trigger = dist[t_trigger_idx]
pos_brake = pos_trigger + v_ego*(t_brake - t_trigger) + 0.5*a_b*(t_brake - t_trigger)**2
pos_brake = np.maximum(pos_brake, 0)

# Rebuild distance array
dist_full = dist.copy()
dist_full[t_trigger_idx:] = pos_brake[:len(dist_full)-t_trigger_idx]
dist_full = np.maximum(dist_full, 0)

# Color zones
ax2.axhspan(0, 1.5, alpha=0.15, color='#b4321e', label='Kollisionsgefahr (TTC < 1,5 s)')
ax2.axhspan(1.5, 2.5, alpha=0.10, color='darkorange', label='Vorwarnung (1,5–2,5 s)')
ax2.axhspan(2.5, 3.5, alpha=0.08, color='#1e641e', label='Sicherer Bereich (> 2,5 s)')

ttc_plot = np.where(dist_full > 0.1, dist_full / v_ego, 0)
ax2.plot(t_vals, ttc_plot, color='#193c8c', linewidth=2.5, label='TTC mit AEB-Eingriff')

# Without AEB
ttc_no_aeb = np.where(dist > 0.1, dist / v_ego, 0)
ax2.plot(t_vals, ttc_no_aeb, color='#b4321e', linewidth=2, linestyle='--', alpha=0.7, label='TTC ohne AEB')

ax2.axvline(t_trigger, color='#193c8c', linewidth=1.5, linestyle=':', alpha=0.8)
ax2.text(t_trigger + 0.05, 3.1, f'AEB-Eingriff\nt={t_trigger:.2f} s', fontsize=8.5,
         color='#193c8c', bbox=dict(boxstyle='round', fc='white', ec='#193c8c', alpha=0.8))

ax2.set_xlabel('Zeit [s]', fontsize=11)
ax2.set_ylabel('Time-to-Collision (TTC) [s]', fontsize=11)
ax2.set_title('(b) TTC-Verlauf: AEB-Eingriff bei 50 km/h\nHindernisabstand d₀ = 45 m', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8.5, framealpha=0.9, loc='lower left')
ax2.set_xlim(0, 3.5)
ax2.set_ylim(0, 3.5)
ax2.grid(True, alpha=0.3, linestyle=':')

fig.suptitle('Abbildung 2: Notbremsassistent (AEB) – Kinematische Analyse',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('home/claude/plot2_aeb_braking.png', format='png', bbox_inches='tight', dpi=150)
print("Plot 2 saved")
