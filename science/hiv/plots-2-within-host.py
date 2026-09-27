import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Within-host model: Target cell limited model (Perelson et al.)
# dT/dt = lambda - d_T * T - beta * T * V
# dI/dt = beta * T * V - delta * I
# dV/dt = p * I - c * V

def hiv_model(y, t, lam, d_T, beta, delta, p, c):
    T, I, V = y
    dT = lam - d_T * T - beta * T * V
    dI = beta * T * V - delta * I
    dV = p * I - c * V
    return [dT, dI, dV]

# Parameters (standard values from literature)
lam   = 1e4      # target cell production rate (cells/mL/day)
d_T   = 0.01     # target cell death rate
beta  = 2.4e-8   # infection rate
delta = 0.7      # infected cell death rate
p     = 300      # viral production rate per infected cell
c     = 3        # viral clearance rate

# Initial conditions: healthy state
T0 = lam / d_T   # ~1e6 cells/mL
I0 = 0
V0 = 1  # single virus introduced

y0 = [T0, I0, V0]

t_acute   = np.linspace(0, 90, 2000)    # acute phase
t_chronic = np.linspace(90, 3650, 5000) # chronic phase (10 years)

sol_a = odeint(hiv_model, y0, t_acute,   args=(lam, d_T, beta, delta, p, c))
y_at90 = sol_a[-1]
sol_c = odeint(hiv_model, y_at90, t_chronic, args=(lam, d_T, beta, delta, p, c))

T_full = np.concatenate([sol_a[:, 0], sol_c[:, 0]])
I_full = np.concatenate([sol_a[:, 1], sol_c[:, 1]])
V_full = np.concatenate([sol_a[:, 2], sol_c[:, 2]])
t_full = np.concatenate([t_acute, t_chronic])

V_full = np.clip(V_full, 1, None)
I_full = np.clip(I_full, 0, None)
T_full = np.clip(T_full, 0, None)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle('HIV Within-Host Dynamik:\nMathematisches Modell (Perelson et al.)',
             fontsize=14, fontweight='bold')

# Viral load
ax1 = axes[0]
ax1.semilogy(t_full, V_full, color='#b4321e', linewidth=2.0)
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.text(200, 70, 'Nachweisgrenze (50 Kopien/mL)', fontsize=8.5, color='gray')
ax1.axvline(x=14, color='#1946a0', linestyle=':', alpha=0.6)
ax1.text(15, 1e6, 'Akuter\nPeak (~14d)', fontsize=8, color='#1946a0')
ax1.set_ylabel('Viruslast (Kopien/mL)', fontsize=10)
ax1.set_ylim(1, 1e9)
ax1.grid(True, alpha=0.3)
ax1.legend(['Viruslast V(t)'], loc='upper right', fontsize=9)

# CD4+ T cells
ax2 = axes[1]
ax2.plot(t_full, T_full, color='#1946a0', linewidth=2.0, label='Gesunde T-Zellen T(t)')
ax2.axhline(y=2e5, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.text(200, 2.1e5, 'AIDS-Grenze (CD4 < 200/μL)', fontsize=8.5, color='orange')
ax2.set_ylabel('CD4$^+$ T-Zellen (Zellen/mL)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Infected cells
ax3 = axes[2]
ax3.plot(t_full, I_full, color='#6a0dad', linewidth=2.0, label='Infizierte Zellen I(t)')
ax3.set_ylabel('Infizierte Zellen (Zellen/mL)', fontsize=10)
ax3.set_xlabel('Zeit (Tage nach Infektion)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

# Add phase labels
for ax in axes:
    ax.axvspan(0, 90, alpha=0.06, color='red', label='_nolegend_')
    ax.axvspan(90, 3650, alpha=0.04, color='blue', label='_nolegend_')

axes[0].text(20, 3, 'Akutphase', fontsize=8, color='darkred',
             bbox=dict(boxstyle='round', fc='#fff0ee', ec='darkred', alpha=0.7))
axes[0].text(500, 3, 'Chronische Phase', fontsize=8, color='#1946a0',
             bbox=dict(boxstyle='round', fc='#f0f0ff', ec='#1946a0', alpha=0.7))

ax3.set_xlim(-20, 3700)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('home/claude/plot2_within_host.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved.")
