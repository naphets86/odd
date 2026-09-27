import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Model parameters
lam   = 1e4
d_T   = 0.01
beta  = 2.4e-8
delta = 0.7
p     = 300
c     = 3

def hiv_model(y, t, lam, d_T, beta, delta, p, c):
    T, I, V = y
    dT = lam - d_T * T - beta * T * V
    dI = beta * T * V - delta * I
    dV = p * I - c * V
    return [dT, dI, dV]

T0 = lam / d_T
y0 = [T0, 0, 1]
t_all = np.linspace(0, 3650, 8000)

# Untreated
sol_unt = odeint(hiv_model, y0, t_all, args=(lam, d_T, beta, delta, p, c))
T_unt = np.clip(sol_unt[:, 0], 0, None)
V_unt = np.clip(sol_unt[:, 2], 1, None)

# ART started at year 1 (day 365): 95% reduction in infection rate
t_before_art = np.linspace(0, 365, 3000)
sol_ba = odeint(hiv_model, y0, t_before_art, args=(lam, d_T, beta, delta, p, c))
y_at_art = sol_ba[-1]
t_after_art = np.linspace(365, 3650, 6000)
beta_art = beta * 0.05  # 95% suppression
sol_aa = odeint(hiv_model, y_at_art, t_after_art,
                args=(lam, d_T, beta_art, delta*0.95, p*0.03, c))
T_art = np.concatenate([sol_ba[:, 0], sol_aa[:, 0]])
V_art = np.concatenate([sol_ba[:, 2], sol_aa[:, 2]])
V_art = np.clip(V_art, 1, None)
T_art = np.clip(T_art, 0, None)
t_art = np.concatenate([t_before_art, t_after_art])

# ART started very early (day 90)
t_before_early = np.linspace(0, 90, 1000)
sol_be = odeint(hiv_model, y0, t_before_early, args=(lam, d_T, beta, delta, p, c))
y_at_early = sol_be[-1]
t_after_early = np.linspace(90, 3650, 7000)
sol_ae = odeint(hiv_model, y_at_early, t_after_early,
                args=(lam, d_T, beta_art, delta*0.95, p*0.03, c))
T_early = np.concatenate([sol_be[:, 0], sol_ae[:, 0]])
V_early = np.concatenate([sol_be[:, 2], sol_ae[:, 2]])
V_early = np.clip(V_early, 1, None)
t_early = np.concatenate([t_before_early, t_after_early])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle('CD4$^+$ T-Zellen und Viruslast:\nVergleich unbehandelt vs. ART',
             fontsize=14, fontweight='bold')

# CD4 count
ax1.plot(t_all/365, T_unt/1e6, color='#b4321e', linewidth=2.0, label='Unbehandelt')
ax1.plot(t_art/365, T_art/1e6, color='#1946a0', linewidth=2.0, label='ART (Start: Jahr 1)')
ax1.plot(t_early/365, T_early/1e6, color='#1e6432', linewidth=2.0, linestyle='--',
         label='ART (früher Start: Monat 3)')
ax1.axhline(y=0.2, color='orange', linestyle=':', linewidth=2, alpha=0.9)
ax1.text(5.5, 0.22, 'AIDS-Schwellenwert\n(CD4 < 200/μL)', fontsize=9, color='darkorange')
ax1.fill_between(t_all/365, 0, 0.2, alpha=0.1, color='red')
ax1.set_ylabel('CD4$^+$ T-Zellen ($\\times 10^6$/mL)', fontsize=11)
ax1.legend(fontsize=9.5, loc='right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1e6/1e6)

# Viral load
ax2.semilogy(t_all/365, V_unt, color='#b4321e', linewidth=2.0, label='Unbehandelt')
ax2.semilogy(t_art/365, V_art, color='#1946a0', linewidth=2.0, label='ART (Start: Jahr 1)')
ax2.semilogy(t_early/365, V_early, color='#1e6432', linewidth=2.0, linestyle='--',
             label='ART (früher Start: Monat 3)')
ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.text(0.2, 70, 'Nachweisgrenze (50 Kopien/mL)', fontsize=8.5, color='gray')
ax2.set_ylabel('Viruslast (Kopien/mL)', fontsize=11)
ax2.set_xlabel('Zeit (Jahre nach Infektion)', fontsize=11)
ax2.legend(fontsize=9.5, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1, 1e9)

# Mark ART start
ax1.axvline(x=1, color='#1946a0', linestyle=':', alpha=0.5)
ax2.axvline(x=1, color='#1946a0', linestyle=':', alpha=0.5)
ax1.axvline(x=90/365, color='#1e6432', linestyle=':', alpha=0.5)
ax2.axvline(x=90/365, color='#1e6432', linestyle=':', alpha=0.5)

ax2.set_xlim(-0.1, 10.1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('home/claude/plot3_cd4_art.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved.")
