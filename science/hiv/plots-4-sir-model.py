import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Extended SI model for HIV (no natural recovery, includes treatment)
# S: Susceptible, I: Infected (untreated), T: on Treatment (suppressed), A: AIDS
# dS/dt = mu*N - beta*S*I/N - mu*S
# dI/dt = beta*S*I/N - tau*I - delta_I*I - mu*I
# dT/dt = tau*I - delta_T*T - mu*T
# dA/dt = delta_I*I - mu_A*A

def hiv_si_model(y, t, beta, mu, tau, delta_I, delta_T, mu_A):
    S, I, Tr, A = y
    N = S + I + Tr + A
    if N <= 0:
        return [0, 0, 0, 0]
    dS  = mu * N - beta * S * I / N - mu * S
    dI  = beta * S * I / N - tau * I - delta_I * I - mu * I
    dTr = tau * I - delta_T * Tr - mu * Tr
    dA  = delta_I * I - mu_A * A
    return [dS, dI, dTr, dA]

N0     = 1e6
I0_frac = 0.001
y0_sir = [N0 * (1 - I0_frac), N0 * I0_frac, 0, 0]

t_sir = np.linspace(0, 50, 3000)  # 50 years

# Scenario 1: No treatment (tau=0)
beta1, mu1, tau1, delta_I1, delta_T1, mu_A1 = 0.25, 0.015, 0.0, 0.05, 0.02, 0.20
sol1 = odeint(hiv_si_model, y0_sir, t_sir,
              args=(beta1, mu1, tau1, delta_I1, delta_T1, mu_A1))

# Scenario 2: Moderate treatment uptake
tau2 = 0.15
sol2 = odeint(hiv_si_model, y0_sir, t_sir,
              args=(beta1, mu1, tau2, delta_I1, delta_T1, mu_A1))

# Scenario 3: High treatment coverage (90-90-90 goal)
tau3 = 0.50
beta3 = 0.10  # treatment also reduces beta (TasP)
sol3 = odeint(hiv_si_model, y0_sir, t_sir,
              args=(beta3, mu1, tau3, delta_I1*0.5, delta_T1, mu_A1))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('HIV-Epidemiologie: SI-Modell mit Therapie\n(Bevölkerung N = 1 Mio.)',
             fontsize=14, fontweight='bold')

scenarios = [
    (sol1, 'Kein ART ($\\tau=0$)', '#b4321e'),
    (sol2, 'Moderates ART ($\\tau=0.15$)', '#e07b00'),
    (sol3, 'Hohes ART-Coverage ($\\tau=0.50$, TasP)', '#1e6432'),
]

panels = [
    ('S', 0, 'Suszeptible S(t)', '#1946a0'),
    ('I', 1, 'Infiziert, unbehandelt I(t)', '#b4321e'),
    ('T', 2, 'In ART-Behandlung T(t)', '#1e6432'),
    ('A', 3, 'AIDS-Stadium A(t)', '#6a0dad'),
]

ax_flat = axes.flatten()

for i, (col_idx, label_y) in enumerate([(0, 'S'), (1, 'I'), (2, 'Tr'), (3, 'A')]):
    ax = ax_flat[i]
    idx_map = {'S': 0, 'I': 1, 'Tr': 2, 'A': 3}
    cidx = idx_map[label_y]
    
    colors = ['#b4321e', '#e07b00', '#1e6432']
    lstyles = ['-', '--', ':']
    for j, (sol, lbl, col) in enumerate(scenarios):
        ax.plot(t_sir, sol[:, cidx] / N0 * 100, 
                color=col, linewidth=2.0, linestyle=lstyles[j], label=lbl)
    
    titles = ['Suszeptible S(t)', 'Infizierte I(t) (unbehandelt)',
              'ART-Behandlung T(t)', 'AIDS-Stadium A(t)']
    ax.set_title(titles[i], fontsize=11, fontweight='bold')
    ax.set_xlabel('Zeit (Jahre)', fontsize=9.5)
    ax.set_ylabel('Anteil (%)', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='best')
    ax.grid(True, alpha=0.3)

# Add R0 annotation
r0_no_art  = beta1 / (delta_I1 + mu1)
r0_art_mod = beta1 / (tau2 + delta_I1 + mu1)
r0_art_hi  = beta3 / (tau3 + delta_I1 * 0.5 + mu1)

ax_flat[1].text(30, 4, f'$R_0$ (kein ART) = {r0_no_art:.2f}\n'
                f'$R_0$ (mod. ART) = {r0_art_mod:.2f}\n'
                f'$R_0$ (hohes ART) = {r0_art_hi:.2f}',
                fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('home/claude/plot4_sir_model.png', dpi=150, bbox_inches='tight')
print("Plot 4 saved.")
