"""
Ebola Paper – matplotlib Plots (je einzeln als PDF gespeichert)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'text.usetex': False,
})

OUT = ' N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]

# Ebola-Parameter (Zaire-Stamm, Literaturwerte)
N       = 1_000_000
beta    = 0.27       # Übertragungsrate (1/Tag)
sigma   = 1/9.4      # Inkubationsrate (Inkubation ~9.4 Tage)
gamma   = 1/8.0      # Genesungsrate   (infektiöse Periode ~8 Tage)
R0_ebola = beta / gamma          # ≈ 2.16
CFR      = 0.55                  # Case Fatality Rate Zaire-Stamm

I0 = 10
E0 = 30
S0 = N - I0 - E0
R0_init = 0
y0 = [S0, E0, I0, R0_init]
t_span = (0, 300)
t_eval = np.linspace(0, 300, 3001)

sol = solve_ivp(seir, t_span, y0, t_eval=t_eval,
                args=(beta, sigma, gamma, N), method='RK45', rtol=1e-8)
S, E, I, R = sol.y

# ─────────────────────────────────────────────────────────────
# PLOT 1 – SEIR-Kurven für Ebola-Ausbruch
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Abbildung 1: SEIR-Modell – Ebolavirus-Infektionsdynamik", fontweight='bold', fontsize=13)

ax = axes[0]
ax.plot(t_eval, S/N*100, 'b-',  lw=2, label='S(t) – Suszeptibel')
ax.plot(t_eval, E/N*100, 'orange', lw=2, linestyle='--', label='E(t) – Exponiert')
ax.plot(t_eval, I/N*100, 'r-',  lw=2.5, label='I(t) – Infektiös')
ax.plot(t_eval, R/N*100, 'g-',  lw=2, label='R(t) – Erholt/Entfernt')
ax.set_xlabel("Zeit (Tage)")
ax.set_ylabel("Anteil der Population (%)")
ax.set_title(f"SEIR-Verlauf (N={N:,}, R0={R0_ebola:.2f})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)
ax.set_ylim(0, 100)

# Subplot 2: Infektiöse Personen absolut + Todesfälle kumulativ
cumD = np.cumsum(gamma * I * CFR) * (t_eval[1]-t_eval[0])
ax2 = axes[1]
color1, color2 = 'crimson', 'black'
lns1 = ax2.plot(t_eval, I, color=color1, lw=2.5, label='Aktiv Infektiöse I(t)')
ax2.set_xlabel("Zeit (Tage)")
ax2.set_ylabel("Aktiv Infektiöse Personen", color=color1)
ax2.tick_params(axis='y', labelcolor=color1)
ax3 = ax2.twinx()
lns2 = ax3.plot(t_eval, cumD, color=color2, lw=2, linestyle=':', label='Kumulat. Todesfälle (CFR=55%)')
ax3.set_ylabel("Kumulierte Todesfälle", color=color2)
ax3.tick_params(axis='y', labelcolor=color2)
lns = lns1 + lns2
ax2.legend(lns, [l.get_label() for l in lns], loc='center right')
ax2.set_title("Infektiöse und Todesfälle (absolut)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/plot1_seir.pdf", bbox_inches='tight')
plt.close()
print("Plot 1 done")

# ─────────────────────────────────────────────────────────────
# PLOT 2 – Basisreproduktionszahl R0 im Vergleich
# ─────────────────────────────────────────────────────────────
diseases = [
    ('Masern',         12.0, '#e74c3c'),
    ('Keuchhusten',     5.5, '#e67e22'),
    ('Mumps',           5.0, '#f39c12'),
    ('Pocken',          5.0, '#d35400'),
    ('Polio',           5.0, '#c0392b'),
    ('COVID-19 Delta',  5.0, '#8e44ad'),
    ('COVID-19 Wuhan',  2.5, '#9b59b6'),
    ('Ebola Zaire',     2.2, '#e74c3c'),
    ('Ebola Sudan',     1.8, '#c0392b'),
    ('Ebola Bundibugyo',1.3, '#922b21'),
    ('Ebola Reston',    1.1, '#7b241c'),
    ('SARS-CoV-1',      2.0, '#2980b9'),
    ('Grippe (saisonal)',1.3,'#27ae60'),
    ('HIV',             2.5, '#1abc9c'),
]

diseases_sorted = sorted(diseases, key=lambda x: x[1], reverse=True)
names = [d[0] for d in diseases_sorted]
r0s   = [d[1] for d in diseases_sorted]
cols  = [d[2] for d in diseases_sorted]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Abbildung 2: Basisreproduktionszahl R0 im Krankheitsvergleich", fontweight='bold', fontsize=13)

ax = axes[0]
bars = ax.barh(names, r0s, color=cols, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, r0s):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(x=1.0, color='red', linestyle='--', lw=2, label='R0 = 1 (epidemischer Schwellenwert)')
ax.set_xlabel("Basisreproduktionszahl R0")
ax.set_title("R0-Vergleich verschiedener Infektionskrankheiten")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 15)

# Subplot 2: Herd-Immunity-Threshold als Funktion von R0
r0_range = np.linspace(1.01, 15, 500)
herd_thresh = 1 - 1/r0_range

ax2 = axes[1]
ax2.plot(r0_range, herd_thresh * 100, 'b-', lw=2.5, label='Herdenimmunitätsschwelle q*(R0)')
ebola_values = {'Ebola Zaire': (2.2, '#e74c3c'),
                'Ebola Sudan': (1.8, '#c0392b'),
                'Ebola Bundibugyo': (1.3, '#922b21'),
                'Masern': (12.0, '#f39c12'),
                'COVID-19 Delta': (5.0, '#8e44ad')}
for disease, (r0, c) in ebola_values.items():
    ht = (1 - 1/r0) * 100
    ax2.axvline(x=r0, color=c, linestyle=':', lw=1.5, alpha=0.8)
    ax2.scatter([r0], [ht], color=c, s=60, zorder=5)
    ax2.annotate(f'{disease}\n({ht:.0f}%)', xy=(r0, ht), xytext=(r0+0.2, ht-5),
                 fontsize=7.5, color=c, fontweight='bold')
ax2.set_xlabel("Basisreproduktionszahl R0")
ax2.set_ylabel("Herdenimmunitätsschwelle q* (%)")
ax2.set_title("Herdenimmunität als Funktion von R0")
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUT}/plot2_r0.pdf", bbox_inches='tight')
plt.close()
print("Plot 2 done")

# ─────────────────────────────────────────────────────────────
# PLOT 3 – Sensitivitätsanalyse: Einfluss von beta und gamma auf R0
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Abbildung 3: Sensitivitätsanalyse der SEIR-Parameter", fontweight='bold', fontsize=13)

beta_range  = np.linspace(0.05, 0.60, 200)
gamma_range = np.linspace(1/20, 1/3,  200)

# Links: Epidemic peak vs beta (gamma fixed)
ax = axes[0]
gamma_fixed = 1/8.0
peak_I_list = []
attack_rate = []
for b in beta_range:
    r0_val = b / gamma_fixed
    sol_tmp = solve_ivp(seir, (0,600), y0, t_eval=np.linspace(0,600,6001),
                        args=(b, sigma, gamma_fixed, N), method='RK45', rtol=1e-8)
    Itmp = sol_tmp.y[2]
    peak_I_list.append(Itmp.max()/N*100)
    attack_rate.append((N - sol_tmp.y[0][-1])/N*100)

r0_x = beta_range / gamma_fixed
ax.plot(r0_x, peak_I_list, 'r-', lw=2.5, label='Maximale Prävalenz I_peak (%)')
ax.plot(r0_x, attack_rate, 'b--', lw=2, label='Gesamtangriffsrate AR (%)')
ax.axvline(x=R0_ebola, color='orange', lw=2, linestyle=':', label=f'Ebola Zaire R0={R0_ebola:.2f}')
ax.axhline(y=50, color='gray', lw=1, linestyle='--', alpha=0.5)
ax.set_xlabel("Basisreproduktionszahl R0 = beta / gamma")
ax.set_ylabel("Prozent der Population (%)")
ax.set_title("Peakprävalenz und Angriffsrate vs. R0")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5)

# Rechts: Heatmap von Peak I(%) als Funktion von beta und gamma
ax2 = axes[1]
beta_g  = np.linspace(0.05, 0.60, 50)
gamma_g = np.linspace(1/20, 1/3,  50)
B, G = np.meshgrid(beta_g, gamma_g)
R0_grid = B / G
# Analytische Näherung: Attack rate aus finaler Größe
# 1 - AR = exp(-R0 * AR) => löse numerisch
def final_size(r0):
    if r0 <= 1:
        return 0.0
    try:
        def eq(z): return z - 1 + np.exp(-r0*z)
        return brentq(eq, 1e-9, 1-1e-9)
    except:
        return 0.0

AR_grid = np.vectorize(final_size)(R0_grid) * 100
im = ax2.contourf(B, 1/G, AR_grid, levels=20, cmap='RdYlGn_r')
plt.colorbar(im, ax=ax2, label='Gesamtangriffsrate (%)')
ax2.contour(B, 1/G, R0_grid, levels=[1.0], colors='blue', linewidths=2)
ax2.contour(B, 1/G, R0_grid, levels=[2.2], colors='orange', linewidths=2, linestyles='--')
ax2.set_xlabel("Übertragungsrate beta (1/Tag)")
ax2.set_ylabel("Infektiöse Periode 1/gamma (Tage)")
ax2.set_title("Angriffsrate AR(beta, gamma) [Heatmap]\nBlau: R0=1, Orange: R0=2.2")
ax2.scatter([beta], [8.0], color='white', s=100, marker='*', zorder=5, label='Ebola Zaire')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUT}/plot3_sensitivity.pdf", bbox_inches='tight')
plt.close()
print("Plot 3 done")

# ─────────────────────────────────────────────────────────────
# PLOT 4 – Impfstoffwirksamkeit und Szenarien
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Abbildung 4: Impfstoffwirksamkeit und epidemiologische Szenarien", fontweight='bold', fontsize=13)

def seir_vaccine(t, y, beta, sigma, gamma, N, VE, p_vacc):
    """SEIR mit Impfung: p_vacc = Anteil geimpfter, VE = Wirksamkeit"""
    S, E, I, R = y
    beta_eff = beta * (1 - VE * p_vacc)  # Effektive Übertragungsrate
    dS = -beta_eff * S * I / N
    dE =  beta_eff * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]

ax = axes[0]
VE_rVSV = 0.976  # rVSV-ZEBOV Wirksamkeit
scenarios = [
    (0.0,  0.0,   'Keine Impfung',          'r-',   2.5),
    (VE_rVSV, 0.30, '30% geimpft (rVSV)',   'orange','--', 2.0),
    (VE_rVSV, 0.50, '50% geimpft (rVSV)',   'gold',  '-.',  2.0),
    (VE_rVSV, 0.70, '70% geimpft (rVSV)',   'g--',   None, 2.0),
    (VE_rVSV, 0.90, '90% geimpft (rVSV)',   'b-',    None, 2.0),
]

for item in scenarios:
    if len(item) == 5:
        ve, pv, label, style, lw = item
    else:
        ve, pv, label, style = item[:4]
        lw = 2.0
    sol_v = solve_ivp(seir_vaccine, (0,300), y0, t_eval=t_eval,
                      args=(beta, sigma, gamma, N, ve, pv), method='RK45', rtol=1e-8)
    ax.plot(t_eval, sol_v.y[2]/N*100, style, lw=lw, label=label)

ax.set_xlabel("Zeit (Tage)")
ax.set_ylabel("Anteil Infektiöser (%)")
ax.set_title(f"Impfszenarien (rVSV-ZEBOV, VE={VE_rVSV*100:.1f}%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

# Subplot 2: Effektives R0 als Funktion der Impfquote
ax2 = axes[1]
vacc_range  = np.linspace(0, 1, 200)
VE_values   = [0.976, 0.80, 0.60, 0.40]  # verschiedene Wirksamkeiten
colors_ve   = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
labels_ve   = ['rVSV-ZEBOV (97.6%)', 'VE = 80%', 'VE = 60%', 'VE = 40%']

for ve_val, c, lbl in zip(VE_values, colors_ve, labels_ve):
    Re_vals = R0_ebola * (1 - ve_val * vacc_range)
    ax2.plot(vacc_range * 100, Re_vals, color=c, lw=2, label=lbl)

ax2.axhline(y=1.0, color='black', lw=2.5, linestyle='--', label='Re = 1 (Ausbruchsschwelle)')
ax2.fill_between(vacc_range*100, 0, 1, alpha=0.1, color='green', label='Ausbruch kontrolliert (Re<1)')
ax2.set_xlabel("Impfquote p (%)")
ax2.set_ylabel("Effektive Reproduktionszahl Re(p)")
ax2.set_title("Effektives Re als Funktion der Impfquote p")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 2.5)

plt.tight_layout()
plt.savefig(f"{OUT}/plot4_vaccine.pdf", bbox_inches='tight')
plt.close()
print("Plot 4 done")

# ─────────────────────────────────────────────────────────────
# PLOT 5 – Virale Kinetik: Within-Host-Modell
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Abbildung 5: Intrazelluläre Virale Kinetik (Within-Host-Modell)", fontweight='bold', fontsize=13)

# Standard-Virale-Kinetik-Modell: T'=-beta*V*T, I'=beta*V*T-delta*I, V'=p*I-c*V
def viral_kinetics(t, y, beta_v, delta, p, c):
    T, Inf, V = y
    dT   = -beta_v * V * T
    dInf =  beta_v * V * T - delta * Inf
    dV   =  p * Inf - c * V
    return [dT, dInf, dV]

# Ebola Within-Host Parameter (Nguyen et al., Madelain et al.)
beta_v = 5e-7   # Infektionsrate (ml/Kopie/Tag)
delta  = 1.0    # Infizierte-Zell-Abbaurate (1/Tag)
p_v    = 100.0  # Virusproduktionsrate (Kopien/Zelle/Tag)
c_v    = 3.0    # Virusabbaurate (1/Tag)

T0   = 1e6   # Zielzellen (hepatische/immunologische Zellen)
Inf0 = 0.0
V0   = 1.0   # Startvirulast (1 Kopie/ml)

y0_vk = [T0, Inf0, V0]
t_vk  = np.linspace(0, 21, 2100)
sol_vk = solve_ivp(viral_kinetics, (0,21), y0_vk, t_eval=t_vk,
                   args=(beta_v, delta, p_v, c_v), method='RK45', rtol=1e-10)
T_arr, Inf_arr, V_arr = sol_vk.y

ax = axes[0]
ax2_twin = ax.twinx()
lns1 = ax.semilogy(t_vk, np.maximum(V_arr, 1e-3), 'r-', lw=2.5, label='Viruslast V(t) [Kopien/ml]')
lns2 = ax2_twin.plot(t_vk, T_arr/T0*100, 'b--', lw=2, label='Gesunde Zielzellen T(t) [%]')
lns3 = ax2_twin.plot(t_vk, Inf_arr/T0*100, 'orange', lw=2, linestyle=':', label='Infizierte Zellen I(t) [%]')
lns = lns1 + lns2 + lns3
ax.set_xlabel("Tage nach Infektion")
ax.set_ylabel("Viruslast (log, Kopien/ml)", color='red')
ax2_twin.set_ylabel("Zellanteil (%)")
ax.tick_params(axis='y', labelcolor='red')
ax.set_title("Virale Kinetik: Zellen und Viruslast")
ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

# Subplot 2: Viruslast mit und ohne Behandlung (mAk-Therapie)
ax3 = axes[1]
# Mit Behandlung: Virusabbaurate c erhöht sich ab Tag 5 (monoklonale Antikörper)
def viral_kinetics_treatment(t, y, beta_v, delta, p, c, t_treat, c_boost):
    T, Inf, V = y
    c_eff = c + c_boost if t >= t_treat else c
    dT   = -beta_v * V * T
    dInf =  beta_v * V * T - delta * Inf
    dV   =  p * Inf - c_eff * V
    return [dT, dInf, dV]

treatment_scenarios = [
    (None, 0,     'Keine Behandlung',    'r-',   2.5),
    (5,    20,  'mAk ab Tag 5',         'b-',   2.0),
    (5,    50,  'mAk ab Tag 5 (stark)', '#2ecc71', 2.0),
    (8,    20,  'mAk ab Tag 8 (spät)', 'orange', 2.0),
]

for t_treat, c_boost, label, style, lw in treatment_scenarios:
    if t_treat is None:
        sol_t = sol_vk
    else:
        sol_t = solve_ivp(viral_kinetics_treatment, (0,21), y0_vk, t_eval=t_vk,
                          args=(beta_v, delta, p_v, c_v, t_treat, c_boost),
                          method='RK45', rtol=1e-10)
    Vt = np.maximum(sol_t.y[2], 1e-4)
    ax3.semilogy(t_vk, Vt, style, lw=lw, label=label)

ax3.set_xlabel("Tage nach Infektion")
ax3.set_ylabel("Viruslast (log-Skala, Kopien/ml)")
ax3.set_title("Viruslastdynamik: Behandlungsszenarien")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(1e-2, 1e10)

plt.tight_layout()
plt.savefig(f"{OUT}/plot5_kinetics.pdf", bbox_inches='tight')
plt.close()
print("Plot 5 done")

# ─────────────────────────────────────────────────────────────
# PLOT 6 – Vergleich Ebola-Stämme + Historische Ausbrüche
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Abbildung 6: Ebolavirus-Stämme: Vergleich und historische Ausbrüche", fontweight='bold', fontsize=13)

strains = {
    'Zaire\n(EBOV)':      {'R0': 2.2,  'CFR': 0.68, 'Inkubation': 6.3,  'color': '#e74c3c'},
    'Sudan\n(SUDV)':      {'R0': 1.8,  'CFR': 0.53, 'Inkubation': 7.0,  'color': '#c0392b'},
    'Bundibugyo\n(BDBV)': {'R0': 1.3,  'CFR': 0.35, 'Inkubation': 8.0,  'color': '#922b21'},
    'Tai Forest\n(TAFV)': {'R0': 1.1,  'CFR': 0.01, 'Inkubation': 7.5,  'color': '#7b241c'},
    'Reston\n(RESTV)':    {'R0': 1.05, 'CFR': 0.00, 'Inkubation': 9.0,  'color': '#95a5a6'},
}

ax = axes[0]
x = np.arange(len(strains))
width = 0.35
names_s = list(strains.keys())
r0s_s   = [v['R0'] for v in strains.values()]
cfrs_s  = [v['CFR']*100 for v in strains.values()]
cols_s  = [v['color'] for v in strains.values()]

bars1 = ax.bar(x - width/2, r0s_s, width, label='R0', color=cols_s, edgecolor='black', lw=0.5)
ax2_s = ax.twinx()
bars2 = ax2_s.bar(x + width/2, cfrs_s, width, label='CFR (%)', color=cols_s, edgecolor='black', lw=0.5, alpha=0.6, hatch='//')
ax.set_xlabel("Ebolavirus-Stamm")
ax.set_ylabel("Basisreproduktionszahl R0")
ax2_s.set_ylabel("Case Fatality Rate CFR (%)")
ax.set_xticks(x)
ax.set_xticklabels(names_s, fontsize=9)
ax.set_title("Vergleich der Ebolavirus-Stämme")
ax.legend(loc='upper left', fontsize=8)
ax2_s.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 3)
ax2_s.set_ylim(0, 100)

# Subplot 2: Historische Ausbrüche (Fälle vs. Todesfälle)
outbreaks = [
    ('Zaire 1976',         318,  280, '#e74c3c'),
    ('Sudan 1976',         284,  151, '#c0392b'),
    ('Sudan 1979',          34,   22, '#c0392b'),
    ('Zaire 1995',         315,  254, '#e74c3c'),
    ('Uganda 2000',        425,  224, '#e74c3c'),
    ('Zaire 2001-02',       57,   43, '#e74c3c'),
    ('W-Afrika 2013-16', 28646,11323, '#8b0000'),
    ('Zaire 2018-20',    3481,  2299, '#e74c3c'),
    ('Zaire 2022',         169,   75, '#e74c3c'),
    ('DRC 2024',           56,   35, '#e74c3c'),
]
ob_names  = [o[0] for o in outbreaks]
ob_cases  = [o[1] for o in outbreaks]
ob_deaths = [o[2] for o in outbreaks]
ob_cols   = [o[3] for o in outbreaks]
ob_cfr    = [d/c*100 for c, d in zip(ob_cases, ob_deaths)]

ax3 = axes[1]
y_pos = np.arange(len(ob_names))
ax3.barh(y_pos, ob_cases,  0.4, label='Gesamtfälle',   color='steelblue',  alpha=0.8, edgecolor='black', lw=0.5)
ax3.barh(y_pos-0.4, ob_deaths, 0.4, label='Todesfälle',color='crimson', alpha=0.8, edgecolor='black', lw=0.5)
for i, (cases, deaths, cfr) in enumerate(zip(ob_cases, ob_deaths, ob_cfr)):
    ax3.text(cases + 50, i,      f'{cases:,}',     va='center', fontsize=7)
    ax3.text(deaths + 50, i-0.4, f'{cfr:.0f}% CFR', va='center', fontsize=7, color='crimson')
ax3.set_yticks(y_pos - 0.2)
ax3.set_yticklabels(ob_names, fontsize=8)
ax3.set_xlabel("Anzahl Personen (log-Skala)")
ax3.set_xscale('log')
ax3.set_title("Historische Ebola-Ausbrüche (Fälle & Todesfälle)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{OUT}/plot6_outbreaks.pdf", bbox_inches='tight')
plt.close()
print("Plot 6 done")

print("\nAlle 6 Plots erfolgreich generiert!")
