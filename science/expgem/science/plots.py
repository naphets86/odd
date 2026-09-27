import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# --- Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

COLORS = ['#1f4e79','#2e75b6','#70ad47','#ed7d31','#a9341f','#7030a0']

# ========== PLOT 1: Grundlegende Ebbinghaus-Gedächtniskurve mit Exponentialfit ==========
fig, ax = plt.subplots(figsize=(9, 5))
t = np.linspace(0, 30, 500)
lambda_vals = [0.08, 0.15, 0.30, 0.55]
labels = [r'$\lambda=0.08$ (stark konsolidiert)', r'$\lambda=0.15$ (moderat)', 
          r'$\lambda=0.30$ (schwach)', r'$\lambda=0.55$ (sehr schwach)']
for i, (lam, lbl) in enumerate(zip(lambda_vals, labels)):
    ax.plot(t, np.exp(-lam*t), color=COLORS[i], lw=2.2, label=lbl)

# Ebbinghaus original data points (approximate)
t_ebb = np.array([0.33, 1, 8.75, 24, 48, 144, 744])  # hours -> we normalize to days
t_ebb_d = t_ebb / 24.0
r_ebb = np.array([1.0, 0.558, 0.442, 0.336, 0.279, 0.258, 0.211])
ax.scatter(t_ebb_d, r_ebb, color='black', zorder=5, s=50, marker='D', label='Ebbinghaus (1885), normiert')

ax.set_xlabel('Zeit $t$ [Tage]')
ax.set_ylabel('Retention $R(t) = e^{-\\lambda t}$')
ax.set_title('Abbildung 1: Exponentielle Gedächtniskurve – Grundmodell')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.axhline(1/np.e, color='gray', ls='--', lw=1.2, alpha=0.7)
ax.text(30.2, 1/np.e, r'$R=e^{-1}$', va='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('plot1_grundkurve.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 done")

# ========== PLOT 2: Log-Linearer Raum – Nachweis der Exponentialität ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
t_long = np.linspace(0.01, 50, 1000)
lam = 0.15
for ax_i, (xscale, yscale, title) in enumerate([
    ('linear','linear','Linearer Maßstab'),
    ('linear','log','Log-Linearer Maßstab (Beweis)')
]):
    axes[ax_i].plot(t_long, np.exp(-lam*t_long), color=COLORS[1], lw=2.2, label='$R(t)=e^{-0.15t}$')
    axes[ax_i].set_xscale(xscale)
    axes[ax_i].set_yscale(yscale)
    axes[ax_i].set_xlabel('Zeit $t$ [Tage]')
    axes[ax_i].set_ylabel('Retention $R(t)$')
    axes[ax_i].set_title(title)
    axes[ax_i].grid(True, alpha=0.3)
    axes[ax_i].legend()

axes[1].set_title('Log-Linearer Maßstab: Gerade = exponentiell')
# Add regression line in log scale
log_r = -lam * t_long
axes[1].plot(t_long, np.exp(log_r), 'r--', lw=1.5, alpha=0.6, label='Lineare Regression (log)')
fig.suptitle('Abbildung 2: Identifikation der Exponentialstruktur', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('plot2_loglinear.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 done")

# ========== PLOT 3: Bidirektionales Modell – Enkodierung und Abruf ==========
fig, ax = plt.subplots(figsize=(10, 5.5))
t = np.linspace(0, 40, 800)
alpha, beta, gamma, delta = 1.0, 0.8, 0.12, 0.25

R_enc = alpha * np.exp(-gamma * t)
R_abruf = beta * np.exp(-delta * t)
R_gesamt = 0.5 * R_enc + 0.5 * R_abruf

ax.plot(t, R_enc,    color=COLORS[0], lw=2.2, ls='-',  label=r'Enkodierungsstärke $R_e(t)$')
ax.plot(t, R_abruf,  color=COLORS[2], lw=2.2, ls='-',  label=r'Abrufwahrscheinlichkeit $R_a(t)$')
ax.plot(t, R_gesamt, color=COLORS[3], lw=2.8, ls='--', label=r'Gesamtretention $R(t)$')

ax.fill_between(t, R_gesamt, alpha=0.08, color=COLORS[3])
ax.set_xlabel('Zeit $t$ [Tage]')
ax.set_ylabel('Normierte Gedächtnisleistung')
ax.set_title('Abbildung 3: Bidirektionales Exponentialmodell – Enkodierung vs. Abruf')
ax.legend(loc='upper right')
ax.set_xlim(0, 40)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_bidirektional.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 done")

# ========== PLOT 4: Wiederholungseffekt – Spaced Repetition ==========
fig, ax = plt.subplots(figsize=(10, 5.5))
lam = 0.22
t_reps = [0, 1, 3, 7, 14, 28]  # Wiederholungszeitpunkte
t_plot = np.linspace(0, 50, 2000)

R = np.zeros_like(t_plot)
colors_sr = plt.cm.Blues(np.linspace(0.4, 0.9, len(t_reps)))

for i, t_r in enumerate(t_reps):
    mask = t_plot >= t_r
    R_i = np.exp(-lam * (t_plot - t_r))
    R_i[~mask] = 0
    # Jede Wiederholung addiert mit Verstärkungsfaktor
    boost = 1.0 - 0.12 * i
    ax.plot(t_plot[mask], boost * R_i[mask], color=colors_sr[i], lw=1.5,
            label=f'Wiederholung bei $t={t_r}$d')
    R += mask * boost * R_i

R_total = np.clip(R / len(t_reps), 0, 1)
ax.plot(t_plot, R_total, 'r-', lw=2.5, label='Akkumulierte Retention', zorder=10)

for t_r in t_reps:
    ax.axvline(t_r, color='gray', ls=':', lw=1, alpha=0.5)

ax.set_xlabel('Zeit $t$ [Tage]')
ax.set_ylabel('Retention')
ax.set_title('Abbildung 4: Spaced Repetition – Superposition exponentieller Zerfälle')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.set_xlim(0, 50)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot4_spaced_repetition.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 done")

# ========== PLOT 5: Emotionale Valenz – Differenzielle Zerfallsraten ==========
fig, ax = plt.subplots(figsize=(10, 5.5))
t = np.linspace(0, 60, 1000)
conditions = {
    'Hoch-positiv ($\\lambda=0.05$)':   (0.05, COLORS[2]),
    'Positiv ($\\lambda=0.10$)':         (0.10, '#a9d18e'),
    'Neutral ($\\lambda=0.20$)':          (0.20, COLORS[1]),
    'Negativ ($\\lambda=0.13$)':          (0.13, '#f4b183'),
    'Hoch-negativ ($\\lambda=0.07$)':    (0.07, COLORS[4]),
    'Traumatisch ($\\lambda=0.03$)':     (0.03, COLORS[5]),
}
for lbl, (lam, col) in conditions.items():
    ax.plot(t, np.exp(-lam*t), color=col, lw=2.2, label=lbl)

ax.set_xlabel('Zeit $t$ [Tage]')
ax.set_ylabel('Retention $R(t)$')
ax.set_title('Abbildung 5: Emotionale Valenz und differentielle Zerfallsraten')
ax.legend(loc='upper right', fontsize=9.5)
ax.set_xlim(0, 60)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.6)
ax.text(61, 0.5, '$R=0.5$', va='center', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('plot5_emotionale_valenz.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 done")

# ========== PLOT 6: Phasenmodell – Kurz- Mittel- Langzeitgedächtnis ==========
fig, ax = plt.subplots(figsize=(11, 5.5))
t_h = np.linspace(0, 720, 5000)  # Stunden

# KZG: sehr schnell
R_kzg = np.exp(-2.5 * t_h)
# MZG: mittel
R_mzg = 0.85 * np.exp(-0.08 * t_h)
# LZG: sehr langsam
R_lzg = 0.70 * np.exp(-0.005 * t_h)
# Gesamtmodell
w1, w2, w3 = 0.3, 0.4, 0.3
R_ges = w1*R_kzg + w2*R_mzg + w3*R_lzg

ax.plot(t_h, R_kzg, color=COLORS[3], lw=2, ls='-.', label='KZG: $\\lambda_{KZ}=2.5\\,h^{-1}$')
ax.plot(t_h, R_mzg, color=COLORS[1], lw=2, ls='--', label='MZG: $\\lambda_{MZ}=0.08\\,h^{-1}$')
ax.plot(t_h, R_lzg, color=COLORS[2], lw=2, ls=':', label='LZG: $\\lambda_{LZ}=0.005\\,h^{-1}$')
ax.plot(t_h, R_ges,  color=COLORS[0], lw=2.8, ls='-',  label='Multikomponenten-Modell $R_{ges}(t)$')

ax.axvline(1, color='gray', ls=':', lw=1, alpha=0.5)
ax.axvline(24, color='gray', ls=':', lw=1, alpha=0.5)
ax.axvline(168, color='gray', ls=':', lw=1, alpha=0.5)
ax.text(1, 1.02, '1h', ha='center', fontsize=8, color='gray')
ax.text(24, 1.02, '1d', ha='center', fontsize=8, color='gray')
ax.text(168, 1.02, '1W', ha='center', fontsize=8, color='gray')

ax.set_xlabel('Zeit $t$ [Stunden]')
ax.set_ylabel('Retention')
ax.set_title('Abbildung 6: Dreiphasen-Exponentialmodell (KZG / MZG / LZG)')
ax.legend(loc='upper right')
ax.set_xlim(0, 720)
ax.set_ylim(0, 1.07)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_phasenmodell.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 done")

# ========== PLOT 7: Neurobiologische Basis – Synaptische Stärke ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

t_min = np.linspace(0, 120, 2000)  # Minuten
# Synaptische Potenzierung
S_early = np.exp(-0.04 * t_min)  # Early-LTP
S_late  = 0.6 * np.exp(-0.008 * t_min)  # Late-LTP (protein synthesis)
S_total = S_early * 0.4 + S_late * 0.6

axes[0].plot(t_min, S_early, color=COLORS[3], lw=2, label='E-LTP (frühe Phase)')
axes[0].plot(t_min, S_late,  color=COLORS[2], lw=2, label='L-LTP (späte Phase)')
axes[0].plot(t_min, S_total, color=COLORS[0], lw=2.5, label='Gesamtstärke $S(t)$')
axes[0].set_xlabel('Zeit $t$ [Minuten]')
axes[0].set_ylabel('Normierte Synaptische Stärke')
axes[0].set_title('LTP-Zerfall: Synaptische Ebene')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AMPA-Rezeptor-Dichte
t_h2 = np.linspace(0, 72, 1000)
rho_AMPA = 0.3 + 0.7 * np.exp(-0.09 * t_h2)
rho_NMDA = 0.2 + 0.8 * np.exp(-0.03 * t_h2)
axes[1].plot(t_h2, rho_AMPA, color=COLORS[1], lw=2.2, label='AMPA-Rezeptordichte')
axes[1].plot(t_h2, rho_NMDA, color=COLORS[4], lw=2.2, label='NMDA-Rezeptordichte')
axes[1].set_xlabel('Zeit $t$ [Stunden]')
axes[1].set_ylabel('Normierte Rezeptordichte $\\rho(t)$')
axes[1].set_title('Rezeptor-Internalisierung')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Abbildung 7: Neurobiologische Substrate des exponentiellen Zerfalls', fontsize=13)
plt.tight_layout()
plt.savefig('plot7_neurobiologie.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 done")

# ========== PLOT 8: Parameterschätzung & Konfidenzintervalle ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

np.random.seed(42)
t_data = np.array([0, 0.5, 1, 2, 4, 7, 14, 21, 30])
lam_true = 0.18
R_true = np.exp(-lam_true * t_data)
noise = 0.04
R_obs = R_true + np.random.normal(0, noise, len(t_data))
R_obs = np.clip(R_obs, 0.01, 1.0)

def exp_model(t, lam):
    return np.exp(-lam * t)

popt, pcov = curve_fit(exp_model, t_data, R_obs, p0=[0.2])
lam_fit = popt[0]
lam_err = np.sqrt(pcov[0,0])
t_fit = np.linspace(0, 32, 500)

axes[0].scatter(t_data, R_obs, color=COLORS[0], s=60, zorder=5, label='Messdaten (synthetisch)')
axes[0].plot(t_fit, exp_model(t_fit, lam_fit), color=COLORS[3], lw=2.2,
             label=f'Fit: $\\lambda={lam_fit:.4f}\\pm{lam_err:.4f}$')
axes[0].fill_between(t_fit,
    exp_model(t_fit, lam_fit - 1.96*lam_err),
    exp_model(t_fit, lam_fit + 1.96*lam_err),
    alpha=0.2, color=COLORS[3], label='95%-KI')
axes[0].plot(t_fit, exp_model(t_fit, lam_true), 'k--', lw=1.5, label=f'Wahres Modell ($\\lambda={lam_true}$)')
axes[0].set_xlabel('Zeit $t$ [Tage]')
axes[0].set_ylabel('Retention')
axes[0].set_title('Nichtlineare Regression')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Residuen
R_fit_data = exp_model(t_data, lam_fit)
residuals = R_obs - R_fit_data
axes[1].bar(t_data, residuals, width=0.6, color=[COLORS[2] if r>=0 else COLORS[4] for r in residuals])
axes[1].axhline(0, color='k', lw=1.2)
axes[1].set_xlabel('Zeit $t$ [Tage]')
axes[1].set_ylabel('Residuum $e_i = R_i - \\hat{R}_i$')
axes[1].set_title('Residualanalyse')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Abbildung 8: Parameterschätzung des Zerfallskoeffizienten', fontsize=13)
plt.tight_layout()
plt.savefig('plot8_parameterschaetzung.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 done")

# ========== PLOT 9: Schlaf-Konsolidierungs-Modell ==========
fig, ax = plt.subplots(figsize=(11, 5.5))
t_h = np.linspace(0, 72, 3000)

# Ohne Schlaf: steiler Zerfall
R_nosleep = np.exp(-0.25 * t_h)

# Mit Schlaf: nächtliche Konsolidierung steigert Retention
R_sleep = np.zeros_like(t_h)
sleep_windows = [(8, 16, 0.35), (32, 40, 0.20), (56, 64, 0.12)]  # (start, end, boost)
base = np.exp(-0.18 * t_h)
R_sleep = base.copy()

for s, e, boost in sleep_windows:
    mask = t_h >= e
    R_sleep[mask] = R_sleep[mask] * (1.0 + boost * np.exp(-0.04*(t_h[mask]-e)))

R_sleep = np.minimum(R_sleep, 1.0)

ax.fill_between(t_h, R_nosleep, alpha=0.12, color=COLORS[3])
ax.fill_between(t_h, R_sleep, alpha=0.12, color=COLORS[2])
ax.plot(t_h, R_nosleep, color=COLORS[3], lw=2.2, label='Ohne Schlaf ($\\lambda=0.25\\,h^{-1}$)')
ax.plot(t_h, R_sleep,   color=COLORS[2], lw=2.2, label='Mit Schlaf (Konsolidierungsboost)')

for s, e, boost in sleep_windows:
    ax.axvspan(s, e, alpha=0.07, color='navy')
    ax.text((s+e)/2, 1.03, 'Schlaf', ha='center', fontsize=8, color='navy')

ax.set_xlabel('Zeit $t$ [Stunden]')
ax.set_ylabel('Retention $R(t)$')
ax.set_title('Abbildung 9: Schlaf-Konsolidierungsmodell – Modifikation der Zerfallsrate')
ax.legend(loc='upper right')
ax.set_xlim(0, 72)
ax.set_ylim(0, 1.10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot9_schlaf.pdf', bbox_inches='tight')
plt.close()
print("Plot 9 done")

# ========== PLOT 10: Alterung und Gedächtniszerfall – Lebensspanne ==========
fig, ax = plt.subplots(figsize=(11, 5.5))
t = np.linspace(0, 90, 1000)
age_groups = {
    '10–20 Jahre ($\\lambda=0.04$)':  0.04,
    '20–30 Jahre ($\\lambda=0.07$)':  0.07,
    '30–50 Jahre ($\\lambda=0.10$)':  0.10,
    '50–65 Jahre ($\\lambda=0.16$)':  0.16,
    '65–80 Jahre ($\\lambda=0.22$)':  0.22,
    '80+  Jahre ($\\lambda=0.30$)':   0.30,
}
colors_age = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(age_groups)))
for i, (lbl, lam) in enumerate(age_groups.items()):
    ax.plot(t, np.exp(-lam*t), color=colors_age[i], lw=2.2, label=lbl)

ax.set_xlabel('Zeit $t$ [Tage]')
ax.set_ylabel('Retention $R(t)$')
ax.set_title('Abbildung 10: Lebensalter und Zerfallsrate – Entwicklungsneurowissenschaft')
ax.legend(loc='upper right', fontsize=9.5)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot10_alterung.pdf', bbox_inches='tight')
plt.close()
print("Plot 10 done")

print("\nAlle 10 Plots erfolgreich generiert.")
