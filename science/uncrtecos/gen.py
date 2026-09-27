"""
PDF-Export aller Plots
======================

Dieses Skript generiert alle Plots direkt als PDF statt PNG.
PDFs haben bessere Qualität für LaTeX-Dokumente und sind vektorbasiert.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Matplotlib-Einstellungen für PDF
rcParams['figure.figsize'] = (14, 8)
rcParams['font.size'] = 11
rcParams['font.family'] = 'sans-serif'
rcParams['axes.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5
rcParams['grid.alpha'] = 0.3
rcParams['pdf.fonttype'] = 42  # Wichtig: Embedded fonts für LaTeX
rcParams['ps.fonttype'] = 42

np.random.seed(42)

print("="*70)
print("Generiere alle Plots als hochwertige PDFs für LaTeX")
print("="*70)

# ============================================================
# PDF 1: Degradationskurven - Vergleich
# ============================================================
tau = np.linspace(0, 365, 1000)

# Degradationsmodelle
delta_max_1 = 2.5
lambda_1 = 0.01
degradation_exp = 1 + (delta_max_1 - 1) * (1 - np.exp(-lambda_1 * tau))

alpha_2 = 0.005
beta_2 = 0.7
degradation_power = 1 + alpha_2 * (tau ** beta_2)

gamma_3 = 0.15
degradation_log = 1 + gamma_3 * np.log(1 + tau)

delta_max_4 = 2.8
lambda_4 = 0.03
tau_0_4 = 100
degradation_logistic = delta_max_4 / (1 + np.exp(-lambda_4 * (tau - tau_0_4)))

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(tau, degradation_exp, 'b-', linewidth=2.5, label='Exponentiell', alpha=0.8)
ax.plot(tau, degradation_power, 'r-', linewidth=2.5, label='Potenzgesetz', alpha=0.8)
ax.plot(tau, degradation_log, 'g-', linewidth=2.5, label='Logarithmisch', alpha=0.8)
ax.plot(tau, degradation_logistic, 'orange', linewidth=2.5, label='Logistisch', alpha=0.8)

ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Maximale Fehlertoleranz (30% Fehler)')

ax.set_xlabel('Zeithorizont τ (Tage)', fontsize=13, fontweight='bold')
ax.set_ylabel(r'Degradation $\delta(\tau)$', fontsize=13, fontweight='bold')
ax.set_title('Vergleich aller vier Degradationsmodelle\nWie schnell wird die Prognose ungenauer?', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 365)
ax.set_ylim(0.95, 3.5)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)

critical_exp = np.where(degradation_exp >= 1.5)[0]
critical_power = np.where(degradation_power >= 1.5)[0]

if len(critical_exp) > 0:
    ax.plot(tau[critical_exp[0]], degradation_exp[critical_exp[0]], 'bo', markersize=8)
    ax.text(tau[critical_exp[0]], degradation_exp[critical_exp[0]]+0.15, 
            f'{tau[critical_exp[0]]:.0f}d', fontsize=9, ha='center', color='blue')
    
if len(critical_power) > 0:
    ax.plot(tau[critical_power[0]], degradation_power[critical_power[0]], 'ro', markersize=8)
    ax.text(tau[critical_power[0]], degradation_power[critical_power[0]]-0.25, 
            f'{tau[critical_power[0]]:.0f}d', fontsize=9, ha='center', color='red')

plt.tight_layout()
plt.savefig('fig_degradation_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 1 saved: fig_degradation_comparison.pdf")
plt.close()

# ============================================================
# PDF 2: Fehlerraten
# ============================================================
baseline_std = 10
error_rate_exp = baseline_std * np.sqrt(degradation_exp)
error_rate_power = baseline_std * np.sqrt(degradation_power)
error_rate_log = baseline_std * np.sqrt(degradation_log)
error_rate_logistic = baseline_std * np.sqrt(degradation_logistic)

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(tau, error_rate_exp, 'b-', linewidth=2.5, label='Exponentiell', alpha=0.8)
ax.plot(tau, error_rate_power, 'r-', linewidth=2.5, label='Potenzgesetz', alpha=0.8)
ax.plot(tau, error_rate_log, 'g-', linewidth=2.5, label='Logarithmisch', alpha=0.8)
ax.plot(tau, error_rate_logistic, 'orange', linewidth=2.5, label='Logistisch', alpha=0.8)

ax.axhspan(0, 15, alpha=0.1, color='green', label='Akzeptabel (< 15% Fehler)')
ax.axhspan(15, 25, alpha=0.1, color='yellow', label='Marginal (15-25%)')
ax.axhspan(25, 40, alpha=0.1, color='red', label='Kritisch (> 25%)')

ax.set_xlabel('Zeithorizont τ (Tage)', fontsize=13, fontweight='bold')
ax.set_ylabel('Prognose-Fehlerrate σ(τ) (%)', fontsize=13, fontweight='bold')
ax.set_title('Prognose-Fehlerraten: Praktische Interpretation\nWann wird Planung unmöglich?', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 365)
ax.set_ylim(8, 40)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('fig_error_rates.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 2 saved: fig_error_rates.pdf")
plt.close()

# ============================================================
# PDF 3: Kundentrajektorien (der angeforderte Plot!)
# ============================================================
num_customers = 100
num_days = 365
days = np.arange(0, num_days)
customer_data = np.zeros((num_customers, num_days))
baseline_trends = np.zeros((num_customers, num_days))

drift_rate = 0.01
ar_coeff = 0.6
noise_std = 10
baseline_demand = 100

for i in range(num_customers):
    baseline_drift = baseline_demand + drift_rate * baseline_demand * days
    baseline_with_variation = baseline_drift * (0.8 + 0.4 * np.random.rand())
    baseline_trends[i] = baseline_with_variation
    
    customer_data[i, 0] = baseline_trends[i, 0] + np.random.normal(0, noise_std)
    
    for t in range(1, num_days):
        ar_component = ar_coeff * customer_data[i, t-1]
        noise = np.random.normal(0, noise_std)
        customer_data[i, t] = (1 - ar_coeff) * baseline_trends[i, t] + ar_component + noise
        customer_data[i, t] = max(customer_data[i, t], 0)

daily_mean = np.mean(customer_data, axis=0)
daily_std = np.std(customer_data, axis=0)
daily_p10 = np.percentile(customer_data, 10, axis=0)
daily_p90 = np.percentile(customer_data, 90, axis=0)

fig, ax = plt.subplots(figsize=(16, 10))

# Alle Kundenkurven in grau
for i in range(num_customers):
    ax.plot(days, customer_data[i], color='gray', alpha=0.08, linewidth=0.5)

# Mittelwert und Perzentile
ax.plot(days, daily_mean, 'r-', linewidth=3, label='Durchschnittliche Nachfrage', zorder=10)
ax.fill_between(days, daily_p10, daily_p90, alpha=0.2, color='red', 
                label='10.-90. Perzentil (80% der Kunden)', zorder=5)
ax.fill_between(days, daily_mean - daily_std, daily_mean + daily_std, alpha=0.15, color='red',
                label='±1 Std. Abw.', zorder=5)

ax.set_xlabel('Tag des Jahres', fontsize=13, fontweight='bold')
ax.set_ylabel('Kundennachfrage (Einheiten)', fontsize=13, fontweight='bold')
ax.set_title('Simulation von 100 Kundennachfragen über ein Jahr\nMit Drift, Autoregression und Zufallsrauschen', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 365)
ax.set_ylim(0, 200)
ax.legend(loc='upper left', fontsize=12, framealpha=0.95)

ax.text(10, 180, f'100 Kundengruppen\nDrift: {drift_rate*100:.1f}% pro Tag\nAR-Koeff: {ar_coeff:.2f}\nRauschen: ±{noise_std} Einheiten',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_customer_trajectories.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 3 saved: fig_customer_trajectories.pdf (HAUPTPLOT!)")
plt.close()

# ============================================================
# PDF 4: Degradation nach Horizont
# ============================================================
horizons = np.array([1, 7, 14, 30, 60, 90, 180, 365])
relative_errors = np.zeros(len(horizons))
absolute_errors = np.zeros(len(horizons))

for h_idx, horizon in enumerate(horizons):
    if horizon <= num_days:
        errors = []
        for t0 in range(0, num_days - horizon, 7):
            t1 = t0 + horizon
            forecast = daily_mean[t0]
            actual = daily_mean[t1]
            error = np.abs(actual - forecast) / forecast * 100
            errors.append(error)
        
        relative_errors[h_idx] = np.mean(errors)
        absolute_errors[h_idx] = np.mean([daily_std[min(t0+horizon, num_days-1)] 
                                          for t0 in range(0, num_days - horizon, 7)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.bar(np.arange(len(horizons)), relative_errors, color='steelblue', alpha=0.7, width=0.6)
ax1.set_xticks(np.arange(len(horizons)))
ax1.set_xticklabels([f'{h}d' if h <= 30 else f'{h//7}w' if h <= 90 else f'{h//30}m' 
                      for h in horizons], fontsize=10)
ax1.set_xlabel('Prognose-Zeithorizont', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mittlere absolute % Fehler (MAPE)', fontsize=12, fontweight='bold')
ax1.set_title('Relative Fehlerquote nach Zeithorizont\nNaïve Prognose: "Heute = Morgen"', 
              fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_ylim(0, max(relative_errors) * 1.2)

for i, (h, err) in enumerate(zip(horizons, relative_errors)):
    if err < 10:
        color = 'green'
        label = 'OK'
    elif err < 20:
        color = 'yellow'
        label = 'Marginal'
    else:
        color = 'red'
        label = 'Kritisch'
    
    ax1.text(i, err + 0.5, f'{err:.1f}%\n{label}', ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

ax2.plot(horizons, absolute_errors, 'o-', linewidth=2.5, markersize=8, color='darkred')
ax2.fill_between(horizons, absolute_errors, alpha=0.2, color='red')
ax2.set_xlabel('Prognose-Zeithorizont (Tage)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mittlere Prognose-Fehlerstandard (Einheiten)', fontsize=12, fontweight='bold')
ax2.set_title('Absolute Unsicherheit nach Zeithorizont\nWie breit werden die Konfidenzintervalle?', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

ax2.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Akzeptable Unsicherheit')
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('fig_degradation_by_horizon.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 4 saved: fig_degradation_by_horizon.pdf")
plt.close()

# ============================================================
# PDF 5: Empirische Degradation
# ============================================================
sigma_0 = daily_std[0]
degradation_empirical = np.zeros(num_days)

for t in range(num_days):
    degradation_empirical[t] = daily_std[t] / sigma_0 if sigma_0 > 0 else 1

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(days, degradation_empirical, 'ko', markersize=3, alpha=0.5, label='Empirische Daten')

ax.set_xlabel('Zeithorizont τ (Tage)', fontsize=13, fontweight='bold')
ax.set_ylabel(r'Degradation $\delta(\tau) = \sigma(\tau) / \sigma(0)$', fontsize=13, fontweight='bold')
ax.set_title('Empirische Degradationskurve aus simulierten Kundendaten\nWie degradiert die Vorhersagbarkeit in diesem System?', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 365)
ax.set_ylim(0.95, max(degradation_empirical[100:]) * 1.05)
ax.legend(loc='upper left', fontsize=12, framealpha=0.95)

if len(np.where(degradation_empirical >= 1.5)[0]) > 0:
    critical_day = np.where(degradation_empirical >= 1.5)[0][0]
    ax.axvline(x=critical_day, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(critical_day, 1.0, f'  Kritisch bei\n  Tag {critical_day}', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_empirical_degradation.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 5 saved: fig_empirical_degradation.pdf")
plt.close()

# ============================================================
# PDF 6: Sicherheitsbestand
# ============================================================
tau_range = np.linspace(0, 365, 100)
sigma_0 = 10
z_alpha = 1.645

def degradation_exp(tau):
    return 1 + (2.5 - 1) * (1 - np.exp(-0.01 * tau))

def degradation_power(tau):
    return 1 + 0.005 * (tau ** 0.7)

def degradation_log(tau):
    return 1 + 0.15 * np.log(1 + tau)

safety_stock_exp = z_alpha * sigma_0 * np.sqrt(degradation_exp(tau_range))
safety_stock_power = z_alpha * sigma_0 * np.sqrt(degradation_power(tau_range))
safety_stock_log = z_alpha * sigma_0 * np.sqrt(degradation_log(tau_range))

holding_cost = 0.5
safety_stock_costs_exp = holding_cost * safety_stock_exp * tau_range
safety_stock_costs_power = holding_cost * safety_stock_power * tau_range
safety_stock_costs_log = holding_cost * safety_stock_log * tau_range

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.plot(tau_range, safety_stock_exp, 'b-', linewidth=2.5, label='Exponentiell', alpha=0.8)
ax1.plot(tau_range, safety_stock_power, 'r-', linewidth=2.5, label='Potenzgesetz', alpha=0.8)
ax1.plot(tau_range, safety_stock_log, 'g-', linewidth=2.5, label='Logarithmisch', alpha=0.8)
ax1.axhline(y=z_alpha * sigma_0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, 
            label='Baseline (Day 0)')

ax1.set_xlabel('Planungs-Zeithorizont τ (Tage)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Optimaler Sicherheitsbestand (Einheiten)', fontsize=12, fontweight='bold')
ax1.set_title('Sicherheitsbestand vs. Zeithorizont\nWie viel lagern wir bei längerer Planung?', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)
ax1.set_xlim(0, 365)

ax1.text(300, z_alpha*sigma_0*1.3, 'Tag 0:', fontsize=10, color='gray')
ax1.text(300, z_alpha*sigma_0*np.sqrt(degradation_exp(365))-1, 
         f'Exp: {safety_stock_exp[-1]:.1f}', fontsize=10, color='blue',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax1.text(300, z_alpha*sigma_0*np.sqrt(degradation_power(365))+1, 
         f'Power: {safety_stock_power[-1]:.1f}', fontsize=10, color='red',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax2.plot(tau_range, safety_stock_costs_exp, 'b-', linewidth=2.5, label='Exponentiell', alpha=0.8)
ax2.plot(tau_range, safety_stock_costs_power, 'r-', linewidth=2.5, label='Potenzgesetz', alpha=0.8)
ax2.plot(tau_range, safety_stock_costs_log, 'g-', linewidth=2.5, label='Logarithmisch', alpha=0.8)

ax2.set_xlabel('Planungs-Zeithorizont τ (Tage)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kumulierte Lagerhaltungskosten (Euro)', fontsize=12, fontweight='bold')
ax2.set_title('Lagerkosten über den Zeithorizont\nWie teuer wird lange Planung?', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=11)
ax2.set_xlim(0, 365)

plt.tight_layout()
plt.savefig('fig_safety_stock.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 6 saved: fig_safety_stock.pdf")
plt.close()

# ============================================================
# PDF 7: Zentral vs. Dezentral
# ============================================================
num_customer_groups = 50
degradation_heterogeneous = np.array([degradation_exp(tau_range) * (1 + 0.2*(np.random.rand()-0.5)) 
                                      for _ in range(num_customer_groups)])

sigma_per_group = 100
degradation_central = np.zeros(len(tau_range))
degradation_local = np.zeros(len(tau_range))

for t_idx, tau in enumerate(tau_range):
    central_var_sq = 0
    local_var_sum = 0
    
    for group_idx in range(num_customer_groups):
        sigma_group_t = sigma_per_group * np.sqrt(degradation_heterogeneous[group_idx, t_idx])
        central_var_sq += sigma_group_t**2
        local_var_sum += sigma_group_t
    
    degradation_central[t_idx] = np.sqrt(central_var_sq) / (sigma_per_group * num_customer_groups)
    degradation_local[t_idx] = (local_var_sum / num_customer_groups) / sigma_per_group

error_rate_central = degradation_central * 10
error_rate_local = degradation_local * 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.plot(tau_range, error_rate_central, 'r-', linewidth=2.5, label='Zentrale Planung', alpha=0.8)
ax1.plot(tau_range, error_rate_local, 'b-', linewidth=2.5, label='Dezentralisierte Planung', alpha=0.8)

ax1.axhspan(0, 15, alpha=0.1, color='green', label='Akzeptabel')
ax1.axhspan(15, 20, alpha=0.1, color='yellow', label='Kritisch')
ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.6)

ax1.set_xlabel('Zeithorizont τ (Tage)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Durchschn. Fehlerquote (%)', fontsize=12, fontweight='bold')
ax1.set_title('Fehlerquoten: Zentral vs. Dezentral\nWer kann länger planen?', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 365)
ax1.set_ylim(8, 25)
ax1.legend(loc='upper left', fontsize=11)

central_critical = np.where(error_rate_central >= 15)[0]
local_critical = np.where(error_rate_local >= 15)[0]

if len(central_critical) > 0:
    tau_central = tau_range[central_critical[0]]
    ax1.plot(tau_central, 15, 'ro', markersize=10)
    ax1.text(tau_central, 16, f'  {tau_central:.0f}d', fontsize=10, color='red')

if len(local_critical) > 0:
    tau_local = tau_range[local_critical[0]]
    ax1.plot(tau_local, 15, 'bo', markersize=10)
    ax1.text(tau_local, 14, f'{tau_local:.0f}d  ', fontsize=10, color='blue', ha='right')

cumulative_error_central = np.cumsum(error_rate_central)
cumulative_error_local = np.cumsum(error_rate_local)

ax2.plot(tau_range, cumulative_error_central, 'r-', linewidth=2.5, label='Zentral', alpha=0.8)
ax2.plot(tau_range, cumulative_error_local, 'b-', linewidth=2.5, label='Dezentral', alpha=0.8)
ax2.fill_between(tau_range, cumulative_error_central, cumulative_error_local, 
                 alpha=0.2, color='green', label='Dezentraler Vorteil')

ax2.set_xlabel('Zeithorizont τ (Tage)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kumulierter Fehler (Summe %)', fontsize=12, fontweight='bold')
ax2.set_title('Kumulierter Fehler: Dezentral ist akkumulativ besser', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=11)
ax2.set_xlim(0, 365)

plt.tight_layout()
plt.savefig('fig_central_vs_local.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 7 saved: fig_central_vs_local.pdf")
plt.close()

# ============================================================
# PDF 8: Planungs-Matrix
# ============================================================
horizons_matrix = [1, 7, 14, 30, 60, 90, 180, 365]
decisions = []

for horizon in horizons_matrix:
    idx = int(horizon * (len(tau_range) - 1) / 365)
    error_c = error_rate_central[idx]
    error_l = error_rate_local[idx]
    
    decisions.append((horizon, error_c, error_l))

fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(decisions))
bars1 = ax.bar(x_pos - 0.175, [d[1] for d in decisions], width=0.35, label='Zentral', 
               color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + 0.175, [d[2] for d in decisions], width=0.35, label='Dezentral',
               color='coral', alpha=0.7)

ax.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Akzeptabel')
ax.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Kritisch')

ax.set_xlabel('Zeithorizont', fontsize=12, fontweight='bold')
ax.set_ylabel('Fehlerquote (%)', fontsize=12, fontweight='bold')
ax.set_title('Planungs-Entscheidungsmatrix: Zentral vs. Dezentral\nWelche Struktur für welchen Zeithorizont?', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{h}d' if h <= 30 else f'{h//7}w' if h <= 90 else f'{h//30}m' 
                     for h, _, _ in decisions], fontsize=11)
ax.set_ylim(0, 25)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('fig_planning_matrix.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✓ PDF 8 saved: fig_planning_matrix.pdf")
plt.close()

# ============================================================
# Zusammenfassung
# ============================================================
print("\n" + "="*70)
print("ALLE PLOTS ALS PDF GENERIERT!")
print("="*70)
print("\nPDFs für LaTeX:")
print("  1. fig_degradation_comparison.pdf   - Vergleich aller 4 Degradationsmodelle")
print("  2. fig_error_rates.pdf              - Fehlerraten in %")
print("  3. fig_customer_trajectories.pdf    - HAUPTPLOT: 100 Kundentrajektorien")
print("  4. fig_degradation_by_horizon.pdf   - Fehler nach Zeithorizont")
print("  5. fig_empirical_degradation.pdf    - Empirische Degradationskurve")
print("  6. fig_safety_stock.pdf             - Sicherheitsbestand & Kosten")
print("  7. fig_central_vs_local.pdf         - Zentral vs. Dezentral")
print("  8. fig_planning_matrix.pdf          - Entscheidungsmatrix")

print("\n✅ Alle PDFs sind vektorbasiert und von hoher Qualität!")
print("✅ Können direkt in LaTeX mit \\includegraphics eingefügt werden!")
print("✅ Sind skaliersam ohne Qualitätsverlust!")

print("\nLaTeX-Code zum Einfügen (Beispiel):")
print("""
\\begin{figure}[h]
  \\centering
  \\includegraphics[width=\\textwidth]{fig_customer_trajectories.pdf}
  \\caption{Abbildung 4: Simulation von 100 Kundennachfragen über ein Jahr}
  \\label{fig:customer_trajectories}
\\end{figure}
""")

print("\n" + "="*70)