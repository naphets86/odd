import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
import os

os.makedirs('plots', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# Plot 1: Reifendruckabweichung - manuell vs. automatisch
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

np.random.seed(42)
t = np.linspace(0, 30, 300)  # Tage

# Manuelle Befüllung: 4 Räder mit Drift und seltenen Korrekturen
manual_pressures = []
labels = ['Vorderlinks (VL)', 'Vorderrechts (VR)', 'Hinterlinks (HL)', 'Hinterrechts (HR)']
target = 2.4  # bar
colors_m = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

for i in range(4):
    base = target + np.random.uniform(-0.15, 0.15)
    drift = -0.008 * t + np.random.normal(0, 0.04, len(t))
    corrections = np.zeros(len(t))
    for j in range(3):
        idx = np.random.randint(50, 250)
        corrections[idx:] += np.random.uniform(0.1, 0.25)
    manual_pressures.append(base + drift + corrections + np.random.normal(0, 0.02, len(t)))

ax1 = axes[0]
for i, (p, label, col) in enumerate(zip(manual_pressures, labels, colors_m)):
    ax1.plot(t, p, label=label, color=col, alpha=0.85, linewidth=1.3)
ax1.axhline(y=target, color='black', linestyle='--', linewidth=1.5, label=f'Sollwert ({target} bar)')
ax1.fill_between(t, target - 0.1, target + 0.1, alpha=0.1, color='green', label='Toleranzband ±0,1 bar')
ax1.set_xlabel('Zeit [Tage]')
ax1.set_ylabel('Reifendruck [bar]')
ax1.set_title('(a) Manuelles Befüllsystem\n(4 getrennte Ventile)')
ax1.legend(fontsize=8.5, loc='lower left')
ax1.set_ylim(1.9, 2.85)
ax1.set_xlim(0, 30)

# Automatisches System: alle 4 Räder nahe am Sollwert
ax2 = axes[1]
for i, (label, col) in enumerate(zip(labels, colors_m)):
    auto = target + np.random.normal(0, 0.015, len(t)) + 0.003 * np.sin(2 * np.pi * t / 7)
    ax2.plot(t, auto, label=label, color=col, alpha=0.85, linewidth=1.3)
ax2.axhline(y=target, color='black', linestyle='--', linewidth=1.5, label=f'Sollwert ({target} bar)')
ax2.fill_between(t, target - 0.1, target + 0.1, alpha=0.1, color='green', label='Toleranzband ±0,1 bar')
ax2.set_xlabel('Zeit [Tage]')
ax2.set_ylabel('Reifendruck [bar]')
ax2.set_title('(b) Automatisches Einventil-System\n(Steuergerät-geregelt)')
ax2.legend(fontsize=8.5, loc='lower left')
ax2.set_ylim(1.9, 2.85)
ax2.set_xlim(0, 30)

plt.suptitle('Reifendruckverlauf: Manuell vs. Automatisch', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot1_druckverlauf.pdf', bbox_inches='tight')
plt.savefig('plots/plot1_druckverlauf.png', bbox_inches='tight')
plt.close()
print("Plot 1 done")

# ============================================================
# Plot 2: Reifenabnutzung über Zeit
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t_km = np.linspace(0, 50000, 500)  # km

# Verschleiß-Modell: W(p, t) = W0 * exp(alpha * |p - p0|) * t^beta
def wear(p_deviation, t, W0=1.0, alpha=3.5, beta=1.1):
    return W0 * np.exp(alpha * np.abs(p_deviation)) * (t / 50000) ** beta

# Manuell: unterschiedliche Abweichungen pro Rad
deviations_m = [0.18, 0.12, 0.22, 0.08]  # bar Abweichung vom Sollwert
ax1 = axes[0]
wear_manual = []
for dev, label, col in zip(deviations_m, labels, colors_m):
    w = wear(dev, t_km)
    wear_manual.append(w)
    ax1.plot(t_km / 1000, w * 8, label=f'{label} (Δp={dev:.2f} bar)', color=col, linewidth=1.8)
ax1.set_xlabel('Kilometerstand [×10³ km]')
ax1.set_ylabel('Profilabnutzung [mm]')
ax1.set_title('(a) Reifenabnutzung: Manuelles System')
ax1.legend(fontsize=9)
ax1.axhline(y=1.6, color='red', linestyle=':', linewidth=1.5, label='Mindestprofiltiefe 1,6 mm')
ax1.set_ylim(0, 9)

# Automatisch: alle nahe Sollwert
deviations_a = [0.015, 0.012, 0.018, 0.010]
ax2 = axes[1]
for dev, label, col in zip(deviations_a, labels, colors_m):
    w = wear(dev, t_km)
    ax2.plot(t_km / 1000, w * 8, label=f'{label} (Δp={dev:.3f} bar)', color=col, linewidth=1.8)
ax2.set_xlabel('Kilometerstand [×10³ km]')
ax2.set_ylabel('Profilabnutzung [mm]')
ax2.set_title('(b) Reifenabnutzung: Automatisches System')
ax2.legend(fontsize=9)
ax2.axhline(y=1.6, color='red', linestyle=':', linewidth=1.5, label='Mindestprofiltiefe 1,6 mm')
ax2.set_ylim(0, 9)

plt.suptitle('Vergleich der Reifenabnutzung in Abhängigkeit des Drucks', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot2_abnutzung.pdf', bbox_inches='tight')
plt.savefig('plots/plot2_abnutzung.png', bbox_inches='tight')
plt.close()
print("Plot 2 done")

# ============================================================
# Plot 3: Standardabweichung des Reifendrucks – Boxplot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

np.random.seed(123)
data_manual = [np.random.normal(2.4 + np.random.uniform(-0.2, 0.2), 0.12, 200) for _ in range(4)]
data_auto   = [np.random.normal(2.4, 0.018, 200) for _ in range(4)]

positions_m = [1, 2, 3, 4]
positions_a = [6, 7, 8, 9]

bp1 = ax.boxplot(data_manual, positions=positions_m, widths=0.6,
                 patch_artist=True, notch=True,
                 boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                 medianprops=dict(color='black', linewidth=2))

bp2 = ax.boxplot(data_auto, positions=positions_a, widths=0.6,
                 patch_artist=True, notch=True,
                 boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                 medianprops=dict(color='black', linewidth=2))

ax.axhline(y=2.4, color='navy', linestyle='--', linewidth=1.5, label='Sollwert 2,4 bar')
ax.set_xticks([1,2,3,4, 6,7,8,9])
ax.set_xticklabels(['VL','VR','HL','HR','VL','VR','HL','HR'])
ax.set_ylabel('Reifendruck [bar]')
ax.set_title('Statistische Druckverteilung pro Rad: Manuell vs. Automatisch', fontweight='bold')

red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Manuelles System')
green_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Automatisches System')
ax.legend(handles=[red_patch, green_patch, plt.Line2D([0],[0],color='navy',linestyle='--',label='Sollwert 2,4 bar')],
          fontsize=10)

ax.text(2.5, 2.78, 'Manuell', ha='center', fontsize=12, color='#c0392b', fontweight='bold')
ax.text(7.5, 2.78, 'Automatisch', ha='center', fontsize=12, color='#27ae60', fontweight='bold')
ax.set_ylim(1.9, 2.85)

plt.tight_layout()
plt.savefig('plots/plot3_boxplot.pdf', bbox_inches='tight')
plt.savefig('plots/plot3_boxplot.png', bbox_inches='tight')
plt.close()
print("Plot 3 done")

# ============================================================
# Plot 4: Lebensdauer-Verlängerung & Kosteneinsparung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Lebensdauer als Funktion der Druckabweichung
dp = np.linspace(0, 0.4, 400)
# Modell: L = L0 * exp(-k * dp^2)
L0 = 60000  # km Nennlebensdauer
k = 12
life = L0 * np.exp(-k * dp**2)

ax1 = axes[0]
ax1.plot(dp * 100, life / 1000, color='#2980b9', linewidth=2.5)
ax1.axvline(x=15, color='#e74c3c', linestyle='--', linewidth=1.8, label='Manuell: Ø Abw. ~15%')
ax1.axvline(x=1.5, color='#27ae60', linestyle='--', linewidth=1.8, label='Automatisch: Ø Abw. ~1,5%')
ax1.fill_between(dp * 100, life / 1000, alpha=0.12, color='#2980b9')
ax1.set_xlabel('Druckabweichung vom Sollwert [%]')
ax1.set_ylabel('Reifenlebensdauer [×10³ km]')
ax1.set_title('(a) Reifenlebensdauer vs. Druckabweichung')
ax1.legend(fontsize=9)

life_manual = L0 * np.exp(-k * 0.15**2)
life_auto   = L0 * np.exp(-k * 0.015**2)
ax1.annotate(f'{life_manual/1000:.1f}k km', xy=(15, life_manual/1000),
             xytext=(22, life_manual/1000 + 3), fontsize=9, color='#e74c3c',
             arrowprops=dict(arrowstyle='->', color='#e74c3c'))
ax1.annotate(f'{life_auto/1000:.1f}k km', xy=(1.5, life_auto/1000),
             xytext=(8, life_auto/1000 - 4), fontsize=9, color='#27ae60',
             arrowprops=dict(arrowstyle='->', color='#27ae60'))

# Kosteneinsparung über Jahre
ax2 = axes[1]
years = np.arange(1, 11)
cost_per_tire = 120  # €
tires_per_set = 4
sets_manual = years * (50000 / life_manual)  # Reifensätze
sets_auto   = years * (50000 / life_auto)
cost_manual = sets_manual * cost_per_tire * tires_per_set
cost_auto   = sets_auto   * cost_per_tire * tires_per_set
savings = cost_manual - cost_auto

ax2.bar(years - 0.2, cost_manual, 0.4, label='Manuell', color='#e74c3c', alpha=0.8)
ax2.bar(years + 0.2, cost_auto,   0.4, label='Automatisch', color='#27ae60', alpha=0.8)
ax2.set_xlabel('Jahr')
ax2.set_ylabel('Kumulative Reifenkosten [€]')
ax2.set_title('(b) Kumulative Reifenkosten über 10 Jahre\n(Fahrleistung 50.000 km/Jahr)')
ax2.legend(fontsize=10)

plt.suptitle('Ökonomische Analyse: Lebensdauer und Kosteneinsparung', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot4_kosten.pdf', bbox_inches='tight')
plt.savefig('plots/plot4_kosten.png', bbox_inches='tight')
plt.close()
print("Plot 4 done")

# ============================================================
# Plot 5: Regelkreis-Sprungantwort des Steuergeräts
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t_ctrl = np.linspace(0, 10, 1000)  # Sekunden

# PT2-Regelstrecke mit PID-Regler
# Gedämpfte Schwingung gegen Sollwert
omega_n = 3.0
zeta = 0.7
p_soll = 2.4

def step_response_pt2(t, p0, p_target, omega_n, zeta):
    delta_p = p_target - p0
    if zeta < 1:
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        resp = delta_p * (1 - np.exp(-zeta * omega_n * t) *
               (np.cos(omega_d * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t)))
    else:
        resp = delta_p * (1 - np.exp(-omega_n * t) * (1 + omega_n * t))
    return p0 + resp

ax1 = axes[0]
# Verschiedene Startdrücke
starts = [1.8, 2.0, 2.2, 2.6, 2.8]
cols_step = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']
for p0, col in zip(starts, cols_step):
    resp = step_response_pt2(t_ctrl, p0, p_soll, omega_n, zeta)
    ax1.plot(t_ctrl, resp, color=col, linewidth=1.8, label=f'p₀ = {p0} bar')
ax1.axhline(y=p_soll, color='black', linestyle='--', linewidth=1.5)
ax1.fill_between(t_ctrl, p_soll * 0.98, p_soll * 1.02, alpha=0.15, color='green', label='2%-Band')
ax1.set_xlabel('Zeit [s]')
ax1.set_ylabel('Reifendruck [bar]')
ax1.set_title('(a) Sprungantworten des Druckregelkreises\n(verschiedene Anfangsdrücke)')
ax1.legend(fontsize=9)
ax1.set_ylim(1.6, 3.1)

# Regelabweichung und Stellgröße
ax2 = axes[1]
p0 = 2.0
resp = step_response_pt2(t_ctrl, p0, p_soll, omega_n, zeta)
e = p_soll - resp  # Regelabweichung

# Stellgröße (proportional zur Ableitung der Sprungantwort)
u = np.gradient(resp, t_ctrl) * 5 + e * 2
u = np.clip(u, 0, 2.5)

ax2.plot(t_ctrl, e, color='#e74c3c', linewidth=1.8, label='Regelabweichung e(t)')
ax2.plot(t_ctrl, u, color='#2980b9', linewidth=1.8, label='Stellgröße u(t) [normiert]')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_xlabel('Zeit [s]')
ax2.set_ylabel('Amplitude')
ax2.set_title('(b) Regelabweichung und Stellgröße\n(Startdruck 2,0 bar → Sollwert 2,4 bar)')
ax2.legend(fontsize=10)

plt.suptitle('Dynamisches Verhalten des automatischen Druckregelkreises', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot5_regelkreis.pdf', bbox_inches='tight')
plt.savefig('plots/plot5_regelkreis.png', bbox_inches='tight')
plt.close()
print("Plot 5 done")

# ============================================================
# Plot 6: Gleichmäßigkeit der Abnutzung - Radar/Polar + Heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Heatmap der Druckabweichungen über 12 Monate
months = ['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez']
np.random.seed(77)
manual_dev = np.abs(np.random.normal(0, 0.15, (4, 12)))
auto_dev   = np.abs(np.random.normal(0, 0.015, (4, 12)))

combined = np.concatenate([manual_dev, auto_dev], axis=1)
wheel_labels = ['VL','VR','HL','HR']

ax1 = axes[0]
im = ax1.imshow(manual_dev, cmap='Reds', aspect='auto', vmin=0, vmax=0.35)
ax1.set_xticks(range(12))
ax1.set_xticklabels(months, fontsize=9)
ax1.set_yticks(range(4))
ax1.set_yticklabels(wheel_labels)
ax1.set_title('(a) Manuell: |Druckabweichung| [bar]')
plt.colorbar(im, ax=ax1, fraction=0.046)
for i in range(4):
    for j in range(12):
        ax1.text(j, i, f'{manual_dev[i,j]:.2f}', ha='center', va='center', fontsize=7, color='black')

ax2 = axes[1]
im2 = ax2.imshow(auto_dev, cmap='Greens', aspect='auto', vmin=0, vmax=0.035)
ax2.set_xticks(range(12))
ax2.set_xticklabels(months, fontsize=9)
ax2.set_yticks(range(4))
ax2.set_yticklabels(wheel_labels)
ax2.set_title('(b) Automatisch: |Druckabweichung| [bar]')
plt.colorbar(im2, ax=ax2, fraction=0.046)
for i in range(4):
    for j in range(12):
        ax2.text(j, i, f'{auto_dev[i,j]:.3f}', ha='center', va='center', fontsize=7, color='black')

plt.suptitle('Heatmap der monatlichen Druckabweichungen pro Rad', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot6_heatmap.pdf', bbox_inches='tight')
plt.savefig('plots/plot6_heatmap.png', bbox_inches='tight')
plt.close()
print("Plot 6 done")

# ============================================================
# Plot 7: Kraftstoffverbrauch und CO2-Reduktion
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

dp_pct = np.linspace(0, 25, 300)  # % Druckabweichung

# Kraftstoffmehrverbrauch: ~0,2% pro 10 kPa = 0,1 bar Unterdrückung
fuel_increase = 0.3 * dp_pct / 10  # % Mehrverbrauch
base_fuel = 7.5  # L/100km
fuel_total = base_fuel * (1 + fuel_increase / 100)
co2_per_liter = 2.31  # kg CO2 per liter petrol
co2_increase = (fuel_total - base_fuel) * co2_per_liter

ax1 = axes[0]
ax1.plot(dp_pct, fuel_total, color='#e67e22', linewidth=2.5, label='Kraftstoffverbrauch')
ax1.axvline(x=15, color='#e74c3c', linestyle='--', linewidth=1.8, label='Manuell: Ø ~15% Abw.')
ax1.axvline(x=1.5, color='#27ae60', linestyle='--', linewidth=1.8, label='Automatisch: Ø ~1,5% Abw.')
ax1.fill_between(dp_pct, base_fuel, fuel_total, alpha=0.2, color='orange')
ax1.set_xlabel('Druckabweichung [%]')
ax1.set_ylabel('Kraftstoffverbrauch [L/100 km]')
ax1.set_title('(a) Kraftstoffverbrauch vs. Druckabweichung')
ax1.legend(fontsize=9)

ax2 = axes[1]
ax2.plot(dp_pct, co2_increase, color='#7f8c8d', linewidth=2.5, label='CO₂-Mehrausstoß')
ax2.fill_between(dp_pct, 0, co2_increase, alpha=0.2, color='gray')
ax2.axvline(x=15, color='#e74c3c', linestyle='--', linewidth=1.8, label='Manuell')
ax2.axvline(x=1.5, color='#27ae60', linestyle='--', linewidth=1.8, label='Automatisch')
ax2.set_xlabel('Druckabweichung [%]')
ax2.set_ylabel('CO₂-Mehrausstoß [kg/100 km]')
ax2.set_title('(b) CO₂-Mehrausstoß vs. Druckabweichung')
ax2.legend(fontsize=9)

plt.suptitle('Umweltwirkung: Kraftstoffverbrauch und CO₂-Emissionen', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot7_umwelt.pdf', bbox_inches='tight')
plt.savefig('plots/plot7_umwelt.png', bbox_inches='tight')
plt.close()
print("Plot 7 done")

print("\nAlle Plots erfolgreich erstellt!")
print(os.listdir('plots'))
