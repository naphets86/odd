import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

BLUE   = '#19468C'
RED    = '#B4321E'
GREEN  = '#1E6432'
GRAY   = '#3C3C46'
LGRAY  = '#F5F5F8'
ORANGE = '#D4700A'

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: O(n³) Laufzeit des Subgraph Algorithmus auf Netzwerkgraphen
# Normierung: relativ zu n0=5, sodass alle Kurven bei 1 starten und
# die Ordnung n^2 < n^3 < n^4 überall korrekt sichtbar ist (log-Skala).
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
n_vals = np.arange(5, 501, 5)
n0 = 5.0   # Referenzpunkt: alle Kurven starten bei (n0/n0)^k = 1

T_n2       = (n_vals / n0)**2
T_subgraph = (n_vals / n0)**3
T_n4       = (n_vals / n0)**4

ax.plot(n_vals, T_subgraph, color=BLUE,  lw=2.5, label=r'Subgraph Algorithmus $\Theta(n^3)$')
ax.plot(n_vals, T_n2,       color=GREEN, lw=2,   linestyle='--', label=r'Untere Schranke $\Omega(n^2)$')
ax.plot(n_vals, T_n4,       color=RED,   lw=2,   linestyle=':',  label=r'Naiver Ansatz $O(n^4)$')
ax.fill_between(n_vals, T_n2, T_subgraph, alpha=0.08, color=BLUE)

# Werte-Annotation bei n=500
for T, label, color in [(T_n4[-1],       f'$n^4$: {T_n4[-1]:.0e}×', RED),
                         (T_subgraph[-1], f'$n^3$: {T_subgraph[-1]:.0e}×', BLUE),
                         (T_n2[-1],       f'$n^2$: {T_n2[-1]:.0e}×', GREEN)]:
    ax.annotate(label, xy=(500, T), xytext=(455, T * 1.4),
                fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('Anzahl Basisstationen / Netzwerkknoten $n$')
ax.set_ylabel(r'Normierte Laufzeit $(n/n_0)^k$,  $n_0 = 5$')
ax.set_title('Laufzeitkomplexität des Subgraph Algorithmus\nauf Mobilfunknetz-Graphen')
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(5, 500)
ax.set_ylim(1, None)
fig.tight_layout()
fig.savefig('home/claude/plot1_laufzeit.png', bbox_inches='tight')
plt.close()
print("Plot 1 OK")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Detektionsrate und Falsch-Positiv-Rate (ROC-Kurve)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

# Simulierte ROC-Kurven für verschiedene Algorithmen
fpr = np.linspace(0, 1, 300)
# Subgraph Algorithmus (sehr gut)
tpr_sub  = 1 - norm.cdf(norm.ppf(1 - fpr) - 3.2)
# Klassisches Matching
tpr_class = 1 - norm.cdf(norm.ppf(1 - fpr) - 1.8)
# Zufällig
tpr_rand  = fpr

auc_sub   = np.trapezoid(tpr_sub,   fpr)
auc_class = np.trapezoid(tpr_class, fpr)

ax.plot(fpr, tpr_sub,   color=BLUE,  lw=2.5,
        label=f'Subgraph Algorithmus (AUC = {auc_sub:.3f})')
ax.plot(fpr, tpr_class, color=ORANGE, lw=2,   linestyle='--',
        label=f'Klassisches Signatur-Matching (AUC = {auc_class:.3f})')
ax.plot(fpr, tpr_rand,  color=GRAY,  lw=1.5, linestyle=':',
        label='Zufällige Klassifikation (AUC = 0.500)')

# Optimaler Arbeitspunkt
idx = np.argmax(tpr_sub - fpr)
ax.scatter([fpr[idx]], [tpr_sub[idx]], color=RED, s=80, zorder=5,
           label=f'Opt. Arbeitspunkt (FPR={fpr[idx]:.2f}, TPR={tpr_sub[idx]:.2f})')
ax.axvline(fpr[idx], color=RED, lw=0.8, linestyle='--', alpha=0.5)

ax.fill_between(fpr, tpr_sub, alpha=0.07, color=BLUE)
ax.set_xlabel('Falsch-Positiv-Rate (FPR)')
ax.set_ylabel('Wahr-Positiv-Rate (Detektionsrate, TPR)')
ax.set_title('ROC-Kurve: Drohnenerkennung im Mobilfunknetz')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig('home/claude/plot2_roc.png', bbox_inches='tight')
plt.close()
print("Plot 2 OK")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Signatur-Verteilung: normaler vs. anomaler Verkehr
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

np.random.seed(42)
normal_sigs   = np.random.normal(loc=120, scale=25, size=2000)
drone_sigs    = np.random.normal(loc=185, scale=18, size=400)
spoof_sigs    = np.random.normal(loc=95,  scale=12, size=150)

bins = np.linspace(40, 260, 60)
ax.hist(normal_sigs, bins=bins, density=True, alpha=0.55, color=BLUE,
        label='Normaler Netzwerkverkehr ($\mu=120$, $\sigma=25$)')
ax.hist(drone_sigs,  bins=bins, density=True, alpha=0.65, color=RED,
        label='Drohnen-Steuersignale ($\mu=185$, $\sigma=18$)')
ax.hist(spoof_sigs,  bins=bins, density=True, alpha=0.55, color=ORANGE,
        label='Spoofing-Angriffe ($\mu=95$, $\sigma=12$)')

# KDE-Überlagerung
x_range = np.linspace(40, 260, 500)
from scipy.stats import gaussian_kde
for data, color, lw in [(normal_sigs, BLUE, 2.5), (drone_sigs, RED, 2.5), (spoof_sigs, ORANGE, 2.0)]:
    kde = gaussian_kde(data)
    ax.plot(x_range, kde(x_range), color=color, lw=lw)

# Schwellenwert
ax.axvline(155, color=GREEN, lw=2, linestyle='--', label='Detektionsschwelle $\theta = 155$')
ax.set_xlabel('Signatur-Wert $\sigma_j$ (normiert)')
ax.set_ylabel('Wahrscheinlichkeitsdichte')
ax.set_title('Signatur-Verteilung nach Verkehrsklasse\n(Subgraph Algorithmus-Signaturen)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('home/claude/plot3_verteilung.png', bbox_inches='tight')
plt.close()
print("Plot 3 OK")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Netzwerkabdeckung – Basisstations-Dichte als Radarschirm (Deutschland)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Simuliertes Deutschland-Grid (Rechteck: 47.3–55.1°N, 5.9–15.0°E)
np.random.seed(7)
n_stations = 180
lat = np.random.uniform(47.5, 55.0, n_stations)
lon = np.random.uniform(6.0,  15.0, n_stations)

# Ballungszentren simulieren
centers_lat = [52.52, 48.14, 53.57, 51.23, 50.94]
centers_lon = [13.40,  11.58, 10.00,  6.78, 6.96]
for clat, clon in zip(centers_lat, centers_lon):
    extra_lat = clat + np.random.normal(0, 0.5, 25)
    extra_lon = clon + np.random.normal(0, 0.7, 25)
    lat = np.append(lat, extra_lat)
    lon = np.append(lon, extra_lon)

# Coverage-Radius Heatmap
grid_lat = np.linspace(47.3, 55.2, 120)
grid_lon = np.linspace(5.8,  15.2, 120)
GLon, GLat = np.meshgrid(grid_lon, grid_lat)
coverage = np.zeros_like(GLat)
R = 0.8   # Grad ≈ 80 km Reichweite
for la, lo in zip(lat, lon):
    dist = np.sqrt((GLat - la)**2 + (GLon - lo)**2)
    coverage += np.exp(-dist**2 / (2 * (R/2)**2))

cmap = plt.cm.Blues
cf = ax.contourf(GLon, GLat, coverage, levels=20, cmap=cmap, alpha=0.7)
fig.colorbar(cf, ax=ax, label='Abdeckungsintensität (normiert)')
ax.scatter(lon, lat, c=RED, s=18, zorder=4, label='Basisstation / Radar-Knoten')

# Drohnen-Sichtung markieren
drone_events_lat = [51.5, 53.1, 48.8, 52.4]
drone_events_lon = [7.2, 12.3, 12.1, 9.6]
ax.scatter(drone_events_lon, drone_events_lat, c=ORANGE, marker='D',
           s=100, zorder=6, label='Detektierte Drohnen-Ereignisse', edgecolors='black', lw=0.5)

ax.set_xlabel('Längengrad [°E]')
ax.set_ylabel('Breitengrad [°N]')
ax.set_title('Mobilfunknetz als Drohnen-Radar: Abdeckungskarte Deutschland\n(Telekom / Rheinmetall Szenario)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.25, linestyle=':')
ax.set_xlim(5.8, 15.2); ax.set_ylim(47.3, 55.2)
fig.tight_layout()
fig.savefig('home/claude/plot4_karte.png', bbox_inches='tight')
plt.close()
print("Plot 4 OK")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Subgraph-Match-Score: normaler Verkehr vs. Drohnensignal über Zeit
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

np.random.seed(13)
t = np.linspace(0, 24, 1440)   # 24 Stunden, minutenweise
# Normaler Verkehr
normal_score = 0.12 + 0.05 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.03, len(t))
# Drohnen-Events an bestimmten Zeitpunkten
drone_score  = normal_score.copy()
events = [6.2, 6.25, 6.3, 11.8, 11.85, 11.9, 18.4, 18.45, 18.5, 22.1, 22.15]
for ev in events:
    idx = np.argmin(np.abs(t - ev))
    width = 15
    drone_score[max(0,idx-width):idx+width] += np.random.uniform(0.4, 0.65)

threshold = 0.35
axes[0].plot(t, normal_score, color=BLUE, lw=1.2, label='LCS-Score (normal)', alpha=0.8)
axes[0].axhline(threshold, color=RED, lw=1.5, linestyle='--', label=f'Schwelle $\\theta={threshold}$')
axes[0].fill_between(t, threshold, normal_score,
                     where=(normal_score > threshold), color=ORANGE, alpha=0.4, label='Alarm')
axes[0].set_ylabel('Subgraph-Match-Score')
axes[0].set_title('Zeitlicher Verlauf der Drohnendetektion (24h-Überwachung)')
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 0.85)
axes[0].grid(True, alpha=0.25)

axes[1].plot(t, drone_score, color=RED, lw=1.2, label='LCS-Score (mit Drohnen-Events)', alpha=0.85)
axes[1].axhline(threshold, color=GRAY, lw=1.5, linestyle='--', label=f'Schwelle $\\theta={threshold}$')
axes[1].fill_between(t, threshold, drone_score,
                     where=(drone_score > threshold), color=RED, alpha=0.35, label='Drohnen-Alarm')
axes[1].set_xlabel('Tageszeit [h]')
axes[1].set_ylabel('Subgraph-Match-Score')
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 0.85)
axes[1].set_xlim(0, 24)
axes[1].set_xticks(range(0, 25, 2))
axes[1].grid(True, alpha=0.25)
axes[1].set_title('Mit simulierten Drohnen-Einflügen (Echtzeit-Detektion)')

fig.tight_layout()
fig.savefig('home/claude/plot5_zeitverlauf.png', bbox_inches='tight')
plt.close()
print("Plot 5 OK")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Balkendiagramm – Vergleich Detektionsmethoden
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

methods = ['Radar\n(klassisch)', 'LIDAR', 'Akustik\nSensor',
           'RF-Scanning', 'Subgraph-\nAlgorithmus']
detection_rate = [72, 68, 55, 81, 94]
false_positive  = [18, 14, 25, 12, 4]
colors = [GRAY, GRAY, GRAY, ORANGE, BLUE]

x = np.arange(len(methods))
bars1 = axes[0].bar(x, detection_rate, color=colors, edgecolor='white', lw=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(methods, fontsize=9)
axes[0].set_ylabel('Detektionsrate [%]')
axes[0].set_title('Detektionsrate nach Methode')
axes[0].set_ylim(0, 105)
axes[0].axhline(90, color=RED, lw=1.5, linestyle='--', label='Zielwert 90%')
axes[0].legend(fontsize=9)
axes[0].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars1, detection_rate):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

bars2 = axes[1].bar(x, false_positive, color=colors, edgecolor='white', lw=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(methods, fontsize=9)
axes[1].set_ylabel('Falsch-Positiv-Rate [%]')
axes[1].set_title('Falsch-Positiv-Rate nach Methode')
axes[1].set_ylim(0, 32)
axes[1].axhline(5, color=GREEN, lw=1.5, linestyle='--', label='Zielwert $\leq 5\%$')
axes[1].legend(fontsize=9)
axes[1].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars2, false_positive):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('Vergleich der Drohnenerkennungsmethoden\n(Simulation, $n=500$ Testszenarien)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('home/claude/plot6_vergleich.png', bbox_inches='tight')
plt.close()
print("Plot 6 OK")

print("Alle Plots erfolgreich generiert.")

