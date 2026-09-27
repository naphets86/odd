#!/usr/bin/env python3
"""
Matplotlib-Plots für die wissenschaftliche Arbeit:
"Resilienz der Erde: Wiederherstellungsfähigkeit natürlicher Ökosysteme"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ----------------------------------------------------------------
# Plot 1: Waldregenerationsmodell (logistische Wachstumskurve)
# ----------------------------------------------------------------
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

t = np.linspace(0, 100, 500)
K_tropen  = 100.0
K_gemässigt = 85.0
K_boreal  = 70.0

y_tropen     = logistic(t, K_tropen,  0.10, 25)
y_gemässigt  = logistic(t, K_gemässigt, 0.07, 35)
y_boreal     = logistic(t, K_boreal,  0.05, 50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, y_tropen,    color='#1a7a3c', lw=2.0, label='Tropischer Regenwald')
ax.plot(t, y_gemässigt, color='#2e86c1', lw=2.0, label='Gemäßigter Laubwald')
ax.plot(t, y_boreal,    color='#7f8c8d', lw=2.0, label='Borealer Nadelwald')
ax.axhline(y=100, color='#e74c3c', lw=1.0, ls='--', label='Ursprüngliche Biomasse (100 %)')
ax.set_xlabel('Zeit nach Störung (Jahre)')
ax.set_ylabel('Biomasse (% des Ausgangswerts)')
ax.set_title('Plot 1 – Waldregeneration: Logistisches Wachstumsmodell')
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot1_waldregeneration.pdf')
plt.close()
print("Plot 1 gespeichert.")

# ----------------------------------------------------------------
# Plot 2: Ozonschicht-Regeneration 1980–2060 (Daten + Projektion)
# ----------------------------------------------------------------
years_obs  = np.array([1980,1985,1990,1995,2000,2005,2010,2015,2020,2023])
ozon_obs   = np.array([304, 294, 283, 275, 271, 273, 276, 279, 283, 287])
years_proj = np.arange(2023, 2065)
# Lineare Projektion (vereinfacht)
slope = (287 - 271) / (2023 - 2000)
ozon_proj = 287 + slope * (years_proj - 2023)
ozon_proj = np.minimum(ozon_proj, 305)  # Sättigungsobergrenze

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(years_obs, ozon_obs, 'o-', color='#2e86c1', lw=2, label='Beobachtete Werte (DU)')
ax.plot(years_proj, ozon_proj, '--', color='#e67e22', lw=2, label='Projektion (Trend)')
ax.axvline(x=1987, color='#8e44ad', lw=1.2, ls=':', label='Montrealer Protokoll (1987)')
ax.axvline(x=2023, color='gray', lw=1.0, ls='-.')
ax.fill_between(years_proj, ozon_proj - 5, ozon_proj + 5, alpha=0.15, color='#e67e22')
ax.set_xlabel('Jahr')
ax.set_ylabel('Gesamtozon (Dobson-Einheiten, DU)')
ax.set_title('Plot 2 – Regeneration der Ozonschicht (1980–2060)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot2_ozonschicht.pdf')
plt.close()
print("Plot 2 gespeichert.")

# ----------------------------------------------------------------
# Plot 3: Meeresbiodiversität – Korallenerholung nach Bleiche
# ----------------------------------------------------------------
t3 = np.linspace(0, 20, 300)
# Schäden + Erholung: exponentieller Abfall + logistische Erholung
def coral_recovery(t, d, k, r, L):
    return L * (1 - np.exp(-r * t)) + d * np.exp(-k * t)

y_mild    = coral_recovery(t3, 40, 0.8, 0.4, 80)
y_mittel  = coral_recovery(t3, 65, 0.5, 0.25, 65)
y_schwer  = coral_recovery(t3, 85, 0.3, 0.12, 45)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t3, y_mild,   color='#27ae60', lw=2, label='Milde Bleiche')
ax.plot(t3, y_mittel, color='#f39c12', lw=2, label='Mittlere Bleiche')
ax.plot(t3, y_schwer, color='#e74c3c', lw=2, label='Schwere Bleiche')
ax.axhline(y=80, color='#27ae60', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Jahre nach Bleichereignis')
ax.set_ylabel('Korallenbedeckung (% Ausgangswert)')
ax.set_title('Plot 3 – Korallenerholung nach Massenbleiche')
ax.legend()
ax.set_xlim(0, 20)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot3_korallenerholung.pdf')
plt.close()
print("Plot 3 gespeichert.")

# ----------------------------------------------------------------
# Plot 4: Tierbestandserholung – Wölfe im Yellowstone
# ----------------------------------------------------------------
jahre = np.arange(1995, 2024)
woelfe = np.array([14, 22, 40, 65, 90, 119, 148, 173, 218, 252, 301,
                   325, 340, 361, 382, 397, 410, 423, 430, 437, 440,
                   445, 448, 450, 452, 453, 455, 456, 458])

# Fitgerade (logistisch)
t4 = np.linspace(0, len(jahre)-1, 500)
def wolf_model(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

popt, _ = curve_fit(wolf_model, np.arange(len(jahre)), woelfe, p0=[460, 0.3, 10], maxfev=5000)
y_fit = wolf_model(t4, *popt)
jahre_fit = 1995 + t4

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(jahre, woelfe, color='#5d6d7e', alpha=0.7, label='Wolfspopulation (Beobachtung)')
ax.plot(jahre_fit, y_fit, color='#e74c3c', lw=2, label=f'Logistisches Modell (K={popt[0]:.0f})')
ax.set_xlabel('Jahr')
ax.set_ylabel('Anzahl Wölfe')
ax.set_title('Plot 4 – Wolfspopulation im Yellowstone (seit Wiederansiedlung 1995)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('plot4_woelfe_yellowstone.pdf')
plt.close()
print("Plot 4 gespeichert.")

# ----------------------------------------------------------------
# Plot 5: Trophische Kaskade – Ökosystemeffekte
# ----------------------------------------------------------------
categories = ['Wölfe\n(Raubtiere)', 'Elche\n(Herbivoren)', 'Weidenfläche\n(%)', 
              'Flussbiber\n(Keystone)', 'Fischbestand\n(Flüsse)', 'Vogelarten\n(Diversität)']
vor_wölfen = [0, 100, 25, 30, 55, 70]
nach_wölfen = [100, 50, 80, 85, 90, 110]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, vor_wölfen,  width, label='Vor Wiederansiedlung (1994)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, nach_wölfen, width, label='Nach Wiederansiedlung (2023)', color='#27ae60', alpha=0.8)
ax.set_ylabel('Relativer Index (Ausgangswert = 100)')
ax.set_title('Plot 5 – Trophische Kaskade im Yellowstone-Ökosystem')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('plot5_trophische_kaskade.pdf')
plt.close()
print("Plot 5 gespeichert.")

# ----------------------------------------------------------------
# Plot 6: Meeresspiegelanstieg vs. Mangrovenbestand (Schutzwirkung)
# ----------------------------------------------------------------
kuestenlaenge = np.linspace(0, 100, 200)  # % Mangrovenbedeckung
erosion_rate = 50 * np.exp(-0.04 * kuestenlaenge) + 2

# Punkte: verschiedene Regionen
regionen = {
    'Bangladesch': (12, 38),
    'Indonesien': (35, 20),
    'Brasilien': (55, 14),
    'Philippinen': (8, 44),
    'Mexiko-Golf': (28, 25),
    'Australien': (60, 11),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kuestenlaenge, erosion_rate, color='#2e86c1', lw=2, label='Erosionsrate (Modell)')
for name, (x_val, y_val) in regionen.items():
    ax.scatter(x_val, y_val, zorder=5, s=60)
    ax.annotate(name, (x_val, y_val), textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xlabel('Mangrovenbedeckung der Küstenlinie (%)')
ax.set_ylabel('Küstenerosionsrate (cm/Jahr)')
ax.set_title('Plot 6 – Mangrovenbestand und Küstenerosion')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot6_mangroven_erosion.pdf')
plt.close()
print("Plot 6 gespeichert.")

# ----------------------------------------------------------------
# Plot 7: Vergleich Regenerationszeiten verschiedener Ökosysteme
# ----------------------------------------------------------------
oekosysteme = [
    'Tropischer\nRegenwald',
    'Gemäßigter\nLaubwald',
    'Borealer\nNadelwald',
    'Korallenriff\n(mild)',
    'Korallenriff\n(schwer)',
    'Steppe /\nSavanne',
    'Meereszone\n(Fischerei)',
    'Feuchtgebiet',
]
min_jahre = [40,  25,  60,  10, 25,  5, 10,  5]
max_jahre = [150, 80, 200,  25, 80, 20, 30, 20]

y_pos = np.arange(len(oekosysteme))
fig, ax = plt.subplots(figsize=(9, 6))
for i, (lo, hi) in enumerate(zip(min_jahre, max_jahre)):
    ax.barh(i, hi - lo, left=lo, height=0.5, color='#2e86c1', alpha=0.75)
    ax.text(hi + 2, i, f'{lo}–{hi} J.', va='center', fontsize=8.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(oekosysteme)
ax.set_xlabel('Regenerationsdauer (Jahre)')
ax.set_title('Plot 7 – Regenerationsdauer verschiedener Ökosysteme')
ax.set_xlim(0, 230)
ax.grid(True, alpha=0.3, axis='x')
fig.tight_layout()
fig.savefig('plot7_regenerationszeiten.pdf')
plt.close()
print("Plot 7 gespeichert.")

# ----------------------------------------------------------------
# Plot 8: Subgraph Algorithmus – Resilienznetzwerk der Biodiversität
# ----------------------------------------------------------------
np.random.seed(42)
n = 8
labels = ['Wälder', 'Ozeane', 'Böden', 'Süßwasser', 'Bestäuber',
          'Raubtiere', 'Mikroben', 'Klima']
# Adjazenzmatrix (Öko-Interaktionsnetz)
A = np.array([
    [0,1,1,1,1,0,1,1],
    [1,0,0,1,0,1,1,1],
    [1,0,0,1,1,1,1,0],
    [1,1,1,0,1,0,1,0],
    [1,0,1,1,0,1,0,0],
    [0,1,1,0,1,0,1,0],
    [1,1,1,1,0,1,0,1],
    [1,1,0,0,0,0,1,0],
])

angles = np.linspace(0, 2*np.pi, n, endpoint=False)
pos = np.column_stack([np.cos(angles), np.sin(angles)])

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Plot 8 – Resilienz-Interaktionsnetz der Ökosystemkomponenten\n(Subgraph Algorithmus Signaturkodierung)', pad=10)

colors = ['#27ae60','#2980b9','#8b4513','#1abc9c','#f39c12','#e74c3c','#9b59b6','#7f8c8d']
for i in range(n):
    for j in range(i+1, n):
        if A[i,j]:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 
                    color='#bdc3c7', lw=1.0, zorder=1)
for i in range(n):
    ax.scatter(*pos[i], s=600, color=colors[i], zorder=3, edgecolors='white', lw=1.5)
    offset = pos[i] * 1.22
    ax.text(offset[0], offset[1], labels[i], ha='center', va='center', fontsize=9,
            fontweight='bold')

# Signaturen berechnen und anzeigen
sigmas = []
for j in range(n):
    col = A[:, j]
    sig = sum(col[i] * (2**i) for i in range(n)) + j * (2**n)
    sigmas.append(sig)
sig_str = ', '.join([f'σ{j}={sigmas[j]}' for j in range(n)])
ax.text(0, -1.55, f'Signaturen: {sig_str}', ha='center', fontsize=6.5,
        style='italic', color='#555')
fig.tight_layout()
fig.savefig('plot8_resilienznetz.pdf')
plt.close()
print("Plot 8 gespeichert.")

print("\nAlle Plots erfolgreich generiert!")

