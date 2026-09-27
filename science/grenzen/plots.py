#!/usr/bin/env python3
"""
Plots für: Grenzen der menschlichen Erforschung – Erde, Sonne, Mond und Meer
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── Plot 1: Entfernungen im Sonnensystem und zu Galaxien (logarithmisch) ──────
fig, ax = plt.subplots(figsize=(10, 6))

objekte = [
    "Erdkern\n(6.371 km)",
    "Meerestiefe\n(10.994 km)",
    "Mond\n(384.400 km)",
    "Sonne\n(149,6 Mio km)",
    "Plutobahn\n(5,9 Mrd km)",
    "Nächster Stern\n(4,07 Lj)",
    "Milchstraßen-\nzentrum (26.000 Lj)",
    "Andromeda\n(2,5 Mio Lj)",
    "Galaxienrand\n(46 Mrd Lj)",
]
entfernungen_km = [
    6371,
    10994,
    384400,
    1.496e8,
    5.9e9,
    3.857e13,   # 4.07 Lichtjahre
    2.461e17,   # 26000 Lj
    2.366e19,   # 2.5 Mio Lj
    4.355e23,   # 46 Mrd Lj
]
farben = ['#1a6b3c', '#1a6b3c', '#2e86c1', '#e67e22',
          '#e67e22', '#8e44ad', '#c0392b', '#c0392b', '#c0392b']

y = np.arange(len(objekte))
bars = ax.barh(y, np.log10(entfernungen_km), color=farben, edgecolor='black', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(objekte, fontsize=9)
ax.set_xlabel("log₁₀(Entfernung in km)")
ax.set_title("Entfernungsskala: Erde und Meer bis zu den Galaxien (logarithmisch)")
ax.axvline(x=np.log10(1.496e8), color='orange', linestyle='--', linewidth=1.2, label='Sonne (1 AU)')
ax.axvline(x=np.log10(3.857e13), color='purple', linestyle=':', linewidth=1.2, label='Nächster Stern')

patch_erde = mpatches.Patch(color='#1a6b3c', label='Erde / Meer (erforschbar)')
patch_ss   = mpatches.Patch(color='#e67e22', label='Sonnensystem')
patch_gal  = mpatches.Patch(color='#c0392b', label='Galaktische / extragalaktische Skala')
ax.legend(handles=[patch_erde, patch_ss, patch_gal], loc='lower right', fontsize=8)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("plot1_entfernungen.pdf", format='pdf')
plt.close()
print("Plot 1 gespeichert.")

# ── Plot 2: Anteil erforschter Erde/Meer-Flächen ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

kategorien = [
    "Meerestiefe\nerkartet (< 25 %)",
    "Meerestiefe\nnoch unbekannt",
    "Landoberfläche\ndetailliert kartiert",
    "Erdkruste\nerforscht",
    "Erdmantel\nindirekt bekannt",
    "Erdkern\nnur theoretisch",
]
werte = [25, 75, 92, 18, 55, 30]
farben2 = ['#2e86c1', '#aed6f1', '#1a6b3c', '#a9dfbf', '#f39c12', '#e74c3c']

bars2 = ax.bar(kategorien, werte, color=farben2, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Schätzung der Erforschungstiefe [%]")
ax.set_title("Schätzung: Erforschungsgrad irdischer Systeme (Stand 2025)")
ax.set_ylim(0, 110)
for bar, val in zip(bars2, werte):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val} %",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("plot2_erforschungsgrad.pdf", format='pdf')
plt.close()
print("Plot 2 gespeichert.")

# ── Plot 3: Wissenschaftliche Publikationen nach Forschungsbereich ─────────────
fig, ax = plt.subplots(figsize=(9, 6))

bereiche = [
    "Ozeanographie",
    "Klimaforschung\n(Erde)",
    "Geophysik\n(Erdinneres)",
    "Solarphysik",
    "Mondforschung",
    "Planetologie\n(Sonnensystem)",
    "Stellarastronomie",
    "Galaxienforschung",
    "Kosmologie",
]
# Relative Anzahl wissenschaftl. Papers (normiert, Schätzwerte)
papers = [18, 42, 14, 22, 8, 16, 28, 35, 30]
farben3 = ['#1a6b3c']*3 + ['#e67e22']*3 + ['#8e44ad']*3

bars3 = ax.bar(bereiche, papers, color=farben3, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Relative Anzahl Publikationen (norm., ×1000 / Jahr)")
ax.set_title("Wissenschaftliche Publikationsvolumina nach Forschungsbereich (2024)")
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)

patch_e = mpatches.Patch(color='#1a6b3c', label='Erde und Meer')
patch_s = mpatches.Patch(color='#e67e22', label='Sonnensystem')
patch_g = mpatches.Patch(color='#8e44ad', label='Galaxien / Kosmos')
ax.legend(handles=[patch_e, patch_s, patch_g], fontsize=9)
plt.tight_layout()
plt.savefig("plot3_publikationen.pdf", format='pdf')
plt.close()
print("Plot 3 gespeichert.")

# ── Plot 4: Kosten vs. Erkenntnisgewinn (Bubble Chart) ────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

missionen = [
    ("Tiefsee-\nForschung", 0.8, 88, 120, '#2e86c1'),
    ("Klimasatelliten\n(Erde)", 2.1, 95, 200, '#1a6b3c'),
    ("ISS", 150, 40, 250, '#e67e22'),
    ("Mondmissionen\nArtemis", 93, 70, 300, '#f39c12'),
    ("Mars-Rover", 2.7, 55, 180, '#e74c3c'),
    ("Hubble / JWST", 12, 80, 220, '#8e44ad'),
    ("Galaxiensurveys\n(Euclid)", 1.5, 45, 160, '#c0392b'),
]

for (name, kosten, nutzen, groesse, farbe) in missionen:
    ax.scatter(kosten, nutzen, s=groesse*2, color=farbe, alpha=0.75,
               edgecolors='black', linewidths=0.7)
    ax.annotate(name, (kosten, nutzen), textcoords="offset points",
                xytext=(5, 5), fontsize=8)

ax.set_xscale('log')
ax.set_xlabel("Kosten [Mrd. USD, log-Skala]")
ax.set_ylabel("Direkte Anwendbarkeit für Menschheit [%]")
ax.set_title("Forschungskosten vs. Direkte Anwendbarkeit für das Leben auf der Erde")
ax.grid(linestyle='--', alpha=0.4)
ax.axhline(y=60, color='gray', linestyle=':', linewidth=1)
ax.text(0.6, 61, "Schwellwert direkter Nutzen", fontsize=8, color='gray')
plt.tight_layout()
plt.savefig("plot4_kosten_nutzen.pdf", format='pdf')
plt.close()
print("Plot 4 gespeichert.")

# ── Plot 5: Extrembedingungen auf der Erde (Radar/Polar) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

kategorien5 = [
    "Temperaturen\n(-89 bis +57 °C)",
    "Meeresdrücke\n(bis 1100 bar)",
    "Winde\n(bis 480 km/h)",
    "Strahlung\n(UV, IR)",
    "Tektonik\n(Erdbeben)",
    "Biologische\nVielfalt",
]
N = len(kategorien5)
werte5 = [0.85, 0.90, 0.75, 0.70, 0.80, 0.95]
werte5 += werte5[:1]
winkel = [n / float(N) * 2 * np.pi for n in range(N)]
winkel += winkel[:1]

ax.plot(winkel, werte5, 'o-', linewidth=2, color='#2e86c1')
ax.fill(winkel, werte5, alpha=0.25, color='#2e86c1')
ax.set_xticks(winkel[:-1])
ax.set_xticklabels(kategorien5, fontsize=9)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["20%","40%","60%","80%","100%"], fontsize=7)
ax.set_title("Extreme Bedingungen auf der Erde –\nForschungsrelevanz (normiert)", pad=20)
plt.tight_layout()
plt.savefig("plot5_extreme_erde.pdf", format='pdf')
plt.close()
print("Plot 5 gespeichert.")

# ── Plot 6: Lebenszeit vs. Forschungsumfang ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

t = np.linspace(0, 80, 1000)  # Lebensjahre

# Sigmoid-artige Kurven: wie viel kann ein Mensch in seinem Leben erforschen?
def sigmoid(t, k, t0, cap):
    return cap / (1 + np.exp(-k * (t - t0)))

erde_meer = sigmoid(t, 0.12, 25, 0.85)
sonne_mond = sigmoid(t, 0.10, 35, 0.45)
galaxien   = sigmoid(t, 0.05, 60, 0.08)

ax.plot(t, erde_meer, '-', color='#1a6b3c', linewidth=2.5, label='Erde & Meer (direkt erforschbar)')
ax.plot(t, sonne_mond, '--', color='#e67e22', linewidth=2.5, label='Sonne & Mond (indirekt erforschbar)')
ax.plot(t, galaxien, ':', color='#c0392b', linewidth=2.5, label='Galaxien (kaum greifbar)')

ax.axvline(x=80, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
ax.text(79, 0.7, "Max.\nLebenserwartung", fontsize=8, ha='right', color='black', alpha=0.6)

ax.fill_between(t, erde_meer, galaxien, alpha=0.08, color='#1a6b3c', label='Wissens­lücke: Erde vs. Galaxien')

ax.set_xlabel("Lebensalter des Forschers [Jahre]")
ax.set_ylabel("Akkumuliertes Verständnis (normiert)")
ax.set_title("Akkumuliertes wissenschaftliches Verständnis im Laufe eines Forscherlebens")
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 85)
ax.set_ylim(0, 1.0)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("plot6_lebenszeit.pdf", format='pdf')
plt.close()
print("Plot 6 gespeichert.")

print("\nAlle 6 Plots erfolgreich erstellt.")
