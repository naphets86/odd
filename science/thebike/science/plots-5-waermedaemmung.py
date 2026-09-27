"""
Plot 5: Wärmedurchgangskoeffizient und Wärmestrom für verschiedene Materialien
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Materialien: Wärmeleitfähigkeit [W/(m*K)], Dichte [kg/m³], cp [J/(kg*K)]
materialien = {
    'Luft (15mm)':        {'lambda': 0.026, 'd': 0.015, 'farbe': '#888888', 'hatch': '..'},
    'Schaumstoff (20mm)': {'lambda': 0.040, 'd': 0.020, 'farbe': '#AAAAFF', 'hatch': '//'},
    'Neopren (10mm)':     {'lambda': 0.050, 'd': 0.010, 'farbe': '#5588CC', 'hatch': '\\\\'},
    'Aerogel (10mm)':     {'lambda': 0.015, 'd': 0.010, 'farbe': '#19468C', 'hatch': 'xx'},
    'PCM+Aerogel (15mm)': {'lambda': 0.012, 'd': 0.015, 'farbe': '#B43220', 'hatch': '++'},
}

# U-Wert = lambda / d  [W/(m²*K)]
namen = list(materialien.keys())
U_vals = [m['lambda'] / m['d'] for m in materialien.values()]
farben = [m['farbe'] for m in materialien.values()]
hatches = [m['hatch'] for m in materialien.values()]

# Wärmestrom bei Delta T = 30 K (z.B. innen 20°C, außen -10°C)
dT = 30  # K
q_vals = [U * dT for U in U_vals]  # W/m²

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Links: U-Werte
bars1 = ax1.barh(namen, U_vals, color=farben, edgecolor='black', linewidth=0.8)
for bar, hatch in zip(bars1, hatches):
    bar.set_hatch(hatch)
for i, (bar, val) in enumerate(zip(bars1, U_vals)):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.2f} W/(m²·K)', va='center', fontsize=10)
ax1.set_xlabel('Wärmedurchgangskoeffizient $U$ [W/(m²·K)]', fontsize=11)
ax1.set_title('U-Werte verschiedener Isolationsmaterialien', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(0, max(U_vals)*1.35)

# Rechts: Wärmestrom
bars2 = ax2.barh(namen, q_vals, color=farben, edgecolor='black', linewidth=0.8)
for bar, hatch in zip(bars2, hatches):
    bar.set_hatch(hatch)
for i, (bar, val) in enumerate(zip(bars2, q_vals)):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} W/m²', va='center', fontsize=10)
ax2.set_xlabel(f'Wärmestrom $\\dot{{q}} = U \\cdot \\Delta T$ [W/m²] bei $\\Delta T = {dT}\\,K$', fontsize=11)
ax2.set_title('Wärmeverlust durch die Dämmschicht', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, max(q_vals)*1.35)

plt.suptitle('Thermische Eigenschaften von Isolationsmaterialien für Thermoumschläge', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plot5_waermedaemmung.pdf', dpi=300, bbox_inches='tight')
print("Plot 5 gespeichert")
