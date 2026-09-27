"""
Plot 4: Zyklenlebensdauer und Kapazitätsdegradation in Abhängigkeit der Betriebstemperatur
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Arrhenius-Modell für Kapazitätsdegradation
# k(T) = A * exp(-Ea / (R*T))
R_gas = 8.314
Ea = 65000  # J/mol (typisch Li-Ion NMC)
A = 1e10

def k_arrhenius(T_C):
    T_K = T_C + 273.15
    return A * np.exp(-Ea / (R_gas * T_K))

T_range = np.linspace(-10, 60, 300)
k_vals = k_arrhenius(T_range)
k_ref = k_arrhenius(25)
k_norm = k_vals / k_ref

# Zyklenlebensdauer: nimmt ab bei hoher Temperatur, aber auch bei zu kalten Lade-Zyklen
def zyklen(T):
    if T < 0:
        # Lithium-Plating bei Kälte: stark degradierend
        return 800 * np.exp(0.03 * T)  # exponentieller Abfall
    elif T <= 25:
        return 800 + (T) * (200/25)
    elif T <= 40:
        return 1000 - (T-25) * (300/15)
    else:
        return 700 * np.exp(-0.04*(T-40))

Z = np.array([zyklen(t) for t in T_range])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Links: Arrhenius Degradationsrate
ax1.semilogy(T_range, k_norm, color='#B43220', linewidth=2.2)
ax1.axvspan(-10, 10, alpha=0.1, color='blue', label='Kalte Lagerung')
ax1.axvspan(10, 35, alpha=0.1, color='green', label='Optimalbereich')
ax1.axvspan(35, 60, alpha=0.1, color='red', label='Überhitzung')
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('Temperatur [°C]', fontsize=11)
ax1.set_ylabel('Normierte Degradationsrate $k(T)/k(25°C)$', fontsize=11)
ax1.set_title('Arrhenius-Degradationsrate ($E_a = 65\\,\\mathrm{kJ/mol}$)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(-10, 60)

# Rechts: Zyklenlebensdauer
ax2.plot(T_range, Z, color='#19468C', linewidth=2.2)
ax2.fill_between(T_range, Z, 0, where=(T_range >= 10) & (T_range <= 35),
                 alpha=0.2, color='green', label='Optimaler Betriebsbereich')
ax2.axvline(x=-10, color='orange', linestyle=':', linewidth=1.5, label='Winterbetrieb ohne Umschlag')
ax2.axvline(x=15,  color='#19468C', linestyle=':', linewidth=1.5, label='Winterbetrieb mit Umschlag')

# Differenz annotieren
z_10 = zyklen(-10)
z_15 = zyklen(15)
ax2.annotate(f'{z_10:.0f} Zyklen\nbei $-10\\,°C$', xy=(-10, z_10), xytext=(-8, z_10+120),
             fontsize=9, color='orange', arrowprops=dict(arrowstyle='->', color='orange'))
ax2.annotate(f'{z_15:.0f} Zyklen\nbei $+15\\,°C$', xy=(15, z_15), xytext=(20, z_15-150),
             fontsize=9, color='#19468C', arrowprops=dict(arrowstyle='->', color='#19468C'))

ax2.set_xlabel('Betriebstemperatur [°C]', fontsize=11)
ax2.set_ylabel('Zyklenlebensdauer [Vollzyklen]', fontsize=11)
ax2.set_title('Zyklenlebensdauer NMC-Akku', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, 60)
ax2.set_ylim(0, 1200)

plt.suptitle('Thermische Alterung und Lebensdauer von Li-Ion-Akkus', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plot4_lebensdauer.pdf', dpi=300, bbox_inches='tight')
print("Plot 4 gespeichert")
