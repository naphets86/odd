"""
Plot 2: Kapazitätsverlust als Funktion der Temperatur (Li-Ion)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Empirische Modellierung der nutzbaren Kapazität
# Basierend auf Literaturwerten für LFP und NMC Zellen
T = np.linspace(-30, 60, 500)

def kapazitaet_NMC(T):
    """Relative Kapazität NMC-Zelle als Funktion der Temperatur"""
    k = np.where(T < 20,
        100 * (1 - 0.008*(20-T)**1.3 / 100),
        np.where(T <= 40,
            100.0,
            100 * (1 - 0.003*(T-40)**1.8 / 100)))
    k = np.clip(k, 0, 100)
    # Kälte stärker
    k = np.where(T < 0,  100 * np.exp(-((T-20)**2)/(2*35**2)) * 1.05, k)
    k = np.clip(k, 0, 100)
    return k

# Einfaches Peukert-ähnliches Modell
def kap(T):
    C = np.zeros_like(T, dtype=float)
    for i, t in enumerate(T):
        if t < -20:
            C[i] = 30
        elif t < 0:
            C[i] = 30 + (t+20) * (50/20)
        elif t < 20:
            C[i] = 80 + (t) * (20/20)
        elif t <= 40:
            C[i] = 100
        else:
            C[i] = 100 - (t-40)**1.5 * 0.4
    return np.clip(C, 0, 100)

C_NMC = kap(T)

# LFP etwas kälterobuster
def kap_LFP(T):
    C = np.zeros_like(T, dtype=float)
    for i, t in enumerate(T):
        if t < -20:
            C[i] = 40
        elif t < 0:
            C[i] = 40 + (t+20) * (45/20)
        elif t < 20:
            C[i] = 85 + (t) * (15/20)
        elif t <= 40:
            C[i] = 100
        else:
            C[i] = 100 - (t-40)**1.5 * 0.35
    return np.clip(C, 0, 100)

C_LFP = kap_LFP(T)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_betweenx([0,110], 10, 35, alpha=0.12, color='green', label='Optimaler Betriebsbereich')
ax.plot(T, C_NMC, color='#B43220', linewidth=2.2, label='NMC-Zelle (typisch E-Bike)')
ax.plot(T, C_LFP, color='#19468C', linewidth=2.2, linestyle='--', label='LFP-Zelle')

ax.axvline(x=-10, color='orange', linestyle=':', linewidth=1.5, label='Typische Wintertemperatur (${-10}\\,°C$)')

# Beschriftung Verlust
for T_mark, col, name in [(-10, '#B43220', 'NMC'), (-10, '#19468C', 'LFP')]:
    idx = np.argmin(np.abs(T - T_mark))
    C_val = C_NMC[idx] if name=='NMC' else C_LFP[idx]
    verlust = 100 - C_val
    ax.annotate(f'{name}: {C_val:.0f}% Kapaz.\n(−{verlust:.0f}%)',
                xy=(T_mark, C_val), xytext=(T_mark-12, C_val-20),
                fontsize=9, color=col,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

ax.set_xlabel('Batterietemperatur [°C]', fontsize=12)
ax.set_ylabel('Nutzbare Kapazität [%]', fontsize=12)
ax.set_title('Nutzbare Kapazität von Li-Ion-Akkus in Abhängigkeit der Temperatur', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-30, 60)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('plot2_kapazitaet_temperatur.pdf', dpi=300, bbox_inches='tight')
print("Plot 2 gespeichert")
