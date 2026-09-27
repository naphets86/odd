"""
Plot 6: PCM Phasenübergang und Wärmepuffer-Effekt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Enthalpie-Temperatur-Kurve mit Phasenübergang
T_PCM = np.linspace(-5, 35, 1000)
T_melt = 18.0  # Schmelzpunkt PCM (z.B. Paraffin RT18)
dT_melt = 2.0  # Breite des Übergangs
H_solid = 200  # J/g Festkörper cp
H_melt  = 180  # J/g latente Wärme
H_liquid = 180  # J/g Flüssigkeit cp

# Sigmoid für Übergang
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))

H = (H_solid * (T_PCM - (-5)) / 40 +
     H_melt * sigmoid(T_PCM, T_melt, 4/dT_melt))

# Zeitlicher Temperaturverlauf mit/ohne PCM-Puffer bei Abkühlung
t2 = np.linspace(0, 120, 500)  # Minuten
T_amb2 = -10

# Ohne PCM: einfaches Newton-Abkühlen
tau_noPCM = 20
T_noPCM = T_amb2 + (20 - T_amb2) * np.exp(-t2/tau_noPCM)

# Mit PCM: Plateau beim Schmelzpunkt
def T_mit_PCM(t, T_start=20, T_amb=-10, T_melt=18, tau1=20, tau2=80, t_melt_end=35):
    T = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t_melt_end:
            # Abkühlung bis Schmelzpunkt, dann Plateau
            T_drop = T_amb + (T_start - T_amb) * np.exp(-ti/tau1)
            T[i] = max(T_drop, T_melt - 0.5)
        else:
            # Nach PCM-Erschöpfung: normale Abkühlung
            T0_phase2 = T_melt - 0.5
            T[i] = T_amb + (T0_phase2 - T_amb) * np.exp(-(ti-t_melt_end)/tau2)
    return T

T_PCM_time = T_mit_PCM(t2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Links: Enthalpie-Temperatur-Kurve
ax1.plot(T_PCM, H, color='#19468C', linewidth=2.5)
ax1.axvspan(T_melt-dT_melt, T_melt+dT_melt, alpha=0.15, color='#B43220',
            label=f'Phasenübergangsbereich ($T_{{\\mathrm{{PCM}}}} = {T_melt:.0f}\\,°C$)')
ax1.axvspan(10, 35, alpha=0.08, color='green', label='Optimaler Betriebsbereich')
ax1.set_xlabel('Temperatur [°C]', fontsize=11)
ax1.set_ylabel('Enthalpie $H$ [J/g]', fontsize=11)
ax1.set_title('Enthalpie-Kurve PCM (RT18-Paraffin)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 35)

# Rechts: Zeitlicher Verlauf
ax2.plot(t2, T_noPCM, color='#B43220', linewidth=2.2, label='Ohne PCM-Schicht')
ax2.plot(t2, T_PCM_time, color='#19468C', linewidth=2.2, label='Mit PCM-Schicht (RT18)')
ax2.fill_between(t2, 10, 35, alpha=0.08, color='green', label='Optimaler Betriebsbereich')
ax2.axhline(y=T_melt, color='#B43220', linestyle=':', linewidth=1.2, alpha=0.6,
            label=f'Schmelzpunkt PCM ({T_melt:.0f} °C)')
ax2.axhline(y=-10, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Plateau annotieren
ax2.annotate('PCM-Plateau\n(latente Wärme)', xy=(17, 17.5), xytext=(40, 14),
             fontsize=9, color='#19468C',
             arrowprops=dict(arrowstyle='->', color='#19468C'))

ax2.set_xlabel('Zeit [min]', fontsize=11)
ax2.set_ylabel('Akkutemperatur [°C]', fontsize=11)
ax2.set_title('PCM-Pufferwirkung im Zeitverlauf', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 120)
ax2.set_ylim(-12, 23)

plt.suptitle('Phase-Change-Material (PCM) als thermischer Puffer für E-Bike-Akkus', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plot6_pcm_phasenuebergang.pdf', dpi=300, bbox_inches='tight')
print("Plot 6 gespeichert")
