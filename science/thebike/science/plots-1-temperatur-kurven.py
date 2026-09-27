"""
Plot 1: Akkutemperatur mit und ohne Thermoumschlag bei Kälte (-10°C Umgebung)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 7200, 1000)  # 0 bis 2 Stunden in Sekunden

# Physikalisches Modell: Newton'sches Abkühlgesetz
# T(t) = T_umgebung + (T_start - T_umgebung) * exp(-t / tau)
T_umgebung = -10.0   # °C
T_start = 20.0       # °C Starttemperatur

# Ohne Thermoumschlag: schnelle Abkühlung
tau_ohne = 900       # Zeitkonstante 15 Minuten
T_ohne = T_umgebung + (T_start - T_umgebung) * np.exp(-t / tau_ohne)

# Mit Thermoumschlag (aerogel + PCM): sehr langsame Abkühlung
tau_mit = 5400       # Zeitkonstante 90 Minuten
T_mit = T_umgebung + (T_start - T_umgebung) * np.exp(-t / tau_mit)

# Optimaler Betriebsbereich
T_opt_low = 10.0
T_opt_high = 35.0

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(t/60, T_opt_low, T_opt_high, alpha=0.15, color='green', label='Optimaler Betriebsbereich (10–35 °C)')
ax.plot(t/60, T_ohne, color='#B43220', linewidth=2.2, label='Ohne Thermoumschlag ($\\tau = 15$ min)')
ax.plot(t/60, T_mit,  color='#19468C', linewidth=2.2, label='Mit Thermoumschlag ($\\tau = 90$ min)')
ax.axhline(y=-10, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Umgebungstemperatur (${-10}$ °C)')
ax.axhline(y=10, color='green', linestyle=':', linewidth=1.2, alpha=0.8)

# Markierung Zeitpunkt Unterschreitung 10°C
for tau, col, ls in [(tau_ohne, '#B43220', '-'), (tau_mit, '#19468C', '-')]:
    T_cross = T_umgebung + (T_start - T_umgebung) * np.exp(-np.linspace(0,7200,100000) / tau)
    idx = np.argmax(T_cross < T_opt_low)
    if idx > 0:
        t_cross = np.linspace(0,7200,100000)[idx] / 60
        ax.axvline(x=t_cross, color=col, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.annotate(f'{t_cross:.0f} min', xy=(t_cross, T_opt_low),
                    xytext=(t_cross+3, T_opt_low+3), fontsize=9, color=col)

ax.set_xlabel('Zeit [min]', fontsize=12)
ax.set_ylabel('Akkutemperatur [°C]', fontsize=12)
ax.set_title('Temperaturverlauf Li-Ion-Akku bei $T_{\\mathrm{amb}} = {-10}\\,°C$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 120)
ax.set_ylim(-12, 25)
plt.tight_layout()
plt.savefig('plot1_temperatur_kurven.pdf', dpi=300, bbox_inches='tight')
print("Plot 1 gespeichert")
