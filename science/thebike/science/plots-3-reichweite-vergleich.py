"""
Plot 3: Reichweitenvergleich mit/ohne Thermoumschlag über Umgebungstemperatur
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

T_amb = np.linspace(-20, 30, 200)

def reichweite(T_akku, R_ref=60.0):
    """km Reichweite basierend auf Akkutemperatur"""
    def kap(t):
        if t < -20: return 0.30
        elif t < 0: return 0.30 + (t+20)*(0.50/20)
        elif t < 20: return 0.80 + t*(0.20/20)
        elif t <= 40: return 1.0
        else: return 1.0 - (t-40)**1.5*0.004
    return R_ref * np.clip(kap(T_akku), 0, 1)

# Temperatur im Akku in Abhängigkeit der Umgebungstemperatur
# Ohne Umschlag: T_akku ≈ T_amb + etwas Eigenerwärmung
def T_akku_ohne(T_amb, fahrzeit_min=60):
    T_start = 20
    tau = 900  # 15 min
    T_end = T_amb + (T_start - T_amb) * np.exp(-fahrzeit_min*60/tau)
    return (T_start + T_end) / 2  # Mittlere Temperatur

def T_akku_mit(T_amb, fahrzeit_min=60):
    T_start = 20
    tau = 5400  # 90 min
    T_end = T_amb + (T_start - T_amb) * np.exp(-fahrzeit_min*60/tau)
    return (T_start + T_end) / 2

R_ohne = np.array([reichweite(T_akku_ohne(t)) for t in T_amb])
R_mit  = np.array([reichweite(T_akku_mit(t))  for t in T_amb])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Links: Reichweite absolut
ax1.plot(T_amb, R_ohne, color='#B43220', linewidth=2.2, label='Ohne Thermoumschlag')
ax1.plot(T_amb, R_mit,  color='#19468C', linewidth=2.2, label='Mit Thermoumschlag')
ax1.fill_between(T_amb, R_ohne, R_mit, alpha=0.15, color='green', label='Reichweitengewinn')
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax1.set_xlabel('Umgebungstemperatur [°C]', fontsize=11)
ax1.set_ylabel('Reichweite [km]', fontsize=11)
ax1.set_title('Absolute Reichweite', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-20, 30)

# Rechts: Prozentualer Gewinn
gewinn = (R_mit - R_ohne) / 60.0 * 100
ax2.bar(T_amb, gewinn, width=0.5, color='#19468C', alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_xlabel('Umgebungstemperatur [°C]', fontsize=11)
ax2.set_ylabel('Reichweitengewinn [%] gegenüber Referenz', fontsize=11)
ax2.set_title('Relativer Reichweitengewinn durch Thermoumschlag', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-20, 30)

plt.suptitle('Reichweitenanalyse E-Bike (60 km Referenz, 60 min Fahrt)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plot3_reichweite_vergleich.pdf', dpi=300, bbox_inches='tight')
print("Plot 3 gespeichert")
