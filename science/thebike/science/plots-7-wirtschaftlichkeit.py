"""
Plot 7: Wirtschaftlichkeitsanalyse – Kostenvergleich und Amortisation
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

Jahre = np.arange(0, 8.1, 0.1)

# Ohne Thermoumschlag:
# Akku muss nach 3 Jahren ersetzt werden (600 Zyklen, 1x pro Tag)
# Akkupreis: 600 EUR, Thermoumschlag: 80 EUR
preis_akku = 600
preis_umschlag = 80
kosten_ohne = np.zeros_like(Jahre)
for i, j in enumerate(Jahre):
    ersatz = int(j / 3)   # alle 3 Jahre neuer Akku
    kosten_ohne[i] = ersatz * preis_akku

# Mit Thermoumschlag:
# Akku hält 5 Jahre (1000 Zyklen), 80 EUR für Umschlag
kosten_mit = np.zeros_like(Jahre)
for i, j in enumerate(Jahre):
    kosten_mit[i] = preis_umschlag
    ersatz = int(j / 5)
    kosten_mit[i] += ersatz * preis_akku

# Stromkosten durch Reichweitengewinn: ~3 EUR/Monat gespart (weniger Laden)
einsparung_pro_jahr = 36  # EUR
einsparung_kum = Jahre * einsparung_pro_jahr
netto_mit = kosten_mit - einsparung_kum

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Links: Kumulierte Kosten
ax1.plot(Jahre, kosten_ohne, color='#B43220', linewidth=2.2, label='Ohne Thermoumschlag')
ax1.plot(Jahre, kosten_mit,  color='#19468C', linewidth=2.2, label='Mit Thermoumschlag (Brutto)')
ax1.plot(Jahre, netto_mit,   color='darkgreen', linewidth=2.2, linestyle='--',
         label='Mit Thermoumschlag (nach Stromersparnis)')
ax1.fill_between(Jahre, kosten_ohne, netto_mit, where=(Jahre>=0),
                 alpha=0.1, color='green', label='Ersparnis')
# Amortisationspunkt
idx_amort = np.argmin(np.abs(netto_mit - kosten_ohne[:len(netto_mit)]))
j_amort = Jahre[idx_amort]
ax1.axvline(x=j_amort, color='green', linestyle=':', linewidth=1.5)
ax1.annotate(f'Amortisation\n({j_amort:.1f} Jahre)', xy=(j_amort, netto_mit[idx_amort]),
             xytext=(j_amort+0.5, netto_mit[idx_amort]+50),
             fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))
ax1.set_xlabel('Jahre', fontsize=11)
ax1.set_ylabel('Kumulierte Kosten [€]', fontsize=11)
ax1.set_title('Kumulierte Akkukosten im Vergleich', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 1500)

# Rechts: Balkenchart Lebensdauer & Kosten pro km
kategorien = ['Ohne\nThermoumschlag', 'Mit\nThermoumschlag']
lebensdauer = [3, 5]
kosten_pro_km = [preis_akku/3/365 / 0.05, (preis_akku/5/365 + preis_umschlag/5/365) / 0.05]
# (Akkupreis / Lebensdauer / 365 Tage / 50 km pro Tag) = EUR pro km

x = np.array([0, 1])
width = 0.35
bars1 = ax2.bar(x - width/2, lebensdauer, width, color=['#B43220', '#19468C'],
                alpha=0.8, label='Akkulebensdauer [Jahre]')
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, kosten_pro_km, width, color=['#FF8866', '#6699CC'],
                     alpha=0.8, label='Kosten pro km [ct/km]')
ax2.set_xticks(x)
ax2.set_xticklabels(kategorien, fontsize=11)
ax2.set_ylabel('Akkulebensdauer [Jahre]', fontsize=11, color='#19468C')
ax2_twin.set_ylabel('Kosten pro km [ct/km]', fontsize=11, color='#B43220')
ax2.set_title('Lebensdauer und Kosteneffizienz', fontsize=12)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 7)
ax2_twin.set_ylim(0, 0.08)

for bar, val in zip(bars1, lebensdauer):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.1, f'{val} J', ha='center', fontsize=10)
for bar, val in zip(bars2, kosten_pro_km):
    ax2_twin.text(bar.get_x()+bar.get_width()/2, val+0.001, f'{val*100:.2f} ct', ha='center', fontsize=10)

plt.suptitle('Wirtschaftlichkeitsanalyse des Thermoumschlags (E-Bike, NMC-Akku 600 €)', fontsize=13)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Platz für Titel reservieren
plt.savefig('plot7_wirtschaftlichkeit.pdf', dpi=300)
print("Plot 7 gespeichert")
