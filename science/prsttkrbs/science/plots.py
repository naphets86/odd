import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, binom
from matplotlib.gridspec import GridSpec

# Global style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE = '#19468C'
RED = '#B4321E'
GREEN = '#1E6432'
GRAY = '#3C3C46'
LIGHT = '#F5F5F8'

# -----------------------------------------------------------------------
# Plot 1: Inzidenz und Mortalität Prostatakrebs in Deutschland 2000-2022
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
years = np.array([2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020,2022])
inzidenz = np.array([48200,52100,55300,58700,62000,65100,67800,66200,65800,65300,64100,65820])
mortalitaet = np.array([11800,11500,11200,11100,11000,12100,13000,13400,14200,14100,13800,14300])

ax.fill_between(years, inzidenz, alpha=0.15, color=BLUE)
ax.fill_between(years, mortalitaet, alpha=0.15, color=RED)
ax.plot(years, inzidenz, 'o-', color=BLUE, linewidth=2.2, markersize=6, label='Neuerkrankungen')
ax.plot(years, mortalitaet, 's-', color=RED, linewidth=2.2, markersize=6, label='Todesfälle')

ax.set_xlabel('Jahr')
ax.set_ylabel('Anzahl Fälle')
ax.set_title('Prostatakrebs in Deutschland: Inzidenz und Mortalität (2000–2022)')
ax.legend(frameon=False)
ax.set_ylim(0, 80000)
ax.set_xlim(1999, 2023)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('plot1_inzidenz_mortalitaet.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 saved.")

# -----------------------------------------------------------------------
# Plot 2: Überleben nach 3 Jahren – Pflaster vs. Injektion (Kaplan–Meier)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
t = np.linspace(0, 36, 500)

# Modellierte Überlebenskurven basierend auf Studiendaten (87% vs 86% nach 36 Mo)
lambda_pflaster = -np.log(0.87) / 36
lambda_injektion = -np.log(0.86) / 36

surv_pflaster = np.exp(-lambda_pflaster * t)
surv_injektion = np.exp(-lambda_injektion * t)

ax.step(t, surv_pflaster * 100, color=BLUE, linewidth=2.2, label='Östrogen-Pflaster ($n = 652$)')
ax.step(t, surv_injektion * 100, color=RED, linewidth=2.2, linestyle='--', label='LHRH-Injektion ($n = 661$)')

# Konfidenzintervall (±1.5% approximiert)
ax.fill_between(t, (surv_pflaster - 0.015) * 100, (surv_pflaster + 0.015) * 100,
                alpha=0.15, color=BLUE)
ax.fill_between(t, (surv_injektion - 0.015) * 100, (surv_injektion + 0.015) * 100,
                alpha=0.15, color=RED)

ax.axhline(87, color=BLUE, linestyle=':', alpha=0.6, linewidth=1)
ax.axhline(86, color=RED, linestyle=':', alpha=0.6, linewidth=1)
ax.axvline(36, color=GRAY, linestyle=':', alpha=0.5, linewidth=1)
ax.text(36.5, 70, '36 Monate', color=GRAY, fontsize=9)

ax.set_xlabel('Zeit (Monate)')
ax.set_ylabel('Progressionsfreies Überleben (%)')
ax.set_title('Kaplan–Meier-Kurven: Progressionsfreies Überleben nach 36 Monaten')
ax.set_ylim(60, 102)
ax.set_xlim(0, 38)
ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('plot2_kaplan_meier.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 saved.")

# -----------------------------------------------------------------------
# Plot 3: Nebenwirkungsvergleich Pflaster vs. Injektion
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
nebenwirkungen = [
    'Hitzewallungen /\nSchweißausbrüche',
    'Gynäkomastie',
    'Knochendichte-\nverlust',
    'Kard. Ereignisse',
    'Fatigue',
]
pflaster_pct = [50, 80, 25, 8, 35]
injektion_pct = [90, 40, 55, 5, 45]

x = np.arange(len(nebenwirkungen))
width = 0.35

bars1 = ax.bar(x - width/2, pflaster_pct, width, color=BLUE, alpha=0.85, label='Östrogen-Pflaster')
bars2 = ax.bar(x + width/2, injektion_pct, width, color=RED, alpha=0.85, label='LHRH-Injektion')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9, color=BLUE)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(nebenwirkungen, fontsize=9.5)
ax.set_ylabel('Häufigkeit (%)')
ax.set_title('Vergleich der Nebenwirkungsprofile: Pflaster vs. LHRH-Injektion')
ax.set_ylim(0, 105)
ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('plot3_nebenwirkungen.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 saved.")

# -----------------------------------------------------------------------
# Plot 4: Testosteron-Suppression über die Zeit (Feedback-Modell)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
t_months = np.linspace(0, 24, 300)

# Testosteron-Verlauf (normiert, 100 = Ausgangswert)
testo_injektion = 100 * np.exp(-0.5 * t_months) + 5
testo_pflaster = 100 * (1 / (1 + 0.6 * t_months)) * np.exp(-0.12 * t_months) + 8

# Östradiol-Spiegel (Pflaster)
oestradiol = 120 * (1 - np.exp(-0.4 * t_months)) * np.exp(-0.015 * t_months) + 20

ax.plot(t_months, testo_injektion, color=RED, linewidth=2, label='Testosteron (LHRH-Injektion)')
ax.plot(t_months, testo_pflaster, color=BLUE, linewidth=2, linestyle='--', label='Testosteron (Östrogen-Pflaster)')
ax2 = ax.twinx()
ax2.plot(t_months, oestradiol, color=GREEN, linewidth=1.8, linestyle='-.', label='Östradiol-Spiegel (Pflaster)')
ax2.set_ylabel('Östradiol (pmol/L, normiert)', color=GREEN)
ax2.tick_params(axis='y', labelcolor=GREEN)
ax2.spines['top'].set_visible(False)

ax.axhline(20, color=GRAY, linestyle=':', alpha=0.5, linewidth=1)
ax.text(0.5, 22, 'Kastrationsniveau', fontsize=8.5, color=GRAY)

ax.set_xlabel('Zeit (Monate)')
ax.set_ylabel('Testosteron (%, normiert)')
ax.set_title('Hormonkinetik: Testosteron-Suppression und Östradiol-Feedback')
ax.set_ylim(0, 115)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.3)
fig.tight_layout()
fig.savefig('plot4_hormonkinetik.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 saved.")

# -----------------------------------------------------------------------
# Plot 5: Kosten-Analyse – Pflaster vs. LHRH-Injektion (pro Jahr)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

kategorien = ['Medikament\n(1 Jahr)', 'Arztbesuche\n(1 Jahr)', 'NW-Behandlung\n(1 Jahr)', 'Gesamtkosten\n(1 Jahr)']
kosten_pflaster = [480, 240, 320, 1040]
kosten_injektion = [3600, 480, 580, 4660]

x = np.arange(len(kategorien))
width = 0.35

b1 = ax.bar(x - width/2, kosten_pflaster, width, color=BLUE, alpha=0.85, label='Östrogen-Pflaster (geschätzt)')
b2 = ax.bar(x + width/2, kosten_injektion, width, color=RED, alpha=0.85, label='LHRH-Injektion (geschätzt)')

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{bar.get_height()}€', ha='center', va='bottom', fontsize=9, color=BLUE)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{bar.get_height()}€', ha='center', va='bottom', fontsize=9, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(kategorien)
ax.set_ylabel('Kosten (EUR/Jahr, geschätzt)')
ax.set_title('Geschätzter Kostenvergleich: Pflaster vs. LHRH-Injektion (pro Patient, pro Jahr)')
ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('plot5_kosten.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 saved.")

# -----------------------------------------------------------------------
# Plot 6: Statistische Auswertung – Chi-Quadrat / Konfidenzintervalle
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))

# 95%-KI für Überlebensraten
groups = ['Pflaster\n(n=652)', 'Injektion\n(n=661)']
means = [87.0, 86.0]
# Wilson-Konfidenzintervall
def wilson_ci(p, n, z=1.96):
    p /= 100
    center = (p + z**2/(2*n)) / (1 + z**2/n)
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return (center - margin)*100, (center + margin)*100

ci_pflaster = wilson_ci(87, 652)
ci_injektion = wilson_ci(86, 661)

yerr_low = [means[0] - ci_pflaster[0], means[1] - ci_injektion[0]]
yerr_high = [ci_pflaster[1] - means[0], ci_injektion[1] - means[1]]

colors = [BLUE, RED]
for i, (g, m, yl, yh) in enumerate(zip(groups, means, yerr_low, yerr_high)):
    ax.errorbar(i, m, yerr=[[yl], [yh]], fmt='o', color=colors[i],
                markersize=12, capsize=8, capthick=2, linewidth=2,
                label=f'{g.strip()}: {m}% (95%-KI)')
    ax.axhspan(m - yl, m + yh, alpha=0.08, color=colors[i], xmin=i*0.45+0.05, xmax=i*0.45+0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(groups)
ax.set_ylabel('Progressionsfreies Überleben nach 36 Mo. (%)')
ax.set_title('95%-Wilson-Konfidenzintervalle: Überlebensraten im Vergleich')
ax.set_ylim(82, 92)
ax.set_xlim(-0.5, 1.5)
ax.legend(frameon=False, loc='lower center')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# p-Wert Annotation
ax.annotate('', xy=(1, 87.5), xytext=(0, 87.5),
            arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.5))
ax.text(0.5, 88, 'p = 0.612 (n.s.)', ha='center', fontsize=9.5, color=GRAY)

fig.tight_layout()
fig.savefig('plot6_konfidenzintervalle.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 saved.")

# -----------------------------------------------------------------------
# Plot 7: Tumorsuppression-Mechanismus (Feedback-Schleife schematisch)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Schematische Darstellung: Östrogenfeedback-Schleife', fontsize=13, pad=12)

# Boxen
def draw_box(ax, xy, w, h, text, color, fontsize=10):
    rx, ry = xy
    rect = mpatches.FancyBboxPatch((rx - w/2, ry - h/2), w, h,
                                    boxstyle='round,pad=0.15',
                                    linewidth=1.5, edgecolor=color,
                                    facecolor=color + '22')
    ax.add_patch(rect)
    ax.text(rx, ry, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight='bold', wrap=True,
            multialignment='center')

def arrow(ax, start, end, color, label='', offset=(0, 0.2)):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    mid = ((start[0]+end[0])/2 + offset[0], (start[1]+end[1])/2 + offset[1])
    if label:
        ax.text(mid[0], mid[1], label, ha='center', color=color, fontsize=8.5)

draw_box(ax, (5, 7), 3.5, 0.8, 'Östradiol-Pflaster\n(transdermale Absorption)', BLUE)
draw_box(ax, (5, 5.5), 3.2, 0.8, 'Hypothalamus / Hypophyse\n(GnRH-Suppression)', GREEN)
draw_box(ax, (5, 4.0), 3.0, 0.8, 'Hoden\n(LH↓ → Testosteron↓)', RED)
draw_box(ax, (5, 2.5), 3.0, 0.8, 'Prostatatumor\n(Wachstumsstop)', GRAY)
draw_box(ax, (1.5, 5.5), 2.2, 0.8, 'Feedback:\nÖstradiol → Gehirn', GREEN, fontsize=9)

arrow(ax, (5, 6.6), (5, 5.9), BLUE, 'Östradiol ↑')
arrow(ax, (5, 5.1), (5, 4.4), RED, 'LH ↓ / FSH ↓')
arrow(ax, (5, 3.6), (5, 2.9), GRAY, 'Testosteron ↓')
# Feedback-Pfeil
ax.annotate('', xy=(2.6, 5.5), xytext=(3.3, 7.0),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8,
                            connectionstyle='arc3,rad=-0.4'))
ax.text(2.0, 6.5, 'neg.\nFeedback', ha='center', color=GREEN, fontsize=8.5)

fig.tight_layout()
fig.savefig('plot7_feedback_schema.pdf', bbox_inches='tight')
plt.close()
print("Plot 7 saved.")

# -----------------------------------------------------------------------
# Plot 8: PSA-Verlauf unter Therapie (Modell)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
t = np.linspace(0, 36, 300)

np.random.seed(42)
# Mittlerer PSA-Verlauf unter Therapie
psa_pflaster_mean = 18 * np.exp(-0.18 * t) + 1.5
psa_injektion_mean = 18 * np.exp(-0.22 * t) + 1.2

noise = 0.5 * np.random.randn(len(t))
ax.plot(t, psa_pflaster_mean, color=BLUE, linewidth=2.5, label='Pflaster – mittlerer PSA-Verlauf')
ax.plot(t, psa_injektion_mean, color=RED, linewidth=2.5, linestyle='--',
        label='Injektion – mittlerer PSA-Verlauf')

ax.fill_between(t, psa_pflaster_mean * 0.7, psa_pflaster_mean * 1.3,
                alpha=0.12, color=BLUE, label='Streubereich Pflaster')
ax.fill_between(t, psa_injektion_mean * 0.7, psa_injektion_mean * 1.3,
                alpha=0.12, color=RED, label='Streubereich Injektion')

ax.axhline(4.0, color=GRAY, linestyle=':', linewidth=1.2)
ax.text(1, 4.4, 'PSA-Schwelle 4 ng/mL', fontsize=8.5, color=GRAY)

ax.set_xlabel('Zeit unter Therapie (Monate)')
ax.set_ylabel('PSA-Wert (ng/mL)')
ax.set_title('Modellierter PSA-Verlauf unter Östrogen-Pflaster vs. LHRH-Injektion')
ax.set_ylim(0, 22)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('plot8_psa_verlauf.pdf', bbox_inches='tight')
plt.close()
print("Plot 8 saved.")

print("\nAlle Plots erfolgreich generiert.")
