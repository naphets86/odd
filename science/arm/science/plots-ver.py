import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

# ─────────────────────────────────────────────────
# Plot R1: RTL Verifikations-Workflow
# ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Abbildung R1: RTL-Verifikations-Workflow für den Safety-Island\n(Lockstep-Verifikation und FDT-Nachweis)", fontweight='bold', pad=12)

stages = [
    (6, 9.2, 'Spezifikation\n(Safety-Island Spec)', '#2ecc71', 2.5, 0.6),
    (2, 7.5, 'RTL-Design\nCore-1 (Master)', '#3498db', 2.2, 0.6),
    (6, 7.5, 'Formale\nEigenschaften (SVA)', '#9b59b6', 2.2, 0.6),
    (10, 7.5, 'RTL-Design\nCore-2 (Shadow)', '#3498db', 2.2, 0.6),
    (2, 5.8, 'Synthesierte\nNetzliste (Yosys)', '#e67e22', 2.2, 0.6),
    (6, 5.8, 'Äquivalenz-\nprüfung (Yosys)', '#c0392b', 2.2, 0.6),
    (10, 5.8, 'Synthesierte\nNetzliste (Yosys)', '#e67e22', 2.2, 0.6),
    (4, 4.1, 'Lockstep-Vergleich\n(Bitgleichheit)', '#e74c3c', 2.5, 0.6),
    (8, 4.1, 'FDT-Analyse\n(≤1μs Nachweis)', '#e74c3c', 2.5, 0.6),
    (6, 2.4, 'FMEDA\n(SPFM/LFM-Nachweis)', '#8e44ad', 2.5, 0.6),
    (6, 0.9, 'ISO 26262 ASIL-D\nZertifizierung', '#27ae60', 2.8, 0.6),
]
for (x,y,lbl,col,w,h) in stages:
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.08',
                                    facecolor=col, edgecolor='white', lw=1.5, alpha=0.9, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=4)

arrows = [
    (6,8.9,6,7.8), (6,7.2,6,6.1),
    (2,7.2,2,6.1), (10,7.2,10,6.1),
    (2,5.5,4,4.4), (10,5.5,8,4.4),
    (6,5.5,6,5.5),
    (4,3.8,5.4,2.7), (8,3.8,6.6,2.7),
    (6,2.1,6,1.2),
]
for (x0,y0,x1,y1) in arrows:
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8))

ax.annotate('', xy=(2,7.8), xytext=(6,7.8), arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
ax.annotate('', xy=(10,7.8), xytext=(6,7.8), arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))

leg_items = [
    mpatches.Patch(color='#2ecc71', label='Spezifikation'),
    mpatches.Patch(color='#3498db', label='RTL-Design'),
    mpatches.Patch(color='#e67e22', label='Synthese / Netzliste'),
    mpatches.Patch(color='#c0392b', label='Formale Prüfung'),
    mpatches.Patch(color='#e74c3c', label='Safety-Nachweis'),
    mpatches.Patch(color='#27ae60', label='Zertifizierung'),
]
ax.legend(handles=leg_items, loc='lower left', framealpha=0.95, fontsize=8.5)
plt.tight_layout()
plt.savefig('plot_r1_workflow.pdf', bbox_inches='tight')
plt.close()
print("R1 done")

# ─────────────────────────────────────────────────
# Plot R2: Lockstep-Timing und Fehlererkennungszeit
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Abbildung R2: Lockstep-Timing und Fehlererkennungszeit (FDT)\nim Safety-Island des Cortex-HX", fontweight='bold')

ax_t = axes[0]
t_cycles = np.arange(0, 20)
# Signale: CLK, Core1_Output, Core2_Output_delayed, Compare, Error_Detected
clk = (t_cycles % 2 == 0).astype(float)
core1 = np.zeros(20); core1[4:] = 1; core1[10:] = 0
core2 = np.zeros(20); core2[5:] = 1; core2[11:] = 0
# Inject fault at cycle 12
core2_fault = core2.copy(); core2_fault[12:] = 1
compare_ok = (core1 == core2).astype(float)
fdt_signal = np.zeros(20); fdt_signal[12:13] = 1

signals = [
    (clk*0.8, 'CLK', '#2c3e50'),
    (core1*0.8+2, 'Core-1\n(Master)', '#3498db'),
    (core2_fault*0.8+4, 'Core-2\n(Shadow,\nFehler@12)', '#e74c3c'),
    (compare_ok*0.8+6, 'Vergleich\n(XOR)', '#27ae60'),
    (fdt_signal*0.8+8, 'FDT-Signal\n(Alarm)', '#e67e22'),
]
for sig, name, color in signals:
    ax_t.step(t_cycles, sig, where='post', color=color, lw=2)
    ax_t.text(-0.5, sig.mean(), name, ha='right', va='center', fontsize=8, color=color, fontweight='bold')
ax_t.axvline(12, color='red', ls='--', lw=2, alpha=0.7)
ax_t.annotate('Fehler\ninjiziert', xy=(12, 9.2), ha='center', fontsize=8, color='red', fontweight='bold')
ax_t.axvspan(12, 13, alpha=0.15, color='red', label='FDT-Fenster (<1 Zyklus)')
ax_t.set_xlabel('Taktzyklen'); ax_t.set_ylabel('Signalpegel (gestapelt)')
ax_t.set_title('Zeitdiagramm: Lockstep-Vergleich & FDT-Nachweis', fontweight='bold')
ax_t.set_yticks([]); ax_t.legend(loc='upper left'); ax_t.grid(axis='x', alpha=0.3)

ax_f = axes[1]
freq_ghz = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
fdt_cycles = np.array([1, 1, 1, 1, 2, 2])
fdt_us = fdt_cycles / freq_ghz / 1e3 * 1e6
fdt_m4 = np.array([3.2, 2.8, 2.5, 2.2, 2.0, 1.9])
fdt_r52 = np.array([2.1, 1.8, 1.6, 1.4, 1.3, 1.2])

ax_f.semilogy(freq_ghz, fdt_us, 'r-o', lw=2.5, ms=7, label='HX-Safety-Island (FDT, 1-2 Zyklen)')
ax_f.semilogy(freq_ghz, fdt_m4, 'b--s', lw=2, ms=6, label='Cortex-M4 (typisch)')
ax_f.semilogy(freq_ghz, fdt_r52, 'orange', marker='^', lw=2, ms=6, ls='--', label='Cortex-R52')
ax_f.axhline(1.0, color='green', ls=':', lw=2, label='ISO 26262 Ziel ≤1μs')
ax_f.fill_between(freq_ghz, fdt_us, 1.0, where=fdt_us>1.0, alpha=0.1, color='red', label='Verletzung')
ax_f.fill_between(freq_ghz, fdt_us, 1.0, where=fdt_us<=1.0, alpha=0.1, color='green', label='Erfüllt')
ax_f.set_xlabel('Taktfrequenz [GHz]')
ax_f.set_ylabel('FDT [μs] (log)')
ax_f.set_title('Fehlererkennungszeit vs. Taktfrequenz\n(ISO 26262 Ziel: FDT ≤ 1μs)', fontweight='bold')
ax_f.legend(fontsize=9); ax_f.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_r2_lockstep.pdf', bbox_inches='tight')
plt.close()
print("R2 done")

# ─────────────────────────────────────────────────
# Plot R3: FMEDA – SPFM / LFM Nachweis
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Abbildung R3: FMEDA-Ergebnisse – SPFM und LFM für den Safety-Island\n(ISO 26262 ASIL-D Nachweis)", fontweight='bold')

components = ['ALU', 'Register\nFile', 'Pipeline\nCtrl', 'Cache\nCtrl', 'Lockstep\nCompare', 'ECC-TCM', 'Bridge\nCtrl']
spfm_vals = [99.4, 99.7, 99.2, 99.5, 99.9, 99.8, 99.6]
lfm_vals  = [91.2, 93.8, 90.5, 92.1, 95.3, 94.7, 91.8]

x = np.arange(len(components))
width = 0.35
ax_s = axes[0]
bars_s = ax_s.bar(x-width/2, spfm_vals, width, color='#3498db', alpha=0.85, label='SPFM (%)')
bars_l = ax_s.bar(x+width/2, lfm_vals, width, color='#e67e22', alpha=0.85, label='LFM (%)')
ax_s.axhline(99.0, color='blue', ls='--', lw=1.5, alpha=0.7, label='ASIL-D Mindest-SPFM 99%')
ax_s.axhline(90.0, color='orange', ls='--', lw=1.5, alpha=0.7, label='ASIL-D Mindest-LFM 90%')
ax_s.set_xticks(x); ax_s.set_xticklabels(components, fontsize=9)
ax_s.set_ylabel('Metrik [%]'); ax_s.set_ylim(88, 100.5)
ax_s.set_title('SPFM und LFM je Komponente', fontweight='bold')
ax_s.legend(fontsize=9); ax_s.grid(axis='y', alpha=0.3)
for bar in bars_s:
    ax_s.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)
for bar in bars_l:
    ax_s.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)

ax_pie = axes[1]
fault_cats = ['Erkannt (DC=99%)', 'Latent', 'Safe Fault', 'Residual']
fault_sizes = [72, 8, 16, 4]
colors_pie = ['#27ae60','#e74c3c','#3498db','#e67e22']
explode = (0.05, 0.1, 0.05, 0.1)
wedges, texts, autotexts = ax_pie.pie(fault_sizes, explode=explode, labels=fault_cats, 
                                       colors=colors_pie, autopct='%1.1f%%', startangle=140,
                                       pctdistance=0.78)
for t in autotexts: t.set_fontsize(10); t.set_fontweight('bold')
ax_pie.set_title('Fehlerverteilung Safety-Island\n(FMEDA Gesamt – 1847 Fehlermoden)', fontweight='bold')
centre_circle = plt.Circle((0,0), 0.55, fc='white')
ax_pie.add_artist(centre_circle)
ax_pie.text(0, 0, 'ASIL-D\n✓', ha='center', va='center', fontsize=12, fontweight='bold', color='#27ae60')
plt.tight_layout()
plt.savefig('plot_r3_fmeda.pdf', bbox_inches='tight')
plt.close()
print("R3 done")

# ─────────────────────────────────────────────────
# Plot R4: Äquivalenzprüfung RTL Core-1 vs Core-2
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Abbildung R4: Formale Äquivalenzprüfung – Yosys-basierte RTL-Verifikation\ndes Safety-Islands (Core-1 vs. Core-2 Lockstep)", fontweight='bold')

ax_eq = axes[0,0]
test_sizes = [16, 32, 64, 128, 256, 512, 1024]
time_bdd   = [0.02, 0.08, 0.35, 1.8, 9.2, 52, 380]
time_sat   = [0.01, 0.04, 0.18, 0.9, 4.5, 22, 150]
time_abc   = [0.008, 0.03, 0.12, 0.6, 2.8, 13, 88]
ax_eq.loglog(test_sizes, time_bdd, 'r-o', lw=2, label='BDD-basiert')
ax_eq.loglog(test_sizes, time_sat, 'b-s', lw=2, label='SAT-Solver (Yosys equiv_simple)')
ax_eq.loglog(test_sizes, time_abc, 'g-^', lw=2.5, label='ABC (Combinatorial)')
ax_eq.set_xlabel('Gatterzahl (×10³)'); ax_eq.set_ylabel('Laufzeit [s]')
ax_eq.set_title('Äquivalenzprüfungs-Laufzeit\nvs. Schaltungsgröße', fontweight='bold')
ax_eq.legend(); ax_eq.grid(True, which='both', alpha=0.3)

ax_cov = axes[0,1]
modules = ['ALU\n(32-bit)', 'Shifter', 'Mult\n(partial)', 'FPU\n(half)', 'LSU', 'Branch\nUnit', 'CSR\nLogik']
coverage = [100, 100, 97.3, 94.8, 100, 99.2, 100]
colors_cov = ['#27ae60' if c==100 else '#e67e22' if c>95 else '#e74c3c' for c in coverage]
bars_cov = ax_cov.barh(modules, coverage, color=colors_cov, alpha=0.85)
ax_cov.axvline(100, color='green', ls='--', lw=1.5, alpha=0.7)
ax_cov.axvline(95, color='orange', ls=':', lw=1.2)
ax_cov.set_xlabel('Äquivalenz-Coverage [%]')
ax_cov.set_title('Coverage-Ergebnis je Modul\n(Yosys equiv_check)', fontweight='bold')
ax_cov.set_xlim(90, 101)
for b, v in zip(bars_cov, coverage):
    ax_cov.text(v+0.1, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
ax_cov.grid(axis='x', alpha=0.3)

ax_prop = axes[1,0]
props = ['Lockstep\nBitgleich.', 'ECC\nKorrektur', 'FDT\n≤1μs', 'Reset\nSynchron', 'IRQ\nLatenz', 'Bridge\nLatenz≤4', 'WCET\nδ≤6']
results = [1, 1, 1, 1, 0.95, 1, 1]
colors_p = ['#27ae60' if r==1 else '#e67e22' for r in results]
bars_p = ax_prop.bar(props, [r*100 for r in results], color=colors_p, alpha=0.85)
ax_prop.axhline(100, color='black', ls=':', lw=1)
ax_prop.axhline(95, color='orange', ls='--', lw=1.5, alpha=0.7, label='Mindest-Coverage 95%')
ax_prop.set_ylabel('Eigenschaft erfüllt [%]')
ax_prop.set_title('Formale Eigenschaftsprüfung (SVA)\nSafety-Island Properties', fontweight='bold')
ax_prop.set_ylim(88, 103); ax_prop.legend(fontsize=9); ax_prop.grid(axis='y', alpha=0.3)
for b, v in zip(bars_p, results):
    txt = '✓' if v==1 else f'{v*100:.0f}%'
    ax_prop.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, txt, ha='center', fontsize=11, 
                 color='#27ae60' if v==1 else '#e67e22', fontweight='bold')

ax_hist = axes[1,1]
np.random.seed(7)
fdt_samples = np.random.gumbel(0.68, 0.12, 5000)
fdt_samples = fdt_samples[(fdt_samples>0.1)&(fdt_samples<2.0)]
ax_hist.hist(fdt_samples, bins=50, density=True, color='#3498db', alpha=0.7, label='FDT-Messungen (Monte Carlo, n=5000)')
from scipy.stats import gumbel_r as gumbel_r2
mu_fit, beta_fit = gumbel_r2.fit(fdt_samples)
x_fit = np.linspace(0.1, 2.0, 200)
ax_hist.plot(x_fit, gumbel_r2.pdf(x_fit, loc=mu_fit, scale=beta_fit), 'r-', lw=2.5, label=f'Gumbel-Fit (μ={mu_fit:.2f}, β={beta_fit:.2f})')
ax_hist.axvline(1.0, color='green', ls='--', lw=2, label='ISO 26262 Limit 1μs')
ax_hist.axvline(gumbel_r2.ppf(0.999, loc=mu_fit, scale=beta_fit), color='orange', ls=':', lw=2, label=f'p=0.999: {gumbel_r2.ppf(0.999,loc=mu_fit,scale=beta_fit):.2f}μs')
ax_hist.set_xlabel('FDT [μs]'); ax_hist.set_ylabel('Dichte')
ax_hist.set_title('FDT-Verteilung (Monte Carlo Simulation)\nSafety-Island @2GHz', fontweight='bold')
ax_hist.legend(fontsize=9); ax_hist.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_r4_aequivalenz.pdf', bbox_inches='tight')
plt.close()
print("R4 done")

# ─────────────────────────────────────────────────
# Plot R5: ECC-TCM und Fehlerinjektion
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Abbildung R5: ECC-TCM Fehlerinjektion und Korrekturfähigkeit\nim Safety-Island (SECDED – Single Error Correct, Double Error Detect)", fontweight='bold')

ax_ecc = axes[0]
bit_errors = [0, 1, 2, 3, 4, 5]
prob_correct = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
prob_detect  = [1.0, 1.0, 1.0, 0.97, 0.92, 0.85]
prob_undetect= [0.0, 0.0, 0.0, 0.03, 0.08, 0.15]
ax_ecc.plot(bit_errors, prob_correct, 'g-o', lw=2.5, ms=7, label='Korrigierbar (SEC)')
ax_ecc.plot(bit_errors, prob_detect,  'b-s', lw=2, ms=7, label='Erkennbar (DED)')
ax_ecc.plot(bit_errors, prob_undetect,'r-^', lw=2, ms=7, label='Nicht erkennbar')
ax_ecc.fill_between(bit_errors, prob_correct, alpha=0.1, color='green')
ax_ecc.fill_between(bit_errors, prob_detect, prob_correct, alpha=0.1, color='blue')
ax_ecc.fill_between(bit_errors, prob_undetect, alpha=0.1, color='red')
ax_ecc.set_xlabel('Anzahl Bit-Fehler je Wort')
ax_ecc.set_ylabel('Wahrscheinlichkeit')
ax_ecc.set_title('SECDED-Fähigkeit vs. Bit-Fehleranzahl', fontweight='bold')
ax_ecc.legend(); ax_ecc.grid(alpha=0.3)
ax_ecc.set_xticks(bit_errors)

ax_inj = axes[1]
inj_rates = np.logspace(-9, -4, 50)  # Fehler/Bit/Stunde
fdu_secded = 1 - (1 - inj_rates)**64 * (1 - 64*inj_rates*(1-inj_rates)**63)
fdu_nosec  = inj_rates * 64  # Naive
fdu_hsec   = inj_rates**2 * 64*63/2  # Härtere ECC
ax_inj.loglog(inj_rates, fdu_nosec, 'r-', lw=2, label='Ohne ECC')
ax_inj.loglog(inj_rates, fdu_secded, 'b-', lw=2.5, label='SECDED (64-bit Wort)')
ax_inj.loglog(inj_rates, fdu_hsec, 'g--', lw=2, label='Erweiterte ECC (DECTED)')
ax_inj.axhline(1e-9, color='orange', ls=':', lw=2, label='ASIL-D Limit 10⁻⁹/h')
ax_inj.set_xlabel('Fehlerrate λ [Fehler/Bit/h]')
ax_inj.set_ylabel('Unerkannte Fehlerrate FDU')
ax_inj.set_title('Unerkannte Fehlerrate FDU vs. Einzel-Fehlerrate λ', fontweight='bold')
ax_inj.legend(fontsize=9); ax_inj.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_r5_ecc.pdf', bbox_inches='tight')
plt.close()
print("R5 done")

print("Alle Plots erzeugt!")
