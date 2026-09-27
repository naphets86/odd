#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots fuer: Inhomogene Wuerzverteilung bei gekochten Nudeln
Sensorische Stimulationsdynamik durch stochastische Gewuerzgradienten
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, beta, gamma as gamma_dist
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

np.random.seed(42)

# ============================================================
# PLOT 1: Wuerzintensitaetsverteilung – homogen vs. inhomogen
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n_bissen = 300
homogen = np.random.normal(5.0, 0.3, n_bissen)
inhomogen = np.random.beta(2, 2, n_bissen) * 9 + 0.5  # 0.5 - 9.5

ax = axes[0]
ax.hist(homogen, bins=30, color='steelblue', edgecolor='white', alpha=0.85, density=True)
x = np.linspace(3, 7, 200)
ax.plot(x, norm.pdf(x, 5.0, 0.3), 'navy', lw=2, label='Gauss-Fit')
ax.set_xlabel('Wuerzintensitaet $I$ [willk. Einh.]')
ax.set_ylabel('Wahrscheinlichkeitsdichte $p(I)$')
ax.set_title('(a) Homogene Wuerzung')
ax.legend()
ax.set_xlim(0, 10)

ax = axes[1]
ax.hist(inhomogen, bins=30, color='tomato', edgecolor='white', alpha=0.85, density=True)
x2 = np.linspace(0.5, 9.5, 300)
a_fit, b_fit = 2.0, 2.0
ax.plot(x2, beta.pdf((x2 - 0.5) / 9.0, a_fit, b_fit) / 9.0, 'darkred', lw=2, label='Beta-Fit')
ax.set_xlabel('Wuerzintensitaet $I$ [willk. Einh.]')
ax.set_ylabel('Wahrscheinlichkeitsdichte $p(I)$')
ax.set_title('(b) Inhomogene Wuerzung')
ax.legend()
ax.set_xlim(0, 10)

fig.suptitle('Abb. 1: Vergleich der Wuerzintensitaetsverteilungen', fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_verteilung.pdf')
plt.savefig('plot1_verteilung.png')
plt.close()
print("Plot 1 erstellt.")

# ============================================================
# PLOT 2: Hedonische Reaktionskurve (Wundt-Kurve)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

I = np.linspace(0, 10, 500)
# Wundt-Kurve: hedonic response as inverted parabola shifted
hedonic = -0.25 * (I - 5.5)**2 + 4.5 + 0.3 * np.sin(I * 1.2)

# Homogen: narrow region
I_hom = 5.0
I_hom_std = 0.3
ax.axvspan(I_hom - 2*I_hom_std, I_hom + 2*I_hom_std, alpha=0.2, color='steelblue', label='Bereich homogen (95%)')
ax.axvline(I_hom, color='steelblue', lw=2, linestyle='--', label='Mittelpunkt homogen')

# Inhomogen: wide region
I_inh_lo, I_inh_hi = 0.5, 9.5
ax.axvspan(I_inh_lo, I_inh_hi, alpha=0.12, color='tomato', label='Bereich inhomogen')

ax.plot(I, hedonic, 'k-', lw=2.5, label='Hedonische Funktion $H(I)$')

# Mark integral areas
from matplotlib.patches import Polygon as MplPolygon
mask_inh = (I >= 0.5) & (I <= 9.5)
ax.fill_between(I[mask_inh], 0, hedonic[mask_inh], alpha=0.15, color='tomato')
mask_hom = (I >= I_hom - 2*I_hom_std) & (I <= I_hom + 2*I_hom_std)
ax.fill_between(I[mask_hom], 0, hedonic[mask_hom], alpha=0.35, color='steelblue')

ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel('Wuerzintensitaet $I$ [willk. Einh.]')
ax.set_ylabel('Hedonischer Wert $H(I)$')
ax.set_title('Abb. 2: Wundt-Kurve der Geschmacksreaktion mit Stimulationsbereichen', fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(0, 10)
plt.tight_layout()
plt.savefig('plot2_wundt.pdf')
plt.savefig('plot2_wundt.png')
plt.close()
print("Plot 2 erstellt.")

# ============================================================
# PLOT 3: Sensorische Adaptation – Zeitverlauf
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

t = np.linspace(0, 60, 600)  # 60 Bissen
bissen_intervall = 1.0

# Homogen: schnelle Adaptation -> Abfall der Erregung
def adaptation_homogen(t, I_const=5.0, tau=8.0):
    return I_const * np.exp(-t / tau) + 1.5 + 0.05*np.random.randn(len(t))

# Inhomogen: wechselnde Stimuli halten Erregung aufrecht
def stimulation_inhomogen(t):
    signal = np.zeros(len(t))
    for i, ti in enumerate(t):
        base = 3.0 + 2.5*np.sin(ti * 0.35) + 1.5*np.cos(ti * 0.7 + 1.0)
        noise = 0.8*np.random.randn()
        signal[i] = np.clip(base + noise, 0.5, 9.0)
    return signal

erregung_hom = adaptation_homogen(t)
erregung_inh = stimulation_inhomogen(t)
erregung_inh_smooth = savgol_filter(erregung_inh, 31, 3)

ax1 = axes[0]
ax1.plot(t, erregung_hom, color='steelblue', lw=1.5, alpha=0.7, label='Erregung (roh)')
ax1.plot(t, savgol_filter(erregung_hom, 21, 3), 'navy', lw=2.5, label='Geglaettet')
ax1.axhline(1.5, color='gray', linestyle=':', lw=1.5, label='Adaptationsniveau')
ax1.set_ylabel('Neuronale Erregung $E(t)$')
ax1.set_title('(a) Homogene Wuerzung – Adaptationsabfall')
ax1.legend()
ax1.set_ylim(0, 8)

ax2 = axes[1]
ax2.plot(t, erregung_inh, color='tomato', lw=0.8, alpha=0.5, label='Erregung (roh)')
ax2.plot(t, erregung_inh_smooth, 'darkred', lw=2.5, label='Geglaettet')
ax2.axhline(np.mean(erregung_inh), color='gray', linestyle=':', lw=1.5, label=f'Mittleres Niveau ({np.mean(erregung_inh):.2f})')
ax2.set_xlabel('Bissen $n$')
ax2.set_ylabel('Neuronale Erregung $E(t)$')
ax2.set_title('(b) Inhomogene Wuerzung – anhaltende Stimulation')
ax2.legend()
ax2.set_ylim(0, 8)

fig.suptitle('Abb. 3: Sensorische Adaptation im Zeitverlauf', fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_adaptation.pdf')
plt.savefig('plot3_adaptation.png')
plt.close()
print("Plot 3 erstellt.")

# ============================================================
# PLOT 4: 2D Wuerzgradient auf Nudel-Oberflaechenmodell
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

nx, ny = 80, 30

# Homogen
Z_hom = np.ones((ny, nx)) * 5.0 + np.random.normal(0, 0.2, (ny, nx))

# Inhomogen: Cluster + Gradienten
Z_inh = np.zeros((ny, nx))
for _ in range(8):
    cx = np.random.randint(5, nx-5)
    cy = np.random.randint(2, ny-2)
    strength = np.random.uniform(3, 9)
    sigma_x = np.random.uniform(5, 15)
    sigma_y = np.random.uniform(2, 6)
    for i in range(nx):
        for j in range(ny):
            Z_inh[j, i] += strength * np.exp(-((i-cx)**2/(2*sigma_x**2) + (j-cy)**2/(2*sigma_y**2)))
Z_inh = np.clip(Z_inh + np.random.normal(0, 0.3, (ny, nx)), 0, 10)

spice_cmap = LinearSegmentedColormap.from_list('spice', ['#FFFDE7','#FFD54F','#FF8F00','#BF360C','#4A0000'])

ax = axes[0]
im0 = ax.imshow(Z_hom, cmap=spice_cmap, vmin=0, vmax=10, aspect='auto')
ax.set_title('(a) Homogene Wuerzverteilung')
ax.set_xlabel('Laengsachse der Nudel [Segmente]')
ax.set_ylabel('Querschnitt [Segmente]')
plt.colorbar(im0, ax=ax, label='Wuerzintensitaet $I$')

ax = axes[1]
im1 = ax.imshow(Z_inh, cmap=spice_cmap, vmin=0, vmax=10, aspect='auto')
ax.set_title('(b) Inhomogene Wuerzverteilung')
ax.set_xlabel('Laengsachse der Nudel [Segmente]')
ax.set_ylabel('Querschnitt [Segmente]')
plt.colorbar(im1, ax=ax, label='Wuerzintensitaet $I$')

fig.suptitle('Abb. 4: Zweidimensionale Wuerzgradienten auf der Nudeloberflaeche', fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_gradient2d.pdf')
plt.savefig('plot4_gradient2d.png')
plt.close()
print("Plot 4 erstellt.")

# ============================================================
# PLOT 5: Informationstheoretische Analyse – Entropie und MHI
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

sigmas = np.linspace(0.05, 3.0, 200)
# Entropie der Normalverteilung: H = 0.5 * ln(2*pi*e*sigma^2)
H_normal = 0.5 * np.log(2 * np.pi * np.e * sigmas**2)
# Mittlerer hedonischer Index: Integral H(I)*p(I) dI
# Approx: fuer breite sigma groesser, da Wundt-Kurve breiter abgetastet
I_vals = np.linspace(0, 10, 1000)
H_wundt = -0.25 * (I_vals - 5.5)**2 + 4.5 + 0.3 * np.sin(I_vals * 1.2)
H_wundt = np.maximum(H_wundt, -1.0)

MHI = []
for sigma in sigmas:
    p = norm.pdf(I_vals, 5.0, sigma)
    p /= p.sum()
    MHI.append(np.dot(H_wundt, p))

ax = axes[0]
ax.plot(sigmas, H_normal, 'navy', lw=2.5)
ax.axvline(0.3, color='steelblue', lw=2, linestyle='--', label='Homogen ($\\sigma=0.3$)')
ax.axvline(1.5, color='tomato', lw=2, linestyle='--', label='Inhomogen ($\\sigma=1.5$)')
ax.set_xlabel('Standardabweichung $\\sigma$')
ax.set_ylabel('Entropie $H$ [nat]')
ax.set_title('(a) Informationsentropie')
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(sigmas, MHI, 'darkred', lw=2.5)
ax.axvline(0.3, color='steelblue', lw=2, linestyle='--', label='Homogen')
ax.axvline(1.5, color='tomato', lw=2, linestyle='--', label='Inhomogen')
ax.set_xlabel('Standardabweichung $\\sigma$')
ax.set_ylabel('Mittl. hedon. Index $\\bar{H}$')
ax.set_title('(b) Mittlerer hedonischer Index')
ax.legend(fontsize=9)

# Varietaetsbonus
variety_bonus = np.array(MHI) + 0.4 * H_normal / max(H_normal)
ax = axes[2]
ax.plot(sigmas, variety_bonus, color='seagreen', lw=2.5)
ax.axvline(0.3, color='steelblue', lw=2, linestyle='--', label='Homogen')
ax.axvline(1.5, color='tomato', lw=2, linestyle='--', label='Inhomogen')
ax.fill_between(sigmas, variety_bonus, alpha=0.15, color='seagreen')
ax.set_xlabel('Standardabweichung $\\sigma$')
ax.set_ylabel('Gesamtgenuss-Index $G$')
ax.set_title('(c) Gesamtgenuss-Index $G = \\bar{H} + \\lambda H$')
ax.legend(fontsize=9)

fig.suptitle('Abb. 5: Informationstheoretische Analyse der Wuerzverteilung', fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_entropie.pdf')
plt.savefig('plot5_entropie.png')
plt.close()
print("Plot 5 erstellt.")

# ============================================================
# PLOT 6: Markov-Modell der Geschmackszustandsfolge
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Simuliere Zustandsfolge (LOW, MED, HIGH)
states = ['Niedrig', 'Mittel', 'Hoch']
# Homogen: fast immer MED
P_hom = np.array([[0.6, 0.35, 0.05],
                   [0.1, 0.80, 0.10],
                   [0.05, 0.35, 0.60]])
# Inhomogen: gleichmaessiger Wechsel
P_inh = np.array([[0.25, 0.45, 0.30],
                   [0.30, 0.40, 0.30],
                   [0.30, 0.45, 0.25]])

def simulate_markov(P, n=200, start=1):
    seq = [start]
    for _ in range(n-1):
        seq.append(np.random.choice(3, p=P[seq[-1]]))
    return np.array(seq)

seq_hom = simulate_markov(P_hom)
seq_inh = simulate_markov(P_inh)

colors_state = ['#5B9BD5', '#70AD47', '#FF7043']
labels_state = ['Niedrig', 'Mittel', 'Hoch']

ax = axes[0]
for s, col, lab in zip([0,1,2], colors_state, labels_state):
    mask = seq_hom == s
    ax.scatter(np.where(mask)[0], seq_hom[mask] + np.random.randn(mask.sum())*0.05,
               color=col, s=6, alpha=0.7, label=lab)
ax.set_yticks([0,1,2]); ax.set_yticklabels(labels_state)
ax.set_xlabel('Bissen $n$'); ax.set_title('(a) Homogen – Zustandsfolge')
ax.legend(markerscale=3, fontsize=9)

ax = axes[1]
for s, col, lab in zip([0,1,2], colors_state, labels_state):
    mask = seq_inh == s
    ax.scatter(np.where(mask)[0], seq_inh[mask] + np.random.randn(mask.sum())*0.05,
               color=col, s=6, alpha=0.7, label=lab)
ax.set_yticks([0,1,2]); ax.set_yticklabels(labels_state)
ax.set_xlabel('Bissen $n$'); ax.set_title('(b) Inhomogen – Zustandsfolge')
ax.legend(markerscale=3, fontsize=9)

fig.suptitle('Abb. 6: Markov-Modell der sensorischen Zustandsfolge', fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_markov.pdf')
plt.savefig('plot6_markov.png')
plt.close()
print("Plot 6 erstellt.")

# ============================================================
# PLOT 7: Statistische Auswertung – Boxplots, Varianz, Skewness
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

n_sim = 500
hom_data = np.random.normal(5.0, 0.35, n_sim)
inh_data = np.random.beta(2.2, 2.2, n_sim) * 9.0 + 0.5

ax = axes[0]
bp = ax.boxplot([hom_data, inh_data], tick_labels=['Homogen', 'Inhomogen'],
                patch_artist=True, notch=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='navy', lw=2))
bp['boxes'][1].set_facecolor('lightsalmon')
ax.set_ylabel('Wuerzintensitaet $I$')
ax.set_title('(a) Boxplot-Vergleich')

ax = axes[1]
x_bins = np.linspace(0, 10, 40)
ax.hist(hom_data, bins=x_bins, alpha=0.7, color='steelblue', density=True, label='Homogen')
ax.hist(inh_data, bins=x_bins, alpha=0.7, color='tomato', density=True, label='Inhomogen')
ax.set_xlabel('Wuerzintensitaet $I$')
ax.set_ylabel('Dichte')
ax.set_title('(b) Dichtehistogramm')
ax.legend()

ax = axes[2]
metrics = ['Mittelwert', 'Std.-Abw.', 'Skewness', 'Kurtosis']
from scipy.stats import skew, kurtosis
vals_hom = [np.mean(hom_data), np.std(hom_data), skew(hom_data), kurtosis(hom_data)]
vals_inh = [np.mean(inh_data), np.std(inh_data), skew(inh_data), kurtosis(inh_data)]
x_pos = np.arange(len(metrics))
w = 0.35
ax.bar(x_pos - w/2, vals_hom, w, color='steelblue', alpha=0.85, label='Homogen')
ax.bar(x_pos + w/2, vals_inh, w, color='tomato', alpha=0.85, label='Inhomogen')
ax.set_xticks(x_pos); ax.set_xticklabels(metrics, fontsize=9)
ax.set_title('(c) Statistische Kenngroessen')
ax.legend()
ax.axhline(0, color='k', lw=0.5)

fig.suptitle('Abb. 7: Statistische Analyse beider Wuerzregimes', fontweight='bold')
plt.tight_layout()
plt.savefig('plot7_statistik.pdf')
plt.savefig('plot7_statistik.png')
plt.close()
print("Plot 7 erstellt.")

# ============================================================
# PLOT 8: Genussdynamik – kumulativer hedonischer Wert
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n_bissen = 80
H_wundt_fn = lambda I: np.clip(-0.25*(I-5.5)**2 + 4.5 + 0.3*np.sin(I*1.2), -2, 5)

# Adaptation factor
def eat_session(regime='hom', n=80):
    hedonic = []
    adapt = 0.0
    tau_adapt = 12.0
    for i in range(n):
        if regime == 'hom':
            I = np.random.normal(5.0, 0.35)
        else:
            I = np.random.beta(2.2, 2.2) * 9.0 + 0.5
        h_raw = H_wundt_fn(I)
        adapt += (h_raw - adapt) / tau_adapt
        h_eff = h_raw - 0.4 * adapt
        hedonic.append(h_eff)
    return np.array(hedonic)

n_runs = 30
all_hom = np.array([eat_session('hom', n_bissen) for _ in range(n_runs)])
all_inh = np.array([eat_session('inh', n_bissen) for _ in range(n_runs)])

bissen = np.arange(1, n_bissen+1)

ax = axes[0]
mean_hom = all_hom.mean(0)
std_hom = all_hom.std(0)
ax.plot(bissen, mean_hom, 'navy', lw=2.5, label='Mittelwert')
ax.fill_between(bissen, mean_hom - std_hom, mean_hom + std_hom, alpha=0.25, color='steelblue', label='$\pm 1\sigma$')
ax.plot(bissen, np.cumsum(mean_hom)/bissen, 'steelblue', lw=1.5, linestyle='--', label='Kumulativer Mittelwert')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Bissen $n$'); ax.set_ylabel('Hedonischer Wert $H_n$')
ax.set_title('(a) Homogen'); ax.legend(fontsize=9)

ax = axes[1]
mean_inh = all_inh.mean(0)
std_inh = all_inh.std(0)
ax.plot(bissen, mean_inh, 'darkred', lw=2.5, label='Mittelwert')
ax.fill_between(bissen, mean_inh - std_inh, mean_inh + std_inh, alpha=0.25, color='tomato', label='$\pm 1\sigma$')
ax.plot(bissen, np.cumsum(mean_inh)/bissen, 'tomato', lw=1.5, linestyle='--', label='Kumulativer Mittelwert')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Bissen $n$'); ax.set_ylabel('Hedonischer Wert $H_n$')
ax.set_title('(b) Inhomogen'); ax.legend(fontsize=9)

fig.suptitle('Abb. 8: Hedonische Dynamik einer Essenseinheit ($N=30$ Simulationen)', fontweight='bold')
plt.tight_layout()
plt.savefig('plot8_hedonik.pdf')
plt.savefig('plot8_hedonik.png')
plt.close()
print("Plot 8 erstellt.")

print("\nAlle 8 Plots erfolgreich erstellt.")


