#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisierung der biologischen Akustik:
Formale mathematische Behandlung der menschlichen und felinen Hörfähigkeit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import erf
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# Sicherstelle hohe Qualität
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['lines.linewidth'] = 2
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

# ============================================================================
# PLOT 1: Auditive Frequenzantwort (Mensch vs. Katze)
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Frequenzbereich [Hz]
f = np.logspace(0, 5, 1000)  # 1 Hz bis 100 kHz

# Menschliches Hörvermögen: Gaussian-approximierte Gleichung
# Fletcher-Munson Kurven, psychoakustisch formalisiert
f_low_human = 20    # untere Grenzfrequenz [Hz]
f_high_human = 20000  # obere Grenzfrequenz [Hz]
Q_human = 0.7  # Qualitätsfaktor

# Hearing sensitivity (normalisiert auf maximale Empfindlichkeit)
sensitivity_human = np.exp(-0.5 * ((np.log10(f) - np.log10(1000))**2 / Q_human**2))
sensitivity_human *= np.heaviside(f - f_low_human, 0) * np.heaviside(f_high_human - f, 0)

# Katze: breiteres Spektrum, höhere obere Grenzfrequenz
f_low_cat = 55      # untere Grenzfrequenz [Hz] (besser bei tiefen Frequenzen)
f_high_cat = 77000  # obere Grenzfrequenz [Hz] (ultraschall!)
Q_cat = 0.5         # schärfere Resonanz

sensitivity_cat = np.exp(-0.5 * ((np.log10(f) - np.log10(8000))**2 / Q_cat**2))
sensitivity_cat *= np.heaviside(f - f_low_cat, 0) * np.heaviside(f_high_cat - f, 0)

ax1.semilogx(f, sensitivity_human, 'b-', linewidth=2.5, label=r'$H_{\text{human}}(f)$: Mensch')
ax1.semilogx(f, sensitivity_cat, 'r-', linewidth=2.5, label=r'$H_{\text{felid}}(f)$: Katze')

ax1.axvline(x=20, color='b', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(x=20000, color='b', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(x=55, color='r', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(x=77000, color='r', linestyle='--', alpha=0.3, linewidth=1)

# Annotationen
ax1.text(20, 0.05, '20 Hz', fontsize=8, ha='center', color='blue')
ax1.text(20000, 0.05, '20 kHz', fontsize=8, ha='center', color='blue')
ax1.text(77000, 0.15, '77 kHz', fontsize=8, ha='center', color='red')

ax1.set_xlim(1, 100000)
ax1.set_ylim(-0.05, 1.1)
ax1.set_xlabel(r'Frequenz $f$ [Hz] (logarithmische Skala)', fontsize=11)
ax1.set_ylabel(r'Relative auditive Empfindlichkeit $H(f)$ [a.u.]', fontsize=11)
ax1.set_title(r'Vergleich der auditiven Frequenzantwort: Mensch vs. Katze', fontsize=12, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('plot_auditive_frequenzantwort.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 1 erstellt: auditive Frequenzantwort")
plt.close()

# ============================================================================
# PLOT 2: Cochleäre Bandenbreite und zeitliche Auflösung
# ============================================================================

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 8))

f_center = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000])

# Cochleäre Bandenbreite nach Geschwindigkeit - proportional zu f^0.8
# (Greenwood-Funktion für Cochlea, biologisch motiviert)
bw_human = 25 * f_center**0.8  # menschliche Bandenbreite [Hz]
bw_cat = 20 * f_center**0.8 * 1.5  # Katze mit besserer Auflösung

ax2a.loglog(f_center, bw_human, 'bo-', markersize=8, linewidth=2.5, label='Mensch')
ax2a.loglog(f_center, bw_cat, 'r^-', markersize=8, linewidth=2.5, label='Katze')
ax2a.loglog(f_center, f_center*0.1, 'k--', alpha=0.3, label=r'$f \cdot 0.1$ (Referenz)')
ax2a.set_xlabel(r'Mittenkante Frequenz $f_c$ [Hz]', fontsize=11)
ax2a.set_ylabel(r'Kritische Bandbreite $\Delta f$ [Hz]', fontsize=11)
ax2a.set_title(r'Cochleäre Bandenbreite: Frequenzauflösung und spektrale Sensitivität', fontsize=12, fontweight='bold')
ax2a.grid(True, which='both', alpha=0.3)
ax2a.legend(loc='upper left', fontsize=10)
ax2a.set_facecolor('#f5f5f5')

# Zeitliche Auflösung (inverse Beziehung zur Bandbreite)
time_resolution_human = 1000 / bw_human  # [ms]
time_resolution_cat = 1000 / bw_cat       # [ms]

ax2b.semilogx(f_center, time_resolution_human, 'bo-', markersize=8, linewidth=2.5, label='Mensch')
ax2b.semilogx(f_center, time_resolution_cat, 'r^-', markersize=8, linewidth=2.5, label='Katze')
ax2b.set_xlabel(r'Mittenkante Frequenz $f_c$ [Hz]', fontsize=11)
ax2b.set_ylabel(r'Zeitliche Auflösung $\Delta t$ [ms]', fontsize=11)
ax2b.set_title(r'Zeitliche Auflösung: Zeitfenster für auditive Wahrnehmung', fontsize=12, fontweight='bold')
ax2b.grid(True, which='both', alpha=0.3)
ax2b.legend(loc='upper right', fontsize=10)
ax2b.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('plot_cochleaere_bandbreite.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 2 erstellt: cochleäre Bandenbreite")
plt.close()

# ============================================================================
# PLOT 3: Raumlokalisationsvektor und 3D-Orientierung
# ============================================================================

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5.5))

# Interaurales Zeit-Differenzen (ITD) und -Pegel-Unterschiede (ILD)
theta = np.linspace(-np.pi/2, np.pi/2, 100)  # Azimut [-90°, +90°]

# Kopfradius [cm]
R_human = 10.5
R_cat = 5.0

# ITD nach Woodworth-Schlosberg Formel: Δt = (R/c) * (sin(θ) + θ)
c_sound = 343  # m/s
ITD_human = 1000 * (R_human/100 / c_sound) * (np.sin(theta) + theta)  # [ms]
ITD_cat = 1000 * (R_cat/100 / c_sound) * (np.sin(theta) + theta)

ax3a.plot(np.degrees(theta), ITD_human*1e3, 'b-', linewidth=2.5, label='Mensch')
ax3a.plot(np.degrees(theta), ITD_cat*1e3, 'r-', linewidth=2.5, label='Katze')
ax3a.set_xlabel(r'Azimutwinkel $\theta$ [Grad]', fontsize=11)
ax3a.set_ylabel(r'Interaurale Zeit-Differenz $\Delta t_{\text{ITD}}$ [µs]', fontsize=11)
ax3a.set_title(r'ITD-basierte Richtungsanalyse (horizontale Ebene)', fontsize=12, fontweight='bold')
ax3a.grid(True, alpha=0.3)
ax3a.legend(loc='upper left', fontsize=10)
ax3a.set_facecolor('#f5f5f5')

# Interauraler Pegel-Unterschied (ILD) als Funktion der Frequenz
f_for_ild = np.array([250, 500, 1000, 2000, 4000, 8000, 16000])
# ILD in dB bei 90° Azimut (seitliche Quelle) nach Wiener-Sphere Modell
ILD_human = 20 * np.log10(np.abs(1 + 0.0001*(f_for_ild)))  # schwache Frequenzabhängigkeit
ILD_cat = 20 * np.log10(np.abs(1 + 0.0002*(f_for_ild)))    # stärkere Frequenzabhängigkeit

ax3b.semilogx(f_for_ild, ILD_human, 'bo-', markersize=8, linewidth=2.5, label='Mensch')
ax3b.semilogx(f_for_ild, ILD_cat, 'r^-', markersize=8, linewidth=2.5, label='Katze')
ax3b.set_xlabel(r'Frequenz $f$ [Hz]', fontsize=11)
ax3b.set_ylabel(r'Interauraler Pegel-Unterschied $\Delta L_{\text{ILD}}$ [dB]', fontsize=11)
ax3b.set_title(r'ILD als Funktion der Frequenz (90° Azimut)', fontsize=12, fontweight='bold')
ax3b.grid(True, which='both', alpha=0.3)
ax3b.legend(loc='lower right', fontsize=10)
ax3b.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('plot_raumlokalisierung.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 3 erstellt: Raumlokalisierung")
plt.close()

# ============================================================================
# PLOT 4: Informations-Kapazität und neuronale Verarbeitungsgeschwindigkeit
# ============================================================================

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5.5))

# Information Capacity nach Shannon: I = BW * log2(1 + SNR)
# Charakteristische Bandbreite
BW_human = 20000 - 20  # [Hz]
BW_cat_us = 77000 - 55  # [Hz]

SNR_range = np.logspace(-2, 2, 100)  # Signal-Rausch-Verhältnis [linear]

I_human = BW_human * np.log2(1 + SNR_range) / 1000  # [kbits/s]
I_cat = BW_cat_us * np.log2(1 + SNR_range) / 1000

ax4a.loglog(SNR_range, I_human, 'b-', linewidth=2.5, label=r'Mensch ($\Delta f \approx 20\,\mathrm{kHz}$)')
ax4a.loglog(SNR_range, I_cat, 'r-', linewidth=2.5, label=r'Katze ($\Delta f \approx 77\,\mathrm{kHz}$)')
ax4a.loglog(SNR_range, I_human*1.5, 'b--', alpha=0.5, linewidth=1.5, label='Mensch (80% Effizienz)')
ax4a.loglog(SNR_range, I_cat*1.5, 'r--', alpha=0.5, linewidth=1.5, label='Katze (80% Effizienz)')
ax4a.set_xlabel(r'Signal-Rausch-Verhältnis SNR [linear]', fontsize=11)
ax4a.set_ylabel(r'Auditive Informations-Kapazität $C$ [kbit/s]', fontsize=11)
ax4a.set_title(r'Shannon-Informationkapazität des auditiven Systems', fontsize=12, fontweight='bold')
ax4a.grid(True, which='both', alpha=0.3)
ax4a.legend(loc='upper left', fontsize=9.5)
ax4a.set_facecolor('#f5f5f5')

# Neuronale Verarbeitungsgeschwindigkeit
# Neuron frequency following response (FFR) bis zu ~200 Hz (synchrone Feuerung)
f_neural = np.array([50, 100, 200, 500, 1000, 2000, 4000])
sync_fraction_human = np.array([0.95, 0.90, 0.70, 0.30, 0.10, 0.05, 0.02])
sync_fraction_cat = np.array([0.97, 0.95, 0.85, 0.50, 0.25, 0.12, 0.05])

ax4b.semilogx(f_neural, sync_fraction_human*100, 'bo-', markersize=8, linewidth=2.5, label='Mensch')
ax4b.semilogx(f_neural, sync_fraction_cat*100, 'r^-', markersize=8, linewidth=2.5, label='Katze')
ax4b.axhline(y=50, color='k', linestyle='--', alpha=0.3, linewidth=1, label='50% Synchronisation')
ax4b.set_xlabel(r'Neuronale Feuerfrequenz [Hz]', fontsize=11)
ax4b.set_ylabel(r'Synchronisationsfraktion [%]', fontsize=11)
ax4b.set_title(r'Neuronale Synchronisation und Phasen-Locking', fontsize=12, fontweight='bold')
ax4b.grid(True, which='both', alpha=0.3)
ax4b.legend(loc='upper right', fontsize=10)
ax4b.set_facecolor('#f5f5f5')
ax4b.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('plot_informationskapazitaet.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 4 erstellt: Informationskapazität")
plt.close()

# ============================================================================
# PLOT 5: 3D Raumorientierung - Polar-Plot der Richtungsempfindlichkeit
# ============================================================================

fig5 = plt.figure(figsize=(12, 5.5))

# Zwei Subplots mit Polar-Koordinaten
ax5a = fig5.add_subplot(121, projection='polar')
ax5b = fig5.add_subplot(122, projection='polar')

# Richtungsmuster (polar pattern) - simuliert mit kombiniertem ITD/ILD
angles = np.linspace(0, 2*np.pi, 100)

# Vereinfachtes Richtungsmuster: Kosinus-ähnliche Funktion
pattern_human = 0.5 + 0.5 * np.cos(angles)  # menschliches Muster
pattern_cat = 0.4 + 0.6 * np.cos(2*angles)  # Katze mit höherer Richtungsspezifität

ax5a.plot(angles, pattern_human, 'b-', linewidth=2.5, label='Mensch')
ax5a.fill(angles, pattern_human, 'b', alpha=0.2)
ax5a.set_title(r'Auditive Richtungsempfindlichkeit: Mensch', fontsize=11, fontweight='bold', pad=20)
ax5a.set_ylim(0, 1)
ax5a.grid(True, alpha=0.3)

ax5b.plot(angles, pattern_cat, 'r-', linewidth=2.5, label='Katze')
ax5b.fill(angles, pattern_cat, 'r', alpha=0.2)
ax5b.set_title(r'Auditive Richtungsempfindlichkeit: Katze', fontsize=11, fontweight='bold', pad=20)
ax5b.set_ylim(0, 1)
ax5b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_raumorientierung_polar.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 5 erstellt: Raumorientierung (Polar)")
plt.close()

# ============================================================================
# PLOT 6: Vergleich Hörverlust: Verursachte Zeit-räumliche Desorientation
# ============================================================================

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5.5))

# Szenario: Audiometrischer Hörverlust auf verschiedenen Frequenzen
freq_aud = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
# Typischer altersgebundener Hörverlust (dB HL)
hearing_loss = np.array([0, 5, 10, 15, 25, 35, 45])

# Resultierendes Lage-Fehler-Profil (localization error in degrees)
localization_error_no_loss = 2 + 1*np.sin(2*np.pi*freq_aud/8000)  # kleine fehler
localization_error_with_loss = 2 + 3*np.sin(2*np.pi*freq_aud/8000) + hearing_loss/10

ax6a.semilogx(freq_aud, hearing_loss, 'ko-', markersize=8, linewidth=2.5)
ax6a.fill_between(freq_aud, 0, hearing_loss, alpha=0.3, color='red')
ax6a.set_xlabel(r'Frequenz $f$ [Hz]', fontsize=11)
ax6a.set_ylabel(r'Hörverlust $L_h$ [dB HL]', fontsize=11)
ax6a.set_title(r'Beispiel: Audiometrischer Hörverlust mit dem Alter', fontsize=12, fontweight='bold')
ax6a.grid(True, which='both', alpha=0.3)
ax6a.set_facecolor('#f5f5f5')

# Mittlerer Lokalisierungsfehler
ax6b.semilogx(freq_aud, localization_error_no_loss, 'b-o', markersize=8, linewidth=2.5, 
              label='Normales Gehör')
ax6b.semilogx(freq_aud, localization_error_with_loss, 'r-^', markersize=8, linewidth=2.5, 
              label='Mit Hörverlust')
ax6b.fill_between(freq_aud, localization_error_no_loss, localization_error_with_loss, 
                   alpha=0.2, color='red')
ax6b.set_xlabel(r'Frequenz $f$ [Hz]', fontsize=11)
ax6b.set_ylabel(r'Mittlerer Lokalisierungsfehler $\epsilon_{\theta}$ [°]', fontsize=11)
ax6b.set_title(r'Induzierter räumlicher Orientierungsfehler durch Hörverlust', fontsize=12, fontweight='bold')
ax6b.grid(True, which='both', alpha=0.3)
ax6b.legend(loc='upper left', fontsize=10)
ax6b.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('plot_hoerverlust_desorientierung.pdf', dpi=300, bbox_inches='tight')
print("✓ Plot 6 erstellt: Hörverlust und Desorientation")
plt.close()

print("\n✓ Alle 6 Plots wurden erfolgreich erstellt!")
print("  Speicherpfad: ")
