#!/usr/bin/env python3
"""
Visualisierungen für die Arbeit:
"KI-basierte Bedrohungen und moderne Cybersicherheit"
Erstellt Plots für alle wichtigen Sätze und Konzepte.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Stil setzen
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

# ============================================================
# Abbildung 1: VC-Theorie - Lernfehler vs. Trainingsmenge
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

m = np.linspace(10, 500, 100)  # Trainingsmenge
d_low = 5  # Niedrige VC-Dimension
d_high = 20  # Hohe VC-Dimension

# Obere Schranke für Lernfehler nach Lemma 3.1
error_low = np.sqrt(d_low * np.log(m) / m) + np.exp(-m)
error_high = np.sqrt(d_high * np.log(m) / m) + np.exp(-m)

ax.plot(m, error_low, 'o-', linewidth=2.5, markersize=4, 
        label=f'VC-Dimension d = {d_low} (z.B. einfaches Modell)', color='#1f77b4')
ax.plot(m, error_high, 's-', linewidth=2.5, markersize=4,
        label=f'VC-Dimension d = {d_high} (z.B. Deep Neural Network)', color='#ff7f0e')

# Lernzahlen-Satz (Satz 3.2)
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Akzeptable Fehlerrate (5%)')
ax.axhline(y=0.01, color='darkred', linestyle='--', linewidth=2, label='Strikte Fehlerrate (1%)')

ax.set_xlabel('Trainingsmenge |T| (Anzahl Beobachtungen)', fontsize=12, fontweight='bold')
ax.set_ylabel('Lernfehler $P_{\\mathrm{Lernfehler}}$', fontsize=12, fontweight='bold')
ax.set_title('Satz 3.2: KI-Angreifer-Lernfehler vs. Trainingsmenge\nVC-Theorie (Lemma 3.1)', 
             fontsize=13, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.4])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig1_vc_theory.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig1_vc_theory.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 1: fig1_vc_theory.pdf")

# ============================================================
# Abbildung 2: Redundanzeffekt (Satz 4.3)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Linker Plot: Erfolgswahrscheinlichkeit vs. Redundanzgrad
n_channels = np.arange(1, 8)
p_compromised = 0.3  # 30% pro Kanal

success_prob = []
for n in n_channels:
    # Binomialverteilung: P(mindestens ceil(n/2)+1 Kanäle nicht kompromittiert)
    # = sum von k=ceil(n/2)+1 bis n von C(n,k) * (1-p)^k * p^(n-k)
    k_min = int(np.ceil(n / 2)) + 1
    prob = 0
    for k in range(k_min, n + 1):
        from scipy.special import comb
        prob += comb(n, k, exact=True) * (1 - p_compromised)**k * p_compromised**(n-k)
    success_prob.append(prob)

ax1.bar(n_channels, success_prob, color='#2ca02c', edgecolor='black', linewidth=1.5, alpha=0.7)
ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% Schwelle')
ax1.set_xlabel('Anzahl redundanter Kanäle (n)', fontsize=11, fontweight='bold')
ax1.set_ylabel('$P_{\\mathrm{KI-Robust}}(n, 0.3)$ - Abwehrerfolgsrate', fontsize=11, fontweight='bold')
ax1.set_title('Satz 4.3 (links): Redundanzeffekt\nMajority-Vote bei p = 30% pro Kanal', 
              fontsize=12, fontweight='bold')
ax1.set_xticks(n_channels)
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()

# Rechter Plot: Erfolgsrate vs. Kompromisskompromittierungsrate
p_range = np.linspace(0, 0.5, 50)
for n in [1, 3, 5, 7]:
    success_rates = []
    for p in p_range:
        k_min = int(np.ceil(n / 2)) + 1
        prob = 0
        for k in range(k_min, n + 1):
            from scipy.special import comb
            prob += comb(n, k, exact=True) * (1 - p)**k * p**(n-k)
        success_rates.append(prob)
    ax2.plot(p_range * 100, success_rates, 'o-', linewidth=2.5, markersize=4, label=f'n = {n} Kanäle')

ax2.set_xlabel('Kompromittierungsrate p pro Kanal (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('$P_{\\mathrm{KI-Robust}}(n,p)$ - Abwehrerfolgsrate', fontsize=11, fontweight='bold')
ax2.set_title('Satz 4.3 (rechts): Robustheit vs. Angriffsrate\nMajority Vote mit variablem n', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_redundancy_effect.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig2_redundancy_effect.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 2: fig2_redundancy_effect.pdf")

# ============================================================
# Abbildung 3: Decoy-Injektions-Effekt (Satz 5.1)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

m_obs = np.linspace(100, 2000, 50)  # Anzahl Beobachtungen
alpha_values = [0, 0.2, 0.33, 0.5]  # Anteil Decoys

# Linker Plot: Lernfehler vs. Beobachtungen mit verschiedenen Decoy-Raten
for alpha in alpha_values:
    # Nach Satz 5.1: Lernfehler >= alpha
    learning_error = [alpha + np.sqrt(np.log(m_val) / m_val) for m_val in m_obs]
    label = f'α = {int(alpha*100)}% Decoys' if alpha > 0 else 'α = 0% (kein Decoy)'
    linestyle = '--' if alpha == 0 else '-'
    ax1.plot(m_obs, learning_error, linestyle=linestyle, linewidth=2.5, markersize=4, label=label)

ax1.set_xlabel('Anzahl Beobachtungen (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Lernfehler $P_{\\mathrm{Lernfehler}}$', fontsize=11, fontweight='bold')
ax1.set_title('Satz 5.1: Decoy-Injektions-Effekt\nAufschlag des Lernfehlers durch False Positives', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 0.7])

# Rechter Plot: Angriffserfolgsrate vs. Decoy-Quote
decoy_rate = np.linspace(0, 0.5, 50)
attack_success_base = 0.5  # Basis-Angriffserfolgsrate ohne Decoys

attack_success = []
for alpha in decoy_rate:
    # Angreifer kann Decoys nicht unterscheiden
    success = attack_success_base * (1 - alpha)  # Effektive Erfolgsrate
    attack_success.append(success * 100)

ax2.fill_between(decoy_rate * 100, 0, attack_success, alpha=0.3, color='red')
ax2.plot(decoy_rate * 100, attack_success, 'o-', linewidth=2.5, markersize=5, color='darkred')

# Optimalzone
optimal_alpha = 0.33
ax2.axvline(x=optimal_alpha * 100, color='green', linestyle='--', linewidth=2, 
            label=f'Optimal: α = {int(optimal_alpha*100)}%')

ax2.set_xlabel('Decoy-Injektionsrate α (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Effektive Angriffserfolgsrate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Effekt: Reduktion der Angriffserfolgsrate durch Decoys\nOptimum bei 2 falsche Alerts pro 1 echten', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 60])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_decoy_injection.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig3_decoy_injection.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 3: fig3_decoy_injection.pdf")

# ============================================================
# Abbildung 4: Vergleich Verteidigungsmechanismen
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

mechanisms = ['Keine\nVerteidigung', 'Einfache IDS\n(Layer 1)', 'Redundante\nIDS (3x)', 
              '+ Anomaly\nDetection', '+ Decoys\n(α=33%)', '+ Crypto-\nAgility',
              'Vollständiges\nSystem (Bechtle)']
success_rates = [50.0, 30.0, 7.5, 4.5, 1.5, 0.5, 1e-145]  # Prozent bzw. extrem klein

# Log-Skala für bessere Visualisierung
colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4', '#9467bd']

# Für die letzte Statistik log-Skala
success_rates_log = []
for rate in success_rates[:-1]:
    success_rates_log.append(np.log10(rate) if rate > 0 else -20)
success_rates_log.append(-145)  # Bechtle: ~10^-145

bars = ax.bar(range(len(mechanisms)), success_rates_log, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Labels
ax.set_ylabel('Log10(Angriffserfolgsrate) [log. Skala]', fontsize=12, fontweight='bold')
ax.set_xlabel('Verteidigungsmechanismus', fontsize=12, fontweight='bold')
ax.set_title('Vergleich: Schrittweise Erhöhung der Verteidigungstiefe\nBased on Sätze 4.3, 5.1, und Algorithmus 7.1', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(range(len(mechanisms)))
ax.set_xticklabels(mechanisms, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Beschriftung der Bars
for i, (bar, rate) in enumerate(zip(bars, success_rates)):
    if rate > 1e-100:
        label = f'{rate:.1f}%' if rate >= 1 else f'{rate:.2e}'
    else:
        label = '~10⁻¹⁴⁵'
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# Sicherheitsschwelle
ax.axhline(y=-6, color='green', linestyle='--', linewidth=2, label='Praktische Sicherheitsschwelle (10⁻⁶)')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_defense_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig4_defense_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 4: fig4_defense_comparison.pdf")

# ============================================================
# Abbildung 5: Bechtle-Szenario - 5-Layer Defense Architektur
# ============================================================
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Titel
ax.text(5, 9.5, 'Fünfschichtige KI-resiliente Sicherheitsarchitektur', 
        ha='center', fontsize=14, fontweight='bold')
ax.text(5, 9.0, 'Bechtle AG - Praktische Implementierung', 
        ha='center', fontsize=11, style='italic', color='gray')

# Layer-Definitionen
layers = [
    ('Layer 1: Diverse IDS', 'Suricata + Zeek + Proprietary\np₁ = 20% pro Kanal', 0.2, '#ff7f0e'),
    ('Layer 2: Behavioral Anomaly', 'Isolation Forest mit täglichem\nParameter-Rotation', 1.8, '#2ca02c'),
    ('Layer 3: Decoy-Injection', 'Zu jedem echten Alert\n2 falsche Alerts', 3.4, '#1f77b4'),
    ('Layer 4: Crypto-Agility', 'AES-256 + ChaCha20\nParallel implementiert', 5.0, '#d62728'),
    ('Layer 5: Human-in-the-Loop', 'Manuelle Review durch\nSicherheitsanalysten (p₅ = 5%)', 6.6, '#9467bd'),
]

y_positions = [7.5, 6.0, 4.5, 3.0, 1.5]
for (title, desc, x_offset, color), y_pos in zip(layers, y_positions):
    # Box
    box = FancyBboxPatch((0.3, y_pos - 0.35), 9.4, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color, alpha=0.3, linewidth=2)
    ax.add_patch(box)
    
    # Text
    ax.text(0.5, y_pos + 0.15, title, fontsize=11, fontweight='bold')
    ax.text(0.5, y_pos - 0.2, desc, fontsize=9, style='italic', color='gray')

# Pfeil und Resultat
ax.arrow(5, 1.0, 0, -0.35, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
ax.text(5, 0.3, '$P_{\\mathrm{Angriff}} \\approx 10^{-145}$', 
        ha='center', fontsize=13, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig5_bechtle_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig5_bechtle_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 5: fig5_bechtle_architecture.pdf")

# ============================================================
# Abbildung 6: Informationstheoretische Äquivokation (Satz 6.2)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Linker Plot: Äquivokation als Funktion der Schlüssellänge
key_length = np.arange(8, 257, 8)  # Bits
# Anzahl mögliche Parametersätze: 2^key_length
equivocation = 2.0 ** (key_length / 8)  # In Log-Skala vereinfacht

ax1.semilogy(key_length, equivocation, 'o-', linewidth=2.5, markersize=6, color='#1f77b4')
ax1.set_xlabel('Schlüssellänge (Bits)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Anzahl mögliche Parameter-Sätze (2^k)', fontsize=11, fontweight='bold')
ax1.set_title('Satz 6.2: Informationstheoretische Äquivokation\nAngreifer kann echte Parameter nicht unterscheiden', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')

# Rechter Plot: Fehler beim Raten vs. Schlüssellänge
# Mit AES-256 kann der Angreifer mit Brute-Force scheitern
key_bits = np.array([64, 128, 192, 256])
bruteforce_attempts = 2.0 ** key_bits
success_rate_bruteforce = 1.0 / bruteforce_attempts * 100

ax2.bar(range(len(key_bits)), np.log10(1.0 / bruteforce_attempts), 
        color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
        edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_xticks(range(len(key_bits)))
ax2.set_xticklabels([f'{kb}-bit\nKey' for kb in key_bits])
ax2.set_ylabel('Log10(Erfolgswahrscheinlichkeit Brute-Force)', fontsize=11, fontweight='bold')
ax2.set_title('Praktische Unmöglichkeit: Brute-Force Angriffe\nGegen informationstheoretisch sichere Systeme', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=-100, color='red', linestyle='--', linewidth=2, label='Praktikal unmöglich')
ax2.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig6_equivocation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig6_equivocation.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 6: fig6_equivocation.pdf")

# ============================================================
# Abbildung 7: Zeitliche Entwicklung - Trainingsphase vs. Exploitierungsphase
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

time = np.linspace(0, 1000, 1000)  # Zeit in Stunden
# Phase 1: Training (0-400 Stunden)
# Phase 2: Exploitation (400-1000 Stunden)

attack_success_phase1 = 0.05 * (1 - np.exp(-time[time <= 400] / 100))  # Exponentielles Lernen
attack_success_phase2 = 0.05 + (0.3 - 0.05) * (1 - np.exp(-(time[time > 400] - 400) / 50))  # Schnellere Exploitierung

time_phase1 = time[time <= 400]
time_phase2 = time[time > 400]

ax.plot(time_phase1, attack_success_phase1 * 100, linewidth=3, color='#ff7f0e', label='Phase 1: Training')
ax.plot(time_phase2, attack_success_phase2 * 100, linewidth=3, color='#d62728', label='Phase 2: Exploitation')

# Verteidigungsintervention
ax.axvline(x=400, color='green', linestyle='--', linewidth=2.5, label='Detektion des Angriffs')

# Mit Verteidigung (Redundanz + Decoys)
defended_success = attack_success_phase2 * 0.15  # 85% Reduktion durch Verteidigung
ax.plot(time_phase2, defended_success * 100, linewidth=3, color='#2ca02c', 
        linestyle='--', label='Mit Redundanz + Decoys')

ax.set_xlabel('Zeit (Stunden)', fontsize=12, fontweight='bold')
ax.set_ylabel('Angriffserfolgswahrscheinlichkeit (%)', fontsize=12, fontweight='bold')
ax.set_title('Def. 2.2: Adaptive KI-Angreifer - Trainings- und Exploitierungsphase\nMit und ohne Verteidigungsmechanismen', 
             fontsize=13, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.fill_between(time_phase1, 0, 40, alpha=0.1, color='orange', label='_nolegend_')
ax.fill_between(time_phase2[time_phase2 <= 600], 0, 40, alpha=0.1, color='red', label='_nolegend_')
ax.text(200, 32, 'Trainingsphase\n(Angreifer lernt)', fontsize=10, ha='center', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(500, 32, 'Exploitierungsphase\n(Angreifer greift an)', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig7_attack_timeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig7_attack_timeline.png', dpi=300, bbox_inches='tight')
print("✓ Abbildung 7: fig7_attack_timeline.pdf")

print("\n" + "="*60)
print("Alle Visualisierungen erfolgreich erstellt!")
print("="*60)
print("Verfügbare Dateien:")
print("  - fig1_vc_theory.pdf/png")
print("  - fig2_redundancy_effect.pdf/png")
print("  - fig3_decoy_injection.pdf/png")
print("  - fig4_defense_comparison.pdf/png")
print("  - fig5_bechtle_architecture.pdf/png")
print("  - fig6_equivocation.pdf/png")
print("  - fig7_attack_timeline.pdf/png")
print("="*60)
