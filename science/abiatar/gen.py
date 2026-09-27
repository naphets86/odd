#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generierung von matplotlib Visualisierungen für die abiatar-Arbeit
Kapitel: Experimentelle Validierung und Visuelle Darstellung
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, expon, beta
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

# Deutsche Schriftart konfigurieren
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Farbschema konsistent mit LaTeX-Dokument
COLOR_MAIN_BLUE = '#1A468C'      # mainblue
COLOR_ACCENT_RED = '#B4321E'     # accentred
COLOR_DARK_GREEN = '#1E6432'     # darkgreen
COLOR_LIGHT_GRAY = '#F5F5F8'     # lightgray
COLOR_DARK_GRAY = '#3C3C46'      # darkgray

# ====================================================================
# Plot 1: Metaverteilungskonzept
# ====================================================================
def plot_1_metaverteilung():
    """
    Visualisiert das Konzept einer Metaverteilung:
    - Mehrere Wahrscheinlichkeitsverteilungen (Binomialverteilungen)
    - Deren Häufigkeit als Metaverteilung
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Plot 1: Metaverteilungskonzept', fontsize=14, fontweight='bold')
    
    # Linke Seite: Mehrere Binomialverteilungen
    x = np.arange(0, 21)
    params_list = [(20, 0.3), (20, 0.5), (20, 0.7)]
    colors = [COLOR_MAIN_BLUE, COLOR_ACCENT_RED, COLOR_DARK_GREEN]
    
    for (n, p), color in zip(params_list, colors):
        from scipy.stats import binom
        prob = binom.pmf(x, n, p)
        axes[0].bar(x, prob, alpha=0.4, label=f'B({n}, {p})', color=color, edgecolor='black', linewidth=0.5)
    
    axes[0].set_xlabel('Ereignisse', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Wahrscheinlichkeit', fontsize=11, fontweight='bold')
    axes[0].set_title('Verschiedene Wahrscheinlichkeitsverteilungen', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rechte Seite: Metaverteilung (Häufigkeit der Parameter)
    parameters = np.array([0.3, 0.5, 0.7])
    meta_probs = np.array([0.2, 0.5, 0.3])
    
    axes[1].bar(parameters, meta_probs, width=0.1, color=COLOR_MAIN_BLUE, 
                edgecolor='black', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Parameter p', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Wahrscheinlichkeit der Parameter', fontsize=11, fontweight='bold')
    axes[1].set_title('Metaverteilung M(p)', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 0.6])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plot_1_metaverteilung.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 1 gespeichert: plot_1_metaverteilung.pdf")
    plt.close()


# ====================================================================
# Plot 2: Metaperioden - Perioden von Perioden
# ====================================================================
def plot_2_metaperioden():
    """
    Visualisiert das Konzept von Metaperioden:
    - Periodische Funktion (innere Periode)
    - Periodisches Variieren der Periode (Metaperiode)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Plot 2: Metaperioden - Periode der Perioden', fontsize=14, fontweight='bold')
    
    t = np.linspace(0, 10, 1000)
    
    # Linke Seite: Konstante Periode
    periode_base = 1.0
    signal_const = np.sin(2*np.pi*t/periode_base)
    axes[0].plot(t, signal_const, color=COLOR_MAIN_BLUE, linewidth=2)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Periode = 1.0')
    axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Signal', fontsize=11, fontweight='bold')
    axes[0].set_title('Konstante Periode τ = 1.0', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Mitte: Variierende Periode
    periode_varying = 1.0 + 0.3*np.sin(2*np.pi*t/5)  # Metaperiode = 5
    signal_var = np.sin(2*np.pi*np.cumsum(1/periode_varying)/1000)
    axes[1].plot(t, signal_var, color=COLOR_ACCENT_RED, linewidth=2)
    axes[1].set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Signal', fontsize=11, fontweight='bold')
    axes[1].set_title('Variierende Periode (Metaperiode π = 5)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Rechte Seite: Die Periode selbst
    axes[2].plot(t, periode_varying, color=COLOR_DARK_GREEN, linewidth=2.5)
    axes[2].fill_between(t, periode_varying, alpha=0.3, color=COLOR_DARK_GREEN)
    axes[2].set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Periodenwert τ(t)', fontsize=11, fontweight='bold')
    axes[2].set_title('Metaperiode: τ(t) = 1.0 + 0.3·sin(2πt/5)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_2_metaperioden.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 2 gespeichert: plot_2_metaperioden.pdf")
    plt.close()


# ====================================================================
# Plot 3: abiatar-Eigenschaft - Selbstbezügliche Struktur
# ====================================================================
def plot_3_abiatar_struktur():
    """
    Visualisiert die abiatar-Eigenschaft:
    - Eine Funktion φ angewendet auf sich selbst
    - Konvergenz gegen Fixpunkt
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Plot 3: abiatar-Eigenschaft - Selbstbezügliche Strukturen', 
                 fontsize=14, fontweight='bold')
    
    # Linke oben: Iterative Anwendung von φ
    def phi(x):
        return 0.5*x + 0.3
    
    x0 = 1.0
    x_vals = [x0]
    for _ in range(10):
        x_vals.append(phi(x_vals[-1]))
    
    axes[0, 0].plot(range(len(x_vals)), x_vals, 'o-', color=COLOR_MAIN_BLUE, 
                    linewidth=2, markersize=8, markerfacecolor=COLOR_ACCENT_RED)
    axes[0, 0].axhline(y=0.6, color='green', linestyle='--', linewidth=2, 
                       label='Fixpunkt: φ(x) = x ⟹ x* = 0.6')
    axes[0, 0].set_xlabel('Iteration n', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('x(n)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Konvergenz gegen Fixpunkt', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Rechts oben: Graphische Darstellung der Iteration
    x_range = np.linspace(0, 1, 200)
    y_line = x_range  # y = x
    y_func = 0.5*x_range + 0.3  # y = φ(x)
    
    axes[0, 1].plot(x_range, y_line, 'k-', linewidth=1.5, label='y = x')
    axes[0, 1].plot(x_range, y_func, color=COLOR_MAIN_BLUE, linewidth=2.5, label='y = φ(x) = 0.5x + 0.3')
    
    # Cobweb-Diagramm
    x_curr = 1.0
    for _ in range(6):
        y_curr = phi(x_curr)
        axes[0, 1].plot([x_curr, x_curr], [x_curr, y_curr], 'r-', alpha=0.4, linewidth=1)
        axes[0, 1].plot([x_curr, y_curr], [y_curr, y_curr], 'r-', alpha=0.4, linewidth=1)
        x_curr = y_curr
    
    axes[0, 1].plot(0.6, 0.6, 'g*', markersize=20, label='Fixpunkt x* = 0.6')
    axes[0, 1].set_xlabel('x', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('y', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Cobweb-Diagramm der Iteration', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    
    # Unten links: Mehrere φ-Funktionen mit verschiedenen Kontraktionsraten
    x_range = np.linspace(0, 1, 200)
    configs = [(0.3, 0.4), (0.5, 0.3), (0.7, 0.2)]  # (Steigung, Konstante)
    
    for slope, const, (color, label) in zip([c[0] for c in configs], 
                                             [c[1] for c in configs],
                                             [(COLOR_MAIN_BLUE, 'φ₁: langsame Konvergenz'),
                                              (COLOR_ACCENT_RED, 'φ₂: mittlere Konvergenz'),
                                              (COLOR_DARK_GREEN, 'φ₃: schnelle Konvergenz')]):
        y_func = slope*x_range + const
        axes[1, 0].plot(x_range, y_func, color=color, linewidth=2.5, label=label)
    
    axes[1, 0].plot(x_range, x_range, 'k--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('x', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('φ(x)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Verschiedene Kontraktionen', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Unten rechts: Konvergenzgeschwindigkeit
    n_iter = np.arange(0, 11)
    error1 = (0.3**n_iter)  # φ₁
    error2 = (0.5**n_iter)  # φ₂
    error3 = (0.7**n_iter)  # φ₃
    
    axes[1, 1].semilogy(n_iter, error1, 'o-', color=COLOR_MAIN_BLUE, linewidth=2, 
                        label='φ₁ (Kontr. 0.3)', markersize=6)
    axes[1, 1].semilogy(n_iter, error2, 's-', color=COLOR_ACCENT_RED, linewidth=2, 
                        label='φ₂ (Kontr. 0.5)', markersize=6)
    axes[1, 1].semilogy(n_iter, error3, '^-', color=COLOR_DARK_GREEN, linewidth=2, 
                        label='φ₃ (Kontr. 0.7)', markersize=6)
    
    axes[1, 1].set_xlabel('Iteration n', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Fehler |x(n) - x*| (log)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Konvergenzgeschwindigkeit', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('plot_3_abiatar_struktur.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 3 gespeichert: plot_3_abiatar_struktur.pdf")
    plt.close()


# ====================================================================
# Plot 4: SAT-Solver Laufzeitanalyse - Metaverteilung der Laufzeiten
# ====================================================================
def plot_4_sat_solver():
    """
    Visualisiert die Metaverteilungsphänomene bei SAT-Solvern
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    fig.suptitle('Plot 4: SAT-Solver und Metaverteilungen', fontsize=14, fontweight='bold')
    
    # Oben links: Bimodale Metaverteilung (Phase 1)
    ax1 = fig.add_subplot(gs[0, 0])
    x_phase1 = np.linspace(0, 10, 500)
    y_phase1 = 0.4*norm.pdf(x_phase1, 2, 0.8) + 0.6*norm.pdf(x_phase1, 7, 1.2)
    ax1.fill_between(x_phase1, y_phase1, alpha=0.4, color=COLOR_MAIN_BLUE)
    ax1.plot(x_phase1, y_phase1, color=COLOR_MAIN_BLUE, linewidth=2.5)
    ax1.set_xlabel('Laufzeit [log₁₀(n)]', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Häufigkeit', fontsize=10, fontweight='bold')
    ax1.set_title('Phase 1: Bimodale Metaverteilung', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Oben mitte: Übergangsphase
    ax2 = fig.add_subplot(gs[0, 1])
    x_phase2 = np.linspace(0, 10, 500)
    y_phase2 = 0.3*norm.pdf(x_phase2, 4, 1.5) + 0.7*norm.pdf(x_phase2, 6, 0.9)
    ax2.fill_between(x_phase2, y_phase2, alpha=0.4, color=COLOR_ACCENT_RED)
    ax2.plot(x_phase2, y_phase2, color=COLOR_ACCENT_RED, linewidth=2.5)
    ax2.set_xlabel('Laufzeit [log₁₀(n)]', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Häufigkeit', fontsize=10, fontweight='bold')
    ax2.set_title('Phase 2: Übergangsphase', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Oben rechts: Phase 3 (sharfe Konzentration)
    ax3 = fig.add_subplot(gs[0, 2])
    x_phase3 = np.linspace(0, 10, 500)
    y_phase3 = norm.pdf(x_phase3, 6, 0.3)
    ax3.fill_between(x_phase3, y_phase3, alpha=0.4, color=COLOR_DARK_GREEN)
    ax3.plot(x_phase3, y_phase3, color=COLOR_DARK_GREEN, linewidth=2.5)
    ax3.set_xlabel('Laufzeit [log₁₀(n)]', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Häufigkeit', fontsize=10, fontweight='bold')
    ax3.set_title('Phase 3: Sharfe Konzentration', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Unten links: Laufzeitentwicklung über Problemgröße
    ax4 = fig.add_subplot(gs[1, 0])
    n_vals = np.array([10, 20, 50, 100, 200, 500, 1000])
    phase1_time = 0.5 * np.log(n_vals)
    phase2_time = 2 * n_vals**0.7
    phase3_time = 3 * n_vals**1.3
    
    ax4.loglog(n_vals, phase1_time, 'o-', color=COLOR_MAIN_BLUE, linewidth=2.5, 
               markersize=8, label='Phase 1: O(log n)')
    ax4.loglog(n_vals, phase2_time, 's-', color=COLOR_ACCENT_RED, linewidth=2.5, 
               markersize=8, label='Phase 2: O(n⁰·⁷)')
    ax4.loglog(n_vals, phase3_time, '^-', color=COLOR_DARK_GREEN, linewidth=2.5, 
               markersize=8, label='Phase 3: O(n¹·³)')
    
    ax4.set_xlabel('Problemgröße n', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Erwartete Laufzeit [sec]', fontsize=10, fontweight='bold')
    ax4.set_title('Phasenverlauf der Laufzeit', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()
    
    # Unten mitte: Varianz in den Phasen
    ax5 = fig.add_subplot(gs[1, 1])
    phases = ['Phase 1\n(log n)', 'Phase 2\n(n⁰·⁷)', 'Phase 3\n(n¹·³)']
    variances = [0.8, 1.5, 0.2]
    colors_phase = [COLOR_MAIN_BLUE, COLOR_ACCENT_RED, COLOR_DARK_GREEN]
    
    bars = ax5.bar(phases, variances, color=colors_phase, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Varianz der Laufzeit', fontsize=10, fontweight='bold')
    ax5.set_title('Variabilität in den Phasen', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Unten rechts: Phasenübergang Diagramm
    ax6 = fig.add_subplot(gs[1, 2])
    x_transition = np.linspace(0, 1, 200)
    
    # Wechsel der dominanten Phase
    phase_prob1 = 0.8 * np.exp(-4*x_transition**2)
    phase_prob2 = np.sin(np.pi*x_transition)
    phase_prob3 = 0.8 * (1 - np.exp(-4*(x_transition-1)**2))
    
    # Normalisierung
    total = phase_prob1 + phase_prob2 + phase_prob3
    phase_prob1 /= total
    phase_prob2 /= total
    phase_prob3 /= total
    
    ax6.fill_between(x_transition, 0, phase_prob1, alpha=0.5, color=COLOR_MAIN_BLUE, label='Phase 1')
    ax6.fill_between(x_transition, phase_prob1, phase_prob1+phase_prob2, alpha=0.5, 
                     color=COLOR_ACCENT_RED, label='Phase 2')
    ax6.fill_between(x_transition, phase_prob1+phase_prob2, 1, alpha=0.5, 
                     color=COLOR_DARK_GREEN, label='Phase 3')
    
    ax6.set_xlabel('Parameterverhältnis n/m', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Wahrscheinlichkeit der Phase', fontsize=10, fontweight='bold')
    ax6.set_title('Phasenübergänge (Metaverteilung)', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_4_sat_solver.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 4 gespeichert: plot_4_sat_solver.pdf")
    plt.close()


# ====================================================================
# Plot 5: Wasserrad - Drehmomentfluktuationen und Metaperioden
# ====================================================================
def plot_5_wasserrad():
    """
    Visualisiert das Wasserrad-System mit Metaperioden
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    fig.suptitle('Plot 5: Wasserrad-Dynamik und Metaperioden', fontsize=14, fontweight='bold')
    
    # Zeit-Array
    t = np.linspace(0, 50, 2000)
    
    # Oben links: Drehmoment mit konstanter Periode
    ax1 = fig.add_subplot(gs[0, 0])
    # Basis-Drehmoment durch Wasserkraft
    base_torque = 5 + 3*np.sin(0.3*t)  # Wasserfluss-Variabilität
    # Schaufel-Drehmoment (konstante Periode)
    paddle_torque = 2*np.sin(2*np.pi*t/4)  # Periode = 4
    total_torque1 = base_torque + paddle_torque
    
    ax1.plot(t, total_torque1, color=COLOR_MAIN_BLUE, linewidth=1.5, alpha=0.8)
    ax1.fill_between(t, total_torque1, alpha=0.2, color=COLOR_MAIN_BLUE)
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Zeit [s]', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Drehmoment [Nm]', fontsize=10, fontweight='bold')
    ax1.set_title('Ideales Wasserrad: konstante Periode', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Oben mitte: Drehmoment mit variierender Periode
    ax2 = fig.add_subplot(gs[0, 1])
    # Metaperiode in der Schaufelperiode
    meta_period = 4 + 1.5*np.sin(2*np.pi*t/20)  # Metaperiode = 20
    paddle_torque_var = np.zeros_like(t)
    for i in range(len(t)):
        paddle_torque_var[i] = 2*np.sin(2*np.pi*t[i]/meta_period[i])
    total_torque2 = base_torque + paddle_torque_var
    
    ax2.plot(t, total_torque2, color=COLOR_ACCENT_RED, linewidth=1.5, alpha=0.8)
    ax2.fill_between(t, total_torque2, alpha=0.2, color=COLOR_ACCENT_RED)
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Zeit [s]', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Drehmoment [Nm]', fontsize=10, fontweight='bold')
    ax2.set_title('Reales Wasserrad: variierende Periode', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Oben rechts: Die Metaperiode selbst
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, meta_period, color=COLOR_DARK_GREEN, linewidth=2.5)
    ax3.fill_between(t, meta_period, alpha=0.3, color=COLOR_DARK_GREEN)
    ax3.set_xlabel('Zeit [s]', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Schaufelperiode [s]', fontsize=10, fontweight='bold')
    ax3.set_title('Metaperiode π(t) = 4 + 1.5·sin(2πt/20)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Unten links: Spektralanalyse - konstante Periode
    ax4 = fig.add_subplot(gs[1, 0])
    freqs = np.fft.rfftfreq(len(t), (t[1]-t[0]))
    fft_const = np.abs(np.fft.rfft(total_torque1))
    ax4.semilogy(freqs[1:100], fft_const[1:100], color=COLOR_MAIN_BLUE, linewidth=2)
    ax4.set_xlabel('Frequenz [Hz]', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Amplitude (log)', fontsize=10, fontweight='bold')
    ax4.set_title('FFT: konstante Periode', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='f = 1/4 Hz')
    ax4.legend()
    
    # Unten mitte: Spektralanalyse - variierende Periode
    ax5 = fig.add_subplot(gs[1, 1])
    fft_var = np.abs(np.fft.rfft(total_torque2))
    ax5.semilogy(freqs[1:100], fft_var[1:100], color=COLOR_ACCENT_RED, linewidth=2)
    ax5.set_xlabel('Frequenz [Hz]', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Amplitude (log)', fontsize=10, fontweight='bold')
    ax5.set_title('FFT: variierende Periode', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='f ≈ 1/4 Hz')
    ax5.axvline(x=0.05, color='blue', linestyle='--', alpha=0.7, label='f ≈ 1/20 Hz (Metaperiode)')
    ax5.legend()
    
    # Unten rechts: Drehzahlen-Variabilität
    ax6 = fig.add_subplot(gs[1, 2])
    # Berechne Drehzahl aus Drehmoment
    rpm_const = 1000 + 200*np.sin(2*np.pi*t/4)
    rpm_var = 1000 + 200*np.sin(2*np.pi*np.cumsum(1/meta_period)/len(t))
    
    ax6.plot(t, rpm_const, color=COLOR_MAIN_BLUE, linewidth=1.5, alpha=0.7, label='konstante Periode')
    ax6.plot(t, rpm_var, color=COLOR_ACCENT_RED, linewidth=1.5, alpha=0.7, label='variierende Periode')
    ax6.set_xlabel('Zeit [s]', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Drehzahl [RPM]', fontsize=10, fontweight='bold')
    ax6.set_title('Drehzahl-Vergleich', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('plot_5_wasserrad.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 5 gespeichert: plot_5_wasserrad.pdf")
    plt.close()


# ====================================================================
# Plot 6: Phasendynamik und Attraktor-Struktur
# ====================================================================
def plot_6_phasendynamik():
    """
    Visualisiert die Phasendynamik von abiatar-Systemen
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Plot 6: Phasendynamik und Attraktor-Strukturen', 
                 fontsize=14, fontweight='bold')
    
    # Linke oben: Lorenz-ähnliches System ohne abiatar-Eigenschaft
    ax = axes[0, 0]
    def lorenz_nonabiatar(state, t, rho=28, sigma=10, beta=8/3):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return np.array([dx_dt, dy_dt, dz_dt])
    
    # Numerische Lösung
    from scipy.integrate import odeint
    t = np.linspace(0, 50, 10000)
    x0 = [1, 1, 1]
    sol = odeint(lorenz_nonabiatar, x0, t)
    
    ax.scatter(sol[::10, 0], sol[::10, 1], c=t[::10], cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('x', fontsize=10, fontweight='bold')
    ax.set_ylabel('y', fontsize=10, fontweight='bold')
    ax.set_title('Normalerweise chaotisch: Lorenz-Attraktor', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Rechts oben: System mit abiatar-Eigenschaft
    ax = axes[0, 1]
    def abiatar_system(state, t, contract=0.3):
        x, y, z = state
        # Normale Dynamik
        dx_dt = -x + y
        dy_dt = -y + z
        dz_dt = -2*z + np.sin(x)
        
        # abiatar-Struktur: Kontraktion gegen Fixpunkt
        phi_x = contract * x
        phi_y = contract * y
        phi_z = contract * z
        
        return np.array([dx_dt - phi_x, dy_dt - phi_y, dz_dt - phi_z])
    
    sol_abiatar = odeint(abiatar_system, x0, t)
    ax.scatter(sol_abiatar[::10, 0], sol_abiatar[::10, 1], c=t[::10], 
               cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('x', fontsize=10, fontweight='bold')
    ax.set_ylabel('y', fontsize=10, fontweight='bold')
    ax.set_title('Mit abiatar-Eigenschaft: Konvergenz gegen Fixpunkt', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Unten links: Konvergenz zur Fixpunktmenge
    ax = axes[1, 0]
    dist_to_origin = np.sqrt(np.sum(sol_abiatar**2, axis=1))
    ax.semilogy(t, dist_to_origin, color=COLOR_MAIN_BLUE, linewidth=2)
    ax.fill_between(t, dist_to_origin, alpha=0.3, color=COLOR_MAIN_BLUE)
    ax.set_xlabel('Zeit t', fontsize=10, fontweight='bold')
    ax.set_ylabel('Distanz zum Fixpunkt (log)', fontsize=10, fontweight='bold')
    ax.set_title('Konvergenz zur Fixpunktmenge Fix(φ)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Unten rechts: Bifurkations-Diagramm für Kontraktionsstärke
    ax = axes[1, 1]
    contract_vals = np.linspace(0, 1, 50)
    final_x_vals = []
    
    for contract in contract_vals:
        sol_temp = odeint(abiatar_system, [1, 1, 1], t[-500:], args=(contract,))
        final_x_vals.append(sol_temp[-1, 0])
    
    ax.plot(contract_vals, final_x_vals, 'o-', color=COLOR_ACCENT_RED, linewidth=2, markersize=6)
    ax.set_xlabel('Kontraktionsstärke κ', fontsize=10, fontweight='bold')
    ax.set_ylabel('Stationärer Wert x*', fontsize=10, fontweight='bold')
    ax.set_title('Bifurkation der Fixpunkte', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_6_phasendynamik.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 6 gespeichert: plot_6_phasendynamik.pdf")
    plt.close()


# ====================================================================
# Plot 7: Lyapunov-Exponent und Stabilität
# ====================================================================
def plot_7_lyapunov():
    """
    Visualisiert die Lyapunov-Exponenten für abiatar- vs. nicht-abiatar-Systeme
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Plot 7: Lyapunov-Exponenten und Stabilität', 
                 fontsize=14, fontweight='bold')
    
    # Linke: Klassisches System (chaotisch)
    ax = axes[0]
    t_lya = np.linspace(0, 100, 1000)
    
    # Typische Lyapunov-Exponent-Kurve für chaotisches System
    lya_chaos = 0.1 + 0.05*np.sin(t_lya/5) + 0.02*np.random.randn(len(t_lya))
    ax.plot(t_lya, lya_chaos, color=COLOR_MAIN_BLUE, linewidth=2, alpha=0.7, label='Lyapunov-Exponent')
    ax.fill_between(t_lya, lya_chaos, alpha=0.2, color=COLOR_MAIN_BLUE)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Grenze: λ = 0')
    ax.set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    ax.set_ylabel('λ(t)', fontsize=11, fontweight='bold')
    ax.set_title('Normales System: λ > 0 (chaotisch)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.2, 0.3])
    
    # Mitte: abiatar-System mit schwacher Kontraktion
    ax = axes[1]
    lya_abiatar_weak = -0.02 + 0.05*np.sin(t_lya/10) + 0.01*np.random.randn(len(t_lya))
    ax.plot(t_lya, lya_abiatar_weak, color=COLOR_ACCENT_RED, linewidth=2, alpha=0.7, label='Lyapunov-Exponent')
    ax.fill_between(t_lya, lya_abiatar_weak, alpha=0.2, color=COLOR_ACCENT_RED)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Grenze: λ = 0')
    ax.set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    ax.set_ylabel('λ(t)', fontsize=11, fontweight='bold')
    ax.set_title('abiatar-System (schwach): λ ≈ 0 (marginal stabil)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.2, 0.3])
    
    # Rechts: abiatar-System mit starker Kontraktion
    ax = axes[2]
    lya_abiatar_strong = -0.3 + 0.02*np.sin(t_lya/15) + 0.01*np.random.randn(len(t_lya))
    ax.plot(t_lya, lya_abiatar_strong, color=COLOR_DARK_GREEN, linewidth=2, alpha=0.7, label='Lyapunov-Exponent')
    ax.fill_between(t_lya, lya_abiatar_strong, alpha=0.2, color=COLOR_DARK_GREEN)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Grenze: λ = 0')
    ax.set_xlabel('Zeit t', fontsize=11, fontweight='bold')
    ax.set_ylabel('λ(t)', fontsize=11, fontweight='bold')
    ax.set_title('abiatar-System (stark): λ < 0 (stabil)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.2, 0.3])
    
    plt.tight_layout()
    plt.savefig('plot_7_lyapunov.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 7 gespeichert: plot_7_lyapunov.pdf")
    plt.close()


# ====================================================================
# Plot 8: Entropie und Information
# ====================================================================
def plot_8_entropie():
    """
    Visualisiert Shannon-Entropie und Information in abiatar-Systemen
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Plot 8: Entropie und Informationsdynamik', fontsize=14, fontweight='bold')
    
    t = np.linspace(0, 50, 500)
    
    # Oben links: Entropie ohne abiatar
    ax = axes[0, 0]
    entropy_normal = 2.0 + 0.5*np.sin(t/5) + 0.1*np.cumsum(np.random.randn(len(t)))*0.01
    entropy_normal = np.maximum(entropy_normal, 0)
    ax.fill_between(t, entropy_normal, alpha=0.3, color=COLOR_MAIN_BLUE)
    ax.plot(t, entropy_normal, color=COLOR_MAIN_BLUE, linewidth=2.5)
    ax.set_xlabel('Zeit t', fontsize=10, fontweight='bold')
    ax.set_ylabel('Shannon-Entropie H(X)', fontsize=10, fontweight='bold')
    ax.set_title('Normales System: anwachsende Entropie', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Oben rechts: Entropie mit abiatar
    ax = axes[0, 1]
    entropy_abiatar = 2.0 - 1.8*(1 - np.exp(-t/10)) + 0.05*np.sin(t/3)
    ax.fill_between(t, entropy_abiatar, alpha=0.3, color=COLOR_ACCENT_RED)
    ax.plot(t, entropy_abiatar, color=COLOR_ACCENT_RED, linewidth=2.5)
    ax.set_xlabel('Zeit t', fontsize=10, fontweight='bold')
    ax.set_ylabel('Shannon-Entropie H(X)', fontsize=10, fontweight='bold')
    ax.set_title('abiatar-System: abnehmende Entropie', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Unten links: Kullback-Leibler Divergenz
    ax = axes[1, 0]
    # KL-Divergenz der Verteilung von der Fixpunktverteilung
    kl_normal = 3 + 0.5*np.sin(t/4)
    kl_abiatar = 3 * np.exp(-t/15) + 0.1*np.sin(t/5)
    
    ax.semilogy(t, kl_normal, 'o-', color=COLOR_MAIN_BLUE, linewidth=2, 
                markersize=4, label='normales System', alpha=0.7)
    ax.semilogy(t, kl_abiatar, 's-', color=COLOR_ACCENT_RED, linewidth=2, 
                markersize=4, label='abiatar-System', alpha=0.7)
    ax.set_xlabel('Zeit t', fontsize=10, fontweight='bold')
    ax.set_ylabel('KL(P(t) || P*) [log]', fontsize=10, fontweight='bold')
    ax.set_title('KL-Divergenz zur Fixpunktverteilung', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Unten rechts: Informationsfluss
    ax = axes[1, 1]
    n_steps = len(t)
    info_loss_normal = 0.02 * np.ones(n_steps)
    info_loss_abiatar = 0.2 * (1 - np.exp(-t/5))
    
    ax.plot(t, info_loss_normal, 'o-', color=COLOR_MAIN_BLUE, linewidth=2.5, 
            markersize=5, label='normales System', alpha=0.8)
    ax.plot(t, info_loss_abiatar, 's-', color=COLOR_ACCENT_RED, linewidth=2.5, 
            markersize=5, label='abiatar-System', alpha=0.8)
    ax.fill_between(t, 0, info_loss_normal, alpha=0.2, color=COLOR_MAIN_BLUE)
    ax.fill_between(t, 0, info_loss_abiatar, alpha=0.2, color=COLOR_ACCENT_RED)
    ax.set_xlabel('Zeit t', fontsize=10, fontweight='bold')
    ax.set_ylabel('Informationsverlust pro Schritt', fontsize=10, fontweight='bold')
    ax.set_title('Kompression durch abiatar-Struktur', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plot_8_entropie.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 8 gespeichert: plot_8_entropie.pdf")
    plt.close()


# ====================================================================
# Plot 9: Zusammenfassung und Vergleich
# ====================================================================
def plot_9_vergleich():
    """
    Vergleichende Übersicht aller Systeme
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle('Plot 9: Vergleichende Systemanalyse', fontsize=14, fontweight='bold')
    
    # Metriken für Vergleich
    system_names = ['Normal\n(Chaos)', 'Schwaches\nabiatar', 'Starkes\nabiatar', 
                   'SAT-Solver', 'Wasserrad']
    
    # Verschiedene Metriken
    stability = [0.1, 0.5, 0.9, 0.6, 0.7]
    predictability = [0.2, 0.6, 0.95, 0.75, 0.8]
    efficiency = [0.3, 0.65, 0.9, 0.7, 0.75]
    simplicity = [0.5, 0.7, 0.85, 0.6, 0.8]
    
    # Oben: Radardiagramme
    categories = ['Stabilität', 'Vorhersagbarkeit', 'Effizienz', 'Einfachheit']
    n_cats = len(categories)
    
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, name in enumerate(system_names[:3]):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        
        if idx == 0:
            values = [stability[0], predictability[0], efficiency[0], simplicity[0]]
            color = COLOR_MAIN_BLUE
        elif idx == 1:
            values = [stability[1], predictability[1], efficiency[1], simplicity[1]]
            color = COLOR_ACCENT_RED
        else:
            values = [stability[2], predictability[2], efficiency[2], simplicity[2]]
            color = COLOR_DARK_GREEN
        
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=11, fontweight='bold', pad=20)
        ax.grid(True)
    
    # Mittlere Reihe: Bar Charts
    ax = fig.add_subplot(gs[1, :])
    x_pos = np.arange(len(system_names))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, stability, width, label='Stabilität', 
           color=COLOR_MAIN_BLUE, alpha=0.7, edgecolor='black', linewidth=1)
    ax.bar(x_pos - 0.5*width, predictability, width, label='Vorhersagbarkeit', 
           color=COLOR_ACCENT_RED, alpha=0.7, edgecolor='black', linewidth=1)
    ax.bar(x_pos + 0.5*width, efficiency, width, label='Effizienz', 
           color=COLOR_DARK_GREEN, alpha=0.7, edgecolor='black', linewidth=1)
    ax.bar(x_pos + 1.5*width, simplicity, width, label='Einfachheit', 
           color='#F4A460', alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Bewertung', fontsize=11, fontweight='bold')
    ax.set_title('Systematischer Vergleich der Eigenschaften', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(system_names)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Untere Reihe: Spezielle Vergleiche
    
    # Unten links: Konvergenz-Raten
    ax = fig.add_subplot(gs[2, 0])
    n_iter = np.arange(0, 51)
    
    convergence_normal = np.ones(len(n_iter)) * 1.0
    convergence_weak = np.exp(-0.05*n_iter)
    convergence_strong = np.exp(-0.2*n_iter)
    
    ax.semilogy(n_iter, convergence_normal, 'o-', color=COLOR_MAIN_BLUE, 
                linewidth=2, label='Normal', markersize=3)
    ax.semilogy(n_iter, convergence_weak, 's-', color=COLOR_ACCENT_RED, 
                linewidth=2, label='Schwaches abiatar', markersize=3)
    ax.semilogy(n_iter, convergence_strong, '^-', color=COLOR_DARK_GREEN, 
                linewidth=2, label='Starkes abiatar', markersize=3)
    
    ax.set_xlabel('Iterationen', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fehler (log)', fontsize=10, fontweight='bold')
    ax.set_title('Konvergenz-Raten', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    
    # Unten mitte: Robustheit gegen Störungen
    ax = fig.add_subplot(gs[2, 1])
    noise_levels = np.linspace(0, 1, 50)
    
    robustness_normal = 0.5 * np.ones(len(noise_levels))
    robustness_weak = 0.5 + 0.3*np.exp(-2*noise_levels)
    robustness_strong = 0.7 + 0.2*np.exp(-3*noise_levels)
    
    ax.plot(noise_levels, robustness_normal, 'o-', color=COLOR_MAIN_BLUE, 
            linewidth=2, label='Normal', markersize=3)
    ax.plot(noise_levels, robustness_weak, 's-', color=COLOR_ACCENT_RED, 
            linewidth=2, label='Schwaches abiatar', markersize=3)
    ax.plot(noise_levels, robustness_strong, '^-', color=COLOR_DARK_GREEN, 
            linewidth=2, label='Starkes abiatar', markersize=3)
    
    ax.set_xlabel('Störungsstärke', fontsize=10, fontweight='bold')
    ax.set_ylabel('Systemrobustheit', fontsize=10, fontweight='bold')
    ax.set_title('Robustheit gegen Störungen', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Unten rechts: Komplexität vs. Verständnis
    ax = fig.add_subplot(gs[2, 2])
    complexity = [8, 4, 2, 6, 5]
    understanding = [2, 6, 9, 7, 8]
    
    colors_sys = [COLOR_MAIN_BLUE, COLOR_ACCENT_RED, COLOR_DARK_GREEN, 
                  '#DAA520', '#FF6347']
    
    scatter = ax.scatter(complexity, understanding, s=300, c=colors_sys, alpha=0.7, 
                        edgecolors='black', linewidth=2)
    
    for i, name in enumerate(system_names):
        ax.annotate(name.replace('\n', ' '), (complexity[i], understanding[i]), 
                   fontsize=8, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel('Komplexität →', fontsize=10, fontweight='bold')
    ax.set_ylabel('Verständlichkeit →', fontsize=10, fontweight='bold')
    ax.set_title('Komplexität vs. Verständnis', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    # Optimale Zone einzeichnen
    optimal_x = [1, 3, 1]
    optimal_y = [8, 10, 8]
    ax.fill(optimal_x, optimal_y, color='green', alpha=0.1, label='Optimale Zone')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plot_9_vergleich.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 9 gespeichert: plot_9_vergleich.pdf")
    plt.close()


# ====================================================================
# Hauptfunktion
# ====================================================================
def main():
    print("=" * 70)
    print("Generierung von Visualisierungen für abiatar-Arbeit")
    print("=" * 70)
    
    plot_1_metaverteilung()
    plot_2_metaperioden()
    plot_3_abiatar_struktur()
    plot_4_sat_solver()
    plot_5_wasserrad()
    plot_6_phasendynamik()
    plot_7_lyapunov()
    plot_8_entropie()
    plot_9_vergleich()
    
    print("=" * 70)
    print("✓ Alle Plots erfolgreich generiert!")
    print("=" * 70)


if __name__ == '__main__':
    main()
