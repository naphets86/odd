#!/usr/bin/env python3
"""
Generierung von matplotlib Plots für die Wahrscheinlichkeitsverteilung
im SAT-Problem über der Kombinationspyramide.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.special import comb
import os

# Ausgabeverzeichnis
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_runtime_evolution():
    """Plot 1: Erwartete Laufzeit über Versuchsnummer"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Simulierte Daten für verschiedene Problemgrößen
    sizes = [12, 14, 16, 18]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for size, color in zip(sizes, colors):
        n_max = 2**size
        k_values = np.logspace(0, np.log10(n_max), 150)
        
        # Theoretische Modellfunktion
        log_component = np.log(size)
        polynomial_component = size**3
        
        # Phasenübergang mit sigmoid-ähnlicher Funktion
        transition_start = np.sqrt(n_max)
        transition_end = n_max / 2
        
        runtime = []
        for k in k_values:
            if k < transition_start:
                # Phase 1: logarithmisch
                rt = log_component + 0.5 * np.log(k)
            else:
                # Übergang und Phase 3
                progress = (np.log(k) - np.log(transition_start)) / (np.log(transition_end) - np.log(transition_start))
                progress = np.clip(progress, 0, 1)
                
                # Superlinearer Übergang
                factor = progress**(1.5)
                rt = log_component * (1 - factor) + polynomial_component * factor
        
            runtime.append(rt)
        
        ax.loglog(k_values, runtime, linewidth=2.5, label=f'$n = {size}$', color=color)
    
    # Theoretische Untergrenzen
    ax.loglog([1, 2**20], [np.log(12), 20*np.log(12)], 'k--', linewidth=1, alpha=0.5, label='Phase 1: $O(\log n)$')
    ax.loglog([100, 2**20], [100, 100*(2**20/100)**1.5], 'k:', linewidth=1, alpha=0.5, label='Phase 2: Übergang')
    
    ax.set_xlabel('Versuchsnummer $k$ (log. Skala)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erwartete Laufzeit $E[T_k]$ (log. Skala)', fontsize=12, fontweight='bold')
    ax.set_title('Verschiebung der Wahrscheinlichkeitsverteilung: Erwartete Laufzeit\nüber Versuchsnummer für verschiedene Problemgrößen', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim([0.5, 2**20])
    ax.set_ylim([0.5, 2*10**6])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot1_runtime_evolution.pdf'), dpi=300, bbox_inches='tight')
    print("✓ Plot 1 erstellt: plot1_runtime_evolution.pdf")
    plt.close()


def generate_probability_density():
    """Plot 2: Wahrscheinlichkeitsdichte für verschiedene Phasen"""
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    phases = [
        ('Phase 1: Initialisierung\n($k < \\sqrt{2^n}$)', 10, 5),
        ('Phase 2a: Früher Übergang\n($\\sqrt{2^n} \\leq k < 2^n/4$)', 300, 200),
        ('Phase 2b: Später Übergang\n($2^n/4 \\leq k < 2^n/2$)', 3000, 1000),
        ('Phase 3: Asymptotik\n($k > 2^n/2$)', 5000, 100)
    ]
    
    n = 16
    max_runtime = 2**n  # Worst-Case
    
    for idx, (title, mode_fast, mode_slow) in enumerate(phases):
        ax = axes[idx]
        
        # Bimodale Verteilung mit zwei Gaussians
        x = np.linspace(0, max_runtime, 1000)
        
        # Phase-abhängiges Mischungsverhältnis
        if idx == 0:
            # Phase 1: Dominanz schnelle Lösungen
            alpha_fast = 0.85
            sigma_fast = 30
            sigma_slow = 500
        elif idx == 1:
            # Phase 2a: Übergang beginnt
            alpha_fast = 0.60
            sigma_fast = 100
            sigma_slow = 1500
        elif idx == 2:
            # Phase 2b: Übergang fortgeschritten
            alpha_fast = 0.30
            sigma_fast = 200
            sigma_slow = 2000
        else:
            # Phase 3: Dominanz Worst-Case
            alpha_fast = 0.05
            sigma_fast = 500
            sigma_slow = 2500
        
        # Bimodale Dichte
        density_fast = alpha_fast * stats.norm.pdf(x, mode_fast, sigma_fast)
        density_slow = (1 - alpha_fast) * stats.norm.pdf(x, mode_slow, sigma_slow)
        density_total = density_fast + density_slow
        
        # Normalisierung
        density_total = density_total / np.max(density_total)
        
        # Plotzen
        ax.fill_between(x, density_total, alpha=0.3, color='#1f77b4', label='Gesamt-Dichte')
        ax.plot(x, density_fast, 'g-', linewidth=2, label=f'Schnelle Mode ($t \\approx {mode_fast}$)')
        ax.plot(x, density_slow, 'r-', linewidth=2, label=f'Langsame Mode ($t \\approx {mode_slow}$)')
        ax.plot(x, density_total, 'b-', linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Laufzeit $t$ (Schritte)', fontsize=10)
        ax.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim([0, max_runtime])
        ax.set_ylim([0, 1.1])
    
    fig.suptitle('Bimodale Wahrscheinlichkeitsdichte über Phasen hinweg', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot2_probability_density.pdf'), dpi=300, bbox_inches='tight')
    print("✓ Plot 2 erstellt: plot2_probability_density.pdf")
    plt.close()


def generate_convergence_analysis():
    """Plot 3: Konvergenzgeschwindigkeit (logarithmische Skalierung)"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linkes Panel: Absolute Annäherung an Worst-Case
    sizes = [12, 14, 16]
    colors_conv = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for size, color in zip(sizes, colors_conv):
        n_max = 2**size
        k_values = np.linspace(n_max/2 + 1, n_max, 100)
        
        # Theoretische Konvergenz: E[T_k] = n^3 - O(n^3) * exp(-mu * (n_max - k))
        worst_case = size**3
        mu = 0.05
        
        distance = worst_case * np.exp(-mu * (n_max - k_values))
        
        ax1.semilogy(k_values, distance, linewidth=2.5, label=f'$n = {size}$', color=color, marker='o', markersize=4)
    
    ax1.set_xlabel('Versuchsnummer $k$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('$|E[T_k] - n^3|$ (log. Skala)', fontsize=11, fontweight='bold')
    ax1.set_title('Annäherung an Worst-Case-Komplexität\n(Phase 3 Detail)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=10)
    
    # Rechtes Panel: Relative Fehler
    for size, color in zip(sizes, colors_conv):
        n_max = 2**size
        k_values = np.linspace(n_max/2 + 1, n_max, 100)
        
        worst_case = size**3
        mu = 0.05
        
        distance = worst_case * np.exp(-mu * (n_max - k_values))
        relative_error = distance / worst_case
        
        ax2.semilogy(k_values / n_max, relative_error, linewidth=2.5, label=f'$n = {size}$', color=color, marker='s', markersize=4)
    
    ax2.set_xlabel('Relative Versuchsnummer $k / 2^n$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relativer Fehler $|E[T_k] - n^3| / n^3$ (log. Skala)', fontsize=11, fontweight='bold')
    ax2.set_title('Relativer Fehler zur Worst-Case-Grenze\n(Normalisiert)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot3_convergence_analysis.pdf'), dpi=300, bbox_inches='tight')
    print("✓ Plot 3 erstellt: plot3_convergence_analysis.pdf")
    plt.close()


def generate_phase_transitions():
    """Plot 4: Phase-Übergänge mit Markierungen"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    size = 16
    n_max = 2**size
    k_values = np.logspace(0, np.log10(n_max), 200)
    
    log_component = np.log(size)
    polynomial_component = size**3
    
    transition_start = np.sqrt(n_max)
    transition_end = n_max / 2
    
    runtime = []
    phases = []
    
    for k in k_values:
        if k < transition_start:
            # Phase 1
            rt = log_component + 0.5 * np.log(k)
            phase = 1
        elif k < transition_end:
            # Phase 2
            progress = (np.log(k) - np.log(transition_start)) / (np.log(transition_end) - np.log(transition_start))
            factor = progress**(1.5)
            rt = log_component * (1 - factor) + polynomial_component * factor
            phase = 2
        else:
            # Phase 3
            factor = 1 - np.exp(-0.05 * (k - transition_end))
            rt = log_component * (1 - factor) + polynomial_component * factor
            phase = 3
        
        runtime.append(rt)
        phases.append(phase)
    
    runtime = np.array(runtime)
    phases = np.array(phases)
    
    # Hauptkurve plotten
    ax.loglog(k_values, runtime, 'b-', linewidth=3.5, label='$E[T_k]$ (Modell)', zorder=5)
    
    # Phase 1
    phase1_mask = phases == 1
    ax.fill_between(k_values[phase1_mask], runtime[phase1_mask]*0.5, runtime[phase1_mask]*2,
                     alpha=0.15, color='green', label='Phase 1: Initialisierung ($k < \\sqrt{2^n}$)', zorder=1)
    
    # Phase 2
    phase2_mask = phases == 2
    ax.fill_between(k_values[phase2_mask], runtime[phase2_mask]*0.5, runtime[phase2_mask]*2,
                     alpha=0.15, color='orange', label='Phase 2: Übergang', zorder=1)
    
    # Phase 3
    phase3_mask = phases == 3
    ax.fill_between(k_values[phase3_mask], runtime[phase3_mask]*0.5, runtime[phase3_mask]*2,
                     alpha=0.15, color='red', label='Phase 3: Asymptotik ($k > 2^n/2$)', zorder=1)
    
    # Übergangspunkte markieren
    ax.axvline(transition_start, color='green', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    ax.axvline(transition_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    # Annotationen
    ax.text(transition_start*0.3, polynomial_component*10, 
            f'$\\sqrt{{2^{{{size}}}}} \\approx {int(transition_start)}$\n(Phase 1→2)', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.text(transition_end*1.5, polynomial_component*5, 
            f'$2^{{{size-1}}} \\approx {int(transition_end)}$\n(Phase 2→3)', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Asymptotische Linie
    ax.axhline(polynomial_component, color='red', linestyle=':', linewidth=2, alpha=0.5, 
               label=f'Worst-Case: $O(n^3) \\approx {polynomial_component:.0f}$')
    
    ax.set_xlabel('Versuchsnummer $k$ (log. Skala)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erwartete Laufzeit $E[T_k]$ (log. Skala)', fontsize=12, fontweight='bold')
    ax.set_title('Phase-Übergänge bei Lösung von SAT-Problemen\n(Problem-Größe $n = 16$, $2^n = 65536$ mögliche Formeln)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.set_xlim([0.5, n_max*1.2])
    ax.set_ylim([1, polynomial_component*20])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot4_phase_transitions.pdf'), dpi=300, bbox_inches='tight')
    print("✓ Plot 4 erstellt: plot4_phase_transitions.pdf")
    plt.close()


def main():
    """Hauptfunktion: Erstelle alle Plots"""
    print("\n" + "="*60)
    print("  Generierung von matplotlib Plots")
    print("  Wahrscheinlichkeitsverteilung im SAT-Problem")
    print("="*60 + "\n")
    
    print("Generiere Plots...")
    generate_runtime_evolution()
    generate_probability_density()
    generate_convergence_analysis()
    generate_phase_transitions()
    
    print("\n" + "="*60)
    print("  ✓ Alle Plots erfolgreich erstellt!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()