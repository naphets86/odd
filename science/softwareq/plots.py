#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import factorial, gammainc
from scipy.integrate import odeint
import warnings

warnings.filterwarnings('ignore')

# Matplotlib-Konfiguration für deutsche Beschriftung
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 300

# ============================================================================
# Plot 1: Skalierungsverhalten - Verfügbarkeit vs. Anzahl Server
# ============================================================================

def plot_scaling():
    """
    Abbildung: Erforderliche Anzahl von Servern c* als Funktion der 
    Verfügbarkeitsanforderung A_req. Basiert auf Satz 1.
    """
    A_req_values = np.linspace(0.90, 0.99999, 100)
    c_values = []
    
    for A_req in A_req_values:
        # Vereinfachte Formel aus Satz 1
        # c* ≈ λ/μ / (1 - (1-A_req)/(1-α))
        # Mit λ/μ = 0.5 (50% Auslastung) und α = 0.001
        lambda_over_mu = 0.5
        alpha = 0.001
        
        if A_req > 0.99:
            c = lambda_over_mu / (1 - (1 - A_req) / (1 - alpha))
            c_values.append(c)
        else:
            c_values.append(lambda_over_mu / 0.9)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.semilogy(A_req_values, c_values, 'b-', linewidth=2.5, label='$c^*(A_{req})$')
    ax.fill_between(A_req_values, c_values, alpha=0.2, color='blue')
    
    ax.set_xlabel('Verfügbarkeitsanforderung $A_{req}$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erforderliche Anzahl Server $c^*$ (log-Skala)', fontsize=12, fontweight='bold')
    ax.set_title('Skalierungsverhalten: Exponentieller Anstieg mit höheren QoS-Anforderungen', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Annotationen für typische Werte
    ax.axvline(x=0.99, color='red', linestyle='--', alpha=0.5, label='99% (zwei Neunen)')
    ax.axvline(x=0.9999, color='orange', linestyle='--', alpha=0.5, label='99.99% (vier Neunen)')
    ax.axvline(x=0.99999, color='green', linestyle='--', alpha=0.5, label='99.999% (fünf Neunen)')
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_01_scaling.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 1 gespeichert: plot_01_scaling.pdf")
    plt.close()

# ============================================================================
# Plot 2: Markov-Zustandsübergänge
# ============================================================================

def plot_markov_states():
    """
    Markov-Kettenmodell mit 3 Zuständen:
    - z1: funktionsfähig
    - z2: degradiertes Verhalten
    - z3: ausgefallen
    """
    # Übergangsraten
    lambda_1_to_2 = 0.1  # Übergang zu degradiert
    lambda_2_to_3 = 0.05  # Übergang zu ausgefallen
    lambda_2_to_1 = 0.02  # Wiederherstellung
    
    # System von Differentialgleichungen
    def markov_system(y, t):
        pi1, pi2, pi3 = y
        dpi1_dt = -lambda_1_to_2 * pi1 + lambda_2_to_1 * pi2
        dpi2_dt = lambda_1_to_2 * pi1 - (lambda_2_to_3 + lambda_2_to_1) * pi2
        dpi3_dt = lambda_2_to_3 * pi2
        return [dpi1_dt, dpi2_dt, dpi3_dt]
    
    # Anfangsbedingung: System funktioniert (z1=1, z2=0, z3=0)
    y0 = [1.0, 0.0, 0.0]
    t = np.linspace(0, 100, 1000)
    
    # Lösen
    solution = odeint(markov_system, y0, t)
    pi1 = solution[:, 0]
    pi2 = solution[:, 1]
    pi3 = solution[:, 2]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t, pi1, 'g-', linewidth=2.5, label='$\pi_1(t)$ - Funktionsfähig')
    ax.plot(t, pi2, 'orange', linewidth=2.5, label='$\pi_2(t)$ - Degradiert')
    ax.plot(t, pi3, 'r-', linewidth=2.5, label='$\pi_3(t)$ - Ausgefallen')
    
    ax.fill_between(t, pi1, alpha=0.1, color='green')
    ax.fill_between(t, pi1 + pi2, pi1, alpha=0.1, color='orange')
    ax.fill_between(t, 1, pi1 + pi2, alpha=0.1, color='red')
    
    ax.set_xlabel('Zeit $t$ (normalisiert)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zustandswahrscheinlichkeit $\pi_i(t)$', fontsize=12, fontweight='bold')
    ax.set_title('Markov-Kettenmodell: Zustandsübergänge über Zeit', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_02_markov_states.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 2 gespeichert: plot_02_markov_states.pdf")
    plt.close()

# ============================================================================
# Plot 3: Warteschlangen-Charakteristiken
# ============================================================================

def plot_queue_characteristics():
    """
    M/M/c Warteschlange: Abhängigkeit von Wartezeit und Systemauslastung
    """
    # Parameter
    c_values = [1, 2, 4, 8]  # verschiedene Anzahl Server
    rho_range = np.linspace(0.1, 0.9, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Durchschnittliche Wartezeit
    for c in c_values:
        W_q = []  # Wartezeit im Queue (ohne Service)
        
        for rho in rho_range:
            # Vereinfachte Erlang-C Approximation
            pw = ((c * rho) ** c) / (factorial(c) * (1 - rho))
            pw_norm = pw / (1 + pw * (1 - rho) / (c * (1 - rho)))
            
            # W_q ≈ P_w / (c*μ*(1-ρ))
            # Normalisiert: W_q ≈ P_w / (c*(1-ρ))
            if rho < 0.95:
                w_q = pw_norm / (c * (1 - rho) + 1e-6)
            else:
                w_q = 100  # numerisch instabil für hohe Auslastung
            W_q.append(w_q)
        
        ax1.plot(rho_range, W_q, linewidth=2.5, label=f'$c={c}$ Server')
    
    ax1.set_xlabel('Systemauslastung $\\rho = \\lambda/(c \\mu)$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Durchschnittliche Wartezeit $W_q$ (normalisiert)', fontsize=11, fontweight='bold')
    ax1.set_title('Wartezeit vs. Auslastung', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 10])
    
    # Subplot 2: Blockierwahrscheinlichkeit (Erlang-C)
    for c in c_values:
        p_w = []
        
        for rho in rho_range:
            if rho < 0.99:
                # Erlang-C Formel
                pw_num = (c * rho) ** c / factorial(c)
                pw_denom = pw_num / (1 - rho)
                pw = pw_denom / (1 + pw_denom)
            else:
                pw = 0.99
            p_w.append(pw)
        
        ax2.plot(rho_range, p_w, linewidth=2.5, label=f'$c={c}$ Server')
    
    ax2.set_xlabel('Systemauslastung $\\rho = \\lambda/(c \\mu)$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Blockierwahrscheinlichkeit $P_w$ (Erlang-C)', fontsize=11, fontweight='bold')
    ax2.set_title('Erlang-C: Blockierwahrscheinlichkeit', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_03_queue_characteristics.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 3 gespeichert: plot_03_queue_characteristics.pdf")
    plt.close()

# ============================================================================
# Plot 4: Amdahl's Gesetz - Optimale Parallelisierung
# ============================================================================

def plot_amdahl_parallelism():
    """
    Amdahl's Gesetz: T(π) = (1-p)*T_seq + p*T_seq/π
    Zeigt optimale Parallelisierungsfaktor basierend auf parallelisierbarem Anteil p
    """
    p_values = [0.5, 0.75, 0.9, 0.95, 0.99]  # Anteil parallelisierbarer Code
    P_range = np.arange(1, 65)  # Anzahl Prozessoren
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for p in p_values:
        speedup = 1 / ((1 - p) + p / P_range)
        ax.plot(P_range, speedup, linewidth=2.5, marker='o', markersize=4, 
                label=f'$p={p}$ (parallelisierbar)')
    
    # Theoretisches Maximum
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Anzahl Prozessoren $P$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup $S(P) = 1 / ((1-p) + p/P)$', fontsize=12, fontweight='bold')
    ax.set_title("Amdahl's Gesetz: Speedup durch Parallelisierung", 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([1, 64])
    ax.set_ylim([0.5, 20])
    
    # Annotationen
    ax.text(30, 1.8, 'p=50% ist typisch\nfür reale Systeme', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_04_amdahl_parallelism.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 4 gespeichert: plot_04_amdahl_parallelism.pdf")
    plt.close()

# ============================================================================
# Plot 5: Cache-Hit-Rate unter Zipf-Verteilung
# ============================================================================

def plot_cache_hit_rate():
    """
    Hit-Rate als Funktion der Cache-Größe unter Zipf-verteilten Zugriffen
    H(C) ≈ 1 - 1 / (β * ln(W))
    """
    W_values = [100, 1000, 10000]  # Working Set Größen
    beta_values = [0.5, 0.8, 0.95]  # Lokalitäts-Koeffiziente
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Hit-Rate vs. Cache-Größe für verschiedene Working Sets
    C_range = np.logspace(0, 4, 100)  # Cache-Größe von 1 bis 10000
    
    for W in W_values:
        beta = 0.8  # konstanter Lokalitätskoeffizient
        hit_rates = []
        
        for C in C_range:
            # Zipf-basierte Hit-Rate Approximation
            if C >= W:
                H = 1.0
            else:
                # H(C) ≈ 1 - W / (e*C + W)
                H = 1 - W / (np.e * C + W)
            hit_rates.append(H)
        
        ax1.semilogx(C_range, hit_rates, linewidth=2.5, marker='o', markersize=3,
                     label=f'Working Set W={W}')
    
    ax1.set_xlabel('Cache-Größe $C$ (log-Skala)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit-Rate $H(C)$', fontsize=11, fontweight='bold')
    ax1.set_title('Cache-Hit-Rate vs. Cache-Größe (β=0.8)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    ax1.set_ylim([0, 1.05])
    
    # Subplot 2: Hit-Rate vs. Lokalitätskoeffizient
    C_fixed = 100  # feste Cache-Größe
    W_fixed = 1000  # festes Working Set
    
    beta_range = np.linspace(0.1, 0.99, 100)
    hit_rates_beta = []
    
    for beta in beta_range:
        # Höherer beta = bessere Lokalität
        H = 1 - (1 - beta) ** (C_fixed / W_fixed)
        hit_rates_beta.append(min(1.0, H))
    
    ax2.plot(beta_range, hit_rates_beta, linewidth=3, color='darkblue')
    ax2.fill_between(beta_range, hit_rates_beta, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Lokalitäts-Koeffizient $\\beta$ (Zipf-Exponent)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Hit-Rate $H(\\beta)$', fontsize=11, fontweight='bold')
    ax2.set_title(f'Hit-Rate vs. Lokalität (C={C_fixed}, W={W_fixed})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_05_cache_hit_rate.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 5 gespeichert: plot_05_cache_hit_rate.pdf")
    plt.close()

# ============================================================================
# Plot 6: Redundanzdesign und Zuverlässigkeitsverbesserung
# ============================================================================

def plot_redundancy_design():
    """
    Auswirkung von Redundanzfaktor k auf die Systemverfügbarkeit
    A(k) ≈ 1 - (1-r)^k
    """
    r_values = [0.95, 0.99, 0.999]  # Komponenten-Zuverlässigkeiten
    k_range = np.arange(1, 21)  # Redundanzfaktor 1..20
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for r in r_values:
        availabilities = 1 - (1 - r) ** k_range
        ax.plot(k_range, availabilities, linewidth=2.5, marker='o', markersize=6,
                label=f'Komponenten-Zuverlässigkeit $r={r}$')
    
    # Anforderungslevel
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='99% Anforderung')
    ax.axhline(y=0.999, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='99.9% Anforderung')
    ax.axhline(y=0.9999, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='99.99% Anforderung')
    
    ax.set_xlabel('Redundanzfaktor $k$ (Anzahl redundanter Komponenten)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Systemverfügbarkeit $A(k) = 1-(1-r)^k$', fontsize=12, fontweight='bold')
    ax.set_title('Redundanzdesign: Verfügbarkeit vs. Redundanzfaktor', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.9, 1.005])
    ax.set_xlim([1, 20])
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_06_redundancy_design.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 6 gespeichert: plot_06_redundancy_design.pdf")
    plt.close()

# ============================================================================
# Plot 7: MTTF (Mean Time To Failure) Vergleich verschiedener Komponenten
# ============================================================================

def plot_mttf_comparison():
    """
    Vergleich von MTTF für verschiedene Komponententypen
    MTTF = 1/λ (bei exponentialverteilter Ausfallzeit)
    """
    components = [
        'HDD Disk',
        'SSD Disk',
        'CPU',
        'RAM Module',
        'Network Card',
        'Power Supply'
    ]
    
    # Typische Ausfallraten (pro Jahr)
    lambda_rates = np.array([0.02, 0.005, 0.001, 0.0005, 0.001, 0.01])
    
    # MTTF in Jahren
    mttf = 1.0 / lambda_rates
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    bars = ax.barh(components, mttf, color=colors, edgecolor='black', linewidth=1.5)
    
    # Werte auf Balken schreiben
    for i, (bar, mttf_val) in enumerate(zip(bars, mttf)):
        ax.text(mttf_val + 5, i, f'{mttf_val:.1f} Jahre', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mittlere Zeit bis Ausfall (MTTF) in Jahren', fontsize=12, fontweight='bold')
    ax.set_title('Vergleich: Mittlere Ausfallzeiten verschiedener Computerkomponenten', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xlim([0, max(mttf) * 1.2])
    
    plt.tight_layout()
    plt.savefig('/home/claude/plot_07_mttf_comparison.pdf', bbox_inches='tight', dpi=300)
    print("✓ Plot 7 gespeichert: plot_07_mttf_comparison.pdf")
    plt.close()

# ============================================================================
# Hauptfunktion: Alle Plots generieren
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Matplotlib-Visualisierungen für wissenschaftliche Arbeit")
    print("="*70 + "\n")
    
    print("Generiere Plots...")
    print()
    
    try:
        plot_scaling()
        plot_markov_states()
        plot_queue_characteristics()
        plot_amdahl_parallelism()
        plot_cache_hit_rate()
        plot_redundancy_design()
        plot_mttf_comparison()
        
        print()
        print("="*70)
        print("✓ ERFOLG: Alle 7 Plots wurden erfolgreich generiert!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ FEHLER: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
