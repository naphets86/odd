#!/usr/bin/env python3
"""
Generierung der matplotlib-Plots für die wissenschaftliche Arbeit:
"Ebbe und Flut als Unsicherheitsstruktur: Formale Ökonomie der Tauschfreiheit"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import erf
from scipy.stats import norm
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Matplotlib-Einstellungen
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================
# PLOT 1: Tidenhub und Unsicherheitsstruktur
# ============================================================
def plot_1_tidal_uncertainty():
    """Plot des Gezeitenverlaufs und der daraus resultierenden Unsicherheit"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Zeitachse
    t = np.linspace(0, 24, 1000)  # 24 Stunden
    
    # Ebbe-Flut-Modell: W(t) = W_max * (1 + sin(2π*t/T))
    W_max = 2.5
    W_t = W_max * (1 + np.sin(2*np.pi*t/12.42))  # 12.42h Gezeitenperiode
    
    # Ableitung für Unsicherheitsgeschwindigkeit
    dW_dt = W_max * (2*np.pi/12.42) * np.cos(2*np.pi*t/12.42)
    
    # Subplot 1: Wasserstand
    ax1.plot(t, W_t, linewidth=2.5, color='#1f77b4', label='$W(t)$ - Wasserstand')
    ax1.axhline(y=W_max, color='r', linestyle='--', alpha=0.5, label='Maximum')
    ax1.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Minimum')
    
    # Markiere aktuelle Position
    t_current = 6
    W_current = W_max * (1 + np.sin(2*np.pi*t_current/12.42))
    ax1.scatter([t_current], [W_current], color='red', s=200, zorder=5, label='Beobachter')
    
    # Vertrauensintervall
    uncertainty = np.abs(dW_dt)
    ax1.fill_between(t, W_t - 0.3*uncertainty/uncertainty.max(), 
                     W_t + 0.3*uncertainty/uncertainty.max(), 
                     alpha=0.2, color='blue', label='Unsicherheitszone')
    
    ax1.set_xlabel('Zeit [Stunden]')
    ax1.set_ylabel('Wasserstand $W(t)$ [Meter]')
    ax1.set_title('(a) Gezeitendynamik: Der physikalische Prozess', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, 24])
    
    # Subplot 2: Verweildauer und Entscheidungsdruck
    # τ(t) = Zeit bis zum nächsten Extremum
    tau_to_max = np.zeros_like(t)
    for i, t_val in enumerate(t):
        phase = (2*np.pi*t_val/12.42) % (2*np.pi)
        if phase < np.pi:
            tau_to_max[i] = (np.pi - phase) * (12.42/(2*np.pi))
        else:
            tau_to_max[i] = (2*np.pi - phase) * (12.42/(2*np.pi))
    
    ax2.plot(t, tau_to_max, linewidth=2.5, color='#ff7f0e', label='$\\tau(t)$ - Verweildauer')
    
    # Entscheidungsschwelle
    decision_threshold = 3.0
    ax2.axhline(y=decision_threshold, color='r', linestyle=':', linewidth=2, 
                label='Kritischer Schwellenwert')
    
    # Markiere Entscheidungszone
    decision_zone = tau_to_max < decision_threshold
    ax2.fill_between(t, 0, tau_to_max, where=decision_zone, alpha=0.3, 
                     color='red', label='Entscheidungszone')
    
    ax2.set_xlabel('Zeit [Stunden]')
    ax2.set_ylabel('Verweildauer $\\tau(t)$ [Stunden]')
    ax2.set_title('(b) Entscheidungsdruck: Notwendigkeit des Handelns', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 24])
    ax2.set_ylim([0, 8])
    
    plt.tight_layout()
    plt.savefig('plot_1_tidal_uncertainty.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 1 erstellt: plot_1_tidal_uncertainty.pdf")
    plt.close()

# ============================================================
# PLOT 2: Uncertainty Index und Tausch-Freiheit
# ============================================================
def plot_2_uncertainty_index():
    """Plot des Unsicherheitsindex Ω und der resultierenden Tausch-Freiheit"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Unsicherheitsindex Ω(t)
    t_eco = np.linspace(0, 100, 500)
    
    # Ω(t) = 1 - |dW/dt|/max(|dW/dt|) = Grad der Unsicherheit
    omega_t = 0.5 + 0.4 * np.sin(2*np.pi*t_eco/24) + 0.1 * np.random.randn(len(t_eco))
    omega_t = np.clip(omega_t, 0.1, 0.9)
    
    # Subplot 1: Unsicherheitsindex
    ax1.plot(t_eco, omega_t, linewidth=2, color='#2ca02c')
    ax1.fill_between(t_eco, omega_t, alpha=0.3, color='#2ca02c')
    ax1.set_xlabel('Zeit $t$ [Stunden]')
    ax1.set_ylabel('$\\Omega(t)$ - Unsicherheitsindex')
    ax1.set_title('(a) Unsicherheitsindex $\\Omega(t)$', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Subplot 2: Tausch-Häufigkeit vs. Unsicherheit
    # Intuition: Tauscher muss häufiger agieren bei hoher Unsicherheit
    tausch_rate = 2 + 3*omega_t + 0.5*np.random.randn(len(t_eco))
    tausch_rate = np.clip(tausch_rate, 0.5, 6)
    
    ax2.scatter(omega_t, tausch_rate, alpha=0.6, s=30, color='#d62728')
    z = np.polyfit(omega_t, tausch_rate, 1)
    p = np.poly1d(z)
    omega_sorted = np.sort(omega_t)
    ax2.plot(omega_sorted, p(omega_sorted), "r--", linewidth=2.5, label='Regressionslinie')
    
    ax2.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax2.set_ylabel('Tausch-Häufigkeit $\\nu(t)$ [Tausche/h]')
    ax2.set_title('(b) Tausch-Häufigkeit vs. Unsicherheit', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Freiheitsgrad F(Ω)
    # F(Ω) = ∫_0^Ω Ω'(s) ds = Freizügigkeit der Handlung
    omega_range = np.linspace(0.1, 0.9, 100)
    # Freiheitsgrad wächst mit Unsicherheit (paradox!)
    freedom_measure = np.log(1/(1-omega_range)) - np.log(1/0.1)
    
    ax3.plot(omega_range, freedom_measure, linewidth=2.5, color='#9467bd')
    ax3.fill_between(omega_range, freedom_measure, alpha=0.3, color='#9467bd')
    ax3.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax3.set_ylabel('Freiheitsgrad $F(\\Omega)$ [Bits]')
    ax3.set_title('(c) Freiheit durch Unsicherheit', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_2_uncertainty_index.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 2 erstellt: plot_2_uncertainty_index.pdf")
    plt.close()

# ============================================================
# PLOT 3: Wahrscheinlichkeitsverteilungen - Extreme vs. Unsicherheit
# ============================================================
def plot_3_distributions():
    """Vergleich: Deterministische vs. unsichere Welten"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.linspace(-5, 5, 1000)
    
    # Subplot 1: Deterministisch (Delta-Dirac)
    ax1.axvline(x=0, color='black', linewidth=3, label='$\\delta(x)$')
    ax1.fill_between([-0.1, 0.1], [0, 0], [5, 5], alpha=0.5, color='red')
    ax1.set_ylabel('Wahrscheinlichkeitsdichte')
    ax1.set_xlabel('Wohlfahrtszustand $x$')
    ax1.set_title('(a) Deterministische Welt', fontweight='bold')
    ax1.set_ylim([0, 5])
    ax1.text(0, 4, 'Keine Freiheit', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Moderate Unsicherheit (Normalverteilung, σ=1)
    y2 = norm.pdf(x, 0, 1)
    ax2.plot(x, y2, linewidth=2.5, color='blue', label='$\\mathcal{N}(0,1)$')
    ax2.fill_between(x, y2, alpha=0.3, color='blue')
    ax2.set_ylabel('Wahrscheinlichkeitsdichte')
    ax2.set_xlabel('Wohlfahrtszustand $x$')
    ax2.set_title('(b) Moderate Unsicherheit', fontweight='bold')
    ax2.text(0, 0.35, 'Optimale Freiheit', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Extreme Unsicherheit (Uniform)
    y3 = np.ones_like(x) * 0.1
    y3[np.abs(x) > 5] = 0
    ax3.fill_between(x, y3, alpha=0.3, color='red')
    ax3.plot(x, y3, linewidth=2.5, color='red', label='$U(-5, 5)$')
    ax3.set_ylabel('Wahrscheinlichkeitsdichte')
    ax3.set_xlabel('Wohlfahrtszustand $x$')
    ax3.set_title('(c) Extreme Unsicherheit', fontweight='bold')
    ax3.text(0, 0.05, 'Chaos / Unmöglichkeit der Planung', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('plot_3_distributions.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 3 erstellt: plot_3_distributions.pdf")
    plt.close()

# ============================================================
# PLOT 4: Monte Carlo Simulation - Tausch unter Unsicherheit
# ============================================================
def plot_4_monte_carlo():
    """Simulation von Tauschtransaktionen bei verschiedenen Unsicherheitsgraden"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    np.random.seed(42)
    n_simulations = 5000
    n_agents = 100
    
    # Szenario 1: Hohe Unsicherheit (Ω = 0.8)
    omega_high = 0.8
    trades_high = np.random.exponential(1/omega_high, n_simulations)
    welfare_high = np.cumsum(np.random.randn(n_simulations) * np.sqrt(omega_high))
    
    # Szenario 2: Moderate Unsicherheit (Ω = 0.5)
    omega_moderate = 0.5
    trades_moderate = np.random.exponential(1/omega_moderate, n_simulations)
    welfare_moderate = np.cumsum(np.random.randn(n_simulations) * np.sqrt(omega_moderate))
    
    # Szenario 3: Geringe Unsicherheit (Ω = 0.2)
    omega_low = 0.2
    trades_low = np.random.exponential(1/omega_low, n_simulations)
    welfare_low = np.cumsum(np.random.randn(n_simulations) * np.sqrt(omega_low))
    
    # Plot 1: Histogramm der Tauschtransaktionen
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(trades_high, bins=50, alpha=0.5, label=f'$\\Omega={omega_high}$', color='red', density=True)
    ax1.hist(trades_moderate, bins=50, alpha=0.5, label=f'$\\Omega={omega_moderate}$', color='blue', density=True)
    ax1.hist(trades_low, bins=50, alpha=0.5, label=f'$\\Omega={omega_low}$', color='green', density=True)
    ax1.set_xlabel('Tausch-Häufigkeit [Ereignisse/h]')
    ax1.set_ylabel('Häufigkeit (normiert)')
    ax1.set_title('(a) Verteilung der Tausch-Häufigkeiten', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])
    
    # Plot 2: Cumulative Wohlfahrts-Trajektorie
    ax2 = fig.add_subplot(gs[0, 1])
    t_array = np.arange(n_simulations)
    ax2.plot(t_array, welfare_high, linewidth=1, alpha=0.7, label=f'$\\Omega={omega_high}$', color='red')
    ax2.plot(t_array, welfare_moderate, linewidth=1, alpha=0.7, label=f'$\\Omega={omega_moderate}$', color='blue')
    ax2.plot(t_array, welfare_low, linewidth=1, alpha=0.7, label=f'$\\Omega={omega_low}$', color='green')
    ax2.set_xlabel('Tausch-Ereignis (kumulativ)')
    ax2.set_ylabel('Kumulierte Wohlfahrt')
    ax2.set_title('(b) Wohlfahrts-Trajektorie über Zeit', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatilität vs. Unsicherheit
    ax3 = fig.add_subplot(gs[1, 0])
    omega_range = np.linspace(0.1, 0.9, 20)
    volatilities = []
    for omega in omega_range:
        sim = np.random.randn(500) * np.sqrt(omega)
        vol = np.std(sim)
        volatilities.append(vol)
    ax3.scatter(omega_range, volatilities, s=100, color='purple', alpha=0.7)
    ax3.plot(omega_range, np.sqrt(omega_range), 'r--', linewidth=2, label='$\\sqrt{\\Omega}$')
    ax3.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax3.set_ylabel('Volatilität $\\sigma$')
    ax3.set_title('(c) Beziehung: Unsicherheit ↔ Volatilität', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Durchschnittliche Tausch-Häufigkeit
    ax4 = fig.add_subplot(gs[1, 1])
    omega_range_fine = np.linspace(0.1, 0.9, 50)
    avg_trades = []
    for omega in omega_range_fine:
        trades_sim = np.random.exponential(1/omega, 1000)
        avg_trades.append(np.mean(trades_sim))
    ax4.plot(omega_range_fine, avg_trades, linewidth=2.5, color='darkblue')
    ax4.fill_between(omega_range_fine, avg_trades, alpha=0.3, color='darkblue')
    ax4.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax4.set_ylabel('Durchschnittliche Tausch-Häufigkeit')
    ax4.set_title('(d) Tausch-Aktivität vs. Unsicherheit', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_4_monte_carlo.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 4 erstellt: plot_4_monte_carlo.pdf")
    plt.close()

# ============================================================
# PLOT 5: Mathematische Funktion F(Ω) - Freiheitsgrad
# ============================================================
def plot_5_freedom_function():
    """Analytische Darstellung des Freiheitsgradsfunktion F(Ω)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    omega = np.linspace(0.01, 0.99, 500)
    
    # Plot 1: F₁(Ω) = -ln(1-Ω) (Informationsentropie)
    f1 = -np.log(1 - omega)
    axes[0, 0].plot(omega, f1, linewidth=2.5, color='#1f77b4')
    axes[0, 0].fill_between(omega, f1, alpha=0.3, color='#1f77b4')
    axes[0, 0].set_xlabel('$\\Omega$ (Unsicherheitsindex)')
    axes[0, 0].set_ylabel('$F_1(\\Omega) = -\\ln(1-\\Omega)$')
    axes[0, 0].set_title('(a) Informationsentropie: Freiheit durch Ungewissheit', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F₂(Ω) = Ω/(1-Ω) (Risiko-Chancen-Ratio)
    f2 = omega / (1 - omega)
    axes[0, 1].plot(omega, f2, linewidth=2.5, color='#ff7f0e')
    axes[0, 1].fill_between(omega, f2, alpha=0.3, color='#ff7f0e')
    axes[0, 1].set_xlabel('$\\Omega$ (Unsicherheitsindex)')
    axes[0, 1].set_ylabel('$F_2(\\Omega) = \\Omega/(1-\\Omega)$')
    axes[0, 1].set_title('(b) Chancen-Risiko-Ratio: Entscheidungsfreiheit', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 20])
    
    # Plot 3: F₃(Ω) = Ω·ln(Ω) + (1-Ω)·ln(1-Ω) (Shannon Entropy)
    f3 = omega * np.log(omega) + (1-omega) * np.log(1-omega)
    f3 = -f3  # Negative für positive Entropie
    axes[1, 0].plot(omega, f3, linewidth=2.5, color='#2ca02c')
    axes[1, 0].fill_between(omega, f3, alpha=0.3, color='#2ca02c')
    axes[1, 0].set_xlabel('$\\Omega$ (Unsicherheitsindex)')
    axes[1, 0].set_ylabel('$F_3(\\Omega) = -H(\\Omega)$ (Shannon-Entropie)')
    axes[1, 0].set_title('(c) Entropie: Maximale Freiheit bei Ω=0.5', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Vergleich aller drei Funktionen (normalisiert)
    f1_norm = f1 / f1.max()
    f2_norm = f2 / f2.max()
    f3_norm = f3 / f3.max()
    
    axes[1, 1].plot(omega, f1_norm, linewidth=2.5, label='$F_1$ (Log)', color='#1f77b4')
    axes[1, 1].plot(omega, f2_norm, linewidth=2.5, label='$F_2$ (Ratio)', color='#ff7f0e')
    axes[1, 1].plot(omega, f3_norm, linewidth=2.5, label='$F_3$ (Shannon)', color='#2ca02c')
    axes[1, 1].set_xlabel('$\\Omega$ (Unsicherheitsindex)')
    axes[1, 1].set_ylabel('Normalisierter Freiheitsgrad')
    axes[1, 1].set_title('(d) Vergleich der Freiheits-Funktionen', fontweight='bold')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_5_freedom_function.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 5 erstellt: plot_5_freedom_function.pdf")
    plt.close()

# ============================================================
# PLOT 6: Dynamische Systeme - Phasenraum
# ============================================================
def plot_6_phase_space():
    """Dynamik des Tauschsystems im Phasenraum"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Phasenraum: (Unsicherheit Ω, Tausch-Häufigkeit ν)
    omega_vals = np.linspace(0.1, 0.9, 20)
    nu_vals = np.linspace(0.5, 5, 20)
    Omega_grid, Nu_grid = np.meshgrid(omega_vals, nu_vals)
    
    # Vektorfeld: dΩ/dt und dν/dt
    # dΩ/dt = 0.5(1-Ω)Ω - 0.1ν  (logistische Dynamik mit Hemmung durch Tausch)
    # dν/dt = 0.2Ω - 0.05ν       (Tausch getrieben durch Unsicherheit, Selbstdämpfung)
    
    dOmega = 0.5*Omega_grid*(1-Omega_grid) - 0.1*Nu_grid
    dNu = 0.2*Omega_grid - 0.05*Nu_grid
    
    # Plot 1: Vektorfeld (Stromlinien)
    speed = np.sqrt(dOmega**2 + dNu**2)
    ax1.streamplot(Omega_grid, Nu_grid, dOmega, dNu, color=speed, cmap='viridis', density=2)
    ax1.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax1.set_ylabel('Tausch-Häufigkeit $\\nu$ [Tausche/h]')
    ax1.set_title('(a) Phasenraum: Strömungsdiagramm', fontweight='bold')
    
    # Markiere Gleichgewichtspunkte (Nullklinen)
    # Omega-Nullkline: 0.5(1-Ω)Ω = 0.1ν  →  ν = 5(1-Ω)Ω
    nu_zero_omega = 5*(1-omega_vals)*omega_vals
    ax1.plot(omega_vals, nu_zero_omega, 'r-', linewidth=2.5, label='Ω-Nullkline')
    
    # Nu-Nullkline: 0.2Ω = 0.05ν  →  ν = 4Ω
    nu_zero_nu = 4*omega_vals
    ax1.plot(omega_vals, nu_zero_nu, 'b-', linewidth=2.5, label='ν-Nullkline')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.1, 0.9])
    ax1.set_ylim([0.5, 5])
    
    # Plot 2: Trajektorien aus verschiedenen Anfangsbedingungen
    initial_conditions = [
        (0.2, 1.0), (0.5, 2.0), (0.8, 3.5), (0.3, 0.7), (0.7, 4.0)
    ]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_conditions)))
    
    for (omega_init, nu_init), color in zip(initial_conditions, colors):
        omega_traj = [omega_init]
        nu_traj = [nu_init]
        
        for _ in range(500):
            omega_curr = omega_traj[-1]
            nu_curr = nu_traj[-1]
            
            dOmega_curr = 0.5*omega_curr*(1-omega_curr) - 0.1*nu_curr
            dNu_curr = 0.2*omega_curr - 0.05*nu_curr
            
            omega_next = omega_curr + 0.01*dOmega_curr
            nu_next = nu_curr + 0.01*dNu_curr
            
            omega_next = np.clip(omega_next, 0.1, 0.9)
            nu_next = np.clip(nu_next, 0.5, 5)
            
            omega_traj.append(omega_next)
            nu_traj.append(nu_next)
        
        ax2.plot(omega_traj, nu_traj, alpha=0.7, linewidth=1.5, color=color)
        ax2.scatter([omega_init], [nu_init], s=100, marker='o', color=color, zorder=5)
        ax2.scatter([omega_traj[-1]], [nu_traj[-1]], s=100, marker='s', color=color, zorder=5)
    
    ax2.set_xlabel('Unsicherheitsindex $\\Omega$')
    ax2.set_ylabel('Tausch-Häufigkeit $\\nu$ [Tausche/h]')
    ax2.set_title('(b) Lösungstrajektorien für verschiedene Anfangsbedingungen', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.1, 0.9])
    ax2.set_ylim([0.5, 5])
    
    plt.tight_layout()
    plt.savefig('plot_6_phase_space.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 6 erstellt: plot_6_phase_space.pdf")
    plt.close()

# ============================================================
# PLOT 7: Pareto-Frontier - Optimal Uncertainty
# ============================================================
def plot_7_pareto_frontier():
    """Pareto-optimale Unsicherheit: Abwägung zwischen Sicherheit und Freiheit"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    omega_range = np.linspace(0.05, 0.95, 100)
    
    # Sicherheit: S(Ω) = -(Ω - 0.5)²  (maximiert bei Ω=0.5)
    security = -(omega_range - 0.5)**2
    security = security - security.min()  # Normalisiert auf [0, max]
    
    # Freiheit: F(Ω) = -ln(1-Ω)
    freedom = -np.log(1 - omega_range)
    freedom = freedom / freedom.max()  # Normalisiert auf [0, 1]
    
    # Pareto-Front: Dominanz-Relation
    pareto_front = []
    for i, omega in enumerate(omega_range):
        is_dominated = False
        for j, omega_other in enumerate(omega_range):
            if (security[j] >= security[i] and freedom[j] >= freedom[i] and
                (security[j] > security[i] or freedom[j] > freedom[i])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append((omega, security[i], freedom[i]))
    
    pareto_front = np.array(pareto_front)
    
    # Plot 1: Sicherheit vs. Freiheit
    axes[0].scatter(freedom, security, alpha=0.5, s=50, c=omega_range, cmap='viridis')
    if len(pareto_front) > 0:
        axes[0].plot(pareto_front[:, 2], pareto_front[:, 1], 'r-', linewidth=3, label='Pareto-Front')
        axes[0].scatter(pareto_front[:, 2], pareto_front[:, 1], color='red', s=100, 
                       marker='*', zorder=5, label='Pareto-optimal')
    
    # Markiere spezielle Punkte
    idx_max_security = np.argmax(security)
    axes[0].scatter([freedom[idx_max_security]], [security[idx_max_security]], 
                   s=200, marker='^', color='green', label='Max. Sicherheit', zorder=5)
    
    idx_max_freedom = np.argmax(freedom)
    axes[0].scatter([freedom[idx_max_freedom]], [security[idx_max_freedom]], 
                   s=200, marker='v', color='orange', label='Max. Freiheit', zorder=5)
    
    axes[0].set_xlabel('Freiheit $F(\\Omega)$')
    axes[0].set_ylabel('Sicherheit $S(\\Omega)$')
    axes[0].set_title('(a) Pareto-Frontier: Freiheit vs. Sicherheit', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Beide Objectives vs. Ω
    axes[1].plot(omega_range, security, linewidth=2.5, label='$S(\\Omega)$', color='blue')
    axes[1].plot(omega_range, freedom, linewidth=2.5, label='$F(\\Omega)$', color='red')
    
    # Markiere optimale Unsicherheit (bei Ω ≈ 0.5)
    axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Optimale Ω*')
    
    axes[1].set_xlabel('Unsicherheitsindex $\\Omega$')
    axes[1].set_ylabel('Normalisierter Wert')
    axes[1].set_title('(b) Sicherheit und Freiheit über Ω', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('plot_7_pareto_frontier.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 7 erstellt: plot_7_pareto_frontier.pdf")
    plt.close()

# ============================================================
# Hauptfunktion - Alle Plots generieren
# ============================================================
def main():
    print("\n" + "="*60)
    print("Generiere matplotlib-Visualisierungen")
    print("="*60 + "\n")
    
    plot_1_tidal_uncertainty()
    plot_2_uncertainty_index()
    plot_3_distributions()
    plot_4_monte_carlo()
    plot_5_freedom_function()
    plot_6_phase_space()
    plot_7_pareto_frontier()
    
    print("\n" + "="*60)
    print("✓ Alle Plots erfolgreich erstellt!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
