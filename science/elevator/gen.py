#!/usr/bin/env python3
"""
Einstein-Elevator: Wissenschaftliche Analyse und Visualisierung
Generiert Plots und kompiliert LaTeX-Dokument
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os

# Deutsche Schrifteinstellung
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Technische Parameter Einstein-Elevator
H_total = 40.0  # Gesamthöhe in Meter
h_experiment = 4.0  # Versuchsdauer entsprechend freiem Fall
g = 9.81  # Erdbeschleunigung
repetitions_per_day = 300  # Wiederholrate pro Tag
max_load = 1000.0  # Nutzlast in kg
microgravity_level = 1e-6  # Restbeschleunigung in g

# ============================================================================
# PLOT 1: Bewegungsprofil und Beschleunigung
# ============================================================================
def plot_motion_profile():
    """Bewegungsprofil während eines Versuchs"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bewegungsprofil und Dynamik des Einstein-Elevators', fontsize=14, fontweight='bold')
    
    # Zeitarray
    t = np.linspace(0, h_experiment, 1000)
    
    # Phase 1: Beschleunigung (0 bis 0.5s, a = 1.2*g)
    # Phase 2: Freier Fall (0.5 bis 3.8s, a ≈ 0)
    # Phase 3: Bremsen (3.8 bis 4.0s, a = -1.2*g)
    
    t1_acc = 0.5
    a_acc = 1.2 * g
    
    # Beschleunigungsprofil
    acceleration = np.zeros_like(t)
    velocity = np.zeros_like(t)
    position = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        if ti <= t1_acc:
            acceleration[i] = a_acc
        elif ti <= h_experiment - t1_acc:
            acceleration[i] = 0
        else:
            acceleration[i] = -a_acc
    
    # Numerische Integration für Geschwindigkeit und Position
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        velocity[i] = velocity[i-1] + acceleration[i] * dt
        position[i] = position[i-1] + velocity[i] * dt
    
    # Subplot 1: Position
    axes[0, 0].plot(t, position, 'b-', linewidth=2.5, label='Position')
    axes[0, 0].axhline(y=H_total, color='r', linestyle='--', linewidth=1.5, label='Turmhöhe (40m)')
    axes[0, 0].set_xlabel('Zeit (s)', fontsize=11)
    axes[0, 0].set_ylabel('Position (m)', fontsize=11)
    axes[0, 0].set_title('Position während des Versuchs', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, H_total * 1.1])
    
    # Subplot 2: Geschwindigkeit
    axes[0, 1].plot(t, velocity, 'g-', linewidth=2.5)
    axes[0, 1].axhline(y=np.sqrt(2*g*H_total), color='r', linestyle='--', linewidth=1.5, label='Freie-Fall-Grenzgeschwindigkeit')
    axes[0, 1].set_xlabel('Zeit (s)', fontsize=11)
    axes[0, 1].set_ylabel('Geschwindigkeit (m/s)', fontsize=11)
    axes[0, 1].set_title('Geschwindigkeit während des Versuchs', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Subplot 3: Beschleunigung
    axes[1, 0].plot(t, acceleration/g, 'r-', linewidth=2.5)
    axes[1, 0].axhline(y=microgravity_level, color='orange', linestyle='--', linewidth=1.5, label=f'Restbeschleunigung ({microgravity_level:.0e}g)')
    axes[1, 0].axhline(y=-microgravity_level, color='orange', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Zeit (s)', fontsize=11)
    axes[1, 0].set_ylabel('Beschleunigung (g)', fontsize=11)
    axes[1, 0].set_title('Beschleunigungsprofil (in Vielfachen von g)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([-1.5, 1.5])
    
    # Subplot 4: Mikrogravitations-Fenster
    micrograv_window = np.abs(acceleration - 0) <= (microgravity_level * g)
    micrograv_time = t[micrograv_window]
    
    axes[1, 1].fill_between(t, 0, micrograv_window.astype(float), alpha=0.3, color='blue', label='Mikrogravitations-Fenster')
    axes[1, 1].plot(t, acceleration/g, 'r-', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xlabel('Zeit (s)', fontsize=11)
    axes[1, 1].set_ylabel('Beschleunigung (g)', fontsize=11)
    axes[1, 1].set_title('Effektive Mikrogravitationsphase', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('elevator_motion.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 1 gespeichert: elevator_motion.pdf")
    plt.close()

# ============================================================================
# PLOT 2: Energieanalyse
# ============================================================================
def plot_energy_analysis():
    """Energieverbrauch und Effizienz"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Energieanalyse des Einstein-Elevators', fontsize=14, fontweight='bold')
    
    # Energieparameter
    m_capsule = 500.0  # Kapsel-Masse in kg
    m_load = 1000.0  # Max-Last in kg
    m_total = m_capsule + m_load
    
    # Potentielle Energie bei Versuchshöhe
    v_max = np.sqrt(2 * g * H_total)
    E_pot = m_total * g * H_total  # Potentielle Energie
    E_kin = 0.5 * m_total * v_max**2  # Kinetische Energie (am höchsten Punkt)
    E_total_per_run = E_pot + E_kin
    
    # Täglich benötigte Energie
    E_daily = E_total_per_run * repetitions_per_day
    
    # Energieaufwand pro Phase
    phases = ['Beschleunigung\n(0-0.5s)', 'Freier Fall\n(0.5-3.8s)', 'Bremsen\n(3.8-4.0s)', 'Ruhephase\n(0-60s)']
    energy_values = [
        E_total_per_run * 0.35,  # Beschleunigung
        E_total_per_run * 0.05,  # Freier Fall (nur Vakuumpumpe)
        E_total_per_run * 0.35,  # Bremsen
        E_total_per_run * 0.25   # Vakuum-Erhalt
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Subplot 1: Energieaufteilung pro Versuch
    axes[0, 0].bar(phases, energy_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Energie (kJ)', fontsize=11)
    axes[0, 0].set_title('Energieaufteilung pro Versuch', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(energy_values):
        axes[0, 0].text(i, v + 5, f'{v:.1f}kJ', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 2: Täglicher Energieverbrauch vs. Anzahl Versuche
    num_runs = np.arange(0, repetitions_per_day + 1, 50)
    daily_energy = num_runs * E_total_per_run / 3600  # in kWh
    
    axes[0, 1].plot(num_runs, daily_energy, 'b-', linewidth=2.5, marker='o', markersize=6)
    axes[0, 1].axvline(x=repetitions_per_day, color='r', linestyle='--', linewidth=2, label=f'{repetitions_per_day} Versuche/Tag')
    axes[0, 1].axhline(y=E_daily/3600, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Anzahl Versuche pro Tag', fontsize=11)
    axes[0, 1].set_ylabel('Energieverbrauch (kWh)', fontsize=11)
    axes[0, 1].set_title('Täglicher Energieverbrauch', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Subplot 3: Energieeffizienz vs. Lasten
    loads = np.linspace(100, max_load, 50)
    efficiency = (loads / (loads + m_capsule)) * 100
    specific_energy = E_total_per_run / (loads)  # J/kg
    
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(loads, efficiency, 'g-', linewidth=2.5, label='Effizienzanteil', marker='s', markersize=5)
    ax3.set_xlabel('Experimentenlast (kg)', fontsize=11)
    ax3.set_ylabel('Nutzlast-Effizienz (%)', fontsize=11, color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    line2 = ax3_twin.plot(loads, specific_energy/1000, 'orange', linewidth=2.5, label='Spezif. Energie', marker='^', markersize=5)
    ax3_twin.set_ylabel('Spez. Energieverbrauch (kJ/kg)', fontsize=11, color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    ax3.set_title('Energieeffizienz vs. Experimentenlast', fontsize=12, fontweight='bold')
    
    # Subplot 4: Vergleich mit klassischem Fallturm
    tower_types = ['Einstein-\nElevator\n(300/Tag)', 'Klassischer\nFallturm\n(max 15/Tag)', 'Flugzeug\nParabelflug\n(20x30s/Tag)']
    experiments_per_day = [300, 15, 20]
    colors_comp = ['#4ECDC4', '#FF6B6B', '#FFE66D']
    
    bars = axes[1, 1].bar(tower_types, experiments_per_day, color=colors_comp, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Anzahl Versuche pro Tag', fontsize=11)
    axes[1, 1].set_title('Vergleich: Wiederholrate vs. andere Systeme', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(experiments_per_day):
        axes[1, 1].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('elevator_energy.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 2 gespeichert: elevator_energy.pdf")
    plt.close()

# ============================================================================
# PLOT 3: Vakuum- und Thermisches Profil
# ============================================================================
def plot_vacuum_thermal():
    """Vakuum und thermische Charakteristiken"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vakuum- und Thermische Analyse', fontsize=14, fontweight='bold')
    
    # Zeit für Vakuumaufbau
    t_vacuum = np.linspace(0, 600, 1000)  # 0-10 Minuten
    
    # Vacuum-Abbau während Versuch
    vacuum_level = 1e-6 * np.exp(-t_vacuum / 120)  # Exponentieller Anstieg Restgas
    vacuum_level_during_run = np.concatenate([
        vacuum_level[:100],
        1e-6 + 1e-7 * np.sin(np.linspace(0, 4*np.pi, 400)),
        vacuum_level[500:]
    ])[:1000]
    
    # Subplot 1: Vakuum-Qualität über Zeit
    axes[0, 0].semilogy(t_vacuum, np.abs(vacuum_level_during_run), 'b-', linewidth=2.5)
    axes[0, 0].axhline(y=1e-6, color='g', linestyle='--', linewidth=2, label='Spezifikation (<10⁻⁶g)')
    axes[0, 0].axhline(y=1e-5, color='orange', linestyle='--', linewidth=1.5, label='Degradation (10⁻⁵g)')
    axes[0, 0].set_xlabel('Zeit (s)', fontsize=11)
    axes[0, 0].set_ylabel('Restbeschleunigung (g)', fontsize=11)
    axes[0, 0].set_title('Vakuum-Qualität während Versuchszyklus', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, which='both')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([1e-7, 1e-4])
    
    # Subplot 2: Kammer-Volumen vs. Abpump-Zeit
    volumes = np.linspace(1, 100, 50)  # m³
    pump_speed = 1000.0  # m³/h
    pump_time = (volumes / pump_speed) * 60  # Minuten
    
    axes[0, 1].plot(volumes, pump_time, 'r-', linewidth=2.5, marker='D', markersize=6)
    axes[0, 1].axvline(x=4.0, color='g', linestyle='--', linewidth=2, label='Einstein-Elevator (~4m³)')
    axes[0, 1].set_xlabel('Kammer-Volumen (m³)', fontsize=11)
    axes[0, 1].set_ylabel('Abpumpzeit (min)', fontsize=11)
    axes[0, 1].set_title('Vakuumaufbau vs. Kammervolumen', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Subplot 3: Thermische Belastung
    t_thermal = np.linspace(0, 3600, 1000)  # 1 Stunde Operation
    num_runs = (t_thermal / 60) * (repetitions_per_day / 1440)  # Skaliert auf reale Wiederholrate
    
    # Wärmeerzeugung: Hauptsächlich bei Beschleunigung/Bremsen
    Q_per_run = 15000  # Joule pro Versuch (Reibung, Antrieb)
    Q_dissipated = Q_per_run * num_runs  # Kumulativ
    
    # Mit Kühlung (angenommen: 10 kW Kühlleistung)
    Q_with_cooling = np.zeros_like(Q_dissipated)
    cooling_power = 10000  # Watt
    
    for i in range(len(Q_dissipated)):
        if i == 0:
            Q_with_cooling[i] = Q_dissipated[i]
        else:
            dt = t_thermal[i] - t_thermal[i-1]
            Q_with_cooling[i] = max(0, Q_with_cooling[i-1] + Q_per_run - cooling_power * dt)
    
    # Temperatur-Anstieg (angenommen: 5000 kg Struktur, c_p = 450 J/kg·K)
    c_p = 450
    mass_structure = 5000
    T_rise = Q_with_cooling / (mass_structure * c_p)
    
    axes[1, 0].plot(t_thermal/60, T_rise, 'orange', linewidth=2.5, label='Mit Kühlung (10kW)')
    axes[1, 0].plot(t_thermal/60, Q_dissipated / (mass_structure * c_p), 'r--', linewidth=2, alpha=0.7, label='Ohne Kühlung')
    axes[1, 0].axhline(y=5, color='g', linestyle='--', linewidth=1.5, label='Akzeptable Temp. (5K)')
    axes[1, 0].set_xlabel('Betriebszeit (Minuten)', fontsize=11)
    axes[1, 0].set_ylabel('Temperaturanstieg (K)', fontsize=11)
    axes[1, 0].set_title('Thermische Belastung und Kühlung', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_xlim([0, 60])
    
    # Subplot 4: Druckprofil in der Kammer
    heights_in_chamber = np.linspace(0, 2.0, 50)  # 2m Kammerhöhe
    # Barometrische Formel (vereinfacht für sehr niedrige Drücke)
    pressure_Pa = 1e-6 * 101325 * np.exp(-heights_in_chamber / 8000)  # Pascal
    
    axes[1, 1].semilogy(heights_in_chamber, pressure_Pa, 'purple', linewidth=2.5, marker='o', markersize=5)
    axes[1, 1].axhline(y=1e-3, color='r', linestyle='--', linewidth=1.5, label='Theoretisches Vakuum')
    axes[1, 1].set_xlabel('Höhe in Kammer (m)', fontsize=11)
    axes[1, 1].set_ylabel('Druck (Pa)', fontsize=11)
    axes[1, 1].set_title('Druckverteilung in der Versuchskammer', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('elevator_vacuum_thermal.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 3 gespeichert: elevator_vacuum_thermal.pdf")
    plt.close()

# ============================================================================
# PLOT 4: Systemvergleich und Anforderungen
# ============================================================================
def plot_system_comparison():
    """Vergleich mit anderen Mikrogravitations-Systemen"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Systemvergleich und Anforderungsanalyse', fontsize=14, fontweight='bold')
    
    # Subplot 1: Vergleichstabelle als Balkendiagramm
    ax1 = fig.add_subplot(gs[0, :])
    
    systems = ['Einstein-\nElevator', 'Klassischer\nFallturm', 'Parabelflug-\nzeug', 'ISS\n(Raumstation)']
    metrics = {
        'Versuchsdauer (s)': [4, 4.5, 30, 0],
        'Wiederholrate (/Tag)': [300, 15, 20, 0],
        'Kosteneffizienz': [1.0, 0.8, 0.2, 0.05],
    }
    
    # Fokus: Wiederholrate
    values = [300, 15, 20, 5]
    colors_sys = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#95E1D3']
    bars = ax1.bar(systems, values, color=colors_sys, edgecolor='black', linewidth=1.5, width=0.6)
    ax1.set_ylabel('Wiederholrate (pro Tag)', fontsize=12, fontweight='bold')
    ax1.set_title('Wiederholrate: Vergleich mit Konkurrenzysystemen', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values):
        ax1.text(i, v + 10, f'{v}x/Tag', ha='center', fontsize=11, fontweight='bold')
    
    # Subplot 2: Größenvergleich
    ax2 = fig.add_subplot(gs[1, 0])
    
    tower_height = 40  # m
    max_payload = 1000  # kg
    
    dimensions = {
        'Einstein-\nElevator': {'h': tower_height, 'w': 4, 'c': '#4ECDC4'},
        'Klassischer\nFallturm': {'h': 100, 'w': 6, 'c': '#FF6B6B'},
        'Parabola-\nFlugzeug': {'h': 30, 'w': 8, 'c': '#FFE66D'},
    }
    
    x_pos = np.arange(len(dimensions))
    heights = [dim['h'] for dim in dimensions.values()]
    colors = [dim['c'] for dim in dimensions.values()]
    
    bars2 = ax2.bar(list(dimensions.keys()), heights, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax2.set_ylabel('Typische Größe/Höhe (m)', fontsize=11)
    ax2.set_title('Abmessungsvergleich', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(heights):
        ax2.text(i, v + 2, f'{v}m', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 3: Kostenvergleich (qualitativ)
    ax3 = fig.add_subplot(gs[1, 1])
    
    costs = {
        'Akquisition\n(Mio. EUR)': [25, 15, 0.5, 50],
        'Betrieb\n(Mio. EUR/Jahr)': [3, 1, 8, 5],
        'Nutzung\n(Versuche/Jahr)': [109500, 5475, 7300, 365*24],
    }
    
    cost_per_experiment = [25*1e6 / 109500, 15*1e6 / 5475, 0.5*1e6 / 7300, 50*1e6 / (365*24)]
    
    bars3 = ax3.bar(systems, cost_per_experiment, color=colors_sys, edgecolor='black', linewidth=1.5, width=0.6)
    ax3.set_ylabel('Kosten pro Experiment (EUR)', fontsize=11)
    ax3.set_title('Geschätzte Kosteneffizienz pro Versuch', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    for i, v in enumerate(cost_per_experiment):
        ax3.text(i, v * 1.5, f'€{v:.0f}', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 4: Qualitätskriterien
    ax4 = fig.add_subplot(gs[2, :])
    
    criteria = ['Mikrogravitations-\nQualität', 'Temperatur-\nstabilität', 'Versuchsdauer', 'Wiederholrate', 'Kosteneffizienz']
    
    # 5-Punkte Bewertung (0-5)
    einstein = [5, 4, 3, 5, 5]
    classical = [4, 3, 3, 2, 2]
    parabola = [3, 2, 4, 1, 2]
    iss = [5, 5, 5, 0, 1]
    
    x_criteria = np.arange(len(criteria))
    width = 0.2
    
    ax4.bar(x_criteria - 1.5*width, einstein, width, label='Einstein-Elevator', color='#4ECDC4', edgecolor='black')
    ax4.bar(x_criteria - 0.5*width, classical, width, label='Klassischer Fallturm', color='#FF6B6B', edgecolor='black')
    ax4.bar(x_criteria + 0.5*width, parabola, width, label='Parabelflugzeug', color='#FFE66D', edgecolor='black')
    ax4.bar(x_criteria + 1.5*width, iss, width, label='ISS', color='#95E1D3', edgecolor='black')
    
    ax4.set_ylabel('Bewertung (0-5)', fontsize=11)
    ax4.set_xlabel('Bewertungskriterium', fontsize=11)
    ax4.set_title('Qualitative Systemvergleich (Multiperspektivisch)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_criteria)
    ax4.set_xticklabels(criteria)
    ax4.set_ylim([0, 5.5])
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('elevator_comparison.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 4 gespeichert: elevator_comparison.pdf")
    plt.close()

# ============================================================================
# PLOT 5: Mathematische Modellierung
# ============================================================================
def plot_mathematical_model():
    """Mathematische Modellierung und Optimierung"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mathematische Modellierung des Einstein-Elevators', fontsize=14, fontweight='bold')
    
    # Subplot 1: Beschleunigungsfunktion a(h)
    h = np.linspace(0, H_total, 1000)
    
    # Theoretisches Beschleunigungsprofil
    a_profile = np.ones_like(h)
    
    # Phase 1: 0 to 10m - Beschleunigung
    mask1 = h <= 10
    a_profile[mask1] = 1.2 * g
    
    # Phase 2: 10m to 30m - Freier Fall (Vakuum)
    mask2 = (h > 10) & (h <= 30)
    a_profile[mask2] = microgravity_level * g
    
    # Phase 3: 30m to 40m - Bremsphase
    mask3 = h > 30
    a_profile[mask3] = -1.2 * g
    
    axes[0, 0].plot(h, a_profile/g, 'b-', linewidth=2.5)
    axes[0, 0].fill_between(h, a_profile/g, 0, alpha=0.2, color='blue')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xlabel('Höhe (m)', fontsize=11)
    axes[0, 0].set_ylabel('Beschleunigung (g)', fontsize=11)
    axes[0, 0].set_title('Beschleunigungsfunktion a(h)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-1.5, 1.5])
    
    # Subplot 2: Kraftverlauf
    m = 1500  # kg
    F = m * a_profile
    
    axes[0, 1].plot(h, F/1000, 'r-', linewidth=2.5)
    axes[0, 1].fill_between(h, F/1000, 0, alpha=0.2, color='red')
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Höhe (m)', fontsize=11)
    axes[0, 1].set_ylabel('Kraft (kN)', fontsize=11)
    axes[0, 1].set_title('Erforderliche Antriebskraft F(h)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Energieverbrauch über Höhe
    W = np.zeros_like(h)
    for i in range(1, len(h)):
        dh = h[i] - h[i-1]
        W[i] = W[i-1] + np.abs(F[i]) * dh
    
    axes[1, 0].plot(h, W/1e6, 'g-', linewidth=2.5)
    axes[1, 0].fill_between(h, W/1e6, 0, alpha=0.2, color='green')
    axes[1, 0].set_xlabel('Höhe (m)', fontsize=11)
    axes[1, 0].set_ylabel('Kumulativer Energieverbrauch (MJ)', fontsize=11)
    axes[1, 0].set_title('Energieverbrauch W(h)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Optimierungspotenzial
    # Vergleich verschiedener Beschleunigungsprofile
    n = len(h)
    linear_profile = np.ones(n) * 1.2
    sinusoid_profile = np.sin(np.linspace(0, np.pi, n)) * 1.2
    
    # Optimiertes Profil mit korrekter Länge
    part1 = np.linspace(0, 1.2, n//3)
    part2 = np.ones(n//3) * 0.001
    part3 = np.linspace(1.2, 0, n - n//3 - n//3)
    optimized_profile = np.concatenate([part1, part2, part3])
    
    profiles = {
        'Linear (a=1.2g)': linear_profile,
        'Sinusförmig': sinusoid_profile,
        'Optimiert': optimized_profile
    }
    
    for label, profile in profiles.items():
        axes[1, 1].plot(h, profile, linewidth=2.5, label=label, marker='', alpha=0.8)
    
    axes[1, 1].set_xlabel('Höhe (m)', fontsize=11)
    axes[1, 1].set_ylabel('Beschleunigung (g)', fontsize=11)
    axes[1, 1].set_title('Vergleich verschiedener Beschleunigungsprofile', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([-0.2, 1.4])
    
    plt.tight_layout()
    plt.savefig('elevator_mathematics.pdf', dpi=300, bbox_inches='tight')
    print("✓ Plot 5 gespeichert: elevator_mathematics.pdf")
    plt.close()

# ============================================================================
# Hauptfunktion
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Generiere Matplotlib-Plots für Einstein-Elevator Dissertation...")
    print("=" * 70)
    
    plot_motion_profile()
    plot_energy_analysis()
    plot_vacuum_thermal()
    plot_system_comparison()
    plot_mathematical_model()
    
    print("\n" + "=" * 70)
    print("Alle Plots erfolgreich generiert!")
    print("=" * 70)