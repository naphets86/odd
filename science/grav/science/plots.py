#!/usr/bin/env python3
"""
Gravitationsmotor – physikalische Untersuchung
Erzeugt alle Plots für das LaTeX-Dokument
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
import os

OUT = 'grav/science/plots"
os.makedirs(OUT, exist_ok=True)

STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(STYLE)

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────
G  = 6.674e-11   # N m² / kg²
g  = 9.81        # m/s²
c  = 3e8         # m/s


# ==============================================================================
# PLOT 1 – Gravitationspotential und Energiebilanz eines einfachen Pendels
# ==============================================================================
def plot1_pendel_energie():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 1a: Potentiallandschaft ---
    ax = axes[0]
    theta = np.linspace(-np.pi, np.pi, 1000)
    m, L = 1.0, 1.0
    U = -m * g * L * np.cos(theta)   # pot. Energie (Minimum bei 0)
    E_ges = [0.5 * m * g * L,        # kleine Amplitude
             m * g * L,              # Separatrix (180°)
             2.0 * m * g * L]        # Rotation

    ax.plot(np.degrees(theta), U, 'b-', lw=2, label=r'$U(\theta)=-mgL\cos\theta$')
    colors = ['green', 'red', 'orange']
    labels = [r'$E < mgL$ (Schwingung)', r'$E = mgL$ (Separatrix)', r'$E > mgL$ (Rotation)']
    for E, col, lab in zip(E_ges, colors, labels):
        ax.axhline(E, color=col, ls='--', lw=1.5, label=lab)
    ax.set_xlabel(r'Winkel $\theta$ [°]')
    ax.set_ylabel(r'Energie $U$ [J]')
    ax.set_title('Potentiallandschaft des Pendels')
    ax.legend(fontsize=8)

    # --- 1b: Phasenraum ---
    ax = axes[1]
    theta_range = np.linspace(-np.pi, np.pi, 400)
    for E in np.linspace(0.1, 3.5, 12):
        omega2 = 2/L * (E/m/g - 1 + np.cos(theta_range))
        mask = omega2 >= 0
        if mask.any():
            om = np.sqrt(np.where(mask, omega2, np.nan))
            ax.plot(np.degrees(theta_range), om, 'b-', lw=0.8, alpha=0.6)
            ax.plot(np.degrees(theta_range), -om, 'b-', lw=0.8, alpha=0.6)

    # Separatrix
    omega_sep = np.sqrt(np.maximum(2*g/L*(1 + np.cos(theta_range)), 0))
    ax.plot(np.degrees(theta_range),  omega_sep, 'r-', lw=2, label='Separatrix')
    ax.plot(np.degrees(theta_range), -omega_sep, 'r-', lw=2)
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel(r'$\dot\theta$ [rad/s]')
    ax.set_title('Phasenraumportrait')
    ax.legend()

    # --- 1c: Energie über Zeit – verlustfrei vs. reib ---
    ax = axes[2]
    t = np.linspace(0, 10, 1000)
    E0 = 1.0
    E_id  = np.full_like(t, E0)
    E_dis = E0 * np.exp(-0.3 * t)
    ax.plot(t, E_id,  'g-',  lw=2, label='Ideal (kein Reibungsverlust)')
    ax.plot(t, E_dis, 'r--', lw=2, label=r'Realistisch ($\gamma=0.3$)')
    ax.fill_between(t, E_dis, E_id, alpha=0.2, color='red', label='Energieverlust')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Energie [J]')
    ax.set_title('Energiebilanz: ideal vs. realistisch')
    ax.legend()

    fig.suptitle('Abbildung 1 – Pendelphysik und Energieerhaltung', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot1_pendel.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot1_pendel.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 1 gespeichert.")


# ==============================================================================
# PLOT 2 – Thermodynamische Grenzen: Carnot, verfügbare Arbeit
# ==============================================================================
def plot2_thermodynamik():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 2a: Carnot-Effizienz ---
    ax = axes[0]
    T_kalt = np.linspace(1, 500, 500)
    T_warm = [400, 600, 800, 1000]
    for Tw in T_warm:
        eta = 1 - T_kalt / Tw
        mask = (T_kalt < Tw) & (eta > 0)
        ax.plot(T_kalt[mask], eta[mask]*100, lw=2, label=f'$T_H={Tw}$ K')
    ax.set_xlabel(r'$T_C$ [K]')
    ax.set_ylabel(r'$\eta_{Carnot}$ [%]')
    ax.set_title('Carnot-Wirkungsgrad')
    ax.legend()

    # --- 2b: Verfügbare Arbeit eines Gewichts ---
    ax = axes[1]
    h = np.linspace(0, 100, 500)
    m_vals = [1, 5, 10, 50, 100]
    for m in m_vals:
        W = m * g * h
        ax.plot(h, W, lw=2, label=f'm = {m} kg')
    ax.set_xlabel('Fallhöhe $h$ [m]')
    ax.set_ylabel('Arbeit $W = mgh$ [J]')
    ax.set_title('Gravitative potentielle Energie')
    ax.legend(fontsize=8)

    # --- 2c: Leistungsvergleich Motor-Konzepte ---
    ax = axes[2]
    kategorien = ['Elektromotor\n(50 kW)', 'Verbrennungsmotor\n(150 kW)',
                  'Gravitations-\nmotor (1 kg, 10 m)', 'Gravitations-\nmotor (1000 kg, 10 m)']
    P = [50000, 150000, 1*9.81*10/1, 1000*9.81*10/1]  # W (letzter: m*g*h in 1 s)
    farben = ['royalblue', 'darkorange', 'tomato', 'tomato']
    bars = ax.bar(kategorien, P, color=farben, edgecolor='black', lw=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('Leistung [W]')
    ax.set_title('Leistungsvergleich')
    for bar, val in zip(bars, P):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.3, f'{val:.0f} W',
                ha='center', va='bottom', fontsize=8)

    fig.suptitle('Abbildung 2 – Thermodynamische Grenzen und Leistungsvergleich',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot2_thermo.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot2_thermo.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 2 gespeichert.")


# ==============================================================================
# PLOT 3 – Perpetuum Mobile: Energiebilanz und Thermodynamische Unmöglichkeit
# ==============================================================================
def plot3_perpetuum():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 3a: Idealer vs. realer Zyklus (E über Zyklen) ---
    ax = axes[0]
    zyklen = np.arange(0, 50)
    E_ideal = np.ones(50)
    eta_arr = [0.99, 0.95, 0.90, 0.80]
    for eta in eta_arr:
        E_real = eta**zyklen
        ax.plot(zyklen, E_real, lw=2, label=f'$\\eta={eta}$')
    ax.plot(zyklen, E_ideal, 'k--', lw=2, label='Ideal (PM I)')
    ax.set_xlabel('Zyklus')
    ax.set_ylabel('Relative Energie')
    ax.set_title('Energieabnahme pro Zyklus')
    ax.legend()

    # --- 3b: Entropieproduktion ---
    ax = axes[1]
    T_arr = np.linspace(100, 1000, 500)
    Q = 1000  # J
    dS_irr = [0.1, 0.5, 1.0, 2.0]
    dS_rev = Q / T_arr
    ax.plot(T_arr, dS_rev, 'b-', lw=2.5, label=r'$\Delta S_{rev} = Q/T$')
    for dS in dS_irr:
        ax.axhline(dS_rev.min() + dS, ls='--', lw=1.5,
                   label=fr'$\Delta S_{{irr}} = {dS}$ J/K (irreversibel)')
    ax.set_xlabel('Temperatur $T$ [K]')
    ax.set_ylabel(r'$\Delta S$ [J/K]')
    ax.set_title('2. Hauptsatz: Entropie')
    ax.legend(fontsize=8)

    # --- 3c: Sankey-ähnliche Energiebilanz ---
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Kreisprozess-Energiebilanz')

    # Kästen
    def kasten(ax, x, y, w, h, label, color):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.1", fc=color, ec='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    kasten(ax, 5, 8.5, 3, 1.2, 'Eingangsenergie\n$E_{in}$',   '#aed6f1')
    kasten(ax, 5, 5.5, 3, 1.2, 'Motor\n(Reibung: $Q_{Reib}$)', '#f9e79f')
    kasten(ax, 2, 2.7, 2.5, 1.2, 'Nutzarbeit\n$W_{nutz}$',    '#a9dfbf')
    kasten(ax, 8, 2.7, 2.5, 1.2, 'Verluste\n$Q_{Reib}+Q_{Luft}$', '#f1948a')

    for (x1,y1),(x2,y2) in [((5,7.9),(5,6.1)), ((4,4.95),(2.5,3.3)), ((6,4.95),(7.5,3.3))]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(3.2, 4.5, r'$W_{nutz} = E_{in} - Q_{ges}$', fontsize=9,
            bbox=dict(fc='lightyellow', ec='gray', pad=3))
    ax.text(3.2, 3.9, r'$\eta = W_{nutz}/E_{in} < 1$', fontsize=9, color='red',
            bbox=dict(fc='lightyellow', ec='gray', pad=3))

    fig.suptitle('Abbildung 3 – Perpetuum Mobile und Energieverluste',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot3_perpetuum.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot3_perpetuum.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 3 gespeichert.")


# ==============================================================================
# PLOT 4 – Radialpendel / Wasserrad: numerische ODE-Simulation
# ==============================================================================
def plot4_ode_pendel():
    """Gedämpftes nichtlineares Pendel – Simulation."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    L_val = 1.0
    m_val = 1.0

    def pendel_ode(t, y, gamma):
        theta, omega = y
        domega = -(g/L_val)*np.sin(theta) - gamma*omega
        return [omega, domega]

    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 3000)
    theta0 = np.radians(60)

    gammas = [0.0, 0.05, 0.2, 0.5]
    farben = ['blue', 'green', 'orange', 'red']

    # --- 4a: theta(t) ---
    ax = axes[0, 0]
    for gam, col in zip(gammas, farben):
        sol = solve_ivp(pendel_ode, t_span, [theta0, 0], args=(gam,), t_eval=t_eval, rtol=1e-8)
        ax.plot(sol.t, np.degrees(sol.y[0]), color=col, lw=1.5, label=rf'$\gamma={gam}$')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel(r'$\theta$ [°]')
    ax.set_title('Pendelauslenkung')
    ax.legend()

    # --- 4b: Phasenraum ---
    ax = axes[0, 1]
    for gam, col in zip(gammas, farben):
        sol = solve_ivp(pendel_ode, t_span, [theta0, 0], args=(gam,), t_eval=t_eval, rtol=1e-8)
        ax.plot(np.degrees(sol.y[0]), sol.y[1], color=col, lw=1, alpha=0.8, label=rf'$\gamma={gam}$')
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel(r'$\dot\theta$ [rad/s]')
    ax.set_title('Phasenraum (gedämpft)')
    ax.legend()

    # --- 4c: Energie(t) ---
    ax = axes[0, 2]
    for gam, col in zip(gammas, farben):
        sol = solve_ivp(pendel_ode, t_span, [theta0, 0], args=(gam,), t_eval=t_eval, rtol=1e-8)
        T = 0.5 * m_val * (L_val * sol.y[1])**2
        V = m_val * g * L_val * (1 - np.cos(sol.y[0]))
        E = T + V
        ax.plot(sol.t, E, color=col, lw=1.5, label=rf'$\gamma={gam}$')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Energie [J]')
    ax.set_title('Gesamtenergie')
    ax.legend()

    # --- 4d: Erzwungenes Pendel – Resonanz ---
    ax = axes[1, 0]
    def pendel_forced(t, y, gamma, F_drive, omega_drive):
        theta, omega = y
        domega = -(g/L_val)*np.sin(theta) - gamma*omega + F_drive*np.cos(omega_drive*t)
        return [omega, domega]

    omega_nat = np.sqrt(g/L_val)
    t_eval2 = np.linspace(0, 60, 6000)
    drive_freqs = [0.5*omega_nat, 0.9*omega_nat, omega_nat, 1.1*omega_nat]
    for od, col in zip(drive_freqs, farben):
        sol = solve_ivp(pendel_forced, (0,60), [0.1, 0],
                        args=(0.1, 0.5, od), t_eval=t_eval2, rtol=1e-7)
        ax.plot(sol.t, np.degrees(sol.y[0]), color=col, lw=1,
                alpha=0.8, label=rf'$\omega_d={od:.2f}$')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel(r'$\theta$ [°]')
    ax.set_title('Erzwungene Schwingung')
    ax.legend(fontsize=8)

    # --- 4e: Amplitudenresonanz ---
    ax = axes[1, 1]
    freqs = np.linspace(0.1, 3.0, 300)
    gammas_res = [0.05, 0.1, 0.3, 0.5]
    omega0 = np.sqrt(g/L_val)
    for gam, col in zip(gammas_res, ['blue','green','orange','red']):
        A = 1.0 / np.sqrt((omega0**2 - freqs**2)**2 + (gam*freqs)**2)
        ax.plot(freqs, A, color=col, lw=2, label=rf'$\gamma={gam}$')
    ax.axvline(omega0, color='gray', ls='--', lw=1, label=r'$\omega_0$')
    ax.set_xlabel(r'$\omega_d$ [rad/s]')
    ax.set_ylabel('Amplitude [a.u.]')
    ax.set_title('Resonanzkurve')
    ax.legend()

    # --- 4f: Energieverlust pro Halbschwingung ---
    ax = axes[1, 2]
    gamma_arr = np.linspace(0.0, 2.0, 400)
    # Theoretisch: E_n = E_0 * exp(-gamma * T_half)
    # T_half ≈ pi / omega_d where omega_d = sqrt(omega0^2 - gamma^2/4)
    E_verlust = []
    for gam in gamma_arr:
        od2 = g/L_val - (gam/2)**2
        if od2 > 0:
            T_half = np.pi / np.sqrt(od2)
            ratio = np.exp(-gam * T_half)
        else:
            ratio = 0.0
        E_verlust.append(1 - ratio)

    ax.plot(gamma_arr, E_verlust, 'purple', lw=2.5)
    ax.axvline(2*np.sqrt(g/L_val), color='red', ls='--', lw=1.5, label='Kritische Dämpfung')
    ax.set_xlabel(r'Dämpfungskoeff. $\gamma$')
    ax.set_ylabel('Relativer Energieverlust pro Halbperiode')
    ax.set_title('Verlust vs. Dämpfung')
    ax.legend()

    fig.suptitle('Abbildung 4 – Numerische Pendelsimulation (ODE)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot4_ode.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot4_ode.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 4 gespeichert.")


# ==============================================================================
# PLOT 5 – Übergewichtiges Rad (Overbalanced Wheel) – Klassische Analyse
# ==============================================================================
def plot5_overbalanced_wheel():
    """Analyse des 'Bessler-Rades' – zeigt warum es nicht funktioniert."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- 5a: Drehmoment als Funktion des Winkels ---
    ax = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 1000)
    R = 1.0  # Radius
    # Annahme: N Massen auf Kreis, Drehmoment jeder Masse = m*g*r*sin(theta_i)
    N_massen = [4, 6, 8, 12]
    for N in N_massen:
        tau_total = np.zeros_like(theta)
        for k in range(N):
            phi_k = theta + k * 2*np.pi/N
            tau_total += np.sin(phi_k)
        ax.plot(np.degrees(theta), tau_total, lw=1.5, label=f'N={N}')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel(r'Winkel $\theta$ [°]')
    ax.set_ylabel(r'$\tau / (mgR)$ [a.u.]')
    ax.set_title('Netto-Drehmoment (N Massen)')
    ax.legend()

    # --- 5b: Integriertes Drehmoment (Arbeit pro Umdrehung) ---
    ax = axes[0, 1]
    N_arr = np.arange(1, 25)
    arbeit = []
    for N in N_arr:
        tau_int, _ = quad(lambda th: sum(np.sin(th + k*2*np.pi/N) for k in range(N)), 0, 2*np.pi)
        arbeit.append(abs(tau_int))
    ax.bar(N_arr, arbeit, color='steelblue', edgecolor='black')
    ax.axhline(0, color='red', lw=2, label='Netto-Arbeit = 0')
    ax.set_xlabel('Anzahl Massen N')
    ax.set_ylabel('|Netto-Arbeit| [a.u.]')
    ax.set_title('Arbeit pro Umdrehung (immer 0)')
    ax.legend()

    # --- 5c: Schwerpunkt des Rades ---
    ax = axes[0, 2]
    t_rot = np.linspace(0, 4*np.pi, 500)
    N = 8
    # Schwerpunkt x und y als Funktion des Rotationswinkels
    sx = np.array([sum(np.cos(t + k*2*np.pi/N) for k in range(N)) / N for t in t_rot])
    sy = np.array([sum(np.sin(t + k*2*np.pi/N) for k in range(N)) / N for t in t_rot])
    ax.plot(np.degrees(t_rot), sx, lw=2, label=r'$x_{SP}$')
    ax.plot(np.degrees(t_rot), sy, lw=2, label=r'$y_{SP}$')
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_xlabel(r'Rotationswinkel [°]')
    ax.set_ylabel('Schwerpunktposition (norm.)')
    ax.set_title('Schwerpunkt bleibt bei Ursprung')
    ax.legend()

    # --- 5d: Bessler-Rad Simulation (gleitendes Gewicht) ---
    ax = axes[1, 0]
    # Modell: Gewicht kann radial verschoben werden zwischen r_min und r_max
    r_min, r_max = 0.3, 1.0
    theta_arr = np.linspace(0, 2*np.pi, 1000)
    # Vereinfachtes Modell: r(theta) = r_min + (r_max - r_min) * (1 + sin(theta))/2
    r = r_min + (r_max - r_min) * (1 + np.sin(theta_arr))/2
    tau = r * np.sin(theta_arr)  # proportional zum Drehmoment
    ax.plot(np.degrees(theta_arr), tau, 'b-', lw=2, label='Drehmoment (gleitendes Gewicht)')
    ax.fill_between(np.degrees(theta_arr), tau, alpha=0.3)
    net = np.trapezoid(tau, theta_arr)
    ax.set_title(f'Gleitendes Gewicht\nNetto-Drehmoment integral = {net:.4f}')
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel(r'$\tau$ [a.u.]')
    ax.legend()

    # --- 5e: Energiefluss im Rotating System ---
    ax = axes[1, 1]
    # Zeige: Was man auf der einen Seite gewinnt, verliert man auf der anderen
    theta_sym = np.linspace(0, 2*np.pi, 1000)
    tau_right = np.maximum(np.sin(theta_sym), 0)  # Positive Seite
    tau_left  = np.minimum(np.sin(theta_sym), 0)  # Negative Seite
    ax.fill_between(np.degrees(theta_sym), tau_right, 0, alpha=0.5, color='green', label='Antreibend (+)')
    ax.fill_between(np.degrees(theta_sym), tau_left,  0, alpha=0.5, color='red',   label='Bremsend (−)')
    ax.plot(np.degrees(theta_sym), np.sin(theta_sym), 'k-', lw=1.5)
    ax.axhline(0, color='black', lw=1)
    int_pos = np.trapezoid(tau_right, np.degrees(theta_sym))
    int_neg = abs(np.trapezoid(tau_left, np.degrees(theta_sym)))
    ax.set_title(f'Antreibend: {int_pos:.1f}, Bremsend: {int_neg:.1f} → Netto=0')
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel(r'$\tau/mgR$')
    ax.legend()

    # --- 5f: Verschiedene 'Maschinen'-Konzepte und ihr Effizienz ---
    ax = axes[1, 2]
    konzepte = ['Wasserrad\n(Gefälle)', 'Windrad', 'Übergewichts-\nRad', 'Permanentmagnet-\nMotor (PM)',
                'Elektromotor']
    effizienz = [85, 45, 0, 0, 95]
    farben2 = ['blue', 'lightblue', 'red', 'red', 'green']
    bars = ax.bar(konzepte, effizienz, color=farben2, edgecolor='black')
    ax.set_ylabel('Tatsächlicher Wirkungsgrad η [%]')
    ax.set_title('Wirkungsgrad verschiedener Konzepte')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, effizienz):
        ax.text(bar.get_x()+bar.get_width()/2, val+2, f'{val}%', ha='center', fontsize=9)

    fig.suptitle('Abbildung 5 – Übergewichtiges Rad: Analyse und Beweis des Scheiterns',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot5_wheel.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot5_wheel.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 5 gespeichert.")


# ==============================================================================
# PLOT 6 – Gravitations-Quanteneffekte und Relativität
# ==============================================================================
def plot6_quantengravitation():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- 6a: Gravitationspotential Erde ---
    ax = axes[0, 0]
    R_E = 6.371e6  # m
    M_E = 5.972e24  # kg
    r = np.linspace(R_E, 5*R_E, 1000)
    Phi = -G * M_E / r
    ax.plot(r/R_E, Phi/1e7, 'b-', lw=2)
    ax.axvline(1, color='brown', ls='--', lw=1.5, label='Erdoberfläche')
    ax.set_xlabel(r'$r / R_E$')
    ax.set_ylabel(r'$\Phi(r)$ [$10^7$ m²/s²]')
    ax.set_title('Gravitationspotential der Erde')
    ax.legend()

    # --- 6b: Gravitationsrotverschiebaung ---
    ax = axes[0, 1]
    dh = np.linspace(0, 1e6, 1000)  # Höhendifferenz in m
    # Gravitational redshift: Δf/f = g*h/c^2
    delta_f_over_f = g * dh / c**2
    ax.plot(dh/1000, delta_f_over_f, 'purple', lw=2)
    ax.set_xlabel('Höhe $h$ [km]')
    ax.set_ylabel(r'$\Delta f / f$')
    ax.set_title('Gravitationsrotverschiebung')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # --- 6c: Casimir-Kraft (Quantenvakuum) ---
    ax = axes[0, 2]
    hbar = 1.0546e-34
    d = np.linspace(1e-9, 1e-6, 1000)  # m
    L_plate = 1e-2  # 1 cm Platte
    # Casimir: F/A = -pi^2 hbar c / (240 d^4)
    F_casimir = -np.pi**2 * hbar * c / (240 * d**4) * L_plate**2
    ax.loglog(d*1e9, np.abs(F_casimir)*1e9, 'darkcyan', lw=2)
    ax.set_xlabel('Plattenabstand $d$ [nm]')
    ax.set_ylabel('Casimir-Kraft [nN]')
    ax.set_title('Casimir-Effekt (1×1 cm² Platten)')

    # --- 6d: Planckskalierung ---
    ax = axes[1, 0]
    kategorien = ['Planck-Länge\n$l_P$', 'Planck-Zeit\n$t_P$', 'Planck-Masse\n$m_P$', 'Planck-Energie\n$E_P$']
    werte = [1.616e-35, 5.391e-44, 2.176e-8, 1.956e9]
    einheiten = ['m', 's', 'kg', 'J']
    colors = ['steelblue', 'seagreen', 'darkorange', 'crimson']
    ax.set_axis_off()
    table_data = [[k, f'{v:.3e} {u}'] for k,v,u in zip(kategorien, werte, einheiten)]
    table = ax.table(cellText=table_data, colLabels=['Größe', 'Wert'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2.2)
    ax.set_title('Planck-Skalen', pad=20)

    # --- 6e: Äquivalenzprinzip – freier Fall ---
    ax = axes[1, 1]
    t = np.linspace(0, 5, 500)
    h_start = 100
    h_fall = h_start - 0.5 * g * t**2
    mask = h_fall >= 0
    v_fall = g * t
    ax.plot(t[mask], h_fall[mask], 'b-', lw=2, label='Höhe $h(t)$')
    ax2 = ax.twinx()
    ax2.plot(t[mask], v_fall[mask], 'r--', lw=2, label='Geschwindigkeit $v(t)$')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Höhe [m]', color='blue')
    ax2.set_ylabel('Geschwindigkeit [m/s]', color='red')
    ax.set_title('Freier Fall ($h_0 = 100$ m)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='center right')

    # --- 6f: Energiedichte Gravitationsfeld vs. EM-Feld ---
    ax = axes[1, 2]
    mu0 = 4*np.pi*1e-7
    epsilon0 = 8.854e-12
    # B-Feld Energie: u_B = B^2/(2*mu0)
    B = np.logspace(-3, 2, 500)
    u_B = B**2 / (2*mu0)
    # Gravitationsfeld: nicht direkt vergleichbar, aber g-Feld-Energie-Äquivalent
    # Scheinbar: u_g = g^2 / (8*pi*G) -- aus linearisierter GR
    g_arr = np.logspace(-3, 2, 500)
    u_g = g_arr**2 / (8*np.pi*G)
    ax.loglog(B, u_B, 'b-', lw=2, label=r'EM: $u_B = B^2/(2\mu_0)$')
    ax.loglog(g_arr, u_g, 'r-', lw=2, label=r'Grav.: $u_g = g^2/(8\pi G)$')
    ax.set_xlabel('Feldstärke [T] oder [m/s²]')
    ax.set_ylabel('Energiedichte [J/m³]')
    ax.set_title('Energiedichte: EM vs. Gravitationsfeld')
    ax.legend()

    fig.suptitle('Abbildung 6 – Quantengravitation, Relativität und Fundamentalphysik',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot6_quanten.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot6_quanten.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 6 gespeichert.")


# ==============================================================================
# PLOT 7 – Elektromotor vs. Gravitationsmotor: Quantitativer Vergleich
# ==============================================================================
def plot7_vergleich():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- 7a: Lorentz-Kraft F = qvB ---
    ax = axes[0, 0]
    B_vals = [0.5, 1.0, 2.0]
    v = np.linspace(0, 100, 500)  # m/s
    q = 1.0  # C
    for B_val in B_vals:
        F = q * v * B_val
        ax.plot(v, F, lw=2, label=f'B = {B_val} T')
    ax.set_xlabel('Geschwindigkeit $v$ [m/s]')
    ax.set_ylabel('Lorentzkraft $F = qvB$ [N]')
    ax.set_title('Lorentzkraft im Elektromotor')
    ax.legend()

    # --- 7b: Energiedichte Vergleich ---
    ax = axes[0, 1]
    technologien = ['Li-Ion\nAkkumulator', 'Benzin', 'Wasserstoff\n(komprimiert)',
                    'Gesp. grav.\nEnergie\n(100m, 1t)', 'Superkond.']
    # Wh/kg
    energiedichte = [250, 12000, 33000, 0.0027, 5]
    farben3 = ['blue', 'darkorange', 'lightblue', 'tomato', 'purple']
    bars = ax.bar(technologien, energiedichte, color=farben3, edgecolor='black', lw=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('Energiedichte [Wh/kg]')
    ax.set_title('Energiedichte verschiedener Speicher')
    for bar, val in zip(bars, energiedichte):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.5, f'{val:.4g}', ha='center', fontsize=8, rotation=0)

    # --- 7c: Kraftfeld-Reichweite ---
    ax = axes[0, 2]
    r = np.linspace(0.1, 10, 500)  # m
    M = 1000  # kg Masse
    q_c = 1e-3  # Coulomb
    E_field = 1e5  # V/m
    F_grav = G * M * 1 / r**2       # Gravitationskraft auf 1 kg
    F_em   = q_c * E_field * np.ones_like(r)  # E-Kraft (uniform field)
    F_mag  = 1.0 / r**3             # magnetisches Dipolfeld ~1/r^3
    ax.semilogy(r, F_grav,  'b-',  lw=2, label=r'Gravitationskraft $\propto r^{-2}$')
    ax.semilogy(r, F_em,    'g--', lw=2, label='Elektrische Kraft (uniform)')
    ax.semilogy(r, F_mag,   'r:',  lw=2, label=r'Magn. Dipol $\propto r^{-3}$')
    ax.set_xlabel('Abstand $r$ [m]')
    ax.set_ylabel('Kraft [N] (normiert)')
    ax.set_title('Kraftfelder: Reichweite und Stärke')
    ax.legend()

    # --- 7d: Leistung vs. Masse für Gravitationsantrieb ---
    ax = axes[1, 0]
    m_arr = np.logspace(0, 6, 500)  # kg
    h_fall = 10  # m
    t_cycle = 1  # s Zykluszeit
    P = m_arr * g * h_fall / t_cycle
    ax.loglog(m_arr, P, 'b-', lw=2.5, label='Leistung $P = mgh/t$')
    ax.axhline(1000, color='green', ls='--', lw=2, label='1 kW (Referenz)')
    ax.axhline(50000, color='orange', ls='--', lw=2, label='50 kW (Elektromotor-Äquivalent)')
    # Welche Masse nötig für 50 kW?
    m_needed = 50000 / (g * h_fall / t_cycle)
    ax.axvline(m_needed, color='red', ls=':', lw=2, label=f'm = {m_needed:.0f} kg nötig')
    ax.set_xlabel('Masse [kg]')
    ax.set_ylabel('Leistung [W]')
    ax.set_title(f'Gravit. Leistung (h={h_fall} m, t={t_cycle} s)')
    ax.legend(fontsize=8)

    # --- 7e: Wirkungsgrad über Frequenz ---
    ax = axes[1, 1]
    freq = np.logspace(-2, 3, 500)
    # Elektromotor: ~konstant hoch bis Eckfrequenz, dann Abfall
    eta_em = 0.95 / (1 + (freq/1000)**2)
    # Hydraulik: gute Wirkungsgrade mittlerer Bereich
    eta_hydr = 0.85 * np.exp(-(np.log(freq/50))**2 / 4)
    # Gravitation: bei sehr niedriger Frequenz maximal, hochfrequent nicht sinnvoll
    eta_grav = 0.70 / (1 + (freq/0.1)**1.5)
    ax.semilogx(freq, eta_em*100,   'b-',  lw=2, label='Elektromotor')
    ax.semilogx(freq, eta_hydr*100, 'g--', lw=2, label='Hydraulikmotor')
    ax.semilogx(freq, eta_grav*100, 'r:',  lw=2, label='Gravitationsantrieb')
    ax.set_xlabel('Betriebsfrequenz [Hz]')
    ax.set_ylabel('Wirkungsgrad η [%]')
    ax.set_title('Wirkungsgrad vs. Frequenz')
    ax.legend()

    # --- 7f: Fundamantale Naturkonstanten im Vergleich ---
    ax = axes[1, 2]
    ax.set_axis_off()
    data = [
        ['Gravitationskonstante $G$', f'{G:.3e}', 'N m²/kg²'],
        ['Elementarladung $e$',        '1.602×10⁻¹⁹', 'C'],
        [r'Permeabilität $\mu_0$',       '1.257×10⁻⁶', 'H/m'],
        ['Lichtgeschwindigkeit $c$',    '3×10⁸', 'm/s'],
        [r'Planck-Konst. $\hbar$',       '1.055×10⁻³⁴', 'J·s'],
        ['Grav./EM Verhältnis',         '~10⁻⁴²', 'dimensionslos'],
    ]
    table = ax.table(cellText=data, colLabels=['Konstante', 'Wert', 'Einheit'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 1.9)
    ax.set_title('Fundamentale Konstanten', pad=20)

    fig.suptitle('Abbildung 7 – Elektromotor vs. Gravitationsmotor: Quantitativer Vergleich',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot7_vergleich.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot7_vergleich.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 7 gespeichert.")


# ==============================================================================
# PLOT 8 – Erweiterte Simulationen: Schwungrad, Gezeitenkraft
# ==============================================================================
def plot8_erweitert():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- 8a: Schwungrad-Energiespeicher ---
    ax = axes[0, 0]
    omega_arr = np.linspace(0, 1000, 500)  # rad/s
    I_vals = [0.1, 1.0, 10.0, 100.0]  # kg m²
    for I in I_vals:
        E = 0.5 * I * omega_arr**2 / 1000  # kJ
        ax.plot(omega_arr, E, lw=2, label=f'I = {I} kg·m²')
    ax.set_xlabel(r'Winkelgeschwindigkeit $\omega$ [rad/s]')
    ax.set_ylabel('Gespeicherte Energie [kJ]')
    ax.set_title('Schwungradspeicher')
    ax.legend()

    # --- 8b: Gezeitenkraft ---
    ax = axes[0, 1]
    M_M = 7.342e22  # Mondmasse
    d_EM = 3.844e8  # Erde-Mond-Abstand
    R_E2 = 6.371e6
    # Gezeitenbeschleunigung auf Erdoberfläche
    theta_arr2 = np.linspace(0, np.pi, 500)
    # Gezeitenforce = 2*G*M_M*R_E*cos(theta)/d^3
    a_tidal = 2 * G * M_M * R_E2 * np.cos(theta_arr2) / d_EM**3
    ax.plot(np.degrees(theta_arr2), a_tidal * 1e7, 'teal', lw=2)
    ax.set_xlabel(r'Winkel $\theta$ (zum Mond) [°]')
    ax.set_ylabel(r'Gezeitenbeschleunigung [$10^{-7}$ m/s²]')
    ax.set_title('Gezeitenkraft des Mondes')

    # --- 8c: Gezeiten als Energiequelle ---
    ax = axes[0, 2]
    installierte_leistung = np.logspace(3, 9, 400)  # W
    effizienz_gezeiten = 0.25
    jahrliche_energie = installierte_leistung * effizienz_gezeiten * 8760  # Wh
    P_gezeitenanlage = [0.4e6, 240e6, 1000e6]  # MW (La Rance, Sihwa, Swansea)
    namen = ['La Rance (240 MW)', 'Sihwa (254 MW)', 'Pentland Firth (~1.9 GW pot.)']
    ax.loglog(installierte_leistung/1e6, jahrliche_energie/1e9, 'b-', lw=2)
    for P, name in zip([240e6, 254e6, 1.9e9], namen):
        E_ann = P * effizienz_gezeiten * 8760 / 1e9
        ax.scatter(P/1e6, E_ann, s=100, zorder=5)
        ax.annotate(name, (P/1e6, E_ann), textcoords='offset points', xytext=(5,5), fontsize=7)
    ax.set_xlabel('Installierte Leistung [MW]')
    ax.set_ylabel('Jährl. Energie [GWh]')
    ax.set_title('Gezeitenkraftwerke (Gravitation legitim nutzbar)')

    # --- 8d: Biomechanik – menschliche Gangenergie ---
    ax = axes[1, 0]
    t2 = np.linspace(0, 2*np.pi, 500)
    # Inverted pendulum model: COM vertical displacement
    h_com = 0.02 * np.cos(t2)  # ~2 cm Hub
    E_pot = 70 * g * 0.02 * np.cos(t2)  # 70 kg Mensch
    E_kin = -E_pot  # In phase exchange (ideal)
    ax.plot(np.degrees(t2), E_pot, 'b-', lw=2, label='Pot. Energie')
    ax.plot(np.degrees(t2), -E_kin, 'r--', lw=2, label='Kin. Energie (ideal)')
    ax.set_xlabel('Gangphase [°]')
    ax.set_ylabel('Energie [J]')
    ax.set_title('Biomechanik: Energie beim Gehen\n(Gravity-Assisted Locomotion)')
    ax.legend()

    # --- 8e: Orbital Mechanics – Gezeitenkraft ---
    ax = axes[1, 1]
    a_orb = np.logspace(6, 12, 500)  # m
    M_planet = 5.972e24  # kg
    v_orb = np.sqrt(G * M_planet / a_orb)
    T_orb = 2 * np.pi * a_orb / v_orb / 3600  # h
    ax.loglog(a_orb/1e6, T_orb, 'purple', lw=2.5)
    # Besondere Orbits
    punkte = {'LEO\n(400 km)': 400e3+6.371e6, 'MEO\n(20200 km)': 20200e3+6.371e6,
              'GEO\n(35786 km)': 35786e3+6.371e6, 'Mond\n(384400 km)': 3.844e8}
    for name, r_val in punkte.items():
        T_val = 2*np.pi*r_val/np.sqrt(G*M_planet/r_val)/3600
        ax.scatter(r_val/1e6, T_val, s=80, zorder=5)
        ax.annotate(name, (r_val/1e6, T_val), textcoords='offset points', xytext=(4,4), fontsize=8)
    ax.set_xlabel('Orbitradius [Mm]')
    ax.set_ylabel('Umlaufzeit [h]')
    ax.set_title("Kepler's 3. Gesetz")

    # --- 8f: Fazit – Stärkenvergleich der fundamentalen Kräfte ---
    ax = axes[1, 2]
    kraefte = ['Starke\nKernkraft', 'Elektro-\nmagnetisch', 'Schwache\nKernkraft', 'Gravitation']
    relative_staerke = [1, 1/137, 1e-6, 6e-39]
    farben4 = ['red', 'blue', 'orange', 'gray']
    bars = ax.bar(kraefte, [np.log10(max(s,1e-40)) + 40 for s in relative_staerke],
                  color=farben4, edgecolor='black')
    for bar, (krft, val) in zip(bars, zip(kraefte, relative_staerke)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.0e}', ha='center', fontsize=8)
    ax.set_ylabel('log₁₀(Stärke) + 40 [normiert]')
    ax.set_title('Stärke der Grundkräfte\n(Gravitation ist ~39 Größenordnungen schwächer)')

    fig.suptitle('Abbildung 8 – Erweiterte Analyse: Schwungrad, Gezeiten, Orbitalmechanik',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot8_erweitert.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot8_erweitert.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 8 gespeichert.")


# ==============================================================================
# PLOT 9 – Zusammenfassung: Warum ein geschlossener Gravitationsmotor scheitert
# ==============================================================================
def plot9_zusammenfassung():
    fig = plt.figure(figsize=(14, 10))
    ax00 = fig.add_subplot(2, 2, 1)
    ax01 = fig.add_subplot(2, 2, 2)
    ax10 = fig.add_subplot(2, 2, 3)
    ax11 = fig.add_subplot(2, 2, 4, projection='polar')
    axes = [[ax00, ax01], [ax10, ax11]]

    # --- 9a: Energiebilanz-Kreisprozess ---
    ax = axes[0][0]
    phasen = ['Start\n(oben)', 'Fall\n(Arbeit E_mech)', 'Unten\n(Wende)', 'Rückhub\n(Energie nötig)']
    energie = [100, 0, 0, -100]
    delta_E = [0, -100, 0, +100]
    phasen_x = [0, 1, 2, 3]
    kumulativ = np.cumsum([100, -100, 0, 100]) - 100 + 100
    ax.bar(phasen, [100, 100, 0, 100], color=['steelblue','green','gray','tomato'],
           edgecolor='black', alpha=0.8)
    ax.set_ylabel('Energie [J]')
    ax.set_title('Kreisprozess: Energiebilanz pro Phase\n(Rückhub kostet genau so viel wie Hinfall bringt)')

    # --- 9b: Thermodynamische Grenzen zusammengefasst ---
    ax = axes[0][1]
    punkte = {r'$\oint \vec{F}_{grav} \cdot d\vec{s} = 0$': (0.5, 0.8),
              r'$\Delta E_{Zyklus} = 0$ (konservativ)': (0.5, 0.65),
              r'$\eta < 1$ (2. HS Thermo.)': (0.5, 0.50),
              r'Reibung: $E_{Nutz} < E_{pot}$': (0.5, 0.35),
              r'Kein PM II. Art möglich': (0.5, 0.20),
              r'Gravitation $\ll$ EM ($G$ vs $k_e$)': (0.5, 0.05)}
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.0)
    ax.set_axis_off()
    ax.set_title('Physikalische Unmöglichkeitsgründe', fontsize=12)
    for i, (text, (x, y)) in enumerate(punkte.items()):
        ax.annotate('✗ ' + text, (x, y), ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', fc='#ffcccc', ec='red', pad=0.4))

    # --- 9c: Skalierungsgesetz Masse vs. Leistung ---
    ax = axes[1][0]
    m_log = np.logspace(0, 6, 500)
    h_10  = m_log * g * 10   # J / Zyklus bei 10 m Fall
    P_1Hz = h_10 * 1          # W bei 1 Hz Zyklus
    P_point1Hz = h_10 * 0.1
    ax.loglog(m_log, P_1Hz,      'b-',  lw=2, label='f = 1 Hz')
    ax.loglog(m_log, P_point1Hz, 'g--', lw=2, label='f = 0.1 Hz')
    ax.axhline(50000, color='red', ls=':', lw=2, label='50 kW (E-Motor Ref.)')
    ax.set_xlabel('Masse [kg]')
    ax.set_ylabel('Leistung [W]')
    ax.set_title('Nötige Masse für gegebene Leistung')
    ax.legend()

    # --- 9d: Radar-Chart Vergleich Antriebe ---
    ax = axes[1][1]
    kategorien_r = ['Leistungs-\ndichte', 'Energie-\ndichte', 'Wirkungsgrad',
                    'Kosten', 'Zuverlässigkeit', 'Skalierbarkeit']
    N_kat = len(kategorien_r)
    winkel = np.linspace(0, 2*np.pi, N_kat, endpoint=False).tolist()
    winkel += winkel[:1]
    # Werte 0-10
    elektromotor = [9, 8, 9, 7, 9, 9]
    gravitationsmotor = [1, 1, 3, 5, 4, 2]
    hydraulik = [6, 4, 7, 6, 7, 7]

    for daten, label, col in [(elektromotor, 'Elektromotor', 'blue'),
                               (gravitationsmotor, 'Gravitationsmotor', 'red'),
                               (hydraulik, 'Hydraulikmotor', 'green')]:
        werte = daten + [daten[0]]
        ax.plot(winkel, werte, 'o-', lw=2, color=col, label=label)
        ax.fill(winkel, werte, alpha=0.15, color=col)

    ax.set_xticks(winkel[:-1])
    ax.set_xticklabels(kategorien_r, size=9)
    ax.set_ylim(0, 10)
    ax.set_title('Radar-Vergleich Antriebstechnologien')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Koordinatenachsen für Radar
    ax.set_rticks([2, 4, 6, 8, 10])
    ax.set_rlabel_position(45)

    fig.suptitle('Abbildung 9 – Zusammenfassung und Schlussfolgerung',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot9_zusammenfassung.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot9_zusammenfassung.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 9 gespeichert.")


# ==============================================================================
# PLOT 10 – Synergiekonzepte I & II: Drehmomentrippel, Nd-Einsparung, Radar
# ==============================================================================
def plot10_synergie_uebersicht():
    """Gravitativer Pendelpuffer im PMSM: Drehmomentrippel & Nd-Einsparung."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    theta = np.linspace(0, 2*np.pi, 1000)
    tau0, tau1 = 100.0, 40.0   # Nm: mittleres Moment und Wechselanteil

    # --- 10a: Drehmomentkurven ---
    ax = axes[0, 0]
    tau_last = tau0 + tau1 * np.sin(theta)
    tau_grav = tau1 * np.sin(theta)          # optimaler Pendelpuffer
    tau_em   = tau_last - tau_grav
    ax.plot(np.degrees(theta), tau_last, 'b-',  lw=2.5, label=r'$\tau_{\rm Last}(\theta)$')
    ax.plot(np.degrees(theta), tau_grav, 'r--', lw=2.0, label=r'$\tau_{\rm grav}(\theta)$')
    ax.plot(np.degrees(theta), tau_em,   'g-',  lw=2.5, label=r'$\tau_{\rm em,red}(\theta)$')
    ax.axhline(tau0, color='gray', ls=':', lw=1.5, label=r'$\tau_0$ (Mittelwert)')
    ax.fill_between(np.degrees(theta), tau_em, tau0,
                    where=(tau_em > tau0), alpha=0.15, color='green')
    ax.fill_between(np.degrees(theta), tau_em, tau0,
                    where=(tau_em < tau0), alpha=0.15, color='green')
    ax.set_xlabel(r'Rotorwinkel $\theta$ [°]')
    ax.set_ylabel('Drehmoment [Nm]')
    ax.set_title('Gravitativer Pendelpuffer:\nDrehmomentrippel-Reduktion')
    ax.legend(fontsize=8)

    # --- 10b: Drehmomentrippel vs. Pufferanteil ---
    ax = axes[0, 1]
    kappa = np.linspace(0, 1.5, 400)   # m_p*g*l / tau1
    ripple_rel = np.abs(1 - kappa)      # normierter Rippel
    ax.plot(kappa, ripple_rel * 100, 'b-', lw=2.5, label='Norm. Drehmomentrippel')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Optimum $\\kappa=1$')
    ax.fill_between(kappa, ripple_rel*100, alpha=0.2, color='blue')
    ax.set_xlabel(r'Pufferanteil $\kappa = m_p g l\,/\,\tau_1$')
    ax.set_ylabel('Relativer Drehmomentrippel [%]')
    ax.set_title('Optimum bei $\\kappa = 1$\n(vollständige Kompensation)')
    ax.legend()
    ax.set_ylim(0, 105)

    # --- 10c: Spitzenstrom und Kupferverluste ---
    ax = axes[0, 2]
    delta_I = np.linspace(0, 0.6, 400)   # relative Stromreduktion
    P_cu_rel = (1 - delta_I)**2
    I_max_rel = 1 - delta_I
    ax.plot(delta_I*100, I_max_rel*100, 'darkorange', lw=2.5,
            label=r'$I_{\max}/I_{\max,0}$')
    ax.plot(delta_I*100, P_cu_rel*100,  'crimson',    lw=2.5,
            label=r'$P_{\rm Cu}/P_{\rm Cu,0}$')
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xlabel('Relative Stromreduktion $\\delta_I$ [%]')
    ax.set_ylabel('Relativer Wert [%]')
    ax.set_title('Spitzenstrom und Kupferverluste\nbei Drehmomentrippel-Senkung')
    ax.legend()
    ax.set_xlim(0, 60)
    ax.set_ylim(40, 105)

    # --- 10d: Benötigtes NdFeB-Volumen vs. Lastmomentschwankung ---
    ax = axes[1, 0]
    gamma_arr = np.linspace(0.0, 0.8, 400)   # tau1/tau0
    # Konventionell: dimensioniert auf tau_max = tau0*(1+gamma)
    V_nd_konv = (1 + gamma_arr)**2
    # Hybridmotor: nur tau0 bestimmend (Konz. II, Satz thm:hybridmotor)
    V_nd_hybrid = np.ones_like(gamma_arr)
    ax.plot(gamma_arr, V_nd_konv,   'b-',  lw=2.5, label='Konventionell')
    ax.plot(gamma_arr, V_nd_hybrid, 'g--', lw=2.5, label='Hybridmotor (Konzept II)')
    ax.fill_between(gamma_arr, V_nd_hybrid, V_nd_konv, alpha=0.2, color='green',
                    label='Nd-Einsparung')
    ax.set_xlabel(r'Lastmoment-Wechselanteil $\tau_1 / \tau_0$')
    ax.set_ylabel('Norm. NdFeB-Volumen [a.u.]')
    ax.set_title('NdFeB-Einsparung durch Hybridmotor')
    ax.legend()

    # --- 10e: Wirkungsgradkarte im Betriebsfeld ---
    ax = axes[1, 1]
    omega_arr2 = np.linspace(10, 200, 300)   # rad/s
    tau_arr2   = np.linspace(10, 150, 300)
    OO, TT = np.meshgrid(omega_arr2, tau_arr2)
    P_out = OO * TT
    # Verluste: Cu + Fe + mech (vereinfacht)
    Rs, Ls, p_pole, Psi_PM = 0.08, 5e-3, 4, 0.2
    I_q = TT / (1.5 * p_pole * Psi_PM)
    P_cu = 1.5 * Rs * I_q**2
    P_fe = 2e-4 * (OO * p_pole)**1.5
    P_mech = 1e-5 * OO**2
    P_tot = P_cu + P_fe + P_mech
    eta_map = P_out / (P_out + P_tot)
    cs = ax.contourf(OO, TT, eta_map*100, levels=np.linspace(85, 98, 20), cmap='RdYlGn')
    ax.contour(OO, TT, eta_map*100, levels=[95], colors='white', linewidths=2, linestyles='--')
    plt.colorbar(cs, ax=ax, label='η [%]')
    ax.set_xlabel(r'Winkelgeschwindigkeit $\omega$ [rad/s]')
    ax.set_ylabel(r'Drehmoment $\tau$ [Nm]')
    ax.set_title('Wirkungsgradkarte PMSM\n(weiß: η = 95 %)')

    # --- 10f: Radar-Vergleich ---
    ax = axes[1, 2]
    ax.set_aspect('equal')
    kategorien = ['Nd-Bedarf\n(invers)', 'Wirkungsgrad', 'Drehmomentrippel\n(invers)',
                  'Gewicht\n(invers)', 'Kosten\n(invers)', 'Zuverlässigkeit']
    N_k = len(kategorien)
    angles = np.linspace(0, 2*np.pi, N_k, endpoint=False).tolist()
    angles += angles[:1]
    konv    = [5, 9, 4, 7, 6, 9]
    hybrid  = [8, 9, 8, 7, 5, 8]
    ferrit  = [10, 7, 6, 9, 9, 8]
    ax_r = fig.add_subplot(2, 3, 6, projection='polar')
    for daten, label, col in [(konv,   'Konv. PMSM',      'blue'),
                               (hybrid, 'Hybridmotor',     'green'),
                               (ferrit, 'Ferritmotor',     'orange')]:
        werte = daten + [daten[0]]
        ax_r.plot(angles, werte, 'o-', lw=2, color=col, label=label)
        ax_r.fill(angles, werte, alpha=0.10, color=col)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(kategorien, size=8)
    ax_r.set_ylim(0, 10)
    ax_r.set_rticks([2, 4, 6, 8, 10])
    ax_r.set_rlabel_position(30)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)
    ax_r.set_title('Vergleich Motorkonzepte\n(höher = besser)', fontsize=9, pad=15)
    axes[1, 2].set_visible(False)   # wird durch polar-Achse ersetzt

    fig.suptitle('Abbildung 10 – Synergiekonzepte I & II: Gravitation + Elektromotor',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot10_synergie_uebersicht.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot10_synergie_uebersicht.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 10 gespeichert.")


# ==============================================================================
# PLOT 11 – FOC mit gravitativer Entlastung: dq-Ebene, Verluste, Thermik, η-Kennlinie
# ==============================================================================
def plot11_foc_gravitation():
    """Feldorientierte Regelung mit gravitativem Puffer: dq-Strom, Verluste, Thermik."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Parameter PMSM (10 kW)
    Rs   = 0.08     # Ohm
    Ls   = 5e-3     # H
    p    = 4        # Polpaare
    Psi  = 0.2      # Wb
    I_N  = 50.0     # A Nennstrom
    w_N  = 2*np.pi*25  # rad/s elektrisch (~150 rpm mech bei p=4 => 600 rpm mech, Beispiel)

    # Nennmoment und Wechselanteil
    tau0, tau1 = 100.0, 40.0

    # --- 11a: dq-Ebene ---
    ax = axes[0, 0]
    # Stromgrenzen
    I_max = I_N
    phi_i = np.linspace(0, 2*np.pi, 500)
    ax.plot(I_max*np.cos(phi_i), I_max*np.sin(phi_i),
            'k-', lw=1.5, label=f'Stromgrenze $I_N={I_N:.0f}$ A', zorder=0)

    # MTPA-Linie für SPMSM (Ld=Lq): iq-Achse
    iq_range = np.linspace(0, I_N, 300)
    ax.plot(np.zeros_like(iq_range), iq_range, 'g--', lw=1.5, label='MTPA-Pfad (SPMSM)')

    # Betriebspunkte konventionell (ohne Puffer)
    iq_konv = np.array([tau0/(1.5*p*Psi), (tau0+tau1)/(1.5*p*Psi)])
    ax.scatter([0, 0], iq_konv, s=100, color='blue', zorder=5, label='Konv. (min/max)')
    ax.annotate(f'$i_q={iq_konv[0]:.1f}$ A', (0, iq_konv[0]),
                xytext=(5, -8), textcoords='offset points', fontsize=8, color='blue')
    ax.annotate(f'$i_q={iq_konv[1]:.1f}$ A', (0, iq_konv[1]),
                xytext=(5, 4), textcoords='offset points', fontsize=8, color='blue')

    # Betriebspunkte mit Puffer (nur tau0 nötig)
    iq_hybrid = tau0/(1.5*p*Psi)
    ax.scatter([0], [iq_hybrid], s=150, color='red', marker='*', zorder=6,
               label=f'Hybridmotor $i_q={iq_hybrid:.1f}$ A (const.)')
    ax.annotate(f'Nd-Einsparung\n$\\approx{100*(1-(tau0/(tau0+tau1))**2):.0f}\\%$',
                (3, iq_hybrid), fontsize=9, color='darkgreen',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='green', pad=0.3))

    ax.set_xlabel('$i_d$ [A]')
    ax.set_ylabel('$i_q$ [A]')
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 65)
    ax.set_title('dq-Betriebsraum:\nKonv. vs. Hybridmotor (SPMSM)')
    ax.legend(fontsize=8)

    # --- 11b: Zeitlicher Strom-Verlauf ---
    ax = axes[0, 1]
    t_sim = np.linspace(0, 0.2, 2000)   # 200 ms
    omega_mech = 2*np.pi*25/p            # rad/s mechanisch
    theta_t = omega_mech * p * t_sim     # elektrischer Winkel
    tau_last_t = tau0 + tau1 * np.sin(theta_t)
    # Konventionell: iq folgt dem Lastmoment
    iq_konv_t = tau_last_t / (1.5 * p * Psi)
    # Hybrid: iq konstant
    iq_hybrid_t = np.full_like(t_sim, tau0 / (1.5 * p * Psi))
    ax.plot(t_sim*1000, iq_konv_t,  'b-',  lw=2, label=r'$i_q(t)$ konventionell')
    ax.plot(t_sim*1000, iq_hybrid_t,'r--', lw=2, label=r'$i_q(t)$ Hybridmotor')
    ax.fill_between(t_sim*1000, iq_hybrid_t, iq_konv_t, alpha=0.2, color='orange',
                    label='Stromreduktion')
    ax.set_xlabel('Zeit [ms]')
    ax.set_ylabel('$i_q$ [A]')
    ax.set_title('Längskomponente des Stroms $i_q(t)$\nbei sinusoidalem Lastmoment')
    ax.legend(fontsize=8)

    # --- 11c: Kupferverluste ---
    ax = axes[0, 2]
    P_cu_konv   = 1.5 * Rs * iq_konv_t**2
    P_cu_hybrid = 1.5 * Rs * iq_hybrid_t**2
    ax.plot(t_sim*1000, P_cu_konv,   'b-',  lw=2, label='$P_{Cu}$ konventionell')
    ax.plot(t_sim*1000, P_cu_hybrid, 'r--', lw=2, label='$P_{Cu}$ Hybridmotor')
    ax.fill_between(t_sim*1000, P_cu_hybrid, P_cu_konv, alpha=0.25, color='tomato',
                    label='Verlustreduzierung')
    ax.set_xlabel('Zeit [ms]')
    ax.set_ylabel('Statorverluste $P_{Cu}$ [W]')
    ax.set_title('Kupferverluste:\nEinsparung durch grav. Entlastung')
    ax.legend(fontsize=8)
    # Mittlere Werte
    delta_pcu = (np.mean(P_cu_konv) - np.mean(P_cu_hybrid)) / np.mean(P_cu_konv) * 100
    ax.text(0.05, 0.05, f'Ø Reduktion: {delta_pcu:.1f} %',
            transform=ax.transAxes, fontsize=10, color='darkred',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='red'))

    # --- 11d: Thermische Simulation ---
    ax = axes[1, 0]
    # Vereinfachtes thermisches Modell: R_th * P_loss = T_steady_state
    R_th   = 3.0     # K/W thermischer Widerstand
    C_th   = 500.0   # J/K thermische Kapazität
    T_amb  = 25.0    # °C Umgebung
    t_therm = np.linspace(0, 3600, 3600)   # 1 Stunde
    # Stationäre Verluste
    P_full   = np.mean(P_cu_konv) + 50     # W Gesamtverlust (konv.)
    P_hybrid_full = np.mean(P_cu_hybrid) + 50
    T_konv   = T_amb + R_th * P_full   * (1 - np.exp(-t_therm / (R_th * C_th)))
    T_hybrid = T_amb + R_th * P_hybrid_full * (1 - np.exp(-t_therm / (R_th * C_th)))
    ax.plot(t_therm/60, T_konv,   'b-',  lw=2, label='Konventionell')
    ax.plot(t_therm/60, T_hybrid, 'r--', lw=2, label='Hybridmotor')
    ax.fill_between(t_therm/60, T_hybrid, T_konv, alpha=0.2, color='blue',
                    label=r'$\Delta T \approx ' + f'{R_th*(P_full-P_hybrid_full):.1f}$ K')
    ax.set_xlabel('Zeit [min]')
    ax.set_ylabel('Statortemperatur [°C]')
    ax.set_title('Thermisches Modell:\nTemperaturentlastung durch grav. Puffer')
    ax.legend(fontsize=8)

    # --- 11e: η-Kennlinie über Lastbereich ---
    ax = axes[1, 1]
    tau_rel = np.linspace(0.05, 1.3, 400)   # tau/tau_N
    tau_abs = tau_rel * tau0
    kappa_vals = [0.0, 0.2, 0.4, 0.6, 1.0]
    colors_kap = ['blue', 'cyan', 'green', 'orange', 'red']
    for kap, col in zip(kappa_vals, colors_kap):
        tau_em_red = tau_abs * (1 - kap * (tau1/tau0))
        tau_em_red = np.maximum(tau_em_red, tau_abs * 0.1)
        iq_val     = tau_em_red / (1.5 * p * Psi)
        P_cu_val   = 1.5 * Rs * iq_val**2
        P_out_val  = tau_abs * w_N / p
        eta_val    = P_out_val / (P_out_val + P_cu_val + 50)
        ax.plot(tau_rel*100, eta_val*100, color=col, lw=2,
                label=f'$\\kappa={kap:.1f}$')
    ax.set_xlabel(r'Relativer Drehmomentwert $\tau\,/\,\tau_N$ [%]')
    ax.set_ylabel('Wirkungsgrad η [%]')
    ax.set_title('η-Kennlinie für verschiedene\ngravitative Entlastungsgrade κ')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 130)

    # --- 11f: Jahresenergiegewinn (Weibull-Lastprofil) ---
    ax = axes[1, 2]
    # Weibull-Verteilung für Windgeschwindigkeit → Lastprofil
    v_wind = np.linspace(0, 25, 300)  # m/s
    c_weib, k_weib = 7.0, 2.0
    pdf_weib = (k_weib/c_weib) * (v_wind/c_weib)**(k_weib-1) * np.exp(-(v_wind/c_weib)**k_weib)
    # Last proportional v^3, begrenzt auf Nennlast
    tau_wind = np.minimum((v_wind/10)**3, 1.0) * tau0
    kappa_arr = np.linspace(0, 1.0, 100)
    energy_gain_arr = []
    for kap in kappa_arr:
        tau_red = tau_wind * (1 - kap * tau1/tau0)
        tau_red = np.maximum(tau_red, tau_wind * 0.05)
        iq_r    = tau_red   / (1.5 * p * Psi)
        iq_c    = tau_wind  / (1.5 * p * Psi)
        dP_cu   = 1.5 * Rs * (iq_c**2 - iq_r**2)
        gain    = np.trapezoid(dP_cu * pdf_weib, v_wind) * 8760 / 1000  # kWh/year
        energy_gain_arr.append(gain)
    ax.plot(kappa_arr, energy_gain_arr, 'darkgreen', lw=2.5, label='Jährl. Gewinn')
    ax.axhline(energy_gain_arr[-1]*0.5, color='gray', ls='--', lw=1.5,
               label='Wirtschaftlichkeitsschwelle (50 %)')
    ax.fill_between(kappa_arr, energy_gain_arr, alpha=0.2, color='green')
    ax.set_xlabel(r'Grav. Entlastungsgrad $\kappa$')
    ax.set_ylabel('Jährl. Energiegewinn [kWh / (MW · Jahr)]')
    ax.set_title('Jahresenergiegewinn\nbei Weibull-Windprofil ($c=7$, $k=2$)')
    ax.legend(fontsize=8)

    fig.suptitle('Abbildung 11 – FOC mit gravitativer Entlastung: dq-Ebene und Wirkungsgradgewinn',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot11_foc_gravitation.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot11_foc_gravitation.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 11 gespeichert.")


# ==============================================================================
# PLOT 12 – Gravitativer TMD + Materialvergleich: Demagnetisierung, GBE, Wertschöpfung
# ==============================================================================
def plot12_tmd_und_material():
    """Gravitativer Schwingungstilger (TMD) und Magnetmaterial-Vergleich."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- 12a: Übertragungsfunktion mit/ohne TMD ---
    ax = axes[0, 0]
    freqs_norm = np.linspace(0.01, 2.5, 2000)  # normiert auf omega0
    mu_tmd = 0.05    # Masseverhältnis
    # Den Hartog optimale Abstimmung
    f_opt     = 1.0 / (1.0 + mu_tmd)
    zeta_opt  = np.sqrt(3 * mu_tmd / (8 * (1 + mu_tmd)**3))

    # Übertragungsfunktion: ohne TMD (SDOF)
    zeta_str = 0.01
    H_ohne = 1.0 / np.sqrt((1 - freqs_norm**2)**2 + (2*zeta_str*freqs_norm)**2)

    # Mit TMD (vereinfachte 2-DOF-Formel, nummerisch)
    H_mit = np.zeros_like(freqs_norm)
    for i, omega_r in enumerate(freqs_norm):
        # Numerische Berechnung der 2-DOF-Übertragungsfunktion
        num = (f_opt**2 - omega_r**2)**2 + (2*zeta_opt*f_opt*omega_r)**2
        den_r = ((1-omega_r**2)*(f_opt**2-omega_r**2) - mu_tmd*f_opt**2*omega_r**2)**2
        den_i = (2*zeta_opt*f_opt*omega_r*(1-omega_r**2+mu_tmd*omega_r**2))**2
        H_mit[i] = np.sqrt(num / (den_r + den_i)) if (den_r + den_i) > 0 else 0

    ax.semilogy(freqs_norm, H_ohne,  'b-',  lw=2, label='Ohne TMD')
    ax.semilogy(freqs_norm, H_mit,   'r--', lw=2, label=f'Mit TMD ($\\mu={mu_tmd}$)')
    ax.axvline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel(r'Normierte Frequenz $\Omega\,/\,\omega_0$')
    ax.set_ylabel(r'$|H(\Omega)|$')
    ax.set_title(f'Übertragungsfunktion:\nLuftspaltmodulation mit/ohne TMD')
    ax.legend()

    # --- 12b: Optimale TMD-Länge vs. Erregerfrequenz ---
    ax = axes[0, 1]
    f_err = np.logspace(-2, 2, 500)   # Hz
    l_tmd = g / (2*np.pi*f_err)**2
    ax.loglog(f_err, l_tmd, 'purple', lw=2.5)
    # Markierung realisierbarer Bereich
    ax.axhspan(0.01, 100, alpha=0.1, color='green', label='Praktikabel (0.01–100 m)')
    ax.axhline(0.01, color='green', ls='--', lw=1)
    ax.axhline(100,  color='green', ls='--', lw=1)
    # Typische Windturbinenfrequenzen
    for f_mark, name in [(0.3, '0.3 Hz\n(Blattdurchgang)'),
                          (1.0, '1 Hz'),
                          (10,  '10 Hz\n(Getriebe)')]:
        l_mark = g / (2*np.pi*f_mark)**2
        ax.scatter(f_mark, l_mark, s=80, color='red', zorder=5)
        ax.annotate(name, (f_mark, l_mark), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel('Erregerfrequenz $f$ [Hz]')
    ax.set_ylabel('Benötigte Pendellänge $l_{TMD}$ [m]')
    ax.set_title('Optimale Pendellänge des\ngravit. TMD vs. Erregerfrequenz')
    ax.legend(fontsize=8)

    # --- 12c: Wirkungsgradgewinn durch TMD vs. mu ---
    ax = axes[0, 2]
    mu_arr = np.linspace(0.005, 0.20, 300)
    # Amplitudenreduktion nach Den Hartog-Näherung
    ampl_red = 1.0 / (1 + np.sqrt(4/mu_arr + 1))   # normiert
    # Zusätzliche Eisenverluste ~ (Delta_delta/delta0)^2
    # Relative Reduktion:
    fe_gain = 1 - ampl_red**2  # rel. Eisenverlust-Reduktion
    # Wirkungsgradgewinn: P_Fe ~ 3-5% der Gesamtverluste
    for p_fe_frac, col, label in [(0.03, 'blue',  '$P_{Fe,0} = 3\\%$'),
                                   (0.05, 'green', '$P_{Fe,0} = 5\\%$'),
                                   (0.08, 'red',   '$P_{Fe,0} = 8\\%$')]:
        eta_gain = fe_gain * p_fe_frac * 100
        ax.plot(mu_arr*100, eta_gain, color=col, lw=2, label=label)
    ax.axvline(5,  color='gray', ls='--', lw=1, label='Opt. μ ≈ 5 %')
    ax.set_xlabel(r'Masseverhältnis $\mu = m_{TMD}/m_{Rotor}$ [%]')
    ax.set_ylabel('Wirkungsgradgewinn Δη [%]')
    ax.set_title('Wirkungsgradgewinn durch\ngravit. TMD vs. Masseverhältnis')
    ax.legend(fontsize=8)

    # --- 12d: B-H-Kurven / Demagnetisierungsdiagramme ---
    ax = axes[1, 0]
    # Zweiter Quadrant: B = mu0*(H + M), vereinfachte Gerade für jeden Magneten
    H_arr = np.linspace(-2000, 0, 500)   # kA/m
    magnete = {
        'NdFeB N52':      (1.45, -1800),   # (Br [T], Hc [kA/m])
        'SmCo 2:17':      (1.08, -800),
        'Ferrit':         (0.40, -260),
        'L1₀-FeNi (proj)':(0.75, -350),
    }
    colors_m = ['royalblue', 'darkorange', 'tomato', 'seagreen']
    for (name, (Br, Hc)), col in zip(magnete.items(), colors_m):
        mu_r_eff = Br / (-Hc * 1e3 * 4*np.pi*1e-7)
        B_vals = Br + mu_r_eff * 4*np.pi*1e-7 * H_arr * 1e3
        B_vals = np.where(B_vals > 0, B_vals, 0)
        ax.plot(H_arr, B_vals, color=col, lw=2, label=name)
        BH = B_vals * H_arr * 1e3
        idx = np.argmin(BH)
        ax.scatter(H_arr[idx], B_vals[idx], s=80, color=col, zorder=5)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel(r'$H$ [kA/m]')
    ax.set_ylabel(r'$B$ [T]')
    ax.set_title('Demagnetisierungsdiagramme\n(2. Quadrant / B-H-Kurven)')
    ax.legend(fontsize=8)
    ax.set_xlim(-2000, 100)

    # --- 12e: (BH)max vs. Nd-Gehalt (incl. GBE) ---
    ax = axes[1, 1]
    nd_konz = np.linspace(10, 35, 300)   # Mol-% Nd in NdFeB
    # Typisch: (BH)max steigt bis ~26 Mol-% Nd, dann wieder fallend
    BH_max = 450 * np.exp(-0.5*((nd_konz-26)/5)**2)   # kJ/m^3
    # Mit GBE: ähnliche BH_max bei weniger Nd (Kurve um ca -15% Nd verschoben)
    nd_GBE = nd_konz - 4  # GBE spart ~4 Mol-pp Nd
    BH_GBE = 450 * np.exp(-0.5*((nd_GBE-22)/5)**2)
    ax.plot(nd_konz, BH_max, 'b-',  lw=2.5, label='Standard NdFeB')
    ax.plot(nd_konz, BH_GBE, 'g--', lw=2.5, label='GBE (Grain-Boundary-Diff.)')
    ax.fill_between(nd_konz, BH_GBE, BH_max, where=(BH_GBE < BH_max),
                    alpha=0.2, color='green', label='Nd-Einsparung durch GBE')
    # Zeige Einsparung bei BH=350 kJ/m^3
    target_BH = 350
    nd_std_at_target = nd_konz[np.argmin(np.abs(BH_max - target_BH))]
    nd_gbe_at_target = nd_konz[np.argmin(np.abs(BH_GBE - target_BH))]
    ax.axhline(target_BH, color='red', ls=':', lw=1.5)
    ax.annotate('', xy=(nd_gbe_at_target, target_BH+5),
                xytext=(nd_std_at_target, target_BH+5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((nd_std_at_target+nd_gbe_at_target)/2, target_BH+12,
            f'${nd_std_at_target-nd_gbe_at_target:.1f}$ Mol-pp\nEinsparung',
            ha='center', fontsize=9, color='darkred')
    ax.set_xlabel('Nd-Gehalt [Mol-%]')
    ax.set_ylabel(r'$(BH)_{\max}$ [kJ/m³]')
    ax.set_title('Energieprodukt vs. Nd-Gehalt:\nGBE ermöglicht Einsparung')
    ax.legend(fontsize=8)

    # --- 12f: Wertschöpfungskette (vereinfachtes Schema) ---
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Wertschöpfungskette:\nNeodym → Motor (Interventionstellen)')
    schritte = [
        (5, 9.0, 'Neodym-Mine\n(CN, AU, USA)', '#f9d5e5', '1'),
        (5, 7.2, 'Separation /\nRaffination', '#ffe0b2', '2'),
        (5, 5.4, 'Legierung\nNdFeB (GBE)', '#fff9c4', '3'),
        (5, 3.6, 'Magnet-\nFertigung', '#dcedc8', '4'),
        (5, 1.8, 'Motor-\nSystem (FOC,TMD)', '#b3e5fc', '5'),
    ]
    synergien = {
        3: 'Konzept: GBE\n(−20% Nd)',
        4: 'Konzept I+II\n(−40−60% Nd)',
        5: 'Konzept III+IV\n(+Wirkungsgrad)',
    }
    for (x, y, label, col, num) in schritte:
        rect = mpatches.FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2,
                                        boxstyle='round,pad=0.1', fc=col,
                                        ec='gray', lw=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, zorder=3)
        if y > 1.8:
            ax.annotate('', xy=(x, y-0.7), xytext=(x, y-0.6-0.5),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        i = int(num)
        if i in synergien:
            ax.text(8.5, y, synergien[i], ha='center', va='center',
                    fontsize=8, color='darkgreen',
                    bbox=dict(boxstyle='round', fc='#e8f5e9', ec='green', pad=0.3))
            ax.annotate('', xy=(x+1.8, y), xytext=(7.4, y),
                        arrowprops=dict(arrowstyle='<-', lw=1.2, color='green',
                                        linestyle='dashed'))

    fig.suptitle('Abbildung 12 – Gravitativer TMD und Magnetmaterial-Vergleich',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot12_tmd_und_material.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot12_tmd_und_material.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 12 gespeichert.")


# ==============================================================================
# PLOT 13 – Gesamtbilanz: Nd-Einsparung, Wirkungsgradgewinn, CO2, Wirtschaftlichkeit
# ==============================================================================
def plot13_gesamtbilanz():
    """Gesamtbilanz: Neodym-Einsparung, Wirkungsgradgewinn, Wirtschaftlichkeit, CO2."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- 13a: Tortendiagramm NdFeB-Verbrauch nach Sektor ---
    ax = axes[0, 0]
    sektoren = ['Windkraft\n35 %', 'E-Mobilität\n30 %',
                'Industrie-\nmotoren 20 %', 'Consumer\nElectronics 10 %',
                'Sonstiges\n5 %']
    anteile = [35, 30, 20, 10, 5]
    farben_p = ['#4472C4', '#ED7D31', '#A9D18E', '#FF6B6B', '#C9C9C9']
    wedges, texts, autotexts = ax.pie(anteile, labels=sektoren, autopct='%1.0f%%',
                                       colors=farben_p, startangle=90,
                                       textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax.set_title('NdFeB-Verbrauch nach Sektor\n(global, ca. 2024)')

    # --- 13b: Kumulierte Nd-Einsparung bis 2040 (Szenarien) ---
    ax = axes[0, 1]
    jahre = np.arange(2025, 2041)
    # Annahmen: jährlicher Nd-Verbrauch wächst, Synergiekonzepte schrittweise eingeführt
    nd_basis_2025 = 80   # kt/Jahr (Basis-Nachfrage)
    wachstum       = 0.08  # 8%/Jahr ohne Gegenmassnahmen
    # Einführungsgrad der Synergiekonzepte: logistisch
    def logistic(t, t0, k):
        return 1 / (1 + np.exp(-k*(t - t0)))
    einfuehrung_niedrig  = logistic(jahre, 2035, 0.5)
    einfuehrung_moderat  = logistic(jahre, 2031, 0.8)
    einfuehrung_hoch     = logistic(jahre, 2028, 1.2)
    nd_ohne = nd_basis_2025 * (1 + wachstum)**(jahre - 2025)
    einspar_faktor = {'Niedrig': 0.20, 'Moderat': 0.40, 'Hoch': 0.55}
    einfuehrungen   = {'Niedrig': einfuehrung_niedrig,
                       'Moderat': einfuehrung_moderat,
                       'Hoch':    einfuehrung_hoch}
    farben_sz = {'Niedrig': 'lightblue', 'Moderat': 'steelblue', 'Hoch': 'navy'}
    nd_save_kum = {}
    ax.plot(jahre, nd_ohne/1000, 'k--', lw=2, label='Ohne Synergien (Referenz)')
    for sz in ['Niedrig', 'Moderat', 'Hoch']:
        nd_mit = nd_ohne * (1 - einspar_faktor[sz] * einfuehrungen[sz])
        nd_save = nd_ohne - nd_mit
        nd_save_kum[sz] = np.cumsum(nd_save)
        ax.plot(jahre, nd_mit/1000, lw=2, color=farben_sz[sz],
                label=f'{sz}: {einspar_faktor[sz]*100:.0f}% Nd-Reduktion')
        ax.fill_between(jahre, nd_mit/1000, nd_ohne/1000, alpha=0.15,
                        color=farben_sz[sz])
    ax.set_xlabel('Jahr')
    ax.set_ylabel('Nd-Jahresbedarf [kt/Jahr]')
    ax.set_title('Prognostizierter Nd-Jahresbedarf\nmit/ohne Synergiekonzepte')
    ax.legend(fontsize=8)

    # --- 13c: Kosteneinsparung vs. Nd-Preis ---
    ax = axes[0, 2]
    nd_preis = np.linspace(50, 500, 300)   # EUR/kg NdFeB-Magnet
    # Typischer Nd-Magnet-Anteil 3-5 % Motorgewicht; ~1 kg Nd/kW (grob)
    nd_kg_per_kW = 0.8   # kg NdFeB pro kW Motorleistung
    nd_einspar_frac = np.array([0.20, 0.40, 0.55])
    eta_gewinn_kWh  = 0.015  # 1.5% durchschnittl. Wirkungsgradgewinn
    E_annual_kWh    = 2000   # kWh/(kW·Jahr) Volllaststunden
    strompreis      = 0.07   # EUR/kWh
    labels_nd = ['20% Nd-Einspar.', '40% Nd-Einspar.', '55% Nd-Einspar.']
    colors_nd = ['lightblue', 'steelblue', 'navy']
    for frac, lab, col in zip(nd_einspar_frac, labels_nd, colors_nd):
        cost_save = nd_kg_per_kW * nd_preis * frac + eta_gewinn_kWh * E_annual_kWh * strompreis
        ax.plot(nd_preis, cost_save, lw=2.5, color=col, label=lab)
    ax.set_xlabel('NdFeB-Magnetpreis [EUR/kg]')
    ax.set_ylabel('Kosteneinsparung [EUR/(kW·Jahr)]')
    ax.set_title('Wirtschaftlichkeit:\nKosteneinsparung vs. Nd-Preis')
    ax.legend(fontsize=8)

    # --- 13d: Vollständige Wirkungsgradkennlinie ---
    ax = axes[1, 0]
    tau_rel = np.linspace(0.05, 1.25, 500)
    Rs_v, Psi_v, p_v = 0.08, 0.2, 4
    P_out_v = tau_rel * 100 * (2*np.pi*25/p_v)
    # Konventionell
    iq_konv_v  = tau_rel * 100 / (1.5 * p_v * Psi_v)
    P_cu_konv_v = 1.5 * Rs_v * iq_konv_v**2
    P_fe_v      = 50 * (tau_rel)**0.5
    eta_konv_v  = P_out_v / (P_out_v + P_cu_konv_v + P_fe_v + 20)
    # Hybrid (alle 4 Konzepte, κ=0.6)
    kap_total  = 0.60
    tau1_over_tau0 = 0.4
    iq_hybrid_v  = tau_rel * 100 * (1 - kap_total*tau1_over_tau0) / (1.5*p_v*Psi_v)
    iq_hybrid_v  = np.maximum(iq_hybrid_v, tau_rel * 100 * 0.2 / (1.5*p_v*Psi_v))
    P_cu_hyb_v   = 1.5 * Rs_v * iq_hybrid_v**2
    P_fe_hyb_v   = P_fe_v * 0.95   # TMD: -5% Eisenverluste
    eta_hybrid_v = P_out_v / (P_out_v + P_cu_hyb_v + P_fe_hyb_v + 20)
    ax.plot(tau_rel*100, eta_konv_v*100,  'b-',  lw=2.5, label='Konv. PMSM')
    ax.plot(tau_rel*100, eta_hybrid_v*100,'r-',  lw=2.5, label='Hybridmotor (alle 4 Konzepte)')
    ax.fill_between(tau_rel*100, eta_konv_v*100, eta_hybrid_v*100,
                    alpha=0.2, color='green', label='Wirkungsgradgewinn')
    ax.set_xlabel(r'Relatives Moment $\tau / \tau_N$ [%]')
    ax.set_ylabel('Wirkungsgrad η [%]')
    ax.set_title('Vollst. Wirkungsgradkennlinie:\nKonv. PMSM vs. Hybridmotor')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 125)
    ax.set_ylim(75, 100)

    # --- 13e: CO2-Äquivalenz-Einsparung ---
    ax = axes[1, 1]
    nd_einspar_kt = np.linspace(0, 200, 300)   # kt/Jahr
    # CO2 für Nd-Gewinnung: ~35 kg CO2/kg NdFeB (inkl. Schmelze, Raffinerie)
    co2_per_kg_nd = 35   # kg CO2/kg NdFeB
    co2_einspar_mining = nd_einspar_kt * 1e6 * co2_per_kg_nd / 1e9  # Mt CO2
    # CO2 durch Wirkungsgradgewinn (1.5% von typisch 2 GWh/MW/Jahr bei 30% Wind CF)
    P_MW = nd_einspar_kt * 50  # grob: 1 kt Nd ~ 50 MW Motorleistung
    E_saved_TWh = P_MW * 1e-6 * 2000 * 0.015  # TWh/Jahr
    co2_saved_energy = E_saved_TWh * 0.4   # Mt CO2 (0.4 kg CO2/kWh globaler Mix)
    ax.plot(nd_einspar_kt, co2_einspar_mining, 'b-',  lw=2.5, label='CO₂ aus Nd-Einsparung')
    ax.plot(nd_einspar_kt, co2_saved_energy,   'g--', lw=2.5, label='CO₂ aus Wirkungsgradgewinn')
    ax.plot(nd_einspar_kt, co2_einspar_mining + co2_saved_energy,
            'darkred', lw=2.5, label='CO₂ gesamt')
    ax.fill_between(nd_einspar_kt,
                    co2_einspar_mining + co2_saved_energy, alpha=0.15, color='red')
    ax.set_xlabel('Nd-Einsparung [kt/Jahr]')
    ax.set_ylabel('CO₂-Äquivalenz-Einsparung [Mt CO₂/Jahr]')
    ax.set_title('CO₂-Einsparungspotenzial\ndurch Synergiekonzepte')
    ax.legend(fontsize=8)

    # --- 13f: Radar-Vergleich aller Motorkonzepte ---
    ax = axes[1, 2]
    ax.set_visible(False)
    ax_r = fig.add_subplot(2, 3, 6, projection='polar')
    kategorien_r = ['Nd-Bedarf\n(inv)', 'Wirkungsgrad', 'Leistungs-\ndichte',
                    'Kosten\n(inv)', 'CO₂-\nFußabdr.(inv)', 'Zuver-\nlässigkeit']
    N_r = len(kategorien_r)
    angles_r = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
    angles_r += angles_r[:1]
    konzepte_radar = {
        'Konv. PMSM':      [5, 9, 8, 6, 5, 9],
        'Hybridmotor':     [8, 9, 7, 5, 8, 8],
        'Ferritmotor':     [10, 7, 5, 9, 9, 8],
        'SRM (no magnet)': [10, 6, 6, 8, 10, 7],
        'HTS-Motor':       [10, 10, 10, 3, 6, 6],
    }
    farben_rad = ['blue', 'red', 'orange', 'purple', 'darkgreen']
    for (name, werte), col in zip(konzepte_radar.items(), farben_rad):
        w = werte + [werte[0]]
        ax_r.plot(angles_r, w, 'o-', lw=1.8, color=col, label=name)
        ax_r.fill(angles_r, w, alpha=0.07, color=col)
    ax_r.set_xticks(angles_r[:-1])
    ax_r.set_xticklabels(kategorien_r, size=8)
    ax_r.set_ylim(0, 10)
    ax_r.set_rticks([2, 4, 6, 8, 10])
    ax_r.set_rlabel_position(20)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.55, 1.2), fontsize=8)
    ax_r.set_title('Gesamtvergleich\n(höher = besser)', fontsize=9, pad=15)

    fig.suptitle('Abbildung 13 – Gesamtbilanz: Neodym-Einsparung, Wirkungsgrad, CO₂, Wirtschaftlichkeit',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT}/plot13_gesamtbilanz.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot13_gesamtbilanz.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 13 gespeichert.")


# ==============================================================================
# PLOT 14 – Entropieproduktion: CMOS vs. IoFET vs. Landauer-Grenze
# ==============================================================================
def plot14_entropie_vergleich():
    """
    Drei Subplots zur Entropie und Reversibilität:
      1) Dissipationsenergie vs. Temperatur (CMOS / IoFET / Landauer)
      2) Kumulative Entropie S_kum(N) für große Schaltanzahlen (log-log)
      3) Energieflussbilanz pro Schaltvorgang als gestapelter Balken
    """
    k_B = 1.381e-23          # J/K
    E_CMOS = 1e-15            # J  (1 fJ)
    E_IoFET = 0.07e-15        # J  (0.07 fJ)
    T_arr = np.linspace(200, 500, 300)   # K
    E_Landauer = k_B * T_arr * np.log(2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Abbildung 14 – Entropieproduktion: CMOS vs. IoFET vs. Landauer-Grenze',
                 fontsize=14, fontweight='bold')

    # ── Subplot 1: Dissipationsenergie vs. Temperatur ──────────────────────
    ax = axes[0]
    ax.semilogy(T_arr, np.full_like(T_arr, E_CMOS),
                color='#d62728', lw=2, label=r'CMOS ($E_\mathrm{CMOS}$)')
    ax.semilogy(T_arr, np.full_like(T_arr, E_IoFET),
                color='#1f77b4', lw=2, label=r'IoFET ($E_\mathrm{IoFET}$)')
    ax.semilogy(T_arr, E_Landauer,
                color='#2ca02c', lw=2, ls='--', label=r'Landauer $k_BT\ln 2$')
    ax.axvline(300, color='gray', ls=':', lw=1, alpha=0.7, label=r'$T=300\,\mathrm{K}$')
    ax.set_xlabel('Temperatur $T$ (K)')
    ax.set_ylabel('Energie pro Schaltvorgang (J)')
    ax.set_title('Dissipationsenergie vs. Temperatur')
    ax.legend(fontsize=9)
    # Faktor-14-Pfeil
    ax.annotate('', xy=(350, E_IoFET), xytext=(350, E_CMOS),
                arrowprops=dict(arrowstyle='<->', color='#ff7f0e', lw=1.5))
    ax.text(355, np.sqrt(E_CMOS * E_IoFET), 'Faktor 14',
            color='#ff7f0e', fontsize=9, va='center')

    # ── Subplot 2: Kumulative Entropie S_kum(N) ────────────────────────────
    ax = axes[1]
    T0 = 300.0                           # Referenztemperatur
    N_arr = np.logspace(0, 12, 200)
    S_cmos  = N_arr * E_CMOS  / T0
    S_iofet = N_arr * E_IoFET / T0
    ax.loglog(N_arr, S_cmos,  color='#d62728', lw=2, label='CMOS')
    ax.loglog(N_arr, S_iofet, color='#1f77b4', lw=2, label='IoFET')
    ax.set_xlabel(r'Anzahl Schaltvorgänge $N$')
    ax.set_ylabel(r'Kumulierte Entropie $S_\mathrm{kum}$ (J/K)')
    ax.set_title(r'Kumulierte Entropie $S_\mathrm{kum}(N)$')
    ax.legend()
    # Faktor-14-Beschriftung bei N=1e8
    n_ref = 1e8
    ax.annotate('Faktor 14',
                xy=(n_ref, n_ref * E_IoFET / T0),
                xytext=(n_ref * 10, n_ref * E_CMOS / T0 * 0.3),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e'),
                color='#ff7f0e', fontsize=9)

    # ── Subplot 3: Energieflussbilanz pro Schaltvorgang ───────────────────
    ax = axes[2]
    labels = ['CMOS', 'IoFET']
    # CMOS: 100% Wärme
    cmos_heat    = 100.0
    cmos_recovery = 0.0
    cmos_viscous  = 0.0
    # IoFET: 93% Wärme + 4.5% Oberflächenrückgewinnung + 2.5% viskos
    iofet_heat      = 93.0
    iofet_recovery  = 4.5
    iofet_viscous   = 2.5
    heat_vals     = [cmos_heat, iofet_heat]
    viscous_vals  = [cmos_viscous, iofet_viscous]
    recovery_vals = [cmos_recovery, iofet_recovery]
    x = np.arange(len(labels))
    w = 0.4
    bars1 = ax.bar(x, heat_vals,     width=w, color='#d62728', label='Wärme (irrev.)')
    bars2 = ax.bar(x, viscous_vals,  width=w, bottom=heat_vals, color='#ff7f0e',
                   label='Viskose Dissipation')
    base3  = [h + v for h, v in zip(heat_vals, viscous_vals)]
    bars3 = ax.bar(x, recovery_vals, width=w, bottom=base3, color='#2ca02c',
                   label='Rückgewonnen (EWOD)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Anteil Energie (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Energieflussbilanz pro Schaltvorgang')
    ax.legend(fontsize=9)
    # Gesamtbalken-Beschriftungen
    for xi, (h, v, r) in enumerate(zip(heat_vals, viscous_vals, recovery_vals)):
        ax.text(xi, h + v + r + 2, f'{h+v+r:.0f}%', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(f'{OUT}/plot14_entropie_vergleich.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot14_entropie_vergleich.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 14 gespeichert.")


# ==============================================================================
# PLOT 15 – IoFET-EM-Vergleich: Kraft, Mobilität, Äquivalenzhöhe
# ==============================================================================
def plot15_iofet_em_vergleich():
    """
    Drei Subplots:
      1) Schaltenergie vs. Temperatur (CMOS / IoFET / Landauer / bio. Synapse)
      2) Kraftverhältnis F_Grav/F_EM vs. Teilchenabstand r
      3) Äquivalente Gravitationshöhe h* für verschiedene Ionen (Balken)
    """
    k_B = 1.381e-23          # J/K
    G   = 6.674e-11          # m^3 kg^-1 s^-2
    k_e = 8.988e9            # N m^2 C^-2
    e   = 1.602e-19          # C
    T_arr = np.linspace(200, 500, 300)
    g   = 9.807              # m/s^2

    E_CMOS    = 1e-15
    E_IoFET   = 0.07e-15
    E_synapse = 1e-12        # 1 pJ (biologische Synapse)
    E_L       = k_B * T_arr * np.log(2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Abbildung 15 – Ionotronisches Computing: EM-Überlegenheit & Kraftvergleich',
                 fontsize=14, fontweight='bold')

    # ── Subplot 1: Schaltenergie vs. T ────────────────────────────────────
    ax = axes[0]
    ax.semilogy(T_arr, np.full_like(T_arr, E_CMOS),
                color='#d62728', lw=2, label='CMOS (1 fJ)')
    ax.semilogy(T_arr, np.full_like(T_arr, E_IoFET),
                color='#1f77b4', lw=2, label='IoFET (0.07 fJ)')
    ax.semilogy(T_arr, np.full_like(T_arr, E_synapse),
                color='#9467bd', lw=2, ls='-.', label='Biol. Synapse (1 pJ)')
    ax.semilogy(T_arr, E_L,
                color='#2ca02c', lw=2, ls='--', label=r'Landauer $k_BT\ln 2$')
    ax.set_xlabel('Temperatur $T$ (K)')
    ax.set_ylabel('Schaltenergie (J)')
    ax.set_title('Schaltenergie vs. Temperatur')
    ax.legend(fontsize=9)

    # ── Subplot 2: F_Grav / F_EM vs. r  (Protonenpaar) ───────────────────
    ax = axes[1]
    m_p = 1.673e-27          # kg (Proton)
    r_arr = np.logspace(-15, -3, 300)   # m
    F_grav = G * m_p**2 / r_arr**2
    F_em   = k_e * e**2     / r_arr**2
    ratio  = F_grav / F_em
    ax.loglog(r_arr, ratio, color='#8c564b', lw=2)
    # horizontale Linie (Wert konstant für gleiche r-Abhängigkeit)
    r_val = ratio[0]
    ax.axhline(r_val, color='gray', ls='--', lw=1, alpha=0.6,
               label=rf'$F_G/F_{{EM}} = {r_val:.2e}$')
    # Bohr-Radius markieren
    a0 = 5.292e-11
    ax.axvline(a0, color='#1f77b4', ls=':', lw=1.5, label=f'Bohr-Radius $a_0$')
    ax.set_xlabel('Teilchenabstand $r$ (m)')
    ax.set_ylabel(r'$F_\mathrm{Grav} / F_\mathrm{EM}$')
    ax.set_title('Kraftverhältnis Gravitation/EM (Protonenpaar)')
    ax.legend(fontsize=9)

    # ── Subplot 3: Äquivalente Gravitationshöhe h* für Ionen ─────────────
    ax = axes[2]
    # h* = E_switch / (m_ion * g);  E_switch = e * V_schalt
    V_schalt = 0.123         # V
    E_sw     = e * V_schalt  # J pro Ion

    ions  = [r'K$^+$', r'Na$^+$', r'Cl$^-$', r'H$^+$']
    masses = np.array([6.492e-26, 3.818e-26, 5.887e-26, 1.673e-27])  # kg
    h_star = E_sw / (masses * g)

    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(ions, h_star, color=colors_bar, edgecolor='k', alpha=0.8)
    ax.set_yscale('log')
    ax.set_ylabel(r'Äquivalente Gravitationshöhe $h^*$ (m)')
    ax.set_title(r'Gravitationsäquivalenz des Schaltfeldes')
    for bar, val in zip(bars, h_star):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                f'{val:.1e} m', ha='center', va='bottom', fontsize=8)
    # Referenzlinien
    ax.axhline(1.1e8, color='gray', ls='--', lw=1, alpha=0.7,
               label=r'$h^*_{\mathrm{K}^+} \approx 1.1\times10^8\,\mathrm{m}$')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(f'{OUT}/plot15_iofet_em_vergleich.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot15_iofet_em_vergleich.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 15 gespeichert.")


# ==============================================================================
# PLOT 16 – IoFET-Verlustbilanz im PMSM-Regelkreis
# ==============================================================================
def plot16_iofet_verlustbilanz():
    """
    Drei Subplots:
      1) Schaltenergie-Vergleich (horizontal Balken, log-Skala)
      2) P_Regel vs. N_gates * f_sw für CMOS und IoFET (doppelt-log)
      3) Aktualisierte Gesamtbilanz mit 5 Synergiemechanismen
    """
    k_B = 1.381e-23
    T0  = 300.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Abbildung 16 – IoFET-Regelungselektronik in der PMSM-Verlustbilanz',
                 fontsize=14, fontweight='bold')

    # ── Subplot 1: Schaltenergie-Vergleich (horizontal) ───────────────────
    ax = axes[0]
    labels  = ['Landauer\n(300 K)', 'IoFET', 'CMOS', 'Biol.\nSynapse']
    values  = [k_B * T0 * np.log(2), 0.07e-15, 1e-15, 1e-12]
    colors  = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']
    y_pos   = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor='k', alpha=0.85)
    ax.set_xscale('log')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Schaltenergie (J)')
    ax.set_title('Schaltenergie-Vergleich')
    for i, v in enumerate(values):
        ax.text(v * 1.3, i, f'{v:.2e} J', va='center', fontsize=8)
    ax.axvline(k_B * T0 * np.log(2), color='#2ca02c', ls='--', lw=1, alpha=0.5)

    # ── Subplot 2: P_Regel vs. N_gates * f_sw ─────────────────────────────
    ax = axes[1]
    Nf_arr  = np.logspace(6, 14, 200)        # N_gates * f_sw in Hz
    E_CMOS  = 1e-15
    E_IoFET = 0.07e-15
    P_cmos  = Nf_arr * E_CMOS
    P_iofet = Nf_arr * E_IoFET
    ax.loglog(Nf_arr, P_cmos,  color='#d62728', lw=2, label='CMOS')
    ax.loglog(Nf_arr, P_iofet, color='#1f77b4', lw=2, label='IoFET')
    # Betriebspunkt markieren: N=1e4, f=1e5 → Nf=1e9
    ax.axvline(1e9, color='gray', ls=':', lw=1.5, alpha=0.8,
               label=r'$N_G=10^4,\,f=10^5\,\mathrm{Hz}$')
    ax.scatter([1e9], [1e9 * E_CMOS],  color='#d62728', s=60, zorder=5)
    ax.scatter([1e9], [1e9 * E_IoFET], color='#1f77b4', s=60, zorder=5)
    ax.set_xlabel(r'$N_\mathrm{Gatter} \cdot f_\mathrm{sw}$ (Hz)')
    ax.set_ylabel(r'$P_\mathrm{Regel}$ (W)')
    ax.set_title(r'Regelungsverlust $P_\mathrm{Regel}$')
    ax.legend(fontsize=9)

    # ── Subplot 3: Aktualisierte Gesamtbilanz (5 Mechanismen) ─────────────
    ax = axes[2]
    mechanisms = ['Konzept I\n(Puffer)', 'Konzept II\n(Hybrid)',
                  'Konzept III\n(FOC+grav.)', 'Konzept IV\n(TMD)',
                  'Konzept V\n(IoFET-Regel.)']
    nd_savings    = [20.0, 40.0, 15.0,  2.5,  1.0]    # % Nd-Einsparung (Mittelwerte)
    eta_gains     = [ 0.5,  1.0,  0.35, 0.2,  0.01]   # % Wirkungsgradgewinn

    x  = np.arange(len(mechanisms))
    w  = 0.35
    ax2 = ax.twinx()
    b1  = ax.bar(x - w/2, nd_savings, width=w, color='#1f77b4',
                 alpha=0.8, label='Nd-Einsparung (%)')
    b2  = ax2.bar(x + w/2, eta_gains, width=w, color='#ff7f0e',
                  alpha=0.8, label=r'$\Delta\eta$ (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=8)
    ax.set_ylabel('Nd-Einsparung (%)', color='#1f77b4')
    ax2.set_ylabel(r'Wirkungsgradgewinn $\Delta\eta$ (%)', color='#ff7f0e')
    ax.set_title('Gesamtbilanz: 5 Synergiemechanismen')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{OUT}/plot16_iofet_verlustbilanz.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot16_iofet_verlustbilanz.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 16 gespeichert.")


# ==============================================================================
# PLOT 17 – Kritische Materialien: CMOS vs. IRF (Seltenerd-Perspektive)
# ==============================================================================
def plot17_irf_material():
    """
    Drei Subplots:
      1) Kritischer Materialbedarf je Element: CMOS vs. IRF (gruppierter Balken)
      2) Supply Risk Index (SRI) je Element mit CRM-Schwelle
      3) Sankey-artiges Übergangsbild: CMOS-Materialien → IRF-Materialien
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Abbildung 17 – Kritische Materialien: CMOS- vs. IRF-Architektur',
                 fontsize=14, fontweight='bold')

    elements  = ['In', 'Co', 'Ta', 'Ru', 'Nd', 'Dy']
    # Relativer Materialbedarf (normiert auf CMOS-Anteil = 1)
    cmos_vals = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    irf_vals  = np.array([0.00, 0.00, 0.00, 0.00, 1.00, 1.00])  # Nd/Dy bleiben im Magnetkreis
    # SRI-Werte (nach EU-CRM-Methodik 2023, normiert [0,1])
    sri_vals  = [0.71, 0.63, 0.82, 0.68, 0.78, 0.91]
    # IRF-Architektur: neue Materialien
    irf_new_elements = ['Al\n(ALD)', 'Si\n(Sub.)', 'KCl\n(Ion.)', 'Glas\n(Geh.)']
    irf_new_sri      = [0.09, 0.12, 0.05, 0.04]

    x   = np.arange(len(elements))
    w   = 0.35
    col_cmos = '#d62728'
    col_irf  = '#1f77b4'

    # ── Subplot 1: Materialbedarf CMOS vs. IRF ────────────────────────────
    ax = axes[0]
    ax.bar(x - w/2, cmos_vals, width=w, color=col_cmos, alpha=0.8, label='CMOS', edgecolor='k')
    ax.bar(x + w/2, irf_vals,  width=w, color=col_irf,  alpha=0.8, label='IRF',  edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel('Relativer Materialeinsatz (normiert)')
    ax.set_title('Materialbedarf: CMOS vs. IRF')
    ax.legend()
    ax.set_ylim(0, 1.35)
    for xi, (cv, iv) in enumerate(zip(cmos_vals, irf_vals)):
        if iv == 0:
            ax.text(xi + w/2, 0.05, '0', ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')

    # ── Subplot 2: Supply Risk Index ──────────────────────────────────────
    ax = axes[1]
    all_elems  = elements + irf_new_elements
    all_sri    = sri_vals + irf_new_sri
    all_colors = [col_cmos] * len(elements) + [col_irf] * len(irf_new_elements)
    x2 = np.arange(len(all_elems))
    ax.bar(x2, all_sri, color=all_colors, edgecolor='k', alpha=0.8)
    ax.axhline(0.5, color='gray', ls='--', lw=1.5, label='CRM-Schwelle (SRI > 0.5)')
    ax.set_xticks(x2)
    ax.set_xticklabels(all_elems, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Supply Risk Index (SRI)')
    ax.set_title('Supply Risk Index je Element')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    # CRM-Bereich schattieren
    ax.axhspan(0.5, 1.05, alpha=0.07, color='red', label='_nolegend_')
    # Legende für Farben
    patch_cmos = mpatches.Patch(color=col_cmos, alpha=0.8, label='CMOS-Materialien')
    patch_irf  = mpatches.Patch(color=col_irf,  alpha=0.8, label='IRF-Materialien')
    ax.legend(handles=[patch_cmos, patch_irf,
                       mpatches.Patch(color='gray', label='CRM-Schwelle')],
              fontsize=9)

    # ── Subplot 3: Sankey-artiges Übergangsbild ───────────────────────────
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Materialübergang CMOS → IRF')

    # CMOS-Seite (links)
    cmos_labels_left = ['In (ITO)', 'Co (Barriere)', 'Ta (Gate)', 'Ru (DRAM)',
                        'Nd (Magnet)', 'Si (Substrat)']
    cmos_colors_left = [col_cmos, col_cmos, col_cmos, col_cmos, '#ff7f0e', '#7f7f7f']
    for i, (lbl, col) in enumerate(zip(cmos_labels_left, cmos_colors_left)):
        y = 9.0 - i * 1.4
        ax.text(0.3, y, lbl, fontsize=9, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.6))

    # IRF-Seite (rechts)
    irf_labels_right = ['— (entfällt)', '— (entfällt)', 'Al₂O₃', '— (entfällt)',
                        'Nd (Magnet)', 'Si (Substrat)']
    irf_colors_right = ['#aec7e8', '#aec7e8', col_irf, '#aec7e8', '#ff7f0e', '#7f7f7f']
    for i, (lbl, col) in enumerate(zip(irf_labels_right, irf_colors_right)):
        y = 9.0 - i * 1.4
        ax.text(9.7, y, lbl, fontsize=9, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.6))

    # Pfeile verbinden
    for i, (col_left, col_right) in enumerate(
            zip(cmos_colors_left, irf_colors_right)):
        y = 9.0 - i * 1.4
        style   = 'dashed' if col_right == '#aec7e8' else 'solid'
        arrowcol = '#d62728' if col_right == '#aec7e8' else '#2ca02c'
        ax.annotate('', xy=(8.5, y), xytext=(2.5, y),
                    arrowprops=dict(arrowstyle='->', color=arrowcol,
                                   lw=1.5,
                                   linestyle=style))

    # Legende
    red_patch   = mpatches.Patch(color='#d62728', alpha=0.6, label='CRM eliminiert')
    green_patch = mpatches.Patch(color='#2ca02c', alpha=0.6, label='CRM erhalten')
    ax.legend(handles=[red_patch, green_patch],
              loc='lower center', fontsize=9, frameon=True)
    ax.text(5, 9.8, 'CMOS', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 0.3, '→ IRF', ha='center', fontsize=11, fontweight='bold',
            color=col_irf)

    plt.tight_layout()
    fig.savefig(f'{OUT}/plot17_irf_material.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/plot17_irf_material.svg', bbox_inches='tight')
    plt.close(fig)
    print("Plot 17 gespeichert.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print("Erzeuge alle Plots ...")
    plot1_pendel_energie()
    plot2_thermodynamik()
    plot3_perpetuum()
    plot4_ode_pendel()
    plot5_overbalanced_wheel()
    plot6_quantengravitation()
    plot7_vergleich()
    plot8_erweitert()
    plot9_zusammenfassung()
    plot10_synergie_uebersicht()
    plot11_foc_gravitation()
    plot12_tmd_und_material()
    plot13_gesamtbilanz()
    plot14_entropie_vergleich()
    plot15_iofet_em_vergleich()
    plot16_iofet_verlustbilanz()
    plot17_irf_material()
    print(f"\nAlle Plots gespeichert in: {OUT}")
