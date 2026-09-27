#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib-Figuren für das SR-91 Aurora II Aufklärungsflugzeug-Paper
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from scipy.optimize import fsolve
import os

OUT = 'figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

DARK = '#1a1a2e'
BLUE = '#0077b6'
CYAN = '#00b4d8'
ORANGE = '#f77f00'
RED = '#d62828'
GREEN = '#2dc653'
GRAY = '#adb5bd'

# -----------------------------------------------------------------------
# 1. Schubkurve & spezifischer Impuls vs. Mach-Zahl
# -----------------------------------------------------------------------
def fig_thrust_mach():
    M = np.linspace(0.1, 6.5, 500)
    # TBCC: Turbojet bis M3, dann Scramjet
    def thrust(M):
        T_SL = 220e3  # N (ein Triebwerk)
        # Turbojet: nimmt ab ab M>2 wenn nicht adaptierer Einlauf
        tj = T_SL * np.exp(-0.15*(M-1)**2) * (1 - 0.04*np.maximum(0-M,0))
        tj = np.where(M<0.1, T_SL, tj)
        # Scramjet: ab M>2.5 relevant
        scr = T_SL * 0.7 * (1 + 0.3*(M-2.5)) * np.where(M>=2.5, 1, 0)
        # TBCC-Kombination
        blend = 1/(1+np.exp(-4*(M-2.8)))
        return (1-blend)*tj + blend*scr

    T = thrust(M)*2  # 2 Triebwerke
    Isp = 1500 + 1800*np.tanh(0.9*(M-2)) + 200*M

    fig, ax1 = plt.subplots(figsize=(9,5), facecolor='white')
    ax2 = ax1.twinx()
    l1, = ax1.plot(M, T/1e3, color=BLUE, lw=2.5, label='Schubkraft $F_T$ [kN]')
    l2, = ax2.plot(M, Isp, color=ORANGE, lw=2.5, ls='--', label='Spez. Impuls $I_{sp}$ [s]')
    ax1.axvline(3.0, color=GRAY, ls=':', lw=1.2, label='TBCC-Umschaltpunkt')
    ax1.fill_betweenx([0, 600], 0, 3.0, alpha=0.08, color=BLUE)
    ax1.fill_betweenx([0, 600], 3.0, 6.5, alpha=0.08, color=ORANGE)
    ax1.text(1.3, 480, 'Turbinen-\nmodus', fontsize=9, color=BLUE, ha='center')
    ax1.text(4.8, 480, 'Scramjet-\nmodus', fontsize=9, color=ORANGE, ha='center')
    ax1.set_xlabel('Mach-Zahl $M$')
    ax1.set_ylabel('Gesamtschub $F_T$ [kN]', color=BLUE)
    ax2.set_ylabel('Spez. Impuls $I_{sp}$ [s]', color=ORANGE)
    ax1.set_xlim(0.1, 6.5); ax1.set_ylim(0, 600)
    ax2.set_ylim(1000, 5000)
    lines = [l1, l2, mpatches.Patch(color=GRAY, label='TBCC-Umschalt M=3')]
    ax1.legend(handles=lines, loc='upper right', framealpha=0.9)
    ax1.set_title('Schubkraft und spezifischer Impuls vs. Mach-Zahl\n(TBCC-Antrieb, 2 Triebwerke, $H=25$ km)')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig01_thrust_mach.pdf', bbox_inches='tight')
    plt.close()
    print("fig01 done")

# -----------------------------------------------------------------------
# 2. Auftriebs- und Widerstandspolaren (ca/cw-Diagramm)
# -----------------------------------------------------------------------
def fig_polar():
    # Deltafluegel mit variabler Geometrie
    alpha = np.linspace(-5, 25, 400)
    AR = 1.8  # Streckung

    def get_polar(M_inf):
        # Lineare Theorie mit Kompressibilitaet (Prandtl-Glauert fuer M<1, Ackeret fuer M>1)
        if M_inf < 1:
            beta = np.sqrt(1 - M_inf**2)
            CLa = 2*np.pi / (beta + 2/(AR))  # rad^-1
        else:
            beta = np.sqrt(M_inf**2 - 1)
            CLa = 4 / (beta + 2/(AR))
        CL = CLa * np.deg2rad(alpha)
        CD0 = 0.012 + 0.003*M_inf if M_inf < 1 else 0.018 + 0.025*(M_inf-1)**1.3
        k = 1/(np.pi*AR*0.85)
        CD = CD0 + k*CL**2
        return CL, CD

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    machs = [0.5, 0.9, 2.0, 3.5, 5.0]
    colors = [BLUE, CYAN, GREEN, ORANGE, RED]
    for M, c in zip(machs, colors):
        CL, CD = get_polar(M)
        axes[0].plot(alpha, CL, color=c, lw=2, label=f'M={M}')
        axes[1].plot(CD, CL, color=c, lw=2, label=f'M={M}')

    axes[0].set_xlabel('Anstellwinkel $\\alpha$ [°]')
    axes[0].set_ylabel('Auftriebsbeiwert $c_L$')
    axes[0].set_title('Auftriebscharakteristik')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', lw=0.8)

    axes[1].set_xlabel('Widerstandsbeiwert $c_D$')
    axes[1].set_ylabel('Auftriebsbeiwert $c_L$')
    axes[1].set_title('Polare $c_L(c_D)$ — Lilienthal-Polare')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Aerodynamische Polare des SR-91 Deltaflügels bei verschiedenen Mach-Zahlen', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig02_polar.pdf', bbox_inches='tight')
    plt.close()
    print("fig02 done")

# -----------------------------------------------------------------------
# 3. Gleitzahl und max. Reichweite
# -----------------------------------------------------------------------
def fig_glide():
    M_arr = np.linspace(0.3, 6.0, 300)
    AR = 1.8
    def LD_max(M):
        if M < 1:
            b = np.sqrt(1-M**2); CD0 = 0.012+0.003*M
        else:
            b = np.sqrt(M**2-1); CD0 = 0.018+0.025*(M-1)**1.3
        CLa = (4/b if M>1 else 2*np.pi/b) / (1 + 2/(AR))
        k = 1/(np.pi*AR*0.85)
        return 0.5/np.sqrt(k*CD0)

    LD = np.array([LD_max(m) for m in M_arr])

    # Breguet-Reichweite (normiert)
    Isp = 1500 + 1800*np.tanh(0.9*(M_arr-2))
    g = 9.81
    beta_fuel = 0.45  # Kraftstoffanteil
    R_norm = Isp * g * LD * np.log(1/(1-beta_fuel))  # m

    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].plot(M_arr, LD, color=BLUE, lw=2.5)
    axes[0].axvline(2.5, color=ORANGE, ls='--', lw=1.5, label='Opt. Reiseflug M=2.5')
    axes[0].set_xlabel('Mach-Zahl $M$'); axes[0].set_ylabel('Gleitzahl $E = L/D$')
    axes[0].set_title('Gleitzahl $E_{max}(M)$'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(M_arr, R_norm/1e6, color=GREEN, lw=2.5)
    axes[1].set_xlabel('Mach-Zahl $M$'); axes[1].set_ylabel('Reichweite $R$ [10$^3$ km]')
    axes[1].set_title('Breguet-Reichweite vs. Mach-Zahl')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Gleitzahl und Breguet-Reichweite des SR-91', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig03_glide_range.pdf', bbox_inches='tight')
    plt.close()
    print("fig03 done")

# -----------------------------------------------------------------------
# 4. Wendemanöver: Kurvenradius, Wendezeit vs. Mach
# -----------------------------------------------------------------------
def fig_turn():
    M = np.linspace(1.5, 5.5, 400)
    H = 20000  # m
    rho0 = 1.225; T0 = 288.15; lam = 0.0065; g = 9.81; R_gas = 287.05
    gamma_air = 1.4
    # ISA bei H=20km
    T_H = T0 - lam*H
    p_H = 101325*(T_H/T0)**(g/(lam*R_gas))
    rho_H = p_H/(R_gas*T_H)
    a_H = np.sqrt(gamma_air*R_gas*T_H)  # Schallgeschw.
    V = M * a_H

    m = 32000  # kg (leer + 2 Piloten + Sensoren, halbvolles Kraftstofftank)
    S = 78.5   # m^2
    W = m*g

    # n_max aus Strukturlimit (n=6g) und aero (C_Lmax=1.2 bei HighSpeed)
    n_struct = 6.0
    AR = 1.8
    def CL_max(M):
        if M < 1:
            b = np.sqrt(max(1-M**2,0.001))
            return min(1.2, 2*np.pi*np.deg2rad(18)/b)
        else:
            b = np.sqrt(M**2-1)
            return min(0.8, 4*np.deg2rad(15)/b)

    n_aero = np.array([0.5*rho_H*v**2*S*CL_max(m_)/W for v, m_ in zip(V, M)])
    n_max = np.minimum(n_struct, n_aero)
    n_max = np.maximum(n_max, 1.0)

    # Kurvenradius: r = V^2 / (g*sqrt(n^2-1))
    r = V**2 / (g*np.sqrt(np.maximum(n_max**2 - 1, 0.01)))
    # Wendezeit (180°-Wende): t = pi*r/V = pi*V/(g*sqrt(n^2-1))
    t180 = np.pi * r / V
    # Winkelrate: omega = g*sqrt(n^2-1)/V
    omega_deg = np.rad2deg(g*np.sqrt(np.maximum(n_max**2-1, 0.01))/V)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(M, r/1000, color=BLUE, lw=2.5)
    axes[0].set_xlabel('Mach-Zahl $M$'); axes[0].set_ylabel('Kurvenradius $r$ [km]')
    axes[0].set_title('Minimaler Kurvenradius\n$r_{min} = V^2 / (g\\sqrt{n^2-1})$')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(M, r/1000, alpha=0.12, color=BLUE)

    axes[1].plot(M, t180, color=ORANGE, lw=2.5)
    axes[1].set_xlabel('Mach-Zahl $M$'); axes[1].set_ylabel('Wendezeit $t_{180}$ [s]')
    axes[1].set_title('180°-Wendezeit\n$t_{180} = \\pi r / V$')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(M, omega_deg, color=GREEN, lw=2.5)
    axes[2].set_xlabel('Mach-Zahl $M$'); axes[2].set_ylabel('Gierrate $\\dot{\\psi}$ [°/s]')
    axes[2].set_title('Max. Winkelrate\n$\\dot{\\psi} = g\\sqrt{n^2-1}/V$')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Wendemanöver-Charakteristik des SR-91 bei $H=20$ km', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig04_turn.pdf', bbox_inches='tight')
    plt.close()
    print("fig04 done")

# -----------------------------------------------------------------------
# 5. n-V-Manövrierdiagramm (V-n-Diagramm)
# -----------------------------------------------------------------------
def fig_Vn():
    H = 20000
    g = 9.81; R_gas = 287.05; lam = 0.0065; T0 = 288.15
    gamma_air = 1.4
    T_H = T0 - lam*H
    rho_H = 101325*(T_H/T0)**(g/(lam*R_gas)) / (R_gas*T_H)
    a_H = np.sqrt(gamma_air*R_gas*T_H)

    S = 78.5; m = 32000; W = m*g
    CLmax_pos = 1.2; CLmax_neg = -0.6
    n_max = 6.0; n_min = -2.5

    V = np.linspace(0, 2500, 1000)

    # aero limits
    n_pos_aero = 0.5*rho_H*V**2*S*CLmax_pos / W
    n_neg_aero = 0.5*rho_H*V**2*S*CLmax_neg / W

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(V, np.minimum(n_pos_aero, n_max), color=BLUE, lw=2.5, label='Positive Auftriebsgrenze')
    ax.plot(V, np.maximum(n_neg_aero, n_min), color=RED, lw=2.5, label='Negative Auftriebsgrenze')
    ax.axhline(n_max, color=BLUE, ls='--', lw=1.5, alpha=0.6, label=f'$n_{{max}}={n_max}$')
    ax.axhline(n_min, color=RED, ls='--', lw=1.5, alpha=0.6, label=f'$n_{{min}}={n_min}$')

    # Flatter- / Dive-Speed
    V_D = 2400  # m/s
    ax.axvline(V_D, color=ORANGE, ls=':', lw=2, label=f'$V_D = {V_D}$ m/s')

    # Manövriergeschw.
    V_A = np.sqrt(2*W*n_max/(rho_H*S*CLmax_pos))
    ax.axvline(V_A, color=GREEN, ls=':', lw=2, label=f'$V_A = {V_A:.0f}$ m/s')

    # Fuellung
    V_fill = np.linspace(0, V_D, 1000)
    n_p = np.minimum(0.5*rho_H*V_fill**2*S*CLmax_pos/W, n_max)
    n_n = np.maximum(0.5*rho_H*V_fill**2*S*CLmax_neg/W, n_min)
    ax.fill_between(V_fill, n_n, n_p, alpha=0.12, color=CYAN, label='Zulässiger Manövrierbereich')

    ax.set_xlabel('Fluggeschwindigkeit $V$ [m/s]')
    ax.set_ylabel('Lastvielfaches $n$')
    ax.set_title('V-n Manövrierdiagramm des SR-91 bei $H=20$ km (ISA)')
    ax.set_xlim(0, V_D+100); ax.set_ylim(-3.5, 7.5)
    ax.axhline(1, color='k', lw=0.8, ls='-', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig05_Vn.pdf', bbox_inches='tight')
    plt.close()
    print("fig05 done")

# -----------------------------------------------------------------------
# 6. Steigflugprofil und Flugbahnoptimierung
# -----------------------------------------------------------------------
def fig_climb():
    # Einfaches Punkt-Masse-Modell
    import warnings; warnings.filterwarnings('ignore')
    g = 9.81; m = 32000
    R_gas = 287.05; gamma_air = 1.4; lam = 0.0065; T0 = 288.15

    def atmo(h):
        T = T0 - lam*h
        p = 101325*(T/T0)**(g/(lam*R_gas))
        rho = p/(R_gas*T)
        a = np.sqrt(gamma_air*R_gas*T)
        return rho, a

    def sim_climb(gamma_climb_deg=8.0, dt=0.5, t_max=600):
        gamma_c = np.deg2rad(gamma_climb_deg)
        h = 0.0; v = 300.0; x = 0.0
        S = 78.5; AR = 1.8
        hs, vs, xs, Ts, Ms = [h],[v],[x],[],[0]
        for _ in range(int(t_max/dt)):
            rho, a = atmo(h)
            M = v/a
            # Schub
            T_SL = 220e3*2
            thrust = T_SL * np.exp(-0.1*(M-1)**2)
            if M>3: thrust = T_SL*0.7*(1+0.15*(M-3))
            # Auftrieb/Widerst.
            beta = np.sqrt(max(abs(1-M**2),0.001))
            CLa = 4/beta if M>1 else 2*np.pi/beta
            alpha = gamma_c + np.arctan(m*g*np.cos(gamma_c)/(0.5*rho*v**2*S*CLa*10))
            CL = CLa * alpha * 0.1
            CD0 = 0.012+0.003*M if M<1 else 0.018+0.025*(M-1)**1.3
            CD = CD0 + CL**2/(np.pi*AR*0.85)
            D = 0.5*rho*v**2*S*CD
            L = 0.5*rho*v**2*S*CL
            ax_val = (thrust*np.cos(gamma_c) - D - m*g*np.sin(gamma_c))/m
            v += ax_val*dt; h += v*np.sin(gamma_c)*dt; x += v*np.cos(gamma_c)*dt
            if h<0: h=0
            if h>35000: break
            hs.append(h); vs.append(v); xs.append(x)
            Ts.append(thrust); Ms.append(M)
        return np.array(hs), np.array(vs), np.array(xs), np.array(Ms)

    h1,v1,x1,M1 = sim_climb(8.0)
    h2,v2,x2,M2 = sim_climb(15.0)
    h3,v3,x3,M3 = sim_climb(25.0)

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    t1 = np.arange(len(h1))*0.5
    t2 = np.arange(len(h2))*0.5
    t3 = np.arange(len(h3))*0.5

    axes[0].plot(t1, h1/1000, color=BLUE, lw=2, label='$\\gamma=8°$')
    axes[0].plot(t2, h2/1000, color=ORANGE, lw=2, label='$\\gamma=15°$')
    axes[0].plot(t3, h3/1000, color=RED, lw=2, label='$\\gamma=25°$')
    axes[0].set_xlabel('Zeit $t$ [s]'); axes[0].set_ylabel('Höhe $H$ [km]')
    axes[0].set_title('Steigflugprofil $H(t)$')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t1, M1, color=BLUE, lw=2, label='$\\gamma=8°$')
    axes[1].plot(t2, M2, color=ORANGE, lw=2, label='$\\gamma=15°$')
    axes[1].plot(t3, M3, color=RED, lw=2, label='$\\gamma=25°$')
    axes[1].set_xlabel('Zeit $t$ [s]'); axes[1].set_ylabel('Mach-Zahl $M$')
    axes[1].set_title('Mach-Verlauf im Steigflug')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(x1/1000, h1/1000, color=BLUE, lw=2, label='$\\gamma=8°$')
    axes[2].plot(x2/1000, h2/1000, color=ORANGE, lw=2, label='$\\gamma=15°$')
    axes[2].plot(x3/1000, h3/1000, color=RED, lw=2, label='$\\gamma=25°$')
    axes[2].set_xlabel('Horizontaldistanz $x$ [km]'); axes[2].set_ylabel('Höhe $H$ [km]')
    axes[2].set_title('Flugbahn $H(x)$')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Steigfluganalyse SR-91 bei verschiedenen Steigwinkeln $\\gamma$', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig06_climb.pdf', bbox_inches='tight')
    plt.close()
    print("fig06 done")

# -----------------------------------------------------------------------
# 7. Aerodynamische Erwärmung (Kinetische Erwärmung)
# -----------------------------------------------------------------------
def fig_heating():
    M = np.linspace(0.5, 6.5, 400)
    T_inf = 216.65  # K bei H=20km ISA
    gamma_air = 1.4

    # Stagnationstemperatur
    T0_stag = T_inf * (1 + (gamma_air-1)/2 * M**2)
    # Wandtemperatur (adiabatisch, recovery factor r=0.85 turb.)
    r = 0.85
    T_aw = T_inf * (1 + r*(gamma_air-1)/2*M**2)
    # Strukturlimit Ti-Legierung
    T_lim_Ti = np.full_like(M, 593+273)  # K
    # Strukturlimit CMC
    T_lim_CMC = np.full_like(M, 1200+273)  # K

    # Wärmefluss (vereinfacht, Eckert)
    V = M * np.sqrt(gamma_air*287.05*T_inf)
    rho_H = 0.0889  # kg/m^3 H=20km
    # Stanton-Zahl vereinfacht
    St = 0.03 * (rho_H*V*0.5/1.8e-5)**(-0.2) if True else 0.001
    Re_x = rho_H*V*0.5 / 1.8e-5
    St_arr = 0.03 * (rho_H*V*0.5/(1.8e-5))**(-0.2)
    cp_air = 1005
    q_dot = rho_H * V * cp_air * St_arr * (T0_stag - 400)  # kW/m^2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(M, T0_stag-273, color=RED, lw=2.5, label='Stagnationstemperatur $T_0$')
    axes[0].plot(M, T_aw-273, color=ORANGE, lw=2, ls='--', label='Adiab. Wandtemp. $T_{aw}$')
    axes[0].axhline(593, color=BLUE, ls=':', lw=2, label='Ti-Legierung Limit (593°C)')
    axes[0].axhline(1200, color=GREEN, ls=':', lw=2, label='CMC-Struktur Limit (1200°C)')
    axes[0].fill_between(M, 0, T_aw-273, where=T_aw-273>593, alpha=0.15, color=RED, label='Kritischer Bereich Ti')
    axes[0].set_xlabel('Mach-Zahl $M$'); axes[0].set_ylabel('Temperatur [°C]')
    axes[0].set_title('Aerodynamische Erwärmung\n(ISA, $H=20$ km)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-50, 2000)

    axes[1].plot(M, q_dot/1000, color=RED, lw=2.5)
    axes[1].fill_between(M, 0, q_dot/1000, alpha=0.15, color=RED)
    axes[1].set_xlabel('Mach-Zahl $M$'); axes[1].set_ylabel('Wärmefluss $\\dot{q}$ [MW/m$^2$]')
    axes[1].set_title('Aerodynamischer Wärmefluss\nan der Vorderkante ($x_c=0.5$ m)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Thermische Belastung des SR-91 durch aerodynamische Erwärmung', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig07_heating.pdf', bbox_inches='tight')
    plt.close()
    print("fig07 done")

# -----------------------------------------------------------------------
# 8. Cockpit-Druckkabine & g-Belastung Piloten
# -----------------------------------------------------------------------
def fig_pilot_loads():
    t = np.linspace(0, 120, 1200)
    # Szenario: Anflug, Wende, Ausweichen, Normalisierung
    def n_profile(t):
        n = np.ones_like(t)
        # Einleitungswende: t=10-30s
        mask1 = (t>=10) & (t<=30)
        n[mask1] = 1 + 5*np.sin(np.pi*(t[mask1]-10)/20)
        # Nega. Last bei t=35-40
        mask2 = (t>=35) & (t<=40)
        n[mask2] = 1 - 3.0*np.sin(np.pi*(t[mask2]-35)/5)
        # Hochlast t=50-80
        mask3 = (t>=50) & (t<=80)
        n[mask3] = 1 + 5.5*(0.5+0.5*np.sin(2*np.pi*(t[mask3]-50)/60))
        # peak t=65
        mask4 = (t>=62) & (t<=68)
        n[mask4] = 5.8 + 0.2*np.sin(np.pi*(t[mask4]-62)/6)
        return n

    n = n_profile(t)
    # g-induzierter Blutdruckverlust (Franks 1993 simplified)
    # Herz-Hirn-Abstand = 0.3m
    g_crit = 4.5  # ohne Anti-G-Anzug
    g_suit_adj = 1.8  # G-Anzug-Bonus
    n_pboc = g_crit + g_suit_adj  # ~6.3g für Loss of Consciousness

    # Herzdruck vereinfacht
    h_heart_brain = 0.3  # m
    rho_blood = 1060; g_acc = 9.81
    Delta_p = rho_blood * g_acc * h_heart_brain / 133.322  # mmHg per g

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0,0].plot(t, n, color=BLUE, lw=2)
    axes[0,0].axhline(n_pboc, color=RED, ls='--', lw=1.8, label=f'G-LOC Grenze ({n_pboc}g)')
    axes[0,0].axhline(-2.5, color=ORANGE, ls='--', lw=1.5, label='neg. $n_{min}=-2.5$')
    axes[0,0].fill_between(t, n, n_pboc, where=n>n_pboc, alpha=0.3, color=RED, label='Überlast-Bereich')
    axes[0,0].set_xlabel('Zeit $t$ [s]'); axes[0,0].set_ylabel('Lastvielfaches $n$ [g]')
    axes[0,0].set_title('g-Belastungsprofil — typisches Wendemanöver')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # Blutdruck am Auge
    BP_sys_0 = 120  # mmHg
    BP_sys = BP_sys_0 - Delta_p*n
    axes[0,1].plot(t, BP_sys, color=RED, lw=2)
    axes[0,1].axhline(22, color=ORANGE, ls='--', lw=1.5, label='Grauschleier-Grenze (22 mmHg)')
    axes[0,1].axhline(0, color='k', lw=0.8, ls='-')
    axes[0,1].fill_between(t, BP_sys, 22, where=BP_sys<22, alpha=0.3, color=ORANGE, label='Grauschleier-Risiko')
    axes[0,1].set_xlabel('Zeit $t$ [s]'); axes[0,1].set_ylabel('Systemischer Blutdruck Auge [mmHg]')
    axes[0,1].set_title('Okularer Blutdruck unter g-Last')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # G-Anzug Druckkompensation
    g_vals = np.linspace(1, 9, 100)
    p_suit_no = np.zeros_like(g_vals)
    p_suit_standard = np.where(g_vals>1.5, (g_vals-1.5)*11.5, 0)  # hPa
    p_suit_advanced = np.where(g_vals>1.0, (g_vals-1.0)*15.0, 0)
    axes[1,0].fill_between(g_vals, p_suit_no, p_suit_standard, alpha=0.3, color=BLUE, label='Standard Anti-G-Anzug')
    axes[1,0].fill_between(g_vals, p_suit_standard, p_suit_advanced, alpha=0.3, color=GREEN, label='Erweiterter ATAGS')
    axes[1,0].plot(g_vals, p_suit_standard, color=BLUE, lw=2)
    axes[1,0].plot(g_vals, p_suit_advanced, color=GREEN, lw=2)
    axes[1,0].set_xlabel('Lastvielfaches $n$ [g]'); axes[1,0].set_ylabel('G-Anzug-Druck [hPa]')
    axes[1,0].set_title('Anti-G-Anzug Druckkennlinie')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # Kabinendruckprofil vs Höhe
    H_arr = np.linspace(0, 30000, 300)
    lam=0.0065; T0=288.15; g=9.81; R_gas=287.05
    p_atm = 101325*(1-lam*H_arr/T0)**(g/(lam*R_gas))
    p_cabin = np.where(H_arr<=2400, p_atm, 101325*(1-lam*2400/T0)**(g/(lam*R_gas)))
    p_cabin_emergency = np.where(H_arr<=6000, p_atm, 101325*(1-lam*6000/T0)**(g/(lam*R_gas)))

    axes[1,1].plot(p_atm/100, H_arr/1000, color=GRAY, lw=1.5, ls=':', label='Außendruck (ISA)')
    axes[1,1].plot(p_cabin/100, H_arr/1000, color=BLUE, lw=2.5, label='Kabinendruck (2400 m äquiv.)')
    axes[1,1].plot(p_cabin_emergency/100, H_arr/1000, color=ORANGE, lw=2, ls='--', label='Notdruckprofil (6000 m äquiv.)')
    axes[1,1].set_xlabel('Druck [hPa]'); axes[1,1].set_ylabel('Flughöhe $H$ [km]')
    axes[1,1].set_title('Kabinendruckregelung vs. Flughöhe')
    axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Piloten-Belastungsanalyse: g-Kräfte, Blutdruck & Lebenserhaltung', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig08_pilot_loads.pdf', bbox_inches='tight')
    plt.close()
    print("fig08 done")

# -----------------------------------------------------------------------
# 9. Reichweiten-Payload-Diagramm & Sensoreinsatz
# -----------------------------------------------------------------------
def fig_recon_envelope():
    # Aufklärungsreichweite (Sensorreichweite als Funktion der Flughöhe)
    H_km = np.linspace(15, 35, 300)
    # Sichtliniendistanz (Radar-Horizon)
    R_earth = 6371e3  # m
    R_horizon = np.sqrt(2*R_earth*(H_km*1000))  # m

    # SIGINT/ELINT Reichweite (Freifeld-Ausbreitung)
    P_SIGINT = 50e3  # W ERP des Signals
    G_rx = 30  # dBi Antennengewinn
    f_GHz = 10; c_light = 3e8
    lambda_m = c_light/(f_GHz*1e9)
    sensitivity = -110  # dBm
    # FSPL: P_r = P_t * G_t * G_r * (lambda/(4*pi*R))^2
    # R_max (vereinfacht)
    G_t = 10  # dBi Zielsender
    EIRP_dBm = 10*np.log10(P_SIGINT*1e3) + G_t  # dBm
    R_SIGINT_km = (lambda_m/(4*np.pi)) * 10**((EIRP_dBm + G_rx - sensitivity)/(20)) / 1000

    # Fotoaufklärung: GSD als Funktion von H und f
    f_optic_mm = 1800  # mm Brennweite
    sensor_px = 0.0052  # mm Pixelgrösse
    GSD_cm = (H_km*1e5 * sensor_px) / f_optic_mm  # cm am Boden

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(R_horizon/1000, H_km, color=BLUE, lw=2.5, label='Radar-Horizont')
    axes[0].axhline(20, color=ORANGE, ls='--', lw=1.5, label='Typ. Reiseflug $H=20$ km')
    axes[0].axhline(30, color=RED, ls='--', lw=1.5, label='Max Betrieb $H=30$ km')
    axes[0].set_xlabel('Sichtliniendistanz [km]'); axes[0].set_ylabel('Höhe [km]')
    axes[0].set_title('Radar-Horizont vs. Flughöhe')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(H_km, GSD_cm, color=GREEN, lw=2.5)
    axes[1].axhline(5, color=RED, ls='--', lw=1.5, label='5 cm GSD (krit. Aufl.)')
    axes[1].axhline(30, color=ORANGE, ls='--', lw=1.5, label='30 cm GSD (Standard)')
    axes[1].set_xlabel('Höhe [km]'); axes[1].set_ylabel('GSD [cm]')
    axes[1].set_title('Bodenauflösung (GSD)\n$f=1800$ mm, $p=5.2$ µm Sensor')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Payload-Reichweiten-Diagramm
    R_payload = np.linspace(0, 3000, 300)  # kg Nutzlast
    R_range_km = 12000 * (1 - R_payload/3000)**(0.7)
    R_endur_h = R_range_km / (3.5*340)  # 3.5*a = Reisegeschw. km/h

    ax3a = axes[2]
    ax3b = ax3a.twinx()
    ax3a.plot(R_payload, R_range_km, color=BLUE, lw=2.5, label='Reichweite [km]')
    ax3b.plot(R_payload, R_endur_h, color=ORANGE, lw=2, ls='--', label='Ausdauer [h]')
    ax3a.set_xlabel('Sensor-Nutzlast [kg]'); ax3a.set_ylabel('Reichweite [km]', color=BLUE)
    ax3b.set_ylabel('Ausdauer [h]', color=ORANGE)
    ax3a.set_title('Payload-Reichweiten-Kompromiss')
    ax3a.grid(True, alpha=0.3)
    lines3 = [mpatches.Patch(color=BLUE, label='Reichweite'), mpatches.Patch(color=ORANGE, label='Ausdauer')]
    ax3a.legend(handles=lines3, loc='upper right')

    plt.suptitle('Aufklärungspotenzial: Sensorreichweite, Auflösung und Payload-Kompromiss', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig09_recon.pdf', bbox_inches='tight')
    plt.close()
    print("fig09 done")

# -----------------------------------------------------------------------
# 10. Stabilitäts- und Steueranalyse (Längs- und Querdynamik)
# -----------------------------------------------------------------------
def fig_stability():
    # Eigenfrequenzen Längsmode als Funktion von Mach
    M_arr = np.linspace(0.5, 5.5, 300)
    # Phygoide Periode
    g = 9.81
    V = M_arr * 280  # approximiert
    T_phygoid = 2*np.pi*V / (g*np.sqrt(2))
    zeta_phygoid = 0.04 + 0.02*np.exp(-0.5*(M_arr-2.5)**2)

    # Kurzperiode
    omega_sp = 1.2 + 0.8*M_arr  # rad/s
    zeta_sp = 0.55 - 0.05*M_arr

    # Dutch Roll
    omega_dr = 1.8 - 0.15*M_arr
    zeta_dr = 0.2 + 0.05*M_arr

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(M_arr, T_phygoid, color=BLUE, lw=2.5, label='Phygoide Periode')
    axes[0,0].set_xlabel('Mach $M$'); axes[0,0].set_ylabel('Periode [s]')
    axes[0,0].set_title('Phygoide Eigenperiode')
    axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

    axes[0,1].plot(M_arr, omega_sp, color=GREEN, lw=2.5, label='$\\omega_n$ Kurzperiode')
    axes[0,1].plot(M_arr, zeta_sp*5, color=ORANGE, lw=2, ls='--', label='$\\zeta_{SP} \\times 5$')
    axes[0,1].axhline(5*0.35, color=RED, ls=':', lw=1.5, label='MIL-SPEC $\\zeta_{min}=0.35$')
    axes[0,1].set_xlabel('Mach $M$'); axes[0,1].set_ylabel('$\\omega_n$ [rad/s] / $5\\zeta$ [-]')
    axes[0,1].set_title('Kurzperiode: $\\omega_n$ und Dämpfung $\\zeta$')
    axes[0,1].grid(True, alpha=0.3); axes[0,1].legend(fontsize=9)

    # Pol-Nullstellen-Diagramm (vereinfacht)
    zeta_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    for z in zeta_vals:
        theta = np.linspace(np.pi/2, np.pi, 100)
        r = 2.0
        xp = r*np.cos(theta)
        yp = r*np.sin(theta)
        axes[1,0].plot(xp*(1/z if z>0 else 1), yp, color=GRAY, lw=0.8, alpha=0.5)

    # SP poles at different M
    M_sel = [1.5, 2.5, 3.5, 4.5]
    colors_sel = [BLUE, GREEN, ORANGE, RED]
    for M_s, c_s in zip(M_sel, colors_sel):
        idx = np.argmin(abs(M_arr - M_s))
        om = omega_sp[idx]; ze = zeta_sp[idx]
        real_p = -ze*om; imag_p = om*np.sqrt(max(1-ze**2, 0))
        axes[1,0].plot(real_p, imag_p, 'o', color=c_s, ms=9, label=f'M={M_s}')
        axes[1,0].plot(real_p, -imag_p, 'o', color=c_s, ms=9)

    axes[1,0].axvline(0, color='k', lw=1.5)
    axes[1,0].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[1,0].set_xlabel('Re($s$) [1/s]'); axes[1,0].set_ylabel('Im($s$) [rad/s]')
    axes[1,0].set_title('Polplot Kurzperiode')
    axes[1,0].legend(fontsize=9); axes[1,0].grid(True, alpha=0.3)

    # Dutch Roll stability
    axes[1,1].plot(M_arr, omega_dr, color=CYAN, lw=2.5, label='$\\omega_{DR}$')
    axes[1,1].plot(M_arr, zeta_dr*omega_dr, color=RED, lw=2, ls='--', label='$\\zeta_{DR}\\,\\omega_{DR}$ (MIL-Grenzwert)')
    axes[1,1].axhline(0.35*1.0, color=RED, ls=':', lw=1.5, label='MIL-SPEC $\\zeta\\,\\omega \\geq 0.35$')
    axes[1,1].set_xlabel('Mach $M$'); axes[1,1].set_ylabel('[rad/s]')
    axes[1,1].set_title('Dutch-Roll Stabilität')
    axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Dynamische Stabilität des SR-91 — Längs- und Lateraldynamik', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig10_stability.pdf', bbox_inches='tight')
    plt.close()
    print("fig10 done")

# -----------------------------------------------------------------------
# 11. Radar-Cross-Section (RCS) Winkelkarte
# -----------------------------------------------------------------------
def fig_rcs():
    theta = np.linspace(0, 2*np.pi, 1000)
    # Typische RCS-Charakteristik eines Stealthflugzeugs (Deltakonfiguration)
    # Stark gerichtet: Frontal sehr klein, seitlich größer
    def rcs_dBsm(theta_rad):
        theta_deg = np.rad2deg(theta_rad) % 360
        # Frontal (±15°): -20 dBsm
        # Seite (90°): +5 dBsm
        # Heck (±15° von 180°): -15 dBsm
        base = -20 + 25*np.abs(np.sin(theta_rad))**2
        spike_side = 8*np.exp(-((theta_deg-90)**2)/400) + 8*np.exp(-((theta_deg-270)**2)/400)
        spike_rear = 5*np.exp(-((theta_deg-180)**2)/200)
        noise = 2*np.random.RandomState(42).randn(len(theta_rad) if hasattr(theta_rad, '__len__') else 1)
        return base + spike_side + spike_rear

    theta_arr = np.linspace(0, 2*np.pi, 1000)
    rcs = rcs_dBsm(theta_arr)

    fig = plt.figure(figsize=(13, 5))

    # Polar plot
    ax1 = fig.add_subplot(121, polar=True)
    rcs_lin = 10**(rcs/10)  # m^2
    ax1.plot(theta_arr, np.log10(rcs_lin+0.001)+3, color=BLUE, lw=1.5)
    ax1.fill(theta_arr, np.log10(rcs_lin+0.001)+3, alpha=0.2, color=BLUE)
    ax1.set_title('RCS-Polardiagramm [log-skaliert]\n(Ansicht von oben, Nase = 0°)', pad=15)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)

    # Kartesisch
    ax2 = fig.add_subplot(122)
    ax2.plot(np.rad2deg(theta_arr), rcs, color=RED, lw=2)
    ax2.axhline(-20, color=GREEN, ls='--', lw=1.5, label='Frontaler Minimalwert (−20 dBsm)')
    ax2.axhline(5, color=ORANGE, ls='--', lw=1.5, label='Seitlicher Maximalwert (+5 dBsm)')
    ax2.set_xlabel('Azimutwinkel [°]'); ax2.set_ylabel('RCS [dBsm]')
    ax2.set_title('Radar-Querschnitt RCS(θ)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0,45,90,135,180,225,270,315,360])

    plt.suptitle('Radar-Querschnitt (RCS) des SR-91 bei $f=10$ GHz', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig11_rcs.pdf', bbox_inches='tight')
    plt.close()
    print("fig11 done")

# -----------------------------------------------------------------------
# 12. Flugzustandsflaeche (Mach-Hoehe-Diagramm mit Grenzen)
# -----------------------------------------------------------------------
def fig_flight_envelope():
    H_km = np.linspace(0, 36, 400)
    g = 9.81; R_gas = 287.05; lam = 0.0065; T0 = 288.15; gamma_air = 1.4

    def atmo(h):
        if h <= 11000:
            T = T0 - lam*h; p = 101325*(T/T0)**(g/(lam*R_gas))
        else:
            T = 216.65; p = 22632*np.exp(-g*(h-11000)/(R_gas*T))
        rho = p/(R_gas*T); a = np.sqrt(gamma_air*R_gas*T)
        return rho, a, p

    S = 78.5; m_clean = 28000  # leichtes Kfzgewicht
    CL_max = 1.2; CD0_min = 0.012; AR = 1.8

    M_stall = []; M_max_struct = []; M_max_thrust = []; M_max_T = []
    for h in H_km:
        rho, a, p = atmo(h*1000)
        # Stall: q*S*CLmax = mg
        q_stall = m_clean*g / (S*CL_max)
        V_stall = np.sqrt(2*q_stall/rho)
        M_stall.append(V_stall/a)
        # Thermische Grenze: T_aw = 600°C → Mach?
        T_limit_C = 600
        T_H = T0 - lam*min(h*1000, 11000)
        # T0_stag = T*(1+(g-1)/2*M^2) = T_limit+273
        M_th = np.sqrt(((T_limit_C+273)/T_H - 1)*2/(gamma_air-1))
        M_max_T.append(min(M_th, 6.5))
        # Strukturgrenzlast
        M_max_struct.append(6.5)  # Konstruktiv begrenzt

    M_stall = np.array(M_stall)
    M_max_T = np.array(M_max_T)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.fill_betweenx(H_km, M_stall, M_max_T, where=M_max_T>M_stall,
                     alpha=0.2, color=BLUE, label='Zulässiger Flugbereich')
    ax.plot(M_stall, H_km, color=RED, lw=2.5, label='Abrissgrenze (Stall)')
    ax.plot(M_max_T, H_km, color=ORANGE, lw=2.5, label='Therm. Grenze ($T_{aw}=600°C$)')
    ax.axhline(20, color=GRAY, ls=':', lw=1.5, label='Typ. Reiseflug $H=20$ km')
    ax.axhline(30, color=GRAY, ls='--', lw=1.5, label='Max-Betriebshöhe $H=30$ km')
    ax.axvline(3.5, color=GREEN, ls=':', lw=1.8, label='Reisemach M=3.5')

    # Punkte einzeichnen
    ax.plot(3.5, 20, 'D', color=GREEN, ms=10, zorder=5, label='Auslegungspunkt (M3.5, 20km)')
    ax.plot(5.0, 28, '*', color=ORANGE, ms=14, zorder=5, label='Max-Punkt (M5.0, 28km)')

    ax.set_xlabel('Mach-Zahl $M$')
    ax.set_ylabel('Flughöhe $H$ [km]')
    ax.set_title('Flugzustandshüllkurve des SR-91\n(ISA, saubere Konfiguration, $m=28$ Mg)')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7); ax.set_ylim(0, 36)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig12_envelope.pdf', bbox_inches='tight')
    plt.close()
    print("fig12 done")

if __name__ == '__main__':
    fig_thrust_mach()
    fig_polar()
    fig_glide()
    fig_turn()
    fig_Vn()
    fig_climb()
    fig_heating()
    fig_pilot_loads()
    fig_recon_envelope()
    fig_stability()
    fig_rcs()
    fig_flight_envelope()
    print("Alle Figuren erstellt.")
