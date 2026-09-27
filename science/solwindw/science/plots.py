#!/usr/bin/env python3
"""
Matplotlib-Plots für die wissenschaftliche Arbeit:
"Transparente Photovoltaik-Verglasung: Solare Energiegewinnung durch
Fenster-integrierte Dioden-Netzwerke"
Autor: Stephan Epp, Universität Bielefeld
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# Farbschema (passend zu subgraph.tex: teal/blau)
FARBE_HAUPT   = '#006666'   # teal
FARBE_AKZENT  = '#008080'
FARBE_HELL    = '#4db8b8'
FARBE_DUNKEL  = '#004444'
FARBE_ORANGE  = '#e07b00'
FARBE_ROT     = '#cc2200'
FARBE_GRAU    = '#555555'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT = 'plots/'

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: AM1.5-Solarspektrum + Absorptionsbereiche
# ─────────────────────────────────────────────────────────────────────────────
def plot_spektrum():
    fig, ax = plt.subplots(figsize=(9, 5))

    # Vereinfachtes AM1.5-Spektrum (W/m²/nm)
    lam = np.linspace(280, 2500, 2000)

    def am15(l):
        # Planck mit T=5778 K, normiert
        h, c, k = 6.626e-34, 3e8, 1.38e-23
        T = 5778
        lm = l * 1e-9
        spec = (2*h*c**2 / lm**5) / (np.exp(h*c/(lm*k*T)) - 1)
        return spec

    irr = am15(lam)
    irr = irr / irr.max() * 1.8  # normiert auf ~1.8 kW/m²/µm

    ax.fill_between(lam, irr, where=(lam < 380),
                    alpha=0.35, color='purple', label='UV (< 380 nm)')
    ax.fill_between(lam, irr, where=((lam >= 380) & (lam <= 780)),
                    alpha=0.45, color=FARBE_HELL, label='Sichtbares Licht (380–780 nm)')
    ax.fill_between(lam, irr, where=(lam > 780),
                    alpha=0.25, color=FARBE_ORANGE, label='Infrarot (> 780 nm)')
    ax.plot(lam, irr, color=FARBE_DUNKEL, lw=1.2)

    # PV-Absorptionsfenster (transparente Zellen)
    ax.axvspan(780, 1100, alpha=0.15, color=FARBE_ROT,
               label='NIR-Absorptions-Bereich transparenter PV')
    ax.axvline(780, color=FARBE_ROT, lw=1.0, ls='--', alpha=0.7)
    ax.axvline(1100, color=FARBE_ROT, lw=1.0, ls='--', alpha=0.7)

    ax.set_xlabel('Wellenlänge $\\lambda$ [nm]')
    ax.set_ylabel('Spektrale Bestrahlungsstärke [kW m$^{-2}$ µm$^{-1}$]')
    ax.set_title('AM1.5-Solarspektrum und Absorptionsbereiche transparenter PV-Dioden')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(280, 2500)
    ax.set_ylim(0, 2.1)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot1_spektrum.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot1_spektrum.png', bbox_inches='tight')
    plt.close()
    print("Plot 1 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: I-U-Kennlinie der transparenten Solarzelle + Leistungskurve
# ─────────────────────────────────────────────────────────────────────────────
def plot_iu_kennlinie():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Parameter (Einzel-Dioden-Modell, One-Diode-Model)
    I_ph  = 8.5e-3   # A  Photostrom
    I_0   = 1e-10    # A  Sättigungsstrom
    n     = 1.4      # Idealitätsfaktor
    V_T   = 0.02585  # V  Thermische Spannung (T=300K)
    R_s   = 2.5      # Ω  Serienwiderstand
    R_sh  = 1000     # Ω  Parallelwiderstand

    def I_zelle(V, I_ph=I_ph, I_0=I_0, n=n, V_T=V_T, R_s=R_s, R_sh=R_sh):
        def f(I):
            return I - I_ph + I_0*(np.exp((V + I*R_s)/(n*V_T)) - 1) + (V + I*R_s)/R_sh
        sol = fsolve(f, I_ph/2, full_output=False)
        return float(sol[0])

    V_arr = np.linspace(0, 0.65, 300)
    I_arr = np.array([I_zelle(v) for v in V_arr])
    P_arr = V_arr * I_arr

    # MPP berechnen
    idx_mpp = np.argmax(P_arr)
    V_mpp, I_mpp, P_mpp = V_arr[idx_mpp], I_arr[idx_mpp], P_arr[idx_mpp]
    V_oc = V_arr[np.where(I_arr <= 0)[0][0]] if any(I_arr <= 0) else 0.62
    I_sc = I_arr[0]

    ax1.plot(V_arr, I_arr * 1000, color=FARBE_HAUPT, lw=2.0, label='$I(V)$')
    ax1.axvline(V_mpp, color=FARBE_ROT, ls='--', lw=1.2, alpha=0.8)
    ax1.axhline(I_mpp*1000, color=FARBE_ROT, ls='--', lw=1.2, alpha=0.8)
    ax1.scatter([V_mpp], [I_mpp*1000], color=FARBE_ROT, zorder=5,
                label=f'MPP ($V_{{\\rm MPP}}$={V_mpp:.3f} V, $I_{{\\rm MPP}}$={I_mpp*1000:.2f} mA)')
    ax1.scatter([0], [I_sc*1000], color=FARBE_ORANGE, zorder=5,
                label=f'$I_{{\\rm SC}}$ = {I_sc*1000:.2f} mA')
    ax1.scatter([V_oc], [0], color=FARBE_DUNKEL, zorder=5,
                label=f'$V_{{\\rm OC}}$ = {V_oc:.3f} V')

    ax1.set_xlabel('Spannung $V$ [V]')
    ax1.set_ylabel('Strom $I$ [mA]')
    ax1.set_title('I-U-Kennlinie der transparenten PV-Diode')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 0.68)
    ax1.set_ylim(-0.5, 10)

    ax2.plot(V_arr, P_arr * 1000, color=FARBE_AKZENT, lw=2.0, label='$P(V)$')
    ax2.axvline(V_mpp, color=FARBE_ROT, ls='--', lw=1.2, alpha=0.8)
    ax2.scatter([V_mpp], [P_mpp*1000], color=FARBE_ROT, zorder=5,
                label=f'$P_{{\\rm MPP}}$ = {P_mpp*1000:.3f} mW')
    ax2.fill_between(V_arr, P_arr*1000, alpha=0.15, color=FARBE_HELL)
    ax2.set_xlabel('Spannung $V$ [V]')
    ax2.set_ylabel('Leistung $P$ [mW]')
    ax2.set_title('Leistungskurve der transparenten PV-Diode')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 0.68)

    FF = P_mpp / (V_oc * I_sc)
    eta = P_mpp / (0.1 * 1000)  # 1 cm² Zelle, 1000 W/m²
    fig.suptitle(f'Füll-Faktor FF = {FF:.3f},   Wirkungsgrad $\\eta$ = {eta*100:.1f}%',
                 fontsize=12, y=1.01, color=FARBE_DUNKEL)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot2_iu_kennlinie.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot2_iu_kennlinie.png', bbox_inches='tight')
    plt.close()
    print("Plot 2 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Energieertragsmodell – Jahressimulation Bielefeld
# ─────────────────────────────────────────────────────────────────────────────
def plot_jahresertrag():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Tagessumme der Globalstrahlung Bielefeld (kWh/m²/Tag) – Monatsmittel
    monate = np.arange(1, 13)
    mon_namen = ['Jan','Feb','Mär','Apr','Mai','Jun',
                 'Jul','Aug','Sep','Okt','Nov','Dez']
    H_monthly = np.array([0.62, 1.08, 2.18, 3.65, 4.82, 5.10,
                           4.90, 4.45, 3.12, 1.85, 0.75, 0.50])  # kWh/m²/d

    # Fensterausrichtung: Süd, West, Ost, Nord
    faktoren = {'Süd': 1.0, 'West': 0.72, 'Ost': 0.72, 'Nord': 0.35}
    farben = [FARBE_HAUPT, FARBE_AKZENT, FARBE_ORANGE, FARBE_GRAU]

    # Wirkungsgrad transparente Zelle: 8%
    eta = 0.08
    # Fenster-Fläche: 2 m² (Standard-Bürofenster)
    A_f = 2.0
    # Tage/Monat
    days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    ax1 = axes[0]
    ax2 = axes[1]

    for (ausrichtung, faktor), farbe in zip(faktoren.items(), farben):
        E_monat = H_monthly * faktor * eta * A_f * days  # kWh/Monat
        E_kum   = np.cumsum(E_monat)
        bar_off = list(faktoren.keys()).index(ausrichtung) * 0.18 - 0.27
        ax1.bar(monate + bar_off, E_monat, width=0.18, color=farbe,
                alpha=0.85, label=f'{ausrichtung} ({E_monat.sum():.1f} kWh/a)')
        ax2.plot(monate, E_kum, marker='o', ms=4, color=farbe,
                 lw=1.8, label=ausrichtung)

    ax1.set_xlabel('Monat')
    ax1.set_ylabel('Energieertrag $E_m$ [kWh/Monat]')
    ax1.set_title('Monatlicher Energieertrag (2 m² Fensterfläche, $\\eta=8\\%$, Bielefeld)')
    ax1.set_xticks(monate)
    ax1.set_xticklabels(mon_namen)
    ax1.legend(framealpha=0.9, ncol=2)

    ax2.set_xlabel('Monat')
    ax2.set_ylabel('Kumulierter Ertrag $E_{\\rm kum}$ [kWh]')
    ax2.set_title('Kumulierter Jahresertrag nach Ausrichtung')
    ax2.set_xticks(monate)
    ax2.set_xticklabels(mon_namen)
    ax2.legend(framealpha=0.9, ncol=2)
    ax2.grid(axis='y', ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot3_jahresertrag.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot3_jahresertrag.png', bbox_inches='tight')
    plt.close()
    print("Plot 3 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Wirkungsgrad & Transmissionsgrad als Funktion der Diodendichte
# ─────────────────────────────────────────────────────────────────────────────
def plot_eta_tau():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Diodendichte ρ: Dioden pro cm²
    rho = np.linspace(0, 50, 500)

    # Flächenbedeckungsgrad φ = ρ * A_diode
    A_diode = 0.008   # cm² pro Diode (0.8 mm × 1.0 mm)
    phi = np.minimum(rho * A_diode, 1.0)

    # Wirkungsgrad: η = φ * η_zelle (Diode hat 12% Wirkungsgrad)
    eta_zelle = 0.12
    eta_ges = phi * eta_zelle

    # Transmissionsgrad: τ = 1 - φ * (1 - τ_diode)
    tau_diode = 0.05   # undurchsichtige Diodenfläche
    tau_glas  = 0.90   # Glas allein
    tau_ges = tau_glas * (1 - phi) + tau_glas * phi * tau_diode

    ax1.plot(rho, eta_ges * 100, color=FARBE_HAUPT, lw=2.2, label='$\\eta_{\\rm ges}(\\rho)$')
    ax1.fill_between(rho, eta_ges*100, alpha=0.12, color=FARBE_HELL)
    ax1.set_xlabel('Diodendichte $\\rho$ [Dioden/cm$^2$]')
    ax1.set_ylabel('Systemwirkungsgrad $\\eta_{\\rm ges}$ [%]')
    ax1.set_title('Wirkungsgrad als Funktion der Diodendichte')
    ax1.legend()

    ax2.plot(rho, tau_ges * 100, color=FARBE_AKZENT, lw=2.2, label='$\\tau_{\\rm ges}(\\rho)$')
    ax2.axhline(40, color=FARBE_ROT, ls='--', lw=1.2,
                label='Min. Tageslicht ($\\tau = 40\\%$)')
    ax2.axhline(70, color=FARBE_ORANGE, ls='--', lw=1.2,
                label='Komfort-Bereich ($\\tau = 70\\%$)')

    # Optimaler Bereich hervorheben
    idx_low = np.argmin(np.abs(tau_ges - 0.40))
    idx_high= np.argmin(np.abs(tau_ges - 0.70))
    ax2.axvspan(rho[idx_high], rho[idx_low], alpha=0.15, color=FARBE_HELL,
                label='Optimaler Dichtebereich')

    ax2.set_xlabel('Diodendichte $\\rho$ [Dioden/cm$^2$]')
    ax2.set_ylabel('Transmissionsgrad $\\tau_{\\rm ges}$ [%]')
    ax2.set_title('Transmissionsgrad als Funktion der Diodendichte')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot4_eta_tau.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot4_eta_tau.png', bbox_inches='tight')
    plt.close()
    print("Plot 4 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Serien- vs. Parallelschaltung von Dioden-Strings
# ─────────────────────────────────────────────────────────────────────────────
def plot_verschaltung():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    I_ph_cell = 8.5e-3
    I_0  = 1e-10
    n    = 1.4
    V_T  = 0.02585

    def I_cell_simple(V):
        return I_ph_cell - I_0 * (np.exp(V / (n * V_T)) - 1)

    V_1 = np.linspace(0, 0.62, 500)
    I_1 = np.clip(I_cell_simple(V_1), 0, None)

    configs = [
        (1, 1, 'Einzelzelle'),
        (4, 1, '4 Seriell'),
        (1, 4, '4 Parallel'),
        (4, 4, '4S×4P Matrix'),
    ]
    farb_list = [FARBE_DUNKEL, FARBE_HAUPT, FARBE_AKZENT, FARBE_ORANGE]

    for ax_idx, ax in enumerate(axes):
        for (Ns, Np, label), farbe in zip(configs, farb_list):
            V_mod = V_1 * Ns
            I_mod = I_1 * Np
            P_mod = V_mod * I_mod
            if ax_idx == 0:
                ax.plot(V_mod, I_mod * 1000, color=farbe, lw=1.8, label=label)
            elif ax_idx == 1:
                ax.plot(V_mod, P_mod * 1000, color=farbe, lw=1.8, label=label)
            else:
                # Kennlinienfeld bei 25% / 50% / 75% / 100% Einstrahlung
                pass

        if ax_idx == 0:
            ax.set_xlabel('Modulspannung $V$ [V]')
            ax.set_ylabel('Strom $I$ [mA]')
            ax.set_title('I-V-Kennlinien Verschaltungskonfigurationen')
            ax.legend(fontsize=9)
        elif ax_idx == 1:
            ax.set_xlabel('Modulspannung $V$ [V]')
            ax.set_ylabel('Leistung $P$ [mW]')
            ax.set_title('Leistungskurven Verschaltungskonfigurationen')
            ax.legend(fontsize=9)
        else:
            # Partielle Verschattung
            for frac, farbe in zip([1.0, 0.75, 0.50, 0.25], farb_list):
                I_ph_eff = I_ph_cell * frac
                I_s = np.clip(I_ph_eff - I_0 * (np.exp(V_1/(n*V_T)) - 1), 0, None)
                ax.plot(V_1, I_s * 1000, color=farbe, lw=1.8,
                        label=f'$G/G_0 = {int(frac*100)}\\%$')
            ax.set_xlabel('Zellspannung $V$ [V]')
            ax.set_ylabel('Strom $I$ [mA]')
            ax.set_title('Einfluss partieller Verschattung')
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot5_verschaltung.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot5_verschaltung.png', bbox_inches='tight')
    plt.close()
    print("Plot 5 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: Thermische Analyse – Wärmeleitung im Verbundglas
# ─────────────────────────────────────────────────────────────────────────────
def plot_thermisch():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Temperaturprofil durch Verbundglasaufbau
    # Schichten: Außenglas 4mm | EVA 0.76mm | PV-Dioden | EVA 0.76mm | Innenglas 4mm
    d_layers = [0, 4, 4.76, 5.52, 10.52, 10.52+4]  # mm kumulative Dicke
    labels   = ['Außenglas', 'EVA₁', 'Dioden', 'EVA₂', 'Innenglas']
    lambdas  = [1.0, 0.20, 148.0, 0.20, 1.0]   # W/(m·K) Wärmeleitfähigkeit
    Q = 50   # W/m² Wärmefluss durch Glas bei Sonneneinstrahlung

    T_aussen = 35   # °C Außentemperatur (Sommer, Sonne)
    T_arr = [T_aussen]
    x_arr = [d_layers[0]]
    for i, (lam, d) in enumerate(zip(lambdas, np.diff(d_layers))):
        dT = Q * (d * 1e-3) / lam
        T_arr.append(T_arr[-1] - dT)
        x_arr.append(d_layers[i+1])

    farben_schichten = ['#aaccff', '#ddeecc', '#ffcc88', '#ddeecc', '#aaccff']
    for i in range(len(labels)):
        ax1.axvspan(d_layers[i], d_layers[i+1],
                    alpha=0.25, color=farben_schichten[i], label=labels[i])
    ax1.plot(x_arr, T_arr, color=FARBE_ROT, lw=2.2, marker='o', ms=5)
    ax1.set_xlabel('Position $x$ [mm]')
    ax1.set_ylabel('Temperatur $T$ [°C]')
    ax1.set_title('Temperaturprofil im Verbundglas-Querschnitt')
    ax1.legend(fontsize=9, loc='upper right')

    # Betriebstemperatur der Dioden in Abhängigkeit der Einstrahlung
    G = np.linspace(0, 1200, 300)   # W/m²
    T_noct = 45   # NOCT-Temperatur
    T_umg  = 20
    # T_zelle = T_umg + (T_NOCT - 20) * G / 800
    T_cell_sommer = T_umg + 15 + (T_noct - 20) * G / 800
    T_cell_winter = (T_umg - 10) + (T_noct - 20) * G / 800

    eta_0    = 0.12
    beta_T   = 0.0045    # 1/K Temperaturkoeffizient (typisch für c-Si)
    T_ref    = 25.0

    eta_sommer = eta_0 * (1 - beta_T * (T_cell_sommer - T_ref))
    eta_winter = eta_0 * (1 - beta_T * (T_cell_winter - T_ref))

    ax2.plot(G, eta_sommer * 100, color=FARBE_ROT,   lw=2.0, label='Sommer ($T_{\\rm Umg}=35°C$)')
    ax2.plot(G, eta_winter * 100, color=FARBE_HAUPT, lw=2.0, label='Winter ($T_{\\rm Umg}=10°C$)')
    ax2.axhline(eta_0*100, color=FARBE_GRAU, ls='--', lw=1.2, label='Referenz (STC, 25°C)')
    ax2.fill_between(G, eta_sommer*100, eta_winter*100, alpha=0.12, color=FARBE_HELL)
    ax2.set_xlabel('Einstrahlung $G$ [W/m$^2$]')
    ax2.set_ylabel('Wirkungsgrad $\\eta$ [%]')
    ax2.set_title('Wirkungsgrad-Temperaturdrift')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot6_thermisch.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot6_thermisch.png', bbox_inches='tight')
    plt.close()
    print("Plot 6 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7: Wirtschaftlichkeitsanalyse – Amortisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_wirtschaftlichkeit():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    Jahre = np.arange(0, 26)

    # Szenarien
    szenarien = {
        'Bürogebäude, Süd (16 m²)': {
            'E_a': 16 * 0.08 * 1100 * 0.85,  # kWh/a: Fläche * η * H * PR
            'C_inv': 16 * 480,                 # € Investition: 480 €/m²
            'C_inst': 2000,
        },
        'Wohnhaus, Süd-West (6 m²)': {
            'E_a': 6 * 0.08 * 980 * 0.82,
            'C_inv': 6 * 520,
            'C_inst': 800,
        },
        'Fassade, Ost (12 m²)': {
            'E_a': 12 * 0.08 * 750 * 0.80,
            'C_inv': 12 * 460,
            'C_inst': 1500,
        },
    }

    p_elekt = 0.32     # €/kWh Strompreis
    preis_steig = 0.03 # 3% jährliche Preissteigerung
    p_O_M  = 0.005     # 0.5% Betriebskosten p.a.
    farb_sz = [FARBE_HAUPT, FARBE_AKZENT, FARBE_ORANGE]

    for (name, par), farbe in zip(szenarien.items(), farb_sz):
        C0 = par['C_inv'] + par['C_inst']
        cash_flow = np.zeros(len(Jahre))
        cash_flow[0] = -C0
        for j in range(1, len(Jahre)):
            ertrag = par['E_a'] * p_elekt * (1 + preis_steig)**j
            O_M    = C0 * p_O_M
            cash_flow[j] = ertrag - O_M
        npv_kum = np.cumsum(cash_flow)

        ax1.plot(Jahre, npv_kum, color=farbe, lw=2.0, marker='', label=name)

    ax1.axhline(0, color='black', lw=1.0, ls='-')
    ax1.fill_between(Jahre, 0, where=(Jahre >= 0), alpha=0.05, color='green')
    ax1.set_xlabel('Jahr')
    ax1.set_ylabel('Kumulierter Cashflow [€]')
    ax1.set_title('Amortisationsanalyse Fenster-PV-Systeme')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 25)

    # Sensitivitätsanalyse: LCOE vs. Wirkungsgrad
    eta_range = np.linspace(0.04, 0.18, 200)
    H_full = 1100   # kWh/(m²·a) Süd
    PR = 0.85
    C_inv_m2 = 480  # €/m²
    C_inst_m2 = 80
    LT = 25         # Lebensdauer
    r  = 0.04       # Diskontrate

    CRF = r * (1+r)**LT / ((1+r)**LT - 1)
    LCOE = (C_inv_m2 + C_inst_m2) * CRF / (eta_range * H_full * PR)

    ax2.plot(eta_range * 100, LCOE, color=FARBE_HAUPT, lw=2.2)
    ax2.axhline(0.32, color=FARBE_ROT, ls='--', lw=1.2, label='Strompreis 0.32 €/kWh')
    ax2.axhline(0.15, color=FARBE_ORANGE, ls='--', lw=1.2,
                label='EEG-Einspeisevergütung 0.15 €/kWh')
    ax2.fill_between(eta_range*100, LCOE, 0.32,
                     where=(LCOE <= 0.32), alpha=0.15, color='green',
                     label='Wirtschaftlicher Betrieb')
    ax2.set_xlabel('Wirkungsgrad $\\eta$ [%]')
    ax2.set_ylabel('LCOE [€/kWh]')
    ax2.set_title('LCOE als Funktion des Wirkungsgrads')
    ax2.legend(fontsize=9)
    ax2.set_xlim(4, 18)
    ax2.set_ylim(0, 2.0)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot7_wirtschaftlichkeit.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot7_wirtschaftlichkeit.png', bbox_inches='tight')
    plt.close()
    print("Plot 7 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8: Netzwerktopologie der elektrischen Verbindungen
# ─────────────────────────────────────────────────────────────────────────────
def plot_netzwerk():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Dioden-Gitter: 5×4
    Nx, Ny = 5, 4
    xs = np.linspace(1.5, 8.5, Nx)
    ys = np.linspace(2.0, 6.5, Ny)

    # Horizontale Sammelschienen
    for y in [1.3, 7.2]:
        ax.plot([1.0, 9.0], [y, y], color=FARBE_DUNKEL, lw=3.0)
    ax.text(0.4, 1.3, '$-$', fontsize=16, color=FARBE_DUNKEL, va='center', ha='center')
    ax.text(0.4, 7.2, '$+$', fontsize=16, color=FARBE_ROT,   va='center', ha='center')
    ax.text(9.7, 4.3, 'Byp.-\nDiode', fontsize=8, color=FARBE_GRAU, ha='center')

    # Strings (Spalten): jede Spalte in Reihe
    for ix, x in enumerate(xs):
        # Vertikale Leiterbahn
        ax.plot([x, x], [1.5, 7.0], color=FARBE_HELL, lw=1.0, ls='--', alpha=0.5)
        for iy, y in enumerate(ys):
            # Diode als Rechteck
            rect = patches.FancyBboxPatch((x-0.25, y-0.18), 0.50, 0.36,
                                          boxstyle='round,pad=0.04',
                                          linewidth=1.2,
                                          edgecolor=FARBE_DUNKEL,
                                          facecolor=FARBE_HELL,
                                          alpha=0.85)
            ax.add_patch(rect)
            ax.text(x, y, f'D$_{{{ix*Ny+iy+1}}}$', ha='center', va='center',
                    fontsize=7, color=FARBE_DUNKEL, fontweight='bold')
        # Verbindung zur Sammelschiene
        ax.plot([x, x], [1.3, ys[0]-0.18], color=FARBE_DUNKEL, lw=1.5)
        ax.plot([x, x], [ys[-1]+0.18, 7.2], color=FARBE_ROT,   lw=1.5)

    # Bypass-Diode rechts
    ax.annotate('', xy=(9.4, 7.0), xytext=(9.4, 1.6),
                arrowprops=dict(arrowstyle='->', color=FARBE_ROT, lw=1.5))
    ax.plot([9.0, 9.8], [4.25, 4.25], color=FARBE_ROT, lw=1.5)
    rect_bp = patches.FancyBboxPatch((9.05, 4.1), 0.70, 0.30,
                                      boxstyle='round,pad=0.04',
                                      linewidth=1.2,
                                      edgecolor=FARBE_ROT,
                                      facecolor='#ffcccc')
    ax.add_patch(rect_bp)
    ax.text(9.4, 4.25, 'BP', ha='center', va='center', fontsize=8, color=FARBE_ROT)

    # Lastanschluss
    ax.annotate('', xy=(5.0, 0.5), xytext=(5.0, 1.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(5.0, 0.2, '$R_{\\rm Last}$ / MPPT', ha='center', fontsize=10, color='black')

    ax.set_title('Netzwerktopologie: 5×4 Dioden-Matrix im Fensterglas\n'
                 '(Serien-Strings mit Bypass-Dioden)', fontsize=12, color=FARBE_DUNKEL)

    plt.tight_layout()
    plt.savefig(OUTPUT + 'plot8_netzwerk.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT + 'plot8_netzwerk.png', bbox_inches='tight')
    plt.close()
    print("Plot 8 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Alle Plots generieren
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generiere Plots ...")
    plot_spektrum()
    plot_iu_kennlinie()
    plot_jahresertrag()
    plot_eta_tau()
    plot_verschaltung()
    plot_thermisch()
    plot_wirtschaftlichkeit()
    plot_netzwerk()
    print("\nAlle 8 Plots erfolgreich generiert.")
