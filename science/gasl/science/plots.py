#!/usr/bin/env python3
"""
Erzeuge alle Matplotlib-Abbildungen fuer die wissenschaftliche Arbeit:
'Systementscheidung fuer kleine Rohrdurchmesser (8-11 cm) gegenueber
 grossen Rohrdurchmessern (30-45 cm) in Gasleitungsnetzwerken'

Aufruf:  python3 generate_plots.py
Ausgabe: fig01_moody.pdf  ...  fig08_geschwindigkeitsprofil.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib import cm
import scipy.optimize as opt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Globale Plot-Einstellungen (LaTeX-aehnliches Erscheinungsbild)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.dpi":      200,
    "lines.linewidth":  2.0,
    "grid.alpha":       0.4,
})

# ---------------------------------------------------------------------------
# Physikalische Konstanten fuer Methan bei 2 bar abs., 20 degC
# ---------------------------------------------------------------------------
MU     = 1.10e-5   # dynamische Viskositaet  [Pa s]
RHO    = 1.436     # Dichte                  [kg m^-3]
NU     = MU / RHO  # kinematische Viskositaet [m^2 s^-1]
K_GAS  = 0.033     # Waermeleitfaehigkeit    [W m^-1 K^-1]
PR     = 0.72      # Prandtl-Zahl            [-]
CP_GAS = 2232.0    # spez. Waermekapazitaet  [J kg^-1 K^-1]
EPSILON = 4.6e-5   # Absolutrauheit Stahl    [m]

# Rohrdurchmesser
D_SMALL = np.array([0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110])  # 8-11 cm
D_LARGE = np.array([0.300, 0.450])                                        # 30, 45 cm
D_REP   = 0.095   # repraesentativer kleiner Durchmesser = 9.5 cm

# Farben
COLOR_SMALL  = "#1f77b4"   # blau
COLOR_30     = "#d62728"   # rot
COLOR_45     = "#ff7f0e"   # orange
COLOR_REGION = "#2ca02c"   # gruen (hervorgehobener Bereich)


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

def colebrook_friction(Re, d, epsilon=EPSILON):
    """Iterative Loesung der Colebrook-White-Gleichung fuer den Reibungsbeiwert."""
    if Re < 2300:
        return 64.0 / Re
    f0 = 0.02
    for _ in range(50):
        rhs = -2.0 * np.log10(epsilon / (3.7 * d) + 2.51 / (Re * np.sqrt(f0)))
        f0 = 1.0 / rhs**2
    return f0

def friction_array(Re_arr, d, epsilon=EPSILON):
    return np.array([colebrook_friction(Re, d, epsilon) for Re in Re_arr])

def pressure_drop(v, d, L=1.0, epsilon=EPSILON):
    """Druckverlust pro Laengeneinheit [Pa m^-1]."""
    Re = v * d / NU
    f  = colebrook_friction(Re, d, epsilon)
    return f * (RHO * v**2 / 2.0) / d

def nusselt_dittus_boelter(Re, Pr=PR, heating=True):
    """Dittus-Boelter-Korrelation fuer turbulente Rohrstroemung."""
    n = 0.4 if heating else 0.3
    if Re < 2300:
        return 3.66   # laminarer Grenzwert
    return 0.023 * Re**0.8 * Pr**n

def heat_transfer_coeff(v, d, Pr=PR, k=K_GAS):
    Re = v * d / NU
    Nu = nusselt_dittus_boelter(Re, Pr)
    return Nu * k / d


# ===========================================================================
# Fig 01 – Moody-Diagramm
# ===========================================================================
def plot_moody():
    fig, ax = plt.subplots(figsize=(10, 7))

    Re_lam  = np.linspace(600, 2300, 300)
    Re_turb = np.logspace(np.log10(4000), 7, 800)

    # Laminare Kurve
    ax.loglog(Re_lam, 64.0 / Re_lam, "k-", lw=2.5, label=r"Laminar: $f = 64/Re$")

    # Turbulente Kurven fuer verschiedene relative Rauheiten
    eps_d_values = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(eps_d_values)))

    for idx, epsd in enumerate(eps_d_values):
        f_arr = []
        for Re in Re_turb:
            if epsd == 0.0:
                # Blasius (glattes Rohr)
                f_arr.append(0.316 * Re**(-0.25) if Re < 1e5 else
                             (-2.0*np.log10(2.51/(Re*np.sqrt(0.01))))**(-2))
            else:
                f0 = 0.02
                for _ in range(50):
                    rhs = -2.0*np.log10(epsd/3.7 + 2.51/(Re*np.sqrt(f0)))
                    f0 = 1.0/rhs**2
                f_arr.append(f0)
        lbl = r"$\varepsilon/d = $" + (f"{epsd:.0e}" if epsd > 0 else "glatt")
        ax.loglog(Re_turb, f_arr, color=palette[idx], label=lbl)

    # Betriebspunkte markieren
    for d_val, label, marker, color in [
        (D_REP,    r"$d=9{,}5\,$cm",  "o", COLOR_SMALL),
        (0.30,     r"$d=30\,$cm",     "s", COLOR_30),
        (0.45,     r"$d=45\,$cm",     "^", COLOR_45),
    ]:
        v_op = 3.0  # m/s Betriebsgeschwindigkeit
        Re_op = v_op * d_val / NU
        f_op  = colebrook_friction(Re_op, d_val)
        ax.loglog(Re_op, f_op, marker=marker, color=color,
                  markersize=10, zorder=5, label=f"Betriebspunkt {label}")

    # Uebergangsbereich schattieren
    ax.axvspan(2300, 4000, alpha=0.15, color="gray", label="Uebergangsbereich")

    ax.set_xlabel(r"Reynolds-Zahl $Re$ $[-]$")
    ax.set_ylabel(r"Darcy-Reibungsbeiwert $f$ $[-]$")
    ax.set_title("Moody-Diagramm: Reibungsbeiwert als Funktion der Reynolds-Zahl")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, which="both")
    ax.set_xlim(600, 1e7)
    ax.set_ylim(0.008, 0.12)
    fig.tight_layout()
    fig.savefig("fig01_moody.pdf")
    plt.close(fig)
    print("fig01_moody.pdf gespeichert")


# ===========================================================================
# Fig 02 – Druckverlust-Vergleich (einzel Rohr)
# ===========================================================================
def plot_pressure_drop_single():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    v_arr = np.linspace(0.05, 15.0, 500)

    # Links: dP/L vs. v fuer drei Durchmesser
    ax = axes[0]
    dp_s  = [pressure_drop(v, D_REP)  for v in v_arr]
    dp_30 = [pressure_drop(v, 0.30)  for v in v_arr]
    dp_45 = [pressure_drop(v, 0.45)  for v in v_arr]

    ax.semilogy(v_arr, dp_s,  color=COLOR_SMALL, label=r"$d=9{,}5\,$cm (kl. Rohr)")
    ax.semilogy(v_arr, dp_30, color=COLOR_30,    label=r"$d=30\,$cm")
    ax.semilogy(v_arr, dp_45, color=COLOR_45,    label=r"$d=45\,$cm")
    ax.axvline(3.0, color="gray", ls="--", lw=1.2, label=r"$v_\mathrm{op}=3\,$m/s")
    ax.set_xlabel(r"Stroemungsgeschwindigkeit $v$ [m/s]")
    ax.set_ylabel(r"Druckverlust pro Meter $\Delta p/L$ [Pa/m]")
    ax.set_title("Einzelrohr-Druckverlust")
    ax.legend()
    ax.grid(True, which="both")

    # Rechts: Parallelnetz (10 kleine vs. 1 grosses 30cm; 23 kleine vs. 1x45cm)
    ax = axes[1]
    Q_arr = np.linspace(1e-4, 0.30, 500)   # Volumenstrom [m^3/s]

    def dp_parallel(Q_total, d_single, N):
        Q_each = Q_total / N
        A      = np.pi * d_single**2 / 4.0
        v_each = Q_each / A
        return pressure_drop(v_each, d_single)

    def dp_one_large(Q_total, d_large):
        A  = np.pi * d_large**2 / 4.0
        v  = Q_total / A
        return pressure_drop(v, d_large)

    # Anzahl kleiner Rohre = gleiche Gesamtquerschnittsflaeche
    N_30 = int(round((0.30 / D_REP)**2))   # ~10
    N_45 = int(round((0.45 / D_REP)**2))   # ~22

    dp_par30 = [dp_parallel(Q, D_REP, N_30)   for Q in Q_arr]
    dp_par45 = [dp_parallel(Q, D_REP, N_45)   for Q in Q_arr]
    dp_gr30  = [dp_one_large(Q, 0.30)         for Q in Q_arr]
    dp_gr45  = [dp_one_large(Q, 0.45)         for Q in Q_arr]

    ax.semilogy(Q_arr*1000, dp_par30, color=COLOR_SMALL, ls="-",
                label=rf"${N_30}\times d=9{{,}}5\,$cm (aeq. $d=30$)")
    ax.semilogy(Q_arr*1000, dp_par45, color=COLOR_SMALL, ls="--",
                label=rf"${N_45}\times d=9{{,}}5\,$cm (aeq. $d=45$)")
    ax.semilogy(Q_arr*1000, dp_gr30,  color=COLOR_30,    ls="-",
                label=r"$1\times d=30\,$cm")
    ax.semilogy(Q_arr*1000, dp_gr45,  color=COLOR_45,    ls="--",
                label=r"$1\times d=45\,$cm")
    ax.set_xlabel(r"Gesamtvolumenstrom $\dot{V}$ [L/s]")
    ax.set_ylabel(r"Druckverlust pro Meter $\Delta p/L$ [Pa/m]")
    ax.set_title(f"Parallelnetz vs. Einzelrohr\n(gleiche Gesamtquerschnittsfläche)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both")

    fig.suptitle("Druckverlustanalyse: Kleine vs. grosse Rohrdurchmesser", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig02_druckverlust.pdf")
    plt.close(fig)
    print("fig02_druckverlust.pdf gespeichert")


# ===========================================================================
# Fig 03 – Reynolds-Zahl als Funktion der Geschwindigkeit und des Durchmessers
# ===========================================================================
def plot_reynolds():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    v_arr = np.linspace(0.01, 15.0, 600)
    d_arr = np.linspace(0.02, 0.55, 600)

    # Links: Re vs v fuer drei Rohre
    ax = axes[0]
    for d_val, lbl, clr in [
        (D_REP, r"$d=9{,}5\,$cm", COLOR_SMALL),
        (0.30,  r"$d=30\,$cm",    COLOR_30),
        (0.45,  r"$d=45\,$cm",    COLOR_45),
    ]:
        Re = v_arr * d_val / NU
        ax.plot(v_arr, Re, color=clr, label=lbl)
    ax.axhline(2300, color="green", ls=":", lw=1.8, label=r"$Re=2300$ (lam.-turb.)")
    ax.axhline(4000, color="purple", ls=":", lw=1.8, label=r"$Re=4000$ (voll turb.)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Stroemungsgeschwindigkeit $v$ [m/s]")
    ax.set_ylabel(r"Reynolds-Zahl $Re$ $[-]$")
    ax.set_title("Re vs. Geschwindigkeit")
    ax.legend()
    ax.grid(True, which="both")
    ax.set_xlim(0, 15)

    # Rechts: 2D-Konturkarte Re(v, d)
    ax = axes[1]
    V_grid, D_grid = np.meshgrid(v_arr, d_arr)
    Re_grid = V_grid * D_grid / NU
    levels = np.logspace(2, 7, 40)
    cs = ax.contourf(V_grid, D_grid * 100, Re_grid, levels=levels,
                     cmap="RdYlGn_r", norm=matplotlib.colors.LogNorm())
    fig.colorbar(cs, ax=ax, label=r"Reynolds-Zahl $Re$ $[-]$")
    # Grenzkurven
    v_lam  = 2300 * NU / d_arr
    v_turb = 4000 * NU / d_arr
    ax.plot(v_lam,  d_arr * 100, "g--", lw=2, label=r"$Re=2300$")
    ax.plot(v_turb, d_arr * 100, "b--", lw=2, label=r"$Re=4000$")
    # Markierungslinien fuer die betrachteten Durchmesser
    ax.axhline(9.5, color=COLOR_SMALL, ls="-",  lw=2, label=r"$d=9{,}5\,$cm")
    ax.axhline(30,  color=COLOR_30,    ls="-",  lw=2, label=r"$d=30\,$cm")
    ax.axhline(45,  color=COLOR_45,    ls="-",  lw=2, label=r"$d=45\,$cm")
    ax.set_xlabel(r"Stroemungsgeschwindigkeit $v$ [m/s]")
    ax.set_ylabel(r"Rohrdurchmesser $d$ [cm]")
    ax.set_title("Stroemungsregime-Karte $Re(v,\,d)$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0.01, 15)
    ax.set_ylim(2, 50)

    fig.suptitle("Reynolds-Zahl-Analyse fuer Gasleitungen", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig03_reynolds.pdf")
    plt.close(fig)
    print("fig03_reynolds.pdf gespeichert")


# ===========================================================================
# Fig 04 – Waermeuebergangszahl (Nusselt / h)
# ===========================================================================
def plot_heat_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    v_arr = np.linspace(0.1, 15.0, 500)

    # N fuer gleiches Gesamtquerschnitt
    N_30 = int(round((0.30 / D_REP)**2))
    N_45 = int(round((0.45 / D_REP)**2))

    # Links: h pro Rohr
    ax = axes[0]
    for d_val, lbl, clr in [
        (D_REP, r"$d=9{,}5\,$cm (einzel)", COLOR_SMALL),
        (0.30,  r"$d=30\,$cm",             COLOR_30),
        (0.45,  r"$d=45\,$cm",             COLOR_45),
    ]:
        h_arr = [heat_transfer_coeff(v, d_val) for v in v_arr]
        ax.plot(v_arr, h_arr, color=clr, label=lbl)
    ax.axvline(3.0, color="gray", ls="--", lw=1.2)
    ax.set_xlabel(r"Stroemungsgeschwindigkeit $v$ [m/s]")
    ax.set_ylabel(r"Waermeuebergangskoeff. $h$ [W m$^{-2}$ K$^{-1}$]")
    ax.set_title("Waermeuebergangskoeffizient pro Rohr")
    ax.legend()
    ax.grid(True)

    # Rechts: Gesamtwaermestrom bei DeltaT = 1 K und L = 1 m
    ax = axes[1]
    Q_flow = np.linspace(1e-4, 0.25, 500)   # Volumenstrom [m^3/s]
    DeltaT = 1.0   # K
    L      = 1.0   # m

    def total_heat(Q_total, d_single, N_pipes):
        Q_each = Q_total / N_pipes
        A      = np.pi * d_single**2 / 4.0
        v_each = Q_each / A
        h      = heat_transfer_coeff(v_each, d_single)
        A_wall = N_pipes * np.pi * d_single * L
        return h * A_wall * DeltaT

    def total_heat_large(Q_total, d_large):
        A  = np.pi * d_large**2 / 4.0
        v  = Q_total / A
        h  = heat_transfer_coeff(v, d_large)
        A_wall = np.pi * d_large * L
        return h * A_wall * DeltaT

    tH_par30 = [total_heat(Q, D_REP, N_30)    for Q in Q_flow]
    tH_par45 = [total_heat(Q, D_REP, N_45)    for Q in Q_flow]
    tH_gr30  = [total_heat_large(Q, 0.30)     for Q in Q_flow]
    tH_gr45  = [total_heat_large(Q, 0.45)     for Q in Q_flow]

    ax.plot(Q_flow*1000, tH_par30, color=COLOR_SMALL, ls="-",
            label=rf"${N_30}\times9{{,}}5\,$cm (aeq. 30\,cm)")
    ax.plot(Q_flow*1000, tH_par45, color=COLOR_SMALL, ls="--",
            label=rf"${N_45}\times9{{,}}5\,$cm (aeq. 45\,cm)")
    ax.plot(Q_flow*1000, tH_gr30,  color=COLOR_30,    ls="-",
            label=r"$1\times30\,$cm")
    ax.plot(Q_flow*1000, tH_gr45,  color=COLOR_45,    ls="--",
            label=r"$1\times45\,$cm")
    ax.set_xlabel(r"Gesamtvolumenstrom $\dot{V}$ [L/s]")
    ax.set_ylabel(r"Gesamtwaermestrom $\dot{Q}$ [W] bei $\Delta T=1\,$K, $L=1\,$m")
    ax.set_title("Vergleich Gesamtwaermestrom (parallele kl. vs. eines gr. Rohrs)")
    ax.legend(fontsize=10)
    ax.grid(True)

    fig.suptitle("Waermeuebergangsanalyse", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig04_waermeuebertragung.pdf")
    plt.close(fig)
    print("fig04_waermeuebertragung.pdf gespeichert")


# ===========================================================================
# Fig 05 – Spezifische Oberflaeche und Oberflaechen-Volumen-Verhaeltnis
# ===========================================================================
def plot_surface_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    d_range = np.linspace(0.01, 0.60, 500)   # m

    # Links: spezifische Oberflaeche = 4/d
    ax = axes[0]
    A_spec = 4.0 / d_range
    ax.plot(d_range * 100, A_spec, "k-", lw=2.5)
    ax.axvspan(8, 11,    alpha=0.25, color=COLOR_SMALL,
               label="Kleiner Bereich (8-11 cm)")
    ax.axvspan(28, 32,   alpha=0.25, color=COLOR_30,
               label="Grosses Rohr 30 cm")
    ax.axvspan(43, 47,   alpha=0.25, color=COLOR_45,
               label="Grosses Rohr 45 cm")
    ax.set_xlabel(r"Rohrdurchmesser $d$ [cm]")
    ax.set_ylabel(r"Spez. Oberflaeche $A_\mathrm{spez} = 4/d$ [m$^{-1}$]")
    ax.set_title("Spezifische Wandoberflaeche pro Gasvolumen")
    ax.legend()
    ax.grid(True)

    # Rechts: Verhaeltnis Kleines/Grosses Rohr fuer gleiches Gesamtquerschnitt
    ax = axes[1]
    d_large_arr = np.linspace(0.15, 0.60, 400)   # grosses Rohr
    d_s = D_REP
    N_arr = (d_large_arr / d_s)**2   # Anzahl kleiner Rohre
    # Gesamtuebertragungsflaeche kl. Rohre / grosse Rohr
    # N * pi * d_s * L  vs  pi * d_L * L --> Verhaeltnis = N*d_s/d_L = d_L/d_s
    ratio = N_arr * d_s / d_large_arr   # = d_large/d_s
    ax.plot(d_large_arr * 100, ratio, "b-", lw=2.5,
            label=r"$\frac{N \cdot d_s}{d_L}=\frac{d_L}{d_s}$")
    ax.axhline(1.0, ls="--", color="gray")
    ax.axvspan(28, 32, alpha=0.2, color=COLOR_30,  label=r"$d_L=30\,$cm")
    ax.axvspan(43, 47, alpha=0.2, color=COLOR_45,  label=r"$d_L=45\,$cm")
    ax.set_xlabel(r"Durchmesser des grossen Rohrs $d_L$ [cm]")
    ax.set_ylabel(r"Oberflaechenverhaeltnis $\frac{A_\mathrm{kl}}{A_\mathrm{gr}}$")
    ax.set_title(rf"Oberflaechenverhaeltnis der Parallelrohre vs. Einzelrohr")
    ax.legend()
    ax.grid(True)

    fig.suptitle(r"Spezifische Oberflaechen: kleine vs. grosse Rohre "
                 rf"($d_s = {D_REP*100:.1f}$\,cm)", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig05_oberflaeche.pdf")
    plt.close(fig)
    print("fig05_oberflaeche.pdf gespeichert")


# ===========================================================================
# Fig 06 – Wirtschaftlichkeitsanalyse (Lebenszykluskosten)
# ===========================================================================
def plot_economics():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    years = np.arange(0, 31)

    # Leitungsparameter
    L_pipe   = 500.0    # m Leitungslaenge
    P_op     = 2e5      # Pa Betriebsdruck
    sigma_s  = 240e6    # Pa Steckgrenze Stahl
    rho_s    = 7850.0   # kg/m^3 Stahldichte
    cost_mat = 1.5      # EUR/kg Stahlerzeugnis (rohe Schaetzung)
    cost_weld= 25.0     # EUR/m Schweisskosten je Rohr
    cost_exc = 200.0    # EUR/m Tiefbaukosten je Trasse (einmalig)
    cost_mnt = 0.005    # Anteil Materialwert / Jahr Wartungskosten
    cost_kWh = 0.15     # EUR/kWh Energiepreis
    pump_eff = 0.70     # Pumpenwirkungsgrad
    op_h     = 6000.0   # Betriebsstunden/Jahr
    Q_gas    = 0.12     # m^3/s Gesamtvolumenstrom

    def wall_thickness(d, P=P_op, sigma=sigma_s, safety=3.0):
        """Barlow-Formel: Mindestwandstaerke [m]"""
        return P * d / (2 * sigma / safety)

    def pipe_mass_per_m(d):
        t = wall_thickness(d)
        return np.pi * d * t * rho_s  # kg/m

    def pump_power(Q_total, d_single, N_pipes, L=L_pipe):
        Q_each = Q_total / N_pipes
        A      = np.pi * d_single**2 / 4.0
        v_each = Q_each / A
        dp_m   = pressure_drop(v_each, d_single)
        P_pump = N_pipes * dp_m * L * Q_each / pump_eff
        return P_pump  # W

    scenarios = {
        r"$10\times d=9{,}5$\,cm": (D_REP, int(round((0.30/D_REP)**2)), COLOR_SMALL, "-"),
        r"$22\times d=9{,}5$\,cm": (D_REP, int(round((0.45/D_REP)**2)), COLOR_SMALL, "--"),
        r"$1\times d=30$\,cm":     (0.30,  1,                            COLOR_30,    "-"),
        r"$1\times d=45$\,cm":     (0.45,  1,                            COLOR_45,    "-"),
    }

    ax = axes[0]
    for lbl, (d_s, N, clr, ls) in scenarios.items():
        m_per_m  = N * pipe_mass_per_m(d_s)
        material = m_per_m * L_pipe * cost_mat
        welding  = N * L_pipe * cost_weld
        # Tiefbau: hoeher fuer grosses Rohr (groessere Trasse)
        exc_f    = 1.0 if d_s <= 0.15 else (0.30 / d_s)
        excavat  = cost_exc * L_pipe * (1 + d_s / D_REP * 0.3)
        capex    = material + welding + excavat
        P_pump   = pump_power(Q_gas, d_s, N)
        en_year  = P_pump * op_h / 1000.0 * cost_kWh   # EUR/Jahr
        mnt_year = material * cost_mnt
        opex_arr = (en_year + mnt_year) * years
        total    = capex + opex_arr
        ax.plot(years, total / 1000, color=clr, ls=ls, label=lbl)

    ax.set_xlabel("Betriebsjahre")
    ax.set_ylabel("Lebenszykluskosten [TEUR]")
    ax.set_title(f"Lebenszykluskosten ueber 30 Jahre\n(L = {L_pipe:.0f} m, "
                 rf"$\dot{{V}}$ = {Q_gas*1000:.0f} L/s)")
    ax.legend()
    ax.grid(True)

    # Rechts: Jährliche Betriebskosten (Energie + Wartung)
    ax = axes[1]
    v_arr2  = np.linspace(0.2, 12.0, 400)
    for d_val, lbl, clr, ls in [
        (D_REP, r"$d=9{,}5$\,cm (einzel)", COLOR_SMALL, "-"),
        (0.30,  r"$d=30$\,cm",             COLOR_30,    "-"),
        (0.45,  r"$d=45$\,cm",             COLOR_45,    "-"),
    ]:
        dp_arr  = [pressure_drop(v, d_val) * L_pipe *
                   (np.pi*d_val**2/4.0)*v * op_h / (1000*pump_eff) * cost_kWh
                   for v in v_arr2]
        ax.semilogy(v_arr2, dp_arr, color=clr, ls=ls, label=lbl)
    ax.axvline(3.0, color="gray", ls=":", lw=1.5, label=r"$v_\mathrm{op}=3$\,m/s")
    ax.set_xlabel(r"Stroemungsgeschwindigkeit $v$ [m/s]")
    ax.set_ylabel("Jaehrl. Pumpenkosten [EUR/Jahr]")
    ax.set_title(f"Jährliche Energiekosten je Rohr\n(L = {L_pipe:.0f} m, "
                 r"$\eta_P = 0{,}70$)")
    ax.legend()
    ax.grid(True, which="both")

    fig.suptitle("Wirtschaftlichkeitsanalyse: Rohrsystemvergleich", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig06_wirtschaftlichkeit.pdf")
    plt.close(fig)
    print("fig06_wirtschaftlichkeit.pdf gespeichert")


# ===========================================================================
# Fig 07 – Effizienzfaktor fuer parallele Rohranordnungen
# ===========================================================================
def plot_parallel_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Effizienz: (Volumenstrom gesamt) / (Pumpenleistung gesamt) normiert
    # auf das einzelne grosse Rohr
    d_large_arr = np.linspace(0.12, 0.55, 400)
    d_s = D_REP
    v_op = 3.0   # m/s im grossen Rohr

    eff_ratio = []
    dp_ratio  = []
    for d_L in d_large_arr:
        N  = (d_L / d_s)**2
        # Kleines Rohr: gleiche Gesamtflaeche --> gleiche v
        Re_L = v_op * d_L / NU
        Re_s = v_op * d_s / NU
        f_L  = colebrook_friction(Re_L, d_L)
        f_s  = colebrook_friction(Re_s, d_s)
        dp_L = f_L * (RHO * v_op**2 / 2.0) / d_L   # Pa/m
        dp_s = f_s * (RHO * v_op**2 / 2.0) / d_s
        dp_ratio.append(dp_s / dp_L)

        # Pumpleistung pro Einheitsquerschnitt: dp * v  (prop. zu dp)
        # Bei gleicher Gesamtflaeche: Leistung prop. dp/m * v * A_total = dp/m * v_total_Q
        # Verhaeltnis: dp_s / dp_L
        # Waerme-Austausch Effizienz: ratio h*A = (h_s/h_L)*(d_L/d_s)
        h_L = heat_transfer_coeff(v_op, d_L)
        h_s = heat_transfer_coeff(v_op, d_s)
        eff_ratio.append((h_s / h_L) * (d_L / d_s))

    ax = axes[0]
    ax.plot(d_large_arr * 100, dp_ratio,  color=COLOR_30, label="Druckverlust-Verhaeltnis")
    ax.plot(d_large_arr * 100, eff_ratio, color=COLOR_SMALL, label="Waermeuebertragungsverhaeltnis")
    ax.axhline(1.0, ls="--", color="gray", lw=1.5)
    ax.axvspan(28, 32, alpha=0.15, color=COLOR_30,  label=r"$d_L=30\,$cm")
    ax.axvspan(43, 47, alpha=0.15, color=COLOR_45,  label=r"$d_L=45\,$cm")
    ax.set_xlabel(r"Durchmesser des grossen Rohrs $d_L$ [cm]")
    ax.set_ylabel(r"Verhaeltnis (parallel kl.) / (einzeln gr.)")
    ax.set_title(f"Leistungsfaehigkeitsverhaeltnis\n"
                 rf"(kl. Rohre: $d_s = {D_REP*100:.1f}$\,cm, gleiche Gesamtflaeche)")
    ax.legend()
    ax.grid(True)

    # Rechts: Barlow-Kurven (zulaessiger Betriebsdruck)
    ax = axes[1]
    d_range2 = np.linspace(0.01, 0.55, 400)
    sigma_s  = 240e6
    t_range  = np.array([0.003, 0.005, 0.008, 0.010, 0.015])   # m
    palette2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_range)))
    for t, col in zip(t_range, palette2):
        P_max = 2 * sigma_s * t / d_range2 / 1e6  # MPa (Barlow ohne Sicherheit)
        ax.semilogy(d_range2 * 100, P_max, color=col,
                    label=rf"$t={t*1000:.0f}$\,mm")
    ax.axhline(0.2, ls=":", color="gray", lw=1.5, label="Betriebsdruck 0,2 MPa")
    ax.axhline(1.0, ls=":", color="red",  lw=1.5, label="Betriebsdruck 1,0 MPa")
    ax.axvspan(8, 11,    alpha=0.2, color=COLOR_SMALL, label="kl. Rohre 8-11 cm")
    ax.axvspan(28, 32,   alpha=0.2, color=COLOR_30)
    ax.axvspan(43, 47,   alpha=0.2, color=COLOR_45)
    ax.set_xlabel(r"Rohrdurchmesser $d$ [cm]")
    ax.set_ylabel(r"Max. Betriebsdruck $p_\mathrm{max}$ [MPa]")
    ax.set_title("Barlow-Kurven: Zulaessiger Betriebsdruck\nvs. Rohrdurchmesser")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both")
    ax.set_ylim(0.01, 10)

    fig.suptitle("Strukturelle Analyse und Effizienzverhaeltnisse", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig07_effizienz_barlow.pdf")
    plt.close(fig)
    print("fig07_effizienz_barlow.pdf gespeichert")


# ===========================================================================
# Fig 08 – Geschwindigkeitsprofile (laminar und turbulent)
# ===========================================================================
def plot_velocity_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    r_norm = np.linspace(-1, 1, 500)   # r/R normiert

    def laminar_profile(r_norm, v_mean):
        return 2 * v_mean * (1 - r_norm**2)

    def turbulent_profile(r_norm, Re):
        """Potenzgesetz-Naeherung: v/v_max = (1 - |r/R|)^(1/n), n = f(Re)"""
        if Re < 2e4:
            n = 6.0
        elif Re < 1e5:
            n = 7.0
        elif Re < 1e6:
            n = 8.5
        else:
            n = 10.0
        return (1.0 - np.abs(r_norm))**(1.0 / n)

    v_mean = 3.0   # m/s

    labels_d = [
        (D_REP, COLOR_SMALL, r"$d=9{,}5$\,cm"),
        (0.30,  COLOR_30,    r"$d=30$\,cm"),
        (0.45,  COLOR_45,    r"$d=45$\,cm"),
    ]

    for ax_idx, (d_val, clr, lbl) in enumerate(labels_d):
        ax = axes[ax_idx]

        # Laminares Profil
        v_lam = laminar_profile(r_norm, v_mean)
        ax.plot(v_lam, r_norm, "b-", lw=2, label="Laminar (HP)")

        # Turbulente Profile bei verschiedenen Geschwindigkeiten
        for v_m, alpha, ls in [(1.0, 0.5, ":"), (3.0, 0.8, "--"), (8.0, 1.0, "-")]:
            Re = v_m * d_val / NU
            prof = turbulent_profile(r_norm, Re) * v_m * 1.22  # auf Mittelwert skaliert
            ax.plot(prof, r_norm, color=clr, lw=1.8, ls=ls, alpha=alpha,
                    label=rf"Turb. $v_m={v_m:.0f}$\,m/s ($Re={Re:.0f}$)")

        ax.axhline(0, ls=":", color="gray", lw=0.8)
        ax.axhline(-1, ls="-", color="black", lw=2)
        ax.axhline(+1, ls="-", color="black", lw=2)
        ax.set_xlabel(r"Geschwindigkeit $v$ [m/s]")
        ax.set_ylabel(r"Normierte radiale Position $r/R$")
        ax.set_title(f"Geschwindigkeitsprofil\n{lbl}")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.set_xlim(left=0)

    fig.suptitle("Geschwindigkeitsprofile in Gasleitungen verschiedener Durchmesser",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig("fig08_geschwindigkeitsprofil.pdf")
    plt.close(fig)
    print("fig08_geschwindigkeitsprofil.pdf gespeichert")


# ===========================================================================
# Fig 09 – Normierter Druckverlust-Vergleich als Balkendiagramm
# ===========================================================================
def plot_bar_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    v_op = 3.0   # m/s
    Q_op = np.pi * 0.30**2 / 4.0 * v_op   # Gesamtvolumenstrom

    N_30 = int(round((0.30/D_REP)**2))
    N_45 = int(round((0.45/D_REP)**2))

    # ---- Druckverlust (normiert auf 30 cm Rohr) ----
    dp_30  = pressure_drop(Q_op / (np.pi*0.30**2/4.0), 0.30)
    dp_45  = pressure_drop(Q_op / (np.pi*0.45**2/4.0), 0.45)
    # Parallelnetz
    q_s_30 = Q_op / N_30;  v_s_30 = q_s_30 / (np.pi*D_REP**2/4)
    q_s_45 = Q_op / N_45;  v_s_45 = q_s_45 / (np.pi*D_REP**2/4)
    dp_par30 = pressure_drop(v_s_30, D_REP)
    dp_par45 = pressure_drop(v_s_45, D_REP)

    # Waermeuebergang (normiert auf 30 cm)
    h_30  = heat_transfer_coeff(v_op, 0.30)
    h_45  = heat_transfer_coeff(Q_op/(np.pi*0.45**2/4.0), 0.45)
    h_s30 = heat_transfer_coeff(v_s_30, D_REP)
    h_s45 = heat_transfer_coeff(v_s_45, D_REP)

    # Gesamtwaermeflaeche (normiert)
    A_30  = np.pi * 0.30
    A_45  = np.pi * 0.45
    A_par30 = N_30 * np.pi * D_REP
    A_par45 = N_45 * np.pi * D_REP

    Q_ht_30  = h_30  * A_30
    Q_ht_45  = h_45  * A_45
    Q_ht_p30 = h_s30 * A_par30
    Q_ht_p45 = h_s45 * A_par45

    labels  = [
        rf"$1\times30$\,cm", rf"$1\times45$\,cm",
        rf"${N_30}\times9.5$\,cm\n(gleiche A wie 30)", rf"${N_45}\times9.5$\,cm\n(gleiche A wie 45)"
    ]
    dp_vals = [dp_30/dp_30, dp_45/dp_30, dp_par30/dp_30, dp_par45/dp_30]
    ht_vals = [Q_ht_30/Q_ht_30, Q_ht_45/Q_ht_30, Q_ht_p30/Q_ht_30, Q_ht_p45/Q_ht_30]

    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width/2, dp_vals, width, label="Norm. Druckverlust", color=[COLOR_30, COLOR_45, COLOR_SMALL, COLOR_SMALL], alpha=0.85)
    b2 = ax.bar(x + width/2, ht_vals, width, label="Norm. Waermeuebertragung", color=[COLOR_30, COLOR_45, COLOR_SMALL, COLOR_SMALL], alpha=0.50, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normierter Wert (Bezug: $1\\times30$\\,cm)")
    ax.set_title("Normierter Vergleich: Druckverlust und Waermeuebertragung\n"
                 rf"(Betriebsgeschwindigkeit $v_\mathrm{{op}} = {v_op}$\,m/s im grossem Rohr)")
    ax.legend()
    ax.grid(axis="y")
    ax.axhline(1.0, ls="--", color="gray", lw=1.2)

    fig.tight_layout()
    fig.savefig("fig09_balkenvergleich.pdf")
    plt.close(fig)
    print("fig09_balkenvergleich.pdf gespeichert")


# ===========================================================================
# Hauptprogramm
# ===========================================================================
if __name__ == "__main__":
    print("Erzeuge Abbildungen ...")
    plot_moody()
    plot_pressure_drop_single()
    plot_reynolds()
    plot_heat_transfer()
    plot_surface_ratio()
    plot_economics()
    plot_parallel_efficiency()
    plot_velocity_profiles()
    plot_bar_comparison()
    print("Alle Abbildungen erfolgreich gespeichert.")
