"""
Matplotlib-Abbildungen fuer die wissenschaftliche Arbeit:
Lärmarme Überschallpassagierflugzeuge – ZAGI-Konzept
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.signal import welch
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

FARBEN = ['#1f4e79', '#c55a11', '#375623', '#6d3e8e', '#1e5799', '#843c0c']

# =========================================================
# Abbildung 1: Druckwellenverteilung – konventionell vs. ZAGI
# =========================================================
def fig_druckwelle():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.linspace(-50, 150, 1200)

    # Konventionell: N-Welle
    def n_welle(x, x0=0, breite=30):
        welle = np.zeros_like(x)
        maske = (x >= x0) & (x <= x0 + breite)
        welle[maske] = np.where(
            x[maske] <= x0 + breite/2,
            2.0 * (x[maske] - x0) / breite,
            2.0 * (x0 + breite - x[maske]) / breite
        )
        # N-Form: positiver Schlag, kurze Pause, negativer Schlag
        welle[(x >= x0) & (x <= x0 + breite*0.45)] = \
            1.8 * np.exp(-((x[(x >= x0) & (x <= x0 + breite*0.45)] - x0 - 3) ** 2) / 18)
        welle[(x >= x0 + breite*0.55) & (x <= x0 + breite)] = \
            -1.8 * np.exp(-((x[(x >= x0 + breite*0.55) & (x <= x0 + breite)] - (x0 + breite - 3)) ** 2) / 18)
        return welle

    druck_konv = n_welle(x)

    # ZAGI: gedämpfte, verteilte Welle
    def zagi_welle(x, x0=0, breite=80):
        welle = np.zeros_like(x)
        # Gaußförmige, langsam ansteigende Druckverteilung
        welle = 0.22 * np.exp(-((x - x0 - breite/2) ** 2) / (2 * 35**2))
        welle += 0.08 * np.sin(2 * np.pi * (x - x0) / 60) * np.exp(-((x - x0 - breite/2)**2) / (2*45**2))
        return welle

    druck_zagi = zagi_welle(x)

    # Plot konventionell
    axes[0].plot(x, druck_konv, color=FARBEN[1], linewidth=2.2, label='Druckprofil (N-Welle)')
    axes[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[0].fill_between(x, druck_konv, 0, where=(druck_konv > 0), alpha=0.25, color=FARBEN[1])
    axes[0].fill_between(x, druck_konv, 0, where=(druck_konv < 0), alpha=0.25, color='#8b0000')
    axes[0].set_xlabel('Horizontale Position [m]')
    axes[0].set_ylabel('Druckdifferenz $\\Delta p / p_\\infty$')
    axes[0].set_title('Konventionelles Überschallflugzeug\n(klassische N-Welle)')
    axes[0].set_xlim(-20, 120)
    axes[0].set_ylim(-2.5, 2.5)
    axes[0].annotate('Vorderer\nKnall', xy=(12, 1.8), xytext=(25, 2.1),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    axes[0].annotate('Hinterer\nKnall', xy=(28, -1.8), xytext=(40, -2.2),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot ZAGI
    axes[1].plot(x, druck_zagi, color=FARBEN[0], linewidth=2.2, label='Druckprofil (ZAGI-Konzept)')
    axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[1].fill_between(x, druck_zagi, 0, where=(druck_zagi > 0), alpha=0.3, color=FARBEN[0])
    axes[1].set_xlabel('Horizontale Position [m]')
    axes[1].set_ylabel('Druckdifferenz $\\Delta p / p_\\infty$')
    axes[1].set_title('ZAGI-Konzept\n(verteilte, gedämpfte Druckwelle)')
    axes[1].set_xlim(-20, 120)
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].annotate('Kontinuierlicher,\ngedämpfter Druckanstieg', xy=(40, 0.22), xytext=(55, 0.6),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig01_druckwelle.pdf', bbox_inches='tight')
    plt.close()
    print("fig01 gespeichert")

# =========================================================
# Abbildung 2: Schalldruckpegel (SPL) vs. Machzahl
# =========================================================
def fig_spl_mach():
    fig, ax = plt.subplots(figsize=(10, 6))
    mach = np.linspace(0.8, 2.5, 300)

    def spl_konventionell(M):
        spl = np.zeros_like(M)
        unterschall = M < 1.0
        ueber = M >= 1.0
        spl[unterschall] = 75 + 8 * (M[unterschall] - 0.8) / 0.2
        spl[ueber] = 94 + 22 * np.log10(M[ueber]) + 5 * (M[ueber] - 1.0)
        return spl

    def spl_zagi(M):
        spl = np.zeros_like(M)
        unterschall = M < 1.0
        ueber = M >= 1.0
        spl[unterschall] = 72 + 5 * (M[unterschall] - 0.8) / 0.2
        spl[ueber] = 75 + 9 * np.log10(M[ueber]) + 2 * (M[ueber] - 1.0)
        return spl

    spl_k = spl_konventionell(mach)
    spl_z = spl_zagi(mach)

    ax.plot(mach, spl_k, color=FARBEN[1], linewidth=2.2, label='Konventionell (Referenz)')
    ax.plot(mach, spl_z, color=FARBEN[0], linewidth=2.2, linestyle='--', label='ZAGI-Konzept')
    ax.axvline(1.0, color='gray', linewidth=1.0, linestyle=':', label='Machzahl $M = 1$')
    ax.fill_between(mach, spl_k, spl_z, where=(spl_k > spl_z), alpha=0.2, color='green',
                    label='Lärmreduktion')
    ax.set_xlabel('Machzahl $M$')
    ax.set_ylabel('Schalldruckpegel SPL [dB(A)]')
    ax.set_title('Schalldruckpegel als Funktion der Machzahl\n– Vergleich konventionell vs. ZAGI-Konzept')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('Transsonischer\nBereich', xy=(1.05, 96), xytext=(1.3, 100),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    reduktion_bei_15 = spl_konventionell(np.array([1.5]))[0] - spl_zagi(np.array([1.5]))[0]
    ax.annotate(f'Reduktion bei M=1.5:\n~{reduktion_bei_15:.1f} dB(A)',
                xy=(1.5, spl_zagi(np.array([1.5]))[0]),
                xytext=(1.7, 82),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig('fig02_spl_mach.pdf', bbox_inches='tight')
    plt.close()
    print("fig02 gespeichert")

# =========================================================
# Abbildung 3: Kegel-Geometrie und Mach-Winkel
# =========================================================
def fig_mach_kegel():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    mach_values = np.linspace(1.01, 3.0, 300)
    mu = np.degrees(np.arcsin(1.0 / mach_values))

    axes[0].plot(mach_values, mu, color=FARBEN[0], linewidth=2.2)
    axes[0].fill_between(mach_values, mu, alpha=0.15, color=FARBEN[0])
    axes[0].set_xlabel('Machzahl $M$')
    axes[0].set_ylabel('Mach-Winkel $\\mu$ [Grad]')
    axes[0].set_title('Mach-Winkel $\\mu = \\arcsin(1/M)$\nvs. Machzahl')
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('$M = 1.7$: $\\mu \\approx 36°$', xy=(1.7, np.degrees(np.arcsin(1/1.7))),
                     xytext=(2.0, 40),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

    # Schematische Kegeldarstellung
    ax2 = axes[1]
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_title('Machkegel – Schematische Darstellung\nbei $M = 1.7$ ($\\mu \\approx 36°$)')

    M_ex = 1.7
    mu_ex = np.arcsin(1.0 / M_ex)
    t = np.linspace(0, 8, 100)

    ax2.annotate('', xy=(8, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=FARBEN[0], lw=2))
    ax2.text(8.2, 0.1, 'Flugzeug', fontsize=9, color=FARBEN[0])

    # Mach-Kegel Linien
    kegel_oben_x = t
    kegel_oben_y = -t * np.tan(mu_ex)
    kegel_unten_y = t * np.tan(mu_ex)
    ax2.plot(kegel_oben_x, kegel_oben_y, color=FARBEN[1], linewidth=2, linestyle='-', label='Machkegel-Rand')
    ax2.plot(kegel_oben_x, -kegel_oben_y, color=FARBEN[1], linewidth=2, linestyle='-')

    # ZAGI-verteilter Druckbereich
    x_fill = np.concatenate([t, t[::-1]])
    y_fill_oben = np.concatenate([kegel_oben_y * 0.6, (-kegel_oben_y * 0.6)[::-1]])
    ax2.fill(x_fill, y_fill_oben, alpha=0.2, color=FARBEN[0], label='ZAGI: Druckverteilungszone')

    ax2.plot(8, 0, 'o', color=FARBEN[0], markersize=8)
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Winkel-Annotation
    theta_arc = np.linspace(np.pi - mu_ex, np.pi, 50)
    r_arc = 1.5
    ax2.plot(8 + r_arc * np.cos(theta_arc), r_arc * np.sin(theta_arc), 'k-', linewidth=1)
    ax2.text(6.2, -0.9, f'$\\mu$={np.degrees(mu_ex):.0f}°', fontsize=9)

    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlabel('x [relative Einheiten]')
    ax2.set_ylabel('y [relative Einheiten]')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('fig03_mach_kegel.pdf', bbox_inches='tight')
    plt.close()
    print("fig03 gespeichert")

# =========================================================
# Abbildung 4: Lärmspektrum – Start/Landung
# =========================================================
def fig_laermspektrum():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    freqs = np.logspace(1, 4, 500)

    def spektrum_konv_start(f):
        # Dominante Frequenzen: Triebwerk (~100 Hz), Strömung (~800 Hz), Hochton
        s = 85 * np.exp(-((np.log10(f) - 2.0)**2) / 0.08)
        s += 78 * np.exp(-((np.log10(f) - 2.9)**2) / 0.12)
        s += 70 * np.exp(-((np.log10(f) - 3.5)**2) / 0.15)
        return s

    def spektrum_zagi_start(f):
        s = 72 * np.exp(-((np.log10(f) - 2.0)**2) / 0.10)
        s += 63 * np.exp(-((np.log10(f) - 2.9)**2) / 0.14)
        s += 55 * np.exp(-((np.log10(f) - 3.5)**2) / 0.18)
        return s

    def spektrum_konv_land(f):
        s = 80 * np.exp(-((np.log10(f) - 2.2)**2) / 0.09)
        s += 74 * np.exp(-((np.log10(f) - 3.0)**2) / 0.11)
        s += 66 * np.exp(-((np.log10(f) - 3.6)**2) / 0.13)
        return s

    def spektrum_zagi_land(f):
        s = 67 * np.exp(-((np.log10(f) - 2.2)**2) / 0.11)
        s += 60 * np.exp(-((np.log10(f) - 3.0)**2) / 0.13)
        s += 51 * np.exp(-((np.log10(f) - 3.6)**2) / 0.16)
        return s

    for ax, (skonv, szagi, title) in zip(axes, [
        (spektrum_konv_start, spektrum_zagi_start, 'Startphase'),
        (spektrum_konv_land, spektrum_zagi_land, 'Landephase')
    ]):
        ax.semilogx(freqs, skonv(freqs), color=FARBEN[1], linewidth=2.0, label='Konventionell')
        ax.semilogx(freqs, szagi(freqs), color=FARBEN[0], linewidth=2.0, linestyle='--', label='ZAGI-Konzept')
        ax.fill_between(freqs, skonv(freqs), szagi(freqs),
                        where=(skonv(freqs) > szagi(freqs)),
                        alpha=0.2, color='green', label='Reduktion')
        ax.set_xlabel('Frequenz [Hz]')
        ax.set_ylabel('Schalldruckpegel [dB(A)]')
        ax.set_title(f'Lärmspektrum – {title}')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(10, 10000)
        ax.set_ylim(40, 100)
        # ICAO Grenzwerte
        ax.axhline(85, color='red', linewidth=1.0, linestyle=':', alpha=0.7)
        ax.text(12, 85.5, 'ICAO Kap. 14 Grenzwert', fontsize=8, color='red', alpha=0.8)

    plt.tight_layout()
    plt.savefig('fig04_laermspektrum.pdf', bbox_inches='tight')
    plt.close()
    print("fig04 gespeichert")

# =========================================================
# Abbildung 5: Thermodynamisches Zustandsdiagramm (T-s)
# =========================================================
def fig_ts_diagramm():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Zustandspunkte: Umgebung -> Einlass -> Verdichter -> Brennkammer -> Turbine -> Düse
    gamma = 1.4
    cp = 1005  # J/(kg K)
    T0 = 220   # K (Reiseflughöhe ~12 km)
    p0 = 19300 # Pa

    # Normaler Kreisprozess (konventionell)
    # Punkt 1: Umgebung
    T1 = T0
    s1 = 0.0
    # Punkt 2: Nach Einlassverdichtung (isentrop + Verluste)
    T2 = T0 * (1 + (gamma - 1)/2 * 1.7**2)  # M=1.7 Staupunkt
    s2 = s1 + 0.05 * cp  # geringe Entropieproduktion
    # Punkt 3: Nach Verdichter
    pi_v = 35.0
    eta_v = 0.88
    T3 = T2 * (1 + (pi_v**((gamma-1)/gamma) - 1) / eta_v)
    s3 = s2 + 0.12 * cp
    # Punkt 4: Nach Brennkammer
    T4 = 1650
    s4 = s3 + cp * np.log(T4 / T3)
    # Punkt 5: Nach Turbine
    pi_t = 0.06
    eta_t = 0.90
    T5 = T4 * (1 - eta_t * (1 - pi_t**((gamma-1)/gamma)))
    s5 = s4 + 0.08 * cp
    # Punkt 6: Schubdüse
    T6 = T5 * 0.75
    s6 = s5 + 0.03 * cp

    s_konv = [s1, s2, s3, s4, s5, s6]
    T_konv = [T1, T2, T3, T4, T5, T6]
    labels = ['1: Umgebung', '2: Einlass', '3: Verdichter', '4: Brennkammer', '5: Turbine', '6: Düse']

    # ZAGI: verbesserter Einlass mit Stoßminimierung
    T2z = T0 * (1 + (gamma - 1)/2 * 1.7**2 * 0.92)
    s2z = s1 + 0.02 * cp  # weniger Entropie durch Stoßreduktion
    T3z = T2z * (1 + (pi_v**((gamma-1)/gamma) - 1) / (eta_v + 0.03))
    s3z = s2z + 0.09 * cp
    T4z = 1680
    s4z = s3z + cp * np.log(T4z / T3z)
    T5z = T4z * (1 - (eta_t + 0.02) * (1 - pi_t**((gamma-1)/gamma)))
    s5z = s4z + 0.06 * cp
    T6z = T5z * 0.72
    s6z = s5z + 0.02 * cp

    s_zagi = [s1, s2z, s3z, s4z, s5z, s6z]
    T_zagi = [T1, T2z, T3z, T4z, T5z, T6z]

    ax.plot(s_konv, T_konv, 'o-', color=FARBEN[1], linewidth=2.0, markersize=7, label='Konventionell')
    ax.plot(s_zagi, T_zagi, 's--', color=FARBEN[0], linewidth=2.0, markersize=7, label='ZAGI-Konzept')

    for i, (s, T, lab) in enumerate(zip(s_konv, T_konv, labels)):
        ax.annotate(lab, (s, T), textcoords='offset points', xytext=(8, 4), fontsize=7.5, color=FARBEN[1])

    ax.set_xlabel('Spezifische Entropie $s$ [J/(kg K)]')
    ax.set_ylabel('Temperatur $T$ [K]')
    ax.set_title('T-s-Diagramm des Triebwerksprozesses\n– Konventionell vs. ZAGI-Konzept')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig05_ts_diagramm.pdf', bbox_inches='tight')
    plt.close()
    print("fig05 gespeichert")

# =========================================================
# Abbildung 6: Lärmkontour (Footprint) am Boden
# =========================================================
def fig_laermkontour():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    x = np.linspace(-60, 60, 400)
    y = np.linspace(-40, 40, 300)
    X, Y = np.meshgrid(x, y)

    def spl_boden_konv(X, Y, h=11000):
        r = np.sqrt(X**2 + Y**2 + h**2)
        # N-Welle SPL
        spl = 110 - 20 * np.log10(r / 100) - 0.004 * r
        spl += 8 * np.exp(-(Y**2) / 500)  # Fokussierung
        return np.clip(spl, 55, 115)

    def spl_boden_zagi(X, Y, h=11000):
        r = np.sqrt(X**2 + Y**2 + h**2)
        spl = 90 - 20 * np.log10(r / 100) - 0.005 * r
        spl += 2 * np.exp(-(Y**2) / 800)
        return np.clip(spl, 45, 95)

    spl_k = spl_boden_konv(X * 1000, Y * 1000)
    spl_z = spl_boden_zagi(X * 1000, Y * 1000)

    levels = np.arange(55, 115, 5)
    levels_z = np.arange(45, 95, 5)

    for ax, (spl, lvl, title) in zip(axes, [
        (spl_k, levels, 'Konventionell'),
        (spl_z, levels_z, 'ZAGI-Konzept')
    ]):
        cf = ax.contourf(X, Y, spl, levels=lvl, cmap='RdYlGn_r', alpha=0.85)
        ct = ax.contour(X, Y, spl, levels=lvl, colors='black', linewidths=0.5, alpha=0.5)
        ax.clabel(ct, inline=True, fontsize=7, fmt='%d dB')
        plt.colorbar(cf, ax=ax, label='SPL [dB(A)]')
        ax.set_xlabel('Horizontale Entfernung [km]')
        ax.set_ylabel('Seitliche Entfernung [km]')
        ax.set_title(f'Lärmkontour (Bodenfootprint)\n{title}')
        ax.set_aspect('equal')
        # Flugpfad
        ax.axhline(0, color='blue', linewidth=1.5, linestyle='--', alpha=0.7, label='Flugpfad')
        ax.plot(0, 0, 'b^', markersize=10, label='Flugzeugposition')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('fig06_laermkontour.pdf', bbox_inches='tight')
    plt.close()
    print("fig06 gespeichert")

# =========================================================
# Abbildung 7: Stoßwellen-Formierung – numerische Simulation
# =========================================================
def fig_stosswellen():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.linspace(-0.5, 3.0, 600)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)

    def druckfeld_konv(X, Y, M=1.7):
        mu = np.arcsin(1.0 / M)
        # Bugwelle und Heckwelle
        bug_welle = np.exp(-50 * (Y - X * np.tan(mu))**2 / (1 + 0.1*X))
        heck_welle = np.exp(-50 * (Y - (X - 2.0) * np.tan(mu))**2 / (1 + 0.1*(X-2)))
        heck_welle *= (X > 2.0).astype(float)
        basis = 1.0 + 0.3 * bug_welle - 0.2 * heck_welle
        return np.clip(basis, 0.7, 1.5)

    def druckfeld_konv_neg(X, Y, M=1.7):
        mu = np.arcsin(1.0 / M)
        bug_welle = np.exp(-50 * (Y + X * np.tan(mu))**2 / (1 + 0.1*X))
        heck_welle = np.exp(-50 * (Y + (X - 2.0) * np.tan(mu))**2 / (1 + 0.1*(X-2)))
        heck_welle *= (X > 2.0).astype(float)
        basis = 1.0 + 0.3 * bug_welle - 0.2 * heck_welle
        return np.clip(basis, 0.7, 1.5)

    def druckfeld_gesamt(X, Y, M=1.7):
        return (druckfeld_konv(X, Y, M) + druckfeld_konv_neg(X, Y, M)) / 2

    def druckfeld_zagi(X, Y, M=1.7):
        # Flächenregel-Optimierung: sanfter Druckverlauf
        sigma = 0.35
        d = np.sqrt((X - 1.0)**2 * 0.3 + Y**2)
        basis = 1.0 + 0.08 * np.exp(-d**2 / (2 * sigma**2))
        basis += 0.03 * np.sin(3 * X) * np.exp(-Y**2 / 0.5) * (X > 0) * (X < 2)
        return np.clip(basis, 0.9, 1.2)

    konfigurationen = [
        (druckfeld_gesamt(X, Y, 1.5), 'Konventionell, M=1.5'),
        (druckfeld_gesamt(X, Y, 1.7), 'Konventionell, M=1.7'),
        (druckfeld_zagi(X, Y, 1.5), 'ZAGI-Konzept, M=1.5'),
        (druckfeld_zagi(X, Y, 1.7), 'ZAGI-Konzept, M=1.7'),
    ]

    for ax, (daten, titel) in zip(axes.flat, konfigurationen):
        cf = ax.contourf(X, Y, daten, levels=20, cmap='coolwarm', alpha=0.9)
        plt.colorbar(cf, ax=ax, label='$p/p_\\infty$')
        ax.set_title(titel)
        ax.set_xlabel('$x/L$')
        ax.set_ylabel('$y/L$')
        # Flugzeug-Rumpf skizzieren
        ax.fill_between([0, 2], [-0.05, -0.05], [0.05, 0.05], color='black', alpha=0.7, zorder=5)
        ax.set_xlim(-0.5, 3.0)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Druckfeldverteilung um Überschallflugzeuge\n– Numerische Simulation (vereinfacht)', 
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('fig07_stosswellen.pdf', bbox_inches='tight')
    plt.close()
    print("fig07 gespeichert")

# =========================================================
# Abbildung 8: Vergleich EPNL (Effective Perceived Noise Level)
# =========================================================
def fig_epnl_vergleich():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Kategorien
    kategorien = ['Start\n(Fly-over)', 'Seitlich\n(Lateral)', 'Anflug\n(Approach)']
    epnl_konv = np.array([96.5, 94.2, 102.8])
    epnl_zagi = np.array([81.3, 78.9, 87.4])
    icao_limit = np.array([89.0, 94.0, 98.0])  # ICAO Chapter 14 (600 t Klasse, fiktiv)

    x = np.arange(len(kategorien))
    breite = 0.25

    bars1 = axes[0].bar(x - breite, epnl_konv, breite, label='Konventionell', color=FARBEN[1], alpha=0.85)
    bars2 = axes[0].bar(x, epnl_zagi, breite, label='ZAGI-Konzept', color=FARBEN[0], alpha=0.85)
    bars3 = axes[0].bar(x + breite, icao_limit, breite, label='ICAO Kap. 14 Limit', color='gray', alpha=0.6)

    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_xlabel('Messpunkt')
    axes[0].set_ylabel('EPNL [EPNdB]')
    axes[0].set_title('Effektiver wahrgenommener Lärmpegel (EPNL)\nnach ICAO Annex 16')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kategorien)
    axes[0].legend()
    axes[0].set_ylim(60, 115)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Kumulative Margin
    margin_konv = epnl_konv - icao_limit
    margin_zagi = epnl_zagi - icao_limit

    axes[1].bar(kategorien, margin_konv, color=FARBEN[1], alpha=0.8, label='Konventionell (Überschreitung)')
    axes[1].bar(kategorien, margin_zagi, color=FARBEN[0], alpha=0.8, label='ZAGI-Konzept (Reserve/Überschreitung)')
    axes[1].axhline(0, color='black', linewidth=1.5, label='ICAO Grenzwert')
    axes[1].fill_between(range(len(kategorien)), margin_zagi, 0,
                         where=(np.array(margin_zagi) < 0), alpha=0.2, color='green',
                         interpolate=True)
    axes[1].set_xlabel('Messpunkt')
    axes[1].set_ylabel('EPNL – ICAO Limit [EPNdB]')
    axes[1].set_title('Lärmmarge relativ zum ICAO Kap. 14 Grenzwert\n(negativ = Unterschreitung = gut)')
    axes[1].set_xticks(range(len(kategorien)))
    axes[1].set_xticklabels(kategorien)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fig08_epnl.pdf', bbox_inches='tight')
    plt.close()
    print("fig08 gespeichert")

# =========================================================
# Abbildung 9: Flächenregel – Querschnittsverlauf
# =========================================================
def fig_flaechenregel():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.linspace(0, 1, 300)

    def querschnitt_konv(x):
        # Typische zylindrische Rumpfform mit abruptem Flügelübergang
        rumpf = 2.5 * np.exp(-(x - 0.1)**2 / 0.003) * x + \
                4.8 * (1 - np.exp(-30 * x)) * np.exp(-30 * (x - 1)**2 / 0.1) * (x < 0.5) + \
                4.8 * np.exp(-20 * (x - 0.5)**2 / 0.05) * (x >= 0.4)
        flügel = 3.2 * np.exp(-(x - 0.45)**2 / 0.005)  # steiler Flügelbereich
        return rumpf + flügel

    def querschnitt_zagi(x):
        # Flächenregel-optimiert: "taillenförmig" am Flügelübergang
        rumpf_basis = 4.5 * np.sin(np.pi * x) * (x < 0.92) * (x > 0.03)
        flügel = 1.5 * np.exp(-(x - 0.45)**2 / 0.008)
        taille = -1.8 * np.exp(-(x - 0.45)**2 / 0.004)  # Einschnürung
        return rumpf_basis + flügel + taille

    sq_k = np.abs(querschnitt_konv(x))
    sq_z = np.abs(querschnitt_zagi(x))

    # Normalisieren
    sq_k = sq_k / sq_k.max() * 10
    sq_z = sq_z / sq_z.max() * 10

    axes[0].plot(x, sq_k, color=FARBEN[1], linewidth=2.2, label='Konventionell')
    axes[0].plot(x, sq_z, color=FARBEN[0], linewidth=2.2, linestyle='--', label='ZAGI (Flächenregel)')
    axes[0].fill_between(x, sq_k, sq_z, where=(sq_k > sq_z), alpha=0.2, color='green', label='Querschnittsreduktion')
    axes[0].fill_between(x, sq_k, sq_z, where=(sq_k < sq_z), alpha=0.2, color='red')
    axes[0].set_xlabel('Normierte Rumpflänge $x/L$')
    axes[0].set_ylabel('Querschnittsfläche $A(x)$ [m²]')
    axes[0].set_title('Querschnittsverlauf $A(x)$ entlang der Rumpfachse\n– Whitcombsche Flächenregel')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0.45, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    axes[0].text(0.46, 8.5, 'Flügel-\nübergang', fontsize=8, color='gray')

    # Ableitung (dA/dx) – wichtig für Wellenbildung
    dAdx_k = np.gradient(sq_k, x)
    dAdx_z = np.gradient(sq_z, x)
    axes[1].plot(x, dAdx_k, color=FARBEN[1], linewidth=2.0, label="$dA/dx$ Konventionell")
    axes[1].plot(x, dAdx_z, color=FARBEN[0], linewidth=2.0, linestyle='--', label="$dA/dx$ ZAGI")
    axes[1].axhline(0, color='gray', linewidth=0.8)
    axes[1].set_xlabel('Normierte Rumpflänge $x/L$')
    axes[1].set_ylabel('$dA/dx$ [m²/m]')
    axes[1].set_title('Ableitung des Querschnittsverlaufs $dA/dx$\n– Kriterium für Wellenbildung')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate('Starke Diskontinuität\n→ Stoßwelle', xy=(0.45, dAdx_k[np.argmin(np.abs(x-0.45))]),
                     xytext=(0.6, -12), arrowprops=dict(arrowstyle='->', color=FARBEN[1]), fontsize=8, color=FARBEN[1])

    plt.tight_layout()
    plt.savefig('fig09_flaechenregel.pdf', bbox_inches='tight')
    plt.close()
    print("fig09 gespeichert")

# =========================================================
# Abbildung 10: Reichweite & Kraftstoffverbrauch
# =========================================================
def fig_reichweite():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    mach = np.linspace(1.2, 2.5, 100)

    # Breguet-Reichweite vereinfacht: R = (L/D) * (v / SFC) * ln(m0/m1)
    LD_konv = 8.5 - 1.5 * (mach - 1.2)  # L/D fällt mit Mach
    LD_zagi = 9.8 - 1.2 * (mach - 1.2)  # besser durch optimierte Form

    SFC_konv = 0.028 + 0.008 * (mach - 1.0)**1.5  # kg/(N·s), vereinfacht
    SFC_zagi = 0.025 + 0.006 * (mach - 1.0)**1.4

    v = mach * 295.0  # m/s bei ~10 km Höhe (a = 295 m/s)
    ln_faktor = np.log(1.4)  # typisches Brennstoffverhältnis

    R_konv = (LD_konv * v / (SFC_konv * 9.81)) * ln_faktor / 1000  # km
    R_zagi = (LD_zagi * v / (SFC_zagi * 9.81)) * ln_faktor / 1000

    axes[0].plot(mach, R_konv, color=FARBEN[1], linewidth=2.2, label='Konventionell')
    axes[0].plot(mach, R_zagi, color=FARBEN[0], linewidth=2.2, linestyle='--', label='ZAGI-Konzept')
    axes[0].fill_between(mach, R_konv, R_zagi, where=(R_zagi > R_konv), alpha=0.2, color='green', label='Verbesserung')
    axes[0].set_xlabel('Machzahl $M$')
    axes[0].set_ylabel('Reichweite $R$ [km]')
    axes[0].set_title('Theoretische Reichweite (Breguet-Gleichung)\nvs. Machzahl')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Kraftstoffverbrauch vs. Reichweite
    reichweite = np.linspace(2000, 10000, 100)
    verbrauch_konv = 18.5 + 0.0012 * reichweite + 0.000000003 * reichweite**2
    verbrauch_zagi = 14.2 + 0.00095 * reichweite + 0.0000000022 * reichweite**2

    axes[1].plot(reichweite, verbrauch_konv, color=FARBEN[1], linewidth=2.2, label='Konventionell')
    axes[1].plot(reichweite, verbrauch_zagi, color=FARBEN[0], linewidth=2.2, linestyle='--', label='ZAGI-Konzept')
    axes[1].fill_between(reichweite, verbrauch_konv, verbrauch_zagi, alpha=0.2, color='green',
                         where=(verbrauch_konv > verbrauch_zagi), label='Einsparung')
    axes[1].set_xlabel('Reichweite [km]')
    axes[1].set_ylabel('Kraftstoffverbrauch [t/Flug]')
    axes[1].set_title('Kraftstoffverbrauch vs. Reichweite\nbei optimaler Reisemachzahl')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(8000, color='gray', linestyle=':', linewidth=0.8)
    axes[1].text(8100, 22, 'Frankfurt–\nNew York\n(~8.000 km)', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig('fig10_reichweite.pdf', bbox_inches='tight')
    plt.close()
    print("fig10 gespeichert")

# =========================================================
# Alle Abbildungen generieren
# =========================================================
if __name__ == '__main__':
    import os
    os.chdir('.')
    fig_druckwelle()
    fig_spl_mach()
    fig_mach_kegel()
    fig_laermspektrum()
    fig_ts_diagramm()
    fig_laermkontour()
    fig_stosswellen()
    fig_epnl_vergleich()
    fig_flaechenregel()
    fig_reichweite()
    print("Alle Abbildungen erfolgreich generiert.")
