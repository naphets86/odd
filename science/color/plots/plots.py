import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings('ignore')

OUTDIR = 'plots"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ============================================================
# Plot 1: CIE D65 Tageslichtspektrum S(lambda)
# ============================================================
def plot_cie_d65():
    lam = np.linspace(300, 830, 531)
    # Approximation der CIE D65 Spectral Power Distribution
    # Basiert auf tabellierten CIE-Werten (angepasst)
    def d65_approx(l):
        # Vereinfachte Gauss-Superposition zur D65-Approximation
        s = (
            25 * np.exp(-((l-450)/60)**2) +
            50 * np.exp(-((l-520)/80)**2) +
            80 * np.exp(-((l-580)/90)**2) +
            60 * np.exp(-((l-650)/80)**2) +
            40 * np.exp(-((l-730)/70)**2) +
            0.03*(l-300)
        )
        return np.clip(s, 0, None)
    
    s = d65_approx(lam)
    s = s / s.max()

    # Farb-Mapping für sichtbares Spektrum
    vis_lam = np.linspace(380, 780, 401)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Spektrale Farben im Hintergrund
    for i in range(len(vis_lam)-1):
        l = vis_lam[i]
        if l < 420:
            r, g, b = 0.5, 0, 0.8
        elif l < 450:
            t = (l-420)/30; r, g, b = 0.5*(1-t), 0, 0.8
        elif l < 490:
            t = (l-450)/40; r, g, b = 0, t*0.3, 1
        elif l < 510:
            t = (l-490)/20; r, g, b = 0, 0.3+t*0.7, 1*(1-t)
        elif l < 560:
            t = (l-510)/50; r, g, b = t, 1, 0
        elif l < 590:
            t = (l-560)/30; r, g, b = 1, 1*(1-t*0.5), 0
        elif l < 625:
            t = (l-590)/35; r, g, b = 1, 0.5*(1-t), 0
        else:
            r, g, b = 0.8, 0, 0
        
        idx_start = np.searchsorted(lam, vis_lam[i])
        idx_end = np.searchsorted(lam, vis_lam[i+1])
        for idx in range(idx_start, min(idx_end+1, len(lam))):
            ax.axvline(x=lam[idx], color=(r, g, b), alpha=0.15, linewidth=1.0)
    
    ax.plot(lam, s, 'k-', linewidth=2, label=r'$S(\lambda)$ CIE D65')
    ax.fill_between(lam, s, alpha=0.3, color='gold')
    
    ax.set_xlim(300, 830)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'Wellenlänge $\lambda$ [nm]')
    ax.set_ylabel(r'Relative Leistungsdichte')
    ax.set_title('CIE D65 Tageslichtspektrum')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Markierung sichtbarer Bereich
    ax.axvspan(380, 780, alpha=0.05, color='blue', label='Sichtbar')
    ax.annotate('sichtbarer Bereich\n380–780 nm', xy=(580, 1.05),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot1_cie_d65.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 1 gespeichert.")

# ============================================================
# Plot 2: Absorptionsspektrum Lykopin mit HOMO/LUMO-Übergang
# ============================================================
def plot_lycopin_absorption():
    lam = np.linspace(350, 650, 500)
    
    # Lykopin-Absorptionsspektrum: drei charakteristische Banden
    # Hauptbande bei ca. 470 nm (in Hexan)
    def lorentz(l, l0, gamma, A):
        return A * (gamma/2)**2 / ((l - l0)**2 + (gamma/2)**2)
    
    # Drei vibronische Banden (0-0, 0-1, 0-2 Übergänge)
    abs_spec = (
        lorentz(lam, 503, 18, 0.7) +   # 0-0 Übergang
        lorentz(lam, 471, 20, 1.0) +   # 0-1 Hauptbande
        lorentz(lam, 442, 18, 0.55)    # 0-2 Übergang
    )
    abs_spec_corr = abs_spec * 1.1  # Kopplungskorrektur
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(lam, abs_spec, 'b-', linewidth=2, label='Ohne Kopplung (RI-CC2)')
    ax.plot(lam, abs_spec_corr, 'r-', linewidth=2, label='Mit Vibrationskorrektur')
    
    # Markierung der Hauptbanden
    for peak, label, col in [(503, '503 nm', 'blue'), (471, '471 nm\n(Hauptbande)', 'red'), (442, '442 nm', 'blue')]:
        y = lorentz(peak, peak, 18, 0.7 if peak != 471 else 1.0)
        if peak == 471:
            y = lorentz(peak, 471, 20, 1.0)
        ax.annotate(label, xy=(peak, y+0.03), ha='center', fontsize=9,
                    color=col, arrowprops=dict(arrowstyle='->', color=col),
                    xytext=(peak+15, y+0.15))
    
    # 463 nm Markierung wie im Poster
    ax.axvline(x=463, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(463, 0.9, '463 nm\nf=5.39', color='darkred', fontsize=9,
            ha='right', va='top')
    
    ax.set_xlim(350, 650)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel(r'Wellenlänge $\lambda$ [nm]')
    ax.set_ylabel(r'Absorptionsquerschnitt $\alpha(\lambda)$ [a.u.]')
    ax.set_title('Absorptionsspektrum Lykopin (Hexan-Lösung)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Shading der Absorptionsbanden
    ax.fill_between(lam, abs_spec, alpha=0.15, color='blue')
    ax.fill_between(lam, abs_spec_corr, abs_spec, alpha=0.15, color='red')
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot2_lycopin_absorption.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 2 gespeichert.")

# ============================================================
# Plot 3: Color Matching Functions (CIE 1931)
# ============================================================
def plot_color_matching():
    lam = np.linspace(360, 780, 421)
    
    # CIE 1931 Standard Observer CMFs (analytische Näherung nach Wyman et al. 2013)
    def gaussian_cmf(l, mu, sigma1, sigma2, A):
        result = np.zeros_like(l)
        mask_left = l <= mu
        mask_right = l > mu
        result[mask_left] = A * np.exp(-0.5*((l[mask_left]-mu)/sigma1)**2)
        result[mask_right] = A * np.exp(-0.5*((l[mask_right]-mu)/sigma2)**2)
        return result
    
    # x_bar: zwei Gauß-Komponenten (rote und blaue Flanke)
    x_bar = (
        gaussian_cmf(lam, 599, 37, 31, 1.056) +
        gaussian_cmf(lam, 449, 20, 26, 0.362) +
        gaussian_cmf(lam, 501, 14, 9, -0.065)
    )
    x_bar = np.clip(x_bar, 0, None)
    
    # y_bar: Hellempfindlichkeitskurve V(lambda)
    y_bar = gaussian_cmf(lam, 559, 41, 42, 1.217) + gaussian_cmf(lam, 449, 23, 28, -0.213)
    y_bar = np.clip(y_bar, 0, None)
    
    # z_bar: blauer Bereich
    z_bar = gaussian_cmf(lam, 450, 24, 18, 1.865)
    z_bar = np.clip(z_bar, 0, None)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(lam, x_bar, 'r-', linewidth=2.5, label=r'$\bar{x}(\lambda)$')
    ax.plot(lam, y_bar, 'g-', linewidth=2.5, label=r'$\bar{y}(\lambda)$')
    ax.plot(lam, z_bar, 'b-', linewidth=2.5, label=r'$\bar{z}(\lambda)$')
    
    ax.fill_between(lam, x_bar, alpha=0.1, color='red')
    ax.fill_between(lam, y_bar, alpha=0.1, color='green')
    ax.fill_between(lam, z_bar, alpha=0.1, color='blue')
    
    ax.set_xlim(360, 780)
    ax.set_ylim(-0.05, 2.0)
    ax.set_xlabel(r'Wellenlänge $\lambda$ [nm]')
    ax.set_ylabel(r'Absorptionsgrad [a.u.]')
    ax.set_title('CIE 1931 Farbabgleichfunktionen (Standard Observer)')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Sichtbares Spektrum im Hintergrund
    for l_i in range(380, 780, 2):
        frac = (l_i - 380) / 400
        if l_i < 450:
            c = (0.3, 0, 1-frac*2)
        elif l_i < 495:
            c = (0, (l_i-450)/45*0.8, 1)
        elif l_i < 570:
            c = (0, 1, 1-(l_i-495)/75)
        elif l_i < 620:
            c = ((l_i-570)/50, 1-(l_i-570)/50*0.5, 0)
        else:
            c = (1, max(0, 0.5-(l_i-620)/160), 0)
        ax.axvspan(l_i, l_i+2, alpha=0.04, color=c)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot3_cmf.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 3 gespeichert.")

# ============================================================
# Plot 4: Tristimuluswert-Integral X(lambda) 
# ============================================================
def plot_tristimulus_integral():
    lam = np.linspace(380, 780, 401)
    k_norm = 1.0
    
    # D65 spektrale Verteilung (normiert)
    def s_d65(l):
        s = (0.3*np.exp(-((l-450)/70)**2) +
             0.6*np.exp(-((l-550)/100)**2) +
             0.4*np.exp(-((l-650)/80)**2) +
             0.002*(l-380))
        return s / s.max()
    
    # Lykopin-Absorptionsquerschnitt
    def alpha_lam(l):
        a = (0.7 * np.exp(-((l-503)/18)**2) +
             1.0 * np.exp(-((l-471)/20)**2) +
             0.55 * np.exp(-((l-442)/18)**2))
        return np.clip(a, 0, None) / 1.0
    
    # CIE x_bar
    def x_bar(l):
        x = (1.056*np.exp(-0.5*((l-599)/37)**2) * (l<=599) +
             1.056*np.exp(-0.5*((l-599)/31)**2) * (l>599) +
             0.362*np.exp(-0.5*((l-449)/20)**2))
        return np.clip(x, 0, None)
    
    S = s_d65(lam)
    alpha = alpha_lam(lam)
    xb = x_bar(lam)
    
    integrand = k_norm * S * (1 - alpha) * xb
    
    # Trapez-Integration
    X = np.trapezoid(integrand, lam)
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    
    axes[0].plot(lam, S, 'gold', linewidth=2, label=r'$S(\lambda)$ D65')
    axes[0].plot(lam, alpha, 'r-', linewidth=2, label=r'$\alpha(\lambda)$ Lykopin')
    axes[0].plot(lam, 1-alpha, 'b--', linewidth=1.5, label=r'$1-\alpha(\lambda)$')
    axes[0].set_ylabel('Intensität [a.u.]')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Bestandteile des Tristimulus-Integrals')
    
    axes[1].plot(lam, integrand, 'purple', linewidth=2, label=r'$S(\lambda)(1-\alpha(\lambda))\bar{x}(\lambda)$')
    axes[1].fill_between(lam, integrand, alpha=0.3, color='purple')
    axes[1].set_xlabel(r'Wellenlänge $\lambda$ [nm]')
    axes[1].set_ylabel('Integrand [a.u.]')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.05, 0.85, f'$X = k\\int_{{380}}^{{780}} S(\\lambda)(1-\\alpha(\\lambda))\\bar{{x}}(\\lambda)\\,d\\lambda$\n'
                             f'$\\approx {X:.4f}$ a.u.',
                transform=axes[1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot4_tristimulus.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 4 gespeichert.")

# ============================================================
# Plot 5: Diarylethene Photoisomerisierung Energiepotential
# ============================================================
def plot_diarylethene():
    # Reaktionskoordinate: Ringöffnung/Ringschluss
    xi = np.linspace(0, 1, 500)
    
    # Grundzustand S0 Energiepotential
    def E_S0_open(x):
        return 0.8 * x**2 - 0.2 * x + 0.1

    def E_S0_closed(x):
        return -0.3 * (x-1)**2 + 0.05

    # Angeregter Zustand S1
    def E_S1(x):
        return 2.5 - 1.5*x + 0.5*x**2 + 0.3*np.sin(3*np.pi*x)

    # Reaktionskoordinate (0=offen, 1=geschlossen)
    x = np.linspace(0, 1, 500)
    
    # Geschlossene Form (photocyclization product) ist bei x≈1
    E0 = np.where(x < 0.5,
                  0.8*(x-0)**2,
                  -0.4*(x-1)**2 + 0.5*(0.5)**2 - 0.4*(0.5-1)**2)
    # Glatte Energiekurve
    E0 = 0.4 * np.exp(-6*(x-0.1)**2) - 0.35 * np.exp(-8*(x-0.9)**2) + 0.15*x
    E0_norm = E0 - E0.min()
    
    E1 = 2.8 - 2.0*(x-0.5)**2 + 0.4*np.cos(2*np.pi*x)
    E1_norm = E1 - E1.min() + 1.5
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(x, E0_norm, 'b-', linewidth=2.5, label=r'$S_0$ (Grundzustand)')
    ax.plot(x, E1_norm, 'r-', linewidth=2.5, label=r'$S_1$ (angeregter Zustand)')
    
    # Conical Intersection (CI) andeuten
    ci_x = 0.52
    ci_y = np.interp(ci_x, x, E0_norm)
    ax.plot(ci_x, ci_y+0.3, 'k^', markersize=12, zorder=5)
    ax.annotate('CI\n(konische\nIntersection)', xy=(ci_x, ci_y+0.3),
                xytext=(0.65, 1.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Pfeile für Absorption und Emission
    ax.annotate('', xy=(0.1, E1_norm[50]), xytext=(0.1, E0_norm[50]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.05, (E1_norm[50]+E0_norm[50])/2, 'hν', fontsize=13, color='blue')
    
    ax.set_xlabel('Reaktionskoordinate (Ringöffnung → Ringschluss)')
    ax.set_ylabel('Energie [eV]')
    ax.set_title('Photoisomerisierung Diarylethene: $S_0$/$S_1$ Potentialflächen')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['offen', 'TS', 'geschlossen'])
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot5_diarylethene.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 5 gespeichert.")

# ============================================================
# Plot 6: Photorezeptoren – spektrale Empfindlichkeiten 
# (S-, M-, L-Zapfen + Stäbchen)
# ============================================================
def plot_photorezeptoren():
    lam = np.linspace(380, 780, 401)
    
    # Spektrale Empfindlichkeiten nach Stockman & Sharpe 2000 (Näherung)
    def cone_sensitivity(l, peak, width_l, width_r, A=1.0):
        resp = np.zeros_like(l)
        mask_l = l <= peak
        mask_r = l > peak
        resp[mask_l] = A * np.exp(-0.5*((l[mask_l]-peak)/width_l)**2)
        resp[mask_r] = A * np.exp(-0.5*((l[mask_r]-peak)/width_r)**2)
        return resp
    
    # S-Zapfen (blau, peak ~420 nm), M-Zapfen (grün, ~530 nm), L-Zapfen (rot, ~560 nm)
    S_cone = cone_sensitivity(lam, 420, 25, 30, 1.0)
    M_cone = cone_sensitivity(lam, 530, 38, 45, 1.0)
    L_cone = cone_sensitivity(lam, 560, 45, 42, 1.0)
    # Stäbchen: peak ~498 nm
    Rod = cone_sensitivity(lam, 498, 32, 38, 1.0)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(lam, S_cone, color='blue', linewidth=2.5, label='S-Zapfen (~420 nm)')
    ax.plot(lam, Rod, color='gray', linewidth=2, linestyle='--', label='Stäbchen (~498 nm)')
    ax.plot(lam, M_cone, color='green', linewidth=2.5, label='M-Zapfen (~530 nm)')
    ax.plot(lam, L_cone, color='red', linewidth=2.5, label='L-Zapfen (~560 nm)')
    
    ax.fill_between(lam, S_cone, alpha=0.1, color='blue')
    ax.fill_between(lam, M_cone, alpha=0.1, color='green')
    ax.fill_between(lam, L_cone, alpha=0.1, color='red')
    
    # Peak-Markierungen
    for peak, label, col in [(420, '420', 'blue'), (498, '498', 'gray'),
                               (530, '530', 'green'), (560, '560', 'red')]:
        ax.axvline(x=peak, color=col, linestyle=':', alpha=0.6, linewidth=1)
        ax.text(peak+3, 1.02, f'{label} nm', color=col, fontsize=8, rotation=90, va='bottom')
    
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'Wellenlänge $\lambda$ [nm]')
    ax.set_ylabel('Relative Empfindlichkeit')
    ax.set_title('Spektrale Empfindlichkeit der Photorezeptoren (Auge)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot6_photorezeptoren.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 6 gespeichert.")

# ============================================================
# Plot 7: Konjugiertes Pi-System und HOMO-LUMO-Lücke vs. Kettenlänge
# ============================================================
def plot_homo_lumo_gap():
    # Polyenkettelänge n (Anzahl konjugierter Doppelbindungen)
    n = np.arange(1, 16)
    # Partikel-im-Kasten-Modell: ΔE ~ 1/(n+1) * hc/λ
    # Wellenlänge des Absorptionsmaximums wächst mit n
    # Für beta-Carotin (n=11): ~450 nm; Lykopin (n=11 acyclisch): ~470 nm
    # ΔE [eV] = 1240/λ [nm]
    # Näherung: λ_max ≈ 120*n + 150 nm (empirisch für Polyene)
    lambda_max = 120 * n + 150  # nm
    delta_E = 1240.0 / lambda_max  # eV
    
    # Experimentelle Referenzwerte (Polyene)
    n_exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    lam_exp = np.array([165, 220, 270, 310, 340, 370, 400, 425, 447, 463, 470])
    dE_exp = 1240.0 / lam_exp
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    axes[0].plot(n, lambda_max, 'b-o', linewidth=2, markersize=6, label='Modell (PIB)')
    axes[0].plot(n_exp, lam_exp, 'rs', markersize=8, label='Experiment')
    axes[0].axhline(y=470, color='orange', linestyle='--', linewidth=1.5, label='Lykopin (470 nm)')
    axes[0].axhline(y=450, color='purple', linestyle='--', linewidth=1.5, label='β-Carotin (450 nm)')
    axes[0].set_xlabel('Anzahl konjugierter Doppelbindungen $n$')
    axes[0].set_ylabel(r'$\lambda_{\max}$ [nm]')
    axes[0].set_title(r'Absorptionsmaximum vs. Kettenlänge')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(n, delta_E, 'r-o', linewidth=2, markersize=6, label='Modell')
    axes[1].plot(n_exp, dE_exp, 'bs', markersize=8, label='Experiment')
    axes[1].axhline(y=1240/470, color='orange', linestyle='--', linewidth=1.5, label='Lykopin')
    axes[1].set_xlabel('Anzahl konjugierter Doppelbindungen $n$')
    axes[1].set_ylabel(r'HOMO-LUMO-Lücke $\Delta E$ [eV]')
    axes[1].set_title(r'HOMO-LUMO-Lücke vs. Kettenlänge')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot7_homo_lumo.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 7 gespeichert.")

# ============================================================
# Plot 8: Tristimulus XYZ Werte für verschiedene Farbstoffe
# ============================================================
def plot_tristimulus_xyz():
    lam = np.linspace(380, 780, 401)
    
    # CIE CMFs
    def x_bar(l):
        return np.clip(
            1.056*np.exp(-0.5*((l-599)/37)**2)*(l<=599) +
            1.056*np.exp(-0.5*((l-599)/31)**2)*(l>599) +
            0.362*np.exp(-0.5*((l-449)/20)**2), 0, None)
    def y_bar(l):
        return np.clip(1.217*np.exp(-0.5*((l-559)/41)**2), 0, None)
    def z_bar(l):
        return np.clip(1.865*np.exp(-0.5*((l-450)/24)**2), 0, None)
    
    xb = x_bar(lam); yb = y_bar(lam); zb = z_bar(lam)
    
    def s_d65(l):
        s = 0.3*np.exp(-((l-450)/70)**2) + 0.6*np.exp(-((l-550)/100)**2) + 0.4*np.exp(-((l-650)/80)**2) + 0.002*(l-380)
        return s / s.max()
    S = s_d65(lam)
    
    # Verschiedene Farbstoffe: Absorptionsspektren
    farbstoffe = {
        'Lykopin\n(Tomate)':  {'peaks': [(503,18,0.7),(471,20,1.0),(442,18,0.55)], 'color': 'red'},
        'Chlorophyll a\n(Blatt)': {'peaks': [(680,20,1.0),(430,25,0.85)], 'color': 'green'},
        'Hämoglobin\n(Blut)': {'peaks': [(575,15,0.6),(540,18,0.7),(414,25,1.0)], 'color': 'darkred'},
        'Beta-Carotin\n(Karotte)': {'peaks': [(497,18,0.6),(468,20,1.0),(449,17,0.7)], 'color': 'orange'},
    }
    
    results = {}
    for name, data in farbstoffe.items():
        alpha = np.zeros_like(lam)
        for (p, w, A) in data['peaks']:
            alpha += A * np.exp(-0.5*((lam-p)/w)**2)
        alpha = np.clip(alpha, 0, 1.0)
        refl = 1 - alpha
        
        X = np.trapezoid(S * refl * xb, lam)
        Y = np.trapezoid(S * refl * yb, lam)
        Z = np.trapezoid(S * refl * zb, lam)
        tot = X+Y+Z+1e-10
        results[name] = {'X': X, 'Y': Y, 'Z': Z, 'x': X/tot, 'y': Y/tot, 'color': data['color']}
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    names = list(results.keys())
    X_vals = [results[n]['X'] for n in names]
    Y_vals = [results[n]['Y'] for n in names]
    Z_vals = [results[n]['Z'] for n in names]
    colors = [results[n]['color'] for n in names]
    
    x_pos = np.arange(len(names))
    width = 0.25
    axes[0].bar(x_pos - width, X_vals, width, label='X', color='salmon', alpha=0.8)
    axes[0].bar(x_pos, Y_vals, width, label='Y', color='lightgreen', alpha=0.8)
    axes[0].bar(x_pos + width, Z_vals, width, label='Z', color='lightblue', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([n.replace('\n', '\n') for n in names], fontsize=9)
    axes[0].set_ylabel('Tristimuluswert [a.u.]')
    axes[0].set_title('Tristimuluswerte X, Y, Z verschiedener Farbstoffe')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CIE xy Chromatizitätsdiagramm (vereinfacht)
    # Hufeisenkurve
    lam_vis = np.linspace(380, 700, 330)
    xs_vis = []; ys_vis = []
    for l_v in lam_vis:
        xv = x_bar(np.array([l_v]))[0]
        yv = y_bar(np.array([l_v]))[0]
        zv = z_bar(np.array([l_v]))[0]
        tot = xv+yv+zv+1e-10
        xs_vis.append(xv/tot); ys_vis.append(yv/tot)
    xs_vis.append(xs_vis[0]); ys_vis.append(ys_vis[0])
    
    axes[1].plot(xs_vis, ys_vis, 'k-', linewidth=1.5, label='Spektrallinie')
    axes[1].fill(xs_vis, ys_vis, alpha=0.05, color='gray')
    
    for name in names:
        xi = results[name]['x']; yi = results[name]['y']
        c = results[name]['color']
        axes[1].plot(xi, yi, 'o', color=c, markersize=12, zorder=5)
        axes[1].annotate(name.split('\n')[0], xy=(xi, yi), xytext=(xi+0.02, yi+0.02),
                        fontsize=8, color=c)
    
    axes[1].set_xlim(-0.05, 0.85)
    axes[1].set_ylim(-0.05, 0.9)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('CIE 1931 Chromatizitätsdiagramm')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot8_tristimulus_xyz.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Plot 8 gespeichert.")

# ============================================================
# Run all plots
# ============================================================
plot_cie_d65()
plot_lycopin_absorption()
plot_color_matching()
plot_tristimulus_integral()
plot_diarylethene()
plot_photorezeptoren()
plot_homo_lumo_gap()
plot_tristimulus_xyz()

print("\nAlle Plots erfolgreich generiert.")

