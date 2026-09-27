#!/usr/bin/env python3
"""
Erzeugung aller Matplotlib-Abbildungen für
dusgrxdnastr.tex
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import math
import os

# ── Ausgabeverzeichnis ────────────────────────────────────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Farben (an das TeX-Farbschema angelehnt) ─────────────────────────────────
C_BLUE   = '#19468C'
C_RED    = '#B4321E'
C_GREEN  = '#1E6432'
C_ORANGE = '#C87820'
C_GRAY   = '#3C3C46'
C_LGRAY  = '#E8E8F0'

plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.6,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 1: Kompressionsfaktor als Funktion des Redundanzgrades ρ
# ═══════════════════════════════════════════════════════════════════════════════
def fig_compression():
    rho = np.linspace(0, 0.95, 500)
    faktor = 1.0 / (1.0 - rho)
    dichte_native = 2.15e15          # Byte/g
    dichte_eff    = dichte_native * faktor

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Linkes Panel: Kompressionsfaktor ──────────────────────────────────────
    ax = axes[0]
    ax.plot(rho, faktor, color=C_BLUE, lw=2.2, label=r'$(1-\rho)^{-1}$')
    ax.axvline(0.8, color=C_RED,    lw=1.4, ls='--', label=r'$\rho=0.8$')
    ax.axvline(0.5, color=C_ORANGE, lw=1.4, ls='--', label=r'$\rho=0.5$')
    ax.annotate(r'$\times 5$', xy=(0.8, 5), xytext=(0.65, 7),
                arrowprops=dict(arrowstyle='->', color=C_RED),
                color=C_RED, fontsize=11)
    ax.annotate(r'$\times 2$', xy=(0.5, 2), xytext=(0.35, 4),
                arrowprops=dict(arrowstyle='->', color=C_ORANGE),
                color=C_ORANGE, fontsize=11)
    ax.set_xlabel(r'Redundanzgrad $\rho$')
    ax.set_ylabel(r'Kompressionsfaktor $(1-\rho)^{-1}$')
    ax.set_title('Kompressionsfaktor des Subgraph Algorithmus')
    ax.set_xlim(0, 0.95)
    ax.set_ylim(1, 22)
    ax.legend(loc='upper left')

    # ── Rechtes Panel: effektive Speicherdichte ───────────────────────────────
    ax2 = axes[1]
    rho_vals = [0.0, 0.5, 0.8, 0.9, 0.95]
    labels   = [r'$\rho=0$ (nativ)', r'$\rho=0.5$', r'$\rho=0.8$',
                r'$\rho=0.9$', r'$\rho=0.95$']
    farben   = [C_GRAY, C_ORANGE, C_GREEN, C_BLUE, C_RED]
    dichten  = [dichte_native / (1 - r) / 1e15 for r in rho_vals]
    bars = ax2.bar(labels, dichten, color=farben, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax2.set_ylabel(r'Effektive Dichte [EB/g]')
    ax2.set_title('Effektive Speicherdichte ($d_\\mathrm{eff}(\\rho) = d_\\mathrm{DNA}/(1-\\rho)$)')
    for bar, d in zip(bars, dichten):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{d:.1f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticklabels(labels, rotation=15, ha='right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_compression.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_compression.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 2: Signatur-Kollisionsrate: Subgraph vs. naive Methoden
# ═══════════════════════════════════════════════════════════════════════════════
def fig_collision():
    n = np.arange(1, 60)

    naive_hash   = 1.0 / n
    inode_hash   = 1.0 / n**2
    subgraph_sig = 1.0 / (2**n)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(n, naive_hash,   color=C_GRAY,   lw=2.0, ls='-',
                label=r'Naives Hashing $1/n$')
    ax.semilogy(n, inode_hash,   color=C_ORANGE, lw=2.0, ls='--',
                label=r'Inode-Hash $1/n^2$')
    ax.semilogy(n, subgraph_sig, color=C_BLUE,   lw=2.5, ls='-',
                label=r'Subgraph-Signatur $1/2^n$')

    # ── Sicherheitszone ───────────────────────────────────────────────────────
    ax.axhline(1e-12, color=C_GREEN, lw=1.2, ls=':', alpha=0.7)
    ax.text(50, 5e-13, 'Krypto-Sicherheitsniveau\n($10^{-12}$)',
            ha='center', fontsize=9, color=C_GREEN)
    ax.fill_between(n, 1e-40, 1e-12, alpha=0.07, color=C_GREEN)

    ax.set_xlabel(r'Graphgröße $n$ (Anzahl Knoten / Basenlänge)')
    ax.set_ylabel('Kollisionswahrscheinlichkeit')
    ax.set_title('Kollisionsrate der Subgraph-Signatur im Vergleich')
    ax.set_xlim(1, 59)
    ax.set_ylim(1e-20, 1.2)
    ax.legend(loc='upper right')

    # ── Annotation ────────────────────────────────────────────────────────────
    ax.annotate('Für $n=40$:\nSubgraph $\\approx 10^{-12}$\nNaiv $\\approx 0.025$',
                xy=(40, 1/40), xytext=(30, 0.1),
                arrowprops=dict(arrowstyle='->', color=C_GRAY),
                fontsize=9, color=C_GRAY)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_collision.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_collision.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 3: Speicherdichte-Vergleich verschiedener Medien (log)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_density():
    medien  = ['Magnetic\nTape', 'HDD\n(7200 rpm)', 'Flash\nSSD', 'Optisch\n(Blu-ray)',
               'DRAM', 'DNA\n(nativ)', 'DNA\n+DSS (ρ=0.8)']
    # Byte pro cm³ (grobe Richtwerte)
    dichten = [1e6, 1e8, 3e9, 5e8, 2e10, 2.15e21, 2.15e21 * 5]
    farben  = [C_GRAY]*5 + [C_BLUE, C_GREEN]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(medien, dichten, color=farben, edgecolor='white', linewidth=0.8, alpha=0.88)
    ax.set_yscale('log')
    ax.set_ylabel(r'Speicherdichte [Byte/cm$^3$]')
    ax.set_title('Speicherdichte-Vergleich verschiedener Speichermedien (logarithmische Skala)')

    # ── Beschriftungen ────────────────────────────────────────────────────────
    einheiten = ['1 MB', '100 MB', '3 GB', '500 MB', '20 GB',
                 '215 EB', '1075 EB\n(× 5)']
    for bar, label in zip(bars, einheiten):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.8,
                label, ha='center', va='bottom', fontsize=8.5)

    # ── Doppelpfeil DNA vs. Flash ─────────────────────────────────────────────
    ax.annotate('', xy=(5, 2.15e21), xytext=(2, 3e9),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.5))
    ax.text(3.5, 1e15, '14 Größen-\nordnungen', ha='center', color=C_RED, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_density.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_density.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 4: Shannon-Kapazität als Funktion der Fehlerrate
# ═══════════════════════════════════════════════════════════════════════════════
def fig_capacity():
    eps = np.linspace(0.001, 0.45, 500)

    def H4(e):
        p = [1-e, e/3, e/3, e/3]
        return -sum(pi * np.log2(pi) for pi in p if pi > 0)

    C_basis = np.array([2 - H4(e) for e in eps])
    rhos    = [0.5, 0.8, 0.9]
    ks      = [50, 100, 150]   # Oligolänge

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Linkes Panel: Kapazität mit Subgraph-Komprimierung ───────────────────
    ax = axes[0]
    ax.plot(eps, C_basis, color=C_GRAY, lw=2.0, label=r'$C_\mathrm{basis}(\varepsilon)$ (kein Kompr.)')
    for rho, c in zip(rhos, [C_ORANGE, C_GREEN, C_BLUE]):
        # k = 100 (Referenz)
        k = 100
        boost = 1 + np.log2(1 + 1/rho) / k
        C_dss = C_basis * boost
        ax.plot(eps, C_dss, color=c, lw=2.0,
                label=rf'$C_\mathrm{{DSS}}$, $\rho={rho}$')
    ax.axvline(1e-3, color=C_RED, lw=1.2, ls=':', label=r'Illumina $\varepsilon_s$')
    ax.set_xlabel(r'Fehlerrate $\varepsilon$')
    ax.set_ylabel(r'Kapazität [Bit/Symbol]')
    ax.set_title('Shannon-Kapazität des DNA-Subgraph-Speicherkanals')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 2.7)

    # ── Rechtes Panel: Nutzbare Bits pro Oligo vs. Oligolänge ────────────────
    ax2 = axes[1]
    k_range = np.arange(50, 1001, 10)
    r_rs = 0.13   # Reed-Solomon Overhead
    for rho, c in zip(rhos, [C_ORANGE, C_GREEN, C_BLUE]):
        I_max = k_range * (2 - r_rs) / (1 - rho)
        ax2.plot(k_range, I_max / 8, color=c, lw=2.0, label=rf'$\rho={rho}$')
    ax2.axvline(150, color=C_GRAY, lw=1.3, ls='--', label='Illumina (150 bp)')
    ax2.set_xlabel(r'Oligolänge $k$ [Basen]')
    ax2.set_ylabel(r'$I_\mathrm{max}$ [Byte/Oligo]')
    ax2.set_title(r'Maximale Informationsdichte $I_\mathrm{max}(k, r_\mathrm{rs}, \rho)$')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_capacity.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_capacity.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 5: Haltbarkeitsmodell P_intakt(t) für verschiedene ρ und Temperaturen
# ═══════════════════════════════════════════════════════════════════════════════
def fig_durability():
    # Degradationsraten (Arrhenius-Modell, grobe Annäherung)
    lambdas = {
        r'Raumtemp. ($20\,°C$)':   1e-6,   # 1/Jahr
        r'Kühlraum ($-20\,°C$)':   1e-7,
        r'Tiefkühlung ($-80\,°C$)':1e-8,
        r'Flüss. N$_2$ ($-196\,°C$)': 1e-9,
    }
    farben_lam = [C_RED, C_ORANGE, C_BLUE, C_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Linkes Panel: P_intakt(t) für verschiedene Temperaturen ──────────────
    ax = axes[0]
    t = np.logspace(0, 10, 1000)  # Jahre
    for (lbl, lam), col in zip(lambdas.items(), farben_lam):
        p = np.exp(-lam * t)
        ax.semilogx(t, p * 100, color=col, lw=2.0, label=lbl)
    ax.axhline(20, color=C_GRAY, lw=1.2, ls=':', label=r'$P=20\%$ (ρ=0.8-Grenze)')
    ax.set_xlabel('Zeit [Jahre]')
    ax.set_ylabel(r'$P_\mathrm{intakt}(t)$ [%]')
    ax.set_title(r'Haltbarkeitsmodell: $P_\mathrm{intakt}(t) = e^{-\lambda t}$')
    ax.legend(fontsize=9)
    ax.set_xlim(1, 1e10)
    ax.set_ylim(0, 105)

    # ── Rechtes Panel: Garantierte Lebensdauer t* vs. ρ ──────────────────────
    ax2 = axes[1]
    rho_vals = np.linspace(0.01, 0.99, 500)
    for (lbl, lam), col in zip(lambdas.items(), farben_lam):
        t_star = -np.log(1 - rho_vals) / lam
        ax2.semilogy(rho_vals, t_star, color=col, lw=2.0, label=lbl)
    ax2.axvline(0.8, color=C_GRAY, lw=1.2, ls='--', label=r'$\rho=0.8$')
    ax2.set_xlabel(r'Redundanzgrad $\rho$')
    ax2.set_ylabel(r'Garantierte Lebensdauer $t^*$ [Jahre]')
    ax2.set_title(r'Garantierte Lebensdauer $t^* = -\ln(1-\rho)/\lambda$')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_durability.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_durability.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 6: Kostenprojektions-Roadmap 2024–2035
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cost_roadmap():
    jahre   = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035])
    c_syn   = np.array([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4, 1e-5])
    c_eff   = c_syn * 0.2   # Faktor 5 durch DSS (ρ=0.8)
    tape_bm = np.full_like(jahre, 1e-9, dtype=float)  # Tape-Benchmark $/Byte

    # Kosten in $/Byte umrechnen: 1 Base speichert 2 Bit = 0.25 Byte
    c_byte    = c_syn  / 0.25
    c_eff_byte= c_eff  / 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Linkes Panel: Synthesekosten ─────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(jahre, c_syn,  'o-', color=C_GRAY,  lw=2.0, label=r'$C_\mathrm{syn}$ ($/Base, ohne DSS)')
    ax.semilogy(jahre, c_eff,  's-', color=C_BLUE,  lw=2.0, label=r'$C_\mathrm{eff}$ ($/Base, mit DSS $\rho=0.8$)')
    ax.fill_between(jahre, c_syn, c_eff, alpha=0.15, color=C_BLUE,
                    label='DSS-Einsparung (80 %)')
    ax.set_xlabel('Jahr')
    ax.set_ylabel('Synthesekosten [$/Base]')
    ax.set_title('Kostenprojektions-Roadmap für DNA-Synthese')
    ax.legend(fontsize=9)
    ax.set_xlim(2024, 2035)

    # ── Rechtes Panel: Break-Even gegenüber Tape ──────────────────────────────
    ax2 = axes[1]
    ax2.semilogy(jahre, c_byte,     'o-', color=C_GRAY,  lw=2.0, label='DNA ohne DSS ($/Byte)')
    ax2.semilogy(jahre, c_eff_byte, 's-', color=C_BLUE,  lw=2.0, label='DNA mit DSS ($/Byte)')
    ax2.semilogy(jahre, tape_bm,    '--', color=C_RED,   lw=1.8, label='Tape-Speicher ($/Byte)')
    # Break-even-Markierung (circa 2033)
    ax2.axvline(2033, color=C_GREEN, lw=1.4, ls=':', label='Break-Even ~2033')
    ax2.set_xlabel('Jahr')
    ax2.set_ylabel('Speicherkosten [$/Byte]')
    ax2.set_title('Break-Even-Analyse: DNA vs. Tape-Speicher')
    ax2.legend(fontsize=9)
    ax2.set_xlim(2024, 2035)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_cost_roadmap.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_cost_roadmap.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 7: Zugriffszeit-Vergleich (Index vs. brute-force vs. andere Medien)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_access_time():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Linkes Panel: Zugriffszeit verschiedener Medien ───────────────────────
    medien_names = ['L1-Cache', 'L3-Cache', 'DRAM', 'NVMe SSD',
                    'HDD', 'Tape (Spool)', 'DNA+Index', 'DNA (direkt)']
    access_ms    = [1e-7, 1e-5, 1e-4, 0.05, 5.0, 60000, 17000, 1020000]
    farben_a     = [C_GREEN]*4 + [C_ORANGE]*2 + [C_BLUE, C_RED]

    ax = axes[0]
    bars = ax.barh(medien_names, access_ms, color=farben_a, edgecolor='white', alpha=0.88)
    ax.set_xscale('log')
    ax.set_xlabel('Zugriffszeit [ms]')
    ax.set_title('Zugriffszeit-Vergleich verschiedener Speichermedien')
    for bar, v in zip(bars, access_ms):
        label = f'{v:.0e}' if v < 1 else (f'{v:.0f} ms' if v < 1000 else f'{v/60000:.1f} min')
        ax.text(v * 1.2, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8.5)

    # ── Rechtes Panel: Skalierung der Index-Zugriffszeit mit Pool-Größe ───────
    ax2 = axes[1]
    M = np.logspace(3, 12, 500)
    t_index     = 0.01 * np.log2(M)  # O(log M), in Sekunden
    t_linear    = 0.001 * M / 1e6    # O(M) naiv
    t_subgraph  = 0.1  * np.log2(M)  # mit Subgraph-Verifikation

    ax2.loglog(M, t_index,    color=C_BLUE,  lw=2.2, label=r'Signatur-Index $O(\log M)$')
    ax2.loglog(M, t_subgraph, color=C_GREEN, lw=2.0, ls='--',
               label=r'Index + Subgraph-Verif. $O(\log M \cdot k^2)$')
    ax2.loglog(M, t_linear,   color=C_RED,   lw=2.0, ls=':',
               label=r'Naiver Scan $O(M)$')
    ax2.axvline(1e6, color=C_GRAY, lw=1.2, ls=':', label=r'$M = 10^6$ (1 Mio. Oligos)')
    ax2.set_xlabel(r'Pool-Größe $M$ (Anzahl Oligomere)')
    ax2.set_ylabel('Zugriffszeit [s]')
    ax2.set_title(r'Skalierung der Zugriffszeit mit $M$')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_access.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_access.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 8: SDD-Komplexitätsvergleich (Inode → DNA)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_complexity():
    n = np.arange(1, 30)

    naive_perm    = np.array([float(math.factorial(ni)) * ni**2 if ni <= 12 else np.nan for ni in n])
    sdd_cubic     = n**3
    hash_linear   = n
    sig_quadratic = n**2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(n, hash_linear,   color=C_GREEN,  lw=2.0, ls='-',
                label=r'Inode-Hash $O(n)$')
    ax.semilogy(n, sig_quadratic, color=C_ORANGE, lw=2.0, ls='--',
                label=r'Signaturberechnung $O(n^2)$')
    ax.semilogy(n, sdd_cubic,     color=C_BLUE,   lw=2.5, ls='-',
                label=r'\textbf{SDD/DSS-Algorithmus $O(n^3)$}')
    valid = ~np.isnan(naive_perm)
    ax.semilogy(n[valid], naive_perm[valid], color=C_RED, lw=2.0, ls=':',
                label=r'Naive Permutation $O(n! \cdot n^2)$')

    ax.axvline(10, color=C_GRAY, lw=1.0, ls=':', alpha=0.7)
    ax.text(10.2, 1.5, 'praktischer\nEinsatzbereich', fontsize=9, color=C_GRAY)
    ax.set_xlabel(r'Problemgröße $n$ (Inodes / Oligomerlänge $k$)')
    ax.set_ylabel('Operationsanzahl (log)')
    ax.set_title('Komplexitätsvergleich: Dateisystem- und DNA-Subgraph-Algorithmen')
    ax.legend(loc='upper left')
    ax.set_xlim(1, 29)
    ax.set_ylim(0.5, 1e20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_complexity.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_complexity.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Abbildung 9: Epigenetische Erweiterung – 3-Bit/Base-Dichte
# ═══════════════════════════════════════════════════════════════════════════════
def fig_epigenetic():
    k = np.arange(50, 1001, 10)
    r_rs = 0.13
    rho  = 0.8

    # Standard 2-Bit/Base
    I_standard = k * (2 - r_rs) / (1 - rho) / 8          # Byte/Oligo
    # Epigenetisch 3-Bit/Base (methylierte Kanten)
    I_epi      = k * (3 - r_rs) / (1 - rho) / 8
    # Ohne Komprimierung
    I_no_comp  = k * (2 - r_rs) / 8

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, I_no_comp,  color=C_GRAY,   lw=2.0, ls=':', label=r'Ohne DSS-Kompr. (2 Bit/Base)')
    ax.plot(k, I_standard, color=C_BLUE,   lw=2.2, ls='-', label=r'DSS 2 Bit/Base ($\rho=0.8$)')
    ax.plot(k, I_epi,      color=C_GREEN,  lw=2.2, ls='--',
            label=r'DSS epigenetisch 3 Bit/Base ($\rho=0.8$)')

    ax.fill_between(k, I_standard, I_epi, alpha=0.15, color=C_GREEN,
                    label='+50 % durch Methylierungskodierung')
    ax.axvline(150, color=C_ORANGE, lw=1.3, ls='-.', label='Illumina (150 bp)')
    ax.axvline(500, color=C_RED,    lw=1.3, ls='-.', label='TdT-Enzym (~500 bp)')

    ax.set_xlabel(r'Oligolänge $k$ [Basen]')
    ax.set_ylabel(r'$I_\mathrm{max}$ [Byte/Oligo]')
    ax.set_title('Informationsdichte: Standard vs. Epigenetische Erweiterung')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dna_fig_epigenetic.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('✓ dna_fig_epigenetic.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Alle Abbildungen erzeugen
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Erzeuge Abbildungen …')
    fig_compression()
    fig_collision()
    fig_density()
    fig_capacity()
    fig_durability()
    fig_cost_roadmap()
    fig_access_time()
    fig_complexity()
    fig_epigenetic()
    print('Alle Abbildungen erzeugt.')

