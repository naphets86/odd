#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLEX-Standard: Matplotlib-Visualisierungen
Generiert 5 Plots als einzelne PDF-Dateien.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
ORANGE    = '#C87010'
PURPLE    = '#6B2D8E'
GREY      = '#5A5A6A'

# ============================================================
# Plot 1: QoS-Zeitverlauf der Suchmaschineanbieter (8 Quartale)
# ============================================================
def plot1_qos_zeitverlauf(outdir='.'):
    quarters = ['Q1/2024','Q2/2024','Q3/2024','Q4/2024',
                'Q1/2025','Q2/2025','Q3/2025','Q4/2025']
    x = np.arange(len(quarters))

    google = np.array([82.1, 83.8, 83.0, 85.5, 85.2, 87.1, 86.4, 88.3])
    bing   = np.array([75.3, 74.8, 76.4, 77.9, 77.2, 79.0, 80.1, 81.4])
    ddg    = np.array([70.5, 72.1, 74.0, 73.3, 75.4, 76.8, 78.0, 79.5])
    ecosia = np.array([65.0, 66.4, 67.9, 67.0, 69.1, 71.2, 72.7, 73.9])
    brave  = np.array([60.2, 62.9, 64.8, 66.5, 69.7, 71.9, 74.1, 75.8])

    fig, ax = plt.subplots(figsize=(9.5, 5))
    kw = dict(lw=2, markersize=6)
    ax.plot(x, google, 'o-', color=MAINBLUE,  label='Google Search', **kw)
    ax.plot(x, bing,   's-', color=ACCENTRED, label='Bing',          **kw)
    ax.plot(x, ddg,    '^-', color=DARKGREEN, label='DuckDuckGo',    **kw)
    ax.plot(x, ecosia, 'D-', color=ORANGE,    label='Ecosia',        **kw)
    ax.plot(x, brave,  'v-', color=PURPLE,    label='Brave Search',  **kw)

    # Vertikale Trennlinien an Quartalsenden
    for xi in x:
        ax.axvline(x=xi, color='lightgrey', lw=0.8, zorder=0)

    # Wettbewerbsentscheidungspunkte markieren
    for xi in x[::1]:
        ax.axvspan(xi - 0.45, xi + 0.45, alpha=0.04, color=MAINBLUE, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=30, ha='right')
    ax.set_ylabel(r'Gewichteter QoS-Score $Q_{\mathbf{w}}$ (0–100)')
    ax.set_xlabel('Wettbewerbsquartal')
    ax.set_title('Abbildung 1: QoS-Zeitverlauf der Suchmaschineanbieter\nim FLEX-Standard-Wettbewerb (8 Quartale)')
    ax.set_ylim(52, 96)
    ax.legend(loc='lower right', framealpha=0.92, fontsize=10)

    # Sieger-Annotierung letztes Quartal
    ax.annotate('Sieger\nQ4/2025', xy=(7, 88.3), xytext=(6.3, 93),
                arrowprops=dict(arrowstyle='->', color=MAINBLUE, lw=1.2),
                fontsize=9, color=MAINBLUE, ha='center')

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot1_qos_zeitverlauf.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('Plot 1 gespeichert: plot1_qos_zeitverlauf.pdf')


# ============================================================
# Plot 2: FLEX-Routing nach geografischer Region
# ============================================================
def plot2_routing_regionen(outdir='.'):
    regions = ['DE', 'EU-West', 'US', 'IN', 'BR', 'JP']
    engines = ['Google Search', 'Bing', 'DuckDuckGo', 'Ecosia', 'Brave Search']

    # Routing-Anteile in % (Summe je Region = 100)
    data = np.array([
        [42, 18, 16, 14, 10],  # DE
        [48, 17, 14, 12,  9],  # EU-West
        [59, 24,  8,  5,  4],  # US
        [54, 21, 13,  7,  5],  # IN
        [51, 23, 13,  8,  5],  # BR
        [62, 22,  8,  5,  3],  # JP
    ])

    colors = [MAINBLUE, ACCENTRED, DARKGREEN, ORANGE, PURPLE]
    bar_w = 0.55
    xpos = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bottom = np.zeros(len(regions))
    for i, (eng, col) in enumerate(zip(engines, colors)):
        bars = ax.bar(xpos, data[:, i], bar_w, bottom=bottom,
                      label=eng, color=col, alpha=0.88, edgecolor='white', lw=0.6)
        # Prozentanzeige nur wenn Segment >= 8%
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if h >= 8:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottom[j] + h/2,
                        f'{int(h)}%', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
        bottom += data[:, i]

    ax.set_xticks(xpos)
    ax.set_xticklabels(regions)
    ax.set_ylabel('Anteil der gerouteten Suchanfragen (%)')
    ax.set_xlabel('Geografische Region')
    ax.set_title('Abbildung 2: FLEX-Routing – Suchmaschinenzuweisung\nnach geografischer Region und regulatorischen Auflagen')
    ax.set_ylim(0, 112)
    ax.legend(loc='upper right', framealpha=0.92, fontsize=10,
              ncol=2, bbox_to_anchor=(1.0, 1.0))
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot2_routing_regionen.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('Plot 2 gespeichert: plot2_routing_regionen.pdf')


# ============================================================
# Plot 3: Latenzen verschiedener FLEX-Dienstketten
# ============================================================
def plot3_latenz_ketten(outdir='.'):
    chains = ['STT\n-> TTS', 'STT -> Trans.\n-> TTS',
              'Trans.\n-> TTS', 'STT\n-> Trans.']
    bw_labels = ['Hoch  (>50 Mbit/s)', 'Mittel (10–50 Mbit/s)', 'Niedrig (<10 Mbit/s)']

    # Latenzen in ms (gemessen / simuliert)
    data = np.array([
        [118, 182, 208, 162],   # Hoch
        [183, 287, 246, 228],   # Mittel
        [375, 518, 415, 468],   # Niedrig
    ])

    x = np.arange(len(chains))
    width = 0.24
    colors = [MAINBLUE, ORANGE, ACCENTRED]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (bw, col) in enumerate(zip(bw_labels, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[i], width, label=bw,
                      color=col, alpha=0.88, edgecolor='white', lw=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                    f'{int(h)} ms', ha='center', va='bottom',
                    fontsize=8.5, color=GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(chains, fontsize=10)
    ax.set_ylabel('Ende-zu-Ende-Latenz (ms)')
    ax.set_xlabel('FLEX-Dienstkette')
    ax.set_title('Abbildung 3: Latenzen der FLEX-Dienstketten\nnach Bandbreitenklasse (STT = Speech-to-Text, Trans. = Translation, TTS = Text-to-Speech)')
    ax.legend(framealpha=0.92, fontsize=10)
    ax.set_ylim(0, 610)

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot3_latenz_ketten.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('Plot 3 gespeichert: plot3_latenz_ketten.pdf')


# ============================================================
# Plot 4: Wettbewerbs-Score-Evolution (8 Quartale, 4 Anbieter)
# ============================================================
def plot4_wettbewerb_ranking(outdir='.'):
    quarters = ['Q1/24','Q2/24','Q3/24','Q4/24',
                'Q1/25','Q2/25','Q3/25','Q4/25']
    x = np.arange(len(quarters))

    p1 = np.array([87.5, 85.9, 88.7, 90.8, 89.9, 91.6, 90.5, 93.8])
    p2 = np.array([82.3, 83.8, 85.1, 84.4, 86.0, 87.9, 88.6, 87.7])
    p3 = np.array([79.1, 80.4, 82.0, 83.2, 85.5, 84.3, 86.1, 87.4])
    p4 = np.array([70.2, 73.1, 75.4, 77.8, 80.0, 81.9, 84.4, 84.9])

    fig, ax = plt.subplots(figsize=(9.5, 5))

    kw = dict(lw=2.5, markersize=7)
    ax.plot(x, p1, 'o-', color=MAINBLUE,  label='Anbieter A (TTS-Leader)', **kw)
    ax.plot(x, p2, 's-', color=ACCENTRED, label='Anbieter B',               **kw)
    ax.plot(x, p3, '^-', color=DARKGREEN, label='Anbieter C',               **kw)
    ax.plot(x, p4, 'D-', color=ORANGE,    label='Anbieter D (Newcomer)',     **kw)

    for xi in x:
        ax.axvline(x=xi, color='lightgrey', lw=1.0, zorder=0)

    # Wettbewerbs-Schwellenwert
    ax.axhline(y=80.0, color=GREY, lw=1.2, ls=':', label='Min.-Schwelle $q_{\\min}=80$')

    # Sieger-Markierung
    all_s = np.vstack([p1, p2, p3, p4])
    winner_idx = np.argmax(all_s, axis=0)
    colors_arr = [MAINBLUE, ACCENTRED, DARKGREEN, ORANGE]
    labels_arr = ['A', 'B', 'C', 'D']
    for qi, wi in enumerate(winner_idx):
        score = all_s[wi, qi]
        ax.scatter(qi, score, s=120, zorder=5,
                   color=colors_arr[wi], edgecolors='white', lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=20, ha='right')
    ax.set_ylabel(r'Gewichteter Wettbewerbsscore $Q_{\mathrm{comp}}$')
    ax.set_xlabel('Wettbewerbsquartal')
    ax.set_title('Abbildung 4: Evolution der Wettbewerbsscores im FLEX-Standard\n(Kreis = Quartalsieger)')
    ax.set_ylim(63, 99)
    ax.legend(loc='lower right', framealpha=0.92, fontsize=10)

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot4_wettbewerb_ranking.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('Plot 4 gespeichert: plot4_wettbewerb_ranking.pdf')


# ============================================================
# Plot 5: Subgraph-Kompatibilitätsmatrix (Schnittstellen x Anbieter)
# ============================================================
def plot5_subgraph_kompatibilitaet(outdir='.'):
    interfaces = ['$I_{\\mathrm{Search}}$', '$I_{\\mathrm{TTS}}$',
                  '$I_{\\mathrm{STT}}$', '$I_{\\mathrm{Trans}}$', '$I_{\\mathrm{NLP}}$']
    services   = ['Google\nSearch', 'Bing', 'Azure\nCognitive', 'AWS\nTranscribe',
                  'DeepL', 'LibreTrans.', 'Google\nNLP']

    # 1 = kompatibel (Subgraph enthalten), 0 = nicht kompatibel
    compat = np.array([
        [1, 1, 0, 0, 0, 0, 0],  # I_Search
        [0, 0, 1, 0, 0, 0, 1],  # I_TTS
        [0, 0, 1, 1, 0, 0, 0],  # I_STT
        [0, 0, 0, 0, 1, 1, 0],  # I_Trans
        [0, 0, 1, 0, 0, 0, 1],  # I_NLP
    ], dtype=float)

    # Laufzeiten des Subgraph Algorithmus (in ms, simuliert)
    runtimes = np.array([
        [1.2, 1.1, 0.9, 0.8, 0.7, 0.8, 0.9],
        [0.8, 0.9, 1.4, 1.1, 0.6, 0.7, 1.5],
        [0.7, 0.8, 1.3, 1.6, 0.5, 0.6, 0.9],
        [0.6, 0.7, 0.8, 0.7, 1.8, 1.7, 0.6],
        [0.5, 0.6, 1.2, 0.9, 0.7, 0.6, 1.4],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Kompatibilitätsmatrix ---
    ax = axes[0]
    cmap_compat = matplotlib.colors.ListedColormap(['#FFDDDD', '#D4EDD4'])
    im = ax.imshow(compat, cmap=cmap_compat, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(services)))
    ax.set_yticks(np.arange(len(interfaces)))
    ax.set_xticklabels(services, fontsize=9.5)
    ax.set_yticklabels(interfaces, fontsize=10)
    ax.set_title('(a) Kompatibilitätsmatrix', fontsize=11)
    ax.set_xlabel('Dienstanbieter', fontsize=10)
    ax.set_ylabel('Schnittstelle', fontsize=10)
    for i in range(len(interfaces)):
        for j in range(len(services)):
            v = compat[i, j]
            txt = 'JA' if v == 1 else 'NEIN'
            col = DARKGREEN if v == 1 else ACCENTRED
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=9, color=col, fontweight='bold')
    ax.grid(False)

    # --- Laufzeit-Heatmap ---
    ax2 = axes[1]
    im2 = ax2.imshow(runtimes, cmap='Blues', aspect='auto')
    ax2.set_xticks(np.arange(len(services)))
    ax2.set_yticks(np.arange(len(interfaces)))
    ax2.set_xticklabels(services, fontsize=9.5)
    ax2.set_yticklabels(interfaces, fontsize=10)
    ax2.set_title('(b) Laufzeit Subgraph-Alg. (ms)', fontsize=11)
    ax2.set_xlabel('Dienstanbieter', fontsize=10)
    for i in range(len(interfaces)):
        for j in range(len(services)):
            ax2.text(j, i, f'{runtimes[i,j]:.1f}', ha='center', va='center',
                     fontsize=9, color='black')
    plt.colorbar(im2, ax=ax2, shrink=0.85, label='Laufzeit (ms)')
    ax2.grid(False)

    fig.suptitle('Abbildung 5: Subgraph-Kompatibilitätsprüfung (Epp 2026) – Schnittstellen × Dienstanbieter',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{outdir}/plot5_subgraph_kompatibilitaet.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('Plot 5 gespeichert: plot5_subgraph_kompatibilitaet.pdf')


if __name__ == '__main__':
    import os
    outdir = '/home/claude/flex_paper'
    os.makedirs(outdir, exist_ok=True)
    plot1_qos_zeitverlauf(outdir)
    plot2_routing_regionen(outdir)
    plot3_latenz_ketten(outdir)
    plot4_wettbewerb_ranking(outdir)
    plot5_subgraph_kompatibilitaet(outdir)
    print('Alle 5 Plots erfolgreich generiert.')

