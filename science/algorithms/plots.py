#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugung aller Abbildungen fuer die wissenschaftliche Arbeit:
"Dynamische Programmierung und Teile-und-Herrsche"
Autor: Stephan Epp, Universitaet Bielefeld, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Einheitliches Farbschema (analog zur LaTeX-Vorlage)
MAINBLUE   = '#19468C'
ACCENTRED  = '#B4321E'
DARKGREEN  = '#1E6432'
DARKGRAY   = '#3C3C46'
LIGHTBLUE  = '#D0E4F7'
LIGHTYELLOW= '#FFF8DC'
LIGHTGREEN = '#D6ECD6'

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Levenshtein-DP-Tabelle (Beispiel: "KITTEN" vs "SITTING")
# ─────────────────────────────────────────────────────────────────────────────
def plot1_levenshtein_table():
    s1 = "KITTEN"
    s2 = "SITTING"
    m, n = len(s1), len(s2)

    # DP-Tabelle berechnen
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)

    # Optimalen Pfad traceback ermitteln
    path = []
    i, j = m, n
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if best == dp[i-1][j-1]:
                i -= 1; j -= 1
            elif best == dp[i-1][j]:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    path = set(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={'width_ratios': [1.6, 1]})

    # --- Linkes Panel: DP-Tabelle ---
    ax = axes[0]
    ax.set_xlim(-0.5, n+0.5)
    ax.set_ylim(-0.5, m+0.5)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f'Levenshtein-DP-Tabelle: \"{s1}\" → \"{s2}\"',
                 fontsize=13, fontweight='bold', color=MAINBLUE, pad=12)

    # Zellen zeichnen
    for i in range(m+1):
        for j in range(n+1):
            val = dp[i][j]
            in_path = (i, j) in path
            match = (i > 0 and j > 0 and s1[i-1] == s2[j-1])

            if in_path:
                color = MAINBLUE
                txt_color = 'white'
            elif match:
                color = LIGHTGREEN
                txt_color = DARKGRAY
            else:
                color = '#F5F7FA'
                txt_color = DARKGRAY

            rect = Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                              linewidth=0.8, edgecolor='#BBBBBB',
                              facecolor=color)
            ax.add_patch(rect)
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, color=txt_color, fontweight='bold' if in_path else 'normal')

    # Spaltenköpfe
    ax.text(-0.5, -0.5, 'ε', ha='center', va='center',
            fontsize=10, color=DARKGRAY, fontweight='bold')
    for j, ch in enumerate(s2):
        ax.text(j+1, -0.5, ch, ha='center', va='center',
                fontsize=11, color=ACCENTRED, fontweight='bold')

    # Zeilenköpfe
    ax.text(-0.5, 0, 'ε', ha='center', va='center',
            fontsize=10, color=DARKGRAY, fontweight='bold')
    for i, ch in enumerate(s1):
        ax.text(-0.5, i+1, ch, ha='center', va='center',
                fontsize=11, color=MAINBLUE, fontweight='bold')

    # Legende
    handles = [
        mpatches.Patch(color=MAINBLUE, label='Optimaler Pfad'),
        mpatches.Patch(color=LIGHTGREEN, label='Zeichenübereinstimmung'),
        mpatches.Patch(color='#F5F7FA', label='Sonstige Zellen'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.9, bbox_to_anchor=(1.0, 0.0))

    # --- Rechtes Panel: Rekurrenzformel ---
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Rekurrenzformel', fontsize=13,
                  fontweight='bold', color=MAINBLUE, pad=12)

    formula_text = (
        r"$d[i][j] = \begin{cases}"
        r"i & \text{falls } j=0 \\"
        r"j & \text{falls } i=0 \\"
        r"d[i{-}1][j{-}1] & \text{falls } s_1[i]=s_2[j] \\"
        r"\min\begin{pmatrix}"
        r"d[i{-}1][j]+1 \\"
        r"d[i][j{-}1]+1 \\"
        r"d[i{-}1][j{-}1]+1"
        r"\end{pmatrix} & \text{sonst}"
        r"\end{cases}$"
    )

    ax2.text(0.05, 0.85, 'Basisfall:', fontsize=10,
             color=DARKGRAY, transform=ax2.transAxes, fontweight='bold')
    ax2.text(0.05, 0.77, r'$d[i][0] = i,\quad d[0][j] = j$',
             fontsize=10, transform=ax2.transAxes)

    ax2.text(0.05, 0.65, 'Rekurrenz:', fontsize=10,
             color=DARKGRAY, transform=ax2.transAxes, fontweight='bold')

    ops = ['Löschung', 'Einfügung', 'Substitution']
    formulas = [
        r'$d[i{-}1][j] + 1$',
        r'$d[i][j{-}1] + 1$',
        r'$d[i{-}1][j{-}1] + 1$',
    ]
    colors = [ACCENTRED, DARKGREEN, MAINBLUE]
    y_pos = [0.54, 0.44, 0.34]

    ax2.text(0.05, 0.60, r'$d[i][j] = \min($', fontsize=10,
             transform=ax2.transAxes, color=DARKGRAY)
    for op, fo, col, yp in zip(ops, formulas, colors, y_pos):
        ax2.text(0.12, yp, f'{op}: {fo}', fontsize=9,
                 transform=ax2.transAxes, color=col)
    ax2.text(0.05, 0.26, r'$)$', fontsize=10,
             transform=ax2.transAxes, color=DARKGRAY)

    ax2.text(0.05, 0.16, 'Laufzeit:', fontsize=10,
             color=DARKGRAY, transform=ax2.transAxes, fontweight='bold')
    ax2.text(0.05, 0.08, r'$\mathcal{O}(m \cdot n)$ mit $m=|s_1|, n=|s_2|$',
             fontsize=11, transform=ax2.transAxes, color=MAINBLUE,
             fontweight='bold')

    ax2.text(0.05, 0.00,
             f'Ergebnis: d("{s1}", "{s2}") = {dp[m][n]}',
             fontsize=10, transform=ax2.transAxes, color=ACCENTRED,
             style='italic')

    fig.suptitle('Abbildung 1: Dynamische Programmierung – Levenshtein-Distanz',
                 fontsize=14, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig('plots/plot1_levenshtein_table.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot1_levenshtein_table.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 1 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Laufzeitkomplexität O(N²) der Dynamischen Programmierung
# ─────────────────────────────────────────────────────────────────────────────
def plot2_dp_complexity():
    N = np.arange(1, 201)
    ops_dp = N**2
    ops_linear = N
    ops_nlogn = N * np.log2(np.maximum(N, 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Links: Absolute Laufzeiten ---
    ax = axes[0]
    ax.plot(N, ops_linear, color=DARKGREEN, lw=2, label=r'$\mathcal{O}(n)$', ls='--')
    ax.plot(N, ops_nlogn, color=MAINBLUE, lw=2, label=r'$\mathcal{O}(n \log n)$', ls='-.')
    ax.plot(N, ops_dp, color=ACCENTRED, lw=2.5, label=r'$\mathcal{O}(n^2)$ – Levenshtein')
    ax.fill_between(N, ops_dp, alpha=0.08, color=ACCENTRED)
    ax.set_xlabel('Stringlänge $N$ (gleich für beide Strings)', fontsize=11)
    ax.set_ylabel('Anzahl elementarer Operationen', fontsize=11)
    ax.set_title(r'Laufzeitwachstum: $\mathcal{O}(N^2)$ im Vergleich', fontsize=12,
                 color=DARKGRAY, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 40500)

    # Annotation
    ax.annotate(r'$N^2 = 40\,000$ bei $N=200$',
                xy=(200, 40000), xytext=(130, 32000),
                arrowprops=dict(arrowstyle='->', color=ACCENTRED, lw=1.5),
                fontsize=9, color=ACCENTRED,
                bbox=dict(boxstyle='round,pad=0.3', fc=LIGHTYELLOW, alpha=0.9))

    # --- Rechts: Wachstumsverhältnis und konkrete Werte ---
    ax2 = axes[1]
    Ns = [10, 20, 50, 100, 200, 500, 1000]
    ops_vals = [n**2 for n in Ns]
    cells = [n**2 for n in Ns]

    bars = ax2.bar(range(len(Ns)), ops_vals, color=MAINBLUE, alpha=0.75,
                   edgecolor=MAINBLUE)
    ax2.set_xticks(range(len(Ns)))
    ax2.set_xticklabels([str(n) for n in Ns])
    ax2.set_xlabel('Stringlänge $N$', fontsize=11)
    ax2.set_ylabel('Anzahl DP-Zellen $= N^2$', fontsize=11)
    ax2.set_title(r'Konkrete Zellanzahl der DP-Tabelle ($N \times N$)',
                  fontsize=12, color=DARKGRAY, fontweight='bold')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    for bar, val in zip(bars, ops_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                 f'{val:,}', ha='center', va='bottom', fontsize=7.5,
                 color=DARKGRAY, rotation=45)

    fig.suptitle(
        r'Abbildung 2: Quadratische Laufzeitkomplexität der Dynamischen Programmierung – $\mathcal{O}(N^2)$',
        fontsize=13, fontweight='bold', color=DARKGRAY, y=1.01
    )
    plt.tight_layout()
    plt.savefig('plots/plot2_dp_complexity.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot2_dp_complexity.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 2 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Merge-Sort-Rekursionsbaum (Visualisierung des Divide-and-Conquer)
# ─────────────────────────────────────────────────────────────────────────────
def plot3_mergesort_tree():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Links: Rekursionsbaum für n=8 ---
    ax = axes[0]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.10)
    ax.axis('off')
    ax.set_title('Rekursionsbaum: Merge Sort mit $n = 8$',
                 fontsize=13, fontweight='bold', color=MAINBLUE, pad=8)

    # Ebenen und Arrays
    levels = [
        [(0.5, 1.00, '[5,2,4,6,1,3,2,6]')],
        [(0.25, 0.78, '[5,2,4,6]'), (0.75, 0.78, '[1,3,2,6]')],
        [(0.125, 0.56, '[5,2]'), (0.375, 0.56, '[4,6]'),
         (0.625, 0.56, '[1,3]'), (0.875, 0.56, '[2,6]')],
        [(0.0625, 0.34, '[5]'), (0.1875, 0.34, '[2]'),
         (0.3125, 0.34, '[4]'), (0.4375, 0.34, '[6]'),
         (0.5625, 0.34, '[1]'), (0.6875, 0.34, '[3]'),
         (0.8125, 0.34, '[2]'), (0.9375, 0.34, '[6]')],
    ]
    merge_levels = [
        [(0.25, 0.14, '[2,4,5,6]'), (0.75, 0.14, '[1,2,3,6]')],
        [(0.5, -0.02, '[1,2,2,3,4,5,6,6]')],
    ]

    colors_by_level = [MAINBLUE, '#2E6DB4', '#4A90E2', LIGHTBLUE]
    node_colors = [MAINBLUE, '#1E6432', ACCENTRED, '#8B4513']

    def draw_node(ax_, x, y, text, level, small=False):
        fs = max(6.5, 9 - level)
        w = min(0.18, max(0.08, len(text) * 0.013))
        h = 0.06
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.01",
                               facecolor=colors_by_level[min(level, 3)],
                               edgecolor=MAINBLUE, linewidth=1.0,
                               alpha=0.85)
        ax_.add_patch(rect)
        ax_.text(x, y, text, ha='center', va='center',
                 fontsize=fs, color='white' if level < 2 else DARKGRAY,
                 fontweight='bold' if level == 0 else 'normal')

    # Divide-Ebenen zeichnen
    node_positions = {}
    for lv, nodes in enumerate(levels):
        for x, y, text in nodes:
            draw_node(ax, x, y, text, lv)
            node_positions[(lv, x)] = (x, y)

    # Kanten von Divide
    for lv in range(len(levels)-1):
        for (xi, yi, _) in levels[lv]:
            for (xj, yj, _) in levels[lv+1]:
                if abs(xj - xi) < 0.30:
                    ax.plot([xi, xj], [yi - 0.03, yj + 0.03],
                            color='#888888', lw=1.0, alpha=0.7)

    # Merge-Ebenen
    merge_colors = ['#1E6432', DARKGREEN]
    for mi, mnodes in enumerate(merge_levels):
        for x, y, text in mnodes:
            w = min(0.22, max(0.10, len(text) * 0.013))
            h = 0.06
            rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.01",
                                   facecolor=merge_colors[mi],
                                   edgecolor=DARKGREEN, linewidth=1.2,
                                   alpha=0.9)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold')

    # Kanten von letzter Divide zu ersten Merge
    for (xi, yi, _) in levels[-1]:
        for (xj, yj, _) in merge_levels[0]:
            if abs(xj - xi) < 0.30:
                ax.plot([xi, xj], [yi - 0.03, yj + 0.03],
                        color=DARKGREEN, lw=1.0, alpha=0.7, ls='--')

    # Kante zu finalem Merge
    for (xi, yi, _) in merge_levels[0]:
        ax.plot([xi, 0.5], [yi - 0.03, -0.02 + 0.03],
                color=DARKGREEN, lw=1.2, alpha=0.8, ls='--')

    # Ebenenbezeichnungen
    level_labels = ['Ebene 0\n(Teilen)', 'Ebene 1', 'Ebene 2', 'Ebene 3\n(Basis)']
    label_y = [1.00, 0.78, 0.56, 0.34]
    for label, ly in zip(level_labels, label_y):
        ax.text(1.03, ly, label, ha='left', va='center',
                fontsize=7.5, color=DARKGRAY, style='italic')

    ax.text(1.03, 0.14, 'Merge 1', ha='left', va='center',
            fontsize=7.5, color=DARKGREEN, style='italic')
    ax.text(1.03, -0.02, 'Merge 2\n(Fertig)', ha='left', va='center',
            fontsize=7.5, color=DARKGREEN, style='italic')

    # Kostenangaben pro Ebene
    costs = ['$cn$', '$cn/2 + cn/2 = cn$', '$4 \\cdot cn/4 = cn$', '$8 \\cdot cn/8 = cn$']
    for cost, ly in zip(costs, label_y):
        ax.text(-0.06, ly, cost, ha='right', va='center',
                fontsize=8, color=ACCENTRED)

    ax.text(-0.06, 1.08, 'Kosten:', ha='right', fontsize=8,
            color=ACCENTRED, fontweight='bold')
    ax.text(1.03, 1.08, 'Tiefe:', ha='left', fontsize=8,
            color=DARKGRAY, fontweight='bold')

    # --- Rechts: Rekurrenzgleichung und Mastertheorem ---
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Mastertheorem-Analyse für Merge Sort',
                  fontsize=13, fontweight='bold', color=MAINBLUE, pad=8)

    content = [
        (0.05, 0.93, 12, DARKGRAY, 'bold', 'Rekurrenzgleichung:'),
        (0.05, 0.84, 13, MAINBLUE, 'bold', r'$T(n) = 2\,T\!\left(\frac{n}{2}\right) + \mathcal{O}(n)$'),
        (0.05, 0.73, 10, DARKGRAY, 'bold', 'Mastertheorem (Fall 2):'),
        (0.05, 0.65, 9, DARKGRAY, 'normal',
         r'$a = 2,\quad b = 2,\quad f(n) = cn$'),
        (0.05, 0.57, 9, DARKGRAY, 'normal',
         r'$n^{\log_b a} = n^{\log_2 2} = n^1 = n$'),
        (0.05, 0.49, 9, DARKGRAY, 'normal',
         r'$f(n) = \Theta(n^{\log_b a}) \Rightarrow$ Fall 2'),
        (0.05, 0.40, 12, MAINBLUE, 'bold',
         r'$\Rightarrow T(n) = \Theta(n \log n)$'),
        (0.05, 0.30, 10, DARKGRAY, 'bold', 'Summenentfaltung:'),
        (0.05, 0.22, 9, DARKGRAY, 'normal',
         r'$T(n) = \sum_{k=0}^{\log_2 n} 2^k \cdot c\,\frac{n}{2^k}$'),
        (0.05, 0.14, 9, DARKGRAY, 'normal',
         r'$\quad = cn \cdot \sum_{k=0}^{\log_2 n} 1 = cn\,(\log_2 n + 1)$'),
        (0.05, 0.05, 12, ACCENTRED, 'bold',
         r'$= \mathcal{O}(n \log n)$'),
    ]

    for (x, y, fs, col, fw, txt) in content:
        ax2.text(x, y, txt, transform=ax2.transAxes,
                 fontsize=fs, color=col, fontweight=fw, va='center')

    # Horizontale Trennlinie
    ax2.plot([0.0, 0.95], [0.70, 0.70], color='#BBBBBB', lw=1.0,
             transform=ax2.transAxes)
    ax2.plot([0.0, 0.95], [0.38, 0.38], color='#BBBBBB', lw=1.0,
             transform=ax2.transAxes)

    fig.suptitle(
        'Abbildung 3: Teile-und-Herrsche – Merge Sort Rekursionsbaum und Mastertheorem-Analyse',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01
    )
    plt.tight_layout()
    plt.savefig('plots/plot3_mergesort_tree.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot3_mergesort_tree.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 3 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Laufzeitkomplexität O(n log n) – Merge Sort
# ─────────────────────────────────────────────────────────────────────────────
def plot4_mergesort_complexity():
    N = np.arange(2, 1001)
    T_nlogn = N * np.log2(N)
    T_n2 = N**2
    T_n3 = N**3
    T_nlogn_norm = T_nlogn / T_nlogn[-1]
    T_n2_norm = T_n2 / T_n2[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Links: Vergleich verschiedener Klassen ---
    ax = axes[0]
    ax.semilogy(N, N, color=DARKGREEN, lw=2, label=r'$\mathcal{O}(n)$', ls='--', alpha=0.8)
    ax.semilogy(N, T_nlogn, color=MAINBLUE, lw=2.5,
                label=r'$\mathcal{O}(n \log n)$ – Merge Sort')
    ax.semilogy(N, T_n2, color=ACCENTRED, lw=2.5,
                label=r'$\mathcal{O}(n^2)$ – Levenshtein', ls='-.')
    ax.semilogy(N[N<=100], T_n3[N<=100], color='#8B4513', lw=2,
                label=r'$\mathcal{O}(n^3)$ (zum Vergleich)', ls=':', alpha=0.8)

    ax.fill_between(N, T_nlogn, N, alpha=0.07, color=MAINBLUE)
    ax.set_xlabel('Eingabegröße $n$', fontsize=11)
    ax.set_ylabel('Operationsanzahl (logarithmische Skala)', fontsize=11)
    ax.set_title(r'Laufzeitklassen im Vergleich (log-Skala)', fontsize=12,
                 color=DARKGRAY, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(2, 1000)

    # Annotation Vorteil n log n gegenüber n²
    n_star = 100
    ax.annotate('',
                xy=(n_star, n_star**2),
                xytext=(n_star, n_star * np.log2(n_star)),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.8))
    ax.text(n_star + 30, np.sqrt(n_star**2 * n_star*np.log2(n_star)),
            f'Faktor ≈ {n_star/np.log2(n_star):.0f}×\nbei $n=100$',
            fontsize=8.5, color='purple',
            bbox=dict(boxstyle='round,pad=0.3', fc=LIGHTYELLOW, alpha=0.9))

    # --- Rechts: Merge Sort Laufzeitmessung (simuliert) ---
    ax2 = axes[1]
    import time

    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        return result + left[i:] + right[j:]

    sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    times_measured = []
    rng = np.random.default_rng(42)
    for sz in sizes:
        arr = rng.integers(0, sz*10, sz).tolist()
        t0 = time.perf_counter()
        for _ in range(max(1, 1000//sz)):
            merge_sort(arr[:])
        elapsed = (time.perf_counter() - t0) / max(1, 1000//sz)
        times_measured.append(elapsed * 1000)  # ms

    times_arr = np.array(times_measured)
    sizes_arr = np.array(sizes, dtype=float)
    theoretical = sizes_arr * np.log2(sizes_arr)
    scale = times_arr[-1] / theoretical[-1]
    times_theory = theoretical * scale

    ax2.plot(sizes_arr, times_arr, 'o-', color=MAINBLUE, lw=2.5,
             markersize=7, label='Gemessene Laufzeit (Python)')
    ax2.plot(sizes_arr, times_theory, '--', color=ACCENTRED, lw=2,
             label=r'Theoretisch $c \cdot n \log_2 n$')
    ax2.fill_between(sizes_arr, times_arr, times_theory,
                     alpha=0.1, color=MAINBLUE)
    ax2.set_xlabel('Arraygröße $n$', fontsize=11)
    ax2.set_ylabel('Laufzeit (ms)', fontsize=11)
    ax2.set_title('Empirische vs.\ theoretische Laufzeit\nMerge Sort', fontsize=12,
                  color=DARKGRAY, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig.suptitle(
        r'Abbildung 4: Laufzeitanalyse Merge Sort – $\mathcal{O}(n \log n)$',
        fontsize=13, fontweight='bold', color=DARKGRAY, y=1.01
    )
    plt.tight_layout()
    plt.savefig('plots/plot4_mergesort_complexity.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot4_mergesort_complexity.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 4 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Vergleich DP (O(N²)) vs. Merge Sort (O(n log n)) – Vollständige Analyse
# ─────────────────────────────────────────────────────────────────────────────
def plot5_comparison():
    N = np.arange(1, 501)
    T_nlogn = np.where(N > 1, N * np.log2(N.astype(float)), N.astype(float))
    T_n2 = N**2.0

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)

    # ── Panel A: Laufzeitkurven ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(N, T_nlogn, color=MAINBLUE, lw=2.5,
              label=r'Merge Sort: $\mathcal{O}(n \log n)$')
    ax_a.plot(N, T_n2, color=ACCENTRED, lw=2.5,
              label=r'Levenshtein: $\mathcal{O}(n^2)$')
    ax_a.fill_between(N, T_nlogn, T_n2, alpha=0.1, color=ACCENTRED,
                      label='Laufzeitvorteil $n \log n$')
    ax_a.set_xlabel('Problemgröße $n$')
    ax_a.set_ylabel('Operationen')
    ax_a.set_title('(A) Absoluter Vergleich', fontweight='bold', color=DARKGRAY)
    ax_a.legend(fontsize=8.5)
    ax_a.set_xlim(1, 500)
    ax_a.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

    # ── Panel B: Verhältnis n²/(n log n) ──
    ax_b = fig.add_subplot(gs[0, 1])
    ratio = T_n2 / np.where(T_nlogn > 0, T_nlogn, 1)
    ax_b.plot(N[1:], ratio[1:], color=DARKGREEN, lw=2.5)
    ax_b.fill_between(N[1:], ratio[1:], alpha=0.12, color=DARKGREEN)
    ax_b.set_xlabel('Problemgröße $n$')
    ax_b.set_ylabel(r'Verhältnis $n^2 \,/\, (n \log_2 n) = n / \log_2 n$')
    ax_b.set_title(r'(B) Effizienzgewinn: $n^2$ vs.\ $n \log n$',
                   fontweight='bold', color=DARKGRAY)
    ax_b.axhline(y=ratio[99], color=ACCENTRED, ls='--', lw=1.5,
                 label=f'Faktor bei $n=100$: {ratio[99]:.1f}')
    ax_b.axhline(y=ratio[499], color=MAINBLUE, ls=':', lw=1.5,
                 label=f'Faktor bei $n=500$: {ratio[499]:.1f}')
    ax_b.legend(fontsize=8.5)
    ax_b.set_xlim(2, 500)

    # ── Panel C: Speicherkomplexität ──
    ax_c = fig.add_subplot(gs[1, 0])
    space_dp = N**2   # O(N²) DP-Tabelle
    space_ms = N      # O(n) für Merge Sort (Hilfsspeicher)
    ax_c.fill_between(N, space_dp, alpha=0.25, color=ACCENTRED,
                      label=r'DP Speicher: $\mathcal{O}(N^2)$')
    ax_c.fill_between(N, space_ms, alpha=0.35, color=MAINBLUE,
                      label=r'Merge Sort Speicher: $\mathcal{O}(n)$')
    ax_c.plot(N, space_dp, color=ACCENTRED, lw=2.5)
    ax_c.plot(N, space_ms, color=MAINBLUE, lw=2.5)
    ax_c.set_xlabel('Problemgröße $n$')
    ax_c.set_ylabel('Speichereinheiten')
    ax_c.set_title('(C) Speicherkomplexität', fontweight='bold', color=DARKGRAY)
    ax_c.legend(fontsize=8.5)
    ax_c.set_xlim(1, 500)
    ax_c.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

    # ── Panel D: Zusammenfassungstabelle ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    table_data = [
        ['Eigenschaft', 'Levenshtein (DP)', 'Merge Sort (T&H)'],
        ['Entwurfsprinzip', 'Dyn. Programmierung', 'Teile-und-Herrsche'],
        ['Zeitkomplexität', r'$\mathcal{O}(N^2)$', r'$\mathcal{O}(n \log n)$'],
        ['Raumkomplexität', r'$\mathcal{O}(N^2)$', r'$\mathcal{O}(n)$'],
        ['Rekurrenz', r'$d[i][j]=\min(\ldots)$', r'$T(n)=2T(n/2)+cn$'],
        ['Optimalitätsprinzip', 'Bellman (OB)', 'Mastertheorem'],
        ['Teilprobleme', 'Überlappend', 'Disjunkt'],
        ['Stabilität', '–', 'Stabil ($\leq$)'],
        ['Vergleich $n=1000$', '$10^6$ Ops.', '$\sim 10\,000$ Ops.'],
    ]

    col_colors = [['#E8E8E8']*3,
                  [LIGHTBLUE, LIGHTBLUE, LIGHTGREEN],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                  ['#F5F7FA', '#FFE8E8', '#E8F8E8'],
                 ]

    t = ax_d.table(cellText=table_data[1:],
                   colLabels=table_data[0],
                   loc='center',
                   cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1.1, 1.55)

    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor(MAINBLUE)
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 1:
            cell.set_facecolor('#FFF0EE')
        elif col == 2:
            cell.set_facecolor('#EEF8EE')
        cell.set_edgecolor('#CCCCCC')

    ax_d.set_title('(D) Vergleichsübersicht beider Algorithmen',
                   fontweight='bold', color=DARKGRAY, pad=12)

    fig.suptitle(
        'Abbildung 5: Vollständiger Vergleich – Dynamische Programmierung vs. Teile-und-Herrsche',
        fontsize=13, fontweight='bold', color=DARKGRAY, y=1.01
    )
    plt.savefig('plots/plot5_comparison.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot5_comparison.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 5 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Merge-Sort Sortierschritte (Visualisierung des Sortierens)
# ─────────────────────────────────────────────────────────────────────────────
def plot6_mergesort_steps():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Links: Merge-Schritt im Detail ---
    ax = axes[0]
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.5, 8.5)
    ax.axis('off')
    ax.set_title('Merge-Schritt im Detail', fontsize=13,
                 fontweight='bold', color=MAINBLUE, pad=8)

    left_arr  = [1, 3, 5, 7]
    right_arr = [2, 4, 6, 8]
    merged    = [1, 2, 3, 4, 5, 6, 7, 8]

    def draw_array(ax_, arr, y, x_start, label, color, small=False):
        ax_.text(x_start - 0.8, y, label, ha='right', va='center',
                 fontsize=9.5, color=DARKGRAY, fontweight='bold')
        for k, val in enumerate(arr):
            rect = FancyBboxPatch((x_start + k*1.5 - 0.42, y - 0.42), 0.84, 0.84,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='#888888', lw=1.0)
            ax_.add_patch(rect)
            ax_.text(x_start + k*1.5, y, str(val), ha='center', va='center',
                     fontsize=11, color='white', fontweight='bold')

    draw_array(ax, left_arr,  6.5, 1.0, 'Linke\nHälfte',  MAINBLUE)
    draw_array(ax, right_arr, 4.5, 1.0, 'Rechte\nHälfte', ACCENTRED)
    draw_array(ax, merged,    2.0, 0.25,'Merged\nOutput',  DARKGREEN)

    # Pointer-Pfeile
    for step, (li, ri, mi) in enumerate([(0,0,0),(0,1,1),(1,1,2),(1,2,3),(2,2,4),(2,3,5),(3,3,6),(3,4,7)]):
        if step < 4:
            if li < len(left_arr):
                ax.annotate('', xy=(0.25 + mi*1.5, 2.42+0.02),
                            xytext=(1.0 + (li if merged[mi] in left_arr[:li+1] and li<len(left_arr) else min(li, len(left_arr)-1)) * 1.5, 6.08),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.3))

    ax.text(7.5, 8.2, 'Zeigervergleich: min(L[i], R[j]) → Output',
            ha='center', fontsize=9, color=DARKGRAY, style='italic')

    # Vergleichstabelle
    for step, (li, ri, out) in enumerate([(0,0,1),(0,1,2),(1,1,3),(1,2,4),(2,2,5),(2,3,6),(3,3,7),(3,4,8)]):
        x = 0.5 + step * 1.85
        y_row = 0.0
        lv = left_arr[li] if li < len(left_arr) else '–'
        rv = right_arr[ri] if ri < len(right_arr) else '–'
        col = MAINBLUE if isinstance(lv, int) and (not isinstance(rv, int) or lv <= rv) else ACCENTRED
        ax.text(x, y_row + 0.6, f'L={lv}', ha='center', fontsize=7.5, color=MAINBLUE)
        ax.text(x, y_row + 0.1, f'R={rv}', ha='center', fontsize=7.5, color=ACCENTRED)
        ax.text(x, y_row - 0.4, f'→{out}', ha='center', fontsize=7.5, color=DARKGREEN, fontweight='bold')

    ax.text(7.5, -1.0, f'Schritt: 1   2   3   4   5   6   7   8',
            ha='center', fontsize=7.5, color='#888888')

    # --- Rechts: Histogramm vor/nach Sortierung ---
    ax2 = axes[1]
    rng = np.random.default_rng(7)
    data_unsorted = rng.integers(1, 50, 16).tolist()
    data_sorted = sorted(data_unsorted)

    x = np.arange(len(data_unsorted))
    w = 0.35
    ax2.bar(x - w/2, data_unsorted, w, color=ACCENTRED, alpha=0.8,
            label='Unsortiert', edgecolor='white', lw=0.8)
    ax2.bar(x + w/2, data_sorted, w, color=MAINBLUE, alpha=0.8,
            label='Sortiert (Merge Sort)', edgecolor='white', lw=0.8)

    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Wert', fontsize=11)
    ax2.set_title('Vor und nach Merge Sort\n(16 zufällige Elemente)',
                  fontsize=12, color=DARKGRAY, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in x], fontsize=8)

    fig.suptitle(
        'Abbildung 6: Merge Sort – Merge-Schritt und Sortiervisualisierung',
        fontsize=13, fontweight='bold', color=DARKGRAY, y=1.01
    )
    plt.tight_layout()
    plt.savefig('plots/plot6_mergesort_steps.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig('plots/plot6_mergesort_steps.png',
                format='png', bbox_inches='tight')
    plt.close()
    print("Plot 6 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Erzeuge alle Abbildungen ...")
    plot1_levenshtein_table()
    plot2_dp_complexity()
    plot3_mergesort_tree()
    plot4_mergesort_complexity()
    plot5_comparison()
    plot6_mergesort_steps()
    print("\nAlle 6 Plots erfolgreich gespeichert.")

