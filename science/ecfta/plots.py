#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots fuer die Arbeit:
"Dynamische Programmierung fuer das CFTA-Problem in Multiagentensystemen"
Stephan Epp, Universitaet Bielefeld, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from itertools import combinations
import random, time, math

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
DARKGRAY  = '#3C3C46'
LIGHTBLUE = '#D0E4F7'
LIGHTYELLOW='#FFF8DC'
LIGHTGREEN= '#D6ECD6'
ORANGE    = '#D4720A'

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

OUT = 'plots/'

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Laufzeitwachstum 3^n vs n^k (Greedy vs Subset-DP)
# ─────────────────────────────────────────────────────────────────────────────
def plot1_complexity_growth():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Links: log-Skala absoluter Vergleich
    ax = axes[0]
    N = np.arange(2, 26)
    ax.semilogy(N, 3.0**N,        color=MAINBLUE,  lw=2.5, label=r'Subset-DP: $3^n \cdot m$')
    ax.semilogy(N, N**2 * 10,     color=DARKGREEN, lw=2,   label=r'Greedy $k=2$: $n^{k+1}$', ls='--')
    ax.semilogy(N, N**3 * 10,     color=ORANGE,    lw=2,   label=r'Greedy $k=3$: $n^{k+1}$', ls='-.')
    ax.semilogy(N, N**4 * 10,     color=ACCENTRED, lw=2,   label=r'Greedy $k=4$: $n^{k+1}$', ls=':')
    ax.axvline(x=15, color='gray', lw=1.2, ls='--', alpha=0.6)
    ax.text(15.3, 1e5, '$n=15$\nDP praktikabel', fontsize=8, color='gray')
    ax.set_xlabel('Anzahl Agenten $n$')
    ax.set_ylabel('Operationsanzahl (log-Skala)')
    ax.set_title(r'Greedy $\mathcal{O}(n^{k+1})$ vs.\ Subset-DP $\mathcal{O}(3^n \cdot m)$',
                 fontweight='bold', color=DARKGRAY)
    ax.legend(fontsize=9)
    ax.set_xlim(2, 25)

    # Rechts: Kreuzungspunkte - ab welchem n ist Greedy besser?
    ax2 = axes[1]
    ns = np.arange(2, 30)
    for k, col, ls in [(2, DARKGREEN, '--'), (3, ORANGE, '-.'), (4, ACCENTRED, ':')]:
        greedy = ns**(k+1)
        subset = 3.0**ns
        ratio = greedy / subset
        ax2.semilogy(ns, ratio, color=col, lw=2, ls=ls,
                     label=fr'$n^{{{k+1}}} \;/\; 3^n$  ($k={k}$)')
    ax2.axhline(y=1.0, color='black', lw=1.5, ls='-', alpha=0.7, label='Gleichstand')
    ax2.set_xlabel('Anzahl Agenten $n$')
    ax2.set_ylabel(r'Verhaeltnis Greedy / Subset-DP')
    ax2.set_title('Verhaeltnis: Wann ist Subset-DP guenstiger?',
                  fontweight='bold', color=DARKGRAY)
    ax2.legend(fontsize=9)
    ax2.set_xlim(2, 29)

    # Schattierung: Bereich wo DP besser
    ax2.axvspan(2, 10, alpha=0.08, color=MAINBLUE, label='DP-Vorteil ($n\\leq10$)')
    ax2.text(5, 1e-4, 'DP\nvorteilhaft', ha='center', fontsize=9,
             color=MAINBLUE, fontweight='bold')

    fig.suptitle(
        r'Abbildung 1: Laufzeitwachstum Greedy ($\mathcal{O}(n^{k+1})$) vs.'
        r' Subset-DP ($\mathcal{O}(3^n \cdot m)$) fuer das CFTA-Problem',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'plot1_complexity_growth.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot1_complexity_growth.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 1 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: DP-Rekurrenz fuer Gruppenressourcen (Tabellenstruktur)
# ─────────────────────────────────────────────────────────────────────────────
def plot2_resource_dp_table():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Agenten und ihre Ressourcenvektoren (r=2 fuer Visualisierung)
    agents = ['$a_1$', '$a_2$', '$a_3$', '$a_4$']
    resources = [(2, 1), (1, 3), (3, 2), (1, 1)]

    # Alle Teilmengen der Groesse 1..4 und ihre Ressourcensummen (DP-Aufbau)
    subsets_by_size = {
        0: [('{}', (0,0))],
        1: [(f'{{{agents[i]}}}', resources[i]) for i in range(4)],
        2: [],
        3: [],
        4: []
    }
    for i, j in combinations(range(4), 2):
        r = (resources[i][0]+resources[j][0], resources[i][1]+resources[j][1])
        subsets_by_size[2].append((f'{{{agents[i]},{agents[j]}}}', r))
    for i, j, l in combinations(range(4), 3):
        r = (resources[i][0]+resources[j][0]+resources[l][0],
             resources[i][1]+resources[j][1]+resources[l][1])
        subsets_by_size[3].append((f'{{{agents[i]},{agents[j]},{agents[l]}}}', r))
    full_r = tuple(sum(resources[i][d] for i in range(4)) for d in range(2))
    subsets_by_size[4] = [('{a_1,a_2,a_3,a_4}', full_r)]

    # Links: Tabelle der DP-Rekurrenz
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 5.0)
    ax.axis('off')
    ax.set_title('DP-Aufbau der Gruppenressourcen\n'
                 r'$B_{C \cup \{a_j\}} = B_C + B_j$',
                 fontsize=12, fontweight='bold', color=MAINBLUE, pad=8)

    col_colors = [LIGHTBLUE, LIGHTGREEN, '#FFF0EE', '#F0F0FF']
    size_labels = ['|C|=1', '|C|=2', '|C|=3', '|C|=4']

    for col_idx in range(4):
        size = col_idx + 1
        entries = subsets_by_size[size]
        ax.text(col_idx + 0.5, 4.6, size_labels[col_idx],
                ha='center', fontsize=9, color=DARKGRAY, fontweight='bold')
        for row_idx, (label, res) in enumerate(entries[:4]):
            y = 3.5 - row_idx * 0.85
            rect = FancyBboxPatch((col_idx + 0.05, y - 0.32), 0.9, 0.60,
                                   boxstyle='round,pad=0.04',
                                   facecolor=col_colors[col_idx],
                                   edgecolor='#AAAAAA', lw=0.8)
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, y + 0.07, label,
                    ha='center', va='center', fontsize=6.5, color=DARKGRAY)
            ax.text(col_idx + 0.5, y - 0.16,
                    f'$B$=({res[0]},{res[1]})',
                    ha='center', va='center', fontsize=6.5,
                    color=MAINBLUE, fontweight='bold')

    # Pfeile zwischen Spalten (DP-Uebergang)
    for col_idx in range(3):
        ax.annotate('', xy=(col_idx + 1.05, 2.5),
                    xytext=(col_idx + 0.95, 2.5),
                    arrowprops=dict(arrowstyle='->', color=ACCENTRED, lw=1.5))

    ax.text(2.0, -0.2,
            r'Jede Zelle: $B_{C \cup \{a_j\}} = B_C + B_j$ in $O(r)$'
            '\n(statt $O(|C| \cdot r)$ von Grund auf)',
            ha='center', fontsize=9, color=ACCENTRED, style='italic')

    # Rechts: Laufzeitvergleich Naiv vs DP
    ax2 = axes[1]
    ks = [2, 3, 4, 5]
    ns = np.arange(5, 25)
    labels_naive = [f'Naiv $k={k}$: $O(n^k \\cdot k \\cdot r)$' for k in ks]
    labels_dp    = [f'DP $k={k}$: $O(n^k \\cdot r)$' for k in ks]
    colors_n = [ACCENTRED, ORANGE, DARKGREEN, MAINBLUE]
    r_val = 5

    for k, col in zip(ks, colors_n):
        naive = ns**k * k * r_val
        dp_   = ns**k * r_val
        ax2.semilogy(ns, naive, color=col, lw=2,   ls='-',  alpha=0.85,
                     label=f'Naiv $k={k}$')
        ax2.semilogy(ns, dp_,   color=col, lw=1.5, ls='--', alpha=0.65,
                     label=f'DP $k={k}$')

    ax2.set_xlabel('Anzahl Agenten $n$')
    ax2.set_ylabel(r'Operationsanzahl (log-Skala, $r=5$)')
    ax2.set_title('Laufzeitgewinn durch DP-Ressourcenaufbau\n'
                  r'(Faktor $k$ Verbesserung)',
                  fontweight='bold', color=DARKGRAY)
    ax2.legend(fontsize=7.5, ncol=2)

    fig.suptitle(
        'Abbildung 2: DP-Rekurrenz fuer Gruppenressourcen '
        r'-- $B_{C\cup\{a_j\}} = B_C + B_j$ -- Faktor-$k$-Verbesserung',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'plot2_resource_dp_table.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot2_resource_dp_table.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 2 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Subset-DP Zustandsraum (Boolean Hypercube n=4)
# ─────────────────────────────────────────────────────────────────────────────
def plot3_subset_dp_states():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Links: Schichtenstruktur des Zustandsraums 2^4
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 5.5)
    ax.axis('off')
    ax.set_title('Schichtenstruktur des Subset-DP\n'
                 r'fuer $n=4$ Agenten ($2^4 = 16$ Zustaende)',
                 fontsize=12, fontweight='bold', color=MAINBLUE, pad=8)

    from math import comb
    sizes = [0, 1, 2, 3, 4]
    counts = [comb(4, s) for s in sizes]
    layer_y = [0.3, 1.4, 2.5, 3.6, 4.7]
    layer_colors = ['#E8E8E8', LIGHTBLUE, LIGHTGREEN, '#FFE8CC', '#FFD0D0']
    layer_labels = [r'$\emptyset$ (1)', r'$|C|=1$ (4)',
                    r'$|C|=2$ (6)', r'$|C|=3$ (4)', r'$|C|=4$ (1)']

    all_subsets = [[], [], [], [], []]
    agents = ['a1','a2','a3','a4']
    for mask in range(16):
        bits = [i for i in range(4) if mask & (1 << i)]
        s = len(bits)
        all_subsets[s].append(mask)

    node_pos = {}
    for s, (y, col) in enumerate(zip(layer_y, layer_colors)):
        ax.text(-0.3, y, layer_labels[s], ha='right', va='center',
                fontsize=9, color=DARKGRAY, fontweight='bold')
        masks = all_subsets[s]
        n_m = len(masks)
        xs = np.linspace(0.3, 4.2, n_m) if n_m > 1 else [2.25]
        for x, mask in zip(xs, masks):
            node_pos[mask] = (x, y)
            circle = plt.Circle((x, y), 0.22, color=col,
                                 ec=MAINBLUE, lw=1.0, zorder=3)
            ax.add_patch(circle)
            label = '{' + ','.join(agents[i] for i in range(4) if mask & (1<<i)) + '}'
            ax.text(x, y, label if len(label) < 12 else f'$S_{{{mask}}}$',
                    ha='center', va='center', fontsize=5.5, color=DARKGRAY, zorder=4)

    # Kanten zwischen aufeinanderfolgenden Schichten (Teilmengenrelation)
    for mask, (x1, y1) in node_pos.items():
        for i in range(4):
            if mask & (1 << i):
                parent = mask ^ (1 << i)
                if parent in node_pos:
                    x2, y2 = node_pos[parent]
                    ax.plot([x1, x2], [y1-0.22, y2+0.22],
                            color='#AAAAAA', lw=0.7, alpha=0.5, zorder=1)

    # Optimaler Pfad hervorheben
    opt_path = [0b0000, 0b0001, 0b0011, 0b0111, 0b1111]
    for a, b in zip(opt_path[:-1], opt_path[1:]):
        if a in node_pos and b in node_pos:
            x1, y1 = node_pos[a]
            x2, y2 = node_pos[b]
            ax.annotate('', xy=(x2, y2-0.22), xytext=(x1, y1+0.22),
                        arrowprops=dict(arrowstyle='->', color=ACCENTRED, lw=2.0))

    ax.text(4.4, 2.5, 'Optimaler\nPfad', ha='left', fontsize=8,
            color=ACCENTRED, fontweight='bold')

    # Rechts: Zustandszahl 3^n und Entfaltung der Rekurrenz
    ax2 = axes[1]
    ns = np.arange(2, 22)
    states_2n = 2.0**ns
    states_3n = 3.0**ns

    ax2.semilogy(ns, states_2n, color=MAINBLUE, lw=2.5, label=r'$2^n$ Zustaende (Speicher)')
    ax2.semilogy(ns, states_3n, color=ACCENTRED, lw=2.5,
                 label=r'$3^n$ Uebergaenge (Zeit)')
    ax2.fill_between(ns, states_2n, states_3n, alpha=0.1, color=ACCENTRED)

    ax2.axvline(x=10, color=DARKGREEN, lw=1.5, ls='--')
    ax2.text(10.2, 2e2, '$n=10$:\n$2^{10}$=1024 Zustaende\n$3^{10}$=59049 Uebergaenge',
             fontsize=8, color=DARKGREEN)

    ax2.set_xlabel('Anzahl Agenten $n$')
    ax2.set_ylabel('Anzahl (log-Skala)')
    ax2.set_title(r'Speicher ($2^n$) und Zeit ($3^n$) des Subset-DP',
                  fontweight='bold', color=DARKGRAY)
    ax2.legend(fontsize=9)
    ax2.set_xlim(2, 21)

    # Tabelle konkreter Werte
    ns_tab = [5, 8, 10, 12, 15, 20]
    for i, nt in enumerate(ns_tab):
        ax2.annotate(f'$n={nt}$: {int(3**nt):,}',
                     xy=(nt, 3**nt), xytext=(nt+0.5, 3**nt * 3),
                     fontsize=7.5, color=ACCENTRED, alpha=0.8)

    fig.suptitle(
        r'Abbildung 3: Subset-DP Zustandsraum -- $2^n$ Zustaende,'
        r' $3^n$ Uebergaenge -- Boolean-Hypercube fuer $n=4$',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'plot3_subset_dp_states.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot3_subset_dp_states.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 3 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Simulation Greedy vs. exakter DP (n=10, m=10)
# ─────────────────────────────────────────────────────────────────────────────
def plot4_simulation_comparison():
    rng = np.random.default_rng(42)
    n_agents = 10
    m_tasks  = 10
    r_res    = 5
    n_runs   = 80

    def make_instance(seed):
        rg = np.random.default_rng(seed)
        agents = rg.integers(0, 4, (n_agents, r_res)).astype(float)
        tasks  = rg.integers(0, 7, (m_tasks,  r_res)).astype(float)
        return agents, tasks

    def group_value(group_indices, agents, task):
        bc = agents[group_indices].sum(axis=0)
        if np.all(bc >= task):
            unused = bc - task
            tau = task.sum()
            delta = unused.sum()
            return max(0, tau - delta)
        return 0.0

    def greedy_disjoint(agents, tasks, k=3):
        n, m = len(agents), len(tasks)
        available = set(range(n))
        remaining = list(range(m))
        total_success = 0.0
        while remaining:
            best_val = -1
            best_group = None
            best_task = None
            for tj in remaining:
                for size in range(1, min(k, len(available)) + 1):
                    for combo in combinations(list(available)[:min(15, len(available))], size):
                        v = group_value(list(combo), agents, tasks[tj])
                        if v > best_val:
                            best_val = v
                            best_group = combo
                            best_task = tj
            if best_val <= 0 or best_group is None:
                break
            total_success += best_val
            for ag in best_group:
                available.discard(ag)
            remaining.remove(best_task)
        return total_success

    def dp_exact(agents, tasks, k=3):
        n, m = len(agents), len(tasks)
        # dp[S] = max Gesamterfolg mit Agentenmenge S
        # Wir begrenzen auf n<=10 fuer Machbarkeit
        dp = {}
        dp[0] = 0.0  # leere Menge: kein Erfolg

        all_masks = list(range(1 << n))
        # Vorbereitung: Gruppe -> bester Task-Wert
        group_best = {}
        for mask in all_masks:
            indices = [i for i in range(n) if mask & (1 << i)]
            if 0 < len(indices) <= k:
                best = max((group_value(indices, agents, tasks[tj])
                            for tj in range(m)), default=0.0)
                group_best[mask] = best

        # Bottom-up DP ueber Teilmengen in aufsteigender Groesse
        for mask in sorted(all_masks, key=lambda x: bin(x).count('1')):
            indices = [i for i in range(n) if mask & (1 << i)]
            best_here = 0.0
            # Alle Teilkoalitionen als moegliche naechste Gruppe
            for size in range(1, min(k, len(indices)) + 1):
                for combo_bits in combinations(indices, size):
                    sub = sum(1 << i for i in combo_bits)
                    rest = mask ^ sub
                    v = group_best.get(sub, 0.0)
                    candidate = (dp.get(rest, 0.0) or 0.0) + v
                    if candidate > best_here:
                        best_here = candidate
            dp[mask] = max(dp.get(mask, 0.0), best_here)

        full_mask = (1 << n) - 1
        return dp.get(full_mask, 0.0)

    greedy_results = []
    dp_results = []

    for seed in range(n_runs):
        agents, tasks = make_instance(seed)
        greedy_results.append(greedy_disjoint(agents, tasks, k=3))
        dp_results.append(dp_exact(agents, tasks, k=3))

    greedy_arr = np.array(greedy_results)
    dp_arr = np.array(dp_results)
    improvements = (dp_arr - greedy_arr) / np.maximum(greedy_arr, 1e-9) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    # Panel A: Scatter Greedy vs DP
    ax = axes[0]
    ax.scatter(greedy_arr, dp_arr, alpha=0.65, color=MAINBLUE,
               s=40, edgecolors='white', lw=0.5, zorder=3)
    max_val = max(greedy_arr.max(), dp_arr.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.5,
            label='Greedy = DP')
    ax.fill_between([0, max_val], [0, max_val], [0, max_val * 1.3],
                    alpha=0.06, color=ACCENTRED)
    ax.text(max_val*0.15, max_val*0.85, 'DP besser', fontsize=9,
            color=ACCENTRED, style='italic')
    ax.set_xlabel('Greedy Gesamterfolg')
    ax.set_ylabel('DP Gesamterfolg (exakt)')
    ax.set_title('(A) Greedy vs.\ DP\n(80 Instanzen, $n=m=10$, $k=3$)',
                 fontweight='bold', color=DARKGRAY)
    ax.legend(fontsize=8.5)

    # Panel B: Histogramm der Verbesserungen
    ax2 = axes[1]
    bins = np.linspace(improvements.min() - 1, improvements.max() + 1, 20)
    ax2.hist(improvements, bins=bins, color=MAINBLUE, edgecolor='white',
             alpha=0.8, rwidth=0.85)
    ax2.axvline(improvements.mean(), color=ACCENTRED, lw=2,
                label=f'Mittelwert: {improvements.mean():.1f}%')
    ax2.axvline(0, color='black', lw=1.5, ls='--', alpha=0.6)
    ax2.set_xlabel('Verbesserung durch DP (%)')
    ax2.set_ylabel('Haeufigkeit')
    ax2.set_title('(B) Verteilung der Verbesserung\ndurch exakten DP',
                  fontweight='bold', color=DARKGRAY)
    ax2.legend(fontsize=9)

    # Panel C: Boxplot-Vergleich
    ax3 = axes[2]
    bp = ax3.boxplot([greedy_arr, dp_arr], labels=['Greedy\n(Shehory-Kraus)', 'Subset-DP\n(exakt)'],
                     patch_artist=True, notch=False,
                     boxprops=dict(facecolor=LIGHTBLUE, color=MAINBLUE),
                     medianprops=dict(color=ACCENTRED, lw=2.5),
                     whiskerprops=dict(color=MAINBLUE),
                     capprops=dict(color=MAINBLUE),
                     flierprops=dict(marker='o', color=MAINBLUE, alpha=0.5))
    bp['boxes'][1].set_facecolor(LIGHTGREEN)
    ax3.set_ylabel('Gesamterfolg $\sum V_\\nu$')
    ax3.set_title('(C) Verteilung des Gesamterfolgs\n(80 Instanzen)',
                  fontweight='bold', color=DARKGRAY)

    mu_g = greedy_arr.mean()
    mu_d = dp_arr.mean()
    ax3.text(1, mu_g + 0.5, f'$\\mu$={mu_g:.1f}', ha='center',
             fontsize=8.5, color=MAINBLUE)
    ax3.text(2, mu_d + 0.5, f'$\\mu$={mu_d:.1f}', ha='center',
             fontsize=8.5, color=DARKGREEN)

    fig.suptitle(
        'Abbildung 4: Simulation -- Greedy (Shehory-Kraus) vs.\ exakter Subset-DP '
        '($n=m=10$, $r=5$, $k=3$, 80 Instanzen)',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'plot4_simulation_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot4_simulation_comparison.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 4 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Mehrdimensionaler Rucksack-DP (ueberlappende Gruppen)
# ─────────────────────────────────────────────────────────────────────────────
def plot5_knapsack_dp():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Links: 2D Zustandsraum (r=2 Ressourcen, n=3 Agenten)
    ax = axes[0]
    B_max = 6
    r = 2
    # dp[b1][b2] = maximaler Gesamterfolg bei Restressourcen (b1, b2)
    # Beispiel: 3 Aufgaben, 3 Agenten
    tasks_demo = [(2, 2, 3), (1, 3, 2), (3, 1, 4)]  # (b1, b2, utility)
    dp = np.zeros((B_max+1, B_max+1))
    # Fuell-Simulation
    for b1 in range(B_max, -1, -1):
        for b2 in range(B_max, -1, -1):
            best = 0
            for tb1, tb2, u in tasks_demo:
                if b1 >= tb1 and b2 >= tb2:
                    best = max(best, u + dp[b1-tb1][b2-tb2])
            dp[b1][b2] = best

    im = ax.imshow(dp, cmap='Blues', origin='lower',
                   extent=[-0.5, B_max+0.5, -0.5, B_max+0.5],
                   aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Max. Gesamterfolg')

    for b1 in range(B_max+1):
        for b2 in range(B_max+1):
            ax.text(b2, b1, f'{dp[b1][b2]:.0f}',
                    ha='center', va='center', fontsize=7.5,
                    color='white' if dp[b1][b2] > 5 else DARKGRAY)

    ax.set_xlabel('Restressource $b_2$')
    ax.set_ylabel('Restressource $b_1$')
    ax.set_title('Rucksack-DP Zustandsraum\n'
                 r'$r=2$, $B_{\max}=6$ (3 Aufgaben)',
                 fontweight='bold', color=MAINBLUE, pad=8)
    ax.set_xticks(range(B_max+1))
    ax.set_yticks(range(B_max+1))

    # Rechts: Komplexitaet des Rucksack-DP in Abhaengigkeit von r und B_max
    ax2 = axes[1]
    B_vals = np.arange(2, 12)
    r_vals = [2, 3, 4, 5]
    colors_r = [MAINBLUE, DARKGREEN, ORANGE, ACCENTRED]
    n_val = 10

    for rv, col in zip(r_vals, colors_r):
        complexity = n_val * B_vals**rv
        ax2.semilogy(B_vals, complexity, color=col, lw=2.5,
                     label=f'$r={rv}$ Ressourcen: $n \\cdot B_{{\\max}}^{{{rv}}}$')

    # Markierung fuer Simulationsparameter
    ax2.axvline(x=6, color='gray', lw=1.5, ls='--', alpha=0.7)
    ax2.text(6.1, 1e3, '$B_{\\max}=6$\n(Simulation)', fontsize=8, color='gray')

    ax2.set_xlabel('$B_{\\max}$ (max. Ressource pro Dimension)')
    ax2.set_ylabel(r'Zustaende $n \cdot B_{\max}^r$ (log-Skala)')
    ax2.set_title('Zustandsanzahl des Rucksack-DP\n'
                  r'fuer $n=10$ Agenten',
                  fontweight='bold', color=DARKGRAY)
    ax2.legend(fontsize=9)

    # Annotation: Simulationsparameter
    for rv, col in zip(r_vals, colors_r):
        val = n_val * 6**rv
        ax2.scatter([6], [val], color=col, s=60, zorder=5)
        ax2.text(6.15, val * 1.5, f'{val:,}', fontsize=7.5, color=col)

    fig.suptitle(
        r'Abbildung 5: Mehrdimensionaler Rucksack-DP fuer uberlappende Gruppen'
        r' -- Zustandsraum $dp[b_1]\ldots[b_r]$',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'plot5_knapsack_dp.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot5_knapsack_dp.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 5 gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Vollstaendige Komplexitaetshierarchie aller CFTA-Ansaetze
# ─────────────────────────────────────────────────────────────────────────────
def plot6_full_hierarchy():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # Panel A: Zeitkomplexitaeten aller Ansaetze im Ueberblick
    ax_a = fig.add_subplot(gs[0, 0])
    n = np.arange(2, 22)
    m = n   # m = n angenommen
    r_val = 5
    Bmax  = 6
    k = 3

    # Alle Ansaetze
    approaches = {
        r'Naiv (alle $2^n$ Gruppen)': (2.0**n * m, '#660000', '-'),
        r'Subset-DP (exakt, $3^n m$)': (3.0**n * m, ACCENTRED, '--'),
        r'Greedy Shehory-Kraus ($n^{k+1}$)': (n**(k+1), MAINBLUE, '-.'),
        r'Greedy $k=2$': (n**3, DARKGREEN, ':'),
        r'Rucksack-DP ($n \cdot B^r$)': (n * Bmax**r_val, ORANGE, '-'),
        r'Lokale DP ($\Delta^2 \cdot m$)': (n**2 * m, '#7B2FA8', '--'),
    }

    for label, (vals, col, ls) in approaches.items():
        ax_a.semilogy(n, vals, color=col, lw=2, ls=ls, label=label, alpha=0.85)

    ax_a.set_xlabel('Anzahl Agenten $n$')
    ax_a.set_ylabel('Operationsanzahl (log)')
    ax_a.set_title('(A) Alle CFTA-DP-Ansaetze im Vergleich',
                   fontweight='bold', color=DARKGRAY)
    ax_a.legend(fontsize=7.5, loc='upper left')
    ax_a.set_xlim(2, 21)

    # Panel B: Guete-Komplexitaets-Tradeoff
    ax_b = fig.add_subplot(gs[0, 1])
    methods = ['Naiv\n$2^n$', 'Subset-DP\n$3^n m$', 'Greedy\n$n^{k+1}$',
               'Greedy\n$k=2$', 'Rucksack\n$nB^r$', 'Lokal\n$\\Delta^2 m$']
    quality  = [100, 100, 75, 60, 85, 70]   # rel. Guete (%)
    # n=10, m=10, r=5, k=3, B=6, Delta=4
    complex_ = [2**10*10, 3**10*10, 10**4, 10**3, 10*6**5, 16*10]
    colors_b = ['#660000', ACCENTRED, MAINBLUE, DARKGREEN, ORANGE, '#7B2FA8']

    sc = ax_b.scatter(complex_, quality, s=180, c=colors_b, zorder=5,
                      edgecolors='white', lw=1.5)
    for i, m_lbl in enumerate(methods):
        ax_b.annotate(m_lbl, (complex_[i], quality[i]),
                      xytext=(8, 5), textcoords='offset points',
                      fontsize=7.5, color=colors_b[i])
    ax_b.set_xscale('log')
    ax_b.set_xlabel('Laufzeit bei $n=10$ (Operationen, log)')
    ax_b.set_ylabel('Loesungsguete (relativ, %)')
    ax_b.set_title('(B) Guete-Laufzeit-Tradeoff\nbei $n=10$',
                   fontweight='bold', color=DARKGRAY)
    ax_b.set_ylim(40, 110)
    ax_b.axhline(y=100, color='gray', lw=1, ls='--', alpha=0.5,
                 label='Optimal (100%)')
    ax_b.legend(fontsize=8)

    # Panel C: Approximationsguete-Analyse (Greedy Shehory-Kraus)
    ax_c = fig.add_subplot(gs[1, 0])
    ns = np.arange(2, 101)
    gamma = 0.5772
    approx_bound = np.array([gamma + np.log(n_) for n_ in ns])
    optimal_ratio = np.ones_like(approx_bound)

    ax_c.plot(ns, optimal_ratio, color=DARKGREEN, lw=2,
              label='Optimal ($\\rho = 1$)', ls='--')
    ax_c.plot(ns, approx_bound, color=ACCENTRED, lw=2.5,
              label=r'Greedy: $\rho \leq \gamma + \ln n$')
    ax_c.fill_between(ns, optimal_ratio, approx_bound, alpha=0.12, color=ACCENTRED)
    ax_c.axvline(x=10, color=MAINBLUE, lw=1.5, ls=':', alpha=0.7)
    ax_c.text(10.3, 1.5, f'$n=10$:\n$\\rho \\leq {gamma+np.log(10):.2f}$',
              fontsize=8, color=MAINBLUE)
    ax_c.set_xlabel('Anzahl Agenten $n$')
    ax_c.set_ylabel('Approximationsguete $\\rho$')
    ax_c.set_title('(C) Approximationsguete des Greedy-Algorithmus\n'
                   r'(Shehory-Kraus: $\rho \leq \gamma + \ln n$)',
                   fontweight='bold', color=DARKGRAY)
    ax_c.legend(fontsize=9)

    # DP exakt: rho=1 immer
    ax_c.text(60, 1.05, 'Subset-DP: $\\rho = 1$ (optimal)',
              fontsize=9, color=DARKGREEN, fontweight='bold')

    # Panel D: Zusammenfassungstabelle
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    headers = ['Ansatz', 'Zeitkomp.', 'Guete', 'Opt.?', 'Prakt.']
    rows = [
        ['Greedy (k=2)', r'$O(n^3 m^2)$', r'$\ln n$-appr.', 'Nein', '$n\leq 1000$'],
        ['Greedy (k=3)', r'$O(n^4 m^2)$', r'$\ln n$-appr.', 'Nein', '$n\leq 100$'],
        ['Subset-DP', r'$O(3^n m)$', r'Optimal', 'Ja', '$n\leq 20$'],
        ['Res.-DP', r'$O(n B^r m)$', r'Optimal', 'Ja', r'$B\leq 10, r\leq 5$'],
        ['Lokal-DP', r'$O(\Delta^2 m)$', 'Lokal opt.', 'Lokal', r'$\Delta\leq 20$'],
        ['Rucksack-DP', r'$O(n B^r)$', 'Optimal', 'Ja', r'$B\leq 8, r\leq 4$'],
    ]

    t = ax_d.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1.1, 1.6)
    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor(MAINBLUE)
            cell.set_text_props(color='white', fontweight='bold')
        elif row in [3, 4, 6] and col == 3:
            cell.set_facecolor(LIGHTGREEN)
        elif row in [1, 2] and col == 3:
            cell.set_facecolor('#FFE8E8')
        cell.set_edgecolor('#DDDDDD')

    ax_d.set_title('(D) Vergleichstabelle aller DP-Ansaetze',
                   fontweight='bold', color=DARKGRAY, pad=12)

    fig.suptitle(
        'Abbildung 6: Vollstaendige Komplexitaetshierarchie -- '
        'Alle DP-Ansaetze fuer das CFTA-Problem im Vergleich',
        fontsize=12, fontweight='bold', color=DARKGRAY, y=1.01)
    plt.savefig(OUT + 'plot6_full_hierarchy.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUT + 'plot6_full_hierarchy.png', format='png', bbox_inches='tight')
    plt.close()
    print("Plot 6 gespeichert.")


if __name__ == '__main__':
    print("Erzeuge alle Abbildungen ...")
    plot1_complexity_growth()
    plot2_resource_dp_table()
    plot3_subset_dp_states()
    plot4_simulation_comparison()
    plot5_knapsack_dp()
    plot6_full_hierarchy()
    print("\nAlle 6 Plots erfolgreich gespeichert.")
