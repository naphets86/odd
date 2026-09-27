"""
Plots fuer das neue Kapitel 7:
  Fehlertolerante Subgraph-Analyse (Abb. 17-23)
Autor: Stephan Epp
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binom
import warnings, math
from itertools import product
warnings.filterwarnings('ignore')

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
LIGHTGRAY = '#F5F5F8'
DARKGRAY  = '#3C3C46'
GOLD      = '#C8A400'
PURPLE    = '#6A1E8C'
TEAL      = '#0E6B6B'
ORANGE    = '#D46B00'

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       150,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

BASE_COLORS = {'A': MAINBLUE, 'T': ACCENTRED, 'G': DARKGREEN, 'C': GOLD}
COMPLEMENT  = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def build_adj(seq):
    n = len(seq)
    A = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        A[i, i + 1] = 1  # backbone
    for i in range(n):
        for j in range(n):
            if i != j and COMPLEMENT.get(seq[i]) == seq[j]:
                A[i, j] = 1  # Watson-Crick
    return A

def row_signatures(adj):
    n = adj.shape[0]
    return [sum(int(adj[i, j]) * (2**i) for i in range(n)) for j in range(n)]

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def count_adj_changes(seq, err_pos, new_base):
    """Count how many adjacency matrix entries change when seq[err_pos] -> new_base."""
    original = build_adj(seq)
    mutated = list(seq)
    mutated[err_pos] = new_base
    mutated_adj = build_adj(mutated)
    return int(np.sum(np.abs(original - mutated_adj)))

def simulate_sig_changes(n, e, n_sim=400):
    """Simulate average changed signatures for n-length seq with e errors."""
    bases = list('ATGC')
    changes = []
    for _ in range(n_sim):
        seq = [np.random.choice(list('ATGC')) for _ in range(n)]
        adj_orig = build_adj(seq)
        sig_orig = row_signatures(adj_orig)

        err_positions = np.random.choice(n, size=e, replace=False)
        seq_err = seq[:]
        for pos in err_positions:
            other = [b for b in bases if b != seq_err[pos]]
            seq_err[pos] = np.random.choice(other)
        adj_err = build_adj(seq_err)
        sig_err = row_signatures(adj_err)

        changed = sum(1 for a, b in zip(sig_orig, sig_err) if a != b)
        changes.append(changed)
    return np.mean(changes), np.std(changes)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 17 – Fehlertypen und ihr Einfluss auf die Adjazenzmatrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_17_fehler_typen():
    seq_ref = list("ATGCATGC")
    n = len(seq_ref)

    # Substitution: Position 3 (C->T)
    seq_sub = seq_ref[:]
    seq_sub[3] = 'T'

    # Insertion: insert 'G' after position 3
    seq_ins = seq_ref[:4] + ['G'] + seq_ref[4:]

    # Deletion: remove position 3
    seq_del = seq_ref[:3] + seq_ref[4:]

    adj_ref = build_adj(seq_ref)
    adj_sub = build_adj(seq_sub)
    adj_ins = build_adj(seq_ins)
    adj_del = build_adj(seq_del)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.45)

    cmap_ref = plt.cm.Blues
    cmap_diff = LinearSegmentedColormap.from_list(
        'diff', ['white', ACCENTRED], N=2)

    def draw_adj(ax, adj, title, diff=None, seq=None):
        display = adj.copy().astype(float)
        ax.imshow(display, cmap=cmap_ref, vmin=0, vmax=1, aspect='equal',
                  interpolation='nearest')
        if diff is not None:
            rows, cols_ = np.where(diff != 0)
            for r, c in zip(rows, cols_):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                    fill=True, color=ACCENTRED, alpha=0.5, zorder=3))
        n_ = adj.shape[0]
        ax.set_xticks(range(n_))
        ax.set_yticks(range(n_))
        if seq is not None:
            labels = [f'{s}\n{i}' for i, s in enumerate(seq)]
            ax.set_xticklabels([f'{s}' for s in seq], fontsize=8)
            ax.set_yticklabels([f'{s}' for s in seq], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Spalte j', fontsize=8)
        ax.set_ylabel('Zeile i', fontsize=8)

    # Row 0: reference + substitution side by side
    ax0 = fig.add_subplot(gs[0, 0:2])
    draw_adj(ax0, adj_ref, 'Referenz: ATGCATGC', seq=seq_ref)

    ax1 = fig.add_subplot(gs[0, 2:4])
    diff_sub = np.abs(adj_ref - adj_sub)
    draw_adj(ax1, adj_sub, 'Substitution C→T (Pos. 3)', diff=diff_sub, seq=seq_sub)

    # Row 1: insertion + deletion
    ax2 = fig.add_subplot(gs[1, 0:2])
    draw_adj(ax2, adj_ins, f'Insertion +G (Pos. 4)\n→ {len(seq_ins)}×{len(seq_ins)} Matrix',
             seq=seq_ins)

    ax3 = fig.add_subplot(gs[1, 2:4])
    draw_adj(ax3, adj_del, f'Deletion −C (Pos. 3)\n→ {len(seq_del)}×{len(seq_del)} Matrix',
             seq=seq_del)

    # Count changed entries
    n_changed = int(np.sum(diff_sub))
    ax1.set_title(f'Substitution C→T (Pos. 3)\n{n_changed} geänderte Einträge (rot)',
                  fontsize=10, fontweight='bold')

    fig.suptitle('Abb. 17 – Fehlertypen und ihr Einfluss auf die DNA-Adjazenzmatrix',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig17_fehler_typen.pdf')
    plt.close()
    print("fig17 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 18 – Bitfehler als Funktion von n und e
# ─────────────────────────────────────────────────────────────────────────────
def plot_18_bitfehler():
    np.random.seed(42)
    ns = np.arange(5, 51)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1, ax2 = axes

    for e_val, color, style in [(1, MAINBLUE, '-'), (5, DARKGREEN, '--'), (10, ACCENTRED, ':')]:
        upper = [2 * e_val * n for n in ns]
        expected = [e_val * (n + 3) / 4 for n in ns]
        empirical_mean = []
        empirical_std  = []
        for n in ns:
            if e_val >= n:
                empirical_mean.append(np.nan)
                empirical_std.append(np.nan)
                continue
            mu, sd = simulate_sig_changes(n, e_val, n_sim=200)
            empirical_mean.append(mu)
            empirical_std.append(sd)

        ax1.plot(ns, upper, color=color, linestyle=':', alpha=0.6,
                 label=f'Obere Schranke $2en$, $e={e_val}$')
        ax1.plot(ns, expected, color=color, linestyle=style,
                 label=f'Erwartungswert $e(n+3)/4$, $e={e_val}$')
        em = np.array(empirical_mean)
        es = np.array(empirical_std)
        valid = ~np.isnan(em)
        ax1.fill_between(ns[valid], em[valid]-es[valid], em[valid]+es[valid],
                         color=color, alpha=0.12)
        ax1.scatter(ns[valid][::5], em[valid][::5], color=color, s=18, zorder=5)

    ax1.set_xlabel('Sequenzlänge $n$')
    ax1.set_ylabel('Geänderte Signatur-Einträge $\\Delta\\rho$')
    ax1.set_title('Geänderte Signaturen vs. Sequenzlänge')
    ax1.legend(fontsize=8, loc='upper left')

    # Right: relative change rate vs normalized error rate
    ax2.set_xlabel('Normierte Fehlerrate $e/n$')
    ax2.set_ylabel('Relative Änderungsrate $\\Delta\\rho / n$')
    ax2.set_title('Relative Signaturkorruption vs. normierter Fehlerrate')

    for n_val, color in [(20, MAINBLUE), (50, DARKGREEN), (100, ACCENTRED), (150, GOLD)]:
        e_fracs = np.linspace(0.01, 0.3, 20)
        rates = []
        for ef in e_fracs:
            e_val = max(1, int(ef * n_val))
            if e_val >= n_val:
                break
            mu, _ = simulate_sig_changes(n_val, e_val, n_sim=150)
            rates.append(mu / n_val)
        e_plot = e_fracs[:len(rates)]
        ax2.plot(e_plot, rates, color=color, marker='o', markersize=4,
                 label=f'$n={n_val}$')
        # theoretical line
        theory = [(n_val + 3) / 4 * ef for ef in e_fracs[:len(rates)]]
        ax2.plot(e_plot, theory, color=color, linestyle='--', alpha=0.5)

    ax2.legend(fontsize=9)
    fig.suptitle('Abb. 18 – Signaturkorruption durch Substitutionsfehler',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig18_bitfehler.pdf')
    plt.close()
    print("fig18 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 19 – Toleranzkurven: e_true vs epsilon für verschiedene n
# ─────────────────────────────────────────────────────────────────────────────
def plot_19_toleranz_kurven():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    eps_range = np.linspace(0.001, 0.5, 200)
    ns = [50, 150, 300, 1000]
    colors = [MAINBLUE, DARKGREEN, GOLD, ACCENTRED]

    # Left: absolute max errors
    for n_val, col in zip(ns, colors):
        e_max = [eps * n_val / ((n_val + 3) / 4) for eps in eps_range]
        ax1.plot(eps_range, e_max, color=col, label=f'$n = {n_val}$')
    ax1.set_xlabel('Toleranzparameter $\\varepsilon$')
    ax1.set_ylabel('Maximale tolerierbare Fehleranzahl $e_{\\max}$')
    ax1.set_title('Absolute Fehlertoleranz $e_{\\max}(\\varepsilon, n)$')
    ax1.legend()

    # Right: normalized error rate eps_true
    for n_val, col in zip(ns, colors):
        eps_true = [4 * eps / (n_val + 3) for eps in eps_range]
        ax2.plot(eps_range, eps_true, color=col, label=f'$n = {n_val}$')

    # Technology lines
    tech = {
        'Illumina ($< 0{,}1\\%$)':    (0.001, DARKGRAY, ':'),
        'PacBio HiFi ($\\sim 0{,}1\\%$)': (0.001, PURPLE, '-.'),
        'Nanopore roh ($\\sim 10\\%$)': (0.10, ACCENTRED, '--'),
    }
    for label, (rate, col, ls) in tech.items():
        ax2.axhline(rate, color=col, linestyle=ls, alpha=0.7,
                    label=f'{label}', linewidth=1.5)

    ax2.set_xlabel('Toleranzparameter $\\varepsilon$')
    ax2.set_ylabel('Max. norm. Fehlerrate $\\varepsilon_{\\mathrm{true}}$')
    ax2.set_title('Normierte Fehlertoleranz vs. Technologie')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)

    fig.suptitle('Abb. 19 – Maximale tolerierbare Fehlerrate als Funktion von $\\varepsilon$',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig19_toleranz_kurven.pdf')
    plt.close()
    print("fig19 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 20 – Falsch-Positiv-Rate (FPR) und ROC-Kurve
# ─────────────────────────────────────────────────────────────────────────────
def plot_20_fpr():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    n_B = 300
    eps_range = np.linspace(0.001, 0.70, 300)
    ns_a = [20, 50, 150]
    colors_a = [MAINBLUE, DARKGREEN, ACCENTRED]

    # Left: FPR(eps)
    for n_A, col in zip(ns_a, colors_a):
        fprs = []
        for eps in eps_range:
            tau = math.ceil((1 - eps) * n_A)
            if tau <= 0:
                fprs.append(1.0)
                continue
            p_fpr = 1 - binom.cdf(tau - 1, n_A, 0.25)
            fpr = min(1.0, n_B * p_fpr)
            fprs.append(fpr)
        ax1.semilogy(eps_range, np.clip(fprs, 1e-100, 1.0),
                     color=col, label=f'$n_A = {n_A}$, $n_B = {n_B}$')

    ax1.axhline(1e-6, color=DARKGRAY, linestyle='--', alpha=0.7,
                label='Akzeptanzgrenze $10^{-6}$')
    ax1.set_xlabel('Toleranzparameter $\\varepsilon$')
    ax1.set_ylabel('Falsch-Positiv-Rate (FPR)')
    ax1.set_title('FPR als Funktion von $\\varepsilon$')
    ax1.legend(fontsize=9)

    # Right: ROC curve
    # Sensitivity: P(detect | true subgraph, e errors)
    # We simulate for n_A=150, varying eps and e (number of errors)
    n_A = 150
    np.random.seed(99)

    ax2.set_xlabel('Falsch-Positiv-Rate (FPR)')
    ax2.set_ylabel('Sensitivität (TPR)')
    ax2.set_title('ROC-Kurve für $n_A = 150$, $n_B = 300$')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Zufälliger Klassifikator')

    tech_points = {
        'Illumina\n($\\varepsilon=0.01$)':  (0.01,  ORANGE),
        'PacBio HiFi\n($\\varepsilon=0.05$)': (0.05,  DARKGREEN),
        'Nanopore korr.\n($\\varepsilon=0.15$)': (0.15,  PURPLE),
    }

    eps_vals = np.linspace(0.001, 0.70, 500)
    fprs_roc = []
    tprs_roc = []
    for eps in eps_vals:
        tau = math.ceil((1 - eps) * n_A)
        if tau <= 0:
            fpr = 1.0; tpr = 1.0
        else:
            p_fp  = 1 - binom.cdf(tau - 1, n_A, 0.25)
            fpr   = min(1.0, n_B * p_fp)
            # TPR: probability that a true subgraph with error fraction eps/2
            # still reaches LCS >= tau
            e_expected = int(eps / 2 * n_A * (n_A + 3) / 4)
            lcs_after  = max(0, n_A - e_expected)
            tpr  = 1.0 if lcs_after >= tau else max(0.0, 1 - binom.cdf(
                tau - lcs_after - 1, n_A - lcs_after, 0.5))
        fprs_roc.append(fpr)
        tprs_roc.append(tpr)

    ax2.plot(fprs_roc, tprs_roc, color=MAINBLUE, lw=2, label='$n_A=150$')

    for label, (eps_val, col) in tech_points.items():
        tau = math.ceil((1 - eps_val) * n_A)
        p_fp = 1 - binom.cdf(tau - 1, n_A, 0.25)
        fpr_pt = min(1.0, n_B * p_fp)
        e_exp  = int(eps_val / 2 * n_A * (n_A + 3) / 4)
        lcs_after = max(0, n_A - e_exp)
        tpr_pt = 1.0 if lcs_after >= tau else 0.95
        ax2.scatter([fpr_pt], [tpr_pt], color=col, s=80, zorder=5,
                    label=label)

    ax2.legend(fontsize=8, loc='lower right')
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle('Abb. 20 – Falsch-Positiv-Rate und ROC-Kurve des $\\varepsilon$-Subgraph Algorithmus',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig20_fpr.pdf')
    plt.close()
    print("fig20 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 21 – Indel-Einfluss auf LCS
# ─────────────────────────────────────────────────────────────────────────────
def plot_21_indel():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Left: alignment schema
    ax1.axis('off')
    ax1.set_xlim(0, 12)
    ax1.set_ylim(-1, 6)

    ref  = list("ATGCATGC")
    read = list("AT-KATGXC")  # '-' = deletion at pos 2, 'X' = insertion at pos 7

    y_ref  = 4.5
    y_read = 2.5
    x_start = 1.0

    ax1.text(0.2, y_ref + 0.7,  'Referenz $S_A$:', fontsize=10, fontweight='bold',
             color=MAINBLUE)
    ax1.text(0.2, y_read + 0.7, 'Read $\\hat{S}_B$ (aligniert):', fontsize=10,
             fontweight='bold', color=DARKGREEN)

    ref_display  = list("ATGCATGC")
    read_display = list("AT_CATGXC")  # '_' = deletion gap, 'X' = insertion

    positions = np.linspace(x_start, 10.5, len(read_display))
    for i, (rb, rd, xp) in enumerate(zip(
            ref_display + [''], read_display, positions)):
        col_r = BASE_COLORS.get(rb, DARKGRAY)
        col_d = BASE_COLORS.get(rd, DARKGRAY)

        if rd == '_':
            # deletion: show gap in read
            ax1.add_patch(plt.Rectangle((xp-0.38, y_read-0.35), 0.76, 0.7,
                fill=True, color=ACCENTRED, alpha=0.25))
            ax1.text(xp, y_read, '—', ha='center', va='center',
                     fontsize=12, color=ACCENTRED)
            ax1.text(xp, y_ref, rb, ha='center', va='center',
                     fontsize=12, fontweight='bold', color=col_r)
            ax1.annotate('', xy=(xp, y_read+0.4), xytext=(xp, y_ref-0.4),
                         arrowprops=dict(arrowstyle='<->', color=ACCENTRED,
                                         lw=1.5))
            ax1.text(xp + 0.5, (y_ref + y_read)/2, 'Del.', fontsize=8,
                     color=ACCENTRED)
        elif rd == 'X':
            # insertion
            ax1.add_patch(plt.Rectangle((xp-0.38, y_read-0.35), 0.76, 0.7,
                fill=True, color=PURPLE, alpha=0.25))
            ax1.text(xp, y_read, 'X', ha='center', va='center',
                     fontsize=12, fontweight='bold', color=PURPLE)
            ax1.annotate('', xy=(xp, y_read+0.35),
                         xytext=(xp, y_ref - 0.35),
                         arrowprops=dict(arrowstyle='->', color=PURPLE,
                                         lw=1.5))
            ax1.text(xp + 0.5, (y_ref + y_read) / 2, 'Ins.', fontsize=8,
                     color=PURPLE)
        else:
            if i < len(ref_display):
                ax1.text(xp, y_ref, ref_display[i], ha='center', va='center',
                         fontsize=12, fontweight='bold', color=col_r)
            ax1.text(xp, y_read, rd, ha='center', va='center',
                     fontsize=12, fontweight='bold', color=col_d)
            if rb == rd:
                ax1.plot([xp, xp], [y_ref - 0.38, y_read + 0.38],
                         color=DARKGRAY, lw=0.8, alpha=0.4)

    ax1.set_title('Alignment-Schema: Deletion (rot) + Insertion (lila)',
                  fontsize=11, fontweight='bold')

    # Right: minimum LCS vs indel count
    ns_a = [50, 100, 150]
    e_range = np.arange(0, 31)
    colors_ = [MAINBLUE, DARKGREEN, ACCENTRED]

    for n_A, col in zip(ns_a, colors_):
        min_lcs = [max(0, n_A - e) for e in e_range]
        ax2.plot(e_range, min_lcs, color=col, lw=2, label=f'$n_A={n_A}$')

    # Thresholds
    for eps_val, ls, label in [(0.10, '--', '$\\varepsilon=0.10$'),
                                (0.15, ':',  '$\\varepsilon=0.15$')]:
        for n_A, col in zip(ns_a, colors_):
            tau = (1 - eps_val) * n_A
            ax2.axhline(tau, color=col, linestyle=ls, alpha=0.6, lw=1.4)

    # legend entries for thresholds
    ax2.plot([], [], 'k--', lw=1.4, alpha=0.7, label='Schwelle $(1-\\varepsilon)n_A$, $\\varepsilon=0.10$')
    ax2.plot([], [], 'k:',  lw=1.4, alpha=0.7, label='Schwelle $(1-\\varepsilon)n_A$, $\\varepsilon=0.15$')

    ax2.set_xlabel('Indel-Anzahl $e = e_I + e_D$')
    ax2.set_ylabel('Minimale LCS nach Indels')
    ax2.set_title('Minimale LCS vs. Indel-Anzahl')
    ax2.legend(fontsize=9)

    fig.suptitle('Abb. 21 – Einfluss von Insertionen und Deletionen auf die Subgraph-Erkennung',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig21_indel.pdf')
    plt.close()
    print("fig21 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 22 – Optimale Schwelle epsilon* für verschiedene Technologien
# ─────────────────────────────────────────────────────────────────────────────
def plot_22_optimale_schwelle():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Left: epsilon*(p_S) for different n, with technology points
    ns_ = [100, 300, 1000]
    colors_ = [MAINBLUE, DARKGREEN, ACCENTRED]
    z_99 = 2.326  # z_{0.99}

    ps_range = np.linspace(0.0001, 0.15, 300)

    for n_val, col in zip(ns_, colors_):
        eps_star = []
        for ps in ps_range:
            E_B = ps * (n_val + 3) / 4
            V_B = ps * (1 - ps) / n_val * ((n_val + 3) / 4)**2
            es  = E_B + z_99 * math.sqrt(V_B)
            eps_star.append(es)
        ax1.semilogy(ps_range, eps_star, color=col, lw=2, label=f'$n={n_val}$')

    # Technology points at n=150
    tech_pts = {
        'Illumina NovaSeq X':     (0.001,  ORANGE,    'o'),
        'PacBio HiFi':            (0.001,  DARKGREEN, 's'),
        'Nanopore R10 (korr.)':   (0.005,  PURPLE,    '^'),
        'Nanopore R10 (roh)':     (0.07,   ACCENTRED, 'D'),
    }
    n_pt = 150
    for label, (ps, col, marker) in tech_pts.items():
        E_B = ps * (n_pt + 3) / 4
        V_B = ps * (1 - ps) / n_pt * ((n_pt + 3) / 4)**2
        es  = E_B + z_99 * math.sqrt(V_B)
        ax1.scatter([ps], [es], color=col, marker=marker, s=80, zorder=5,
                    label=label + f' ($n=150$)')

    ax1.axhline(1.0, color=DARKGRAY, linestyle='--', alpha=0.5,
                label='$\\varepsilon^* = 1$ (Grenze)')
    ax1.set_xlabel('Substitutionsrate $p_S$')
    ax1.set_ylabel('Optimale Schwelle $\\varepsilon^*$')
    ax1.set_title('Optimale Schwelle $\\varepsilon^*$ vs. Fehlerrate \n(nur Substitutionen, $z_{0.99}$)')
    ax1.legend(fontsize=7.5, loc='upper left')

    # Right: Heatmap eps*(p_S, p_I + p_D) for n=150
    n_h = 150
    ps_vals = np.linspace(0.0005, 0.08, 60)
    pi_vals = np.linspace(0.0000, 0.08, 60)
    Z = np.zeros((len(pi_vals), len(ps_vals)))
    for i, pi in enumerate(pi_vals):
        for j, ps in enumerate(ps_vals):
            E_B = ps * (n_h + 3) / 4 + pi + pi  # p_I = p_D = pi/2
            V_S = ps * (1-ps) / n_h * ((n_h+3)/4)**2
            V_I = pi * (1-pi) / n_h
            es  = E_B + z_99 * math.sqrt(V_S + 2*V_I)
            Z[i, j] = min(es, 2.0)

    cmap_hm = LinearSegmentedColormap.from_list(
        'hm', [LIGHTGRAY, MAINBLUE, ACCENTRED], N=256)
    im = ax2.contourf(ps_vals * 100, pi_vals * 100, Z,
                      levels=20, cmap=cmap_hm)
    plt.colorbar(im, ax=ax2, label='$\\varepsilon^*$')
    ax2.set_xlabel('Substitutionsrate $p_S$ (%)')
    ax2.set_ylabel('Indel-Rate $p_I + p_D$ (%)')
    ax2.set_title(f'Heatmap $\\varepsilon^*(p_S, p_I+p_D)$ für $n={n_h}$')

    # Mark technologies
    for label, (ps, col, marker) in list(tech_pts.items())[:3]:
        ax2.scatter([ps*100], [0.2], color=col, marker=marker, s=60, zorder=5)
        ax2.annotate(label.split()[0], (ps*100, 0.2), textcoords='offset points',
                     xytext=(5, 5), fontsize=7, color=col)

    fig.suptitle('Abb. 22 – Optimale Schwellwahl $\\varepsilon^*$ nach Sequenziertechnologie',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig22_optimale_schwelle.pdf')
    plt.close()
    print("fig22 saved")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 23 – Vollständiges Beispiel: epsilon-Subgraph Algorithmus
# ─────────────────────────────────────────────────────────────────────────────
def plot_23_fuzzy_beispiel():
    np.random.seed(12)
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.45)

    seq_A  = list("ATGCATGC")
    n_A    = len(seq_A)
    adj_A  = build_adj(seq_A)
    rho_A  = row_signatures(adj_A)

    # Introduce 1 substitution (pos 4: A->T) + 1 insertion (pos 8: +G)
    seq_B_err  = list("CATGTTGCTAT")  # longer read with errors
    adj_B_err  = build_adj(seq_B_err)
    rho_B_err  = row_signatures(adj_B_err)

    eps        = 0.15
    tau        = math.ceil((1 - eps) * n_A)      # = 7

    # ---- Panel A: Adj matrix of S_A ----
    ax_A = fig.add_subplot(gs[0, 0])
    im = ax_A.imshow(adj_A, cmap=plt.cm.Blues, vmin=0, vmax=1, aspect='equal',
                     interpolation='nearest')
    ax_A.set_xticks(range(n_A))
    ax_A.set_yticks(range(n_A))
    ax_A.set_xticklabels(seq_A, fontsize=9)
    ax_A.set_yticklabels(seq_A, fontsize=9)
    ax_A.set_title(f'Referenz $G_A$: ATGCATGC\n($n_A = {n_A}$ Knoten)',
                   fontsize=11, fontweight='bold')
    ax_A.set_xlabel('Spalte j')
    ax_A.set_ylabel('Zeile i')

    # ---- Panel B: Adj matrix of B with errors marked ----
    ax_B = fig.add_subplot(gs[0, 1])
    adj_B_clean  = build_adj(list("CATGCATGC"))   # clean version for diff
    adj_B_err_pad = build_adj(seq_B_err)
    im2 = ax_B.imshow(adj_B_err_pad, cmap=plt.cm.Blues, vmin=0, vmax=1,
                      aspect='equal', interpolation='nearest')
    ax_B.set_xticks(range(len(seq_B_err)))
    ax_B.set_yticks(range(len(seq_B_err)))
    ax_B.set_xticklabels(seq_B_err, fontsize=8)
    ax_B.set_yticklabels(seq_B_err, fontsize=8)
    # Mark substitution position (index 5 in read = 'T' instead of 'A')
    ax_B.add_patch(plt.Rectangle((4.5, -0.5), 1, len(seq_B_err),
        fill=True, color=ACCENTRED, alpha=0.18, zorder=3))
    ax_B.add_patch(plt.Rectangle((-0.5, 4.5), len(seq_B_err), 1,
        fill=True, color=ACCENTRED, alpha=0.18, zorder=3))
    # Mark insertion position (last 'T' at end)
    ax_B.add_patch(plt.Rectangle((9.5, -0.5), 1, len(seq_B_err),
        fill=True, color=PURPLE, alpha=0.18, zorder=3))
    ax_B.set_title(f'Read $\\hat{{S}}_B$: CATG\\ T\\ TGCTAT\n'
                   f'(Subst. rot, Ins. lila; {len(seq_B_err)} Basen)',
                   fontsize=10, fontweight='bold')
    ax_B.set_xlabel('Spalte j')

    # ---- Panel C: LCS values per rotation ----
    ax_C = fig.add_subplot(gs[1, 0])
    n_B = len(rho_B_err)
    lcs_vals = []
    for r in range(n_B):
        rot_B = rho_B_err[r:] + rho_B_err[:r]
        best_lcs = 0
        for s in range(n_B - n_A + 1):
            window = rot_B[s: s + n_A]
            best_lcs = max(best_lcs, lcs_len(rho_A, window))
        lcs_vals.append(best_lcs)

    rotations = list(range(n_B))
    bar_colors = [DARKGREEN if v >= tau else ACCENTRED for v in lcs_vals]
    bars = ax_C.bar(rotations, lcs_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    ax_C.axhline(tau, color=DARKGRAY, linestyle='--', lw=2,
                 label=f'Schwelle $\\tau = {tau}$ ($\\varepsilon = {eps}$)')
    ax_C.axhline(2, color=TEAL, linestyle=':', lw=1.5,
                 label='Original-Schwelle LCS $\\geq 2$')
    ax_C.set_xlabel('Rotation $r$')
    ax_C.set_ylabel('Bester LCS-Wert (über alle Fenster)')
    ax_C.set_title('LCS-Werte je Rotation', fontweight='bold')
    ax_C.set_xticks(rotations)
    ax_C.legend(fontsize=9)
    # Annotate max
    best_r = int(np.argmax(lcs_vals))
    ax_C.annotate(f'Max LCS={lcs_vals[best_r]}\n$r={best_r}$',
                  xy=(best_r, lcs_vals[best_r]),
                  xytext=(best_r + 0.5, lcs_vals[best_r] + 0.3),
                  fontsize=9, color=DARKGREEN,
                  arrowprops=dict(arrowstyle='->', color=DARKGREEN))

    # ---- Panel D: Original vs epsilon comparison ----
    ax_D = fig.add_subplot(gs[1, 1])
    eps_values = np.linspace(0, 0.5, 200)
    tau_vals   = [math.ceil((1 - e) * n_A) for e in eps_values]

    # TPR approximation: fraction of lcs_vals >= tau
    tpr_eps = [np.mean([1 if v >= t else 0 for v in lcs_vals])
               for t in tau_vals]
    # FPR approximation using Binomial model
    fpr_eps = []
    n_B_ref = 300
    for eps_v in eps_values:
        tau_v = math.ceil((1 - eps_v) * n_A)
        if tau_v <= 0:
            fpr_eps.append(1.0)
        else:
            p_fp = 1 - binom.cdf(tau_v - 1, n_A, 0.25)
            fpr_eps.append(min(1.0, n_B_ref * p_fp))

    ax_D.plot(eps_values, tpr_eps, color=DARKGREEN, lw=2.5,
              label='Sensitivität (TPR)')
    ax_D.plot(eps_values, np.clip(fpr_eps, 0, 1), color=ACCENTRED, lw=2.5,
              label='Falsch-Pos.-Rate (FPR)')
    ax_D.axvline(eps, color=DARKGRAY, linestyle='--', lw=1.5,
                 label=f'Gewähltes $\\varepsilon = {eps}$')
    # Mark operating point
    idx = np.argmin(np.abs(eps_values - eps))
    ax_D.scatter([eps], [tpr_eps[idx]], color=DARKGREEN, s=80, zorder=5)
    ax_D.scatter([eps], [min(1.0, fpr_eps[idx])], color=ACCENTRED, s=80, zorder=5)
    ax_D.set_xlabel('Toleranzparameter $\\varepsilon$')
    ax_D.set_ylabel('Rate')
    ax_D.set_title('Sensitivität vs. FPR als Funktion von $\\varepsilon$',
                   fontweight='bold')
    ax_D.legend(fontsize=9)

    fig.suptitle('Abb. 23 – Vollständiges Beispiel: $\\varepsilon$-Subgraph Algorithmus',
                 fontweight='bold', fontsize=12)
    plt.savefig('figures/fig23_fuzzy_beispiel.pdf')
    plt.close()
    print("fig23 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    print("Generating figures 17–23 ...")
    plot_17_fehler_typen()
    plot_18_bitfehler()
    plot_19_toleranz_kurven()
    plot_20_fpr()
    plot_21_indel()
    plot_22_optimale_schwelle()
    plot_23_fuzzy_beispiel()
    print("All figures generated successfully.")

