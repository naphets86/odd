#!/usr/bin/env python3
"""
Generates all matplotlib figures for the ARM Cortex Architecture paper.
Author: Stephan Epp
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit
import os

OUT = "."
# os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'A': '#0077CC',
    'M': '#00AA55',
    'R': '#DD4422',
    'H': '#9933CC',   # hypothetical HX
    'accent': '#FF9900',
    'grey': '#888888',
}

# ────────────────────────────────────────────────────────────
# PLOT 1: Performance vs. Power  (DMIPS/MHz  vs  mW/MHz)
# ────────────────────────────────────────────────────────────
def plot1_perf_power():
    cores = {
        # name, DMIPS/MHz, mW/MHz, family
        'M0+':   (0.93,  0.011, 'M'),
        'M3':    (1.25,  0.032, 'M'),
        'M4':    (1.27,  0.038, 'M'),
        'M7':    (2.14,  0.10,  'M'),
        'M33':   (1.50,  0.045, 'M'),
        'A5':    (1.57,  0.05,  'A'),
        'A7':    (2.00,  0.12,  'A'),
        'A53':   (2.30,  0.25,  'A'),
        'A55':   (2.70,  0.22,  'A'),
        'A72':   (4.70,  0.90,  'A'),
        'A76':   (6.30,  1.20,  'A'),
        'R4':    (1.57,  0.10,  'R'),
        'R5':    (2.00,  0.18,  'R'),
        'R52':   (2.20,  0.30,  'R'),
        'R7':    (3.77,  0.55,  'R'),
        # New proposed HX-1 and HX-2
        'HX-1': (8.10,  1.50,  'H'),
        'HX-2': (10.5,  2.10,  'H'),
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, (perf, pwr, fam) in cores.items():
        col = COLORS[fam]
        marker = {'A': 'o', 'M': 's', 'R': '^', 'H': 'D'}[fam]
        ax.scatter(pwr, perf, color=col, marker=marker, s=90, zorder=5)
        ax.annotate(name, (pwr, perf), textcoords='offset points',
                    xytext=(6, 3), fontsize=8.5, color=col)

    # Pareto frontier
    pts = sorted(cores.values(), key=lambda x: x[1])
    pareto = []
    best = -1
    for perf, pwr, _ in pts:
        if perf > best:
            pareto.append((pwr, perf))
            best = perf
    px, py = zip(*pareto)
    ax.plot(px, py, '--', color=COLORS['grey'], lw=1.2, label='Pareto-Grenze')

    # Legend patches
    patches = [
        mpatches.Patch(color=COLORS['A'], label='Cortex-A'),
        mpatches.Patch(color=COLORS['M'], label='Cortex-M'),
        mpatches.Patch(color=COLORS['R'], label='Cortex-R'),
        mpatches.Patch(color=COLORS['H'], label='Cortex-HX (neu)'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='upper left')

    ax.set_xlabel('Leistungsaufnahme [mW/MHz]')
    ax.set_ylabel('Rechenleistung [DMIPS/MHz]')
    ax.set_title('Abb. 1: Performance–Power-Raum der ARM-Cortex-Familien\n(inkl. neuer HX-Architektur)')
    ax.set_xscale('log')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{OUT}/plot1_perf_power.pdf')
    plt.savefig(f'{OUT}/plot1_perf_power.png')
    plt.close()
    print("Plot 1 done")

# ────────────────────────────────────────────────────────────
# PLOT 2: Latenz-Deterministik  (Worst-Case Execution Time)
# ────────────────────────────────────────────────────────────
def plot2_wcet():
    families = ['Cortex-A\n(A76)', 'Cortex-M\n(M7)', 'Cortex-R\n(R52)', 'Cortex-HX\n(HX-1, neu)']
    bcet   = [12,  3,  2,  1.5]   # Best-Case  [us]
    avg    = [45, 10,  5,  3.2]   # Average    [us]
    wcet   = [310, 22, 8,  5.1]   # Worst-Case [us]
    colors = [COLORS['A'], COLORS['M'], COLORS['R'], COLORS['H']]

    x = np.arange(len(families))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5.5))

    b1 = ax.bar(x - width, bcet,  width, label='BCET', color=[c+'88' for c in colors], edgecolor=colors, linewidth=1.2)
    b2 = ax.bar(x,          avg,  width, label='ACET', color=colors, alpha=0.75)
    b3 = ax.bar(x + width,  wcet, width, label='WCET', color=colors, edgecolor='black', linewidth=1.2, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel('Ausführungszeit [µs]')
    ax.set_title('Abb. 2: BCET / ACET / WCET-Vergleich der Cortex-Familien')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    # WCET-Spread ratio annotations
    for i, (b, w) in enumerate(zip(bcet, wcet)):
        ratio = w / b
        ax.annotate(f'×{ratio:.0f}', xy=(x[i] + width, w),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=8.5, color='black')

    plt.tight_layout()
    plt.savefig(f'{OUT}/plot2_wcet.pdf')
    plt.savefig(f'{OUT}/plot2_wcet.png')
    plt.close()
    print("Plot 2 done")

# ────────────────────────────────────────────────────────────
# PLOT 3: Pipeline-Tiefe vs. IPC  (neue Metrik: RDI)
# ────────────────────────────────────────────────────────────
def plot3_pipeline_ipc():
    data = {
        'M0':  (2,  0.90), 'M3': (3, 1.01), 'M4': (3, 1.02),
        'M7':  (6,  1.85), 'M33': (3, 1.15),
        'A5':  (8,  1.60), 'A7': (8, 2.10), 'A53': (8, 2.20),
        'A55': (8,  2.60), 'A72': (13, 3.50), 'A76': (13, 4.30),
        'R4':  (8,  1.57), 'R5': (8, 1.75), 'R52': (8, 2.10), 'R7': (11, 2.80),
        'HX-1': (10, 5.20), 'HX-2': (10, 6.80),
    }
    families = {'M0':'M','M3':'M','M4':'M','M7':'M','M33':'M',
                'A5':'A','A7':'A','A53':'A','A55':'A','A72':'A','A76':'A',
                'R4':'R','R5':'R','R52':'R','R7':'R',
                'HX-1':'H','HX-2':'H'}

    depths, ipcs, cols, names = [], [], [], []
    for name, (d, ipc) in data.items():
        depths.append(d); ipcs.append(ipc)
        cols.append(COLORS[families[name]])
        names.append(name)

    depths = np.array(depths); ipcs = np.array(ipcs)

    # Fit: IPC = a * log(depth) + b
    def log_fit(x, a, b): return a * np.log(x) + b
    popt, _ = curve_fit(log_fit, depths, ipcs)
    xfit = np.linspace(2, 14, 200)
    yfit = log_fit(xfit, *popt)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(depths, ipcs, c=cols, s=90, zorder=5)
    ax.plot(xfit, yfit, '--', color=COLORS['grey'], lw=1.4,
            label=f'Fit: IPC = {popt[0]:.2f}·ln(d) + {popt[1]:.2f}')
    for i, name in enumerate(names):
        ax.annotate(name, (depths[i], ipcs[i]), fontsize=8,
                    xytext=(4, 3), textcoords='offset points',
                    color=cols[i])

    patches = [mpatches.Patch(color=COLORS[f], label=f'Cortex-{f}' if f != 'H' else 'Cortex-HX (neu)')
               for f in ['A','M','R','H']]
    ax.legend(handles=patches + [plt.Line2D([0],[0],ls='--',color=COLORS['grey'],label='Log-Fit')],
              fontsize=9, loc='upper left')

    ax.set_xlabel('Pipeline-Tiefe [Stufen]')
    ax.set_ylabel('IPC (Instructions Per Cycle)')
    ax.set_title('Abb. 3: Pipeline-Tiefe vs. IPC — logarithmisches Skalierungsgesetz')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{OUT}/plot3_pipeline_ipc.pdf')
    plt.savefig(f'{OUT}/plot3_pipeline_ipc.png')
    plt.close()
    print("Plot 3 done")

# ────────────────────────────────────────────────────────────
# PLOT 4: AISA-Metrik (neue formale Metrik)
# ────────────────────────────────────────────────────────────
def plot4_aisa():
    # AISA = (DMIPS/MHz) / sqrt(mW/MHz * Pipeline-Stufen)
    # Normalisiert auf Cortex-M4 = 1.0
    cores_data = {
        'M0+':  (0.93,  0.011, 2),
        'M3':   (1.25,  0.032, 3),
        'M4':   (1.27,  0.038, 3),   # Ref
        'M7':   (2.14,  0.10,  6),
        'M33':  (1.50,  0.045, 3),
        'A5':   (1.57,  0.05,  8),
        'A7':   (2.00,  0.12,  8),
        'A53':  (2.30,  0.25,  8),
        'A55':  (2.70,  0.22,  8),
        'A72':  (4.70,  0.90,  13),
        'A76':  (6.30,  1.20,  13),
        'R4':   (1.57,  0.10,  8),
        'R5':   (2.00,  0.18,  8),
        'R52':  (2.20,  0.30,  8),
        'R7':   (3.77,  0.55,  11),
        'HX-1': (8.10,  1.50,  10),
        'HX-2': (10.5,  2.10,  10),
    }
    families = {'M0+':'M','M3':'M','M4':'M','M7':'M','M33':'M',
                'A5':'A','A7':'A','A53':'A','A55':'A','A72':'A','A76':'A',
                'R4':'R','R5':'R','R52':'R','R7':'R',
                'HX-1':'H','HX-2':'H'}

    ref_perf, ref_pwr, ref_pipe = cores_data['M4']
    ref_aisa = ref_perf / np.sqrt(ref_pwr * ref_pipe)

    names, aisas, cols = [], [], []
    for name, (p, pw, d) in cores_data.items():
        aisa = (p / np.sqrt(pw * d)) / ref_aisa
        names.append(name); aisas.append(aisa)
        cols.append(COLORS[families[name]])

    order = np.argsort(aisas)
    names  = [names[i]  for i in order]
    aisas  = [aisas[i]  for i in order]
    cols   = [cols[i]   for i in order]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(names, aisas, color=cols, edgecolor='white', linewidth=0.5)
    ax.axvline(1.0, color='black', ls='--', lw=1.2, label='Referenz: Cortex-M4')

    for bar, val in zip(bars, aisas):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    patches = [mpatches.Patch(color=COLORS[f], label=f'Cortex-{f}' if f != 'H' else 'Cortex-HX (neu)')
               for f in ['A','M','R','H']]
    ax.legend(handles=patches + [plt.Line2D([0],[0],ls='--',color='black',label='Referenz M4')],
              fontsize=9)

    ax.set_xlabel('AISA-Metrik (normiert auf Cortex-M4)')
    ax.set_title('Abb. 4: AISA — Architecture-Integrated Scaled Aptitude\n'
                 r'AISA $= \frac{\mathrm{DMIPS/MHz}}{\sqrt{P_{\mathrm{dyn}} \cdot d_{\mathrm{pipe}}}}$')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{OUT}/plot4_aisa.pdf')
    plt.savefig(f'{OUT}/plot4_aisa.png')
    plt.close()
    print("Plot 4 done")

# ────────────────────────────────────────────────────────────
# PLOT 5: Cortex-HX Architekturschema (Block-Diagramm)
# ────────────────────────────────────────────────────────────
def plot5_hx_block():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Abb. 5: Cortex-HX Architektur — Blockdiagramm (Entwurf)', fontsize=13, pad=14)

    def box(x, y, w, h, label, color='#DDEEFF', fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.08', facecolor=color,
            edgecolor='#334466', linewidth=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color='#334466', lw=1.3))

    # Fetch + Decode
    box(0.3, 6.4, 2.5, 1.0, 'Fetch\n(Dual-Stream)', '#DDEEFF', bold=True)
    box(3.1, 6.4, 2.5, 1.0, 'Branch Pred.\n(Hybrid TAGE)', '#DDEEFF')
    box(5.9, 6.4, 3.7, 1.0, 'Decode / Rename\n(6-breit)', '#DDEEFF', bold=True)

    # Dispatch
    box(0.3, 4.9, 9.3, 0.9, 'Out-of-Order Dispatch (ROB, 256 Einträge)', '#D0F0D0', bold=True)

    # Execution units
    box(0.3, 3.2, 1.8, 1.3, 'ALU ×4\n(INT)', '#FFF0CC')
    box(2.3, 3.2, 1.8, 1.3, 'FPU/SIMD\n×2 (128b)', '#FFF0CC')
    box(4.3, 3.2, 1.8, 1.3, 'Load/Store\n×2 AGU', '#FFF0CC')
    box(6.3, 3.2, 1.5, 1.3, 'Branch\n×1', '#FFF0CC')
    box(8.0, 3.2, 1.6, 1.3, 'DSP/MAC\n×2', '#FFF0CC')

    # Memory hierarchy
    box(0.3, 1.5, 2.8, 1.2, 'L1-I$ 64 kB\nL1-D$ 64 kB', '#FFE0E0')
    box(3.3, 1.5, 3.0, 1.2, 'L2 Unified 1 MB\n(ECC, 8-way)', '#FFE0E0')
    box(6.5, 1.5, 3.0, 1.2, 'L3 Shared 8 MB\n(eDRAM, opt.)', '#FFE0E0')

    # TrustZone + Safety
    box(0.3, 0.1, 4.2, 0.9, 'TrustZone-X (S/NS-Partition, 4 Welten)', '#E8D0FF')
    box(4.7, 0.1, 4.9, 0.9, 'Safety-Island: Lock-Step, ECC, ISO 26262 ASIL-D', '#FFD0D0')

    # Arrows
    arrow(1.55, 6.4, 1.55, 5.8)
    arrow(4.35, 6.4, 4.35, 5.8)
    arrow(7.75, 6.4, 7.75, 5.8)
    for cx in [1.2, 3.2, 5.2, 7.05, 8.8]:
        arrow(cx, 4.9, cx, 4.5)
    for cx in [1.2, 3.2, 5.2, 7.05, 8.8]:
        arrow(cx, 3.2, cx, 2.7)

    plt.tight_layout()
    plt.savefig(f'{OUT}/plot5_hx_block.pdf')
    plt.savefig(f'{OUT}/plot5_hx_block.png')
    plt.close()
    print("Plot 5 done")

# ────────────────────────────────────────────────────────────
# PLOT 6: Skalierungsgesetz  DMIPS = f(N_core, Clock)
# ────────────────────────────────────────────────────────────
def plot6_scaling():
    clocks = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # GHz
    n_cores = [1, 2, 4, 8]
    ipc_hx1 = 5.20

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: DMIPS vs freq, per core count
    ax = axes[0]
    for n in n_cores:
        # Amdahl-Gesetz: S(n) = 1/(f_serial + (1-f_serial)/n), f_serial=0.05
        f_s = 0.05
        speedup = 1 / (f_s + (1 - f_s) / n)
        dmips = ipc_hx1 * clocks * 1000 * speedup  # DMIPS (approx: 1 GHz * IPC * 1000)
        ax.plot(clocks, dmips, marker='o', label=f'{n} Kern(e)')
    ax.set_xlabel('Taktfrequenz [GHz]')
    ax.set_ylabel('DMIPS (geschätzt)')
    ax.set_title('Abb. 6a: Cortex-HX Skalierung\n(Amdahl, f_serial=0.05)')
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.4)

    # Right: Efficiency = DMIPS / (n_cores * f)
    ax2 = axes[1]
    f_fixed = 2.0  # GHz
    ns = np.arange(1, 17)
    f_s = 0.05
    speedups = 1 / (f_s + (1 - f_s) / ns)
    total_dmips = ipc_hx1 * f_fixed * 1000 * speedups
    efficiency  = total_dmips / (ns * f_fixed * 1000 * ipc_hx1)
    ax2.plot(ns, efficiency * 100, 'o-', color=COLORS['H'])
    ax2.axhline(100, color=COLORS['grey'], ls='--', lw=1)
    ax2.set_xlabel('Anzahl Kerne N')
    ax2.set_ylabel('Parallelisierungseffizienz [%]')
    ax2.set_title('Abb. 6b: Parallelisierungseffizienz\n(Amdahl-Gesetz)')
    ax2.grid(linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{OUT}/plot6_scaling.pdf')
    plt.savefig(f'{OUT}/plot6_scaling.png')
    plt.close()
    print("Plot 6 done")

# ────────────────────────────────────────────────────────────
# PLOT 7: Domänenmatrix  (Eignung pro Anwendungsbereich)
# ────────────────────────────────────────────────────────────
def plot7_domain_matrix():
    domains = ['Automotive\nASIL-D', 'Cloud\nServer', 'IoT\nEdge', 'Industrie\nRobotik',
               'Mobile\nSmartphone', 'Wearable\n< 1 mW', 'HPC\nEmbedded']
    families = ['Cortex-A', 'Cortex-M', 'Cortex-R', 'Cortex-HX\n(neu)']

    # Eignungsmatrix 0..5
    matrix = np.array([
        # CortA  CortM  CortR  HX
        [1,      1,     5,     4],   # Automotive
        [5,      0,     1,     5],   # Cloud
        [2,      5,     2,     3],   # IoT
        [3,      3,     4,     5],   # Industrie
        [5,      1,     1,     4],   # Mobile
        [0,      5,     1,     2],   # Wearable
        [4,      2,     3,     5],   # HPC Embedded
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.YlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=5, aspect='auto')
    ax.set_xticks(range(len(families))); ax.set_xticklabels(families, fontsize=10)
    ax.set_yticks(range(len(domains)));  ax.set_yticklabels(domains,  fontsize=9.5)
    ax.set_title('Abb. 7: Domäneneignungs-Matrix (0 = ungeeignet, 5 = optimal)')

    for i in range(len(domains)):
        for j in range(len(families)):
            val = matrix[i, j]
            color = 'black' if val < 4 else 'white'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, label='Eignungsscore')
    plt.tight_layout()
    plt.savefig(f'{OUT}/plot7_domain_matrix.pdf')
    plt.savefig(f'{OUT}/plot7_domain_matrix.png')
    plt.close()
    print("Plot 7 done")

if __name__ == '__main__':
    plot1_perf_power()
    plot2_wcet()
    plot3_pipeline_ipc()
    plot4_aisa()
    plot5_hx_block()
    plot6_scaling()
    plot7_domain_matrix()
    print("All plots generated.")
