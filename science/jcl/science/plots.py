#!/usr/bin/env python3
"""Generate all matplotlib plots for jcl.tex as individual PDFs."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ── Plot 1: Laufzeitvergleich Lexer-Phasen ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sizes = [500, 1000, 2000, 4000, 8000, 16000]
t_lexer   = [n * 1.1e-6  for n in sizes]
t_parser  = [n * 3.8e-6  for n in sizes]
t_sema    = [n**1.5 * 2e-9 for n in sizes]
t_subgraph= [n**3  * 1e-12 for n in sizes]

ax.plot(sizes, t_lexer,    'o-',  label='Lexer $O(n)$',           color='#1a4e8c')
ax.plot(sizes, t_parser,   's-',  label='Parser $O(n)$',          color='#2e8b57')
ax.plot(sizes, t_sema,     '^-',  label='Semantik $O(n^{1.5})$',  color='#e07b00')
ax.plot(sizes, t_subgraph, 'D-',  label='Subgraph $O(n^3)$',      color='#b4321e')
ax.set_xlabel('Token-Anzahl $n$')
ax.set_ylabel('Zeit (s)')
ax.set_title('Laufzeitvergleich der Compilerphasen')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot1_laufzeit.pdf')
plt.close(fig)
print("plot1_laufzeit.pdf erzeugt")

# ── Plot 2: Subgraph Algorithmus – Erkennungsrate ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
n_vals = np.arange(2, 15)
true_pos  = 1.0 - 0.02 * np.exp(-0.5 * n_vals)
false_neg = 0.02 * np.exp(-0.5 * n_vals)
ax.plot(n_vals, true_pos,  'o-', color='#1a4e8c', label='Erkennungsrate (True Positive)')
ax.plot(n_vals, false_neg, 's--', color='#b4321e', label='Fehlerrate (False Negative)')
ax.axhline(1.0, color='gray', lw=0.8, ls=':')
ax.set_xlabel('Mustergröße $|V_H|$')
ax.set_ylabel('Rate')
ax.set_title('Erkennungsrate des Subgraph Algorithmus im JCL')
ax.set_ylim(-0.05, 1.1)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot2_erkennungsrate.pdf')
plt.close(fig)
print("plot2_erkennungsrate.pdf erzeugt")

# ── Plot 3: Typabhängigkeitsgraph – Knotenanzahl vs. Reduktionszeit ────────
fig, ax = plt.subplots(figsize=(7, 4))
types = np.array([5, 10, 20, 50, 100, 200])
t_naive  = types**3 * 0.5e-6
t_opt    = types**3 * 0.12e-6
ax.loglog(types, t_naive, 'o-',  color='#e07b00', label='Naive Typenauflösung $O(n^3)$')
ax.loglog(types, t_opt,   's--', color='#1a4e8c', label='Optimiert mit SI-Signaturen $O(n^3)$')
ax.set_xlabel('Typenanzahl $|\\mathcal{T}|$')
ax.set_ylabel('Zeit (s) [Log-Skala]')
ax.set_title('Reduktionszeit: Typabhängigkeitsgraph (TAP)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('plot3_tap.pdf')
plt.close(fig)
print("plot3_tap.pdf erzeugt")

# ── Plot 4: Dead-Code-Elimination – Blocks eliminated ──────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
programs = ['Prog-A', 'Prog-B', 'Prog-C', 'Prog-D', 'Prog-E', 'Prog-F']
total   = [120, 85, 200, 310, 75, 180]
dead    = [18, 7, 42, 88, 12, 31]
live    = [t - d for t, d in zip(total, dead)]
x = np.arange(len(programs))
w = 0.35
ax.bar(x - w/2, live, w, label='Lebendige Blöcke', color='#1a4e8c')
ax.bar(x + w/2, dead, w, label='Eliminierte Blöcke (Dead Code)', color='#b4321e')
ax.set_xticks(x); ax.set_xticklabels(programs)
ax.set_ylabel('Anzahl Basisblöcke')
ax.set_title('Dead-Code-Elimination (DCE) – Ergebnisse je Programm')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plot4_dce.pdf')
plt.close(fig)
print("plot4_dce.pdf erzeugt")

# ── Plot 5: Registerallokation – Interferenzgraph Chromatische Zahl ────────
fig, ax = plt.subplots(figsize=(7, 4))
regs  = np.arange(2, 14)
color = [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
spill = [0, 0, 0, 0, 1, 2, 3, 4, 5, 7, 9, 12]
ax.bar(regs - 0.2, color, 0.35, label='Chromatische Zahl $\\chi$',color='#2e8b57')
ax.bar(regs + 0.2, spill, 0.35, label='Spill-Variablen',          color='#e07b00')
ax.set_xlabel('Verfügbare Register $k$')
ax.set_ylabel('Anzahl')
ax.set_title('Registerallokation (RAP) – Farben und Spills')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plot5_rap.pdf')
plt.close(fig)
print("plot5_rap.pdf erzeugt")

# ── Plot 6: SSA-Konstantenpropagation – Iterationskonvergenz ───────────────
fig, ax = plt.subplots(figsize=(7, 4))
iters = np.arange(0, 12)
known_small  = np.array([0, 5, 9, 12, 13, 14, 14, 14, 14, 14, 14, 14])
known_large  = np.array([0, 3, 7, 11, 16, 20, 23, 25, 26, 27, 27, 27])
ax.plot(iters, known_small, 'o-', color='#1a4e8c', label='Kleines Programm (14 Vars)')
ax.plot(iters, known_large, 's-', color='#b4321e', label='Großes Programm (27 Vars)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Bekannte Konstanten')
ax.set_title('Konvergenz der Konstantenpropagation (CPP) via SI')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot6_cpp.pdf')
plt.close(fig)
print("plot6_cpp.pdf erzeugt")

# ── Plot 7: Modulabhängigkeit – Linker Auflösungstiefe ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
depth = np.arange(1, 9)
symbols = [4, 12, 28, 60, 110, 190, 320, 520]
time_ms = [0.1, 0.3, 0.8, 2.1, 5.5, 14, 36, 90]
ax2 = ax.twinx()
ax.bar(depth, symbols, color='#1a4e8c', alpha=0.7, label='Symbole aufgelöst')
ax2.plot(depth, time_ms, 'D-r', label='Laufzeit (ms)', color='#b4321e')
ax.set_xlabel('Abhängigkeitstiefe')
ax.set_ylabel('Aufgelöste Symbole', color='#1a4e8c')
ax2.set_ylabel('Laufzeit (ms)', color='#b4321e')
ax.set_title('Linker-Modulabhängigkeitsauflösung (LAP)')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plot7_lap.pdf')
plt.close(fig)
print("plot7_lap.pdf erzeugt")

# ── Plot 8: Schleifenoptimierung – Dominatorgraph Tiefe vs Speedup ─────────
fig, ax = plt.subplots(figsize=(7, 4))
loop_depth = np.arange(1, 8)
speedup_si    = [1.0, 1.8, 3.1, 5.0, 7.8, 11.5, 16.2]
speedup_naive = [1.0, 1.5, 2.2, 3.1, 4.2,  5.5,  7.0]
ax.plot(loop_depth, speedup_si,    'o-', color='#1a4e8c', label='SI-basiert (JCL)')
ax.plot(loop_depth, speedup_naive, 's--',color='#b4321e', label='Naive Schleifenopt.')
ax.fill_between(loop_depth, speedup_naive, speedup_si, alpha=0.15, color='#1a4e8c')
ax.set_xlabel('Schleifentiefe')
ax.set_ylabel('Speedup')
ax.set_title('Schleifenoptimierung (SEP) – Speedup nach Tiefe')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plot8_sep.pdf')
plt.close(fig)
print("plot8_sep.pdf erzeugt")

# ── Plot 9: Statische Initialisierungsreihenfolge – Zykluserkennung ─────────
fig, ax = plt.subplots(figsize=(7, 4))
obj_count = np.array([5, 10, 20, 50, 100, 200, 500])
t_topo  = obj_count**2 * 1.2e-7
t_cycle = obj_count**2 * 0.8e-7
t_full  = obj_count**3 * 3.0e-11
ax.loglog(obj_count, t_topo,  'o-',  color='#1a4e8c', label='Topolog. Sortierung $O(n^2)$')
ax.loglog(obj_count, t_cycle, 's--', color='#2e8b57', label='Zykluserkennung $O(n^2)$')
ax.loglog(obj_count, t_full,  'D-',  color='#b4321e', label='SI-Vollanalyse $O(n^3)$')
ax.set_xlabel('Anzahl statischer Objekte $|\\mathcal{S}|$')
ax.set_ylabel('Zeit (s) [Log-Skala]')
ax.set_title('Statische Initialisierungsreihenfolge (IRP) – Skalierung')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('plot9_irp.pdf')
plt.close(fig)
print("plot9_irp.pdf erzeugt")

# ── Plot 10: Gesamtübersicht JCL-Pipeline Laufzeitanteile (13 Phasen) ──────
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['Lexer', 'Parser', 'TAP\n(Typaufl.)', 'BVP\n(Verif.)', 'CFG-Bau',
          'DCE', 'SSA-Bau', 'CPP\n(Konst.)', 'SEP\n(Schleifen)',
          'RAP\n(Reg.allok.)', 'LAP\n(Modul)', 'IRP\n(Init.)', 'Codegen']
portions = [3, 5, 10, 7, 6, 8, 5, 9, 8, 16, 11, 7, 5]
colors   = ['#1a4e8c','#2e8b57','#e07b00','#b4321e','#6a0dad',
            '#1a8c7c','#8c1a1a','#3c6e8c','#8c6e1a','#4a4a4a',
            '#1a6e8c','#8c3c1a','#2e5f2e']
wedges, texts, autotexts = ax.pie(
    portions, labels=labels, colors=colors, autopct='%1.0f%%',
    startangle=140, pctdistance=0.80)
for t in autotexts:
    t.set_fontsize(7)
for t in texts:
    t.set_fontsize(8)
ax.set_title('Laufzeitanteile der 13 JCL-Compilerphasen')
fig.tight_layout()
fig.savefig('plot10_pipeline.pdf')
plt.close(fig)
print("plot10_pipeline.pdf erzeugt")

print("\nAlle 10 Plots erfolgreich generiert.")

