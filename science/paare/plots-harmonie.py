#!/usr/bin/env python3
"""
Matplotlib-Plots fuer: Paare als Harmoniestruktur der Natur
Stephan Epp, Universitaet Bielefeld, 2026
6 Plots, einzeln als PDF gespeichert.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.integrate import odeint

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})

MAINBLUE  = '#19468C'
ACCENTRED = '#B4321E'
DARKGREEN = '#1E6432'
ORANGE    = '#C06000'
PURPLE    = '#5A0A8A'
GRAY      = '#707080'
OUT = '/home/claude/'

# ============================================================
# PLOT 1: Harmoniegraph vs. Disharmoniegraph
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Harmoniegraph
G1 = nx.Graph()
paare = [('Mann','Frau'), ('A','T'), ('G','C'),
         ('+q','-q'), ('N-Pol','S-Pol'), ('Biene','Bluete')]
for a, b in paare:
    G1.add_edge(a, b)

ax = axes[0]
pos1 = nx.spring_layout(G1, seed=42, k=1.8)
nx.draw_networkx_nodes(G1, pos1, ax=ax, node_color=MAINBLUE, node_size=750, alpha=0.93)
nx.draw_networkx_labels(G1, pos1, ax=ax, font_color='white', font_size=7.5, font_weight='bold')
nx.draw_networkx_edges(G1, pos1, ax=ax, edge_color=DARKGREEN, width=2.8, alpha=0.85)
ax.set_title('(a) Harmoniegraph $G_H$: alle Knoten gepaart', fontsize=11)
ax.axis('off')
H_a = 2 * G1.number_of_edges() / G1.number_of_nodes()
ax.text(0.5, -0.05, f'Harmoniemass $H(G_H) = {H_a:.2f}$ (maximal)',
        transform=ax.transAxes, ha='center', fontsize=10, color=DARKGREEN, fontweight='bold')

# (b) Disharmoniegraph
G2 = nx.Graph()
G2.add_edges_from([('Mann','Frau'), ('A','T'), ('+q','-q')])
G2.add_nodes_from(['G*', 'N-Pol', 'X', 'Y'])

ax = axes[1]
pos2 = nx.spring_layout(G2, seed=99, k=1.5)
paired_n   = [n for n in G2.nodes() if G2.degree(n) > 0]
isolated_n = [n for n in G2.nodes() if G2.degree(n) == 0]
nx.draw_networkx_nodes(G2, pos2, ax=ax, nodelist=paired_n,   node_color=MAINBLUE,  node_size=750, alpha=0.93)
nx.draw_networkx_nodes(G2, pos2, ax=ax, nodelist=isolated_n, node_color=ACCENTRED, node_size=750, alpha=0.93)
nx.draw_networkx_labels(G2, pos2, ax=ax, font_color='white', font_size=7.5, font_weight='bold')
nx.draw_networkx_edges(G2, pos2, ax=ax, edge_color=DARKGREEN, width=2.8, alpha=0.85)
ax.set_title('(b) Disharmoniegraph $G_D$: isolierte Knoten (rot)', fontsize=11)
ax.axis('off')
H_b = 2 * G2.number_of_edges() / G2.number_of_nodes()
ax.text(0.5, -0.05, f'Harmoniemass $H(G_D) = {H_b:.2f} < 1$ (nicht maximal)',
        transform=ax.transAxes, ha='center', fontsize=10, color=ACCENTRED, fontweight='bold')

ph_b = mpatches.Patch(color=MAINBLUE,  label='Gepaarter Knoten')
ph_r = mpatches.Patch(color=ACCENTRED, label='Isolierter (ungepaarter) Knoten')
ph_g = mpatches.Patch(color=DARKGREEN, label='Harmoniekante (Paarverbindung)')
fig.legend(handles=[ph_b, ph_r, ph_g], loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, 0.01))
fig.suptitle('Abbildung 1: Harmoniegraph und Disharmoniegraph natuerlicher Paare',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.09, 1, 0.96])
plt.savefig(OUT + 'plot1_harmonie_graph.pdf', bbox_inches='tight')
plt.close()
print("Plot 1 gespeichert.")

# ============================================================
# PLOT 2: DNA-Basenpaarung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Bindungsenergien
ax = axes[0]
paare_dna = ['A-T\n(2 H-Br.)', 'T-A\n(2 H-Br.)', 'G-C\n(3 H-Br.)', 'C-G\n(3 H-Br.)']
# Approximate Gibbs free energy contribution per H-bond ~-2.9 kJ/mol in DNA context
energien_kJ = [5.8, 5.8, 8.7, 8.7]
farben = [MAINBLUE, MAINBLUE, ACCENTRED, ACCENTRED]
bars = ax.bar(paare_dna, energien_kJ, color=farben, edgecolor='white', linewidth=1.2, alpha=0.88)
ax.set_ylabel('Bindungsenergie (kJ/mol, relativ)')
ax.set_title('(a) Watson-Crick-Basenpaar-Bindungsstaerke')
ax.set_ylim(0, 12)
ax.axhline(7.25, color=GRAY, linestyle='--', alpha=0.5, label='Mittlere Bindungsstaerke')
for bar, val in zip(bars, energien_kJ):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ph_at = mpatches.Patch(color=MAINBLUE,  label='A-T Paare (2 H-Bruecken)')
ph_gc = mpatches.Patch(color=ACCENTRED, label='G-C Paare (3 H-Bruecken)')
ax.legend(handles=[ph_at, ph_gc], fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (b) Komplementaritaetsfunktion phi
ax = axes[1]
basen = ['A', 'T', 'G', 'C']
kompl = ['T', 'A', 'C', 'G']
x_pos = np.arange(4)
ax.bar(x_pos - 0.2, [1]*4, 0.35, color=MAINBLUE,  alpha=0.85, label='Basis $b$')
ax.bar(x_pos + 0.2, [1]*4, 0.35, color=ACCENTRED, alpha=0.85, label='Komplement $\\phi(b)$')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'$\\phi({b})={k}$' for b, k in zip(basen, kompl)], fontsize=11)
ax.set_yticks([])
ax.set_ylim(0, 1.6)
ax.set_title('(b) Watson-Crick-Komplementaritaet $\\phi$ (Involution)')
for i in range(4):
    ax.annotate('', xy=(i+0.2, 0.55), xytext=(i-0.2, 0.55),
                arrowprops=dict(arrowstyle='->', color=DARKGREEN, lw=2.0))
ax.text(1.5, 1.3, '$\\phi \\circ \\phi = \\mathrm{id}$', ha='center',
        fontsize=13, color=DARKGREEN, fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Abbildung 2: DNA-Basenpaarung und Watson-Crick-Komplementaritaet',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot2_dna_paare.pdf', bbox_inches='tight')
plt.close()
print("Plot 2 gespeichert.")

# ============================================================
# PLOT 3: Reproduktionsdynamik
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Reproduktionsrate vs Geschlechterverhaeltnis
ax = axes[0]
N  = 100
p  = np.linspace(0, 1, 400)
R  = np.minimum(p * N, (1 - p) * N)
R_norm = R / (N / 2)
ax.plot(p, R_norm, color=MAINBLUE, lw=3.0, label='$\\hat{R}(p) = \\min(pN,(1-p)N)\\,/\\,\\frac{N}{2}$')
ax.axvline(0.5, color=DARKGREEN, linestyle='--', lw=1.8, label='Optimum $p^* = 0{,}5$')
ax.axhline(1.0, color=GRAY, linestyle=':', alpha=0.5)
ax.scatter([0.5], [1.0], color=ACCENTRED, s=90, zorder=6, label='Maximum $\\hat{R}(0{,}5)=1$')
ax.axvspan(0.4, 0.6, alpha=0.10, color=DARKGREEN, label='Harmoniebereich')
ax.set_xlabel('Maennchen-Anteil $p = m/N$')
ax.set_ylabel('Normierte Reproduktionsrate $\\hat{R}(p)$')
ax.set_title('(a) Reproduktionsoptimum bei Paarungsharmonie ($p^*=0{,}5$)')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)

# (b) Populationsdynamik: sexuell mit Allee-Effekt vs. asexuell
ax = axes[1]
K = 500; r = 0.35; A = 60

def sex_ode(y, t):
    N = max(y[0], 0)
    return r * N * (N - A) / K * (1 - N / K)

def asex_ode(y, t):
    return r * y[0] * (1 - y[0] / K)

t_vec = np.linspace(0, 40, 800)
sol_sex_hi = odeint(sex_ode,  [200.0], t_vec)[:,0]
sol_sex_lo = odeint(sex_ode,  [30.0],  t_vec)[:,0]
sol_asex   = odeint(asex_ode, [30.0],  t_vec)[:,0]

ax.plot(t_vec, sol_sex_hi, color=MAINBLUE,  lw=2.5, label='Sexuell, gepaart ($N_0>A$): Wachstum')
ax.plot(t_vec, sol_sex_lo, color=ACCENTRED, lw=2.5, linestyle='--',
        label='Sexuell, gepaart ($N_0<A$): Aussterben')
ax.plot(t_vec, sol_asex,   color=DARKGREEN, lw=2.5, linestyle=':',
        label='Asexuell, ungepaart: stabiles Wachstum')
ax.axhline(A, color=GRAY, linestyle=':', alpha=0.6, label=f'Allee-Schwelle $A={A}$')
ax.set_xlabel('Zeit $t$')
ax.set_ylabel('Populationsgroesse $N(t)$')
ax.set_title('(b) Allee-Effekt: Paarung bedingt kritische Populationsdichte')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)
ax.set_ylim(-30, K + 60)

fig.suptitle('Abbildung 3: Reproduktionsdynamik und der Allee-Effekt bei sexueller Paarung',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot3_reproduktion.pdf', bbox_inches='tight')
plt.close()
print("Plot 3 gespeichert.")

# ============================================================
# PLOT 4: Coulomb-Potential
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

r_m   = np.linspace(0.25e-10, 6e-10, 600)
k_e   = 8.9875e9
q_e   = 1.6e-19
eV_J  = 1.6e-19
r_A   = r_m * 1e10  # Angstrom

V_att = k_e * (-q_e) * q_e / r_m / eV_J
V_rep = k_e * q_e * q_e    / r_m / eV_J

# (a) Potentialkurven
ax = axes[0]
ax.plot(r_A, V_att, color=MAINBLUE,  lw=2.8, label='Anziehung $+q/-q$ (Harmonisches Paar)')
ax.plot(r_A, V_rep, color=ACCENTRED, lw=2.8, label='Abstossung $+q/+q$ (kein Paar)')
ax.axhline(0, color=GRAY, lw=1)
ax.fill_between(r_A, V_att, 0, where=(V_att < 0),
                alpha=0.13, color=MAINBLUE, label='Bindungsbereich (Harmonie)')
ax.set_xlim(0.2, 6); ax.set_ylim(-60, 60)
ax.set_xlabel('Abstand $r$ (\\AA)')
ax.set_ylabel('Potential $V(r)$ (eV)')
ax.set_title('(a) Coulomb-Potential: Ladungspaare')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.text(4.2, -42, 'Harmoniebereich\n$V < 0$\n(gebundener Zustand)',
        ha='center', color=MAINBLUE, fontsize=8.5)

# (b) Visualisierung der Lade-Paare
ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('(b) Ladungspaare: Harmonie vs. Disharmonie')

# Oben: gleiche Ladungen (Repulsion)
c1 = plt.Circle((0.27, 0.76), 0.09, color=ACCENTRED, zorder=3)
c2 = plt.Circle((0.73, 0.76), 0.09, color=ACCENTRED, zorder=3)
ax.add_patch(c1); ax.add_patch(c2)
ax.text(0.27, 0.76, '+', ha='center', va='center', fontsize=18, color='white', fontweight='bold', zorder=4)
ax.text(0.73, 0.76, '+', ha='center', va='center', fontsize=18, color='white', fontweight='bold', zorder=4)
ax.annotate('', xy=(0.60, 0.76), xytext=(0.40, 0.76),
            arrowprops=dict(arrowstyle='<->', color=ACCENTRED, lw=2.2))
ax.text(0.50, 0.90, 'Gleiche Ladungen: Abstossung', ha='center',
        fontsize=10, color=ACCENTRED, fontweight='bold')
ax.text(0.50, 0.63, r'$H(G) = 0$ (kein Paar)', ha='center', fontsize=9, color=ACCENTRED)

# Unten: entgegengesetzte Ladungen (Attraktion)
c3 = plt.Circle((0.27, 0.30), 0.09, color=ACCENTRED, zorder=3)
c4 = plt.Circle((0.73, 0.30), 0.09, color=MAINBLUE,  zorder=3)
ax.add_patch(c3); ax.add_patch(c4)
ax.text(0.27, 0.30, '+', ha='center', va='center', fontsize=18, color='white', fontweight='bold', zorder=4)
ax.text(0.73, 0.30, u'\u2212', ha='center', va='center', fontsize=18, color='white', fontweight='bold', zorder=4)
ax.annotate('', xy=(0.60, 0.30), xytext=(0.40, 0.30),
            arrowprops=dict(arrowstyle='->', color=MAINBLUE, lw=2.5))
ax.text(0.50, 0.44, 'Entgegengesetzte Ladungen: Anziehung', ha='center',
        fontsize=10, color=MAINBLUE, fontweight='bold')
ax.text(0.50, 0.17, r'$H(G) = 1$ (stabiles Paar)', ha='center', fontsize=9, color=MAINBLUE)
ax.axhline(0.52, xmin=0.05, xmax=0.95, color=GRAY, lw=0.8, linestyle='--', alpha=0.5)

fig.suptitle('Abbildung 4: Coulomb-Potential - Elektrische Ladungspaare als physikalische Harmonie',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot4_coulomb.pdf', bbox_inches='tight')
plt.close()
print("Plot 4 gespeichert.")

# ============================================================
# PLOT 5: Lotka-Volterra
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def lv(y, t, alpha, beta, gamma, delta):
    x, z = y
    return [alpha*x - beta*x*z, delta*x*z - gamma*z]

alpha, beta, gamma, delta = 1.0, 0.10, 0.75, 0.075
t_lv = np.linspace(0, 80, 4000)
y0 = [10.0, 5.0]
sol = odeint(lv, y0, t_lv, args=(alpha, beta, gamma, delta))
xv, zv = sol[:,0], sol[:,1]
x_star = gamma / delta
z_star = alpha / beta

# (a) Zeitreihe
ax = axes[0]
ax.plot(t_lv, xv, color=MAINBLUE,  lw=2.0, label='Beute $x(t)$ (z.B. Hase)')
ax.plot(t_lv, zv, color=ACCENTRED, lw=2.0, label='Raeuber $z(t)$ (z.B. Fuchs)')
ax.axhline(x_star, color=MAINBLUE,  linestyle=':', alpha=0.5, lw=1.2,
           label=f'$x^* = \\gamma/\\delta = {x_star:.0f}$')
ax.axhline(z_star, color=ACCENTRED, linestyle=':', alpha=0.5, lw=1.2,
           label=f'$z^* = \\alpha/\\beta = {z_star:.0f}$')
ax.set_xlabel('Zeit $t$')
ax.set_ylabel('Populationsgroesse')
ax.set_title('(a) Lotka-Volterra-Zeitreihe: stabiles Paargleichgewicht')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) Phasenportraet
ax = axes[1]
ax.plot(xv, zv, color=PURPLE, lw=1.6, alpha=0.9, label='Trajektorie')
ax.scatter([x_star], [z_star], color=DARKGREEN, s=110, zorder=5,
           label=f'GGW $({x_star:.0f},\\,{z_star:.0f})$')
# Multiple orbits
for n0 in [(6,3), (14,8), (20,12)]:
    s2 = odeint(lv, list(n0), t_lv, args=(alpha, beta, gamma, delta))
    ax.plot(s2[:,0], s2[:,1], lw=1.0, alpha=0.40, color=PURPLE)
ax.set_xlabel('Beutepopulation $x$')
ax.set_ylabel('Raeuber population $z$')
ax.set_title('(b) Phasenportraet: geschlossene Orbits um stabiles Gleichgewicht')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.annotate('Stabiles\nGleichgewicht\n$(x^*, z^*)$',
            xy=(x_star, z_star), xytext=(x_star+7, z_star+4),
            arrowprops=dict(arrowstyle='->', color=DARKGREEN, lw=1.8),
            fontsize=9, color=DARKGREEN)

fig.suptitle('Abbildung 5: Lotka-Volterra-Modell - Raeuber-Beute-Paar als oekologische Harmonie',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'plot5_lotka_volterra.pdf', bbox_inches='tight')
plt.close()
print("Plot 5 gespeichert.")

# ============================================================
# PLOT 6: Harmoniemass-Vergleich
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

kategorien = [
    'Mann-Frau\n(Reproduktion)',
    'DNA\n(A-T, G-C)',
    'Elektr.\n(+/-)',
    'Magnetpole\n(N/S)',
    'Saeure-\nBase',
    'Raeuber-\nBeute',
]
H_pair   = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
H_unpair = [0.40, 0.25, 0.50, 0.30, 0.60, 0.45]
x = np.arange(len(kategorien))

# (a) Balkendiagramm
ax = axes[0]
bars1 = ax.bar(x - 0.20, H_pair,   0.35, color=MAINBLUE,  alpha=0.88, label='Gepaart $H=1$')
bars2 = ax.bar(x + 0.20, H_unpair, 0.35, color=ACCENTRED, alpha=0.88, label='Ungepaart $H<1$')
ax.set_xticks(x); ax.set_xticklabels(kategorien, fontsize=8.5)
ax.set_ylabel('Harmoniemass $H(G)$')
ax.set_title('(a) Harmoniemass: gepaart vs. ungepaart')
ax.axhline(1.0, color=DARKGREEN, linestyle='--', lw=1.5, alpha=0.7, label='Maximale Harmonie $H=1$')
ax.set_ylim(0, 1.30)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, '1.00',
            ha='center', fontsize=8, color=MAINBLUE, fontweight='bold')

# (b) Radar-Diagramm
N_cat  = len(kategorien)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

# Use polar inset
ax2 = axes[1]
ax2.axis('off')
inset = fig.add_axes([0.565, 0.11, 0.40, 0.78], polar=True)

H_pair_r   = H_pair   + H_pair[:1]
H_unpair_r = H_unpair + H_unpair[:1]

inset.plot(angles, H_pair_r,   'o-', color=MAINBLUE,  lw=2.4, label='Gepaart')
inset.fill(angles, H_pair_r,   alpha=0.14, color=MAINBLUE)
inset.plot(angles, H_unpair_r, 's--', color=ACCENTRED, lw=2.4, label='Ungepaart')
inset.fill(angles, H_unpair_r, alpha=0.14, color=ACCENTRED)

inset.set_xticks(angles[:-1])
inset.set_xticklabels(['M/F', 'DNA', 'Elekt.', 'Magn.', 'S/B', 'R/B'], fontsize=9.5)
inset.set_ylim(0, 1.25)
inset.set_yticks([0.25, 0.5, 0.75, 1.0])
inset.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=8)
inset.set_title('(b) Radar: $H$ ueber alle\nnaturlichen Paartypen',
                fontsize=10, fontweight='bold', pad=12)
inset.legend(loc='upper right', bbox_to_anchor=(1.45, 1.12), fontsize=9)
inset.grid(True, alpha=0.4)

fig.suptitle('Abbildung 6: Harmoniemass-Vergleich uber alle naturlichen Paarungstypen',
             fontsize=12, fontweight='bold')
plt.savefig(OUT + 'plot6_harmoniemass.pdf', bbox_inches='tight')
plt.close()
print("Plot 6 gespeichert.")

print("\nAlle 6 Plots erfolgreich als PDF gespeichert.")
