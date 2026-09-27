#!/usr/bin/env python3
"""Alle matplotlib-Plots für das Maschinenbau-Buch"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Arc, Circle
import matplotlib.patches as mpatches

# ============================================================
# Plot 1: Statik -- Kräftezerlegung und Gleichgewicht
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
F_mag = 5.0
alpha = np.radians(37)
F = np.array([F_mag*np.cos(alpha), F_mag*np.sin(alpha)])
Fx = np.array([F[0], 0])
Fy = np.array([0, F[1]])

ax.annotate('', xy=F, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax.annotate('', xy=Fx, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=Fy, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.plot([Fx[0], F[0]], [0, F[1]], 'b--', lw=1, alpha=0.6)
ax.plot([0, F[0]], [Fy[1], F[1]], 'r--', lw=1, alpha=0.6)

theta = np.linspace(0, alpha, 60)
ax.plot(0.8*np.cos(theta), 0.8*np.sin(theta), 'k-', lw=1)
ax.text(0.95, 0.18, r'$\alpha$', fontsize=12)
ax.text(F[0]/2 - 0.3, -0.35, r'$F_x = F\cos\alpha$', color='blue', fontsize=11)
ax.text(-1.4, F[1]/2, r'$F_y = F\sin\alpha$', color='red', fontsize=11)
ax.text(F[0]+0.1, F[1]+0.1, r'$\vec{F}$', fontsize=13, fontweight='bold')
ax.set_xlim(-0.5, 6); ax.set_ylim(-0.8, 5)
ax.set_aspect('equal'); ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel(r'$x$', fontsize=12); ax.set_ylabel(r'$y$', fontsize=12)
ax.set_title('Kraftzerlegung in Komponenten', fontsize=12)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Drei Kräfte im Gleichgewicht
F1 = np.array([3.0, 0.0])
F2 = np.array([-1.5, 2.598])   # 3 N bei 120°
F3 = -F1 - F2
for Fi, col, lbl in [(F1,'blue',r'$\vec{F}_1$'),(F2,'red',r'$\vec{F}_2$'),(F3,'green',r'$\vec{F}_3$')]:
    ax2.annotate('', xy=Fi, xytext=(0,0),
                 arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    ax2.text(Fi[0]*1.12+0.05, Fi[1]*1.12, lbl, color=col, fontsize=12)
ax2.plot(0, 0, 'ko', ms=8, zorder=5)
ax2.set_xlim(-4, 4.5); ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.axhline(0, color='k', lw=0.5); ax2.axvline(0, color='k', lw=0.5)
ax2.set_xlabel(r'$x$ [N]', fontsize=12); ax2.set_ylabel(r'$y$ [N]', fontsize=12)
ax2.set_title(r'Kräftegleichgewicht: $\sum \vec{F}_i = \vec{0}$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.text(0.2, 0.2, 'Knoten', fontsize=10)

plt.tight_layout()
plt.savefig('plot_statik.pdf', bbox_inches='tight')
plt.close()
print("plot_statik.pdf OK")

# ============================================================
# Plot 2: Kinematik -- s(t), v(t), a(t)
# ============================================================
t = np.linspace(0, 4, 400)
a0 = 2.0
s = 0.5 * a0 * t**2
v = a0 * t
a = np.full_like(t, a0)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
axes[0].plot(t, s, 'b-', lw=2, label=r'$s(t)=\frac{1}{2}a_0 t^2$')
axes[0].fill_between(t, s, alpha=0.1, color='blue')
axes[0].set_ylabel(r'Weg $s$ [m]', fontsize=12)
axes[0].set_title(r'Gleichmäßig beschleunigte Bewegung ($a_0 = 2\,\mathrm{m/s^2}$)', fontsize=12)
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

axes[1].plot(t, v, 'r-', lw=2, label=r'$v(t)=a_0 t$')
axes[1].set_ylabel(r'Geschwindigkeit $v$ [m/s]', fontsize=12)
axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)

axes[2].plot(t, a, 'g-', lw=2, label=r'$a(t)=a_0=\mathrm{const}$')
axes[2].set_ylabel(r'Beschleunigung $a$ [m/s$^2$]', fontsize=12)
axes[2].set_xlabel(r'Zeit $t$ [s]', fontsize=12)
axes[2].set_ylim(0, 3); axes[2].legend(fontsize=11); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_kinematik.pdf', bbox_inches='tight')
plt.close()
print("plot_kinematik.pdf OK")

# ============================================================
# Plot 3: Festigkeitslehre -- Spannungs-Dehnungs-Diagramm
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
# Stahl S235 typisch
eps = np.linspace(0, 0.25, 1000)
# elastischer Bereich bis eps_e=0.001, dann Fließen, Verfestigung, Bruch
sig = np.zeros_like(eps)
E = 210000  # MPa
Re = 235    # MPa
Rm = 360    # MPa
eps_e = Re/E
eps_y = 0.02
eps_max = 0.22

for i, e in enumerate(eps):
    if e <= eps_e:
        sig[i] = E * e
    elif e <= eps_y:
        sig[i] = Re + (Re*0.02)*(e - eps_e)/(eps_y - eps_e)
    elif e <= eps_max:
        # Verfestigung
        sig[i] = Re + (Rm - Re) * ((e - eps_y)/(eps_max - eps_y))**0.3
    else:
        sig[i] = Rm * max(0, 1 - 5*(e - eps_max))

ax.plot(eps*100, sig, 'b-', lw=2.5, label='Baustahl S235')
ax.axvline(eps_e*100, color='gray', lw=1, ls='--')
ax.axhline(Re, color='red', lw=1, ls='--', label=f'$R_e = {Re}$ MPa')
ax.axhline(Rm, color='green', lw=1, ls='--', label=f'$R_m = {Rm}$ MPa')
ax.text(eps_e*100+0.05, 10, r'$\varepsilon_e$', fontsize=11, color='gray')
ax.text(0.01, Re+5, r'Streckgrenze $R_e$', fontsize=10, color='red')
ax.text(0.01, Rm+5, r'Zugfestigkeit $R_m$', fontsize=10, color='green')
ax.annotate('', xy=(0.05, Re/2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(0.05, Re/2 - 15, r'Elastisch: $\sigma = E\,\varepsilon$', color='blue', fontsize=10)

ax.set_xlabel(r'Dehnung $\varepsilon$ [%]', fontsize=12)
ax.set_ylabel(r'Spannung $\sigma$ [MPa]', fontsize=12)
ax.set_title('Spannungs-Dehnungs-Diagramm (Zugversuch, Stahl S235)', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
ax.set_xlim(-0.002*100, 0.26*100); ax.set_ylim(-5, 420)

plt.tight_layout()
plt.savefig('plot_festigkeit.pdf', bbox_inches='tight')
plt.close()
print("plot_festigkeit.pdf OK")

# ============================================================
# Plot 4: Biegelinie eines einseitig eingespannten Balkens
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

L = 1.0
E_mod = 210e9   # Pa
I = 1e-6        # m^4  (z.B. HEA 100)
q = 5000        # N/m  gleichstreckenlast

x = np.linspace(0, L, 300)
# Biegelinie: w(x) = q/(24EI) * (x^4 - 4Lx^3 + 6L^2 x^2)  -- keine, das ist beidseitig aufgelegt
# Kragarm mit Gleichstreckenlast: w(x) = q/(24EI)*(x^4 - 4Lx^3 + 6L^2 x^2)  nein
# Kragarm, Einspannung bei x=0, freies Ende x=L, Gleichstreckenlast:
# w(x) = q/(24EI) * (6L^2 x^2 - 4L x^3 + x^4)
w = (q / (24*E_mod*I)) * (6*L**2*x**2 - 4*L*x**3 + x**4)
w_mm = w * 1000

ax = axes[0]
ax.plot(x*100, -w_mm, 'b-', lw=2.5, label='Biegelinie $w(x)$')
ax.fill_between(x*100, -w_mm, alpha=0.1, color='blue')
ax.axhline(0, color='k', lw=1)
ax.set_xlabel(r'Position $x$ [cm]', fontsize=12)
ax.set_ylabel(r'Durchbiegung $w$ [mm]', fontsize=12)
ax.set_title('Kragarm mit Gleichstreckenlast $q$', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
w_max = (q*L**4)/(8*E_mod*I)*1000
ax.text(80, -w_max*0.5, f'$w_{{max}}={w_max:.2f}$ mm', fontsize=10)

# Biegemoment
ax2 = axes[1]
# M(x) = q/2 * (L-x)^2  (am freien Ende = 0, an Einspannung = qL^2/2)
M = q/2 * (L - x)**2
ax2.plot(x*100, M, 'r-', lw=2.5, label='Biegemoment $M(x)$')
ax2.fill_between(x*100, M, alpha=0.1, color='red')
ax2.set_xlabel(r'Position $x$ [cm]', fontsize=12)
ax2.set_ylabel(r'Biegemoment $M$ [Nm]', fontsize=12)
ax2.set_title(r'Biegemomentenverlauf $M(x) = \frac{q}{2}(L-x)^2$', fontsize=12)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_biegung.pdf', bbox_inches='tight')
plt.close()
print("plot_biegung.pdf OK")

# ============================================================
# Plot 5: Thermodynamik -- Kreisprozesse (p-V-Diagramm)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# Carnot-Prozess: 1->2 isotherme Expansion, 2->3 Adiabate, 3->4 isotherme Komp, 4->1 Adiabate
T_H = 600  # K
T_C = 300  # K
R = 8.314
n = 1  # mol
V1, V2, V3, V4 = 0.01, 0.02, 0.04, 0.02

# p = nRT/V
V12 = np.linspace(V1, V2, 100)
p12 = n*R*T_H / V12

kappa = 1.4
p2 = n*R*T_H/V2
p3 = n*R*T_C/V3
V23 = np.linspace(V2, V3, 100)
p23 = p2*(V2/V23)**kappa

V34 = np.linspace(V3, V4, 100)
p34 = n*R*T_C / V34

p4 = n*R*T_C/V4
V41 = np.linspace(V4, V1, 100)
p41 = p4*(V4/V41)**kappa

ax.plot(V12*1000, p12/1000, 'r-', lw=2.5, label='1→2: Isotherm ($T_H$)')
ax.plot(V23*1000, p23/1000, 'b-', lw=2.5, label='2→3: Adiabat')
ax.plot(V34*1000, p34/1000, 'c-', lw=2.5, label='3→4: Isotherm ($T_C$)')
ax.plot(V41*1000, p41/1000, 'm-', lw=2.5, label='4→1: Adiabat')

for Vi, pi, lbl in [(V1,n*R*T_H/V1,'1'),(V2,p2,'2'),(V3,p3,'3'),(V4,p4,'4')]:
    ax.plot(Vi*1000, pi/1000, 'ko', ms=7, zorder=5)
    ax.text(Vi*1000+0.1, pi/1000+0.5, lbl, fontsize=12, fontweight='bold')

ax.set_xlabel(r'Volumen $V$ [l]', fontsize=12)
ax.set_ylabel(r'Druck $p$ [kPa]', fontsize=12)
ax.set_title('Carnot-Kreisprozess im $p$-$V$-Diagramm', fontsize=12)
ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)

ax2 = axes[1]
eta_carnot = lambda T_C, T_H: 1 - T_C/T_H
T_H_arr = np.linspace(400, 1200, 200)
for TC in [300, 400, 500]:
    eta = eta_carnot(TC, T_H_arr)
    ax2.plot(T_H_arr, eta*100, lw=2, label=f'$T_C = {TC}$ K')
ax2.set_xlabel(r'Heißgastemperatur $T_H$ [K]', fontsize=12)
ax2.set_ylabel(r'Carnot-Wirkungsgrad $\eta_C$ [%]', fontsize=12)
ax2.set_title(r'$\eta_C = 1 - T_C/T_H$', fontsize=12)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 80)

plt.tight_layout()
plt.savefig('plot_thermodynamik.pdf', bbox_inches='tight')
plt.close()
print("plot_thermodynamik.pdf OK")

# ============================================================
# Plot 6: Werkstofftechnik -- Eisen-Kohlenstoff-Diagramm (vereinfacht)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Vereinfachtes Fe-C-Diagramm mit den wichtigsten Linien
C = np.linspace(0, 6.67, 300)

# Liquiduslinie (vereinfacht)
# A-B: 1536 bis 1493 (Peritektkum)
# B-C: 1493 bis 1147 (Eutektikum)
# C-D: 1147 konstant -> 6.67%

# Soliduslinie
# A-H: bis 0.1% bei 1493
# H-J: 0.1 bis 0.51 bei 1493
# ... vereinfacht:

# Liquiduskurve Stahl-Bereich (0-2.06%)
C_s = np.linspace(0, 2.06, 100)
T_liq_s = 1536 - (1536-1147)/2.06 * C_s

# Liquiduskurve Gusseisen (2.06-6.67%)
C_g = np.linspace(2.06, 6.67, 100)
T_liq_g = 1147 + (1493-1147)*(6.67-C_g)/(6.67-2.06)

ax.plot(C_s, T_liq_s, 'r-', lw=2, label='Liquiduslinie')
ax.plot(C_g, T_liq_g, 'r-', lw=2)

# Soliduslinie (vereinfacht)
C_sol = np.array([0, 0.8, 2.06])
T_sol = np.array([1536, 723, 1147])
ax.plot([0, 0.8], [1536, 723], 'b-', lw=1.5, label='Soliduslinie (vereinf.)')

# Eutektoidale Linie (723°C, PSK)
ax.axhline(723, xmin=0, xmax=0.8/6.67+0.01, color='green', lw=1.5, ls='--', label=r'$A_1 = 723\,°C$ (Eutektoid)')
# Eutektikale Linie (1147°C, ECF)
ax.axhline(1147, color='orange', lw=1.5, ls='--', label=r'$A_{ec} = 1147\,°C$ (Eutektikum)')

# Wichtige Punkte
ax.plot(0.8, 723, 'ko', ms=8, zorder=5)
ax.text(0.85, 710, 'P (0.8% C)', fontsize=9)
ax.plot(4.3, 1147, 'ko', ms=8, zorder=5)
ax.text(4.35, 1130, 'C (4.3% C)', fontsize=9)
ax.plot(0, 1536, 'rs', ms=8, zorder=5)
ax.text(0.05, 1540, 'A (1536°C)', fontsize=9)

# Bereiche beschriften
ax.text(0.4, 900, 'Austenit\n$\gamma$', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(0.4, 600, 'Ferrit + Perlit\n$\alpha$ + Fe$_3$C', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(3.5, 900, 'Austenit +\nLedeburit', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.axvline(2.06, color='purple', lw=1, ls=':', label='2.06% C (max. Löslichkeit)')
ax.text(2.1, 400, 'Stahl | Gusseisen', fontsize=9, color='purple', rotation=90)

ax.set_xlabel(r'Kohlenstoffgehalt [Masse-%]', fontsize=12)
ax.set_ylabel(r'Temperatur $T$ [°C]', fontsize=12)
ax.set_title('Eisen-Kohlenstoff-Diagramm (vereinfacht)', fontsize=13)
ax.set_xlim(0, 6.7); ax.set_ylim(300, 1600)
ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_eisen_kohlenstoff.pdf', bbox_inches='tight')
plt.close()
print("plot_eisen_kohlenstoff.pdf OK")

# ============================================================
# Plot 7: Maschinenelemente -- Schraubenverbindung (Kraft vs. Anzugsmoment)
#         und Welle-Nabe-Verbindung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# Anzugsmoment M_A = F_V * (0.5*p/pi + mu_G * d_2/2 + mu_K * D_Km/2)
# Vereinfacht: M_A ≈ 0.2 * F_V * d
d_arr = np.array([6, 8, 10, 12, 16, 20, 24])  # mm Schraubendurchmesser
mu = 0.12  # Reibungszahl
F_V_list = [0.75 * 600 * d**1.6 for d in d_arr]  # grobe Näherung Vorspannkraft [N]
M_A_list = [0.2 * F * d/1000 for F, d in zip(F_V_list, d_arr)]  # [Nm]

ax.bar(d_arr, M_A_list, color='steelblue', edgecolor='black', width=1.2, label='Anzugsmoment $M_A$')
ax.set_xlabel(r'Schraubendurchmesser $d$ [mm]', fontsize=12)
ax.set_ylabel(r'Anzugsmoment $M_A$ [Nm]', fontsize=12)
ax.set_title(r'Richtwerte Anzugsmoment ($\mu = 0{,}12$, Festigkl. 8.8)', fontsize=11)
ax.set_xticks(d_arr)
for x, y in zip(d_arr, M_A_list):
    ax.text(x, y+0.5, f'{y:.1f}', ha='center', fontsize=9)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
# Welle-Nabe: übertragbares Moment einer Passfederverbindung
# M_t = F_t * d/2 = p_zul * l * h/2 * d/2
b_arr = np.array([4, 5, 6, 8, 10, 12, 14, 16])  # Passfederbreite [mm]
d_w = 2.5 * b_arr  # grobe Abschätzung Wellendurchmesser
l_eff = 3 * b_arr
h = 0.55 * b_arr
p_zul = 100  # MPa (Stahl)
M_t = p_zul * l_eff/1000 * h/2/1000 * d_w/2/1000 * 1000  # [Nm]

ax2.plot(d_w, M_t, 'ro-', lw=2, ms=6, label=r'$M_t = p_{zul} \cdot l \cdot h/2 \cdot d/2$')
ax2.set_xlabel(r'Wellendurchmesser $d_w$ [mm]', fontsize=12)
ax2.set_ylabel(r'Übertragbares Moment $M_t$ [Nm]', fontsize=12)
ax2.set_title(r'Passfederverbindung ($p_{zul} = 100$ MPa)', fontsize=11)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_maschinenelemente.pdf', bbox_inches='tight')
plt.close()
print("plot_maschinenelemente.pdf OK")

# ============================================================
# Plot 8: Strömungslehre -- Bernoulli und Rohrströmung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# Bernoulli: Venturi-Effekt
D1, D2 = 0.1, 0.05  # m Rohrdurchmesser
v1 = 2.0  # m/s
v2 = v1 * (D1/D2)**2  # Kontinuität
rho = 1000  # kg/m3 Wasser
p1 = 200000  # Pa
p2 = p1 + 0.5*rho*(v1**2 - v2**2)  # Bernoulli

x = np.linspace(0, 5, 200)
# Rohrprofil
r_top = np.where(x < 1.5, 0.05,
         np.where(x < 2.5, 0.05 - 0.025*(x-1.5)/0.5,
         np.where(x < 3.0, 0.025,
         np.where(x < 4.0, 0.025 + 0.025*(x-3.0)/1.0, 0.05))))
# Druckverlauf (vereinfacht)
p_arr = np.where(x < 1.5, p1,
        np.where(x < 2.5, p1 + (p2-p1)*(x-1.5)/1.0,
        np.where(x < 3.0, p2,
        np.where(x < 4.0, p2 + (p1-p2)*(x-3.0)/1.0, p1))))

ax.plot(x, r_top*1000, 'b-', lw=2)
ax.plot(x, -r_top*1000, 'b-', lw=2)
ax.fill_between(x, r_top*1000, -r_top*1000, alpha=0.15, color='blue')
ax.set_xlabel(r'Position $x$ [m]', fontsize=12)
ax.set_ylabel(r'Rohrhalbmesser [mm]', fontsize=12)
ax.set_title('Venturi-Rohr: Querschnittsverengung', fontsize=12)
ax.text(0.5, 58, r'$D_1 = 100$ mm', fontsize=9)
ax.text(2.6, 30, r'$D_2 = 50$ mm', fontsize=9)
ax.grid(True, alpha=0.3); ax.set_ylim(-80, 80)

ax2 = axes[1]
ax2.plot(x, p_arr/1000, 'r-', lw=2.5, label='Druck $p(x)$')
ax2.axhline(p1/1000, color='gray', ls='--', lw=1)
ax2.axhline(p2/1000, color='gray', ls='--', lw=1)
ax2.text(0.2, p1/1000+1, f'$p_1={p1/1000:.0f}$ kPa', fontsize=10)
ax2.text(2.6, p2/1000-4, f'$p_2={p2/1000:.1f}$ kPa', fontsize=10)
ax2.set_xlabel(r'Position $x$ [m]', fontsize=12)
ax2.set_ylabel(r'Druck $p$ [kPa]', fontsize=12)
ax2.set_title(r'Druckverlauf (Bernoulli: $p + \frac{\rho v^2}{2} = \mathrm{const}$)', fontsize=11)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_stroemung.pdf', bbox_inches='tight')
plt.close()
print("plot_stroemung.pdf OK")

# ============================================================
# Plot 9: Regelungstechnik -- PT1/PT2 Sprungantwort
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t = np.linspace(0, 10, 500)

ax = axes[0]
# PT1 für verschiedene T
for T, col in [(0.5,'blue'),(1.0,'red'),(2.0,'green')]:
    y = 1 - np.exp(-t/T)
    ax.plot(t, y, lw=2, color=col, label=f'$T = {T}$ s')
ax.axhline(1, color='k', ls='--', lw=1)
ax.set_xlabel(r'Zeit $t$ [s]', fontsize=12)
ax.set_ylabel(r'Ausgangsgröße $y(t)$', fontsize=12)
ax.set_title('Sprungantwort PT1-Glied: $y(t)=1-e^{-t/T}$', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

ax2 = axes[1]
# PT2 für verschiedene Dämpfungen
T2 = 1.0
w_n = 1.0/T2
for D, col in [(0.3,'blue'),(0.7,'red'),(1.0,'green'),(2.0,'orange')]:
    if D < 1:
        w_d = w_n * np.sqrt(1 - D**2)
        y = 1 - np.exp(-D*w_n*t)*(np.cos(w_d*t) + D/np.sqrt(1-D**2)*np.sin(w_d*t))
    else:
        s1 = -w_n*(D - np.sqrt(D**2-1)) if D > 1 else -w_n
        s2 = -w_n*(D + np.sqrt(D**2-1)) if D > 1 else -w_n
        if D == 1:
            y = 1 - (1 + w_n*t)*np.exp(-w_n*t)
        else:
            y = 1 + (s1*np.exp(s2*t) - s2*np.exp(s1*t))/(s2-s1)
    ax2.plot(t, y, lw=2, color=col, label=f'$D = {D}$')

ax2.axhline(1, color='k', ls='--', lw=1)
ax2.set_xlabel(r'Zeit $t$ [s]', fontsize=12)
ax2.set_ylabel(r'Ausgangsgröße $y(t)$', fontsize=12)
ax2.set_title('Sprungantwort PT2-Glied (verschiedene Dämpfungen $D$)', fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.1, 1.8)

plt.tight_layout()
plt.savefig('plot_regelung.pdf', bbox_inches='tight')
plt.close()
print("plot_regelung.pdf OK")

# ============================================================
# Plot 10: Schwingungslehre -- erzwungene Schwingung
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
eta = np.linspace(0.01, 3, 500)
for D, col in [(0.05,'blue'),(0.15,'red'),(0.3,'green'),(0.7,'orange'),(1.0,'purple')]:
    V = 1/np.sqrt((1 - eta**2)**2 + (2*D*eta)**2)
    ax.plot(eta, V, lw=2, color=col, label=f'$D={D}$')
ax.axvline(1, color='gray', ls='--', lw=1)
ax.set_xlabel(r'Frequenzverhältnis $\eta = \Omega/\omega_0$', fontsize=11)
ax.set_ylabel(r'Vergrößerungsfaktor $V$', fontsize=12)
ax.set_title('Resonanzkurven des erzwungenen Schwingers', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 8)
ax.text(1.02, 7.5, r'Resonanz $\eta=1$', fontsize=9, color='gray')

ax2 = axes[1]
t = np.linspace(0, 20, 1000)
D = 0.1; w0 = 1.0; Omega = 0.95
# Erzwungene Schwingung
F0_m = 1.0
# Homogene + partikuläre Lösung
wd = w0*np.sqrt(1-D**2)
eta_v = Omega/w0
V = 1/np.sqrt((1-eta_v**2)**2 + (2*D*eta_v)**2)
phi = np.arctan2(2*D*eta_v, 1-eta_v**2)
y_p = V * F0_m/w0**2 * np.cos(Omega*t - phi)
# Homogene (transient)
A = -V * F0_m/w0**2 * np.cos(-phi)
B = -(-A*D*w0 - V*F0_m/w0**2*(-Omega)*np.sin(-phi))/wd
y_h = np.exp(-D*w0*t) * (A*np.cos(wd*t) + B*np.sin(wd*t))
y = y_h + y_p

ax2.plot(t, y, 'b-', lw=1.5, label='Gesamtschwingung $x(t)$')
ax2.plot(t, y_p, 'r--', lw=1.5, label='Partikuläre Lösung')
ax2.plot(t, y_h, 'g:', lw=1.5, label='Einschwingvorgang')
ax2.set_xlabel(r'Zeit $t$ [s]', fontsize=12)
ax2.set_ylabel(r'Auslenkung $x(t)$', fontsize=12)
ax2.set_title(r'Erzwungene Schwingung ($D=0{,}1$, $\eta=0{,}95$)', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_schwingung.pdf', bbox_inches='tight')
plt.close()
print("plot_schwingung.pdf OK")

print("\nAlle Plots erstellt!")
