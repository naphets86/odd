#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Plot 5 (Fe-C fix)
fig, ax = plt.subplots(figsize=(10, 7))
C_s = np.linspace(0, 2.06, 100)
T_liq_s = 1536 - (1536-1147)/2.06 * C_s
C_g = np.linspace(2.06, 6.67, 100)
T_liq_g = 1147 + (1493-1147)*(6.67-C_g)/(6.67-2.06)

ax.plot(C_s, T_liq_s, 'r-', lw=2, label='Liquiduslinie')
ax.plot(C_g, T_liq_g, 'r-', lw=2)
ax.axhline(723, color='green', lw=1.5, ls='--', label='$A_1 = 723$ Grad C (Eutektoid)')
ax.axhline(1147, color='orange', lw=1.5, ls='--', label='$A_{ec} = 1147$ Grad C (Eutektikum)')
ax.plot(0.8, 723, 'ko', ms=8)
ax.text(0.85, 705, 'P (0.8% C)', fontsize=9)
ax.plot(4.3, 1147, 'ko', ms=8)
ax.text(4.35, 1125, 'C (4.3% C)', fontsize=9)
ax.plot(0, 1536, 'rs', ms=8)
ax.text(0.05, 1538, 'A (1536 Grad C)', fontsize=9)
ax.text(0.4, 900, 'Austenit', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(0.4, 580, 'Ferrit + Perlit', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(3.5, 900, 'Austenit + Ledeburit', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.axvline(2.06, color='purple', lw=1, ls=':', label='2.06% C (max. Loeslichkeit)')
ax.text(2.1, 380, 'Stahl | Gusseisen', fontsize=9, color='purple', rotation=90)
ax.set_xlabel('Kohlenstoffgehalt [Masse-%]', fontsize=12)
ax.set_ylabel('Temperatur T [Grad C]', fontsize=12)
ax.set_title('Eisen-Kohlenstoff-Diagramm (vereinfacht)', fontsize=13)
ax.set_xlim(0, 6.7); ax.set_ylim(300, 1600)
ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_eisen_kohlenstoff.pdf', bbox_inches='tight')
plt.close()
print("plot_eisen_kohlenstoff.pdf OK")

# Plot 6: Maschinenelemente
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
d_arr = np.array([6, 8, 10, 12, 16, 20, 24])
F_V_list = [0.75 * 600 * d**1.6 for d in d_arr]
M_A_list = [0.2 * F * d/1000 for F, d in zip(F_V_list, d_arr)]
ax.bar(d_arr, M_A_list, color='steelblue', edgecolor='black', width=1.2)
ax.set_xlabel('Schraubendurchmesser d [mm]', fontsize=12)
ax.set_ylabel('Anzugsmoment M_A [Nm]', fontsize=12)
ax.set_title('Richtwerte Anzugsmoment (mu=0.12, Festigkl. 8.8)', fontsize=11)
ax.set_xticks(d_arr)
for x, y in zip(d_arr, M_A_list):
    ax.text(x, y+0.5, f'{y:.1f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
b_arr = np.array([4, 5, 6, 8, 10, 12, 14, 16])
d_w = 2.5 * b_arr
l_eff = 3 * b_arr
h = 0.55 * b_arr
p_zul = 100
M_t = p_zul * l_eff/1000 * h/2/1000 * d_w/2/1000 * 1000
ax2.plot(d_w, M_t, 'ro-', lw=2, ms=6, label='M_t (Passfeder)')
ax2.set_xlabel('Wellendurchmesser d_w [mm]', fontsize=12)
ax2.set_ylabel('Uebertragbares Moment M_t [Nm]', fontsize=12)
ax2.set_title('Passfederverbindung (p_zul = 100 MPa)', fontsize=11)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_maschinenelemente.pdf', bbox_inches='tight')
plt.close()
print("plot_maschinenelemente.pdf OK")

# Plot 7: Stroemung Bernoulli
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
D1, D2 = 0.1, 0.05
v1 = 2.0
v2 = v1 * (D1/D2)**2
rho = 1000
p1 = 200000
p2 = p1 + 0.5*rho*(v1**2 - v2**2)
x = np.linspace(0, 5, 200)
r_top = np.where(x < 1.5, 0.05,
         np.where(x < 2.5, 0.05 - 0.025*(x-1.5)/1.0,
         np.where(x < 3.0, 0.025,
         np.where(x < 4.0, 0.025 + 0.025*(x-3.0)/1.0, 0.05))))
p_arr = np.where(x < 1.5, p1,
        np.where(x < 2.5, p1 + (p2-p1)*(x-1.5)/1.0,
        np.where(x < 3.0, p2,
        np.where(x < 4.0, p2 + (p1-p2)*(x-3.0)/1.0, p1))))

ax = axes[0]
ax.plot(x, r_top*1000, 'b-', lw=2)
ax.plot(x, -r_top*1000, 'b-', lw=2)
ax.fill_between(x, r_top*1000, -r_top*1000, alpha=0.15, color='blue')
ax.set_xlabel('Position x [m]', fontsize=12)
ax.set_ylabel('Rohrradius [mm]', fontsize=12)
ax.set_title('Venturi-Rohr: Querschnittsverengung', fontsize=12)
ax.text(0.5, 58, 'D1 = 100 mm', fontsize=9)
ax.text(2.6, 30, 'D2 = 50 mm', fontsize=9)
ax.grid(True, alpha=0.3); ax.set_ylim(-80, 80)

ax2 = axes[1]
ax2.plot(x, p_arr/1000, 'r-', lw=2.5, label='Druck p(x)')
ax2.axhline(p1/1000, color='gray', ls='--', lw=1)
ax2.axhline(p2/1000, color='gray', ls='--', lw=1)
ax2.text(0.2, p1/1000+1, f'p1={p1/1000:.0f} kPa', fontsize=10)
ax2.text(2.6, p2/1000-4, f'p2={p2/1000:.1f} kPa', fontsize=10)
ax2.set_xlabel('Position x [m]', fontsize=12)
ax2.set_ylabel('Druck p [kPa]', fontsize=12)
ax2.set_title('Druckverlauf nach Bernoulli', fontsize=11)
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_stroemung.pdf', bbox_inches='tight')
plt.close()
print("plot_stroemung.pdf OK")

# Plot 8: Regelungstechnik
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
t = np.linspace(0, 10, 500)
ax = axes[0]
for T, col in [(0.5,'blue'),(1.0,'red'),(2.0,'green')]:
    y = 1 - np.exp(-t/T)
    ax.plot(t, y, lw=2, color=col, label=f'T = {T} s')
ax.axhline(1, color='k', ls='--', lw=1)
ax.set_xlabel('Zeit t [s]', fontsize=12)
ax.set_ylabel('Ausgangsgr. y(t)', fontsize=12)
ax.set_title('Sprungantwort PT1-Glied', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

ax2 = axes[1]
w_n = 1.0
for D, col in [(0.3,'blue'),(0.7,'red'),(1.0,'green'),(2.0,'orange')]:
    if D < 1:
        w_d = w_n*np.sqrt(1-D**2)
        y = 1 - np.exp(-D*w_n*t)*(np.cos(w_d*t) + D/np.sqrt(1-D**2)*np.sin(w_d*t))
    elif D == 1:
        y = 1 - (1 + w_n*t)*np.exp(-w_n*t)
    else:
        s1 = -w_n*(D - np.sqrt(D**2-1))
        s2 = -w_n*(D + np.sqrt(D**2-1))
        y = 1 + (s1*np.exp(s2*t) - s2*np.exp(s1*t))/(s2-s1)
    ax2.plot(t, y, lw=2, color=col, label=f'D = {D}')
ax2.axhline(1, color='k', ls='--', lw=1)
ax2.set_xlabel('Zeit t [s]', fontsize=12)
ax2.set_ylabel('Ausgangsgr. y(t)', fontsize=12)
ax2.set_title('Sprungantwort PT2-Glied (verschiedene D)', fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.1, 1.8)
plt.tight_layout()
plt.savefig('plot_regelung.pdf', bbox_inches='tight')
plt.close()
print("plot_regelung.pdf OK")

# Plot 9: Schwingungen
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
eta = np.linspace(0.01, 3, 500)
for D, col in [(0.05,'blue'),(0.15,'red'),(0.3,'green'),(0.7,'orange'),(1.0,'purple')]:
    V = 1/np.sqrt((1 - eta**2)**2 + (2*D*eta)**2)
    ax.plot(eta, V, lw=2, color=col, label=f'D={D}')
ax.axvline(1, color='gray', ls='--', lw=1)
ax.set_xlabel('Frequenzverh. eta = Omega/omega_0', fontsize=11)
ax.set_ylabel('Vergroesserungsfaktor V', fontsize=12)
ax.set_title('Resonanzkurven des erzwungenen Schwingers', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 8)

ax2 = axes[1]
t = np.linspace(0, 30, 1500)
D = 0.1; w0 = 1.0; Omega = 0.95
eta_v = Omega/w0
V = 1/np.sqrt((1-eta_v**2)**2 + (2*D*eta_v)**2)
phi = np.arctan2(2*D*eta_v, 1-eta_v**2)
y_p = V * np.cos(Omega*t - phi)
wd = w0*np.sqrt(1-D**2)
# IC: y(0)=0, y'(0)=0
A = -V*np.cos(-phi)
B = (-(-A*D*w0) - V*(-Omega)*np.sin(-phi))/wd
y_h = np.exp(-D*w0*t)*(A*np.cos(wd*t) + B*np.sin(wd*t))
y = y_h + y_p
ax2.plot(t, y, 'b-', lw=1.5, label='Gesamtschwingung')
ax2.plot(t, y_p, 'r--', lw=1.5, label='Stationaer')
ax2.set_xlabel('Zeit t [s]', fontsize=12)
ax2.set_ylabel('Auslenkung x(t)', fontsize=12)
ax2.set_title('Erzwungene Schwingung (D=0.1, eta=0.95)', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_schwingung.pdf', bbox_inches='tight')
plt.close()
print("plot_schwingung.pdf OK")

print("\nAlle verbleibenden Plots OK!")
