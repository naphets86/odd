#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, poisson
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11,
                     'legend.fontsize': 10, 'figure.dpi': 150})
OUT = ''

# ===== Plot 1: Kontinuierlich vs. diskret =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
t_cont = np.linspace(0, 4*np.pi, 2000)
y_cont = np.sin(t_cont) + 0.3*np.sin(3*t_cont) + 0.1*np.cos(5*t_cont)
t_disc = np.linspace(0, 4*np.pi, 30)
y_disc = np.sin(t_disc) + 0.3*np.sin(3*t_disc) + 0.1*np.cos(5*t_disc)

axes[0].plot(t_cont, y_cont, color='#1946AC', lw=1.8, label=r'Analoges Signal $x(t)$')
axes[0].set_title('Analoges (kontinuierliches) Signal')
axes[0].set_xlabel(r'Zeit $t$ [s]'); axes[0].set_ylabel(r'Amplitude $x(t)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_xlim([0, 4*np.pi])

axes[1].plot(t_cont, y_cont, color='#AAAACC', lw=1.2, linestyle='--', label='Analoges Signal (Referenz)')
markerline, stemlines, baseline = axes[1].stem(t_disc, y_disc, label=r'Diskretes Signal $x[n]$')
markerline.set_color('#B43218'); stemlines.set_color('#B43218'); baseline.set_color('black')
axes[1].set_title('Diskretes (abgetastetes) Signal')
axes[1].set_xlabel(r'Zeit $t$ [s]'); axes[1].set_ylabel(r'Amplitude $x[n]$')
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xlim([0, 4*np.pi])
plt.tight_layout()
plt.savefig(OUT + 'plot1_kontinuierlich_diskret.pdf', bbox_inches='tight'); plt.close()
print("Plot 1 OK")

# ===== Plot 2: Nyquist =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
f_signal = 5.0
fs_list = [7.0, 10.0, 50.0]
titles = [r'Unterabtastung ($f_s < 2f$)', r'Kritische Abtastung ($f_s = 2f$)', r'Korrekte Abtastung ($f_s \gg 2f$)']
colors_stem = ['#B43218', '#E8A020', '#1946AC']
t_ref = np.linspace(0, 1.0, 5000)
y_ref = np.sin(2*np.pi*f_signal*t_ref)
for i, (fs, title, col) in enumerate(zip(fs_list, titles, colors_stem)):
    t_s = np.arange(0, 1.0, 1.0/fs)
    y_s = np.sin(2*np.pi*f_signal*t_s)
    axes[i].plot(t_ref, y_ref, color='#AAAACC', lw=1.5, linestyle='--', label='Original')
    ml, sl, bl = axes[i].stem(t_s, y_s, label=f'$f_s={fs}$ Hz')
    ml.set_color(col); sl.set_color(col); bl.set_color('black')
    axes[i].set_title(title); axes[i].set_xlabel('Zeit $t$ [s]'); axes[i].set_ylabel('Amplitude')
    axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3); axes[i].set_xlim([0, 1.0])
plt.tight_layout()
plt.savefig(OUT + 'plot2_nyquist.pdf', bbox_inches='tight'); plt.close()
print("Plot 2 OK")

# ===== Plot 3: RC-Glied =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
R, C = 1000.0, 1e-3; tau = R*C; V0 = 5.0
t = np.linspace(0, 5*tau, 2000)
V_charge = V0*(1 - np.exp(-t/tau)); V_discharge = V0*np.exp(-t/tau)
axes[0].plot(t*1000, V_charge, color='#1946AC', lw=2.0, label=r'$U_C(t)=U_0(1-e^{-t/\tau})$')
axes[0].axhline(V0, color='gray', linestyle=':', lw=1.0, label=r'$U_0=5\,\mathrm{V}$')
axes[0].axvline(tau*1000, color='#B43218', linestyle='--', lw=1.5, label=r'$\tau=RC$')
axes[0].set_title('RC-Glied: Aufladung'); axes[0].set_xlabel(r'Zeit $t$ [ms]')
axes[0].set_ylabel(r'Spannung $U_C(t)$ [V]'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(t*1000, V_discharge, color='#B43218', lw=2.0, label=r'$U_C(t)=U_0 e^{-t/\tau}$')
axes[1].axvline(tau*1000, color='#1946AC', linestyle='--', lw=1.5, label=r'$\tau=RC$')
axes[1].set_title('RC-Glied: Entladung'); axes[1].set_xlabel(r'Zeit $t$ [ms]')
axes[1].set_ylabel(r'Spannung $U_C(t)$ [V]'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'plot3_rc_glied.pdf', bbox_inches='tight'); plt.close()
print("Plot 3 OK")

# ===== Plot 4: Fourier =====
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
t = np.linspace(-5, 5, 10000); dt = t[1]-t[0]
f_g = np.exp(-t**2/(2*0.5**2))
N = len(t); freqs = np.fft.fftshift(np.fft.fftfreq(N, dt))
F_g = np.fft.fftshift(np.fft.fft(f_g)) * dt
f_r = np.where(np.abs(t) <= 1.0, 1.0, 0.0)
F_r = np.fft.fftshift(np.fft.fft(f_r)) * dt
axes[0,0].plot(t, f_g, color='#1946AC', lw=2.0)
axes[0,0].set_title(r'Gauss-Funktion $x(t)=e^{-t^2/(2\sigma^2)}$')
axes[0,0].set_xlabel('Zeit $t$'); axes[0,0].set_ylabel('Amplitude')
axes[0,0].set_xlim([-4,4]); axes[0,0].grid(True, alpha=0.3)
axes[0,1].plot(freqs, np.abs(F_g), color='#1946AC', lw=2.0)
axes[0,1].set_title(r'Fourier-Spektrum $|\hat{x}(f)|$ (Gauss)')
axes[0,1].set_xlabel('Frequenz $f$ [Hz]'); axes[0,1].set_ylabel('Spektrale Amplitude')
axes[0,1].set_xlim([-5,5]); axes[0,1].grid(True, alpha=0.3)
axes[1,0].plot(t, f_r, color='#B43218', lw=2.0)
axes[1,0].set_title(r'Rechteck-Funktion $\Pi(t/T)$')
axes[1,0].set_xlabel('Zeit $t$'); axes[1,0].set_ylabel('Amplitude')
axes[1,0].set_xlim([-4,4]); axes[1,0].grid(True, alpha=0.3)
axes[1,1].plot(freqs, np.abs(F_r), color='#B43218', lw=2.0)
axes[1,1].set_title(r'Fourier-Spektrum: $|\hat{x}(f)|=T\,\mathrm{sinc}(\pi fT)$')
axes[1,1].set_xlabel('Frequenz $f$ [Hz]'); axes[1,1].set_ylabel('Spektrale Amplitude')
axes[1,1].set_xlim([-5,5]); axes[1,1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'plot4_fourier.pdf', bbox_inches='tight'); plt.close()
print("Plot 4 OK")

# ===== Plot 5: Quantisierung =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
t = np.linspace(0, 2*np.pi, 2000); y = np.sin(t)
for bits, ax in zip([3, 8], axes):
    levels = 2**bits; step = 2.0/levels
    y_q = np.round(y/step)*step
    ax.plot(t, y, color='#1946AC', lw=1.8, label='Analog', alpha=0.85)
    ax.step(t, y_q, color='#B43218', lw=1.2, label=f'Digital ({bits} Bit, {levels} Stufen)')
    ax.fill_between(t, y, y_q, alpha=0.25, color='#E8A020', label='Quantisierungsfehler')
    ax.set_title(f'{bits}-Bit-Quantisierung (LSB = {step:.4f})')
    ax.set_xlabel('Zeit $t$'); ax.set_ylabel('Amplitude')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'plot5_quantisierung.pdf', bbox_inches='tight'); plt.close()
print("Plot 5 OK")

# ===== Plot 6: Lorenz =====
def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
t_span = np.linspace(0, 40, 10000)
sol = odeint(lorenz, [1.0, 1.0, 1.0], t_span)
fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.4, color='#1946AC', alpha=0.7)
ax1.set_title('Lorenz-Attraktor (kontinuierlich)'); ax1.set_xlabel('$x(t)$'); ax1.set_ylabel('$y(t)$'); ax1.set_zlabel('$z(t)$')
ax2 = fig.add_subplot(122)
ax2.plot(t_span[:3000], sol[:3000,0], color='#1946AC', lw=0.8, label='$x(t)$')
ax2.plot(t_span[:3000], sol[:3000,1], color='#B43218', lw=0.8, alpha=0.7, label='$y(t)$')
ax2.set_title('Lorenz: Zeitverlauf'); ax2.set_xlabel('Zeit $t$'); ax2.set_ylabel('Zustand')
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'plot6_lorenz.pdf', bbox_inches='tight'); plt.close()
print("Plot 6 OK")

# ===== Plot 7: Verteilungen =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_cont = np.linspace(-4, 4, 1000)
pdf_norm = norm.pdf(x_cont, 0, 1)
axes[0].plot(x_cont, pdf_norm, color='#1946AC', lw=2.2, label=r'$\mathcal{N}(0,1)$')
axes[0].fill_between(x_cont, pdf_norm, alpha=0.18, color='#1946AC')
axes[0].set_title('Kontinuierliche Normalverteilung (PDF)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel(r'Wahrscheinlichkeitsdichte $f(x)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
k_vals = np.arange(0, 15)
pmf_pois = poisson.pmf(k_vals, 4.0)
axes[1].bar(k_vals, pmf_pois, color='#B43218', alpha=0.75, label=r'Poisson($\mu=4$)')
axes[1].set_title('Diskrete Poisson-Verteilung (PMF)')
axes[1].set_xlabel('$k$'); axes[1].set_ylabel(r'$P(X=k)$')
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT + 'plot7_wahrscheinlichkeit.pdf', bbox_inches='tight'); plt.close()
print("Plot 7 OK")

# ===== Plot 8: Shannon =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
SNR_dB = np.linspace(0, 40, 500)
SNR_lin = 10**(SNR_dB/10)
C_shannon = np.log2(1 + SNR_lin)
axes[0].plot(SNR_dB, C_shannon, color='#1946AC', lw=2.2, label=r'$C=B\log_2(1+\mathrm{SNR})$')
axes[0].set_title('Shannon-Kanalkapazität ($B=1$ Hz)')
axes[0].set_xlabel('SNR [dB]'); axes[0].set_ylabel(r'Kapazität $C$ [bit/s/Hz]')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
bits = np.arange(1, 17)
SQNR_dB_vals = 6.02*bits + 1.76
axes[1].plot(bits, SQNR_dB_vals, color='#B43218', lw=2.2, marker='o', ms=5,
             label=r'$\mathrm{SQNR}\approx 6.02n+1.76\,\mathrm{dB}$')
axes[1].set_title('Quantisierungs-SNR vs. Bittiefe')
axes[1].set_xlabel('Bittiefe $n$'); axes[1].set_ylabel('SQNR [dB]')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'plot8_shannon.pdf', bbox_inches='tight'); plt.close()
print("Plot 8 OK")

print("\nAlle Plots fertig.")

