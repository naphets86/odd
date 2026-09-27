#!/usr/bin/env python3
"""
Plot generation script for the scientific paper on matplotlib, numpy, and scipy.
Run this script before compiling the LaTeX document.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal, stats, optimize, integrate, fft, linalg, interpolate, special
from scipy.ndimage import gaussian_filter
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ─────────────────────────────────────────────────────────
# Figure 1: NumPy – Array operations and broadcasting
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Broadcasting visualisation
x = np.linspace(0, 2 * np.pi, 200)
freqs = np.array([1, 2, 3, 4, 5])[:, np.newaxis]          # (5,1)
signals = np.sin(freqs * x)                                  # (5,200) – broadcasting
for i, sig in enumerate(signals):
    axes[0].plot(x, sig, label=f'$f={i+1}$')
axes[0].set_title('Broadcasting: $\\sin(f \\cdot x)$')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# NumPy random distributions
rng = np.random.default_rng(42)
data_norm   = rng.normal(0, 1, 5000)
data_exp    = rng.exponential(1, 5000)
data_poisson = rng.poisson(5, 5000)
axes[1].hist(data_norm,    bins=60, alpha=0.6, label='Normal(0,1)')
axes[1].hist(data_exp,     bins=60, alpha=0.6, label='Exp(1)')
axes[1].hist(data_poisson, bins=30, alpha=0.6, label='Poisson(5)')
axes[1].set_title('Zufallsverteilungen (NumPy)')
axes[1].set_xlabel('Wert')
axes[1].set_ylabel('Häufigkeit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# NumPy linear algebra: eigenvalue decomposition
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
theta = np.linspace(0, 2*np.pi, 300)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
axes[2].plot(circle_x, circle_y, 'k--', alpha=0.3, label='Einheitskreis')
colors = ['tab:blue', 'tab:orange']
for i in range(2):
    v = eigenvectors[:, i]
    axes[2].quiver(0, 0, v[0]*eigenvalues[i], v[1]*eigenvalues[i],
                   angles='xy', scale_units='xy', scale=1,
                   color=colors[i], label=f'$\\lambda_{i+1}={eigenvalues[i]:.1f}$')
axes[2].set_xlim(-5, 5); axes[2].set_ylim(-5, 5)
axes[2].set_aspect('equal')
axes[2].set_title('Eigenvektoren / Eigenwerte')
axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle('NumPy – Array-Operationen, Zufallsverteilungen und lineare Algebra', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/numpy_overview.pdf')
plt.close(fig)
print("Fig 1 done")

# ─────────────────────────────────────────────────────────
# Figure 2: NumPy – FFT and polynomials
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# FFT
fs = 1000.0
t  = np.linspace(0, 1, int(fs), endpoint=False)
sig_fft = (np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
           + 0.3*np.random.default_rng(0).normal(size=len(t)))
freqs_fft = np.fft.rfftfreq(len(t), 1/fs)
spectrum  = np.abs(np.fft.rfft(sig_fft)) / len(t) * 2
axes[0].plot(t[:200], sig_fft[:200])
axes[0].set_title('Zeitsignal (Ausschnitt)')
axes[0].set_xlabel('Zeit [s]'); axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(freqs_fft, spectrum)
axes[1].set_title('FFT-Spektrum (NumPy)')
axes[1].set_xlabel('Frequenz [Hz]'); axes[1].set_ylabel('|X(f)|')
axes[1].set_xlim(0, 200); axes[1].grid(True, alpha=0.3)

# Polynomial fitting
xp = np.linspace(-3, 3, 50)
yp = 2*xp**3 - xp**2 + 0.5*xp - 1 + np.random.default_rng(7).normal(0, 2, 50)
for deg, col in zip([1, 3, 7], ['tab:red', 'tab:green', 'tab:purple']):
    coeffs = np.polyfit(xp, yp, deg)
    poly   = np.poly1d(coeffs)
    xfit   = np.linspace(-3, 3, 300)
    axes[2].plot(xfit, poly(xfit), label=f'Grad {deg}', color=col)
axes[2].scatter(xp, yp, s=10, color='k', zorder=5, label='Daten')
axes[2].set_title('Polynominterpolation (NumPy)')
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$y$')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle('NumPy – Fourier-Transformation und Polynomfit', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/numpy_fft_poly.pdf')
plt.close(fig)
print("Fig 2 done")

# ─────────────────────────────────────────────────────────
# Figure 3: SciPy – Signal Processing
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

fs2 = 500.0
t2  = np.linspace(0, 2, int(2*fs2), endpoint=False)
chirp_sig  = signal.chirp(t2, f0=1, f1=100, t1=2, method='linear')
noisy_chirp = chirp_sig + 0.4 * np.random.default_rng(1).normal(size=len(t2))

# Butterworth lowpass
b, a = signal.butter(5, 40, btype='low', fs=fs2)
filtered = signal.filtfilt(b, a, noisy_chirp)
axes[0,0].plot(t2[:500], noisy_chirp[:500], alpha=0.6, label='verrauscht')
axes[0,0].plot(t2[:500], filtered[:500],    lw=2,       label='gefiltert (Butterworth 5. Ord.)')
axes[0,0].set_title('Butterworth-Tiefpassfilter')
axes[0,0].set_xlabel('Zeit [s]'); axes[0,0].set_ylabel('Amplitude')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(chirp_sig, fs=fs2, nperseg=128)
im = axes[0,1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx+1e-12), shading='gouraud', cmap='inferno')
axes[0,1].set_ylabel('Frequenz [Hz]')
axes[0,1].set_xlabel('Zeit [s]')
axes[0,1].set_title('Spektrogramm (Chirp-Signal)')
plt.colorbar(im, ax=axes[0,1], label='Leistung [dB]')

# Welch PSD
f_w, Pxx = signal.welch(noisy_chirp, fs=fs2, nperseg=256)
axes[1,0].semilogy(f_w, Pxx)
axes[1,0].set_title('Welch-Leistungsspektrum')
axes[1,0].set_xlabel('Frequenz [Hz]'); axes[1,0].set_ylabel('PSD [V²/Hz]')
axes[1,0].grid(True, alpha=0.3)

# Filter frequency response
w, h = signal.freqz(b, a, worN=2048, fs=fs2)
axes[1,1].plot(w, 20*np.log10(np.abs(h)+1e-12))
axes[1,1].axvline(40, color='r', linestyle='--', label='Grenzfreq. 40 Hz')
axes[1,1].set_title('Frequenzgang Butterworth-Filter')
axes[1,1].set_xlabel('Frequenz [Hz]'); axes[1,1].set_ylabel('Verstärkung [dB]')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

fig.suptitle('SciPy – Signalverarbeitung', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_signal.pdf')
plt.close(fig)
print("Fig 3 done")

# ─────────────────────────────────────────────────────────
# Figure 4: SciPy – Optimization and root-finding
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Nonlinear optimization: Rosenbrock
from scipy.optimize import minimize, rosen
def rosen_2d(xy):
    return (1 - xy[0])**2 + 100*(xy[1] - xy[0]**2)**2

x_r = np.linspace(-2, 2, 400)
y_r = np.linspace(-1, 3, 400)
X_r, Y_r = np.meshgrid(x_r, y_r)
Z_r = rosen_2d([X_r, Y_r])
cs = axes[0].contourf(X_r, Y_r, np.log1p(Z_r), levels=40, cmap='viridis')
plt.colorbar(cs, ax=axes[0], label='log(1+f)')
result = minimize(rosen_2d, [-1.5, 0.5], method='L-BFGS-B')
axes[0].plot(*result.x, 'r*', markersize=14, label=f'Minimum ({result.x[0]:.2f},{result.x[1]:.2f})')
axes[0].set_title('Rosenbrock-Funktion (L-BFGS-B)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].legend(fontsize=8); 

# Curve fitting
def model_func(x, a, b, c):
    return a * np.exp(-b * x) * np.cos(c * x)

x_cf = np.linspace(0, 4*np.pi, 100)
y_true = model_func(x_cf, 3, 0.3, 2)
y_noisy = y_true + 0.2 * np.random.default_rng(3).normal(size=len(x_cf))
popt, pcov = optimize.curve_fit(model_func, x_cf, y_noisy, p0=[2, 0.5, 1.5])
axes[1].scatter(x_cf, y_noisy, s=5, alpha=0.7, label='Messdaten')
axes[1].plot(x_cf, model_func(x_cf, *popt), 'r-', lw=2, label=f'Fit: a={popt[0]:.2f}, b={popt[1]:.2f}')
axes[1].set_title('Kurvenanpassung (curve\\_fit)')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Root finding
def f_root(x):
    return x**3 - 2*x**2 - 5
x_rf = np.linspace(-2, 4, 400)
axes[2].plot(x_rf, f_root(x_rf), label='$f(x) = x^3 - 2x^2 - 5$')
axes[2].axhline(0, color='k', lw=0.8)
root = optimize.brentq(f_root, 2, 4)
axes[2].plot(root, 0, 'ro', markersize=10, label=f'Nullstelle $x^*={root:.4f}$')
axes[2].set_title('Nullstellensuche (Brent)')
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$f(x)$')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle('SciPy – Optimierung und Nullstellensuche', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_optimize.pdf')
plt.close(fig)
print("Fig 4 done")

# ─────────────────────────────────────────────────────────
# Figure 5: SciPy – Statistics
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

rng2 = np.random.default_rng(99)
sample1 = rng2.normal(5, 1.5, 200)
sample2 = rng2.normal(6, 2.0, 200)

# KDE
x_kde = np.linspace(0, 12, 400)
kde1 = stats.gaussian_kde(sample1)
kde2 = stats.gaussian_kde(sample2)
axes[0,0].hist(sample1, bins=30, density=True, alpha=0.4, label='Stichprobe 1')
axes[0,0].hist(sample2, bins=30, density=True, alpha=0.4, label='Stichprobe 2')
axes[0,0].plot(x_kde, kde1(x_kde), lw=2, label='KDE 1')
axes[0,0].plot(x_kde, kde2(x_kde), lw=2, label='KDE 2')
axes[0,0].set_title('Histogramm + Kernel-Dichteschätzung')
axes[0,0].set_xlabel('Wert'); axes[0,0].set_ylabel('Dichte')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# QQ-Plot
(osm, osr), (slope, intercept, r) = stats.probplot(sample1, dist='norm')
axes[0,1].plot(osm, osr, 'o', markersize=3, label='Daten')
axes[0,1].plot(osm, slope*np.array(osm)+intercept, 'r-', lw=2, label=f'Theoretisch ($r={r:.3f}$)')
axes[0,1].set_title('Q-Q-Plot (Normalverteilung)')
axes[0,1].set_xlabel('Theoretische Quantile'); axes[0,1].set_ylabel('Stichprobenquantile')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Confidence intervals bootstrap
n_boot = 2000
boot_means = np.array([np.mean(rng2.choice(sample1, size=len(sample1))) for _ in range(n_boot)])
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
axes[1,0].hist(boot_means, bins=50, edgecolor='none', alpha=0.8)
axes[1,0].axvline(ci_low,  color='r', linestyle='--', label=f'95%-KI [{ci_low:.2f}, {ci_high:.2f}]')
axes[1,0].axvline(ci_high, color='r', linestyle='--')
axes[1,0].axvline(np.mean(sample1), color='k', lw=2, label=f'Mittelwert {np.mean(sample1):.2f}')
axes[1,0].set_title('Bootstrap-Konfidenzintervall')
axes[1,0].set_xlabel('Bootstrap-Mittelwert'); axes[1,0].set_ylabel('Häufigkeit')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# Scipy continuous distributions
dists_info = [
    ('norm',  {'loc': 0,   'scale': 1},   'Normal(0,1)'),
    ('t',     {'df': 5},                  'Student-t(5)'),
    ('gamma', {'a': 2, 'scale': 1},       'Gamma(2,1)'),
    ('beta',  {'a': 2, 'b': 5},           'Beta(2,5)'),
]
x_d = np.linspace(-4, 8, 500)
for dist_name, params, label in dists_info:
    dist = getattr(stats, dist_name)
    pdf_vals = dist.pdf(x_d, **params)
    axes[1,1].plot(x_d[pdf_vals > 0.001], pdf_vals[pdf_vals > 0.001], label=label)
axes[1,1].set_title('Wahrscheinlichkeitsdichten (SciPy)')
axes[1,1].set_xlabel('$x$'); axes[1,1].set_ylabel('$f(x)$')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

fig.suptitle('SciPy – Statistik und Wahrscheinlichkeitsverteilungen', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_stats.pdf')
plt.close(fig)
print("Fig 5 done")

# ─────────────────────────────────────────────────────────
# Figure 6: SciPy – Integration and ODEs
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Numerical integration: visualise trapezoid vs. quad
def f_int(x):
    return np.exp(-x**2) * np.cos(4*x)

x_int = np.linspace(-3, 3, 500)
axes[0].fill_between(x_int, f_int(x_int), alpha=0.3)
axes[0].plot(x_int, f_int(x_int), lw=2)
val, _ = integrate.quad(f_int, -3, 3)
axes[0].set_title(f'Numerische Integration\n$\\int_{{-3}}^{{3}} = {val:.4f}$ (quad)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$f(x) = e^{-x^2}\\cos(4x)$')
axes[0].grid(True, alpha=0.3)

# ODE: Lotka-Volterra
def lotka_volterra(t, y, alpha, beta, delta, gamma):
    x_lv, y_lv = y
    dxdt = alpha*x_lv - beta*x_lv*y_lv
    dydt = delta*x_lv*y_lv - gamma*y_lv
    return [dxdt, dydt]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)
sol = integrate.solve_ivp(lotka_volterra, t_span, [10, 5], args=(1.5, 0.1, 0.075, 1.5),
                           t_eval=t_eval, method='RK45', rtol=1e-8)
axes[1].plot(sol.t, sol.y[0], label='Beute')
axes[1].plot(sol.t, sol.y[1], label='Räuber')
axes[1].set_title('Lotka-Volterra (solve\\_ivp RK45)')
axes[1].set_xlabel('Zeit'); axes[1].set_ylabel('Population')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Phase portrait
axes[2].plot(sol.y[0], sol.y[1], lw=1.2, alpha=0.8)
axes[2].set_title('Phasenportrait Lotka-Volterra')
axes[2].set_xlabel('Beute'); axes[2].set_ylabel('Räuber')
axes[2].grid(True, alpha=0.3)

fig.suptitle('SciPy – Numerische Integration und DGL', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_integrate.pdf')
plt.close(fig)
print("Fig 6 done")

# ─────────────────────────────────────────────────────────
# Figure 7: SciPy – Interpolation and special functions
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Interpolation
x_ip_pts = np.array([0, 1, 2, 3, 4, 5, 6])
y_ip_pts = np.array([0, 0.8, 0.9, 0.1, -0.8, -1, -0.4])
x_fine = np.linspace(0, 6, 400)
for kind, col in [('linear', 'tab:blue'), ('cubic', 'tab:orange'), ('quadratic', 'tab:green')]:
    f_ip = interpolate.interp1d(x_ip_pts, y_ip_pts, kind=kind)
    axes[0].plot(x_fine, f_ip(x_fine), label=kind, color=col)
axes[0].plot(x_ip_pts, y_ip_pts, 'ko', zorder=5, label='Datenpunkte')
axes[0].set_title('Spline-Interpolation (SciPy)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2D RBF interpolation
rng3 = np.random.default_rng(5)
n_pts = 60
x2d = rng3.uniform(-2, 2, n_pts)
y2d = rng3.uniform(-2, 2, n_pts)
z2d = np.sin(np.pi*x2d) * np.cos(np.pi*y2d)
from scipy.interpolate import RBFInterpolator
xi2d = np.column_stack([x2d, y2d])
rbf = RBFInterpolator(xi2d, z2d, kernel='thin_plate_spline')
xg = np.linspace(-2, 2, 100)
yg = np.linspace(-2, 2, 100)
XG, YG = np.meshgrid(xg, yg)
ZG = rbf(np.column_stack([XG.ravel(), YG.ravel()])).reshape(100, 100)
im2 = axes[1].pcolormesh(XG, YG, ZG, shading='gouraud', cmap='RdBu_r')
axes[1].scatter(x2d, y2d, c='k', s=10, zorder=5)
plt.colorbar(im2, ax=axes[1], label='$z$')
axes[1].set_title('RBF 2D-Interpolation')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')

# Special functions: Bessel, Legendre, Gamma
x_sf = np.linspace(0, 20, 500)
for n in range(4):
    axes[2].plot(x_sf, special.jv(n, x_sf), label=f'$J_{n}(x)$')
axes[2].axhline(0, color='k', lw=0.8)
axes[2].set_title('Bessel-Funktionen $J_n(x)$ (SciPy)')
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$J_n(x)$')
axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.5, 1.1)

fig.suptitle('SciPy – Interpolation und spezielle Funktionen', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_interp_special.pdf')
plt.close(fig)
print("Fig 7 done")

# ─────────────────────────────────────────────────────────
# Figure 8: Matplotlib – diverse plot types
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Polar plot
theta_p = np.linspace(0, 2*np.pi, 1000)
ax_pol = fig.add_subplot(2, 3, 1, polar=True)
fig.delaxes(axes[0, 0])
r_spiral = theta_p / (2*np.pi) * 3
r_rose  = np.cos(4*theta_p)
ax_pol.plot(theta_p, r_spiral, label='Archimedische Spirale')
ax_pol.plot(theta_p, np.abs(r_rose), '--', label='Rosenplot $|\\cos 4\\theta|$')
ax_pol.set_title('Polarkoordinaten', pad=15)
ax_pol.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

# Box & Violin
rng4 = np.random.default_rng(7)
data_box = [rng4.normal(i, 1, 100) for i in range(5)]
axes[0, 1].boxplot(data_box, notch=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[0, 1].set_title('Boxplot')
axes[0, 1].set_xlabel('Gruppe'); axes[0, 1].set_ylabel('Wert')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].violinplot(data_box, showmedians=True)
axes[0, 2].set_title('Violin Plot')
axes[0, 2].set_xlabel('Gruppe'); axes[0, 2].set_ylabel('Wert')
axes[0, 2].grid(True, alpha=0.3)

# Heatmap / Correlation matrix
data_mat = rng4.normal(size=(6, 200))
corr = np.corrcoef(data_mat)
im3 = axes[1, 0].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im3, ax=axes[1, 0], label='Korrelation')
labels_mat = [f'$X_{i}$' for i in range(1, 7)]
axes[1, 0].set_xticks(range(6)); axes[1, 0].set_xticklabels(labels_mat)
axes[1, 0].set_yticks(range(6)); axes[1, 0].set_yticklabels(labels_mat)
axes[1, 0].set_title('Korrelationsmatrix (Heatmap)')

# Scatter with colormap
N_sc = 200
x_sc = rng4.normal(0, 1, N_sc)
y_sc = x_sc * 0.8 + rng4.normal(0, 0.6, N_sc)
c_sc = x_sc**2 + y_sc**2
sc = axes[1, 1].scatter(x_sc, y_sc, c=c_sc, cmap='plasma', s=30, alpha=0.8)
plt.colorbar(sc, ax=axes[1, 1], label='$x^2+y^2$')
axes[1, 1].set_title('Scatter mit Farbkodierung')
axes[1, 1].set_xlabel('$x$'); axes[1, 1].set_ylabel('$y$')

# Stem plot
x_stem = np.linspace(-np.pi, np.pi, 25)
y_stem = np.sinc(x_stem)
axes[1, 2].stem(x_stem, y_stem)
axes[1, 2].set_title('Stem-Plot ($\\mathrm{sinc}$)')
axes[1, 2].set_xlabel('$x$'); axes[1, 2].set_ylabel('$\\mathrm{sinc}(x)$')
axes[1, 2].grid(True, alpha=0.3)

fig.suptitle('Matplotlib – Vielfalt der Diagrammtypen', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/matplotlib_types.pdf')
plt.close(fig)
print("Fig 8 done")

# ─────────────────────────────────────────────────────────
# Figure 9: 3D Visualization
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
x3 = np.linspace(-3, 3, 80)
y3 = np.linspace(-3, 3, 80)
X3, Y3 = np.meshgrid(x3, y3)
Z3 = np.sin(np.sqrt(X3**2 + Y3**2))
surf = ax1.plot_surface(X3, Y3, Z3, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, ax=ax1, shrink=0.5, label='$z$')
ax1.set_title('3D-Oberfläche $\\sin(\\sqrt{x^2+y^2})$')
ax1.set_xlabel('$x$'); ax1.set_ylabel('$y$'); ax1.set_zlabel('$z$')

# 3D parametric curve (helix)
ax2 = fig.add_subplot(132, projection='3d')
t3 = np.linspace(0, 4*np.pi, 500)
ax2.plot(np.cos(t3), np.sin(t3), t3/(4*np.pi), lw=2, color='steelblue')
ax2.set_title('3D-Helix (parametrisch)')
ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$'); ax2.set_zlabel('$z$')

# 3D scatter
ax3 = fig.add_subplot(133, projection='3d')
rng5 = np.random.default_rng(11)
xs = rng5.normal(0, 1, 200); ys = rng5.normal(0, 1, 200)
zs = xs**2 - ys**2 + rng5.normal(0, 0.5, 200)
sc3 = ax3.scatter(xs, ys, zs, c=zs, cmap='RdYlBu', s=20, alpha=0.7)
fig.colorbar(sc3, ax=ax3, shrink=0.5)
ax3.set_title('3D-Scatter mit Farbkodierung')
ax3.set_xlabel('$x$'); ax3.set_ylabel('$y$'); ax3.set_zlabel('$z$')

fig.suptitle('Matplotlib – 3D-Visualisierung', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/matplotlib_3d.pdf')
plt.close(fig)
print("Fig 9 done")

# ─────────────────────────────────────────────────────────
# Figure 10: Contour + Quiver + Streamplot
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

xv = np.linspace(-3, 3, 200)
yv = np.linspace(-3, 3, 200)
XV, YV = np.meshgrid(xv, yv)
ZV = XV * np.exp(-(XV**2 + YV**2))

cs2 = axes[0].contourf(XV, YV, ZV, levels=20, cmap='RdBu_r')
axes[0].contour(XV, YV,  ZV, levels=20, colors='k', linewidths=0.5, alpha=0.5)
plt.colorbar(cs2, ax=axes[0], label='$z$')
axes[0].set_title('Konturplot')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')

# Vector field
xq = np.linspace(-2, 2, 20)
yq = np.linspace(-2, 2, 20)
XQ, YQ = np.meshgrid(xq, yq)
U = -YQ; V = XQ
axes[1].quiver(XQ, YQ, U, V, np.sqrt(U**2+V**2), cmap='autumn', alpha=0.9)
axes[1].set_title('Quiver-Plot (Vektorfeld)')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
axes[1].set_aspect('equal')

# Streamplot
xs2 = np.linspace(-3, 3, 100)
ys2 = np.linspace(-3, 3, 100)
XS, YS = np.meshgrid(xs2, ys2)
US = -YS; VS = XS
speed = np.sqrt(US**2 + VS**2)
axes[2].streamplot(xs2, ys2, US, VS, color=speed, cmap='Blues', linewidth=1,
                   density=1.5, arrowsize=1.2)
axes[2].set_title('Stromlinienplot (Rotation)')
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$y$')

fig.suptitle('Matplotlib – Kontur-, Quiver- und Stromlinienplots', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/matplotlib_contour_quiver.pdf')
plt.close(fig)
print("Fig 10 done")

# ─────────────────────────────────────────────────────────
# Figure 11: SciPy Linear Algebra
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

rng6 = np.random.default_rng(21)
A_svd = rng6.normal(0, 1, (6, 6))
U_s, s_sv, Vt_s = linalg.svd(A_svd)
axes[0].bar(range(1, len(s_sv)+1), s_sv, color='steelblue')
axes[0].set_title('Singuläre Werte (SVD)')
axes[0].set_xlabel('Index'); axes[0].set_ylabel('$\\sigma_i$')
axes[0].grid(True, alpha=0.3)

# LU decomposition visualisation
P, L_lu, U_lu = linalg.lu(A_svd)
im4 = axes[1].imshow(np.abs(A_svd), cmap='Blues', aspect='auto')
plt.colorbar(im4, ax=axes[1], label='$|a_{ij}|$')
axes[1].set_title('Matrix $|A|$')

im5 = axes[2].imshow(np.abs(L_lu @ U_lu), cmap='Blues', aspect='auto')
plt.colorbar(im5, ax=axes[2], label='$|l_{ij}||u_{ij}|$')
axes[2].set_title('Rekonstruktion $|LU|$')

fig.suptitle('SciPy – Lineare Algebra (SVD, LU)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/scipy_linalg.pdf')
plt.close(fig)
print("Fig 11 done")

# ─────────────────────────────────────────────────────────
# Figure 12: Combined real-world: Heat equation (PDE)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Solve 1D heat equation with FFT (spectral method)
N_heat = 256
L_heat = 2 * np.pi
x_heat = np.linspace(0, L_heat, N_heat, endpoint=False)
alpha_heat = 0.01
dt_heat = 0.01
u0 = np.exp(-4*(x_heat - np.pi)**2)  # Gaussian initial condition

times_heat = [0, 1, 3, 5]
u = u0.copy()
k = np.fft.rfftfreq(N_heat, d=L_heat/(2*np.pi*N_heat))
u_hat = np.fft.rfft(u)

snap = {0: u0.copy()}
for step in range(int(max(times_heat)/dt_heat) + 10):
    t_now = step * dt_heat
    u_hat = np.fft.rfft(np.fft.irfft(u_hat))
    u_hat *= np.exp(-alpha_heat * k**2 * dt_heat)
    for tt in times_heat[1:]:
        if abs(t_now - tt) < dt_heat/2 and tt not in snap:
            snap[tt] = np.fft.irfft(u_hat).copy()

for tt, col in zip(times_heat, ['tab:blue','tab:orange','tab:green','tab:red']):
    if tt in snap:
        axes[0].plot(x_heat, snap[tt], label=f'$t={tt}$', color=col)
axes[0].set_title('1D-Wärmeleitungsgleichung (spektral)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$u(x,t)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2D Gaussian blur (image processing analogy)
img = np.zeros((100, 100))
img[30:70, 30:70] = 1.0
img[40:60, 40:60] = 2.0
for sigma, ax in zip([0, 3, 8], axes):
    sm = gaussian_filter(img, sigma=sigma)
    ax.imshow(sm, cmap='hot', vmin=0, vmax=2)
    ax.set_title(f'Gauss-Glättung $\\sigma={sigma}$')
    ax.axis('off')

fig.suptitle('NumPy/SciPy – PDE und Bildverarbeitung', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/pde_image.pdf')
plt.close(fig)
print("Fig 12 done")

# ─────────────────────────────────────────────────────────
# Figure 13: Matplotlib – colormaps and styles
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
cmaps = ['viridis', 'plasma', 'inferno', 'magma',
         'cividis', 'RdBu_r', 'coolwarm', 'spectral' if 'spectral' in plt.colormaps() else 'nipy_spectral']
x_cm = np.linspace(-np.pi, np.pi, 100)
y_cm = np.linspace(-np.pi, np.pi, 100)
XC, YC = np.meshgrid(x_cm, y_cm)
ZC = np.sin(XC) * np.cos(YC)
for ax, cmap in zip(axes.ravel(), cmaps):
    ax.imshow(ZC, cmap=cmap, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi])
    ax.set_title(cmap, fontsize=9)
    ax.axis('off')
fig.suptitle('Matplotlib – Farbpaletten (Colormaps)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/matplotlib_colormaps.pdf')
plt.close(fig)
print("Fig 13 done")

# ─────────────────────────────────────────────────────────
# Figure 14: Real-world application – Pendulum & FFT
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Damped pendulum ODE
def pendulum_ode(t, y, b, g, l):
    theta, omega = y
    return [omega, -(b/l)*omega - (g/l)*np.sin(theta)]

t_pend = np.linspace(0, 30, 3000)
b_vals = [0.0, 0.5, 1.5]
for b_v, col in zip(b_vals, ['tab:blue', 'tab:orange', 'tab:red']):
    sol_p = integrate.solve_ivp(pendulum_ode, (0, 30), [np.pi/3, 0],
                                 args=(b_v, 9.81, 1.0), t_eval=t_pend,
                                 method='DOP853', rtol=1e-9)
    axes[0].plot(sol_p.t, sol_p.y[0], label=f'$b={b_v}$', color=col)
axes[0].set_title('Gedämpftes Pendel')
axes[0].set_xlabel('Zeit [s]'); axes[0].set_ylabel('Winkel [rad]')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Lorenz attractor
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x_l, y_l, z_l = state
    return [sigma*(y_l - x_l), x_l*(rho - z_l) - y_l, x_l*y_l - beta*z_l]

sol_l = integrate.solve_ivp(lorenz, (0, 50), [1, 1, 1], max_step=0.01,
                              t_eval=np.linspace(0, 50, 20000), method='RK45')
axes[1].plot(sol_l.y[0], sol_l.y[2], lw=0.3, alpha=0.8, color='navy')
axes[1].set_title('Lorenz-Attraktor ($xz$-Projektion)')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$z$')
axes[1].grid(True, alpha=0.3)

# Power spectrum of Lorenz x-component
f_lor = np.fft.rfftfreq(len(sol_l.t), d=50/20000)
P_lor = np.abs(np.fft.rfft(sol_l.y[0]))**2
axes[2].semilogy(f_lor[1:2000], P_lor[1:2000])
axes[2].set_title('Leistungsspektrum (Lorenz $x$)')
axes[2].set_xlabel('Frequenz'); axes[2].set_ylabel('Leistung')
axes[2].grid(True, alpha=0.3)

fig.suptitle('DGL-Anwendungen: Gedämpftes Pendel und Lorenz-Attraktor', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/pendulum_lorenz.pdf')
plt.close(fig)
print("Fig 14 done")

# ─────────────────────────────────────────────────────────
# Figure 15: Matplotlib – Annotations and error bars
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Error bars with fill_between
x_eb = np.linspace(0, 4*np.pi, 60)
y_mean = np.sin(x_eb) * np.exp(-0.1*x_eb)
y_std  = 0.1 + 0.05*np.abs(np.cos(x_eb))
axes[0].fill_between(x_eb, y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.25, label='$2\\sigma$')
axes[0].fill_between(x_eb, y_mean - y_std,   y_mean + y_std,   alpha=0.4,  label='$1\\sigma$')
axes[0].plot(x_eb, y_mean, lw=2, label='Mittelwert')
axes[0].set_title('Unsicherheitsband ($\\pm\\sigma$, $\\pm 2\\sigma$)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Rich annotation example
x_an = np.linspace(0, 10, 300)
y_an = np.sin(x_an)
ax_an = axes[1]
ax_an.plot(x_an, y_an, lw=2)
max_idx = np.argmax(y_an)
ax_an.annotate('Maximum', xy=(x_an[max_idx], y_an[max_idx]),
                xytext=(x_an[max_idx]+1.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
min_idx = np.argmin(y_an)
ax_an.annotate('Minimum', xy=(x_an[min_idx], y_an[min_idx]),
                xytext=(x_an[min_idx]-2.5, -0.5),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
                fontsize=10, color='darkblue')
ax_an.set_title('Matplotlib – Annotationen')
ax_an.set_xlabel('$x$'); ax_an.set_ylabel('$\\sin(x)$')
ax_an.grid(True, alpha=0.3)

fig.suptitle('Matplotlib – Fehlerbänder und Annotationen', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/matplotlib_annotations.pdf')
plt.close(fig)
print("Fig 15 done")

print("\nAll figures generated successfully!")
