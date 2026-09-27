"""Generate all missing PDF figures for signalth.tex"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

OUT = 'signalth/science"

# --------------------------------------------------------------------------
# Figure 1: rl_circuit.pdf
# RL circuit: current vs. time  R=10, L=0.1, tau=0.01 s, U0=5V -> I_inf=0.5A
# --------------------------------------------------------------------------
def make_rl_circuit():
    R, L, U0 = 10, 0.1, 5.0
    tau = L / R          # 0.01 s
    t = np.linspace(0, 5 * tau, 1000)
    i = (U0 / R) * (1 - np.exp(-t / tau))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(t * 1000, i, 'b-', linewidth=2, label=r'$i(t)=\frac{U_0}{R}(1-e^{-t/\tau})$')
    ax1.axhline(U0 / R, color='r', linestyle='--', linewidth=1.5, label=r'$i_\infty=0.5\,\mathrm{A}$')
    ax1.axvline(tau * 1000, color='g', linestyle=':', linewidth=1.5, label=rf'$\tau={tau*1000:.1f}\,\mathrm{{ms}}$')
    ax1.scatter([tau * 1000], [(U0 / R) * (1 - np.exp(-1))], color='g', zorder=5, s=60)
    ax1.annotate('63.2 %', xy=(tau * 1000, (U0 / R) * (1 - np.exp(-1))),
                 xytext=(tau * 1000 * 1.5, 0.38), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax1.set_xlabel('Zeit $t$ [ms]')
    ax1.set_ylabel('Strom $i(t)$ [A]')
    ax1.set_title('RL-Stromkreis: Stromanstieg')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # Energy stored in inductor
    W_L = 0.5 * L * i**2
    P_R = R * i**2
    ax2.plot(t * 1000, W_L * 1000, 'b-', linewidth=2, label=r'$W_L(t)$ [mJ]')
    ax2.plot(t * 1000, P_R, 'r-', linewidth=2, label=r'$P_R(t)$ [W]')
    ax2.set_xlabel('Zeit $t$ [ms]')
    ax2.set_ylabel('Energie / Leistung')
    ax2.set_title('Energie und Leistung')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.suptitle('RL-Stromkreis: Experimentelle Validierung ($R=10\,\Omega$, $L=0.1\,\mathrm{H}$, $U_0=5\,\mathrm{V}$)', fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{OUT}/rl_circuit.pdf", bbox_inches='tight')
    plt.close(fig)
    print("rl_circuit.pdf created")


# --------------------------------------------------------------------------
# Figure 2: rlc_circuit.pdf
# RLC circuit: three damping cases + pole diagram
# L=0.1H, C=1e-4F, R in {5, 20, 40}
# --------------------------------------------------------------------------
def make_rlc_circuit():
    L, C = 0.1, 1e-4
    Rs = [5, 20, 40]
    colors = ['b', 'g', 'r']
    t = np.linspace(0, 0.08, 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for R, col in zip(Rs, colors):
        sigma = R / (2 * L)
        omega0 = 1 / np.sqrt(L * C)
        discriminant = sigma**2 - omega0**2
        zeta = R / 2 * np.sqrt(C / L)
        if discriminant < 0:
            omega_d = np.sqrt(omega0**2 - sigma**2)
            # Step response of underdamped RLC
            # voltage across capacitor from unit step
            # v_C(t) = 1 - e^(-sigma*t)*(cos(omega_d*t) + sigma/omega_d * sin(omega_d*t))
            v = 1 - np.exp(-sigma * t) * (np.cos(omega_d * t) + sigma / omega_d * np.sin(omega_d * t))
        else:
            s1 = -sigma + np.sqrt(discriminant)
            s2 = -sigma - np.sqrt(discriminant)
            A1 = s2 / (s2 - s1)
            A2 = -s1 / (s2 - s1)
            v = 1 + A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
        ax1.plot(t * 1000, v, color=col, linewidth=2,
                 label=rf'$R={R}\,\Omega$, $\zeta={zeta:.4f}$')

    ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Zeit $t$ [ms]')
    ax1.set_ylabel('Kondensatorspannung $U_C/U_0$')
    ax1.set_title('RLC-Stromkreis: Drei Dämpfungsfälle')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(-0.3, 1.5)

    # Pole diagram
    omega0 = 1 / np.sqrt(L * C)
    for R, col, marker in zip(Rs, colors, ['o', 's', '^']):
        sigma = R / (2 * L)
        discriminant = sigma**2 - omega0**2
        if discriminant < 0:
            omega_d = np.sqrt(omega0**2 - sigma**2)
            poles = [complex(-sigma, omega_d), complex(-sigma, -omega_d)]
        else:
            s1 = -sigma + np.sqrt(discriminant)
            s2 = -sigma - np.sqrt(discriminant)
            poles = [s1, s2]
        for p in poles:
            ax2.scatter(p.real, p.imag, color=col, marker=marker, s=80, zorder=5,
                       label=rf'$R={R}\,\Omega$' if p.imag >= 0 else '')

    ax2.axvline(0, color='k', linewidth=0.8)
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_xlabel(r'Re$(s)$')
    ax2.set_ylabel(r'Im$(s)$')
    ax2.set_title('Pole in der komplexen Ebene')
    ax2.grid(True, alpha=0.4)
    handles, labels = ax2.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax2.legend(unique.values(), unique.keys(), fontsize=9)

    plt.suptitle('RLC-Stromkreis ($L=0.1\,\mathrm{H}$, $C=10^{-4}\,\mathrm{F}$)', fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{OUT}/rlc_circuit.pdf", bbox_inches='tight')
    plt.close(fig)
    print("rlc_circuit.pdf created")


# --------------------------------------------------------------------------
# Figure 3: fourier_analysis.pdf
# FFT of three signals
# --------------------------------------------------------------------------
def make_fourier_analysis():
    fs = 10000  # sample rate
    T_dur = 0.5
    t = np.arange(0, T_dur, 1 / fs)
    freqs = np.fft.rfftfreq(len(t), 1 / fs)

    sig1 = np.sin(2 * np.pi * 100 * t)
    sig2 = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    sig3 = np.sin(2 * np.pi * (100 + 50 * t) * t)  # chirp

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    titles = ['Einzelne Sinuswelle (100 Hz)',
              'Summe von Sinuswellen (100 + 300 Hz)',
              'Chirp-Signal (100–150 Hz)']
    sigs = [sig1, sig2, sig3]

    for row, (s, title) in enumerate(zip(sigs, titles)):
        t_show = t[t < 0.05]
        ax_t = axes[row, 0]
        ax_f = axes[row, 1]

        ax_t.plot(t_show * 1000, s[:len(t_show)], 'b-', linewidth=1.2)
        ax_t.set_xlabel('Zeit [ms]')
        ax_t.set_ylabel('Amplitude')
        ax_t.set_title(f'{title}\n(Zeitbereich)')
        ax_t.grid(True, alpha=0.4)

        N = len(s)
        S = np.abs(np.fft.rfft(s)) / N * 2
        mask = freqs <= 500
        ax_f.plot(freqs[mask], S[mask], 'r-', linewidth=1.5)
        ax_f.set_xlabel('Frequenz [Hz]')
        ax_f.set_ylabel('|X(f)|')
        ax_f.set_title(f'{title}\n(Frequenzbereich)')
        ax_f.grid(True, alpha=0.4)

    plt.suptitle('Fourier-Analyse: Signalzerlegung in Frequenzkomponenten', fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fourier_analysis.pdf", bbox_inches='tight')
    plt.close(fig)
    print("fourier_analysis.pdf created")


# --------------------------------------------------------------------------
# Figure 4: filter_design.pdf
# Butterworth, Chebyshev I, Elliptic lowpass, fc=100 Hz, order=4
# --------------------------------------------------------------------------
def make_filter_design():
    fs = 10000
    fc = 100
    order = 4
    wn = fc / (fs / 2)  # Normalized cutoff

    filters = {
        'Butterworth': signal.butter(order, wn, btype='low', output='sos'),
        'Chebyshev I (1 dB)': signal.cheby1(order, 1, wn, btype='low', output='sos'),
        'Elliptic (1 dB / 40 dB)': signal.ellip(order, 1, 40, wn, btype='low', output='sos'),
    }
    colors = ['b', 'g', 'r']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for (name, sos), col in zip(filters.items(), colors):
        b, a = signal.sos2tf(sos)
        w, h = signal.freqz(b, a, worN=4096, fs=fs)
        ax1.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=col, linewidth=2, label=name)
        ax2.plot(w, np.unwrap(np.angle(h)) * 180 / np.pi, color=col, linewidth=2, label=name)

    ax1.axvline(fc, color='k', linestyle='--', linewidth=1, label=f'$f_c={fc}$ Hz')
    ax1.axhline(-3, color='gray', linestyle=':', linewidth=1, label='−3 dB')
    ax1.set_xlim(0, 500)
    ax1.set_ylim(-80, 5)
    ax1.set_xlabel('Frequenz [Hz]')
    ax1.set_ylabel('Betrag [dB]')
    ax1.set_title('Amplitudengang')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    ax2.axvline(fc, color='k', linestyle='--', linewidth=1, label=f'$f_c={fc}$ Hz')
    ax2.set_xlim(0, 500)
    ax2.set_xlabel('Frequenz [Hz]')
    ax2.set_ylabel('Phase [°]')
    ax2.set_title('Phasengang')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.suptitle('Filterdesign: Tiefpass-Filter ($f_c=100\,\mathrm{Hz}$, Ordnung 4)', fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{OUT}/filter_design.pdf", bbox_inches='tight')
    plt.close(fig)
    print("filter_design.pdf created")


# --------------------------------------------------------------------------
# Figure 5: network_propagation.pdf
# Signal propagation in 3 network types
# --------------------------------------------------------------------------
def make_network_propagation():
    rng = np.random.default_rng(42)
    n = 10  # nodes
    t_steps = 50

    def make_random_adj(n, p=0.3):
        A = (rng.random((n, n)) < p).astype(float)
        np.fill_diagonal(A, 0)
        A = (A + A.T) / 2
        # Normalize so spectral radius < 1
        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        return A / (rho + 0.05)

    def make_scalefree_adj(n):
        # Simple preferential attachment approximation
        A = np.zeros((n, n))
        degrees = np.ones(n)
        for i in range(2, n):
            probs = degrees[:i] / degrees[:i].sum()
            j = rng.choice(i, p=probs)
            A[i, j] = A[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1
        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        return A / (rho + 0.05)

    def make_smallworld_adj(n, k=2):
        # Ring lattice + rewiring
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(1, k + 1):
                A[i, (i + j) % n] = 1
                A[(i + j) % n, i] = 1
        # Rewire with prob 0.2
        for i in range(n):
            for j in range(1, k + 1):
                if rng.random() < 0.2:
                    new_j = rng.integers(0, n)
                    if new_j != i:
                        A[i, (i + j) % n] = 0
                        A[(i + j) % n, i] = 0
                        A[i, new_j] = 1
                        A[new_j, i] = 1
        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        return A / (rho + 0.05)

    networks = {
        'Zufälliges Netzwerk': make_random_adj(n),
        'Scale-Free-Netzwerk': make_scalefree_adj(n),
        'Small-World-Netzwerk': make_smallworld_adj(n),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors_net = ['b', 'g', 'r']
    for ax, (name, A), col in zip(axes, networks.items(), colors_net):
        x0 = np.zeros(n)
        x0[0] = 1.0  # initial impulse at node 0
        traj = np.zeros((t_steps, n))
        traj[0] = x0
        for k in range(1, t_steps):
            traj[k] = A @ traj[k - 1]
        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        for node in range(min(n, 5)):
            ax.plot(traj[:, node], alpha=0.7, linewidth=1.2,
                    label=f'Knoten {node+1}' if node < 3 else None)
        ax.set_title(f'{name}\n$\\rho(A)={rho:.4f}$')
        ax.set_xlabel('Zeitschritt')
        ax.set_ylabel('Signalstärke')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    plt.suptitle('Signalausbreitung in Netzwerken', fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/network_propagation.pdf", bbox_inches='tight')
    plt.close(fig)
    print("network_propagation.pdf created")


# --------------------------------------------------------------------------
# Figure 6: basilar_membrane.pdf
# Basilar membrane tonotopic organization
# --------------------------------------------------------------------------
def make_basilar_membrane():
    f0 = 20000   # Hz at basal end
    alpha = 0.06  # mm^-1
    L_total = 35  # mm

    x = np.linspace(0, L_total, 1000)
    f_x = f0 * np.exp(-alpha * x)  # characteristic frequency at position x

    test_freqs = [100, 500, 1000, 5000, 10000, 20000]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(test_freqs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: tonotopic map
    ax1.semilogy(x, f_x, 'k-', linewidth=2, label=r'$f(x)=f_0 e^{-\alpha x}$')
    for f, col in zip(test_freqs, colors):
        x_peak = -np.log(f / f0) / alpha
        if 0 <= x_peak <= L_total:
            ax1.axhline(f, color=col, linestyle='--', alpha=0.7, linewidth=1)
            ax1.axvline(x_peak, color=col, linestyle=':', alpha=0.7, linewidth=1)
            ax1.scatter([x_peak], [f], color=col, s=60, zorder=5)
            label = f'{f} Hz → {x_peak:.1f} mm'
            ax1.annotate(label, xy=(x_peak, f), xytext=(x_peak + 1, f * 1.3),
                         fontsize=7.5, color=col)
    ax1.set_xlabel('Position $x$ [mm]\n(0 = basal, 35 = apikal)')
    ax1.set_ylabel('Charakteristische Frequenz $f(x)$ [Hz]')
    ax1.set_title('Tonotopische Karte der Basilarmembran')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4, which='both')

    # Right: membrane response profiles
    r = 0.05 * L_total  # half-width of response
    for f, col in zip(test_freqs, colors):
        x_peak = -np.log(f / f0) / alpha
        if 0 <= x_peak <= L_total:
            # Gaussian-like amplitude profile
            amp = np.exp(-((x - x_peak) ** 2) / (2 * r**2))
            ax2.plot(x, amp, color=col, linewidth=1.8, label=f'{f} Hz')
    ax2.set_xlabel('Position $x$ [mm]')
    ax2.set_ylabel('Normierte Membranauslenkung $|Y(x)|$')
    ax2.set_title('Frequenzspezifische Auslenkungsprofile')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.4)

    plt.suptitle('Basilarmembran: Tonotopische Organisation und biologische Fourier-Transformation', fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{OUT}/basilar_membrane.pdf", bbox_inches='tight')
    plt.close(fig)
    print("basilar_membrane.pdf created")


if __name__ == '__main__':
    make_rl_circuit()
    make_rlc_circuit()
    make_fourier_analysis()
    make_filter_design()
    make_network_propagation()
    make_basilar_membrane()
    print("All figures generated successfully.")
