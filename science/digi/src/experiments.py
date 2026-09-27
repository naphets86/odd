"""
experiments.py – Comprehensive experiments from "Von der Diskretion"
=====================================================================
Runs all experiments, writes CSV tables and SVG plots to src/results/.

Usage (from project root, with the venv active)::

    python src/experiments.py

Each experiment produces:
  • one SVG figure with **two sub-plots**
  • one CSV file with the underlying data

Experiments
-----------
01  Harmonic signal and its derivative / frequency spectrum
02  Nyquist sampling and aliasing
03  Quantization error and SNR vs bit depth
04  Acoustic pressure wave and SPL reference scale
05  Basilar membrane step response – cat vs dog damping
06  RLC circuit: impedance and bandpass transfer function
07  Hodgkin-Huxley action potential & gating variables
08  Tonotopic maps cat vs dog + gammatone impulse responses
09  Shannon capacity landscape – cat / dog / human
10  Membrane optimality – multi-metric Ω comparison
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure "src/" is on the search path so that "import digi" works whether
# the package is installed in dev-mode or the script is run directly.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed – pure file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from digi.signals import harmonic, harmonic_derivative, spectrum_magnitude
from digi.sampling import (
    nyquist_rate, sample, sinc_reconstruct, alias_frequency,
)
from digi.quantization import quantize, snr_max_db, quantization_step
from digi.acoustics import pressure_wave, spl, P_REF, C_SOUND_AIR
from digi.mechanics import natural_frequency, damping_ratio, step_response_membrane
from digi.rlc import (
    resonance_frequency, quality_factor, impedance_magnitude, bandpass_magnitude,
    step_response as rlc_step_response,
)
from digi.neuron import (
    hodgkin_huxley, leaky_integrate_and_fire,
)
from digi.basilar import (
    tonotopic_frequency, tonotopic_position,
    gammatone_impulse_response,
    active_quality_factor,
    snr_gain_db,
)
from digi.hearing import (
    HearingProfile, CAT, DOG, HUMAN,
    shannon_capacity, membrane_efficiency_omega, compare_profiles,
)
from digi.information import capacity_over_bandwidth

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = _SRC / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
_CAT_COLOR   = "#E05A2B"   # warm orange-red
_DOG_COLOR   = "#3A7FC1"   # medium blue
_HU_COLOR    = "#5BAD6F"   # muted green
_GRAY        = "#888888"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
})

# ---------------------------------------------------------------------------
# Helper: tonotopic β computed from physical endpoints
# ---------------------------------------------------------------------------

def _beta(profile: HearingProfile) -> float:
    """Return the correct tonotopic gradient β [mm⁻¹] consistent with
    f_base = f_apex · exp(β · L_membrane)."""
    return math.log(profile.f_max / profile.f_apex) / profile.L_membrane


def _save_csv(filename: str, rows: list[list], header: list[str]) -> Path:
    path = RESULTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    path = RESULTS_DIR / filename
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Experiment 01 – Harmonic signal, derivative, frequency spectrum
# ===========================================================================

def exp01_harmonic_derivative() -> None:
    """Sinusoidal signal, its analytical derivative, and magnitude spectrum."""
    t = np.linspace(0, 1.0, 4000)
    f0, A = 5.0, 2.0
    x = harmonic(t, amplitude=A, frequency=f0)
    dx = harmonic_derivative(t, amplitude=A, frequency=f0)
    dt_val = float(t[1] - t[0])
    freqs, mag = spectrum_magnitude(x, dt_val)

    # CSV
    rows = list(zip(t.round(6), x.round(6), dx.round(6)))
    _save_csv("exp01_harmonic.csv", rows, ["t_s", "x", "dx_dt"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 01 – Harmonic Signal and Frequency Spectrum", fontweight="bold")

    ax1.plot(t, x,  color=_CAT_COLOR, label=r"$x(t)=A\sin(2\pi f_0 t)$")
    ax1.plot(t, dx, color=_DOG_COLOR, label=r"$\dot{x}(t)=A\omega_0\cos(2\pi f_0 t)$",
             linestyle="--")
    ax1.axhline(0, color=_GRAY, linewidth=0.6)
    ax1.set_xlim(0, 0.6)
    ax1.set_xlabel("Zeit t [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Signal und Ableitung  (f₀ = {f0} Hz, A = {A})")
    ax1.legend()

    # Show only positive frequencies
    pos = freqs >= 0
    ax2.stem(freqs[pos], mag[pos], linefmt=_CAT_COLOR, markerfmt=f"o",
             basefmt=_GRAY)
    ax2.set_xlim(-0.5, 20)
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylabel("|X(f)|")
    ax2.set_title("Betragsspektrum (einseitig)")

    fig.tight_layout()
    _save_fig(fig, "exp01_harmonic.svg")
    print("  exp01 ✓")


# ===========================================================================
# Experiment 02 – Nyquist sampling and aliasing
# ===========================================================================

def exp02_sampling_aliasing() -> None:
    """Perfect reconstruction vs. aliasing demonstration."""
    f_signal = 5.0       # Hz – signal to be sampled
    f_alias  = 90.0      # Hz – alias signal
    fs_good  = 100.0     # Hz – satisfies Nyquist for both
    fs_bad   = 20.0      # Hz – aliases 90 Hz → 10 Hz

    t_cont = np.linspace(0, 0.5, 2000)
    x_cont = np.sin(2 * np.pi * f_signal * t_cont)

    # Good sampling
    t_s, x_s = sample(lambda t: np.sin(2 * np.pi * f_signal * t),
                      0.0, 0.5, fs=fs_good)
    t_rec = np.linspace(min(t_s) + 0.02, max(t_s) - 0.02, 500)
    x_rec = sinc_reconstruct(x_s, t_s, t_rec)

    # Aliasing: 90 Hz signal sampled at 20 Hz → apparent frequency
    t_alias = np.linspace(0, 0.5, 2000)
    x_alias_true = np.sin(2 * np.pi * f_alias * t_alias)
    t_alias_s, x_alias_s = sample(lambda t: np.sin(2 * np.pi * f_alias * t),
                                   0.0, 0.5, fs=fs_bad)
    f_apparent = alias_frequency(f_alias, fs_bad)
    x_apparent = np.sin(2 * np.pi * f_apparent * t_alias)

    # CSV
    rows = [[round(float(t), 6), round(float(a), 6), round(float(b), 6)]
            for t, a, b in zip(t_alias, x_alias_true, x_apparent)]
    _save_csv("exp02_aliasing.csv", rows,
              [f"t_s", f"x_true_{f_alias}Hz", f"x_apparent_{f_apparent}Hz"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 02 – Nyquist-Abtastung und Aliasing", fontweight="bold")

    ax1.plot(t_cont, x_cont, color=_GRAY, linewidth=0.8, label="Original (5 Hz)")
    ax1.stem(t_s, x_s, linefmt=_CAT_COLOR, markerfmt="o", basefmt=_GRAY,
             label=f"Abtastwerte (fs={fs_good} Hz)")
    ax1.plot(t_rec, x_rec, color=_DOG_COLOR, linestyle="--",
             label="Rekonstruktion (sinc)")
    ax1.set_xlim(0, 0.5)
    ax1.set_xlabel("Zeit [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Perfekte Rekonstruktion  (f₀={f_signal} Hz, fs={fs_good} Hz)")
    ax1.legend(loc="upper right")

    ax2.plot(t_alias, x_alias_true, color=_GRAY, linewidth=0.8,
             label=f"Original ({f_alias} Hz)")
    ax2.stem(t_alias_s, x_alias_s, linefmt=_CAT_COLOR, markerfmt="o",
             basefmt=_GRAY, label=f"Abtastwerte (fs={fs_bad} Hz)")
    ax2.plot(t_alias, x_apparent, color=_DOG_COLOR, linestyle="--",
             label=f"Apparent alias ({f_apparent:.0f} Hz)")
    ax2.set_xlim(0, 0.5)
    ax2.set_xlabel("Zeit [s]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Aliasing: {f_alias} Hz bei fs={fs_bad} Hz → {f_apparent} Hz")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    _save_fig(fig, "exp02_aliasing.svg")
    print("  exp02 ✓")


# ===========================================================================
# Experiment 03 – Quantization error and SNR vs. bit depth
# ===========================================================================

def exp03_quantization_snr() -> None:
    """3-bit quantization of a sine wave; theoretical SNR formula."""
    # Subplot 1: quantization
    n_bits_demo = 3
    t = np.linspace(0, 1.0, 500)
    x = np.sin(2 * np.pi * 3.0 * t)
    x_q = quantize(x, n_bits=n_bits_demo, x_min=-1.0, x_max=1.0)
    err = x - x_q

    # Subplot 2: SNR vs N
    bits = np.arange(1, 25)
    snr_theory = np.array([snr_max_db(n) for n in bits])
    step_sizes  = np.array([quantization_step(n, -1.0, 1.0) for n in bits])

    # CSV
    rows_snr = [[int(n), round(s, 3), round(d, 8)]
                for n, s, d in zip(bits, snr_theory, step_sizes)]
    _save_csv("exp03_quantization_snr.csv", rows_snr,
              ["n_bits", "SNR_max_dB", "step_size"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 03 – Quantisierung und SNR", fontweight="bold")

    ax1.plot(t, x,   color=_GRAY,     linewidth=0.8, label="Original")
    ax1.step(t, x_q, color=_CAT_COLOR, linewidth=1.2, label=f"{n_bits_demo}-Bit quantisiert")
    ax1.fill_between(t, 0, err * 5, alpha=0.35, color=_DOG_COLOR,
                     label="Fehler × 5")
    ax1.set_xlabel("Zeit [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{n_bits_demo}-Bit Quantisierung (×5 Fehler)")
    ax1.legend()

    ax2.plot(bits, snr_theory, color=_CAT_COLOR, marker="o", markersize=3)
    ax2.axhline(96.3, color=_DOG_COLOR, linestyle="--",
                label="16-Bit CD (96.3 dB)")
    ax2.axhline(144.5, color=_HU_COLOR, linestyle=":",
                label="24-Bit Studio (144.5 dB)")
    ax2.set_xlabel("Bit-Tiefe N")
    ax2.set_ylabel("SNR_max [dB]")
    ax2.set_title("Theoretisches SNR: 6.02·N + 1.76 dB")
    ax2.legend()
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _save_fig(fig, "exp03_quantization.svg")
    print("  exp03 ✓")


# ===========================================================================
# Experiment 04 – Acoustic pressure wave and SPL scale
# ===========================================================================

def exp04_acoustic_pressure_spl() -> None:
    """Pressure wave snapshot and annotated SPL reference scale."""
    f_hz = 440.0          # A4
    p0 = 1.0              # Pa (amplitude)
    x_arr = np.linspace(0, 2.0, 800)   # metres
    t0 = 0.0
    wave = pressure_wave(x_arr, t0, p0=p0, frequency=f_hz, c=C_SOUND_AIR)

    # SPL reference levels (dB re 20 µPa)
    spl_levels = {
        "Hörschwelle (0 dB)":  20e-6,
        "Flüstern (30 dB)":    20e-3,
        "Gespräch (60 dB)":    0.020,
        "Straßenlärm (80 dB)": 0.200,
        "Presslufthammer (100 dB)": 2.0,
        "Schmerzschwelle (130 dB)": 63.2,
    }
    labels  = list(spl_levels.keys())
    p_vals  = list(spl_levels.values())
    db_vals = [spl(p) for p in p_vals]

    # CSV
    rows_wave = [[round(float(x), 5), round(float(pr), 6)]
                 for x, pr in zip(x_arr, wave)]
    _save_csv("exp04_acoustic_pressure.csv", rows_wave, ["x_m", "p_Pa"])

    rows_spl = [[lbl, round(pv, 8), round(db, 2)]
                for lbl, pv, db in zip(labels, p_vals, db_vals)]
    _save_csv("exp04_spl_levels.csv", rows_spl, ["description", "p_Pa", "SPL_dB"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 04 – Schalldruck und Schalldruckpegel", fontweight="bold")

    ax1.plot(x_arr, wave, color=_CAT_COLOR)
    ax1.axhline(1.0, color=_GRAY, linewidth=0.6, linestyle="--")
    ax1.axhline(-1.0, color=_GRAY, linewidth=0.6, linestyle="--")
    ax1.set_xlabel("Ort x [m]")
    ax1.set_ylabel("Schalldruck p [Pa]")
    ax1.set_title(f"Druckwelle: A4 ({f_hz} Hz), t = {t0} s")

    y_pos = range(len(labels))
    bars = ax2.barh(list(y_pos), db_vals, color=_CAT_COLOR, alpha=0.75)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("Schalldruckpegel [dB re 20 µPa]")
    ax2.set_title("SPL-Referenzpegel")
    ax2.axvline(0,   color=_GRAY, linewidth=0.6)
    ax2.axvline(130, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    fig.tight_layout()
    _save_fig(fig, "exp04_acoustics.svg")
    print("  exp04 ✓")


# ===========================================================================
# Experiment 05 – Basilar membrane step response: cat vs dog damping
# ===========================================================================

def exp05_membrane_step_response() -> None:
    """Step response and frequency response for cat-like vs dog-like membrane."""
    f0 = 1000.0          # resonance frequency [Hz]
    omega_0 = 2 * np.pi * f0

    # Cat: high Q → low damping ratio D ≈ 1/(2Q)
    D_cat = 1 / (2 * CAT.Q_mean)     # ≈ 0.029
    D_dog = 1 / (2 * DOG.Q_mean)     # ≈ 0.071

    t = np.linspace(0, 5e-3, 2000)   # 5 ms

    resp_cat = step_response_membrane(t, omega_0, D_cat, U0=1.0)
    resp_dog = step_response_membrane(t, omega_0, D_dog, U0=1.0)

    # Frequency response magnitude (band around resonance)
    freqs = np.linspace(100, 5000, 2000)
    omega = 2 * np.pi * freqs
    # |H(jω)| = ω₀² / √((ω₀²-ω²)² + (2Dω₀ω)²)
    def freq_resp(om0, D_val, w):
        return om0**2 / np.sqrt((om0**2 - w**2)**2 + (2 * D_val * om0 * w)**2)

    H_cat = freq_resp(omega_0, D_cat, omega)
    H_dog = freq_resp(omega_0, D_dog, omega)

    # CSV
    rows = [[round(tt * 1e3, 5), round(float(rc), 6), round(float(rd), 6)]
            for tt, rc, rd in zip(t, resp_cat, resp_dog)]
    _save_csv("exp05_step_response.csv", rows,
              ["t_ms", "u_cat_D{:.3f}".format(D_cat), "u_dog_D{:.3f}".format(D_dog)])

    rows_fr = [[round(f, 2), round(float(hc), 6), round(float(hd), 6)]
               for f, hc, hd in zip(freqs, H_cat, H_dog)]
    _save_csv("exp05_freq_response.csv", rows_fr, ["f_Hz", "H_cat", "H_dog"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Experiment 05 – Membran-Sprungantwort & Frequenzgang: Katze vs. Hund",
        fontweight="bold")

    ax1.plot(t * 1e3, resp_cat, color=_CAT_COLOR,
             label=f"Katze  Q={CAT.Q_mean:.0f}, D={D_cat:.3f}")
    ax1.plot(t * 1e3, resp_dog, color=_DOG_COLOR, linestyle="--",
             label=f"Hund   Q={DOG.Q_mean:.0f}, D={D_dog:.3f}")
    ax1.axhline(1.0, color=_GRAY, linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Zeit [ms]")
    ax1.set_ylabel("Normierte Auslenkung")
    ax1.set_title(f"Sprungantwort (f₀ = {f0} Hz)")
    ax1.legend()

    ax2.plot(freqs, 20 * np.log10(H_cat + 1e-12), color=_CAT_COLOR,
             label=f"Katze  Q={CAT.Q_mean:.0f}")
    ax2.plot(freqs, 20 * np.log10(H_dog + 1e-12), color=_DOG_COLOR, linestyle="--",
             label=f"Hund   Q={DOG.Q_mean:.0f}")
    ax2.axvline(f0, color=_GRAY, linewidth=0.6, linestyle=":")
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylabel("|H(f)| [dB]")
    ax2.set_title("Frequenzgang-Betrag (Bandpasscharakteristik)")
    ax2.legend()
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _save_fig(fig, "exp05_step_response.svg")
    print("  exp05 ✓")


# ===========================================================================
# Experiment 06 – RLC circuit: impedance and step response
# ===========================================================================

def exp06_rlc_circuit() -> None:
    """RLC impedance magnitude and step response for cat-like and dog-like Q."""
    # Map Q of each species to RLC parameters at 1 kHz resonance.
    # ω₀ = 2π·1000, Q = (1/R)√(L/C)  → choose L=1 mH, C=1/(ω₀²·L), R=1/(Q·ω₀·L)
    f_res = 1000.0
    omega0 = 2 * np.pi * f_res
    L = 1e-3   # 1 mH

    def rlc_params(Q_val):
        C = 1.0 / (omega0**2 * L)
        R = 1.0 / (Q_val * omega0 * L)
        return R, L, C

    R_cat, L_cat, C_cat = rlc_params(CAT.Q_mean)
    R_dog, L_dog, C_dog = rlc_params(DOG.Q_mean)

    freqs = np.linspace(100, 5000, 3000)
    Z_cat = impedance_magnitude(freqs * 2 * np.pi, R_cat, L_cat, C_cat)
    Z_dog = impedance_magnitude(freqs * 2 * np.pi, R_dog, L_dog, C_dog)
    BP_cat = bandpass_magnitude(freqs * 2 * np.pi, R_cat, L_cat, C_cat)
    BP_dog = bandpass_magnitude(freqs * 2 * np.pi, R_dog, L_dog, C_dog)

    t_step = np.linspace(0, 5e-3, 5000)
    u_cat, _ = rlc_step_response(t_step, R_cat, L_cat, C_cat, U0=1.0)
    u_dog, _ = rlc_step_response(t_step, R_dog, L_dog, C_dog, U0=1.0)

    # CSV
    rows = [[round(f, 2), round(float(zc), 6), round(float(zd), 6),
             round(float(bc), 6), round(float(bd), 6)]
            for f, zc, zd, bc, bd in zip(freqs, Z_cat, Z_dog, BP_cat, BP_dog)]
    _save_csv("exp06_rlc_impedance.csv", rows,
              ["f_Hz", "Z_cat_Ohm", "Z_dog_Ohm", "BP_cat", "BP_dog"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 06 – RLC-Schaltung: Impedanz und Sprungantwort",
                  fontweight="bold")

    ax1.semilogy(freqs, Z_cat, color=_CAT_COLOR,
                 label=f"Katze  Q={CAT.Q_mean:.0f}, R={R_cat*1000:.2f} mΩ")
    ax1.semilogy(freqs, Z_dog, color=_DOG_COLOR, linestyle="--",
                 label=f"Hund   Q={DOG.Q_mean:.0f}, R={R_dog*1000:.2f} mΩ")
    ax1.axvline(f_res, color=_GRAY, linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Frequenz [Hz]")
    ax1.set_ylabel("|Z(f)| [Ω]")
    ax1.set_title("Impedanzbetrag |Z(f)|")
    ax1.legend()
    ax1.grid(True, which="both", linewidth=0.3)

    ax2.plot(t_step * 1e3, u_cat, color=_CAT_COLOR,
             label=f"Katze  Q={CAT.Q_mean:.0f}")
    ax2.plot(t_step * 1e3, u_dog, color=_DOG_COLOR, linestyle="--",
             label=f"Hund   Q={DOG.Q_mean:.0f}")
    ax2.axhline(1.0, color=_GRAY, linewidth=0.6, linestyle=":")
    ax2.set_xlabel("Zeit [ms]")
    ax2.set_ylabel("u_C(t) / U₀")
    ax2.set_title("Sprungantwort der Kondensatorspannung")
    ax2.legend()

    fig.tight_layout()
    _save_fig(fig, "exp06_rlc.svg")
    print("  exp06 ✓")


# ===========================================================================
# Experiment 07 – Hodgkin-Huxley action potential
# ===========================================================================

def exp07_hodgkin_huxley() -> None:
    """HH action potential and gating variables (m, h, n)."""
    dt = 0.01e-3           # 0.01 ms
    n_steps = 8000         # 80 ms total
    I_ext = np.zeros(n_steps)
    I_ext[500:1000] = 10.0   # 10 µA/cm², 5 ms pulse → triggers spike

    V, t = hodgkin_huxley(I_ext, dt=dt)

    # Also run explicit gating variable population for subplot 2
    from digi.neuron import (
        _hh_alpha_m, _hh_beta_m,
        _hh_alpha_h, _hh_beta_h,
        _hh_alpha_n, _hh_beta_n,
        _hh_m_inf, _hh_h_inf, _hh_n_inf,
        _HH_DEFAULTS,
    )
    p = _HH_DEFAULTS.copy()
    m_arr = np.zeros(n_steps)
    h_arr = np.zeros(n_steps)
    n_arr = np.zeros(n_steps)
    v_arr = V.copy()

    m_arr[0] = _hh_m_inf(0.0)
    h_arr[0] = _hh_h_inf(0.0)
    n_arr[0] = _hh_n_inf(0.0)

    for i in range(1, n_steps):
        v = v_arr[i - 1]
        dm = _hh_alpha_m(v) * (1 - m_arr[i-1]) - _hh_beta_m(v) * m_arr[i-1]
        dh = _hh_alpha_h(v) * (1 - h_arr[i-1]) - _hh_beta_h(v) * h_arr[i-1]
        dn = _hh_alpha_n(v) * (1 - n_arr[i-1]) - _hh_beta_n(v) * n_arr[i-1]
        m_arr[i] = float(np.clip(m_arr[i-1] + dm * dt * 1e3, 0, 1))
        h_arr[i] = float(np.clip(h_arr[i-1] + dh * dt * 1e3, 0, 1))
        n_arr[i] = float(np.clip(n_arr[i-1] + dn * dt * 1e3, 0, 1))

    t_ms = t * 1e3

    # CSV
    rows = [[round(float(tm), 5), round(float(v), 4),
             round(float(m), 5), round(float(h), 5), round(float(n), 5)]
            for tm, v, m, h, n in zip(t_ms, V, m_arr, h_arr, n_arr)]
    _save_csv("exp07_hodgkin_huxley.csv", rows, ["t_ms", "V_mV", "m", "h", "n"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 07 – Hodgkin-Huxley Aktionspotential",
                  fontweight="bold")

    stim_start = 500 * dt * 1e3
    stim_end   = 1000 * dt * 1e3
    ax1.axvspan(stim_start, stim_end, alpha=0.15, color="gold", label="Stimulus")
    ax1.plot(t_ms, V, color=_CAT_COLOR, linewidth=1.2, label="V(t) [mV]")
    ax1.axhline(0, color=_GRAY, linewidth=0.5)
    ax1.set_xlabel("Zeit [ms]")
    ax1.set_ylabel("Membranpotential V [mV]")
    ax1.set_title("Aktionspotential (HH 1952, I=10 µA/cm²)")
    ax1.legend()

    ax2.plot(t_ms, m_arr**3, color=_CAT_COLOR,  label="m³ (Na-Aktivierung)")
    ax2.plot(t_ms, h_arr,    color=_DOG_COLOR,   label="h (Na-Inaktivierung)")
    ax2.plot(t_ms, n_arr**4, color=_HU_COLOR,    label="n⁴ (K-Aktivierung)")
    ax2.axvspan(stim_start, stim_end, alpha=0.1, color="gold")
    ax2.set_xlabel("Zeit [ms]")
    ax2.set_ylabel("Gating-Variable (normiert)")
    ax2.set_title("Gating-Variablen m³, h, n⁴")
    ax2.legend()

    fig.tight_layout()
    _save_fig(fig, "exp07_hodgkin_huxley.svg")
    print("  exp07 ✓")


# ===========================================================================
# Experiment 08 – Tonotopic maps cat vs dog + gammatone impulse responses
# ===========================================================================

def exp08_tonotopy_filterbank() -> None:
    """Tonotopic frequency-place maps and gammatone filters for cat and dog."""
    beta_cat  = _beta(CAT)
    beta_dog  = _beta(DOG)

    x_cat = np.linspace(0, CAT.L_membrane, 500)
    x_dog = np.linspace(0, DOG.L_membrane, 500)

    f_cat = tonotopic_frequency(x_cat, CAT.f_apex, beta_cat)
    f_dog = tonotopic_frequency(x_dog, DOG.f_apex, beta_dog)

    # Gammatone filter at 4 kHz center frequency
    t_gt = np.linspace(0, 15e-3, 2000)   # 15 ms
    cf = 4000.0
    b_cat_filter = cf / CAT.Q_mean   # ERB ≈ CF/Q
    b_dog_filter = cf / DOG.Q_mean
    gt_cat = gammatone_impulse_response(t_gt, f_center=cf, n=4, b=b_cat_filter)
    gt_dog = gammatone_impulse_response(t_gt, f_center=cf, n=4, b=b_dog_filter)

    # CSV: tonotopy
    rows_tono = [[round(float(xc), 3), round(float(fc), 2),
                  round(float(xd), 3), round(float(fd), 2)]
                 for xc, fc, xd, fd in zip(x_cat, f_cat,
                                           np.linspace(0, DOG.L_membrane, 500),
                                           f_dog)]
    _save_csv("exp08_tonotopy.csv", rows_tono,
              ["x_cat_mm", "f_cat_Hz", "x_dog_mm", "f_dog_Hz"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Experiment 08 – Tonotopie und Gammatone-Filter: Katze vs. Hund",
                  fontweight="bold")

    ax1.semilogy(x_cat, f_cat / 1e3, color=_CAT_COLOR,
                 label=f"Katze  β={beta_cat:.3f} mm⁻¹, L={CAT.L_membrane} mm")
    ax1.semilogy(x_dog, f_dog / 1e3, color=_DOG_COLOR, linestyle="--",
                 label=f"Hund   β={beta_dog:.3f} mm⁻¹, L={DOG.L_membrane} mm")
    ax1.set_xlabel("Position x [mm] (Apex → Basis)")
    ax1.set_ylabel("Charakteristische Frequenz [kHz]")
    ax1.set_title("Tonotopische Abbildung f(x) = f_apex · e^(β·x)")
    ax1.legend()
    ax1.grid(True, which="both", linewidth=0.3)

    gt_cat_norm = gt_cat / (np.max(np.abs(gt_cat)) + 1e-30)
    gt_dog_norm = gt_dog / (np.max(np.abs(gt_dog)) + 1e-30)
    ax2.plot(t_gt * 1e3, gt_cat_norm, color=_CAT_COLOR,
             label=f"Katze  Q={CAT.Q_mean:.0f}, b={b_cat_filter:.0f} Hz")
    ax2.plot(t_gt * 1e3, gt_dog_norm, color=_DOG_COLOR, linestyle="--",
             label=f"Hund   Q={DOG.Q_mean:.0f}, b={b_dog_filter:.0f} Hz")
    ax2.axhline(0, color=_GRAY, linewidth=0.4)
    ax2.set_xlabel("Zeit [ms]")
    ax2.set_ylabel("Normierte Amplitude")
    ax2.set_title(f"Gammatone-Impulsantwort  CF = {cf/1e3:.0f} kHz")
    ax2.legend()

    fig.tight_layout()
    _save_fig(fig, "exp08_tonotopy.svg")
    print("  exp08 ✓")


# ===========================================================================
# Experiment 09 – Shannon capacity landscape: cat / dog / human
# ===========================================================================

def exp09_shannon_capacity() -> None:
    """Shannon capacity C(B, SNR) landscape with species marked."""
    profiles = [CAT, DOG, HUMAN]
    colors   = [_CAT_COLOR, _DOG_COLOR, _HU_COLOR]

    # Capacity landscape on a 2-D grid
    B_range   = np.logspace(3, 5.2, 200)    # 1 kHz … 160 kHz
    snr_range = np.linspace(0, 70, 200)     # 0 … 70 dB
    BB, SS = np.meshgrid(B_range, snr_range)
    CC = BB * np.log2(1.0 + 10.0 ** (SS / 10.0)) / 1e6   # Mbit/s

    # Capacity breakdown (how much does B vs SNR contribute)
    # C = B · CE_snr  where CE_snr = log2(1+SNR)
    CE_vals   = [np.log2(1.0 + p.snr_linear)   for p in profiles]
    B_vals    = [p.bandwidth / 1e3              for p in profiles]   # kHz
    C_total   = [p.shannon_capacity_bits_per_second / 1e6 for p in profiles]

    # CSV
    rows_sp = []
    for pr in profiles:
        rows_sp.append([
            pr.species,
            round(pr.bandwidth / 1e3, 1),
            round(pr.snr_db, 1),
            round(pr.shannon_capacity_bits_per_second / 1e6, 3),
        ])
    _save_csv("exp09_shannon_capacity.csv", rows_sp,
              ["species", "bandwidth_kHz", "snr_dB", "capacity_Mbit_s"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Experiment 09 – Shannon-Kapazitätslandschaft: Katze / Hund / Mensch",
                  fontweight="bold")

    # Contour landscape
    cf = ax1.contourf(B_range / 1e3, snr_range, CC,
                      levels=30, cmap="viridis", alpha=0.85)
    fig.colorbar(cf, ax=ax1, label="C [Mbit/s]")
    for pr, col in zip(profiles, colors):
        ax1.scatter(pr.bandwidth / 1e3, pr.snr_db,
                    s=100, color=col, zorder=5,
                    edgecolors="white", linewidths=0.8)
        ax1.annotate(
            f"{pr.species}\n{pr.shannon_capacity_bits_per_second/1e6:.2f} Mbit/s",
            xy=(pr.bandwidth / 1e3, pr.snr_db),
            xytext=(6, -18), textcoords="offset points",
            fontsize=7, color=col,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Bandbreite B [kHz]")
    ax1.set_ylabel("SNR [dB]")
    ax1.set_title("C(B, SNR) = B · log₂(1 + SNR)  [Mbit/s]")

    # Bar chart: capacity breakdown
    x_pos = np.arange(len(profiles))
    width = 0.35
    bars1 = ax2.bar(x_pos - width / 2, B_vals, width,
                    color=colors, alpha=0.5, label="B [kHz]", edgecolor="k",
                    linewidth=0.5)
    ax2r = ax2.twinx()
    bars2 = ax2r.bar(x_pos + width / 2, C_total, width,
                     color=colors, alpha=0.9, label="C [Mbit/s]", edgecolor="k",
                     linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([p.species for p in profiles])
    ax2.set_ylabel("Bandbreite [kHz]", color=_GRAY)
    ax2r.set_ylabel("Kapazität C [Mbit/s]")
    ax2.set_title("Bandbreite und Shannon-Kapazität im Vergleich")

    # Manual legend
    patch_b = mpatches.Patch(color=_GRAY, alpha=0.5, label="Bandbreite [kHz]")
    patch_c = mpatches.Patch(color=_GRAY, alpha=0.9, label="Kapazität [Mbit/s]")
    ax2.legend(handles=[patch_b, patch_c], loc="upper left", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, "exp09_shannon_capacity.svg")
    print("  exp09 ✓")


# ===========================================================================
# Experiment 10 – Membrane optimality: multi-metric Ω analysis
# ===========================================================================

def exp10_membrane_optimality() -> None:
    """Quantitative proof of cat membrane optimality across six metrics."""
    profiles = [CAT, DOG, HUMAN]
    colors   = [_CAT_COLOR, _DOG_COLOR, _HU_COLOR]

    beta_cat = _beta(CAT)
    beta_dog = _beta(DOG)
    beta_hu  = _beta(HUMAN)
    betas    = [beta_cat, beta_dog, beta_hu]

    # Relative metrics (cat = 1.0 reference)
    metrics = {
        "Ω (Gesamt)":              [membrane_efficiency_omega(p, reference=CAT) for p in profiles],
        "Bandbreite η_B\n[kHz/mm]":  [p.bandwidth_efficiency / 1e3 for p in profiles],
        "Gütefaktor Q̄":             [p.Q_mean for p in profiles],
        "Dynamikumfang D\n[dB]":     [p.dynamic_range_db for p in profiles],
        "Temporal. Aufl.\n1/(2σ_t) [kHz]": [p.temporal_resolution_hz / 1e3 for p in profiles],
        "Kapazität C\n[Mbit/s]":     [p.shannon_capacity_bits_per_second / 1e6 for p in profiles],
        "Tonotop. Gradient β\n[mm⁻¹]": [b for b in betas],
    }

    # CSV: full comparison table
    cmp = compare_profiles(CAT, DOG, HUMAN)
    rows_cmp = []
    for sp, d in cmp.items():
        rows_cmp.append([
            sp, d["f_min_Hz"], d["f_max_Hz"],
            d["bandwidth_Hz"], d["L_membrane_mm"],
            d["Q_mean"], d["sigma_t_us"],
            d["dynamic_range_dB"],
            d["bandwidth_efficiency_kHz_per_mm"],
            d["temporal_resolution_kHz"],
            d["shannon_capacity_Mbit_per_s"],
            d["omega"],
        ])
    _save_csv(
        "exp10_membrane_optimality.csv",
        rows_cmp,
        ["species", "f_min_Hz", "f_max_Hz", "B_Hz", "L_mm",
         "Q_mean", "sigma_t_us", "DR_dB",
         "eta_B_kHz_per_mm", "temporal_kHz", "C_Mbit_s", "Omega"],
    )

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.suptitle(
        "Experiment 10 – Optimalität der Katzenmembran: Mehrkriterien-Analyse",
        fontweight="bold")

    # ---- Subplot 1: grouped bar chart of normalized metrics ----
    metric_labels = list(metrics.keys())
    n_m = len(metric_labels)
    n_p = len(profiles)
    x = np.arange(n_m)
    bar_w = 0.24

    # Normalize each metric to the cat value
    for j, (lbl, vals) in enumerate(metrics.items()):
        ref = vals[0] if vals[0] != 0 else 1.0
        for k, (pr, col) in enumerate(zip(profiles, colors)):
            bar = ax1.bar(j + (k - 1) * bar_w, vals[k] / ref,
                          bar_w, color=col, alpha=0.85,
                          label=pr.species if j == 0 else "_nolegend_",
                          edgecolor="white", linewidth=0.4)

    ax1.axhline(1.0, color=_GRAY, linewidth=0.7, linestyle="--",
                label="Referenz: Katze = 1.0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=7, rotation=15, ha="right")
    ax1.set_ylabel("Normierter Wert (Katze = 1.0)")
    ax1.set_title("Normierte Leistungsmerkmale: Katze / Hund / Mensch")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)

    # ---- Subplot 2: Ω breakdown as stacked component bars ----
    sp_labels = [p.species for p in profiles]
    eta_ratio = [p.bandwidth_efficiency / CAT.bandwidth_efficiency for p in profiles]
    q_ratio   = [p.Q_mean / CAT.Q_mean for p in profiles]
    d_ratio   = [p.dynamic_range_db / CAT.dynamic_range_db for p in profiles]
    t_ratio   = [CAT.sigma_t / p.sigma_t for p in profiles]

    x2 = np.arange(len(profiles))
    b1 = ax2.bar(x2, eta_ratio,
                 label="η_B-Anteil (Bandbreiten-Effizienz)", color="#E8A080", edgecolor="k", lw=0.5)
    b2 = ax2.bar(x2, [q * e for q, e in zip(q_ratio, eta_ratio)],
                 bottom=eta_ratio, label="Q̄-Anteil (Gütefaktor)", color=_CAT_COLOR, edgecolor="k", lw=0.5,
                 alpha=0.0)  # invisible – used for stacking reference only

    # Instead show all four Ω components as grouped bars
    ax2.cla()
    comp_labels = ["η_B", "Q̄", "D", "1/σ_t"]
    comp_data = [eta_ratio, q_ratio, d_ratio, t_ratio]
    comp_colors = ["#E8A080", "#C0392B", "#2980B9", "#27AE60"]
    x2c = np.arange(len(comp_labels))
    w2 = 0.25
    for k, (pr, col) in enumerate(zip(profiles, colors)):
        vals_k = [comp_data[ci][k] for ci in range(4)]
        ax2.bar(x2c + (k - 1) * w2, vals_k, w2,
                label=pr.species, color=col, alpha=0.85,
                edgecolor="white", linewidth=0.4)

    # Annotate Ω total
    for k, pr in enumerate(profiles):
        om = membrane_efficiency_omega(pr, reference=CAT)
        ax2.annotate(f"Ω={om:.3f}", xy=(x2c[-1] + (k - 1) * w2 + 0.5, 0),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=7, color=colors[k])

    ax2.axhline(1.0, color=_GRAY, linewidth=0.7, linestyle="--",
                label="Referenz: Katze = 1.0")
    ax2.set_xticks(x2c)
    ax2.set_xticklabels(comp_labels, fontsize=9)
    ax2.set_ylabel("Normierter Komponentenwert (Katze = 1.0)")
    ax2.set_title("Ω-Komponenten: η_B · Q̄ · D · (1/σ_t)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "exp10_membrane_optimality.svg")
    print("  exp10 ✓")


# ===========================================================================
# Experiment 11 – Active cochlear amplifier: Q(g_motor) for cat and dog
# ===========================================================================

def exp11_active_amplifier() -> None:
    """Cochlear amplifier gain and Q enhancement vs. Prestin motor conductance."""
    # Passive Q values from profiles
    Q_pass_cat = CAT.Q_range[0]    # lower end ≈ passive
    Q_pass_dog = DOG.Q_range[0]

    # Sweep motor conductance g relative to passive damping R (normalised)
    R_passive = 1.0               # normalised
    g_range = np.linspace(0, 0.95, 500)   # must stay < R for stability

    Q_active_cat = np.array([
        active_quality_factor(Q_pass_cat, g, R_passive)
        for g in g_range
    ])
    Q_active_dog = np.array([
        active_quality_factor(Q_pass_dog, g, R_passive)
        for g in g_range
    ])

    # SNR gain relative to dog (Q_cat vs Q_dog) over realistic g for cat
    g_realistic_cat = np.linspace(0, 0.85, 400)
    G_cat = np.array([active_quality_factor(Q_pass_cat, g, R_passive)
                      for g in g_realistic_cat])
    G_dog = Q_pass_dog * np.ones_like(g_realistic_cat)  # dog stays passive
    snr_gain = 20 * np.log10(G_cat / G_dog)

    # CSV
    rows = [[round(float(g), 4),
             round(float(qac), 4),
             round(float(qad), 4)]
            for g, qac, qad in zip(g_range, Q_active_cat, Q_active_dog)]
    _save_csv("exp11_active_amplifier.csv", rows,
              ["g_motor_norm", "Q_active_cat", "Q_active_dog"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Experiment 11 – Aktiver Kohleorverstärker: Q(g_motor) und SNR-Gewinn",
        fontweight="bold")

    ax1.plot(g_range, Q_active_cat, color=_CAT_COLOR,
             label=f"Katze  Q_pass={Q_pass_cat:.1f}")
    ax1.plot(g_range, Q_active_dog, color=_DOG_COLOR, linestyle="--",
             label=f"Hund   Q_pass={Q_pass_dog:.1f}")
    ax1.axhline(CAT.Q_mean, color=_CAT_COLOR, linewidth=0.7,
                linestyle=":", alpha=0.7, label=f"Katze Q_mean={CAT.Q_mean:.0f}")
    ax1.axhline(DOG.Q_mean, color=_DOG_COLOR, linewidth=0.7,
                linestyle=":", alpha=0.7, label=f"Hund  Q_mean={DOG.Q_mean:.0f}")
    ax1.set_xlabel("Normierte Motorkondukanz g / R_passiv")
    ax1.set_ylabel("Q_aktiv")
    ax1.set_title("Gütefaktor-Verstärkung durch Prestin-Motor")
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 0.95)
    ax1.set_ylim(0, min(100, ax1.get_ylim()[1]))

    ax2.plot(g_realistic_cat, snr_gain, color=_CAT_COLOR, linewidth=1.4)
    ax2.fill_between(g_realistic_cat, 0, snr_gain, alpha=0.2, color=_CAT_COLOR)
    ax2.axhline(0, color=_GRAY, linewidth=0.6)
    ax2.axhline(snr_gain_db(CAT.Q_mean, DOG.Q_mean), color="k",
                linewidth=0.8, linestyle="--",
                label=f"Bei Q_cat={CAT.Q_mean:.0f}: {snr_gain_db(CAT.Q_mean, DOG.Q_mean):.1f} dB")
    ax2.set_xlabel("Normierte Motorkondukanz g / R_passiv")
    ax2.set_ylabel("SNR-Gewinn Katze vs. Hund [dB]")
    ax2.set_title("SNR-Vorteil durch aktivere Kochlea der Katze")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _save_fig(fig, "exp11_active_amplifier.svg")
    print("  exp11 ✓")


# ===========================================================================
# Experiment 12 – LIF spike train + spectral efficiency
# ===========================================================================

def exp12_lif_spectral_efficiency() -> None:
    """LIF spike train under varying current; Shannon spectral efficiency."""
    # --- LIF: threshold comparison ------------------------------------------
    dt = 5e-5   # 50 µs
    T = 0.5     # simulation duration [s]
    n = int(T / dt)

    # Sweep current amplitude
    I_levels = np.linspace(1.5e-9, 3.5e-9, 8)   # 1.5 … 3.5 nA
    spike_rates = []
    for I_amp in I_levels:
        I_ext = np.ones(n) * I_amp
        _, spikes = leaky_integrate_and_fire(
            I_ext, dt=dt,
            C_m=1e-10, R_m=1e7,
            V_rest=-70e-3, V_thresh=-55e-3,
        )
        spike_rates.append(float(np.sum(spikes)) / T)

    # Full trace at 2.5 nA
    I_demo = np.ones(n) * 2.5e-9
    V_trace, spikes_trace = leaky_integrate_and_fire(
        I_demo, dt=dt,
        C_m=1e-10, R_m=1e7,
        V_rest=-70e-3, V_thresh=-55e-3,
    )
    t_trace = np.arange(n) * dt * 1e3

    # --- Spectral efficiency C/B = log2(1+SNR) -------------------------------
    snr_db_vals = np.linspace(-10, 60, 500)
    C_over_B = capacity_over_bandwidth(snr_db_vals)

    # Cat, Dog, Human operating points
    species_pts = [(CAT, _CAT_COLOR), (DOG, _DOG_COLOR), (HUMAN, _HU_COLOR)]

    # CSV
    rows_lif = [[round(float(ia * 1e9), 3), round(sr, 2)]
                for ia, sr in zip(I_levels, spike_rates)]
    _save_csv("exp12_spike_rate.csv", rows_lif, ["I_nA", "spike_rate_Hz"])

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Experiment 12 – LIF-Entladerate und spektrale Effizienz",
        fontweight="bold")

    # Spike train trace (first 100 ms)
    mask = np.arange(n) * dt < 0.1
    ax1.plot(t_trace[mask], V_trace[mask] * 1e3, color=_CAT_COLOR,
             linewidth=0.8, label="V(t) [mV]")
    spike_t = t_trace[mask][spikes_trace[mask].astype(bool)]
    ax1.vlines(spike_t, -75, 40, color=_DOG_COLOR, linewidth=1.0,
               label=f"Spikes  (I=2.5 nA)")
    ax1.set_xlabel("Zeit [ms]")
    ax1.set_ylabel("Membranpotential [mV]")
    ax1.set_title("LIF-Entladung (Lecky Integrate & Fire)")
    ax1.legend(fontsize=8)

    ax2.plot(snr_db_vals, C_over_B, color=_CAT_COLOR, linewidth=1.4,
             label="C/B = log₂(1 + SNR)")
    for pr, col in species_pts:
        cb = np.log2(1.0 + pr.snr_linear)
        ax2.scatter(pr.snr_db, cb, s=70, color=col, zorder=5,
                    edgecolors="white", linewidths=0.6)
        ax2.annotate(f"{pr.species}\n{cb:.1f} bit/s/Hz",
                     xy=(pr.snr_db, cb),
                     xytext=(5, -18), textcoords="offset points",
                     fontsize=7, color=col)
    ax2.set_xlabel("SNR [dB]")
    ax2.set_ylabel("C/B [bit/s/Hz]")
    ax2.set_title("Spektrale Effizienz C/B = log₂(1 + SNR)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _save_fig(fig, "exp12_lif_spectral_efficiency.svg")
    print("  exp12 ✓")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(f"\nDigi-Experimente – Ausgabeverzeichnis: {RESULTS_DIR}\n")
    experiments = [
        exp01_harmonic_derivative,
        exp02_sampling_aliasing,
        exp03_quantization_snr,
        exp04_acoustic_pressure_spl,
        exp05_membrane_step_response,
        exp06_rlc_circuit,
        exp07_hodgkin_huxley,
        exp08_tonotopy_filterbank,
        exp09_shannon_capacity,
        exp10_membrane_optimality,
        exp11_active_amplifier,
        exp12_lif_spectral_efficiency,
    ]
    for exp in experiments:
        try:
            exp()
        except Exception as exc:
            print(f"  !! {exp.__name__} fehlgeschlagen: {exc}")

    svg_files = sorted(RESULTS_DIR.glob("*.svg"))
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    print(f"\n✓ {len(svg_files)} SVG-Grafiken und {len(csv_files)} CSV-Dateien erstellt.")
    print(f"  → {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
