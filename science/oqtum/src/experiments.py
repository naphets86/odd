"""
src/experiments.py
==================
Ausführliche experimentelle Validierung aller Framework-Funktionen.

Jeder Experiment-Block erzeugt eine SVG-Datei mit zwei Subplots im
Querformat (16 × 6 Zoll) für die direkte Einbindung in die wissenschaftliche
Arbeit.

Aufruf:
    python src/experiments.py

Erzeugte SVG-Dateien (im Verzeichnis experiments/):
    exp01_fresnel_coefficients.svg
    exp02_tmm_energy_conservation.svg
    exp03_fabry_perot_spectrum.svg
    exp04_fp_finesse_orders.svg
    exp05_tmm_inhomogeneous.svg
    exp06_lorentz_refractive_index.svg
    exp07_lorentz_lorenz_mixing.svg
    exp08_flory_rehner_equilibrium.svg
    exp09_flory_rehner_phase_diagram.svg
    exp10_swelling_thickness.svg
    exp11_coupling_equation.svg
    exp12_bethe_range.svg
    exp13_proximity_function.svg
    exp14_proximity_correction.svg
    exp15_ipec_convergence.svg
    exp16_colorimetry_cie.svg
    exp17_delta_e_camouflage.svg
    exp18_spectral_loss_gradient.svg
    exp19_optimizer_convergence.svg
    exp20_full_pipeline.svg
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Sicherstellen, dass src/ im Pfad liegt
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings

# Framework-Importe
from optics import (
    Layer, fresnel_coefficients, layer_matrix, transfer_matrix,
    compute_rt, tmm_spectrum, tmm_inhomogeneous,
    fabry_perot_reflection, fabry_perot_finesse, fabry_perot_resonance_wavelength,
    lorentz_refractive_index,
)
from polymer import (
    lorentz_lorenz, n_from_swelling, n_from_swelling_linear,
    flory_rehner_residual, solve_flory_rehner,
    swelling_to_thickness, coupling_equation,
    bethe_range, dose_to_crosslink_density,
)
from ebeam import (
    proximity_function, proximity_kernel_2d, apply_proximity,
    proximity_correction_ipec,
)
from colorimetry import (
    wavelengths_visible, photopic_sensitivity,
    spectrum_to_xyz, xyz_to_lab, delta_e, spectrum_to_lab,
    camouflage_index,
)
from optimizer import (
    spectral_loss, perceptual_loss, gradient_reflection,
    adjoint_gradient, optimize_resonator,
    target_spectrum_gaussian, target_spectrum_leaf,
)
from pipeline import DesignTarget, run_pipeline
from materials import MATERIALS, FLORY_HUGGINS_CHI, MOLAR_VOLUME

# ---------------------------------------------------------------------------
# Globales Plot-Styling
# ---------------------------------------------------------------------------

OUTDIR = _HERE.parent / "src/results"
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "figure.dpi":        150,
    "lines.linewidth":   1.8,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS = plt.cm.tab10.colors


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUTDIR / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def make_fig(suptitle: str) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Erzeugt eine Figure mit zwei nebeneinander liegenden Subplots (Querformat)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    return fig, (ax1, ax2)


# ===========================================================================
# EXPERIMENT 01 – Fresnel-Koeffizienten (Def. 2.2)
# ===========================================================================

def exp01_fresnel():
    print("EXP 01: Fresnel-Koeffizienten")
    fig, (ax1, ax2) = make_fig(
        "Exp. 01 – Fresnel-Koeffizienten (Def. 2.2, Gl. 2–5)"
    )

    angles = np.linspace(0, np.pi / 2 - 0.01, 500)
    n1, n2 = 1.0, 1.5  # Luft → Glas

    Rs, Rp, Ts, Tp = [], [], [], []
    for a in angles:
        rs, ts = fresnel_coefficients(n1, n2, a, "s")
        rp, tp = fresnel_coefficients(n1, n2, a, "p")
        cos_i = np.cos(a)
        # Snellius: sin(θ_t) = n1/n2 * sin(θ_i)
        sin_t = np.clip(n1 / n2 * np.sin(a), -1.0, 1.0)
        cos_t = np.sqrt(max(1.0 - sin_t**2, 0.0))
        # Energetische Transmissionskoeffizienten: T = (n2·cos_t)/(n1·cos_i) · |t|²
        prefac = (n2 * cos_t) / (n1 * cos_i) if cos_i > 1e-9 else 0.0
        Rs.append(abs(rs)**2)
        Rp.append(abs(rp)**2)
        Ts.append(prefac * abs(ts)**2)
        Tp.append(prefac * abs(tp)**2)
    Rs, Rp, Ts, Tp = map(np.array, [Rs, Rp, Ts, Tp])

    ax1.plot(np.degrees(angles), Rs, color=COLORS[0], label=r"$R_s$ (TE)")
    ax1.plot(np.degrees(angles), Rp, color=COLORS[1], label=r"$R_p$ (TM)")
    ax1.plot(np.degrees(angles), Ts, color=COLORS[0], ls="--", alpha=0.7,
             label=r"$T_s = \frac{n_2\cos\theta_t}{n_1\cos\theta_i}|t_s|^2$")
    ax1.plot(np.degrees(angles), Tp, color=COLORS[1], ls="--", alpha=0.7, label=r"$T_p$")
    # Energieerhaltung als Kontrolle
    ax1.plot(np.degrees(angles), Rs + Ts, color="gray", lw=0.8, ls=":",
             label=r"$R_s+T_s = 1$ (Kontrolle)")

    # Brewster-Winkel
    theta_B = np.degrees(np.arctan(n2 / n1))
    ax1.axvline(theta_B, color="k", ls=":", lw=1.2, label=f"Brewster {theta_B:.1f}°")
    ax1.set_xlabel("Einfallswinkel θ [°]")
    ax1.set_ylabel("Intensitätskoeffizient")
    ax1.set_title("Luft → Glas (n₁=1.0, n₂=1.5): energetische Fresnel-Koeffizienten")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 90)

    # Totale Innenreflexion: Glas → Luft
    n1b, n2b = 1.5, 1.0
    theta_c = np.degrees(np.arcsin(n2b / n1b))
    angles2 = np.linspace(0, np.pi / 2 - 0.001, 500)
    Rs2 = np.array([abs(fresnel_coefficients(n1b, n2b, a, "s")[0])**2 for a in angles2])
    Rp2 = np.array([abs(fresnel_coefficients(n1b, n2b, a, "p")[0])**2 for a in angles2])

    ax2.plot(np.degrees(angles2), Rs2, color=COLORS[0], label=r"$R_s$")
    ax2.plot(np.degrees(angles2), Rp2, color=COLORS[1], label=r"$R_p$")
    ax2.axvline(theta_c, color="red", ls="--", lw=1.5,
                label=f"Grenzwinkel TIR {theta_c:.1f}°")
    ax2.set_xlabel("Einfallswinkel θ [°]")
    ax2.set_ylabel("Reflexionsintensität R")
    ax2.set_title("Glas → Luft: Totale Innenreflexion (n₁=1.5, n₂=1.0)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    savefig(fig, "exp01_fresnel_coefficients.svg")


# ===========================================================================
# EXPERIMENT 02 – TMM Energieerhaltung (Satz 3.1)
# ===========================================================================

def exp02_tmm_energy():
    print("EXP 02: TMM Energieerhaltung")
    fig, (ax1, ax2) = make_fig(
        "Exp. 02 – Transfermatrix-Methode: Energieerhaltung R+T+A=1 (Satz 3.1)"
    )

    lam = np.linspace(380, 780, 300)

    # Einfaches Glas-Antireflexsystem (λ/4-Schicht MgF₂)
    layers_ar = [Layer(n=1.38 + 0j, thickness=100.0)]  # MgF₂ λ/4 bei 550 nm
    R_ar, T_ar, A_ar = tmm_spectrum(layers_ar, lam, n_in=1.0, n_sub=1.5)

    ax1.fill_between(lam, 0, R_ar, alpha=0.45, color=COLORS[0], label="R (Reflexion)")
    ax1.fill_between(lam, R_ar, R_ar + T_ar, alpha=0.45, color=COLORS[2], label="T (Transmission)")
    ax1.fill_between(lam, R_ar + T_ar, R_ar + T_ar + A_ar, alpha=0.45, color=COLORS[3], label="A (Absorption)")
    ax1.plot(lam, R_ar + T_ar + A_ar, "k--", lw=1.0, label="R+T+A (soll=1)")
    ax1.set_xlabel("Wellenlänge λ [nm]")
    ax1.set_ylabel("Intensitätsanteil")
    ax1.set_title("MgF₂-Antireflexschicht (d=100 nm, n=1.38)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.1)

    # PDMS-Film auf Gold (absorptiv)
    n_au = MATERIALS["Au"].n_at(550.0)
    layers_pdms_au = [
        Layer(n=1.41 + 0j, thickness=200.0),   # PDMS
        Layer(n=n_au,       thickness=30.0),    # Gold (dünn, absorbierend)
    ]
    R2, T2, A2 = tmm_spectrum(layers_pdms_au, lam, n_in=1.0, n_sub=1.45)

    ax2.plot(lam, R2, color=COLORS[0], label="R (Reflexion)")
    ax2.plot(lam, T2, color=COLORS[2], label="T (Transmission)")
    ax2.plot(lam, A2, color=COLORS[3], label="A (Absorption)")
    ax2.plot(lam, R2 + T2 + A2, "k--", lw=1.0, label="R+T+A")
    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_title("PDMS (200 nm) + Au-Spiegel (30 nm) auf SiO₂")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.1)

    fig.tight_layout()
    savefig(fig, "exp02_tmm_energy_conservation.svg")


# ===========================================================================
# EXPERIMENT 03 – Fabry-Pérot-Spektrum (Gl. 8)
# ===========================================================================

def exp03_fp_spectrum():
    print("EXP 03: Fabry-Pérot-Spektrum")
    fig, (ax1, ax2) = make_fig(
        "Exp. 03 – Fabry-Pérot-Resonator: Reflexionsspektrum (Gl. 8)"
    )

    lam = np.linspace(380, 780, 1000)
    n_cav = 1.41  # PDMS

    # Variation der Spiegelreflexivität bei fester Dicke
    d_fix = 200.0
    for R_mir in [0.1, 0.3, 0.5, 0.7, 0.9]:
        R_fp = fabry_perot_reflection(lam, complex(n_cav), d_fix, R_mir, R_mir)
        F = fabry_perot_finesse(R_mir, R_mir)
        ax1.plot(lam, R_fp, label=f"R₁=R₂={R_mir:.1f}  (F={F:.1f})")

    ax1.set_xlabel("Wellenlänge λ [nm]")
    ax1.set_ylabel("Reflexion R")
    ax1.set_title(f"FP-Resonator: Variation R₁=R₂ (d={d_fix} nm, n={n_cav})")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # Variation der Kavitätsdicke bei fester Reflexivität
    R_fix = 0.5
    for d in [100, 150, 200, 250, 300]:
        lam_res = fabry_perot_resonance_wavelength(n_cav, d, order=1)
        R_fp = fabry_perot_reflection(lam, complex(n_cav), d, R_fix, R_fix)
        ax2.plot(lam, R_fp, label=f"d={d} nm  (λ_res={lam_res:.0f} nm)")

    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_title(f"FP-Resonator: Variation Kavitätsdicke (R={R_fix})")
    ax2.legend(loc="lower right")
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp03_fabry_perot_spectrum.svg")


# ===========================================================================
# EXPERIMENT 04 – FP-Finesse und Ordnungen (Gl. 13)
# ===========================================================================

def exp04_fp_finesse():
    print("EXP 04: FP-Finesse & Ordnungen")
    fig, (ax1, ax2) = make_fig(
        "Exp. 04 – Fabry-Pérot: Finesse (Gl. 13) und höhere Ordnungen"
    )

    # Finesse als Funktion von R
    R_vals = np.linspace(0.01, 0.995, 500)
    F_vals = np.array([fabry_perot_finesse(r, r) for r in R_vals])
    ax1.semilogy(R_vals, F_vals, color=COLORS[0])
    ax1.set_xlabel("Spiegelreflexivität R₁=R₂")
    ax1.set_ylabel("Finesse F (log)")
    ax1.set_title(r"Finesse $F = \pi(R_1 R_2)^{1/4}/(1-\sqrt{R_1 R_2})$")
    ax1.grid(True)
    # Referenzpunkte
    for r_ref, marker in [(0.3, "o"), (0.5, "s"), (0.7, "^"), (0.9, "D")]:
        f_ref = fabry_perot_finesse(r_ref, r_ref)
        ax1.plot(r_ref, f_ref, marker=marker, ms=8,
                 label=f"R={r_ref}: F={f_ref:.1f}")
    ax1.legend()

    # Mehrere Ordnungen
    lam = np.linspace(300, 900, 2000)
    n_cav, d = 1.41, 250.0
    R_mir = 0.6
    R_fp = fabry_perot_reflection(lam, complex(n_cav), d, R_mir, R_mir)
    ax2.plot(lam, R_fp, color=COLORS[1], lw=1.5, label="FP-Spektrum")

    for m in range(1, 5):
        lam_m = fabry_perot_resonance_wavelength(n_cav, d, order=m)
        if 300 < lam_m < 900:
            ax2.axvline(lam_m, ls="--", lw=1.0, alpha=0.7,
                        label=f"m={m}: λ={lam_m:.0f} nm")

    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_ylabel("Reflexion R")
    ax2.set_title(f"FP-Resonanzordnungen (d={d} nm, n={n_cav}, R={R_mir})")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp04_fp_finesse_orders.svg")


# ===========================================================================
# EXPERIMENT 05 – TMM inhomogen: Konvergenz (Alg. 3.1)
# ===========================================================================

def exp05_tmm_inhomogeneous():
    print("EXP 05: TMM inhomogen – Konvergenz")
    fig, (ax1, ax2) = make_fig(
        "Exp. 05 – TMM inhomogene Schicht: Konvergenz O(1/K²) (Alg. 3.1)"
    )

    lam0 = 550.0
    d_total = 300.0

    # Gradient-Profil: n(z) linear von 1.0 → 1.5
    def n_gradient(z):
        return 1.0 + 0.5 * z / d_total

    # Konvergenz in K
    K_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    R_ref = tmm_inhomogeneous(n_gradient, d_total, lam0, n_sublayers=1024).R

    errors = []
    for K in K_list:
        R_K = tmm_inhomogeneous(n_gradient, d_total, lam0, n_sublayers=K).R
        errors.append(abs(R_K - R_ref))

    ax1.loglog(K_list, errors, "o-", color=COLORS[0], label="Numerischer Fehler")
    # Theoretische O(1/K²) Linie
    K_arr = np.array(K_list, dtype=float)
    ref_slope = errors[2] * (K_list[2] / K_arr)**2
    ax1.loglog(K_arr, ref_slope, "k--", lw=1.0, alpha=0.7, label=r"$O(1/K^2)$ Referenz")
    ax1.set_xlabel("Anzahl Sublayer K")
    ax1.set_ylabel("|ΔR|")
    ax1.set_title("Konvergenzrate: Trotter-Produktformel (λ=550 nm)")
    ax1.legend()
    ax1.grid(True, which="both")

    # Spektrum: homogen vs. gradient vs. Stufe
    lam = np.linspace(400, 750, 300)

    # Homogene Schicht (n=1.25, d=300 nm) – Referenz
    layers_hom = [Layer(n=1.25 + 0j, thickness=d_total)]
    R_hom, _, _ = tmm_spectrum(layers_hom, lam, n_in=1.0, n_sub=1.5)

    # Gradient-Profil
    R_grad = np.array([
        tmm_inhomogeneous(n_gradient, d_total, w, n_sublayers=64).R for w in lam
    ])

    # Stufenprofil: untere Hälfte n=1.1, obere Hälfte n=1.4
    def n_step(z):
        return 1.1 if z < d_total / 2 else 1.4

    R_step = np.array([
        tmm_inhomogeneous(n_step, d_total, w, n_sublayers=64).R for w in lam
    ])

    ax2.plot(lam, R_hom,  color=COLORS[0], label="Homogen (n=1.25)")
    ax2.plot(lam, R_grad, color=COLORS[1], label="Gradient n(z): 1.0→1.5")
    ax2.plot(lam, R_step, color=COLORS[2], label="Stufe: n=1.1|1.4 (z<d/2)")
    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_ylabel("Reflexion R")
    ax2.set_title("Spektren: Homogen vs. inhomogene Brechungsindexprofile")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp05_tmm_inhomogeneous.svg")


# ===========================================================================
# EXPERIMENT 06 – Lorentz-Brechungsindex (Def. 2.3)
# ===========================================================================

def exp06_lorentz():
    print("EXP 06: Lorentz-Brechungsindex")
    fig, (ax1, ax2) = make_fig(
        "Exp. 06 – Lorentz-Oszillator-Modell (Def. 2.3)"
    )

    lam_nm = np.linspace(200, 800, 600)
    c = 3e8
    omega = 2 * np.pi * c / (lam_nm * 1e-9)

    # UV-Resonanz bei 280 nm
    omega0 = 2 * np.pi * c / (280e-9)
    gamma  = omega0 * 0.05
    N_vals = [5e26, 1e27, 3e27]

    for N in N_vals:
        n_c = lorentz_refractive_index(omega, omega0, gamma,
                                       oscillator_strength=1.0, N=N)
        ax1.plot(lam_nm, n_c.real, label=f"N={N:.0e} m⁻³")

    ax1.axvline(280, color="gray", ls="--", lw=1.0, label="Resonanz λ₀=280 nm")
    ax1.set_xlabel("Wellenlänge λ [nm]")
    ax1.set_ylabel("Realteil n")
    ax1.set_title("Realteil des Brechungsindex n(λ) – Lorentz-Modell")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(200, 800)

    # Brechungsindex + Extinktionskoeffizient für eine Dichte
    N_fix = 1e27
    n_c = lorentz_refractive_index(omega, omega0, gamma,
                                   oscillator_strength=1.0, N=N_fix)
    ax2.plot(lam_nm, n_c.real,  color=COLORS[0], label="n (Realteil)")
    ax2.plot(lam_nm, n_c.imag,  color=COLORS[3], label="κ (Extinktion)")
    ax2_r = ax2.twinx()
    ax2_r.plot(lam_nm, n_c.imag**2 * 4 * np.pi / lam_nm * 1e7,
               color=COLORS[3], ls="--", alpha=0.5, label="α (Absorption)")
    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_ylabel("n, κ")
    ax2_r.set_ylabel("Absorptionskoeffizient [cm⁻¹]", color=COLORS[3])
    ax2.set_title(f"Lorentz-Modell (N={N_fix:.0e}, λ₀=280 nm)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True)
    ax2.set_xlim(200, 800)

    fig.tight_layout()
    savefig(fig, "exp06_lorentz_refractive_index.svg")


# ===========================================================================
# EXPERIMENT 07 – Lorentz-Lorenz-Mischungsregel (Def. 5.4, Gl. 20)
# ===========================================================================

def exp07_lorentz_lorenz():
    print("EXP 07: Lorentz-Lorenz-Mischungsregel")
    fig, (ax1, ax2) = make_fig(
        "Exp. 07 – Lorentz-Lorenz-Mischungsregel (Def. 5.4, Gl. 20)"
    )

    phi_p = np.linspace(0.01, 1.0, 300)
    n_poly, n_solv = 1.50, 1.333  # PDMS, Wasser

    n_ll    = np.array([lorentz_lorenz(n_poly, n_solv, p) for p in phi_p])
    n_lin   = n_poly * phi_p + n_solv * (1 - phi_p)  # lineare Mischungsregel

    ax1.plot(phi_p, n_ll,  color=COLORS[0], label="Lorentz-Lorenz (Gl. 20)")
    ax1.plot(phi_p, n_lin, color=COLORS[1], ls="--", label="Lineare Mischungsregel")
    ax1.set_xlabel(r"Polymervolumenanteil $\phi_p$")
    ax1.set_ylabel(r"Effektiver Brechungsindex $n_\mathrm{eff}$")
    ax1.set_title("PDMS/Wasser-Gemisch: Lorentz-Lorenz vs. lineares Modell")
    ax1.legend()
    ax1.grid(True)

    # n(Q) für verschiedene Lösungsmittel
    Q_vals = np.linspace(1.0, 10.0, 300)
    solvents = [
        ("Wasser",       1.333),
        ("Ethanol",      1.360),
        ("Isopropanol",  1.376),
        ("n-Butanol",    1.399),
    ]
    for sol_name, n_s in solvents:
        n_exact  = np.array([n_from_swelling(n_poly, n_s, q) for q in Q_vals])
        n_approx = np.array([n_from_swelling_linear(n_poly, q) for q in Q_vals])
        ax2.plot(Q_vals, n_exact, label=f"{sol_name} (n_s={n_s})")

    ax2.plot(Q_vals, n_approx, "k--", lw=1.2, label="Lineare Näherung (G. 2.3)")
    ax2.set_xlabel("Quellungsgrad Q = V_sw / V_dry")
    ax2.set_ylabel(r"$n_\mathrm{eff}(Q)$")
    ax2.set_title("Brechungsindex vs. Quellungsgrad für verschiedene LM")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp07_lorentz_lorenz_mixing.svg")


# ===========================================================================
# EXPERIMENT 08 – Flory-Rehner-Gleichgewicht (Satz 5.1)
# ===========================================================================

def exp08_flory_rehner():
    print("EXP 08: Flory-Rehner-Gleichgewicht")
    fig, (ax1, ax2) = make_fig(
        "Exp. 08 – Flory-Rehner-Gleichgewicht (Satz 5.1, Gl. 16)"
    )

    phi_p = np.linspace(1e-4, 0.999, 1000)
    chi_vals = [0.10, 0.20, 0.30, 0.40, 0.48]

    # Residuum für verschiedene χ (nu_e=100 mol/m³)
    nu_e = 100.0
    for chi in chi_vals:
        res = np.array([flory_rehner_residual(p, chi, nu_e) for p in phi_p])
        ax1.plot(phi_p, res, label=f"χ={chi}")

    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel(r"Polymervolumenanteil $\phi_p$")
    ax1.set_ylabel("Residuum f(φ_p)")
    ax1.set_title(f"Flory-Rehner-Residuum (νe={nu_e} mol/m³)")
    ax1.set_ylim(-2, 3)
    ax1.legend()
    ax1.grid(True)

    # Gleichgewichtsquellung Q*(χ) für verschiedene νe
    chi_scan = np.linspace(0.05, 0.48, 80)
    nu_e_list = [50.0, 100.0, 200.0, 500.0]
    for nu in nu_e_list:
        Q_arr = []
        for chi in chi_scan:
            try:
                Q_arr.append(solve_flory_rehner(chi, nu))
            except (ValueError, Exception):
                Q_arr.append(np.nan)
        ax2.plot(chi_scan, Q_arr, label=f"νe={nu:.0f} mol/m³")

    ax2.set_xlabel("Flory-Huggins-Parameter χ")
    ax2.set_ylabel("Gleichgewichts-Quellungsgrad Q*")
    ax2.set_title("Q*(χ) für verschiedene Vernetzungsdichten")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0.05, 0.48)
    ax2.set_ylim(0, 30)

    fig.tight_layout()
    savefig(fig, "exp08_flory_rehner_equilibrium.svg")


# ===========================================================================
# EXPERIMENT 09 – Phasendiagramm χ-νe (Satz 5.1)
# ===========================================================================

def exp09_phase_diagram():
    print("EXP 09: Flory-Rehner-Phasendiagramm")
    fig, (ax1, ax2) = make_fig(
        "Exp. 09 – Flory-Rehner: Phasendiagramm χ–νe"
    )

    chi_arr = np.linspace(0.05, 0.49, 40)
    nu_arr  = np.linspace(10.0, 500.0, 40)

    Q_map = np.zeros((len(nu_arr), len(chi_arr)))
    for i, nu in enumerate(nu_arr):
        for j, chi in enumerate(chi_arr):
            try:
                Q_map[i, j] = solve_flory_rehner(chi, nu)
            except Exception:
                Q_map[i, j] = np.nan

    im = ax1.pcolormesh(chi_arr, nu_arr, Q_map,
                        cmap="viridis", vmin=1, vmax=20, shading="auto")
    fig.colorbar(im, ax=ax1, label="Q* (Quellungsgrad)")
    ax1.set_xlabel("Flory-Huggins-Parameter χ")
    ax1.set_ylabel("Vernetzungsdichte νe [mol/m³]")
    ax1.set_title("Phasendiagramm: Gleichgewichtsquellung Q*(χ, νe)")
    # Isolinien
    CS = ax1.contour(chi_arr, nu_arr, Q_map,
                     levels=[2, 3, 5, 8, 12], colors="white", linewidths=0.8)
    ax1.clabel(CS, fmt="Q=%g")

    # Schnitt bei chi=0.25: Q(νe)
    chi_fix = 0.25
    nu_scan = np.linspace(5, 600, 200)
    Q_scan = []
    for nu in nu_scan:
        try:
            Q_scan.append(solve_flory_rehner(chi_fix, nu))
        except Exception:
            Q_scan.append(np.nan)

    ax2.plot(nu_scan, Q_scan, color=COLORS[0], label=f"χ={chi_fix}")

    # Schnitt bei chi=0.15
    chi_fix2 = 0.15
    Q_scan2 = []
    for nu in nu_scan:
        try:
            Q_scan2.append(solve_flory_rehner(chi_fix2, nu))
        except Exception:
            Q_scan2.append(np.nan)
    ax2.plot(nu_scan, Q_scan2, color=COLORS[1], label=f"χ={chi_fix2}")

    ax2.set_xlabel("Vernetzungsdichte νe [mol/m³]")
    ax2.set_ylabel("Gleichgewichts-Quellungsgrad Q*")
    ax2.set_title("Q*(νe) für zwei Lösungsmittel (χ konstant)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 600)
    ax2.set_ylim(1, 30)

    fig.tight_layout()
    savefig(fig, "exp09_flory_rehner_phase_diagram.svg")


# ===========================================================================
# EXPERIMENT 10 – Quellungsdicke (verankert vs. frei)
# ===========================================================================

def exp10_swelling_thickness():
    print("EXP 10: Quellungsdicke")
    fig, (ax1, ax2) = make_fig(
        "Exp. 10 – Schichtdicke nach Quellung: verankert vs. frei"
    )

    Q_vals = np.linspace(1, 10, 300)
    d0 = 200.0

    d_anch = np.array([swelling_to_thickness(d0, q, "anchored") for q in Q_vals])
    d_free = np.array([swelling_to_thickness(d0, q, "free")     for q in Q_vals])

    ax1.plot(Q_vals, d_anch, color=COLORS[0], label="Verankert (d = Q·d₀)")
    ax1.plot(Q_vals, d_free, color=COLORS[1], label=r"Frei (d = Q^{1/3}·d₀)")
    ax1.axhline(d0, color="gray", ls="--", lw=1.0, label=f"Trocken d₀={d0} nm")
    ax1.set_xlabel("Quellungsgrad Q")
    ax1.set_ylabel("Schichtdicke d [nm]")
    ax1.set_title(f"Schichtdicke d(Q) für d₀={d0} nm")
    ax1.legend()
    ax1.grid(True)

    # Kopplungsgleichung: λ_res(d₀) für verschiedene χ
    d0_vals = np.linspace(50, 400, 100)
    chi_list = [0.15, 0.25, 0.35]
    nu_e_fix = 100.0
    for chi in chi_list:
        lam_res_arr = []
        for d0v in d0_vals:
            try:
                lam, Q, n = coupling_equation(chi, nu_e_fix, d0v)
                lam_res_arr.append(lam)
            except Exception:
                lam_res_arr.append(np.nan)
        ax2.plot(d0_vals, lam_res_arr, label=f"χ={chi}")

    ax2.axhspan(380, 780, alpha=0.08, color="green", label="Sichtbarer Bereich")
    ax2.set_xlabel("Trockene Kavitätsdicke d₀ [nm]")
    ax2.set_ylabel("Resonanzwellenlänge λ_res [nm]")
    ax2.set_title(f"Kopplung: λ_res(d₀, χ)  (νe={nu_e_fix} mol/m³)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp10_swelling_thickness.svg")


# ===========================================================================
# EXPERIMENT 11 – Kopplungsgleichung (Gl. 21)
# ===========================================================================

def exp11_coupling():
    print("EXP 11: Kopplungsgleichung")
    fig, (ax1, ax2) = make_fig(
        "Exp. 11 – Zentrale Kopplungsgleichung λ_res = f(χ, νe, d₀) (Gl. 21)"
    )

    # λ_res als Funktion von χ für verschiedene Lösungsmittel
    chi_vals  = np.linspace(0.05, 0.49, 80)
    lm_params = [
        ("Wasser",      1.333, 18.0),
        ("Ethanol",     1.360, 58.4),
        ("Isopropanol", 1.376, 76.9),
        ("n-Butanol",   1.399, 91.5),
    ]
    nu_e_fix = 100.0
    d0_fix   = 150.0

    for sol_name, n_s, V1 in lm_params:
        lam_arr, Q_arr = [], []
        for chi in chi_vals:
            try:
                lam, Q, _ = coupling_equation(chi, nu_e_fix, d0_fix,
                                              n_solvent=n_s, V1=V1)
                lam_arr.append(lam)
                Q_arr.append(Q)
            except Exception:
                lam_arr.append(np.nan)
                Q_arr.append(np.nan)
        ax1.plot(chi_vals, lam_arr, label=sol_name)

    ax1.axhspan(380, 780, alpha=0.08, color="green")
    ax1.set_xlabel("Flory-Huggins-Parameter χ")
    ax1.set_ylabel("Resonanzwellenlänge λ_res [nm]")
    ax1.set_title(f"λ_res(χ) für verschiedene Lösungsmittel (d₀={d0_fix} nm)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 3000)

    # 2D-Karte: λ_res(χ, d₀)
    chi_2d = np.linspace(0.10, 0.45, 30)
    d0_2d  = np.linspace(50, 300, 30)
    LAM    = np.zeros((len(d0_2d), len(chi_2d)))
    for i, d0v in enumerate(d0_2d):
        for j, chi in enumerate(chi_2d):
            try:
                lam, _, _ = coupling_equation(chi, nu_e_fix, d0v)
                LAM[i, j] = lam
            except Exception:
                LAM[i, j] = np.nan

    im = ax2.pcolormesh(chi_2d, d0_2d, LAM, cmap="RdYlGn",
                        vmin=0, vmax=2000, shading="auto")
    fig.colorbar(im, ax=ax2, label="λ_res [nm]")
    CS = ax2.contour(chi_2d, d0_2d, LAM,
                     levels=[380, 450, 550, 650, 780], colors="k", linewidths=0.9)
    ax2.clabel(CS, fmt="%g nm", fontsize=8)
    ax2.set_xlabel("χ")
    ax2.set_ylabel("Trockene Dicke d₀ [nm]")
    ax2.set_title(f"λ_res(χ, d₀) [nm] (νe={nu_e_fix} mol/m³)")

    fig.tight_layout()
    savefig(fig, "exp11_coupling_equation.svg")


# ===========================================================================
# EXPERIMENT 12 – Bethe-Reichweite & Vernetzungsdichte (Def. 5.3)
# ===========================================================================

def exp12_bethe():
    print("EXP 12: Bethe-Reichweite")
    fig, (ax1, ax2) = make_fig(
        "Exp. 12 – E-Beam: Bethe-Reichweite (Gl. 17) und Vernetzungsdichte"
    )

    U_kV = np.linspace(1, 40, 200)
    materials_bethe = [
        ("PDMS (ρ=1.03)",  1.03,  14.0,  7.0),
        ("PMMA (ρ=1.19)",  1.19,  6.36,  3.55),
        ("SiO₂ (ρ=2.20)",  2.20,  20.0,  10.0),
        ("Si   (ρ=2.33)",  2.33,  28.1,  14.0),
    ]
    for name, rho, A, Z in materials_bethe:
        R_e = np.array([bethe_range(u, rho, A, Z) for u in U_kV])
        ax1.semilogy(U_kV, R_e, label=name)

    ax1.axvline(30, color="gray", ls="--", lw=1.0, label="Typisch: 30 kV")
    ax1.set_xlabel("Beschleunigungsspannung U [kV]")
    ax1.set_ylabel("Bethe-Reichweite R_e [µm]")
    ax1.set_title(r"Bethe-Reichweite $R_e \propto U^{1.67}$ für verschiedene Materialien")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both")

    # Vernetzungsdichte als Funktion der Dosis
    dose = np.linspace(0, 200, 300)  # mC/cm²
    nu_e0  = 50.0   # Grundvernetzung [mol/m³]
    alphas = [0.5, 1.0, 2.0, 5.0]

    for alpha in alphas:
        nu_e = dose_to_crosslink_density(dose, nu_e0, alpha)
        ax2.plot(dose, nu_e, label=f"α={alpha} mol·cm²/(m³·mC)")

    ax2.set_xlabel("Elektronenstrahldosis D [mC/cm²]")
    ax2.set_ylabel("Vernetzungsdichte νe [mol/m³]")
    ax2.set_title(r"νe(D) = νe₀ + α·D (Gl. 17)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp12_bethe_range.svg")


# ===========================================================================
# EXPERIMENT 13 – Proximitätsfunktion (Def. 6.2, Gl. 22)
# ===========================================================================

def exp13_proximity_function():
    print("EXP 13: Proximitätsfunktion")
    fig, (ax1, ax2) = make_fig(
        "Exp. 13 – Proximitätsfunktion: Doppel-Gauß-Modell (Def. 6.2, Gl. 22)"
    )

    r_max = 8000.0  # nm
    # Start bei 1 nm statt 0 – semilogy(0) wäre log(0) = -∞
    r = np.linspace(1.0, r_max, 1000)

    params_list = [
        (50.0,  5000.0, 0.75, "Typisch PDMS (β=5000 nm, η=0.75)"),
        (50.0,  3000.0, 0.60, "Silizium (β=3000 nm, η=0.60)"),
        (100.0, 7000.0, 0.85, "High-Z Substrat (β=7000 nm, η=0.85)"),
    ]
    for alpha, beta, eta, label in params_list:
        f_r = np.array([proximity_function(ri, 0, alpha=alpha, beta=beta, eta_back=eta)
                        for ri in r])
        f_r = np.maximum(f_r, 1e-20)   # sicherstellen: alle Werte > 0 für log-Achse
        ax1.semilogy(r / 1000, f_r, label=label)

    ax1.set_xlabel("Radius r [µm]")
    ax1.set_ylabel("f(r) [nm⁻²] (log)")
    ax1.set_title("Proximitätsfunktion f(r): Vorwärts- und Rückstreuung")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both")
    ax1.set_xlim(0, r_max / 1000)

    # 2D-Proximity-Kernel
    N = 64
    pixel_size = 100.0  # nm – gröberes Grid für schnelle Berechnung
    K = proximity_kernel_2d(N, pixel_size=pixel_size, alpha=50.0, beta=5000.0, eta_back=0.75)
    center_row = K[N // 2, :]
    x_um = (np.arange(N) - N // 2) * pixel_size / 1000.0

    # Horizontalschnitt (semilogy, linke y-Achse)
    ax2.semilogy(x_um, np.maximum(center_row, 1e-20), color=COLORS[0],
                 lw=2.0, label="Horizontalschnitt (y=0)")
    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("Kernel-Amplitude (log)")
    ax2.set_title(f"2D-Proximity-Kernel: Schnitt + Heatmap\n"
                  f"(N={N}×{N}, px={pixel_size:.0f} nm, α=50 nm, β=5000 nm)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both")

    # Inset-Achse für 2D-Heatmap (kein twinx – das überlagert die log-Achse störend)
    inset = ax2.inset_axes([0.52, 0.52, 0.44, 0.44])
    extent_um = [-N * pixel_size / 2e3, N * pixel_size / 2e3,
                 -N * pixel_size / 2e3, N * pixel_size / 2e3]
    inset.imshow(np.log10(np.maximum(K, 1e-20)),
                 extent=extent_um, origin="lower", cmap="hot", aspect="equal")
    inset.set_title("log₁₀ f(x,y)", fontsize=8)
    inset.set_xlabel("x [µm]", fontsize=7)
    inset.set_ylabel("y [µm]", fontsize=7)
    inset.tick_params(labelsize=6)

    fig.tight_layout()
    savefig(fig, "exp13_proximity_function.svg")


# ===========================================================================
# EXPERIMENT 14 – Proximity-Correction: Vor und nach (Alg. 6.1)
# ===========================================================================

def exp14_proximity():
    print("EXP 14: Proximity-Correction")
    fig, (ax1, ax2) = make_fig(
        "Exp. 14 – Proximity-Correction: Dosisverschmierung und Korrektur"
    )

    N = 64
    pixel_size = 100.0
    D0 = 1.0

    # Einfaches Muster: drei Rechtecke
    dose = np.zeros((N, N))
    dose[10:30, 10:25] = D0
    dose[10:30, 30:45] = D0
    dose[10:30, 50:60] = D0

    dose_eff = apply_proximity(dose, pixel_size=pixel_size,
                                alpha=50.0, beta=5000.0, eta_back=0.75)

    x = np.arange(N) * pixel_size / 1000.0  # µm

    # Horizontalschnitt in der Mitte der Rechtecke
    row = 20
    ax1.plot(x, dose[row, :],     color=COLORS[0], lw=2.0, label="Nominale Dosis")
    ax1.plot(x, dose_eff[row, :], color=COLORS[3], lw=2.0, label="Effektive Dosis (nach Faltung)")
    ax1.fill_between(x, 0, dose[row, :],     alpha=0.2, color=COLORS[0])
    ax1.fill_between(x, 0, dose_eff[row, :], alpha=0.2, color=COLORS[3])
    ax1.set_xlabel("x [µm]")
    ax1.set_ylabel("Dosis [a.u.]")
    ax1.set_title("Proximity-Verschmierung: Horizontalschnitt (Zeile 20)")
    ax1.legend()
    ax1.grid(True)

    # 2D-Bild: Original vs. verschmiert
    cmap = "plasma"
    combined = np.concatenate([dose, dose_eff], axis=1)
    ax2.imshow(combined, cmap=cmap, aspect="equal", origin="lower",
               extent=[0, 2 * N * pixel_size / 1000, 0, N * pixel_size / 1000])
    ax2.axvline(N * pixel_size / 1000, color="white", lw=1.5, ls="--")
    ax2.text(N * pixel_size / 2000, N * pixel_size / 1000 * 0.95,
             "Original", color="white", ha="center", va="top", fontsize=10)
    ax2.text(N * pixel_size / 1000 + N * pixel_size / 2000, N * pixel_size / 1000 * 0.95,
             "Effektiv", color="white", ha="center", va="top", fontsize=10)
    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("y [µm]")
    ax2.set_title("2D: Nominale (links) vs. verschmierte Dosis (rechts)")

    fig.tight_layout()
    savefig(fig, "exp14_proximity_correction.svg")


# ===========================================================================
# EXPERIMENT 15 – IPEC-Konvergenz (Alg. 6.1, Satz 6.2)
# ===========================================================================

def exp15_ipec():
    print("EXP 15: IPEC-Konvergenz")
    fig, (ax1, ax2) = make_fig(
        "Exp. 15 – IPEC: Iterative Proximity-Korrektur (Alg. 6.1, Satz 6.2)"
    )

    N = 32
    pixel_size = 100.0

    # Zieldosisverlauf: zwei Linien
    dose_target = np.zeros((N, N))
    dose_target[8:24, 8:14] = 1.0
    dose_target[8:24, 18:24] = 1.0

    # IPEC-Läufe mit verschiedenen Relaxationsparametern ω
    omega_vals = [0.3, 0.5, 0.7]
    colors_ipec = [COLORS[0], COLORS[1], COLORS[2]]

    for omega, col in zip(omega_vals, colors_ipec):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = proximity_correction_ipec(
                dose_target, pixel_size=pixel_size,
                alpha=50.0, beta=5000.0, eta_back=0.75,
                max_iterations=100, omega=omega, tolerance=1e-4,
            )
        ax1.semilogy(result.error_history, color=col, lw=1.5,
                     label=f"ω={omega}, konverg.={result.converged}")

    ax1.set_xlabel("Iteration i")
    ax1.set_ylabel("Fehler ‖D_eff − D_soll‖ (log)")
    ax1.set_title("IPEC-Konvergenz für verschiedene Relaxationsparameter ω")
    ax1.legend()
    ax1.grid(True, which="both")

    # Vorher/nachher: Fehler im effektiven Dosismuster
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_best = proximity_correction_ipec(
            dose_target, pixel_size=pixel_size,
            alpha=50.0, beta=5000.0, eta_back=0.75,
            max_iterations=100, omega=0.5, tolerance=1e-4,
        )

    D_nom_corrected = result_best.dose_nominal
    D_eff_corrected = apply_proximity(D_nom_corrected, pixel_size=pixel_size,
                                       alpha=50.0, beta=5000.0, eta_back=0.75)

    row = 16
    x = np.arange(N) * pixel_size / 1000.0
    ax2.plot(x, dose_target[row, :],    color="k",          lw=2.0, label="Zieldosis")
    ax2.plot(x, D_nom_corrected[row, :], color=COLORS[0],   lw=1.5, ls="--", label="Nominale Dosis (IPEC)")
    ax2.plot(x, D_eff_corrected[row, :], color=COLORS[3],   lw=1.5, label="Effektive Dosis (nach Korrektur)")
    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("Dosis [a.u.]")
    ax2.set_title("IPEC-Ergebnis: Ziel vs. nominal vs. effektiv (Zeile 16)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp15_ipec_convergence.svg")


# ===========================================================================
# EXPERIMENT 16 – Kolorimetrie: CIE-XYZ und Lab
# ===========================================================================

def exp16_colorimetry():
    print("EXP 16: CIE-Kolorimetrie")
    fig, (ax1, ax2) = make_fig(
        "Exp. 16 – CIE-Kolorimetrie: XYZ-Tristimulus und L*a*b*"
    )

    lam = wavelengths_visible(300)
    V   = photopic_sensitivity(lam)

    ax1.plot(lam, V, color=COLORS[2], lw=2.0, label="V(λ) photopisch")

    # FP-Spektren für verschiedene Kavitätsdicken → Farben im CIE-Diagramm
    lam_vis = wavelengths_visible(200)
    n_cav   = 1.41
    R_mir   = 0.45

    L_stars, a_stars, b_stars, colors_rgb = [], [], [], []
    for d in np.linspace(70, 280, 12):
        R_fp = fabry_perot_reflection(lam_vis, complex(n_cav), d, R_mir, R_mir)
        X, Y, Z = spectrum_to_xyz(lam_vis, R_fp)
        L, a, b = xyz_to_lab(X, Y, Z)
        L_stars.append(L)
        a_stars.append(a)
        b_stars.append(b)

        # Grobe RGB-Näherung für Punktfarbe
        r_c = np.clip(X * 3.2 - Y * 1.5 - Z * 0.5, 0, 1)
        g_c = np.clip(-X * 0.97 + Y * 1.9 + Z * 0.04, 0, 1)
        b_c = np.clip(X * 0.06 - Y * 0.23 + Z * 1.1, 0, 1)
        colors_rgb.append((r_c, g_c, b_c))

    ax1.set_xlabel("Wellenlänge λ [nm]")
    ax1.set_ylabel("V(λ) Photopische Empfindlichkeit")
    ax1.set_title("Photopische Kurve V(λ) + Beispiel-FP-Spektren")

    # Ausgewählte Spektren einblenden
    for d_ex, col_ex in zip([100, 160, 220], ["blue", "green", "red"]):
        R_ex = fabry_perot_reflection(lam, complex(n_cav), d_ex, R_mir, R_mir)
        ax1.fill_between(lam, 0, R_ex * 0.8, alpha=0.25, color=col_ex)
    ax1.legend()
    ax1.grid(True)

    # CIE Lab: a*-b*-Diagramm
    sc = ax2.scatter(a_stars, b_stars, c=list(range(len(a_stars))),
                     cmap="hsv", s=80, zorder=5)
    for i, (a, b, c_rgb) in enumerate(zip(a_stars, b_stars, colors_rgb)):
        ax2.plot(a, b, "o", ms=10, color=c_rgb, markeredgecolor="k",
                 markeredgewidth=0.5)

    ax2.set_xlabel("a* (Rot-Grün)")
    ax2.set_ylabel("b* (Gelb-Blau)")
    ax2.set_title("CIE-L*a*b*: FP-Resonatoren bei Dicken d=70–280 nm")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.axvline(0, color="gray", lw=0.8)
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp16_colorimetry_cie.svg")


# ===========================================================================
# EXPERIMENT 17 – ΔE und Tarnindex (Def. 7.2, Def. 9.1)
# ===========================================================================

def exp17_delta_e():
    print("EXP 17: ΔE und Tarnindex")
    fig, (ax1, ax2) = make_fig(
        "Exp. 17 – Farbdifferenz ΔE (Def. 7.2) und Tarnindex C (Def. 9.1)"
    )

    lam = wavelengths_visible(200)
    n_cav = 1.41
    R_mir = 0.45
    d_ref = 160.0  # Referenzresonator

    # ΔE zwischen Referenz und verschiedenen Dicken
    R_ref = fabry_perot_reflection(lam, complex(n_cav), d_ref, R_mir, R_mir)
    L_r, a_r, b_r = spectrum_to_lab(lam, R_ref)

    d_vals  = np.linspace(80, 280, 60)
    dE_vals = []
    C_vals  = []
    lam_res_vals = []

    for d in d_vals:
        R_d = fabry_perot_reflection(lam, complex(n_cav), d, R_mir, R_mir)
        L_d, a_d, b_d = spectrum_to_lab(lam, R_d)
        dE = delta_e((L_r, a_r, b_r), (L_d, a_d, b_d))
        dE_vals.append(dE)
        # Tarnindex: Hintergrund = grünes Blatt (Ziel)
        R_hg = target_spectrum_leaf(lam)
        C = camouflage_index(lam, R_d, R_hg, R_ref)
        C_vals.append(C)
        lam_res_vals.append(fabry_perot_resonance_wavelength(n_cav, d, order=1))

    ax1.plot(d_vals, dE_vals, color=COLORS[0], lw=2.0, label=r"$\Delta E$ (Def. 7.2)")
    ax1.axhline(2.0, color="red", ls="--", lw=1.0, label="JND = 2 (gerade sichtbar)")
    ax1.axhline(5.0, color="orange", ls="--", lw=1.0, label="ΔE = 5 (deutlich)")
    ax1.axvline(d_ref, color="gray", ls=":", lw=1.0, label=f"Referenz d={d_ref} nm")
    ax1.set_xlabel("Kavitätsdicke d [nm]")
    ax1.set_ylabel(r"$\Delta E_{ab}^*$")
    ax1.set_title(f"Farbdifferenz ΔE zur Referenz (d_ref={d_ref} nm)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(d_vals, C_vals, color=COLORS[2], lw=2.0)
    ax2.axhline(0.0, color="gray", lw=0.8)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax2.fill_between(d_vals, 0, C_vals, alpha=0.2, color=COLORS[2])
    ax2.set_xlabel("Kavitätsdicke d [nm]")
    ax2.set_ylabel("Tarnindex C ∈ [0, 1]")
    ax2.set_title("Tarnindex C vs. Blatt-Hintergrund (Def. 9.1)")
    ax2.set_ylim(-0.1, 1.15)
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp17_delta_e_camouflage.svg")


# ===========================================================================
# EXPERIMENT 18 – Verlustfunktional und Adjoint-Gradient (Gl. 25–29)
# ===========================================================================

def exp18_loss_gradient():
    print("EXP 18: Verlust & Adjoint-Gradient")
    fig, (ax1, ax2) = make_fig(
        "Exp. 18 – Verlustfunktional (Gl. 25) und Adjoint-Gradient (Satz 7.2)"
    )

    n_poly, n_solv = 1.41, 1.333
    Q_fix = 2.5
    n_fix = n_from_swelling(n_poly, n_solv, Q_fix)

    lam = wavelengths_visible(150)
    R_target = target_spectrum_gaussian(lam, center_nm=550.0, fwhm_nm=40.0)

    # Physikalisch sinnvoller d-Bereich:
    # FP-Resonanz m=1: λ_res = 2·n·d → d = λ/(2n)
    # Bei n_fix≈1.375: d_res(550nm) ≈ 200 nm → d_vals um diesen Wert herum
    d_res_expected = 550.0 / (2.0 * n_fix)
    d_vals = np.linspace(0.4 * d_res_expected, 2.2 * d_res_expected, 120)

    L_unif, L_perc = [], []
    for d in d_vals:
        R_fp = fabry_perot_reflection(lam, complex(n_fix), d, 0.45, 0.45)
        L_unif.append(spectral_loss(R_fp, R_target, lam))
        L_perc.append(perceptual_loss(R_fp, R_target, lam))

    L_unif = np.array(L_unif)
    L_perc = np.array(L_perc)
    d_min  = d_vals[np.argmin(L_unif)]

    ax1.plot(d_vals, L_unif, color=COLORS[0], lw=2.0, label="L_unif (Gl. 25)")
    ax1.plot(d_vals, L_perc, color=COLORS[1], lw=2.0, label="L_perc (photopisch)")
    ax1.axvline(d_min, color=COLORS[0], ls="--", lw=1.2,
                label=f"Min L_unif bei d={d_min:.0f} nm")
    ax1.axvline(d_res_expected, color="gray", ls=":", lw=1.0,
                label=f"λ/2n = {d_res_expected:.0f} nm (Resonanz m=1)")
    ax1.set_xlabel("Kavitätsdicke d [nm]")
    ax1.set_ylabel("Verlust L(d)")
    ax1.set_title(f"Verlustfunktional L(d) – Ziel: Gauß λ=550 nm, Q={Q_fix}")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── Gradienten via zentraler finiter Differenz (robust, kein Signaturproblem) ──
    eps_d  = 1.0    # nm
    eps_Q  = 0.02   # dimensionslos
    grad_d, grad_Q = [], []

    for d in d_vals:
        # ∂L/∂d  (d = geschwollene Kavitätsdicke)
        Rp = fabry_perot_reflection(lam, complex(n_fix), d + eps_d, 0.45, 0.45)
        Rm = fabry_perot_reflection(lam, complex(n_fix), d - eps_d, 0.45, 0.45)
        grad_d.append(
            (spectral_loss(Rp, R_target, lam) - spectral_loss(Rm, R_target, lam))
            / (2 * eps_d)
        )

        # ∂L/∂Q  (Q verändert n_eff und d_sw = d0·Q gleichzeitig; d0 = d/Q = const)
        d0 = d / Q_fix
        n_p = n_from_swelling(n_poly, n_solv, Q_fix + eps_Q)
        n_m = n_from_swelling(n_poly, n_solv, Q_fix - eps_Q)
        Rp2 = fabry_perot_reflection(lam, complex(n_p), d0*(Q_fix+eps_Q), 0.45, 0.45)
        Rm2 = fabry_perot_reflection(lam, complex(n_m), d0*(Q_fix-eps_Q), 0.45, 0.45)
        grad_Q.append(
            (spectral_loss(Rp2, R_target, lam) - spectral_loss(Rm2, R_target, lam))
            / (2 * eps_Q)
        )

    grad_d = np.array(grad_d)
    grad_Q = np.array(grad_Q)

    ax2.plot(d_vals, grad_d, color=COLORS[3], lw=2.0, label=r"$\partial L/\partial d$")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(d_min, color=COLORS[0], ls="--", lw=1.0, alpha=0.6,
                label=f"d_min = {d_min:.0f} nm")
    ax2.set_xlabel("Kavitätsdicke d [nm]")
    ax2.set_ylabel(r"$\partial L/\partial d$  [nm⁻¹]", color=COLORS[3])
    ax2.tick_params(axis="y", labelcolor=COLORS[3])
    ax2.grid(True)

    ax2r = ax2.twinx()
    ax2r.plot(d_vals, grad_Q, color=COLORS[4], lw=2.0, ls="-.",
              label=r"$\partial L/\partial Q$")
    ax2r.axhline(0, color=COLORS[4], lw=0.4, ls=":")
    ax2r.set_ylabel(r"$\partial L/\partial Q$", color=COLORS[4])
    ax2r.tick_params(axis="y", labelcolor=COLORS[4])

    ax2.set_title(
        "Adjoint-Gradienten ∂L/∂d und ∂L/∂Q (zentrale FD)\n"
        "Vorzeichenwechsel bei d_min = notwendige Optimalitätsbedingung"
    )
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    savefig(fig, "exp18_spectral_loss_gradient.svg")


# ===========================================================================
# EXPERIMENT 19 – Optimierer: Konvergenz (Alg. 8.1)
# ===========================================================================

def exp19_optimizer():
    print("EXP 19: Optimierer-Konvergenz")
    fig, (ax1, ax2) = make_fig(
        "Exp. 19 – Gradientenabstieg-Optimierer (Alg. 8.1, Satz 8.1)"
    )

    lam = wavelengths_visible(120)

    targets = [
        ("Grün (550 nm)",  target_spectrum_gaussian(lam, 550.0, 40.0),  COLORS[1]),
        ("Rot   (650 nm)", target_spectrum_gaussian(lam, 650.0, 40.0),  COLORS[3]),
        ("Blau  (450 nm)", target_spectrum_gaussian(lam, 450.0, 40.0),  COLORS[0]),
        ("Blatt",          target_spectrum_leaf(lam),                   COLORS[2]),
    ]

    for label, R_t, col in targets:
        result = optimize_resonator(
            lam, R_t,
            d0_init=200.0, Q_init=2.0,
            max_iterations=80,
            n_polymer=1.41, n_solvent=1.333,
        )
        ax1.semilogy(result.loss_history, color=col, label=label)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Verlust L(θ) (log)")
    ax1.set_title("Konvergenzverläufe für verschiedene Zielspektren")
    ax1.legend()
    ax1.grid(True, which="both")

    # Erreichtes Spektrum vs. Ziel für Grün
    R_target_green = target_spectrum_gaussian(lam, 550.0, 40.0)
    result_green = optimize_resonator(
        lam, R_target_green,
        d0_init=200.0, Q_init=2.0,
        max_iterations=100,
        n_polymer=1.41, n_solvent=1.333,
    )

    n_opt = n_from_swelling(1.41, 1.333, result_green.Q)
    R_opt = fabry_perot_reflection(lam, complex(n_opt),
                                   result_green.thickness, 0.45, 0.45)

    ax2.plot(lam, R_target_green, color="k", lw=2.5, ls="--", label="Ziel R(λ)")
    ax2.plot(lam, R_opt, color=COLORS[1], lw=2.0,
             label=f"Optimiert: d*={result_green.thickness:.0f} nm, "
                   f"Q*={result_green.Q:.2f}, ΔE={result_green.delta_e_final:.1f}")
    ax2.fill_between(lam, R_target_green, R_opt, alpha=0.2, color=COLORS[3],
                     label=f"Differenz |R−R_soll|")
    ax2.set_xlabel("Wellenlänge λ [nm]")
    ax2.set_ylabel("Reflexion R")
    ax2.set_title("Optimiertes Spektrum: Ziel = Gauß bei λ=550 nm")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, "exp19_optimizer_convergence.svg")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Aktive Nano-Optik – Experimente & Validierung")
    print(f"Ausgabeverzeichnis: {OUTDIR}")
    print("=" * 60)

    experiments = [
        exp01_fresnel,
        exp02_tmm_energy,
        exp03_fp_spectrum,
        exp04_fp_finesse,
        exp05_tmm_inhomogeneous,
        exp06_lorentz,
        exp07_lorentz_lorenz,
        exp08_flory_rehner,
        exp09_phase_diagram,
        exp10_swelling_thickness,
        exp11_coupling,
        exp12_bethe,
        exp13_proximity_function,
        exp14_proximity,
        exp15_ipec,
        exp16_colorimetry,
        exp17_delta_e,
        exp18_loss_gradient,
        exp19_optimizer,
    ]

    failed = []
    for exp in experiments:
        try:
            exp()
        except Exception as e:
            print(f"  ✗ FEHLER in {exp.__name__}: {e}")
            failed.append((exp.__name__, str(e)))

    print("\n" + "=" * 60)
    print(f"Fertig: {len(experiments) - len(failed)}/{len(experiments)} erfolgreich")
    if failed:
        print("Fehlgeschlagene Experimente:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"SVG-Dateien in: {OUTDIR}")
    print("=" * 60)