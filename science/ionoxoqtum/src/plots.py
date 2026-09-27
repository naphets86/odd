"""
plots.py
=======================
Visualisierungsmodul für das HydroPhotonic Framework.

Erzeugt:
  - Fabry-Pérot-Reflexionsspektren für verschiedene Quellungsgrade
  - IoFET-Kennlinien (Transfer- und Ausgangskennlinien)
  - EWOD-Kontaktwinkelverlauf
  - Quellungskinetik
  - EBL-PSF und Proximity-Korrektur
  - Optimierungsverlauf
  - CIE-Farbgamut
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Dict, Tuple

# Farbpalette
PALETTE = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93", "#1d3557"]


def plot_fp_spectra(
    resonator,
    Q_values: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reflexionsspektren des FP-Resonators für verschiedene Quellungsgrade.

    Parameters
    ----------
    resonator : FabryPerotResonator
    Q_values : list[float], optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if Q_values is None:
        Q_values = [1.0, 1.8, 2.5, 3.0, 3.8]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Spectral_r(np.linspace(0, 1, len(Q_values)))

    for Q, color in zip(Q_values, colors):
        lam, R = resonator.reflection_spectrum(Q=Q)
        lam_res_nm = resonator.resonance_wavelength(Q) * 1e9
        label = f"Q={Q:.1f}  (λ_res={lam_res_nm:.0f} nm)"
        ax.plot(lam, R, color=color, linewidth=2, label=label)

    ax.set_xlabel("Wellenlänge λ (nm)", fontsize=12)
    ax.set_ylabel("Reflexion R", fontsize=12)
    ax.set_title("Fabry-Pérot-Reflexionsspektren bei verschiedenen Quellungsgraden", fontsize=13)
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ewod_characteristics(
    ewod_system,
    v_max: float = 2.0,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """EWOD-Kontaktwinkelverlauf und Energiekurve.

    Parameters
    ----------
    ewod_system : EWODSystem
    v_max : float
        Maximale Spannung [V].

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    V, theta = ewod_system.voltage_sweep(v_max=v_max)

    ax1.plot(V, theta, color=PALETTE[3], linewidth=2.5)
    ax1.axhline(ewod_system.theta0_deg, color="gray", linestyle="--", label=f"θ₀={ewod_system.theta0_deg:.0f}°")
    ax1.set_xlabel("Spannung V (V)", fontsize=11)
    ax1.set_ylabel("Kontaktwinkel θ (°)", fontsize=11)
    ax1.set_title("Young-Lippmann: θ(V)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Schaltenergie
    electrode_area = (100e-9)**2
    E_values = np.array([ewod_system.switching_energy(v, electrode_area) for v in V[1:]])
    ax2.semilogy(V[1:], E_values * 1e21, color=PALETTE[0], linewidth=2.5)
    ax2.set_xlabel("Spannung V (V)", fontsize=11)
    ax2.set_ylabel("Schaltenergie E (zJ)", fontsize=11)
    ax2.set_title("Schaltenergie E(V)", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"EWOD-System (η={ewod_system.ewod_efficiency:.1f} V⁻²)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_iofet_characteristics(
    transistor,
    V_gs_values: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """IoFET-Kennlinien: Transfer und Ausgang.

    Parameters
    ----------
    transistor : IoFET
    V_gs_values : list[float], optional
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    if V_gs_values is None:
        V_gs_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Transferkennlinie
    V_gs, I_ds_lin = transistor.transfer_characteristics(V_ds=0.1)
    _, I_ds_log = transistor.transfer_characteristics(V_ds=0.1)
    ax1.semilogy(V_gs, np.abs(I_ds_log) + 1e-30, color=PALETTE[3], linewidth=2)
    ax1.axvline(transistor.V_T, color="red", linestyle="--", alpha=0.7, label=f"V_T={transistor.V_T} V")
    ax1.set_xlabel("V_GS (V)", fontsize=11)
    ax1.set_ylabel("|I_DS| (A)  [log]", fontsize=11)
    ax1.set_title("Transferkennlinie (log)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # Ausgangskennlinien
    output = transistor.output_characteristics(V_gs_values)
    colors_out = plt.cm.viridis(np.linspace(0.2, 0.9, len(V_gs_values)))
    for (V_gs_val, (V_ds, I_ds)), col in zip(output.items(), colors_out):
        ax2.plot(V_ds, I_ds * 1e9, color=col, linewidth=2, label=f"V_GS={V_gs_val:.1f}V")
    ax2.set_xlabel("V_DS (V)", fontsize=11)
    ax2.set_ylabel("I_DS (nA)", fontsize=11)
    ax2.set_title("Ausgangskennlinienfeld", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"IoFET-Kennlinien (σ={transistor.sigma} S/m)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_swelling_kinetics(
    polymer_model,
    h_values_nm: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Quellungskinetik für verschiedene Schichtdicken.

    Returns
    -------
    plt.Figure
    """
    if h_values_nm is None:
        h_values_nm = [200, 500, 1000, 2000]

    Q_eq = polymer_model.equilibrium_swelling()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(h_values_nm)))

    for h_nm, color in zip(h_values_nm, colors):
        h = h_nm * 1e-9
        tau = polymer_model.relaxation_time(h)
        t, Q_t = polymer_model.kinetic_swelling(h, Q_init=1.0, Q_eq=Q_eq)
        t_ms = t * 1e3
        ax1.plot(t_ms, Q_t, color=color, linewidth=2, label=f"h={h_nm} nm")

    ax1.set_xlabel("Zeit (ms)", fontsize=11)
    ax1.set_ylabel("Quellungsgrad Q", fontsize=11)
    ax1.set_title("Quellungskinetik Q(t)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Relaxationszeit vs. Dicke
    h_arr = np.linspace(50, 2000, 200) * 1e-9
    tau_arr = np.array([polymer_model.relaxation_time(h) * 1e3 for h in h_arr])
    ax2.loglog(h_arr * 1e9, tau_arr, color=PALETTE[2], linewidth=2.5)
    ax2.set_xlabel("Schichtdicke h (nm)", fontsize=11)
    ax2.set_ylabel("Relaxationszeit τ_Q (ms)", fontsize=11)
    ax2.set_title("τ_Q vs. Schichtdicke", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Flory-Rehner Quellungskinetik (χ={polymer_model.p.chi:.2f})", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ebl_psf(
    proximity_model,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """EBL-PSF: 1D-Profil und 2D-Karte.

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # 1D-PSF
    r = np.linspace(0, 500, 1000)
    psf_1d = proximity_model.psf_1d(r)
    ax1.semilogy(r, psf_1d / psf_1d.max(), color=PALETTE[0], linewidth=2)
    bf_nm = proximity_model.params.beta_f * 1e9
    bb_nm = proximity_model.params.beta_b * 1e9
    ax1.axvline(bf_nm, color="blue", linestyle="--", alpha=0.7, label=f"β_f={bf_nm:.0f}nm")
    ax1.axvline(bb_nm, color="red", linestyle="--", alpha=0.7, label=f"β_b={bb_nm:.0f}nm")
    ax1.set_xlabel("r (nm)", fontsize=11)
    ax1.set_ylabel("PSF (norm.)", fontsize=11)
    ax1.set_title("Chang-PSF (1D)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # 2D-PSF (Ausschnitt)
    psf_2d = proximity_model.psf_2d()
    n = proximity_model.n_grid
    c = n // 2
    zoom = 20
    psf_zoom = psf_2d[c-zoom:c+zoom, c-zoom:c+zoom]
    im2 = ax2.imshow(np.log1p(psf_zoom / psf_zoom.max() * 1000),
                     cmap="inferno", origin="lower")
    ax2.set_title("Chang-PSF 2D (log)", fontsize=12)
    ax2.set_xlabel("x (Gitter)", fontsize=11)
    ax2.set_ylabel("y (Gitter)", fontsize=11)
    plt.colorbar(im2, ax=ax2, label="log(PSF+1)")

    # Proximity-Korrektur Beispiel
    rng = np.random.default_rng(0)
    target = np.zeros((n, n))
    target[n//3:2*n//3, n//3:2*n//3] = 1.0
    corrected = proximity_model.proximity_correction(target)
    ax3.imshow(corrected, cmap="viridis", origin="lower")
    ax3.set_title("Proximity-Korrigiertes Muster", fontsize=12)
    ax3.set_xlabel("x (Gitter)", fontsize=11)
    ax3.set_ylabel("y (Gitter)", fontsize=11)

    fig.suptitle("EBL Proximity-Effekte (Chang-Modell)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_optimization_result(
    opt_result,
    designer,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Optimierungsergebnis: Spektrenvergleich und Konvergenzkurve.

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    R_pred = designer.predict_spectrum(opt_result)
    ax1.plot(designer.lam_nm, designer.R_target, color="black",
             linewidth=2.5, linestyle="--", label="Ziel R_Ziel(λ)")
    ax1.plot(designer.lam_nm, R_pred, color=PALETTE[0],
             linewidth=2, label=f"Optimiert (RMSE={opt_result.final_loss**0.5:.4f})")
    ax1.set_xlabel("Wellenlänge λ (nm)", fontsize=11)
    ax1.set_ylabel("Reflexion R", fontsize=11)
    ax1.set_title("Inverses Design: Ziel vs. Ergebnis", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(380, 780)

    ax2.semilogy(opt_result.loss_history, color=PALETTE[3], linewidth=2)
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Kostenfunktion L (log)", fontsize=11)
    ax2.set_title("Konvergenzkurve", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Adjungierter Gradientenabstieg – Inverses Photonisches Design", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_system_overview(
    fp_resonator,
    polymer_model,
    ewod_system,
    transistor,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Übersichtsplot aller Systemkomponenten (2×3-Layout).

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. FP-Spektren
    ax1 = fig.add_subplot(gs[0, 0])
    for Q, color in zip([1.0, 2.0, 3.0, 4.0],
                         ["red", "orange", "green", "blue"]):
        lam, R = fp_resonator.reflection_spectrum(Q=Q)
        ax1.plot(lam, R, color=color, linewidth=1.8, label=f"Q={Q:.0f}")
    ax1.set_title("FP-Spektren", fontsize=11)
    ax1.set_xlabel("λ (nm)", fontsize=10)
    ax1.set_ylabel("R", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. EWOD-Kennlinie
    ax2 = fig.add_subplot(gs[0, 1])
    V, theta = ewod_system.voltage_sweep(v_max=2.0)
    ax2.plot(V, theta, color=PALETTE[3], linewidth=2)
    ax2.set_title("EWOD: θ(V)", fontsize=11)
    ax2.set_xlabel("V (V)", fontsize=10)
    ax2.set_ylabel("θ (°)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Transferkennlinie IoFET
    ax3 = fig.add_subplot(gs[0, 2])
    V_gs, I_ds = transistor.transfer_characteristics(V_ds=0.1)
    ax3.semilogy(V_gs, np.abs(I_ds) + 1e-30, color=PALETTE[2], linewidth=2)
    ax3.set_title("IoFET Transfer", fontsize=11)
    ax3.set_xlabel("V_GS (V)", fontsize=10)
    ax3.set_ylabel("|I_DS| (A)", fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    # 4. Quellungsgrad vs. chi
    ax4 = fig.add_subplot(gs[1, 0])
    chi_arr, Q_arr = polymer_model.swelling_vs_chi()
    ax4.plot(chi_arr, Q_arr, color=PALETTE[4], linewidth=2)
    ax4.set_title("Quellung Q(χ)", fontsize=11)
    ax4.set_xlabel("χ (Flory-Huggins)", fontsize=10)
    ax4.set_ylabel("Q_eq", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. λ_res vs. Q
    ax5 = fig.add_subplot(gs[1, 1])
    Q_range = np.linspace(1.0, 4.5, 100)
    lam_res = np.array([
        polymer_model.resonance_wavelength(fp_resonator.d0, Q) * 1e9
        for Q in Q_range
    ])
    ax5.plot(Q_range, lam_res, color=PALETTE[0], linewidth=2)
    ax5.axhspan(380, 780, alpha=0.1, color="yellow", label="sichtbar")
    ax5.set_title("Resonanzwellenlänge λ(Q)", fontsize=11)
    ax5.set_xlabel("Quellungsgrad Q", fontsize=10)
    ax5.set_ylabel("λ_res (nm)", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Quellungskinetik
    ax6 = fig.add_subplot(gs[1, 2])
    Q_eq = polymer_model.equilibrium_swelling()
    for h_nm, color in zip([200, 500, 1000], ["#2196F3", "#FF9800", "#4CAF50"]):
        t, Q_t = polymer_model.kinetic_swelling(h_nm * 1e-9, 1.0, Q_eq)
        ax6.plot(t * 1e3, Q_t, color=color, linewidth=1.8, label=f"{h_nm} nm")
    ax6.set_title("Quellungskinetik", fontsize=11)
    ax6.set_xlabel("Zeit (ms)", fontsize=10)
    ax6.set_ylabel("Q(t)", fontsize=10)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        "HydroPhotonic Framework – Systemübersicht\n"
        "Effiziente Nanostrukturen mit Wasser und Licht",
        fontsize=14, fontweight="bold"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
