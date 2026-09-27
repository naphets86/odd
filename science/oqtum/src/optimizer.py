"""
nanooptik.optimizer
===================
Inverses Optimierungsproblem: Optimale Resonatorgestaltung
gemäß Kapitel 7–8 der wissenschaftlichen Arbeit.

Implementiert:
  - Definition 7.1:  Spektrales Verlustfunktional (Gl. 25)
  - Definition 7.2:  Perceptuelles Verlustfunktional
  - Satz 7.1:        Analytischer Gradient ∂R/∂d₀ (Gl. 27–28)
  - Satz 7.2:        Adjungierte Methode (Gl. 29)
  - Algorithmus 8.1: Optimale Resonatorgestaltung (Gradientenabstieg)
  - Satz 8.1:        Konvergenzgarantie
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple, Callable
from dataclasses import dataclass, field

from src.optics import fabry_perot_reflection, fabry_perot_resonance_wavelength
from src.polymer import n_from_swelling, swelling_to_thickness, solve_flory_rehner
from src.colorimetry import (
    photopic_sensitivity, spectrum_to_lab, delta_e, wavelengths_visible
)


# ---------------------------------------------------------------------------
# Verlustfunktionale (Definitionen 7.1 und 7.2)
# ---------------------------------------------------------------------------

def spectral_loss(
    R_computed: NDArray,
    R_target: NDArray,
    weights: Optional[NDArray] = None,
    wavelengths: Optional[NDArray] = None,
) -> float:
    """
    Spektrales Verlustfunktional (Gleichung 25):

    L(θ) = ∫ w(λ)|R(λ;θ) - R_soll(λ)|² dλ

    Parameters
    ----------
    R_computed : ndarray
        Berechnetes Reflexionsspektrum.
    R_target : ndarray
        Zielspektrum R_soll.
    weights : ndarray, optional
        Gewichtsfaktor w(λ). Standard: gleichgewichtig (w ≡ 1).
    wavelengths : ndarray, optional
        Wellenlängen für Trapezintegration. Standard: uniformes Gitter.

    Returns
    -------
    loss : float
        Verlustfunktional ≥ 0.
    """
    R_c = np.asarray(R_computed, dtype=float)
    R_t = np.asarray(R_target,   dtype=float)

    if weights is None:
        weights = np.ones_like(R_c)

    diff_sq = weights * (R_c - R_t) ** 2

    if wavelengths is not None:
        return float(np.trapezoid(diff_sq, wavelengths))
    else:
        return float(np.mean(diff_sq))


def perceptual_loss(
    wavelengths: NDArray,
    R_computed: NDArray,
    R_target: NDArray,
) -> float:
    """
    Perceptuelles Verlustfunktional mit photopischer Gewichtung V(λ).

    L_perc(θ) = ∫ V(λ)|R(λ;θ) - R_soll(λ)|² dλ
    """
    V = photopic_sensitivity(wavelengths)
    return spectral_loss(R_computed, R_target, weights=V, wavelengths=wavelengths)


# ---------------------------------------------------------------------------
# Satz 7.1: Analytischer Gradient des FP-Reflexionsspektrums
# ---------------------------------------------------------------------------

def gradient_reflection(
    wavelengths: ArrayLike,
    n_cavity: float,
    thickness: float,
    R1: float,
    R2: float,
    theta: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    """
    Analytischer Gradient der Fabry-Pérot-Reflexion nach Schichtdicke d
    und Brechungsindex n (Gleichungen 27–28).

    ∂R_FP/∂d = 2·Re(r* · ∂r/∂d)
    ∂r/∂δ = 2i·r₂·e^(2iδ)·(1-r₁²) / (1+r₁r₂e^(2iδ))²

    Parameters
    ----------
    wavelengths : array_like
        Wellenlängen [nm].
    n_cavity : float
        Brechungsindex der Kavität.
    thickness : float
        Kavitätsdicke [nm].
    R1, R2 : float
        Intensitätsreflexivitäten.
    theta : float
        Einfallswinkel [rad].

    Returns
    -------
    dR_dd : ndarray
        Gradient nach Schichtdicke ∂R/∂d.
    dR_dn : ndarray
        Gradient nach Brechungsindex ∂R/∂n.
    """
    lam = np.asarray(wavelengths, dtype=float)
    cos_t = np.cos(theta)
    delta = 2 * np.pi * n_cavity * thickness * cos_t / lam

    r1 = np.sqrt(R1)
    r2 = np.sqrt(R2)
    e2id = np.exp(2j * delta)

    denom = 1 + r1 * r2 * e2id
    r_fp  = (r1 + r2 * e2id) / denom

    # ∂r/∂δ (Gleichung 28)
    dr_ddelta = (2j * r2 * e2id * (1 - r1**2)) / denom**2

    # δ = 2π·n·d·cos(θ)/λ → ∂δ/∂d = 2π·n·cos(θ)/λ
    ddelta_dd = 2 * np.pi * n_cavity * cos_t / lam
    ddelta_dn = 2 * np.pi * thickness * cos_t / lam

    # ∂R/∂d = 2·Re(r* · ∂r/∂d) = 2·Re(r* · (∂r/∂δ)·(∂δ/∂d))
    dR_dd = 2 * np.real(np.conj(r_fp) * dr_ddelta * ddelta_dd)
    dR_dn = 2 * np.real(np.conj(r_fp) * dr_ddelta * ddelta_dn)

    return dR_dd, dR_dn


def adjoint_gradient(
    wavelengths: NDArray,
    R_computed: NDArray,
    R_target: NDArray,
    n_cavity: float,
    thickness: float,
    R1: float,
    R2: float,
    theta: float = 0.0,
    weights: Optional[NDArray] = None,
) -> Tuple[float, float]:
    """
    Gradienten des Verlustfunktionals via adjungierter Methode (Gl. 29).

    ∂L/∂d = ∫ w(λ) · 2(R-R_soll) · ∂R/∂d dλ
    ∂L/∂n = ∫ w(λ) · 2(R-R_soll) · ∂R/∂n dλ

    Laufzeit: O(Λ) – unabhängig von der Anzahl Parameter (Satz 7.2).

    Parameters
    ----------
    wavelengths, R_computed, R_target : ndarray
    n_cavity, thickness, R1, R2, theta : float
    weights : ndarray, optional
        Gewichtung w(λ). Standard: V(λ) (photopisch).

    Returns
    -------
    grad_d : float
        ∂L/∂d
    grad_n : float
        ∂L/∂n
    """
    if weights is None:
        weights = photopic_sensitivity(wavelengths)

    dR_dd, dR_dn = gradient_reflection(
        wavelengths, n_cavity, thickness, R1, R2, theta
    )

    residual = R_computed - R_target  # shape (Λ,)

    grad_d = float(np.trapezoid(weights * 2 * residual * dR_dd, wavelengths))
    grad_n = float(np.trapezoid(weights * 2 * residual * dR_dn, wavelengths))

    return grad_d, grad_n


# ---------------------------------------------------------------------------
# Algorithmus 8.1: Optimale Resonatorgestaltung
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Ergebnis der Resonator-Optimierung."""
    thickness: float           # Optimale Schichtdicke d* [nm]
    Q: float                   # Optimaler Quellungsgrad Q*
    n_eff: float               # Effektiver Brechungsindex n_eff*
    lambda_res: float          # Resonanzwellenlänge [nm]
    loss: float                # Finales Verlustfunktional L*
    loss_history: list[float]  # Verlauf L^(i)
    n_iterations: int          # Benötigte Iterationen
    converged: bool
    delta_e_final: float       # CIE-ΔE zum Zielspektrum


def optimize_resonator(
    wavelengths: NDArray,
    R_target: NDArray,
    d0_init: float = 250.0,
    Q_init: float = 2.0,
    n_polymer: float = 1.50,
    n_solvent: float = 1.333,
    R1: float = 0.5,
    R2: float = 0.5,
    theta: float = 0.0,
    learning_rate: float = 0.5,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    d_bounds: Tuple[float, float] = (50.0, 1000.0),
    Q_bounds: Tuple[float, float] = (1.0, 10.0),
    use_perceptual_weights: bool = True,
) -> OptimizationResult:
    """
    Optimale Resonatorgestaltung durch adjungierten Gradientenabstieg –
    Algorithmus 8.1 der Arbeit.

    Minimiert L(d, Q) = ∫ w(λ)|R_FP(λ; d, Q) - R_soll(λ)|² dλ

    wobei R_FP(λ; d, Q) = Fabry-Pérot-Reflexion mit Kavitätsdicke d·Q
    und Brechungsindex n_eff(Q).

    Konvergenz: geometrisch mit Rate (1 - μ/L) < 1 (Satz 8.1).

    Parameters
    ----------
    wavelengths : ndarray [nm]
        Wellenlängengitter Λ Punkte.
    R_target : ndarray
        Zielreflexionsspektrum.
    d0_init : float
        Start-Schichtdicke [nm].
    Q_init : float
        Start-Quellungsgrad.
    n_polymer, n_solvent : float
        Brechungsindizes für n_eff(Q) via Lorentz-Lorenz.
    R1, R2 : float
        Spiegel-Reflexivitäten.
    theta : float
        Einfallswinkel [rad].
    learning_rate : float
        Lernrate η.
    max_iterations : int
        Maximale Iterationen I_max.
    tolerance : float
        Konvergenztoleranz L < ε.
    d_bounds : tuple
        Zulässiger Bereich für d [nm].
    Q_bounds : tuple
        Zulässiger Bereich für Q.
    use_perceptual_weights : bool
        Wenn True: photopische Gewichtung V(λ).

    Returns
    -------
    OptimizationResult
    """
    # Gewichtsfaktor
    weights = photopic_sensitivity(wavelengths) if use_perceptual_weights else np.ones_like(wavelengths)

    # Startparameter
    d0 = float(d0_init)
    Q  = float(Q_init)

    loss_history: list[float] = []
    converged = False

    for i in range(max_iterations):
        # Aktuelle Kavitätsparameter
        n_eff = n_from_swelling(n_polymer, n_solvent, Q)
        d_cav = swelling_to_thickness(d0, Q, mode="anchored")  # d = Q·d₀

        # Reflexionsspektrum berechnen
        R_computed = fabry_perot_reflection(wavelengths, complex(n_eff), d_cav, R1, R2, theta)

        # Verlustfunktional
        loss = spectral_loss(R_computed, R_target, weights=weights, wavelengths=wavelengths)
        loss_history.append(loss)

        # Konvergenzcheck
        if loss < tolerance:
            converged = True
            break

        # Gradienten via adjungierter Methode
        grad_d, grad_n = adjoint_gradient(
            wavelengths, R_computed, R_target,
            n_eff, d_cav, R1, R2, theta, weights=weights
        )

        # Kettenregel: ∂L/∂d₀ = ∂L/∂d_cav · ∂d_cav/∂d₀ + ∂L/∂n_eff · ∂n_eff/∂d₀
        # d_cav = Q·d₀ → ∂d_cav/∂d₀ = Q
        # n_eff hängt nicht von d₀ ab → ∂n_eff/∂d₀ = 0
        grad_d0 = grad_d * Q

        # Kettenregel für Q:
        # d_cav = Q·d₀ → ∂d_cav/∂Q = d₀
        # n_eff(Q) → ∂n_eff/∂Q numerisch
        eps_Q   = 1e-4
        dn_dQ   = (n_from_swelling(n_polymer, n_solvent, Q + eps_Q)
                 - n_from_swelling(n_polymer, n_solvent, max(1.0, Q - eps_Q))) / (2 * eps_Q)
        grad_Q  = grad_d * d0 + grad_n * dn_dQ

        # Gradientenabstieg mit Projektion
        d0_new = np.clip(d0 - learning_rate * grad_d0, d_bounds[0], d_bounds[1])
        Q_new  = np.clip(Q  - learning_rate * grad_Q,  Q_bounds[0], Q_bounds[1])

        # Adaptive Lernrate: Backtracking Line Search
        if loss_history[-1] > (loss_history[-2] if len(loss_history) > 1 else np.inf) * 1.01:
            learning_rate *= 0.5

        d0, Q = float(d0_new), float(Q_new)

    # Finale Berechnung
    n_eff_final = n_from_swelling(n_polymer, n_solvent, Q)
    d_cav_final = swelling_to_thickness(d0, Q, mode="anchored")
    R_final     = fabry_perot_reflection(wavelengths, complex(n_eff_final), d_cav_final, R1, R2, theta)
    loss_final  = spectral_loss(R_final, R_target, weights=weights, wavelengths=wavelengths)

    # ΔE berechnen
    lab_computed  = spectrum_to_lab(wavelengths, R_final)
    lab_target    = spectrum_to_lab(wavelengths, R_target)
    dE            = delta_e(lab_computed, lab_target)

    lambda_res = fabry_perot_resonance_wavelength(n_eff_final, d_cav_final, order=1)

    return OptimizationResult(
        thickness     = d0,
        Q             = Q,
        n_eff         = n_eff_final,
        lambda_res    = lambda_res,
        loss          = loss_final,
        loss_history  = loss_history,
        n_iterations  = len(loss_history),
        converged     = converged,
        delta_e_final = dE,
    )


def target_spectrum_gaussian(
    wavelengths: NDArray,
    center_nm: float,
    fwhm_nm: float = 60.0,
    amplitude: float = 0.8,
) -> NDArray:
    """
    Erzeugt ein gaußförmiges Zielspektrum – nützlich für Tests.

    Parameters
    ----------
    wavelengths : ndarray
    center_nm : float
        Zentrum des Gauss [nm].
    fwhm_nm : float
        Halbwertsbreite [nm].
    amplitude : float
        Maximale Reflexion.

    Returns
    -------
    R_target : ndarray
    """
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((wavelengths - center_nm) / sigma) ** 2)


def target_spectrum_leaf(wavelengths: NDArray) -> NDArray:
    """
    Typisches Reflexionsspektrum eines grünen Laubblatts (Näherung).
    Reflexionsmaximum bei ~550 nm, niedrige Reflexion im Blau und Rot.
    """
    lam = np.asarray(wavelengths, dtype=float)
    # Grüner Peak
    green = 0.35 * np.exp(-0.5 * ((lam - 550) / 30) ** 2)
    # Infrarot-Anstieg (Vegetation)
    ir_rise = 0.45 * np.where(lam > 700, (lam - 700) / 80, 0.0)
    # Chlorophyll-Absorption bei 680 nm
    chl = 0.08 * np.exp(-0.5 * ((lam - 680) / 20) ** 2)
    R = np.clip(green + ir_rise - chl + 0.03, 0.0, 1.0)
    return R
