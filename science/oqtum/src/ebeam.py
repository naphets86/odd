"""
nanooptik.ebeam
===============
Elektronenstrahl-Kodierung und Proximity-Effekt-Korrektur
gemäß Kapitel 6 der wissenschaftlichen Arbeit.

Implementierte Algorithmen:
  - Definition 6.2: Proximity-Funktion (Gl. 22)
  - Gl. 23:         Faltungsgleichung D_eff = f * D_nom
  - Definition 6.3: Proximity-Korrekturproblem (Gl. 24)
  - Algorithmus 6.1: Iterative Proximity-Effekt-Korrektur (IPEC)
  - Satz 6.2:       Konvergenz der IPEC
  - Satz 6.1:       Invertierbarkeit des Topographie-Operators
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
import warnings


# ---------------------------------------------------------------------------
# Definition 6.2: Proximity-Funktion
# ---------------------------------------------------------------------------

def proximity_function(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 50.0,
    beta: float = 5000.0,
    eta_back: float = 0.75,
) -> NDArray:
    """
    Proximity-Funktion des Elektronenstrahls (Gleichung 22):

    f(r) = D₀/[π(α² + η_back·β²)] · [exp(-r²/α²) + η_back·exp(-r²/β²)]

    Parameters
    ----------
    x, y : array_like
        Koordinaten relativ zum Strahlmittelpunkt [nm].
    alpha : float
        Vorwärtsstreuradius [nm]. Typisch: 50 nm bei 100 keV.
    beta : float
        Rückstreuradius [nm]. Typisch: 5000 nm bei 100 keV.
    eta_back : float
        Rückstreukoeffizient η. Typisch: 0.75 für PMMA.

    Returns
    -------
    f : ndarray
        Normierte Proximity-Funktion [1/nm²].

    Notes
    -----
    Die Funktion ist normiert sodass ∫∫ f(x,y) dx dy = 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x**2 + y**2

    norm = np.pi * (alpha**2 + eta_back * beta**2)
    forward  = np.exp(-r2 / alpha**2)
    backward = eta_back * np.exp(-r2 / beta**2)

    return (forward + backward) / norm


def proximity_kernel_2d(
    grid_size: int,
    pixel_size: float,
    alpha: float = 50.0,
    beta: float = 5000.0,
    eta_back: float = 0.75,
) -> NDArray:
    """
    Erzeugt einen 2D-Proximity-Kernel für FFT-Faltung.

    Parameters
    ----------
    grid_size : int
        Gittergröße M (quadratisch: M×M).
    pixel_size : float
        Pixelgröße [nm/Pixel].
    alpha, beta, eta_back : float
        Proximity-Parameter.

    Returns
    -------
    kernel : ndarray, shape (M, M)
        Normierter Proximity-Kernel.
    """
    M = grid_size
    ax = np.fft.fftfreq(M) * M * pixel_size
    X, Y = np.meshgrid(ax, ax)
    kernel = proximity_function(X, Y, alpha, beta, eta_back)
    # Normierung: ∑ kernel · pixel_size² ≈ 1
    kernel /= kernel.sum()
    return kernel


def apply_proximity(
    dose_nominal: NDArray,
    pixel_size: float,
    alpha: float = 50.0,
    beta: float = 5000.0,
    eta_back: float = 0.75,
) -> NDArray:
    """
    Wendet den Proximity-Effekt auf ein nominales Dosismuster an (Gleichung 23):

    D_eff(r) = (f * D_nom)(r)

    Berechnung via 2D-FFT in O(M² log M).

    Parameters
    ----------
    dose_nominal : ndarray, shape (M, M)
        Nominales Dosismuster [mC/cm²].
    pixel_size : float
        Pixelgröße [nm/Pixel].
    alpha, beta, eta_back : float
        Proximity-Parameter.

    Returns
    -------
    dose_effective : ndarray, shape (M, M)
        Effektives Dosismuster nach Proximity-Verschmierung.
    """
    M = dose_nominal.shape[0]
    kernel = proximity_kernel_2d(M, pixel_size, alpha, beta, eta_back)

    # Energie-erhaltende FFT-Faltung:
    # Da der Backscatter-Radius oft größer als das Grid ist, verlässt
    # ein Teil der Kernelenergie das Grid → naiver Conv verliert Energie.
    # Lösung: Teile durch die "lokale Normierung" = Faltung von Einsen mit Kernel.
    # Das entspricht dem tatsächlichen Kernel-Integral über den sichtbaren Bereich.
    M2 = 2 * M
    ones_pad   = np.zeros((M2, M2))
    dose_pad   = np.zeros((M2, M2))
    kernel_pad = np.zeros((M2, M2))

    ones_pad[:M, :M]   = 1.0
    dose_pad[:M, :M]   = dose_nominal
    kernel_pad[:M, :M] = kernel

    F_kernel = np.fft.fft2(kernel_pad)
    F_dose   = np.fft.fft2(dose_pad)
    F_ones   = np.fft.fft2(ones_pad)

    # Gefaltete Dosis und lokale Kernelnormierung
    conv_dose = np.real(np.fft.ifft2(F_dose * F_kernel))
    conv_norm = np.real(np.fft.ifft2(F_ones * F_kernel))

    result = conv_dose[:M, :M]
    norm   = conv_norm[:M, :M]

    # Lokale Normierung: jeder Pixel durch seine Kernelabdeckung dividieren
    mask = norm > 1e-10 * norm.max()
    result[mask] /= norm[mask]
    result[~mask] = dose_nominal[~mask]

    # Globale Energieerhaltung: out.sum() == dose_in.sum()
    # (Faltung verteilt Dosis um, erzeugt/vernichtet sie nicht)
    dose_total = dose_nominal.sum()
    result_total = result.sum()
    if result_total > 1e-15:
        result *= dose_total / result_total

    return result


# ---------------------------------------------------------------------------
# Algorithmus 6.1: Iterative Proximity-Effekt-Korrektur (IPEC)
# ---------------------------------------------------------------------------

@dataclass
class IPECResult:
    """Ergebnis der iterativen Proximity-Korrektur."""
    dose_nominal: NDArray         # Korrigiertes nominales Dosismuster
    dose_effective: NDArray       # Effektives Muster nach finaler Faltung
    error_history: list[float]   # max|err| über Iterationen
    n_iterations: int             # Tatsächliche Anzahl Iterationen
    converged: bool               # Konvergenzflag


def proximity_correction_ipec(
    dose_target: NDArray,
    pixel_size: float,
    alpha: float = 50.0,
    beta: float = 5000.0,
    eta_back: float = 0.75,
    omega: float = 0.8,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> IPECResult:
    """
    Iterative Proximity-Effekt-Korrektur (IPEC) – Algorithmus 6.1.

    Löst das lineare Inverseproblem f * D_nom = D_soll durch relaxierte
    Richardson-Iteration:

    D_nom^(k+1) = max(0, D_nom^(k) + ω · (D_soll - f * D_nom^(k)))

    Laufzeit: O(M² log M) pro Iteration.
    Konvergenz: geometrisch mit Rate ρ = |1 - ω · ||f̂||_∞| < 1
    (Satz 6.2 der Arbeit).

    Parameters
    ----------
    dose_target : ndarray, shape (M, M)
        Gewünschte effektive Dosisverteilung D_soll [mC/cm²].
    pixel_size : float
        Pixelgröße [nm/Pixel].
    alpha, beta, eta_back : float
        Proximity-Parameter.
    omega : float
        Relaxationsparameter ω ∈ (0, 2/||f̂||_∞).
    tolerance : float
        Konvergenztoleranz ||err||_∞ < tolerance.
    max_iterations : int
        Maximale Iterationsanzahl.

    Returns
    -------
    IPECResult
    """
    M = dose_target.shape[0]
    kernel = proximity_kernel_2d(M, pixel_size, alpha, beta, eta_back)

    # Fouriertransformierte des Kernels (für Faltung)
    F_kernel = np.fft.fft2(kernel)

    # Überprüfe Stabilitätsbedingung: ω < 2/||F_kernel||_∞
    f_hat_max = float(np.max(np.abs(F_kernel)))
    omega_max = 2.0 / f_hat_max
    if omega >= omega_max:
        warnings.warn(
            f"Relaxationsparameter ω={omega:.3f} ≥ ω_max={omega_max:.3f}. "
            f"Algorithmus könnte divergieren. Setze ω = {0.9*omega_max:.3f}.",
            UserWarning,
        )
        omega = 0.9 * omega_max

    # Initiale Schätzung
    D_nom = dose_target.copy()
    error_history: list[float] = []

    for iteration in range(max_iterations):
        # Vorwärtsfaltung: D_eff = f * D_nom
        F_nom = np.fft.fft2(D_nom)
        D_eff = np.real(np.fft.ifft2(F_kernel * F_nom))

        # Fehler
        err = dose_target - D_eff
        err_max = float(np.max(np.abs(err)))
        error_history.append(err_max)

        # Konvergenzcheck
        if err_max < tolerance:
            return IPECResult(
                dose_nominal=D_nom,
                dose_effective=D_eff,
                error_history=error_history,
                n_iterations=iteration + 1,
                converged=True,
            )

        # Richardson-Update mit Projektion auf D ≥ 0
        D_nom = np.maximum(0.0, D_nom + omega * err)

    # Nicht konvergiert
    warnings.warn(
        f"IPEC hat nach {max_iterations} Iterationen nicht konvergiert. "
        f"Letzter Fehler: {error_history[-1]:.2e}",
        UserWarning,
    )
    return IPECResult(
        dose_nominal=D_nom,
        dose_effective=apply_proximity(D_nom, pixel_size, alpha, beta, eta_back),
        error_history=error_history,
        n_iterations=max_iterations,
        converged=False,
    )


# ---------------------------------------------------------------------------
# Satz 6.1: Topographie-Operator und Inverse
# ---------------------------------------------------------------------------

def topography_operator(
    dose: ArrayLike,
    d0: float,
    nu_e0: float,
    alpha_ebeam: float,
    chi: float,
    V1: float = 18.0,
    n_polymer: float = 1.50,
    n_solvent: float = 1.333,
) -> NDArray:
    """
    Topographie-Operator T: D(r) → h(r) = d₀·Q(νe(D), χ) - d₀.

    Berechnet die Oberflächenhöhe nach Quellung als Funktion des
    lokalen Dosismusters.

    Parameters
    ----------
    dose : array_like
        Dosismuster D(r) [mC/cm²].
    d0 : float
        Trockene Filmdicke [nm].
    nu_e0 : float
        Grundvernetzungsdichte [mol/m³].
    alpha_ebeam : float
        E-Beam-Empfindlichkeitskoeffizient [mol/m³ / (mC/cm²)].
    chi : float
        Flory-Huggins-Parameter des Quellungsmittels.
    V1 : float
        Molares Lösungsmittelvolumen [cm³/mol].
    n_polymer, n_solvent : float
        Brechungsindizes für Quellungsberechnung.

    Returns
    -------
    h : ndarray
        Höhenprofil [nm].
    """
    from .polymer import dose_to_crosslink_density, solve_flory_rehner, swelling_to_thickness

    dose = np.asarray(dose, dtype=float)
    nu_e = dose_to_crosslink_density(dose, nu_e0, alpha_ebeam)

    h = np.zeros_like(dose)
    for idx in np.ndindex(dose.shape):
        try:
            Q = solve_flory_rehner(chi, float(nu_e[idx]), V1)
        except (ValueError, RuntimeError):
            Q = 1.0
        d_sw = swelling_to_thickness(d0, Q, mode="anchored")
        h[idx] = d_sw - d0

    return h


def inverse_topography_operator(
    height_target: ArrayLike,
    d0: float,
    nu_e0: float,
    alpha_ebeam: float,
    chi: float,
    V1: float = 18.0,
    Q_max: float = 10.0,
) -> NDArray:
    """
    Inverser Topographie-Operator: h_soll(r) → D(r).

    Nutzt die Monotonie von T (Satz 6.1) für elementweise Invertierung.

    Parameters
    ----------
    height_target : array_like
        Zielhöhenprofil h_soll(r) [nm].
    d0 : float
        Trockene Filmdicke [nm].
    nu_e0 : float
        Grundvernetzungsdichte [mol/m³].
    alpha_ebeam : float
        E-Beam-Empfindlichkeitskoeffizient [mol/m³ / (mC/cm²)].
    chi : float
        Flory-Huggins-Parameter.
    V1 : float
        Molares Lösungsmittelvolumen [cm³/mol].
    Q_max : float
        Maximaler Quellungsgrad für Suchbereich.

    Returns
    -------
    dose : ndarray
        Benötigtes Dosismuster [mC/cm²].
    """
    from .polymer import solve_flory_rehner, swelling_to_thickness
    from scipy.optimize import brentq

    height_target = np.asarray(height_target, dtype=float)
    dose = np.zeros_like(height_target)

    for idx in np.ndindex(height_target.shape):
        h_t = float(height_target[idx])
        Q_t = (h_t + d0) / d0          # Q = (h + d0) / d0

        Q_t = max(1.0, min(Q_t, Q_max))

        # Inverse Flory-Rehner: finde νe sodass Q(νe) = Q_t
        def residual_Q(nu_e_val):
            try:
                Q = solve_flory_rehner(chi, nu_e_val, V1)
            except (ValueError, RuntimeError):
                return Q_max - Q_t
            return Q - Q_t

        try:
            nu_e_star = brentq(residual_Q, nu_e0, nu_e0 + 1e5, xtol=1e-6)
        except ValueError:
            nu_e_star = nu_e0

        dose[idx] = max(0.0, (nu_e_star - nu_e0) / alpha_ebeam)

    return dose
