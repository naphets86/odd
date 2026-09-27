"""
nanooptik.optics
================
Optische Kernberechnungen gemäß Kapitel 2–3 der wissenschaftlichen Arbeit.

Implementierte Theoreme und Definitionen:
  - Definition 2.1:  Komplexer Brechungsindex
  - Definition 2.2:  Fresnel-Koeffizienten (Gl. 2–5)
  - Lemma 2.1:       Energieerhaltung an verlustfreien Grenzflächen
  - Definition 3.1:  Charakteristische Matrix einer Schicht (Gl. 9)
  - Definition 3.2:  Systemtransfermatrix (Gl. 10)
  - Satz 3.1:        Reflexions-/Transmissionskoeffizient via TMM (Gl. 11–12)
  - Satz 3.2:        Trotter-Produktformel für inhomogene Schichten
  - Algorithmus 3.1: TMM für inhomogene Schichten
  - Definition 3.1:  Fabry-Pérot-Resonanz (Gl. 6–8)
  - Definition 3.2:  Finesse (Gl. 13)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

# Physikalische Konstanten
Z0 = 376.730  # Freiraum-Wellenwiderstand [Ω]


# ---------------------------------------------------------------------------
# Hilfsdatenklassen
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """Beschreibt eine einzelne optische Schicht."""
    n: complex          # Komplexer Brechungsindex ñ = n + i·κ
    thickness: float    # Schichtdicke [nm]

    def __post_init__(self):
        if self.thickness < 0:
            raise ValueError(f"Schichtdicke muss ≥ 0 sein, erhalten: {self.thickness}")

    @property
    def is_lossless(self) -> bool:
        return abs(self.n.imag) < 1e-12


@dataclass
class OpticalResult:
    """Ergebnis einer optischen Berechnung."""
    R: float           # Reflexionsintensität [0, 1]
    T: float           # Transmissionsintensität [0, 1]
    A: float           # Absorption = 1 - R - T
    r: complex         # Amplitudenreflexionskoeffizient
    t: complex         # Amplitudentransmissionskoeffizient
    M: NDArray         # 2×2 Systemtransfermatrix

    def __post_init__(self):
        # Sicherstellen, dass A ≥ 0 (numerische Fehler)
        self.A = max(0.0, self.A)


# ---------------------------------------------------------------------------
# Definition 2.2: Fresnel-Koeffizienten
# ---------------------------------------------------------------------------

def fresnel_coefficients(
    n1: complex,
    n2: complex,
    theta_i: float = 0.0,
    polarization: str = "s",
) -> Tuple[complex, complex]:
    """
    Berechnet Fresnel-Amplitudenkoeffizienten r und t an einer ebenen
    Grenzfläche gemäß Gleichungen (2)–(5) der Arbeit.

    Parameters
    ----------
    n1 : complex
        Komplexer Brechungsindex des Eingangsmediums.
    n2 : complex
        Komplexer Brechungsindex des Ausgangsmediums.
    theta_i : float
        Einfallswinkel [rad], Standard: senkrechter Einfall (0.0).
    polarization : str
        's' für TE-Polarisation, 'p' für TM-Polarisation.

    Returns
    -------
    r : complex
        Amplitudenreflexionskoeffizient.
    t : complex
        Amplitudentransmissionskoeffizient.

    Notes
    -----
    Snellius-Gesetz: n1·sin(θ_i) = n2·sin(θ_t)
    Energieerhaltung bei reellen n: R + T = 1 (Lemma 2.1)
    """
    cos_i = np.cos(theta_i)
    # Verallgemeinertes Snellius-Gesetz (komplex)
    sin_t_sq = (n1 / n2) ** 2 * (1 - cos_i**2)
    cos_t = np.sqrt(1 - sin_t_sq + 0j)

    if polarization.lower() == "s":
        # TE-Polarisation (Gl. 2 und 3)
        denom = n1 * cos_i + n2 * cos_t
        r = (n1 * cos_i - n2 * cos_t) / denom
        t = 2 * n1 * cos_i / denom
    elif polarization.lower() == "p":
        # TM-Polarisation (Gl. 4 und 5)
        denom = n2 * cos_i + n1 * cos_t
        r = (n2 * cos_i - n1 * cos_t) / denom
        t = 2 * n1 * cos_i / denom
    else:
        raise ValueError(f"Polarisation muss 's' oder 'p' sein, erhalten: {polarization!r}")

    return r, t


# ---------------------------------------------------------------------------
# Definition 3.1: Charakteristische Matrix einer Schicht
# ---------------------------------------------------------------------------

def layer_matrix(
    n: complex,
    thickness: float,
    wavelength: float,
    theta: float = 0.0,
    polarization: str = "s",
) -> NDArray:
    """
    Berechnet die charakteristische 2×2-Transfermatrix einer homogenen
    Schicht gemäß Gleichung (9) der Arbeit.

    Parameters
    ----------
    n : complex
        Komplexer Brechungsindex der Schicht.
    thickness : float
        Schichtdicke [nm].
    wavelength : float
        Vakuumwellenlänge [nm].
    theta : float
        Brechungswinkel in der Schicht [rad].
    polarization : str
        's' oder 'p'.

    Returns
    -------
    M : ndarray, shape (2, 2), dtype complex
        Charakteristische Matrix.

    Notes
    -----
    Für verlustfreie Medien gilt det(M) = 1.
    """
    cos_theta = np.cos(theta)
    # Optische Admittanz η [1/Z0 Einheiten absorbiert]
    if polarization.lower() == "s":
        eta = n * cos_theta / Z0
    else:  # p
        eta = n / (cos_theta * Z0) if abs(cos_theta) > 1e-12 else complex(1e12)

    # Phasendicke δ = 2π·n·d·cos(θ)/λ
    delta = 2 * np.pi * n * thickness * cos_theta / wavelength

    cd = np.cos(delta)
    sd = np.sin(delta)

    M = np.array([
        [cd,              -1j * sd / eta],
        [-1j * eta * sd,  cd            ],
    ], dtype=complex)
    return M


# ---------------------------------------------------------------------------
# Definition 3.2 + Satz 3.1: Systemtransfermatrix und r, t
# ---------------------------------------------------------------------------

def transfer_matrix(layers: list[Layer], wavelength: float,
                    theta_i: float = 0.0, polarization: str = "s") -> NDArray:
    """
    Berechnet die Gesamttransfermatrix M_ges = M_1 · M_2 · ... · M_N
    für ein Schichtsystem aus N Schichten (Gleichung 10).

    Parameters
    ----------
    layers : list of Layer
        Geordnete Liste der Schichten (von Eingangsmedium zu Substrat).
    wavelength : float
        Vakuumwellenlänge [nm].
    theta_i : float
        Einfallswinkel im Eingangsmedium [rad].
    polarization : str
        's' oder 'p'.

    Returns
    -------
    M : ndarray, shape (2, 2), dtype complex
    """
    M = np.eye(2, dtype=complex)
    for layer in layers:
        # Winkel in dieser Schicht (Snellius, ausgehend von erster Schicht n0)
        # Vereinfachung: theta konstant für senkrechten Einfall
        M_j = layer_matrix(layer.n, layer.thickness, wavelength,
                           theta_i, polarization)
        M = M @ M_j
    return M


def compute_rt(
    M: NDArray,
    n_in: complex,
    n_sub: complex,
    theta_i: float = 0.0,
    polarization: str = "s",
) -> OpticalResult:
    """
    Berechnet r, t, R, T, A aus der Systemtransfermatrix gemäß
    Gleichungen (11) und (12) der Arbeit.

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        Systemtransfermatrix.
    n_in : complex
        Brechungsindex des Eingangsmediums.
    n_sub : complex
        Brechungsindex des Substrats.
    theta_i : float
        Einfallswinkel [rad].
    polarization : str
        's' oder 'p'.

    Returns
    -------
    OpticalResult
    """
    cos_i = np.cos(theta_i)
    if polarization.lower() == "s":
        eta0 = n_in  * cos_i / Z0
        etas = n_sub * cos_i / Z0  # Näherung: gleicher Winkel
    else:
        eta0 = n_in  / (cos_i * Z0) if abs(cos_i) > 1e-12 else 1e12
        etas = n_sub / (cos_i * Z0) if abs(cos_i) > 1e-12 else 1e12

    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]

    # Gl. 11
    numer_r = m11 * eta0 + m12 * eta0 * etas - m21 - m22 * etas
    denom   = m11 * eta0 + m12 * eta0 * etas + m21 + m22 * etas
    r = numer_r / denom

    # Gl. 12
    t = 2 * eta0 / denom

    R = float(abs(r) ** 2)
    # Transmissionsintensität: Poynting-Vektor-Verhältnis
    T = float((etas / eta0).real * abs(t) ** 2)
    A = max(0.0, 1.0 - R - T)

    return OpticalResult(R=R, T=T, A=A, r=r, t=t, M=M)


# ---------------------------------------------------------------------------
# Fabry-Pérot-Resonator (Kapitel 3)
# ---------------------------------------------------------------------------

def fabry_perot_reflection(
    wavelengths: ArrayLike,
    n_cavity: complex,
    thickness: float,
    R1: float,
    R2: float,
    theta: float = 0.0,
) -> NDArray:
    """
    Reflexionsspektrum eines Fabry-Pérot-Resonators gemäß Gleichung (8).

    R_FP(λ) = (R1 + R2 + 2√(R1·R2)·cos(2δ)) / (1 + R1·R2 + 2√(R1·R2)·cos(2δ))

    Parameters
    ----------
    wavelengths : array_like
        Wellenlängen [nm].
    n_cavity : complex
        Komplexer Brechungsindex der Kavität.
    thickness : float
        Kavitätsdicke [nm].
    R1, R2 : float
        Intensitätsreflexivitäten der Spiegel [0, 1].
    theta : float
        Einfallswinkel [rad].

    Returns
    -------
    R : ndarray
        Reflexionsspektrum.
    """
    lam = np.asarray(wavelengths, dtype=float)
    # Phasendicke δ = 2π·n·d·cos(θ)/λ
    delta = 2 * np.pi * n_cavity.real * thickness * np.cos(theta) / lam

    sqrt_R = np.sqrt(R1 * R2)
    cos2d  = np.cos(2 * delta)

    R_fp = (R1 + R2 + 2 * sqrt_R * cos2d) / (1 + R1 * R2 + 2 * sqrt_R * cos2d)
    return np.clip(R_fp, 0.0, 1.0)


def fabry_perot_finesse(R1: float, R2: float) -> float:
    """
    Finesse eines Fabry-Pérot-Resonators gemäß Gleichung (13).

    F = π·(R1·R2)^(1/4) / (1 - √(R1·R2))

    Parameters
    ----------
    R1, R2 : float
        Intensitätsreflexivitäten der Spiegel.

    Returns
    -------
    F : float
        Finesse.
    """
    sqrt_R = np.sqrt(R1 * R2)
    if sqrt_R >= 1.0:
        raise ValueError("Produkt R1·R2 muss < 1 sein für endliche Finesse.")
    return np.pi * (R1 * R2) ** 0.25 / (1 - sqrt_R)


def fabry_perot_resonance_wavelength(
    n_cavity: float,
    thickness: float,
    order: int = 1,
    theta: float = 0.0,
) -> float:
    """
    Resonanzwellenlänge des Fabry-Pérot-Resonators, m-te Ordnung.
    λ_m = 2·n·d·cos(θ) / m

    Parameters
    ----------
    n_cavity : float
    thickness : float  [nm]
    order : int        Ordnung m ≥ 1
    theta : float      Einfallswinkel [rad]

    Returns
    -------
    lambda_res : float  [nm]
    """
    if order < 1:
        raise ValueError(f"Ordnung muss ≥ 1 sein, erhalten: {order}")
    # Reflexionsminimum (= Transmissionsmaximum) bei δ = (m-½)π:
    # δ = 2πnd cosθ/λ = (m-½)π  →  λ = 4nd cosθ/(2m-1)
    # Für m=1: λ = 4nd (klassische Viertelwellenresonanz)
    # Alternativformulierung: 2nd cosθ = (m-½)λ  →  λ = 2nd cosθ/(m-0.5)
    return 2 * n_cavity * thickness * np.cos(theta) / (order - 0.5)


# ---------------------------------------------------------------------------
# Algorithmus 3.1: TMM für inhomogene Schicht
# ---------------------------------------------------------------------------

def tmm_inhomogeneous(
    n_profile: Callable[[float], complex],
    thickness: float,
    wavelength: float,
    n_sublayers: int = 100,
    n_in: complex = 1.0 + 0j,
    n_sub: complex = 1.45 + 0j,
    polarization: str = "s",
) -> OpticalResult:
    """
    TMM für eine Schicht mit kontinuierlichem Brechungsindexprofil n(z).
    Implementiert Algorithmus 3.1 der Arbeit (Trotter-Produktformel).

    Parameters
    ----------
    n_profile : callable
        Funktion z ↦ n(z), z ∈ [0, thickness] in nm.
    thickness : float
        Gesamtdicke der inhomogenen Schicht [nm].
    wavelength : float
        Vakuumwellenlänge [nm].
    n_sublayers : int
        Anzahl der Sublayer K (Konvergenz: O(1/K²)).
    n_in : complex
        Brechungsindex des Eingangsmediums.
    n_sub : complex
        Brechungsindex des Substrats.
    polarization : str
        's' oder 'p'.

    Returns
    -------
    OpticalResult
    """
    dz = thickness / n_sublayers
    M = np.eye(2, dtype=complex)

    for k in range(n_sublayers):
        z_mid = (k + 0.5) * dz          # Mittelpunkt (Mittelpunktsregel)
        n_k   = complex(n_profile(z_mid))
        M_k   = layer_matrix(n_k, dz, wavelength, polarization=polarization)
        M     = M @ M_k

    return compute_rt(M, n_in, n_sub, polarization=polarization)


# ---------------------------------------------------------------------------
# Spektrumberechnung über Wellenlängenbereich
# ---------------------------------------------------------------------------

def tmm_spectrum(
    layers: list[Layer],
    wavelengths: ArrayLike,
    n_in: complex = 1.0 + 0j,
    n_sub: complex = 1.45 + 0j,
    theta_i: float = 0.0,
    polarization: str = "s",
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Berechnet Reflexions-, Transmissions- und Absorptionsspektrum für ein
    homogenes Schichtsystem über einen Wellenlängenbereich.

    Laufzeit: O(N · Λ)  (Satz 3.2)

    Parameters
    ----------
    layers : list of Layer
        Schichtsystem.
    wavelengths : array_like
        Wellenlängen [nm].
    n_in : complex
        Brechungsindex des Eingangsmediums.
    n_sub : complex
        Brechungsindex des Substrats.
    theta_i : float
        Einfallswinkel [rad].
    polarization : str
        's' oder 'p'.

    Returns
    -------
    R, T, A : ndarray
        Spektren (jeweils Werte in [0, 1]).
    """
    lam = np.asarray(wavelengths, dtype=float)
    R = np.zeros_like(lam)
    T = np.zeros_like(lam)
    A = np.zeros_like(lam)

    for i, wl in enumerate(lam):
        M   = transfer_matrix(layers, wl, theta_i, polarization)
        res = compute_rt(M, n_in, n_sub, theta_i, polarization)
        R[i], T[i], A[i] = res.R, res.T, res.A

    return R, T, A


# ---------------------------------------------------------------------------
# Lorentz-Brechungsindex-Modell (Abschnitt 2.3)
# ---------------------------------------------------------------------------

def lorentz_refractive_index(
    omega: ArrayLike,
    omega0: float,
    gamma: float,
    oscillator_strength: float,
    N: float,
    e: float = 1.602e-19,
    m_e: float = 9.109e-31,
    eps0: float = 8.854e-12,
) -> NDArray:
    """
    Dielektrische Funktion im Lorentz-Oszillator-Modell (Gleichung in Def. 2.3).

    ε(ω) = 1 + (N·e²·f) / (ε₀·mₑ) · 1/(ω₀² - ω² - i·γ·ω)

    Parameters
    ----------
    omega : array_like
        Kreisfrequenzen [rad/s].
    omega0 : float
        Resonanzfrequenz [rad/s].
    gamma : float
        Dämpfungsrate [rad/s].
    oscillator_strength : float
        Oszillatorstärke f.
    N : float
        Resonatordichte [1/m³].

    Returns
    -------
    n_complex : ndarray
        Komplexer Brechungsindex ñ = n + i·κ.
    """
    omega = np.asarray(omega, dtype=complex)
    prefactor = N * e**2 * oscillator_strength / (eps0 * m_e)
    eps = 1.0 + prefactor / (omega0**2 - omega**2 - 1j * gamma * omega)
    # ñ = sqrt(ε)
    n_complex = np.sqrt(eps)
    return n_complex
