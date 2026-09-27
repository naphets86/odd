"""
nanooptik.polymer
=================
Polymerphysik-Modul gemäß Kapitel 5 der wissenschaftlichen Arbeit.

Implementierte Theoreme:
  - Definition 5.1:  Flory-Huggins-Mischungsfreie Energie (Gl. 14)
  - Definition 5.2:  Elastische freie Energie / Gummi-Elastizität (Gl. 15)
  - Satz 5.1:        Flory-Rehner-Gleichgewichtsbedingung (Gl. 16)
  - Definition 5.3:  E-Beam-induzierte Vernetzung (Gl. 17–18)
  - Definition 5.4:  Lorentz-Lorenz-Mischungsregel (Gl. 20)
  - Zentrale Kopplung: Gleichung (21): λ_res = f(χ, ν_e, d₀)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Definition 5.4: Lorentz-Lorenz-Mischungsregel
# ---------------------------------------------------------------------------

def lorentz_lorenz(
    n_polymer: float,
    n_solvent: float,
    phi_polymer: float,
) -> float:
    """
    Effektiver Brechungsindex eines Zwei-Komponenten-Systems gemäß der
    Lorentz-Lorenz-Mischungsregel (Gleichung 20 der Arbeit).

    (n_eff² - 1)/(n_eff² + 2) = φ_p·(n_p² - 1)/(n_p² + 2)
                                + (1-φ_p)·(n_s² - 1)/(n_s² + 2)

    Parameters
    ----------
    n_polymer : float
        Brechungsindex des reinen Polymers (z.B. 1.50 für PDMS).
    n_solvent : float
        Brechungsindex des Lösungsmittels (z.B. 1.333 für Wasser).
    phi_polymer : float
        Polymervolumenanteil φ_p = 1/Q ∈ (0, 1].

    Returns
    -------
    n_eff : float
        Effektiver Brechungsindex des gequollenen Hydrogels.

    Notes
    -----
    Gilt für nicht-absorptive Materialien (κ ≈ 0).
    """
    if not 0.0 < phi_polymer <= 1.0:
        raise ValueError(f"φ_p muss in (0, 1] liegen, erhalten: {phi_polymer}")

    def ll_term(n: float) -> float:
        return (n**2 - 1) / (n**2 + 2)

    # Gewichtete Mischung im Clausius-Mossotti-Sinn
    ll_mix = phi_polymer * ll_term(n_polymer) + (1 - phi_polymer) * ll_term(n_solvent)

    # Invertierung: n² = (1 + 2·ll) / (1 - ll)
    n_sq = (1 + 2 * ll_mix) / (1 - ll_mix)
    return float(np.sqrt(max(n_sq, 1.0)))


def n_from_swelling(
    n_polymer: float,
    n_solvent: float,
    Q: float,
) -> float:
    """
    Effektiver Brechungsindex als Funktion des Quellungsgrads Q.

    Parameters
    ----------
    n_polymer : float
        Brechungsindex des trockenen Polymers.
    n_solvent : float
        Brechungsindex des Lösungsmittels.
    Q : float
        Quellungsgrad Q = V_gequollen / V_trocken ≥ 1.

    Returns
    -------
    n_eff : float

    Notes
    -----
    Lineare Näherung (Gl. n-quellung in Abschnitt 2.3):
        n(Q) ≈ 1 + (n₀ - 1)/Q
    Diese Funktion verwendet die exakte Lorentz-Lorenz-Regel.
    """
    if Q < 1.0:
        raise ValueError(f"Quellungsgrad Q muss ≥ 1 sein, erhalten: {Q}")
    phi_p = 1.0 / Q
    return lorentz_lorenz(n_polymer, n_solvent, phi_p)


def n_from_swelling_linear(n_polymer: float, Q: float) -> float:
    """
    Lineare Näherung des Brechungsindex:  n(Q) ≈ 1 + (n₀ - 1)/Q.
    Gilt für Q ≲ 5 mit Fehler < 1 %.
    """
    if Q < 1.0:
        raise ValueError(f"Q muss ≥ 1 sein, erhalten: {Q}")
    return 1.0 + (n_polymer - 1.0) / Q


# ---------------------------------------------------------------------------
# Satz 5.1: Flory-Rehner-Gleichgewichtsbedingung
# ---------------------------------------------------------------------------

def flory_rehner_residual(
    phi_p: float,
    chi: float,
    nu_e: float,
    V1: float = 18.0,
) -> float:
    """
    Residuum der Flory-Rehner-Gleichgewichtsbedingung (Gleichung 16):

    ln(1-φ) + φ + χ·φ² + V₁·νe·(φ - φ²/2) = 0

    Für substrat-verankerte Filme (laterale Quellung gehemmt).

    Parameters
    ----------
    phi_p : float
        Polymervolumenanteil φ_p = 1/Q ∈ (0, 1).
    chi : float
        Flory-Huggins-Wechselwirkungsparameter χ.
    nu_e : float
        Effektive Vernetzungsdichte [mol/m³].
    V1 : float
        Molares Lösungsmittelvolumen V₁ [cm³/mol].

    Returns
    -------
    residual : float
        Soll im Gleichgewicht = 0 sein.
    """
    if not 0.0 < phi_p < 1.0:
        return np.inf
    phi_s = 1.0 - phi_p
    # Einheiten: nu_e [mol/L] = [mol/dm³], V1 [cm³/mol]
    # 1 L = 1000 cm³ → nu_e [mol/L] * V1 [cm³/mol] / 1000 [cm³/L] = dimensionslos
    # Beispiel: nu_e=100 mol/L, V1=18 cm³/mol → nu_V1 = 1.8 (physikalisch sinnvoll)
    nu_V1 = nu_e * V1 / 1000.0  # dimensionslos

    residual = (
        np.log(phi_s)
        + phi_p
        + chi * phi_p**2
        + nu_V1 * (phi_p - phi_p**2 / 2.0)
    )
    return float(residual)


def solve_flory_rehner(
    chi: float,
    nu_e: float,
    V1: float = 18.0,
    phi_bounds: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
) -> float:
    """
    Löst die Flory-Rehner-Gleichgewichtsbedingung numerisch für den
    Gleichgewichts-Quellungsgrad Q*.

    Verwendet Brentq (Bisektionsverfahren) für garantierte Konvergenz.

    Parameters
    ----------
    chi : float
        Flory-Huggins-Parameter χ. Für χ < 0.5: gutes Lösungsmittel.
    nu_e : float
        Vernetzungsdichte [mol/L]. Größere Werte → geringere Quellung.
    V1 : float
        Molares Lösungsmittelvolumen [cm³/mol].
    phi_bounds : tuple
        Suchintervall für φ_p ∈ (0, 1).

    Returns
    -------
    Q_eq : float
        Gleichgewichts-Quellungsgrad Q* = 1/φ_p*.

    Raises
    ------
    ValueError
        Wenn kein Gleichgewicht im angegebenen Bereich existiert.
    """
    f = lambda phi: flory_rehner_residual(phi, chi, nu_e, V1)

    lo, hi = phi_bounds
    f_lo, f_hi = f(lo), f(hi)

    if f_lo * f_hi > 0:
        # Kein Vorzeichenwechsel – kein Gleichgewicht oder Rand-Gleichgewicht
        # Versuche, ein Gleichgewicht zu finden (Q=1 für schlechtes Lösungsmittel)
        if chi >= 0.5 or nu_e > 1e4:
            return 1.0  # kein Quellen
        raise ValueError(
            f"Kein Flory-Rehner-Gleichgewicht gefunden für χ={chi}, νe={nu_e}. "
            f"f(φ_lo)={f_lo:.3f}, f(φ_hi)={f_hi:.3f}"
        )

    phi_eq = brentq(f, lo, hi, xtol=1e-10, rtol=1e-10)
    Q_eq = 1.0 / phi_eq
    return float(Q_eq)


# ---------------------------------------------------------------------------
# Schichtdicke nach Quellung
# ---------------------------------------------------------------------------

def swelling_to_thickness(d0: float, Q: float, mode: str = "anchored") -> float:
    """
    Schichtdicke nach Quellung.

    Parameters
    ----------
    d0 : float
        Trockene Schichtdicke [nm].
    Q : float
        Quellungsgrad.
    mode : str
        'anchored' : Film ist lateral ans Substrat gebunden →
                     Ausdehnung nur senkrecht: d = Q · d₀
        'free'     : Freistehender Film → isotrope Quellung: d = Q^(1/3) · d₀

    Returns
    -------
    d_swollen : float
        Gequollene Schichtdicke [nm].
    """
    if Q < 1.0:
        raise ValueError(f"Q muss ≥ 1 sein, erhalten: {Q}")
    if mode == "anchored":
        return d0 * Q
    elif mode == "free":
        return d0 * Q ** (1.0 / 3.0)
    else:
        raise ValueError(f"Modus muss 'anchored' oder 'free' sein, erhalten: {mode!r}")


# ---------------------------------------------------------------------------
# Zentrale Kopplung: Gleichung (21)
# ---------------------------------------------------------------------------

def coupling_equation(
    chi: float,
    nu_e: float,
    d0: float,
    n_polymer: float = 1.50,
    n_solvent: float = 1.333,
    V1: float = 18.0,
    order: int = 1,
) -> Tuple[float, float, float]:
    """
    Zentrale Kopplungsgleichung (Gleichung 21 der Arbeit):

    λ_res = f(χ, νe, d₀) = 2·n_eff(Q*)·d₀·Q*

    Verbindet Lösungsmittel (χ), Vernetzungsdichte (νe) und trockene
    Filmdicke (d₀) mit der Fabry-Pérot-Resonanzwellenlänge.

    Parameters
    ----------
    chi : float
        Flory-Huggins-Parameter des Lösungsmittels.
    nu_e : float
        Vernetzungsdichte [mol/L].
    d0 : float
        Trockene Kavitätsdicke [nm].
    n_polymer : float
        Brechungsindex des trockenen Polymers.
    n_solvent : float
        Brechungsindex des Lösungsmittels.
    V1 : float
        Molares Lösungsmittelvolumen [cm³/mol].
    order : int
        Ordnung m der Fabry-Pérot-Resonanz.

    Returns
    -------
    lambda_res : float
        Resonanzwellenlänge [nm].
    Q_eq : float
        Gleichgewichts-Quellungsgrad.
    n_eff : float
        Effektiver Brechungsindex im gequollenen Zustand.
    """
    Q_eq  = solve_flory_rehner(chi, nu_e, V1)
    n_eff = n_from_swelling(n_polymer, n_solvent, Q_eq)
    d_sw  = swelling_to_thickness(d0, Q_eq, mode="anchored")
    lambda_res = 2 * n_eff * d_sw / order
    return float(lambda_res), float(Q_eq), float(n_eff)


# ---------------------------------------------------------------------------
# Definition 5.3: E-Beam-induzierte Vernetzung
# ---------------------------------------------------------------------------

def bethe_range(
    U_kV: float,
    rho: float = 1.03,
    A: float = 14.0,
    Z: float = 7.0,
) -> float:
    """
    Bethe-Reichweite des Elektronenstrahls in einem Material (Gleichung 17):

    R_e ≈ 0.0276·A / (ρ·Z^0.889) · U^1.67  [µm]

    Parameters
    ----------
    U_kV : float
        Beschleunigungsspannung [kV].
    rho : float
        Dichte des Materials [g/cm³].
    A : float
        Mittlere Molmasse [g/mol].
    Z : float
        Mittlere Ordnungszahl.

    Returns
    -------
    R_e : float
        Bethe-Reichweite [µm].
    """
    return 0.0276 * A / (rho * Z ** 0.889) * U_kV ** 1.67


def dose_to_crosslink_density(
    dose: ArrayLike,
    nu_e0: float,
    alpha: float,
    V1: float = 18.0,
) -> NDArray:
    """
    Vernetzungsdichte als Funktion der Elektronenstrahldosis (Gleichung 17).

    νe(r) = νe0 + α · D(r)

    Parameters
    ----------
    dose : array_like
        Dosismuster D(r) [mC/cm²].
    nu_e0 : float
        Grundvernetzungsdichte [mol/m³].
    alpha : float
        Empfindlichkeitskoeffizient [mol/m³ / (mC/cm²)].
    V1 : float
        Molares Lösungsmittelvolumen [cm³/mol] (nicht direkt verwendet, für Konsistenz).

    Returns
    -------
    nu_e : ndarray
        Ortsabhängige Vernetzungsdichte [mol/m³].
    """
    dose = np.asarray(dose, dtype=float)
    return nu_e0 + alpha * dose
