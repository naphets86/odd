"""
nanooptik.pipeline
==================
Vollständige Design-Pipeline: Zielbild → Dosismuster.
Implementiert Algorithmus 8.2 der wissenschaftlichen Arbeit.

Pipeline-Schritte:
  1. Farboptimierung: Zielfarbe → (d₀*, Q*) via Alg. 8.1
  2. Vernetzungsdichte: Q* → νe* via inverse Flory-Rehner
  3. Dosisberechnung: νe* → D_soll via inverse Vernetzungsformel
  4. Proximity-Korrektur: D_soll → D_nom via IPEC (Alg. 6.1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple
from dataclasses import dataclass, field

from src.optimizer import optimize_resonator, OptimizationResult
from src.polymer import solve_flory_rehner, dose_to_crosslink_density
from src.ebeam import proximity_correction_ipec, IPECResult, inverse_topography_operator
from src.colorimetry import wavelengths_visible, spectrum_to_lab, delta_e
from src.optics import fabry_perot_reflection
from src.polymer import n_from_swelling, swelling_to_thickness


@dataclass
class DesignTarget:
    """Beschreibt das Ziel-Erscheinungsbild für die Pipeline."""
    wavelengths:    NDArray                  # Wellenlängen [nm]
    R_target:       NDArray                  # Zielreflexionsspektrum
    height_map:     Optional[NDArray] = None # Zielhöhenkarte [nm] (2D)
    # Materialparameter
    n_polymer:      float = 1.50
    n_solvent_color: float = 1.333           # Lösungsmittel für Farbe
    n_solvent_tex:  float = 1.333            # Lösungsmittel für Textur
    chi_color:      float = 0.25             # χ für Farbschicht
    chi_texture:    float = 0.10             # χ für Texturschicht
    V1:             float = 18.0             # Mol. Lösungsmittelvolumen [cm³/mol]
    d0:             float = 250.0            # Trockene Filmdicke [nm]
    nu_e0:          float = 100.0            # Grundvernetzungsdichte [mol/m³]
    alpha_ebeam:    float = 5.0              # E-Beam-Koeffizient [mol/m³/(mC/cm²)]
    R1:             float = 0.5              # Spiegel-Reflexivität 1
    R2:             float = 0.5              # Spiegel-Reflexivität 2
    # E-Beam-Parameter
    ebeam_alpha_nm: float = 50.0            # Vorwärtsstreuradius [nm]
    ebeam_beta_nm:  float = 5000.0          # Rückstreuradius [nm]
    ebeam_eta:      float = 0.75            # Rückstreukoeffizient
    pixel_size_nm:  float = 100.0           # Pixelgröße [nm]


@dataclass
class PipelineResult:
    """Vollständiges Ergebnis der Design-Pipeline."""
    # Phase 1: Farboptimierung
    opt_result:       OptimizationResult
    # Phase 2: Vernetzungsdichte
    nu_e_color:       float
    # Phase 3: Dosisberechnung
    dose_target:      Optional[NDArray] = None
    dose_nominal:     Optional[NDArray] = None
    # Phase 4: Proximity-Korrektur
    ipec_result:      Optional[IPECResult] = None
    # Metriken
    delta_e_color:    float = 0.0
    camouflage_index: float = 0.0
    summary:          str = ""


def run_pipeline(
    target: DesignTarget,
    optimize_kwargs: Optional[dict] = None,
    ipec_kwargs: Optional[dict] = None,
    run_proximity: bool = True,
) -> PipelineResult:
    """
    Vollständige Design-Pipeline gemäß Algorithmus 8.2 der Arbeit.

    Parameters
    ----------
    target : DesignTarget
        Ziel-Erscheinungsbild und Materialparameter.
    optimize_kwargs : dict, optional
        Zusätzliche Argumente für optimize_resonator().
    ipec_kwargs : dict, optional
        Zusätzliche Argumente für proximity_correction_ipec().
    run_proximity : bool
        Wenn False: Proximity-Korrektur überspringen (Schritt 4).

    Returns
    -------
    PipelineResult
    """
    opt_kwargs  = optimize_kwargs or {}
    prox_kwargs = ipec_kwargs or {}

    # ---------------------------------------------------------------
    # Phase 1: Farboptimierung → optimale Resonatorparameter
    # ---------------------------------------------------------------
    opt = optimize_resonator(
        wavelengths = target.wavelengths,
        R_target    = target.R_target,
        d0_init     = target.d0,
        n_polymer   = target.n_polymer,
        n_solvent   = target.n_solvent_color,
        R1          = target.R1,
        R2          = target.R2,
        **opt_kwargs,
    )

    # ---------------------------------------------------------------
    # Phase 2: Vernetzungsdichte aus Gleichgewichts-Q*
    # ---------------------------------------------------------------
    try:
        nu_e_color = _inverse_flory_rehner(
            Q_target=opt.Q,
            chi=target.chi_color,
            nu_e0=target.nu_e0,
            V1=target.V1,
        )
    except Exception:
        nu_e_color = target.nu_e0

    # ---------------------------------------------------------------
    # Phase 3: Dosisberechnung (optional mit Textur-Höhenkarte)
    # ---------------------------------------------------------------
    dose_target = None
    if target.height_map is not None:
        # Textur-Pipeline: Höhenkarte → Dosis
        dose_target = inverse_topography_operator(
            height_target = target.height_map,
            d0            = target.d0,
            nu_e0         = target.nu_e0,
            alpha_ebeam   = target.alpha_ebeam,
            chi           = target.chi_texture,
            V1            = target.V1,
        )
    else:
        # Nur Farbe: uniforme Dosis
        dose_target = np.full((10, 10),
                              max(0.0, (nu_e_color - target.nu_e0) / target.alpha_ebeam))

    # ---------------------------------------------------------------
    # Phase 4: Proximity-Korrektur
    # ---------------------------------------------------------------
    ipec_result = None
    dose_nominal = dose_target.copy()

    if run_proximity and dose_target is not None:
        ipec_result = proximity_correction_ipec(
            dose_target  = dose_target,
            pixel_size   = target.pixel_size_nm,
            alpha        = target.ebeam_alpha_nm,
            beta         = target.ebeam_beta_nm,
            eta_back     = target.ebeam_eta,
            **prox_kwargs,
        )
        dose_nominal = ipec_result.dose_nominal

    # ---------------------------------------------------------------
    # Metriken
    # ---------------------------------------------------------------
    # Finales Reflexionsspektrum
    n_eff_final = n_from_swelling(target.n_polymer, target.n_solvent_color, opt.Q)
    d_final     = swelling_to_thickness(target.d0, opt.Q, mode="anchored")
    R_final     = fabry_perot_reflection(
        target.wavelengths, complex(n_eff_final), d_final, target.R1, target.R2
    )

    lab_computed = spectrum_to_lab(target.wavelengths, R_final)
    lab_target   = spectrum_to_lab(target.wavelengths, target.R_target)
    dE           = delta_e(lab_computed, lab_target)

    summary = (
        f"Pipeline abgeschlossen:\n"
        f"  d₀* = {opt.thickness:.1f} nm, Q* = {opt.Q:.3f}\n"
        f"  n_eff = {opt.n_eff:.4f}, λ_res = {opt.lambda_res:.1f} nm\n"
        f"  ΔE = {dE:.2f} ({'✓ unsichtbar' if dE < 2 else '✗ sichtbar'})\n"
        f"  L_final = {opt.loss:.2e}, Iterationen = {opt.n_iterations}\n"
        f"  νe* = {nu_e_color:.1f} mol/m³\n"
    )
    if ipec_result is not None:
        summary += (
            f"  IPEC: {'konvergiert' if ipec_result.converged else 'nicht konvergiert'} "
            f"nach {ipec_result.n_iterations} Iterationen\n"
        )

    return PipelineResult(
        opt_result      = opt,
        nu_e_color      = nu_e_color,
        dose_target     = dose_target,
        dose_nominal    = dose_nominal,
        ipec_result     = ipec_result,
        delta_e_color   = dE,
        summary         = summary,
    )


def _inverse_flory_rehner(
    Q_target: float,
    chi: float,
    nu_e0: float,
    V1: float = 18.0,
    nu_e_max: float = 1e5,
) -> float:
    """
    Inverse Flory-Rehner: Finde νe sodass Q(νe, χ) = Q_target.
    Verwendet Bisektion.
    """
    from scipy.optimize import brentq

    def residual(nu_e):
        try:
            Q = solve_flory_rehner(chi, nu_e, V1)
        except (ValueError, RuntimeError):
            return 100.0 - Q_target
        return Q - Q_target

    try:
        nu_e_star = brentq(residual, nu_e0, nu_e_max, xtol=1.0, maxiter=100)
    except ValueError:
        nu_e_star = nu_e0

    return float(nu_e_star)
