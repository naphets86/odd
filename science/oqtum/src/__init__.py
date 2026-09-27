"""
nanooptik – Aktive Nanooptik: Programmierbare photonische Oberflächen
======================================================================
Python-Framework zur wissenschaftlichen Arbeit:
  "Aktive Nanooptik: Programmierbare photonische Oberflächen –
   Dynamische Textur- und Farbkontrolle durch weiche Polymersysteme"

Module:
  optics      – Fresnel-Koeffizienten, Fabry-Pérot, Transfermatrix-Methode
  polymer     – Flory-Rehner-Quellungsmodell, Brechungsindex-Kopplung
  ebeam       – Elektronenstrahl-Kodierung, Proximity-Korrektur
  optimizer   – Inverses Problem, adjungierter Gradientenabstieg
  colorimetry – CIE-Farbraum, Delta-E, photopische Kurve
  pipeline    – Vollständige Zielbild -> Dosismuster Pipeline
  materials   – Materialdatenbank (PDMS, Au, SiO2, ...)
"""

from .optics import (
    fresnel_coefficients,
    layer_matrix,
    transfer_matrix,
    compute_rt,
    fabry_perot_reflection,
    fabry_perot_finesse,
    tmm_spectrum,
    tmm_inhomogeneous,
)
from .polymer import (
    lorentz_lorenz,
    n_from_swelling,
    flory_rehner_residual,
    solve_flory_rehner,
    swelling_to_thickness,
    coupling_equation,
    dose_to_crosslink_density,
    bethe_range,
)
from .ebeam import (
    proximity_function,
    proximity_kernel_2d,
    apply_proximity,
    proximity_correction_ipec,
    topography_operator,
    inverse_topography_operator,
)
from .optimizer import (
    spectral_loss,
    gradient_reflection,
    adjoint_gradient,
    optimize_resonator,
)
from .colorimetry import (
    photopic_sensitivity,
    spectrum_to_xyz,
    xyz_to_lab,
    delta_e,
    wavelengths_visible,
    camouflage_index,
)
from .pipeline import (
    DesignTarget,
    run_pipeline,
)
from .materials import MATERIALS

__version__ = "1.0.0"
__author__  = "Stephan Epp"
