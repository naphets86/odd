"""hawk — Bioinspiriertes Sturzangriffssystem (BioSAS).

Dieses Paket implementiert das aerodynamische Modell und den Regelungsalgorithmus
des bioinspirierten Sturzangriffssystems, das auf den Flugeigenschaften des
Wanderfalken (*Falco peregrinus*) basiert.

Öffentliche Schnittstelle
-------------------------
Aerodynamik:
    `air_density`, `drag_coefficient`, `lift_coefficient`,
    `reference_area`, `drag_force`, `lift_force`, `terminal_velocity`

Regelung:
    `Phase`, `BGSGController`, `BioFalconTracker`, `proportional_navigation`
"""

from hawk.aerodynamics import (
    air_density,
    drag_coefficient,
    drag_force,
    lift_coefficient,
    lift_force,
    reference_area,
    terminal_velocity,
)
from hawk.controller import (
    BGSGController,
    BioFalconTracker,
    Phase,
    proportional_navigation,
)

__all__ = [
    # aerodynamics
    "air_density",
    "drag_coefficient",
    "drag_force",
    "lift_coefficient",
    "lift_force",
    "reference_area",
    "terminal_velocity",
    # controller
    "Phase",
    "BGSGController",
    "BioFalconTracker",
    "proportional_navigation",
]
