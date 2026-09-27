"""liionp – KI-Power-Management-Modul für Lithium-Ionen-Akkumulatoren.

Dieses Paket stellt alle Bausteine bereit, die in *liionp.tex* beschrieben werden:
Degradationsmodelle, thermische Modelle, Zellmodelle, Regler, Mechanikmodelle,
Analyseformeln sowie einen Simulationsmotor zum Vergleich von statischem PMS und KI-PMM.

Öffentliche API
---------------
Aus :mod:`liionp.constants`:
    Physikalische Konstanten (``R_GAS``, ``F_FARADAY``) und Standard-Zellparameter.

Aus :mod:`liionp.degradation`:
    ``arrhenius_rate``, ``arrhenius_ratio``,
    ``sei_growth_rate``, ``sei_thickness``, ``capacity_loss_sei``,
    ``capacity_drift``,
    ``overvoltage_degradation_rate``, ``undervoltage_degradation_rate``,
    ``voltage_degradation_rate``, ``cyclic_degradation_rate``,
    ``total_degradation_rate``,
    ``cumulative_degradation``, ``lifetime_from_rate``

Aus :mod:`liionp.thermal`:
    ``ThermalModel``

Aus :mod:`liionp.cell`:
    ``CellParameters``, ``CellState``, ``CellModel``

Aus :mod:`liionp.controllers`:
    ``PIDController``, ``SimpleMPC``

Aus :mod:`liionp.mechanics`:
    ``swelling_strain``, ``tortuosity``, ``ionic_resistance``,
    ``mechanical_stress``, ``mechanical_degradation_rate``,
    ``mechanical_reduction_factor``, ``PlateDistanceMPCController``

Aus :mod:`liionp.analysis`:
    ``improvement_factor_thermal``, ``improvement_factor_voltage``,
    ``improvement_factor_combined``, ``improvement_factor_total``,
    ``phi_factor_from_rates``,
    ``range_vs_cycles``, ``range_loss_ratio``,
    ``cold_capacity``,
    ``ev_lifetime_years``, ``ev_cumulative_km``, ``preheating_efficiency``

Aus :mod:`liionp.simulation`:
    ``LoadProfile``, ``SimulationResult``,
    ``run_simulation``, ``compare_systems``
"""

from .analysis import (
    cold_capacity,
    ev_cumulative_km,
    ev_lifetime_years,
    improvement_factor_combined,
    improvement_factor_thermal,
    improvement_factor_total,
    improvement_factor_voltage,
    phi_factor_from_rates,
    preheating_efficiency,
    range_loss_ratio,
    range_vs_cycles,
)
from .cell import CellModel, CellParameters, CellState
from .constants import (
    D_EOL,
    E_A_SEI_DEFAULT,
    F_FARADAY,
    K_SEI_DEFAULT,
    Q0_DEFAULT,
    R_GAS,
    V_MAX_DEFAULT,
    V_MIN_DEFAULT,
)
from .controllers import PIDController, SimpleMPC
from .degradation import (
    arrhenius_rate,
    arrhenius_ratio,
    capacity_drift,
    capacity_loss_sei,
    cumulative_degradation,
    cyclic_degradation_rate,
    lifetime_from_rate,
    overvoltage_degradation_rate,
    sei_growth_rate,
    sei_thickness,
    total_degradation_rate,
    undervoltage_degradation_rate,
    voltage_degradation_rate,
)
from .mechanics import (
    PlateDistanceMPCController,
    ionic_resistance,
    mechanical_degradation_rate,
    mechanical_reduction_factor,
    mechanical_stress,
    swelling_strain,
    tortuosity,
)
from .simulation import LoadProfile, SimulationResult, compare_systems, run_simulation
from .thermal import ThermalModel

__all__ = [
    # constants
    "R_GAS",
    "F_FARADAY",
    "Q0_DEFAULT",
    "V_MAX_DEFAULT",
    "V_MIN_DEFAULT",
    "K_SEI_DEFAULT",
    "E_A_SEI_DEFAULT",
    "D_EOL",
    # degradation
    "arrhenius_rate",
    "arrhenius_ratio",
    "sei_growth_rate",
    "sei_thickness",
    "capacity_loss_sei",
    "capacity_drift",
    "overvoltage_degradation_rate",
    "undervoltage_degradation_rate",
    "voltage_degradation_rate",
    "cyclic_degradation_rate",
    "total_degradation_rate",
    "cumulative_degradation",
    "lifetime_from_rate",
    # thermal
    "ThermalModel",
    # cell
    "CellParameters",
    "CellState",
    "CellModel",
    # controllers
    "PIDController",
    "SimpleMPC",
    # mechanics
    "swelling_strain",
    "tortuosity",
    "ionic_resistance",
    "mechanical_stress",
    "mechanical_degradation_rate",
    "mechanical_reduction_factor",
    "PlateDistanceMPCController",
    # analysis
    "improvement_factor_thermal",
    "improvement_factor_voltage",
    "improvement_factor_combined",
    "improvement_factor_total",
    "phi_factor_from_rates",
    "range_vs_cycles",
    "range_loss_ratio",
    "cold_capacity",
    "ev_lifetime_years",
    "ev_cumulative_km",
    "preheating_efficiency",
    # simulation
    "LoadProfile",
    "SimulationResult",
    "run_simulation",
    "compare_systems",
]
