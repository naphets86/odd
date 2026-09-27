"""
IRF - Ionotronic Reconfigurable Fabric Framework
=================================================

Ein Python-Framework zur Simulation und Analyse von ionotronischen,
wasservolumen-basierten Rechenarchitekturen.

Module:
    physics     - Physikalische Grundmodelle (EWOD, Ionenleitung, Oberflächenspannung)
    transistor  - IoFET-Transistormodell und Kennlinien
    logic       - Tropfenlogik: Gatter, Verknüpfungen, Schaltnetze
    matrix      - Elektrodenmatrix und Topologieverwaltung
    droplet     - Tropfendynamik und Selbstorganisation
    memory      - Pinning-Well-Speicher und ionische Konzentrationspeicherung
    architecture- Systemarchitektur, Hybridsteuerung und Durchsatzanalyse
    simulation  - Simulationsengine und Zeitverlaufsanalyse
    fabrication - Fertigungsprozessmodell und Parametervalidierung
    utils       - Hilfsfunktionen, Einheitenkonvertierung, Logging
"""

__version__ = "1.0.0"
__author__ = "Stephan Epp"
__license__ = "LICENSE"

from .physics import (
    EWODModel,
    IonicConductivity,
    SurfaceTension,
    ElectrokineticTransport,
)
from src.transistor import IoFET, IoFETState, TransferCharacteristic
from src.logic import (
    DropletGate,
    GateType,
    LogicNetwork,
    TruthTable,
)
from src.matrix import ElectrodeMatrix, ElectrodeState, MatrixConfig
from src.droplet import Droplet, DropletMerger, SelfOrganizer
from src.memory import PinningWell, IonicMemoryCell, MemoryArray
from src.architecture import (
    IRFSystem,
    HybridController,
    ThroughputAnalyzer,
    VonNeumannBottleneck,
)
from src.simulation import (
    Simulator,
    SimulationResult,
    TimeStep,
    SimulationConfig,
)
from src.fabrication import (
    FabricationProcess,
    ProcessStep,
    ProcessValidator,
    TechnologyNode,
)
from src.utils import (
    UnitConverter,
    Logger,
    PhysicalConstants,
    ValidationError,
)

__all__ = [
    # physics
    "EWODModel",
    "IonicConductivity",
    "SurfaceTension",
    "ElectrokineticTransport",
    # transistor
    "IoFET",
    "IoFETState",
    "TransferCharacteristic",
    # logic
    "DropletGate",
    "GateType",
    "LogicNetwork",
    "TruthTable",
    # matrix
    "ElectrodeMatrix",
    "ElectrodeState",
    "MatrixConfig",
    # droplet
    "Droplet",
    "DropletMerger",
    "SelfOrganizer",
    # memory
    "PinningWell",
    "IonicMemoryCell",
    "MemoryArray",
    # architecture
    "IRFSystem",
    "HybridController",
    "ThroughputAnalyzer",
    "VonNeumannBottleneck",
    # simulation
    "Simulator",
    "SimulationResult",
    "TimeStep",
    "SimulationConfig",
    # fabrication
    "FabricationProcess",
    "ProcessStep",
    "ProcessValidator",
    "TechnologyNode",
    # utils
    "UnitConverter",
    "Logger",
    "PhysicalConstants",
    "ValidationError",
]
