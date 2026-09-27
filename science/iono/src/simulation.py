"""
irf.simulation - Simulationsengine und Zeitverlaufsanalyse
===========================================================

Implementiert:
  - SimulationConfig  Konfiguration eines Simulationslaufs
  - TimeStep          Einzelner Zeitschritt mit Zustand und Metriken
  - SimulationResult  Ergebnis eines vollständigen Simulationslaufs
  - Simulator         Hauptsimulationsengine
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.architecture import IRFSystem
from src.droplet import Droplet, SelfOrganizer
from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """
    Konfiguration eines IRF-Simulationslaufs.

    Attributes:
        total_time_s:      Simulationsdauer [s]
        time_step_s:       Zeitschrittgröße [s]
        threshold_voltage_V: Schaltspannungsschwelle [V]
        record_all_steps:  Ob alle Zeitschritte aufgezeichnet werden sollen
        random_seed:       Zufallssamen für reproduzierbare Simulation
    """

    total_time_s: float
    time_step_s: float
    threshold_voltage_V: float = 0.12
    record_all_steps: bool = True
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.total_time_s <= 0:
            raise ValidationError("total_time_s", self.total_time_s, "> 0")
        if self.time_step_s <= 0:
            raise ValidationError("time_step_s", self.time_step_s, "> 0")
        if self.time_step_s > self.total_time_s:
            raise ValidationError(
                "time_step_s", self.time_step_s, "<= total_time_s"
            )
        if self.threshold_voltage_V <= 0:
            raise ValidationError("threshold_voltage_V", self.threshold_voltage_V, "> 0")

    @property
    def num_steps(self) -> int:
        """Anzahl der Simulationsschritte."""
        return math.ceil(self.total_time_s / self.time_step_s)


# ---------------------------------------------------------------------------
# Zeitschritt
# ---------------------------------------------------------------------------

@dataclass
class TimeStep:
    """
    Zustand und Metriken eines einzelnen Simulationszeitschritts.

    Attributes:
        step_index:      Index des Schritts (0-basiert)
        time_s:          Simulationszeit [s]
        num_droplets:    Anzahl der Tropfen
        active_electrodes: Anzahl der aktiven Elektroden
        energy_J:        In diesem Schritt verbrauchte Energie [J]
        merges:          Anzahl der Tropfenverschmelzungen in diesem Schritt
        utilization:     Matrixauslastung [0, 1]
        extra:           Zusätzliche benutzerdefinierte Metriken
    """

    step_index: int
    time_s: float
    num_droplets: int
    active_electrodes: int
    energy_J: float
    merges: int = 0
    utilization: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulationsergebnis
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Vollständiges Ergebnis eines Simulationslaufs.

    Attributes:
        config:       Verwendete Konfiguration
        steps:        Liste aller aufgezeichneten Zeitschritte
        final_droplets: Tropfen am Ende der Simulation
    """

    config: SimulationConfig
    steps: list[TimeStep] = field(default_factory=list)
    final_droplets: list[Droplet] = field(default_factory=list)

    @property
    def total_energy_J(self) -> float:
        """Gesamtenergieverbrauch der Simulation [J]."""
        return sum(s.energy_J for s in self.steps)

    @property
    def total_merges(self) -> int:
        """Gesamtzahl der Tropfenverschmelzungen."""
        return sum(s.merges for s in self.steps)

    @property
    def mean_utilization(self) -> float:
        """Mittlere Matrixauslastung über die Simulation."""
        if not self.steps:
            return 0.0
        return sum(s.utilization for s in self.steps) / len(self.steps)

    @property
    def num_steps_recorded(self) -> int:
        """Anzahl aufgezeichneter Schritte."""
        return len(self.steps)

    def peak_droplet_count(self) -> int:
        """Maximale Tropfenanzahl über alle Schritte."""
        if not self.steps:
            return 0
        return max(s.num_droplets for s in self.steps)

    def time_series(self, metric: str) -> list[tuple[float, float]]:
        """
        Gibt eine Zeitreihe einer Metrik zurück.

        Args:
            metric: Name der Metrik (z.B. 'num_droplets', 'energy_J')

        Returns:
            Liste von (Zeit [s], Wert)-Tupeln
        """
        result = []
        for step in self.steps:
            val = getattr(step, metric, step.extra.get(metric))
            if val is not None:
                result.append((step.time_s, float(val)))
        return result


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Hauptsimulationsengine für das IRF-System.

    Kombiniert Tropfendynamik mit Matrixsteuerung und zeichnet
    alle relevanten Zustandsgrößen zeitlich auf.
    """

    def __init__(
        self,
        system: IRFSystem,
        config: SimulationConfig,
    ) -> None:
        self.system = system
        self.config = config
        self._organizer = SelfOrganizer(
            threshold_voltage_V=config.threshold_voltage_V
        )
        self._result: Optional[SimulationResult] = None

    def add_droplet(self, droplet: Droplet) -> None:
        """Fügt einen Anfangstropfen zur Simulation hinzu."""
        self._organizer.add_droplet(droplet)

    @property
    def result(self) -> Optional[SimulationResult]:
        """Ergebnis des letzten Simulationslaufs."""
        return self._result

    def run(
        self,
        voltage_schedule: Callable[[float], dict[tuple[int, int], float]],
    ) -> SimulationResult:
        """
        Führt die vollständige Simulation aus.

        Args:
            voltage_schedule: Funktion (Zeit [s]) → Spannungsschema
                              Mapping (row, col) → Spannung [V]

        Returns:
            SimulationResult mit allen aufgezeichneten Daten
        """
        result = SimulationResult(config=self.config)
        self.system.reset()

        for i in range(self.config.num_steps):
            t = i * self.config.time_step_s
            voltages = voltage_schedule(t)

            # Tropfen-Schritt
            n_before = self._organizer.num_droplets
            current_droplets = self._organizer.step(voltages)
            n_after = self._organizer.num_droplets
            merges = max(0, n_before - n_after)

            # Matrixzustand aktualisieren
            cfg = self.system.config
            pattern = [[0.0] * cfg.cols for _ in range(cfg.rows)]
            for (r, c), v in voltages.items():
                if 0 <= r < cfg.rows and 0 <= c < cfg.cols:
                    pattern[r][c] = v

            metrics = self.system.execute_configuration(pattern)
            utilization = metrics["utilization"]

            if self.config.record_all_steps:
                ts = TimeStep(
                    step_index=i,
                    time_s=t,
                    num_droplets=len(current_droplets),
                    active_electrodes=int(utilization * cfg.num_electrodes),
                    energy_J=metrics["energy_J"],
                    merges=merges,
                    utilization=utilization,
                )
                result.steps.append(ts)

        result.final_droplets = list(self._organizer.droplets)
        self._result = result
        return result

    def reset(self) -> None:
        """Setzt den Simulator für einen neuen Lauf zurück."""
        self._organizer = SelfOrganizer(
            threshold_voltage_V=self.config.threshold_voltage_V
        )
        self._result = None
        self.system.reset()
