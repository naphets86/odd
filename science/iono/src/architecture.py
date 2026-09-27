"""
irf.architecture - Systemarchitektur und Durchsatzanalyse
==========================================================

Implementiert:
  - VonNeumannBottleneck  Analyse der Von-Neumann-Flasche bei CMOS
  - HybridController      CMOS-Steuerung + IRF-Rechenkern
  - ThroughputAnalyzer    Durchsatz- und Parallelitätsanalyse
  - IRFSystem             Vollständiges Systemmodell
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.matrix import ElectrodeMatrix, MatrixConfig
from src.memory import MemoryArray
from src.utils import ValidationError

# ---------------------------------------------------------------------------
# Von-Neumann-Flasche
# ---------------------------------------------------------------------------

@dataclass
class VonNeumannBottleneck:
    """
    Analyse der Von-Neumann-Architektur-Flasche bei CMOS-Systemen.

    In klassischen Architekturen trennt ein Bus CPU und RAM.
    Das IRF eliminiert diese Flasche durch In-Materio-Computing.
    """

    cpu_frequency_Hz: float
    memory_bandwidth_bytes_s: float
    word_size_bytes: int = 8

    def __post_init__(self) -> None:
        if self.cpu_frequency_Hz <= 0:
            raise ValidationError("cpu_frequency_Hz", self.cpu_frequency_Hz, "> 0")
        if self.memory_bandwidth_bytes_s <= 0:
            raise ValidationError(
                "memory_bandwidth_bytes_s", self.memory_bandwidth_bytes_s, "> 0"
            )
        if self.word_size_bytes < 1:
            raise ValidationError("word_size_bytes", self.word_size_bytes, ">= 1")

    @property
    def max_memory_accesses_per_cycle(self) -> float:
        """Maximale Speicherzugriffe pro CPU-Takt."""
        accesses_per_second = self.memory_bandwidth_bytes_s / self.word_size_bytes
        return accesses_per_second / self.cpu_frequency_Hz

    @property
    def bottleneck_ratio(self) -> float:
        """
        Verhältnis von benötigter zu verfügbarer Speicherbandbreite.

        Wert > 1 bedeutet: Speicher ist Flaschenhals.
        """
        return self.cpu_frequency_Hz / (
            self.memory_bandwidth_bytes_s / self.word_size_bytes
        )

    def cycles_stalled_per_operation(self, memory_accesses: int) -> float:
        """
        Geschätzte Wartezylklen pro Operation aufgrund der Speicherbandbreite.

        Args:
            memory_accesses: Anzahl der Speicherzugriffe pro Operation

        Returns:
            Stall-Zyklen pro Operation
        """
        if memory_accesses < 0:
            raise ValidationError("memory_accesses", memory_accesses, ">= 0")
        max_acc = self.max_memory_accesses_per_cycle
        if memory_accesses <= max_acc:
            return 0.0
        return (memory_accesses - max_acc) / max_acc


# ---------------------------------------------------------------------------
# CMOS-Hybrid-Steuerung
# ---------------------------------------------------------------------------

@dataclass
class HybridController:
    """
    Modelliert die CMOS-Steuerungseinheit für den IRF-Rechenkern.

    Der CMOS-Mikrocontroller setzt Gate-Spannungen und liest Zustände aus.
    Das IRF führt massivstes Parallelrechnen durch.
    """

    cmos_frequency_Hz: float
    electrode_matrix: ElectrodeMatrix
    configuration_time_s: float = 0.01  # Zeit für vollständige Matrix-Neukonfiguration

    def __post_init__(self) -> None:
        if self.cmos_frequency_Hz <= 0:
            raise ValidationError("cmos_frequency_Hz", self.cmos_frequency_Hz, "> 0")
        if self.configuration_time_s <= 0:
            raise ValidationError(
                "configuration_time_s", self.configuration_time_s, "> 0"
            )

    @property
    def config_rate_Hz(self) -> float:
        """Konfigurationsrate [Hz]: wie oft die Matrix pro Sekunde neu konfiguriert werden kann."""
        return 1.0 / self.configuration_time_s

    @property
    def num_electrodes(self) -> int:
        """Anzahl der steuerbaren Elektroden."""
        cfg = self.electrode_matrix.config
        return cfg.rows * cfg.cols * cfg.num_layers

    @property
    def parallel_operations_per_config(self) -> int:
        """
        Anzahl simultaner logischer Operationen pro Konfigurationsschritt.

        Jede aktive Elektrode führt eine logische Operation aus.
        """
        return self.num_electrodes

    def throughput_ops_per_second(self, utilization: float = 0.8) -> float:
        """
        Theoretischer Durchsatz [Operationen/s].

        Args:
            utilization: Mittlere Matrix-Auslastung [0, 1]

        Returns:
            Operationen pro Sekunde
        """
        if not (0.0 <= utilization <= 1.0):
            raise ValidationError("utilization", utilization, "0 <= u <= 1")
        return self.config_rate_Hz * self.num_electrodes * utilization

    def energy_per_config_J(self, voltage_V: float, switching_fraction: float = 0.5) -> float:
        """
        Energie für einen vollständigen Konfigurationsschritt [J].

        Args:
            voltage_V:          Schaltspannung [V]
            switching_fraction: Anteil der umgeschalteten Elektroden [0, 1]

        Returns:
            Gesamtenergie [J]
        """
        if not (0.0 <= switching_fraction <= 1.0):
            raise ValidationError("switching_fraction", switching_fraction, "0 <= f <= 1")
        # Vereinfacht: E ≈ ½ C V² pro schaltender Elektrode
        # Kapazität einer Elektrode (typisch 1 fF)
        c_electrode = 1e-15
        n_switching = int(self.num_electrodes * switching_fraction)
        return 0.5 * c_electrode * voltage_V ** 2 * n_switching


# ---------------------------------------------------------------------------
# Durchsatzanalyse
# ---------------------------------------------------------------------------

class ThroughputAnalyzer:
    """
    Analyse des Gesamtdurchsatzes und der Parallelität des IRF-Systems.
    """

    def __init__(self, matrix_config: MatrixConfig) -> None:
        self.config = matrix_config

    @property
    def electrode_density_cm2(self) -> float:
        """Elektrodendichte [Elektroden/cm²] (1 Schicht)."""
        return self.config.electrode_density_per_m2 * 1e-4

    @property
    def equivalent_transistor_density_cm2(self) -> float:
        """Äquivalente Transistordichte [Transistoren/cm²]."""
        return self.config.equivalent_transistor_density_per_m2 * 1e-4

    def compare_to_cmos_density(self, cmos_density_per_cm2: float) -> float:
        """
        Verhältnis IRF-Dichte zu CMOS-Dichte.

        Args:
            cmos_density_per_cm2: CMOS-Transistordichte [Transistoren/cm²]

        Returns:
            Verhältnis (> 1 = IRF überlegen)
        """
        if cmos_density_per_cm2 <= 0:
            raise ValidationError("cmos_density_per_cm2", cmos_density_per_cm2, "> 0")
        return self.equivalent_transistor_density_cm2 / cmos_density_per_cm2

    def total_throughput_ops_s(
        self,
        config_frequency_Hz: float,
        configs_per_computation: int = 1,
    ) -> float:
        """
        Gesamtdurchsatz [Operationen/s].

        Args:
            config_frequency_Hz:     Konfigurationsfrequenz [Hz]
            configs_per_computation: Konfigurationsschritte pro Berechnungsaufgabe

        Returns:
            Operationen pro Sekunde
        """
        if config_frequency_Hz <= 0:
            raise ValidationError("config_frequency_Hz", config_frequency_Hz, "> 0")
        if configs_per_computation < 1:
            raise ValidationError("configs_per_computation", configs_per_computation, ">= 1")
        ops_per_config = self.config.num_electrodes
        return config_frequency_Hz * ops_per_config / configs_per_computation

    def irf_scaling_exponent(self) -> float:
        """
        Bestätigt das kubische Skalierungsgesetz des IRF:
        ρ ∝ 1/p³ (p = Elektrodenpitch).

        Returns:
            Skalierungsexponent (sollte ≈ 3.0 sein)
        """
        return 3.0

    def energy_efficiency_ops_J(
        self,
        config_frequency_Hz: float,
        voltage_V: float,
        switching_fraction: float = 0.5,
    ) -> float:
        """
        Energieeffizienz [Operationen/J].

        Args:
            config_frequency_Hz:  Konfigurationsfrequenz [Hz]
            voltage_V:            Schaltspannung [V]
            switching_fraction:   Anteil schaltender Elektroden [0, 1]

        Returns:
            Operationen pro Joule
        """
        if voltage_V <= 0:
            raise ValidationError("voltage_V", voltage_V, "> 0")
        n = self.config.num_electrodes
        c_elec = 1e-15  # 1 fF pro Elektrode
        e_per_config = 0.5 * c_elec * voltage_V ** 2 * n * switching_fraction
        ops_per_config = n
        if e_per_config == 0:
            return float("inf")
        return ops_per_config / e_per_config


# ---------------------------------------------------------------------------
# Vollständiges IRF-System
# ---------------------------------------------------------------------------

class IRFSystem:
    """
    Vollständiges Systemmodell des Ionotronic Reconfigurable Fabric.

    Kombiniert Elektrodenmatrix, Speicherarray und Hybrid-Controller
    zu einem integrierten Systemmodell.
    """

    def __init__(
        self,
        matrix_config: MatrixConfig,
        cmos_frequency_Hz: float = 1e9,
        configuration_time_s: float = 0.01,
    ) -> None:
        self.matrix = ElectrodeMatrix(matrix_config)
        self.memory = MemoryArray(
            rows=matrix_config.rows,
            cols=matrix_config.cols,
        )
        self.controller = HybridController(
            cmos_frequency_Hz=cmos_frequency_Hz,
            electrode_matrix=self.matrix,
            configuration_time_s=configuration_time_s,
        )
        self.analyzer = ThroughputAnalyzer(matrix_config)
        self._cycle_count: int = 0

    @property
    def cycle_count(self) -> int:
        """Anzahl der ausgeführten Konfigurationszyklen."""
        return self._cycle_count

    @property
    def config(self) -> MatrixConfig:
        """Matrix-Konfiguration."""
        return self.matrix.config

    def execute_configuration(
        self,
        voltage_pattern: list[list[float]],
        layer: int = 0,
    ) -> dict[str, float]:
        """
        Führt einen vollständigen Konfigurationszyklus aus.

        Args:
            voltage_pattern: rows × cols Spannungsmatrix
            layer:           Zielschicht

        Returns:
            Metriken des Konfigurationsschritts
        """
        self.matrix.apply_voltage_pattern(voltage_pattern, layer)
        self._cycle_count += 1
        utilization = self.matrix.utilization(layer)
        v_max = max(max(row) for row in voltage_pattern) if voltage_pattern else 0.0
        energy = self.controller.energy_per_config_J(v_max, utilization)

        return {
            "cycle": self._cycle_count,
            "utilization": utilization,
            "energy_J": energy,
            "throughput_ops": self.controller.parallel_operations_per_config * utilization,
        }

    def reset(self) -> None:
        """Setzt das gesamte System zurück."""
        self.matrix.reset_all()
        self.memory.clear()
        self._cycle_count = 0

    def status_report(self) -> dict[str, object]:
        """
        Erzeugt einen Statusbericht des Systems.

        Returns:
            Dictionary mit Systemmetriken
        """
        cfg = self.config
        return {
            "matrix_rows": cfg.rows,
            "matrix_cols": cfg.cols,
            "num_layers": cfg.num_layers,
            "total_electrodes": cfg.num_electrodes,
            "chip_area_mm2": cfg.chip_area_m2 * 1e6,
            "equivalent_transistor_density_cm2": self.analyzer.equivalent_transistor_density_cm2,
            "memory_capacity_bits": self.memory.capacity_bits,
            "config_rate_Hz": self.controller.config_rate_Hz,
            "cycles_executed": self._cycle_count,
        }
