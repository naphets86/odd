"""
irf.matrix - Elektrodenmatrix und Topologieverwaltung
======================================================

Implementiert:
  - ElectrodeState   Zustand einer einzelnen Elektrode
  - MatrixConfig     Konfigurationsparameter der Matrix
  - ElectrodeMatrix  Vollständige 2D-Elektrodenmatrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional

from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Elektrodenzustand
# ---------------------------------------------------------------------------

class ElectrodeState(Enum):
    """Logischer und physikalischer Zustand einer Elektrode."""
    OFF = auto()      # Keine Spannung, hydrophobe Oberfläche dominiert
    ON = auto()       # Spannung angelegt, Benetzung aktiv
    FLOODED = auto()  # Von Tropfen überdeckt (leitend)
    DRY = auto()      # Explizit trocken (kein Tropfen)
    DISABLED = auto() # Elektrode deaktiviert (Defekt oder Reservierung)


# ---------------------------------------------------------------------------
# Matrixkonfiguration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatrixConfig:
    """
    Konfigurationsparameter für eine Elektrodenmatrix.

    Attributes:
        rows:              Anzahl der Zeilen
        cols:              Anzahl der Spalten
        electrode_pitch_m: Pitch (Mitte-zu-Mitte-Abstand) [m]
        electrode_size_m:  Elektrodengröße (quadratisch) [m]
        channel_height_m:  Höhe des Flüssigkeitskanals [m]
        num_layers:        Anzahl der gestapelten Schichten (3D-IRF)
    """

    rows: int
    cols: int
    electrode_pitch_m: float
    electrode_size_m: float
    channel_height_m: float
    num_layers: int = 1

    def __post_init__(self) -> None:
        for name, val in [("rows", self.rows), ("cols", self.cols), ("num_layers", self.num_layers)]:
            if val < 1:
                raise ValidationError(name, val, ">= 1")
        for name, val in [
            ("electrode_pitch_m", self.electrode_pitch_m),
            ("electrode_size_m", self.electrode_size_m),
            ("channel_height_m", self.channel_height_m),
        ]:
            if val <= 0:
                raise ValidationError(name, val, "> 0")
        if self.electrode_size_m >= self.electrode_pitch_m:
            raise ValidationError(
                "electrode_size_m",
                self.electrode_size_m,
                "< electrode_pitch_m",
            )

    @property
    def num_electrodes(self) -> int:
        """Gesamtanzahl der Elektroden in allen Schichten."""
        return self.rows * self.cols * self.num_layers

    @property
    def chip_area_m2(self) -> float:
        """Gesamtchipfläche [m²]."""
        return (self.rows * self.electrode_pitch_m) * (self.cols * self.electrode_pitch_m)

    @property
    def electrode_density_per_m2(self) -> float:
        """Elektrodendichte [Elektroden/m²] (pro Schicht)."""
        return (self.rows * self.cols) / self.chip_area_m2

    @property
    def equivalent_transistor_density_per_m2(self) -> float:
        """
        Äquivalente Transistordichte [Transistoren/m²].

        Faktor 2, da jede Elektrode gleichzeitig Schalter UND Speicher ist.
        Multipliziert mit num_layers für 3D-Stapelung.
        """
        return 2.0 * self.electrode_density_per_m2 * self.num_layers


# ---------------------------------------------------------------------------
# Elektrodenmatrix
# ---------------------------------------------------------------------------

class ElectrodeMatrix:
    """
    Vollständige 2D-Elektrodenmatrix des IRF-Chips.

    Verwaltet den Zustand jeder Elektrode und berechnet Verbindungstopologien.
    """

    def __init__(self, config: MatrixConfig) -> None:
        self.config = config
        # Zustandsmatrix: [layer][row][col]
        self._state: list[list[list[ElectrodeState]]] = [
            [
                [ElectrodeState.DRY] * config.cols
                for _ in range(config.rows)
            ]
            for _ in range(config.num_layers)
        ]
        # Spannungsmatrix: [layer][row][col]
        self._voltage: list[list[list[float]]] = [
            [[0.0] * config.cols for _ in range(config.rows)]
            for _ in range(config.num_layers)
        ]

    # -----------------------------------------------------------------------
    # Zugriff
    # -----------------------------------------------------------------------

    def _check_coords(self, layer: int, row: int, col: int) -> None:
        if not (0 <= layer < self.config.num_layers):
            raise ValidationError("layer", layer, f"0 <= layer < {self.config.num_layers}")
        if not (0 <= row < self.config.rows):
            raise ValidationError("row", row, f"0 <= row < {self.config.rows}")
        if not (0 <= col < self.config.cols):
            raise ValidationError("col", col, f"0 <= col < {self.config.cols}")

    def set_voltage(self, layer: int, row: int, col: int, voltage: float) -> None:
        """Setzt die Spannung einer Elektrode."""
        self._check_coords(layer, row, col)
        self._voltage[layer][row][col] = voltage

    def get_voltage(self, layer: int, row: int, col: int) -> float:
        """Gibt die aktuelle Spannung einer Elektrode zurück."""
        self._check_coords(layer, row, col)
        return self._voltage[layer][row][col]

    def set_state(self, layer: int, row: int, col: int, state: ElectrodeState) -> None:
        """Setzt den Zustand einer Elektrode."""
        self._check_coords(layer, row, col)
        self._state[layer][row][col] = state

    def get_state(self, layer: int, row: int, col: int) -> ElectrodeState:
        """Gibt den aktuellen Zustand einer Elektrode zurück."""
        self._check_coords(layer, row, col)
        return self._state[layer][row][col]

    # -----------------------------------------------------------------------
    # Massenoperationen
    # -----------------------------------------------------------------------

    def apply_voltage_pattern(
        self,
        pattern: list[list[float]],
        layer: int = 0,
    ) -> None:
        """
        Setzt Spannungen für eine komplette Schicht anhand einer 2D-Matrix.

        Args:
            pattern: rows × cols Matrix mit Spannungswerten
            layer:   Zielschicht
        """
        if len(pattern) != self.config.rows:
            raise ValidationError("pattern rows", len(pattern), f"== {self.config.rows}")
        for r, row_vals in enumerate(pattern):
            if len(row_vals) != self.config.cols:
                raise ValidationError(
                    f"pattern[{r}] cols", len(row_vals), f"== {self.config.cols}"
                )
            for c, v in enumerate(row_vals):
                self.set_voltage(layer, r, c, v)

    def reset_layer(self, layer: int) -> None:
        """Setzt alle Spannungen und Zustände einer Schicht zurück."""
        self._check_coords(layer, 0, 0)
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                self._voltage[layer][r][c] = 0.0
                self._state[layer][r][c] = ElectrodeState.DRY

    def reset_all(self) -> None:
        """Setzt die gesamte Matrix zurück."""
        for lay in range(self.config.num_layers):
            self.reset_layer(lay)

    # -----------------------------------------------------------------------
    # Topologieanalyse
    # -----------------------------------------------------------------------

    def active_electrodes(self, layer: int = 0) -> list[tuple[int, int]]:
        """
        Gibt Koordinaten aller aktiven (ON oder FLOODED) Elektroden zurück.

        Returns:
            Liste von (row, col)-Tupeln
        """
        self._check_coords(layer, 0, 0)
        result = []
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if self._state[layer][r][c] in (
                    ElectrodeState.ON, ElectrodeState.FLOODED
                ):
                    result.append((r, c))
        return result

    def neighbors(
        self,
        layer: int,
        row: int,
        col: int,
    ) -> list[tuple[int, int, int]]:
        """
        Gibt alle gültigen Nachbarelektroden in der gleichen Schicht zurück.

        Returns:
            Liste von (layer, row, col)-Tupeln
        """
        self._check_coords(layer, row, col)
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.config.rows and 0 <= nc < self.config.cols:
                result.append((layer, nr, nc))
        return result

    def count_state(self, state: ElectrodeState, layer: int = 0) -> int:
        """Zählt die Elektroden in einem bestimmten Zustand."""
        self._check_coords(layer, 0, 0)
        return sum(
            1
            for r in range(self.config.rows)
            for c in range(self.config.cols)
            if self._state[layer][r][c] == state
        )

    def utilization(self, layer: int = 0) -> float:
        """
        Auslastungsgrad: Anteil aktiver Elektroden.

        Returns:
            Wert zwischen 0.0 und 1.0
        """
        total = self.config.rows * self.config.cols
        active = self.count_state(ElectrodeState.ON, layer) + \
                 self.count_state(ElectrodeState.FLOODED, layer)
        return active / total if total > 0 else 0.0

    def all_states(self, layer: int = 0) -> list[list[ElectrodeState]]:
        """Gibt die vollständige Zustandsmatrix einer Schicht zurück (Kopie)."""
        self._check_coords(layer, 0, 0)
        return [row[:] for row in self._state[layer]]
