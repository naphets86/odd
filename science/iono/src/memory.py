"""
irf.memory - Pinning-Well-Speicher und ionische Konzentrationspeicherung
========================================================================

Implementiert:
  - PinningWell      Geometrischer Speicher (Tropfenposition als Information)
  - IonicMemoryCell  Analoger Speicher durch Ionenkonzentration
  - MemoryArray      Feld aus Speicherzellen mit Lese-/Schreibzugriff
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.droplet import Droplet
from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Pinning-Well (geometrischer Speicher)
# ---------------------------------------------------------------------------

class PinningWell:
    """
    Geometrischer Speicher durch hydrophile Vertiefung.

    Ein Tropfen in der Vertiefung = logisch '1'.
    Kein Tropfen in der Vertiefung = logisch '0'.

    Die Information ist nichtflüchtig (kein Strom nötig für Erhalt).
    """

    def __init__(
        self,
        position: tuple[int, int],
        well_area_m2: float,
        pinning_energy_J: float,
        label: Optional[str] = None,
    ) -> None:
        """
        Args:
            position:       (row, col) auf der Elektrodenmatrix
            well_area_m2:   Fläche der hydrophilen Vertiefung [m²]
            pinning_energy_J: Energie zum Herausbewegen eines Tropfens [J]
            label:          Optionaler Bezeichner
        """
        if well_area_m2 <= 0:
            raise ValidationError("well_area_m2", well_area_m2, "> 0")
        if pinning_energy_J < 0:
            raise ValidationError("pinning_energy_J", pinning_energy_J, ">= 0")

        self.position = position
        self.well_area_m2 = well_area_m2
        self.pinning_energy_J = pinning_energy_J
        self.label = label
        self._stored_droplet: Optional[Droplet] = None

    @property
    def is_occupied(self) -> bool:
        """True wenn ein Tropfen im Well gespeichert ist."""
        return self._stored_droplet is not None

    @property
    def logical_value(self) -> bool:
        """Logischer Wert: True = besetzt ('1'), False = leer ('0')."""
        return self.is_occupied

    def write(self, droplet: Optional[Droplet]) -> None:
        """
        Schreibt einen Tropfen in den Well (oder löscht ihn).

        Args:
            droplet: Tropfen zum Speichern, oder None zum Löschen
        """
        if droplet is not None and droplet.volume_m3 <= 0:
            raise ValidationError("droplet.volume_m3", droplet.volume_m3, "> 0")
        self._stored_droplet = droplet

    def read(self) -> Optional[Droplet]:
        """
        Liest den gespeicherten Tropfen (nicht-destruktiv).

        Returns:
            Gespeicherter Tropfen oder None
        """
        return self._stored_droplet

    def erase(self) -> Optional[Droplet]:
        """
        Löscht den Well und gibt den Tropfen zurück.

        Returns:
            Der entfernte Tropfen oder None
        """
        d = self._stored_droplet
        self._stored_droplet = None
        return d

    def can_hold(self, droplet: Droplet, applied_energy_J: float = 0.0) -> bool:
        """
        Prüft, ob der Well einen Tropfen halten kann.

        Ein Tropfen kann gehalten werden, wenn die anliegende Energie
        kleiner als die Pinning-Energie ist.

        Args:
            droplet:           Zu prüfender Tropfen
            applied_energy_J:  Von außen anliegende Energie [J]

        Returns:
            True wenn der Well den Tropfen halten kann
        """
        return applied_energy_J < self.pinning_energy_J


# ---------------------------------------------------------------------------
# Ionische Speicherzelle
# ---------------------------------------------------------------------------

class IonicMemoryCell:
    """
    Analoger Speicher durch Ionenkonzentration.

    Speichert Information als Ionenkonzentration in einem abgeschlossenen
    Volumen. Höhere Konzentration → niedrigerer Widerstand → 'stärkere' Verbindung.

    Dies realisiert einen synaptischen Plastizitätsmechanismus (ionische Plastizität).
    """

    def __init__(
        self,
        position: tuple[int, int],
        volume_m3: float,
        base_concentration_mol_m3: float = 100.0,
        plasticity_rate: float = 0.1,
    ) -> None:
        """
        Args:
            position:                     (row, col) auf der Matrix
            volume_m3:                    Volumen der Speicherzelle [m³]
            base_concentration_mol_m3:    Ruhekonzentration [mol/m³]
            plasticity_rate:              Lernrate δc/δt pro Aktivierung
        """
        if volume_m3 <= 0:
            raise ValidationError("volume_m3", volume_m3, "> 0")
        if base_concentration_mol_m3 < 0:
            raise ValidationError(
                "base_concentration_mol_m3", base_concentration_mol_m3, ">= 0"
            )
        if not (0.0 < plasticity_rate <= 1.0):
            raise ValidationError("plasticity_rate", plasticity_rate, "0 < rate <= 1")

        self.position = position
        self.volume_m3 = volume_m3
        self.base_concentration_mol_m3 = base_concentration_mol_m3
        self.plasticity_rate = plasticity_rate
        self._concentration_mol_m3 = base_concentration_mol_m3
        self._activation_count: int = 0

    @property
    def concentration_mol_m3(self) -> float:
        """Aktuelle Ionenkonzentration [mol/m³]."""
        return self._concentration_mol_m3

    @property
    def activation_count(self) -> int:
        """Anzahl der bisherigen Aktivierungen."""
        return self._activation_count

    @property
    def normalized_strength(self) -> float:
        """
        Normierte synaptische Stärke im Bereich [0, 1].

        0 = Ruhekonzentration, 1 = maximale gespeicherte Stärke.
        """
        if self.base_concentration_mol_m3 == 0:
            return 0.0
        excess = self._concentration_mol_m3 - self.base_concentration_mol_m3
        return min(1.0, max(0.0, excess / self.base_concentration_mol_m3))

    def activate(self) -> float:
        """
        Aktiviert die Zelle (Hebbsches Lernen: Konzentration erhöhen).

        Returns:
            Neue Konzentration [mol/m³]
        """
        self._activation_count += 1
        delta = self.base_concentration_mol_m3 * self.plasticity_rate
        self._concentration_mol_m3 += delta
        return self._concentration_mol_m3

    def decay(self, decay_factor: float = 0.01) -> float:
        """
        Exponentieller Zerfall zurück zur Ruhekonzentration.

        Args:
            decay_factor: Zerfallsrate pro Zeitschritt [0, 1]

        Returns:
            Neue Konzentration [mol/m³]
        """
        if not (0.0 <= decay_factor <= 1.0):
            raise ValidationError("decay_factor", decay_factor, "0 <= factor <= 1")
        diff = self._concentration_mol_m3 - self.base_concentration_mol_m3
        self._concentration_mol_m3 = (
            self.base_concentration_mol_m3 + diff * (1.0 - decay_factor)
        )
        return self._concentration_mol_m3

    def reset(self) -> None:
        """Setzt die Zelle auf Ruhekonzentration zurück."""
        self._concentration_mol_m3 = self.base_concentration_mol_m3
        self._activation_count = 0

    def write_analog(self, concentration_mol_m3: float) -> None:
        """
        Schreibt direkt eine Konzentration (analoges Schreiben).

        Args:
            concentration_mol_m3: Zielkonzentration [mol/m³]
        """
        if concentration_mol_m3 < 0:
            raise ValidationError("concentration_mol_m3", concentration_mol_m3, ">= 0")
        self._concentration_mol_m3 = concentration_mol_m3


# ---------------------------------------------------------------------------
# Speicher-Array
# ---------------------------------------------------------------------------

class MemoryArray:
    """
    Feld aus Pinning-Well-Speicherzellen (digitaler Speicher).

    Unterstützt wahlfreien Lese-/Schreibzugriff auf Bit-Ebene.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        well_area_m2: float = 25e-12,
        pinning_energy_J: float = 1e-15,
    ) -> None:
        """
        Args:
            rows, cols:         Dimensionen des Arrays
            well_area_m2:       Fläche jedes Wells [m²]
            pinning_energy_J:   Pinning-Energie jedes Wells [J]
        """
        if rows < 1:
            raise ValidationError("rows", rows, ">= 1")
        if cols < 1:
            raise ValidationError("cols", cols, ">= 1")

        self.rows = rows
        self.cols = cols
        self._cells: list[list[PinningWell]] = [
            [
                PinningWell(
                    position=(r, c),
                    well_area_m2=well_area_m2,
                    pinning_energy_J=pinning_energy_J,
                    label=f"W[{r},{c}]",
                )
                for c in range(cols)
            ]
            for r in range(rows)
        ]

    @property
    def capacity_bits(self) -> int:
        """Speicherkapazität in Bits."""
        return self.rows * self.cols

    def write_bit(
        self,
        row: int,
        col: int,
        value: bool,
        droplet_volume_m3: float = 1e-15,
    ) -> None:
        """
        Schreibt ein einzelnes Bit.

        Args:
            row, col:            Adresse
            value:               True = '1' (Tropfen einsetzen), False = '0'
            droplet_volume_m3:   Tropfenvolumen für '1' [m³]
        """
        self._check(row, col)
        well = self._cells[row][col]
        if value:
            d = Droplet(volume_m3=droplet_volume_m3, position=(row, col))
            well.write(d)
        else:
            well.erase()

    def read_bit(self, row: int, col: int) -> bool:
        """Liest ein einzelnes Bit."""
        self._check(row, col)
        return self._cells[row][col].logical_value

    def write_word(
        self,
        row: int,
        value: int,
        droplet_volume_m3: float = 1e-15,
    ) -> None:
        """
        Schreibt ein ganzzahliges Wort in eine Zeile (LSB = Spalte 0).

        Args:
            row:   Zeilenadresse
            value: Ganzzahliger Wert (nicht-negativ)
        """
        if row < 0 or row >= self.rows:
            raise ValidationError("row", row, f"0 <= row < {self.rows}")
        if value < 0:
            raise ValidationError("value", value, ">= 0")
        for c in range(self.cols):
            bit = bool((value >> c) & 1)
            self.write_bit(row, c, bit, droplet_volume_m3)

    def read_word(self, row: int) -> int:
        """
        Liest ein ganzzahliges Wort aus einer Zeile (LSB = Spalte 0).

        Returns:
            Ganzzahliger Wert
        """
        if row < 0 or row >= self.rows:
            raise ValidationError("row", row, f"0 <= row < {self.rows}")
        val = 0
        for c in range(self.cols):
            if self.read_bit(row, c):
                val |= (1 << c)
        return val

    def clear(self) -> None:
        """Löscht alle gespeicherten Bits (setzt alle Wells auf '0')."""
        for r in range(self.rows):
            for c in range(self.cols):
                self._cells[r][c].erase()

    def count_ones(self) -> int:
        """Zählt die gesetzten Bits ('1') im gesamten Array."""
        return sum(
            1
            for r in range(self.rows)
            for c in range(self.cols)
            if self._cells[r][c].logical_value
        )

    def _check(self, row: int, col: int) -> None:
        if not (0 <= row < self.rows):
            raise ValidationError("row", row, f"0 <= row < {self.rows}")
        if not (0 <= col < self.cols):
            raise ValidationError("col", col, f"0 <= col < {self.cols}")
